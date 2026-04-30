import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.profiler as profiler
from torch.cuda.amp import autocast, GradScaler
from imblearn.over_sampling import SMOTE

"""训练器"""
class GraphTrainer:
    def __init__(self, model, device, relation, modelName):
        self.model = model              #正在训练的模型
        self.device = device            #使用的设备
        self.relation = relation        #关系类型判断条件
        self.modelName = modelName      #使用的模型（调用的参数不同）
        self.history = {}               #训练历史，用于k折交叉验证时保存的列表
        self.confusions = []            #保存所有的混淆矩阵
        self.bestModel = {}             #用于保存训练中正确率最高的模型状态
        self.valProbs = []
        self.valLabels = []

    def collectEmbed(self, loader):
        self.model.eval()
        embeds = []
        labels = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                # 根据模型名决定如何调用
                if self.modelName == 'GCNModel':
                    _, embed = self.model(batch.x, batch.edge_index, None, batch.batch)
                embeds.append(embed.cpu())
                labels.append(batch.y.cpu())
        return torch.cat(embeds, dim=0).numpy(), torch.cat(labels, dim=0).numpy()

    def smoteAugment(self, embeddings, labels, minority_class=None, k_neighbors=5, sampling_strategy='auto'):
        """
        embeddings: numpy array (n_samples, embed_dim)
        labels: numpy array (n_samples,)
        minority_class: 指定少数类（若为 None 则自动检测）
        """
        if minority_class is not None:
            # 只对指定的少数类进行过采样
            # 但 SMOTE 默认会对所有类按比例过采样，这里可以手动筛选
            # 简单起见，使用 auto
            pass
        sm = SMOTE(k_neighbors=k_neighbors, sampling_strategy=sampling_strategy, random_state=42)
        X_res, y_res = sm.fit_resample(embeddings, labels)
        return X_res, y_res

    def train(self, trainLoader, valLoader, epochs, classWeights, lr=0.001, smoteFreq=5, smoteK=5, useSmote = False):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        scaler = GradScaler()

        #开始训练
        self.history['trainLosses'] = []     #在训练集上的损失值
        self.history['valLosses'] = []       #在验证集上的损失值
        self.history['trainAccs'] = []       #训练集上的准确率
        self.history['valAccs'] = []         #验证集上的准确率
        #将验证过程中每一轮里的混淆矩阵记录下来，并添加正确率创新高时的矩阵
        baseAcc = 0

        # 初始化进度条
        pbar = tqdm(total=epochs, desc='训练进度', unit='epoch')

        # 配置 profiler
        # prof = profiler.profile(
        #     activities=[
        #         profiler.ProfilerActivity.CPU,
        #         profiler.ProfilerActivity.CUDA,
        #     ],
        #     schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        #     on_trace_ready=profiler.tensorboard_trace_handler('./log'),  # 保存到 tensorboard
        #     record_shapes=True,
        #     with_stack=True,
        #     profile_memory=True
        # )
        # prof.start()  # 手动启动

        for epoch in range(epochs):
            """
                将模型设置为训练模式，该模式下：
                正则化层会丢弃一部分神经元
            """
            correct_gpu = torch.tensor(0, device=self.device)  # 在 GPU 上累积
            total_gpu = torch.tensor(0, device=self.device)


            self.model.train()
            totalLoss = 0       #计算每个batch的损失值


            # ------------ 阶段1：训练 ------------
            for batch in trainLoader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()


                with autocast():
                    if self.modelName == 'GCNModel':
                        out, _ = self.model(batch.x, batch.edge_index, batch.batch)
                    elif self.modelName == 'GraphTransFormer':
                        out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    elif self.modelName == 'RGCNModel':
                        out = self.model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                    elif self.modelName == 'GATModel':
                        out, _ = self.model(batch.x, batch.edge_index, batch.batch)
                    else:
                        raise ValueError(f'{self.modelName}是不存在的模型名字')
                    loss = F.nll_loss(out, batch.y, weight=classWeights)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()


                totalLoss += loss.item()  # loss.item() 仍需，但次数少
                pred = out.argmax(dim=1)
                correct_gpu += (pred == batch.y).sum()
                total_gpu += batch.y.size(0)

                # Profiler step
                # if epoch == 0:
                #     prof.step()

            #计算平均损失值，计算正确率，并加入到类属性中
            correct = correct_gpu.item()
            total = total_gpu.item()
            trainLoss = totalLoss / len(trainLoader)
            trainAcc = correct / total
            self.history['trainLosses'].append(trainLoss)
            self.history['trainAccs'].append(trainAcc)

            # --- 阶段2：每 smote_freq 轮执行 SMOTE ---
            if ((epoch + 1) % smoteFreq == 0) & useSmote:
                # 收集嵌入
                X_emb, y_labels = self.collectEmbed(trainLoader)
                # 执行 SMOTE
                X_res, y_res = self.smoteAugment(X_emb, y_labels, k_neighbors=smoteK)
                # 转为 tensor
                X_res = torch.tensor(X_res, dtype=torch.float32).to(self.device)
                y_res = torch.tensor(y_res, dtype=torch.long).to(self.device)

                # 冻结编码器部分
                for param in self.model.parameters():
                    param.requires_grad = False
                for param in self.model.classifier.parameters():
                    param.requires_grad = True

                # 仅训练分类器
                classifier_optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=lr)
                for _ in range(5):  # 内循环次数
                    classifier_optimizer.zero_grad()
                    logits = self.model.classifier(X_res)
                    loss = F.nll_loss(logits, y_res)
                    loss.backward()
                    classifier_optimizer.step()

                # 恢复整个模型的梯度
                for param in self.model.parameters():
                    param.requires_grad = True

            #验证阶段
            valLoss, valAcc, confus, probs, tlabels = self.validate(valLoader, classWeights)
            if valAcc > baseAcc:
                self.confusions.append(confus)
                self.valProbs.append(probs)
                self.valLabels.append(tlabels)
                baseAcc = valAcc
                self.bestModel = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            self.history['valLosses'].append(valLoss)
            self.history['valAccs'].append(valAcc)
            scheduler.step()
            pbar.update(1)

            # 如果只 profile 第一个 epoch，完成后停止 profiler 并退出循环（可选）
            # if epoch == 0:
            #     prof.stop()
            #     # 打印简洁的结果表格
            #     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
            #     # 如果不希望继续训练，可以 break；如果想继续训练，则注释掉下面的 break
            #     # break

        # pbar.close()

        # prof.stop()  # 确保停止（如果上面 break 了，这里可能不会执行）

            # #每训练10轮输出一次信息
            # if (epoch + 1) % 10 == 0:
            #     print(f"训练轮数：{epoch + 1:03d}，训练损失：{trainLoss:.4f}，训练准确率：{trainAcc:.4f}，"
            #           f"验证损失：{valLoss:.4f}，验证准确率：{valAcc:.4f}")

    """模型评估器，取消梯度计算直接显示分类结果"""
    def validate(self, valLoader, classWeights):
        """
            将模型设置为评估模式，该模式下：
            正则化层会保持所有神经元激活
            其他三个参数同训练模式
        """
        self.model.eval()
        totalLoss = 0.0
        correct_gpu = torch.tensor(0, device=self.device)
        total_gpu = torch.tensor(0, device=self.device)

        #模型验证阶段，取消梯度计算，之后的步骤同训练阶段
        confusion = np.zeros((3, 3), dtype=int)
        allProbs = []
        allLabels = []
        with torch.no_grad():
            for batch in valLoader:
                batch = batch.to(self.device)

                if self.modelName == 'GCNModel':
                    out, _ = self.model(batch.x, batch.edge_index,  batch.batch)
                elif self.modelName == 'GraphTransFormer':
                    out = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                elif self.modelName == 'RGCNModel':
                    out = self.model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                elif self.modelName == 'GATModel':
                    out, _ = self.model(batch.x, batch.edge_index, batch.batch)
                else:
                    raise ValueError(f'{self.modelName}是不存在的模型名字')

                probs = torch.exp(out)
                allProbs.append(probs.cpu().numpy())
                allLabels.append(batch.y.cpu().numpy())
                loss = F.nll_loss(out, batch.y, weight=classWeights)

                #计算平均损失和正确率
                totalLoss += loss.item()
                pred = out.argmax(dim=1)
                correct_gpu += (pred == batch.y).sum()
                total_gpu += batch.y.size(0)
                for i in range(len(batch.y)):
                    trueLabel = batch.y[i].item()
                    predLabel = pred[i].item()
                    confusion[trueLabel, predLabel] += 1

        allProbs = np.concatenate(allProbs, axis=0)
        allLabels = np.concatenate(allLabels, axis=0)

        valLoss = totalLoss / len(valLoader)
        valAcc = correct_gpu.item() / total_gpu.item()
        return valLoss, valAcc, confusion, allProbs, allLabels

    #模型测试阶段,直接将测试数据放入模型评估器返回结果
    def test(self, testLoader, classWeights):
        return self.validate(testLoader, classWeights)

    """统计每个类别的样本数量"""
    def countClasses(self, dataLoader):

        classCounts = {}
        for batch in dataLoader:
            for label in batch.y.unique():
                label = label.item()
                count = (batch.y == label).sum().item()
                classCounts[label] = classCounts.get(label, 0) + count
        return classCounts
