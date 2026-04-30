import datetime
import json
import os
import yaml
import torch
import numpy as np
from FeatureExtraction.Model.GCNModel import SandBoxModel
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize



"""定义一个绘图类，用于处理训练器里保存好的历史训练结果"""
class PlotHistory:
    def __init__(self):
        self.trainLosses = []  # 在训练集上的损失值
        self.valLosses = []  # 在验证集上的损失值
        self.trainAccs = []  # 训练集上的准确率
        self.valAccs = []  # 验证集上的准确率

    """
        将多个字典中对应的四项键值列表逐项求平均
            history: 包含多个字典的列表，
            每个字典都有 'trainLosses', 'valLosses', 'trainAccs', 'valAccs' 四个键 
    """
    def average_history_dicts(self, his):

        # 获取字典中列表的长度
        hisLength = len(his[0]['trainLosses'])

        # 对每个位置进行求平均
        for i in range(hisLength):

            trainLosses = []
            valLosses = []
            trainAccs = []
            valAccs = []
            for d in his:
                # 训练损失
                trainLosses.append(d['trainLosses'][i])
                # 验证损失
                valLosses.append(d['valLosses'][i])
                # 训练准确率
                trainAccs.append(d['trainAccs'][i])
                # 验证准确率
                valAccs.append(d['valAccs'][i])

            # 训练损失
            tl = sum(trainLosses) / len(trainLosses)
            self.trainLosses.append(tl)
            # 验证损失
            vl = sum(valLosses) / len(valLosses)
            self.valLosses.append(vl)
            # 训练准确率
            ta = sum(trainAccs) / len(trainAccs)
            self.trainAccs.append(ta)
            # 验证准确率
            va = sum(valAccs) / len(valAccs)
            self.valAccs.append(va)

    # 绘制结果曲线
    def plotTrainHistory(self, resultPath):
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.trainLosses, label='训练集损失')
        ax1.plot(self.valLosses, label='验证集损失')
        ax1.set_xlabel('训练轮数')
        ax1.set_ylabel('平均损失')
        ax1.legend()
        ax1.set_title('训练集和验证集上的平均损失')

        ax2.plot(self.trainAccs, label='训练集正确率')
        ax2.plot(self.valAccs, label='验证集准确率')
        ax2.set_xlabel('训练轮数')
        ax2.set_ylabel('正确率')
        ax2.legend()
        ax2.set_title('训练集和验证集上的正确率')

        plt.tight_layout()
        plt.savefig(resultPath, dpi=300, bbox_inches='tight')
        # plt.show()

    def drawROCClass(self, results_dir='../GNN-ROCResult', output_dir='../GNN-ROCClass'):
        """
        对每个类别，绘制所有模型的 ROC 曲线叠加图

        Args:
            results_dir: 存储各模型结果 JSON 文件的目录，文件名即为模型名
            output_dir: 输出图片的目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 获取所有 JSON 文件
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]


        model_results = {}  # 模型名 -> {"probs": np.ndarray, "labels": np.ndarray}
        n_classes = None

        for file in json_files:
            model_name = os.path.splitext(file)[0]
            file_path = os.path.join(results_dir, file)
            with open(file_path, 'r') as f:
                data = json.load(f)

            # 将列表转换为 numpy 数组
            probs = np.array(data['probs'])
            labels = np.array(data['labels'])

            # 检查维度
            if probs.ndim == 3:
                # 如果保存的是 batch 列表形式，需要拼接
                # 假设形状为 (num_batches, batch_size, num_classes)
                probs = np.concatenate(probs, axis=0)
                labels = np.concatenate(labels, axis=0)
            elif probs.ndim == 2:
                # 已经是 (N, num_classes) 形式
                pass
            else:
                raise ValueError(f"Unexpected probs shape for {model_name}: {probs.shape}")

            model_results[model_name] = {'probs': probs, 'labels': labels}

            # 确定类别数
            cur_n_classes = probs.shape[1]
            if n_classes is None:
                n_classes = cur_n_classes
            elif n_classes != cur_n_classes:
                print(f"Warning: {model_name} has {cur_n_classes} classes, expected {n_classes}. Skipping.")
                continue

        if n_classes is None:
            raise ValueError("No valid model data loaded.")

        # 对每个类别绘制 ROC 曲线
        for class_idx in range(n_classes):
            plt.figure(figsize=(10, 8))

            for model_name, res in model_results.items():
                probs = res['probs']
                labels = res['labels']

                # 二值化标签：当前类别为正类，其余为负类
                y_true = (labels == class_idx).astype(int)

                # 计算 ROC 曲线
                fpr, tpr, _ = roc_curve(y_true, probs[:, class_idx])
                roc_auc = auc(fpr, tpr)

                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

            # 绘制对角线（随机分类器）
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curves for Class {class_idx}')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)

            # 保存图片
            save_path = os.path.join(output_dir, f'ROC_class_{class_idx}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved: {save_path}")

    def dealValROC(self, allProbs, allLabels, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(len(allProbs)):
            flodPath = f'{path}/第{i + 1}折下的验证集ROC曲线'
            if not os.path.exists(flodPath):
                os.makedirs(flodPath)
            flodProbs = allProbs[i]
            flodLabels = allLabels[i]
            for j in range(len(flodProbs)):
                ROCPath = f'{flodPath}/验证集第{j + 1}次创新高后的ROC曲线.png'
                self.drawROC(flodProbs[j], flodLabels[j], ROCPath)

    def dealTestROC(self, allProbs, allLabels, path):
        testPath = f'{path}/交叉验证下测试集的ROC曲线'
        if not os.path.exists(testPath):
            os.makedirs(testPath)
        for i in range(len(allProbs)):
            ROCPath = f'{testPath}/第{i + 1}折下的测试集ROC曲线.png'
            self.drawROC(allProbs[i], allLabels[i], ROCPath)

    """绘制ROC曲线"""
    def drawROC(self, allProbs, allLabels, drawPath):
        nClasses = allProbs.shape[1]
        yOnehot = label_binarize(allLabels, classes=range(nClasses))

        #计算每个类别的ROC曲线和AUC
        fpr = dict()
        tpr = dict()
        rocAuc = dict()
        for i in range(nClasses):
            fpr[i], tpr[i], _ = roc_curve(yOnehot[:, i], allProbs[:, i])
            rocAuc[i] = auc(fpr[i], tpr[i])

        #计算宏观平均ROC曲线（对所有类别取平均）
        allFpr = np.unique(np.concatenate([fpr[i] for i in range(nClasses)]))
        meanTpr = np.zeros_like(allFpr)
        for i in range(nClasses):
            meanTpr += np.interp(allFpr, fpr[i], tpr[i])
        meanTpr /= nClasses
        macroAuc = auc(allFpr, meanTpr)

        #计算微观平均ROC曲线
        fprMicro, tprMicro, _ = roc_curve(yOnehot.ravel(), allProbs.ravel())
        microAuc = auc(fprMicro, tprMicro)

        #绘图
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10', nClasses)
        for i, color in zip(range(nClasses), colors.colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'Class {i} (AUC = {rocAuc[i]:.2f})')

        plt.plot(fprMicro, tprMicro, color='deeppink', linestyle=':', linewidth=4,
                 label=f'micro-average (AUC = {microAuc:.2f})')
        plt.plot(allFpr, meanTpr, color='navy', linestyle='--', linewidth=4,
                 label=f'macro-average (AUC = {macroAuc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig(drawPath, dpi=300, bbox_inches='tight')


    # 将所有的混淆矩阵绘制下来，并添加附加信息
    def drawConfus(self, photoPath, foldCon):
        # 取出所有的混淆矩阵
        # i:代表折数，j:代表矩阵的次序,k代表类别
        i = 1
        for conDict in foldCon:
            fileFirst = os.path.join(photoPath, f'第{i}折交叉验证的结果')
            os.mkdir(fileFirst)
            j = 1
            for con in conDict:
                text = ''
                info = self.getInfo(con)
                valNum = 0
                for k in range(len(info)):
                    text += f"类别{k}的精确率：{info[k]['accurate']:.4f}，" \
                        f"召回率：{info[k]['recall']:.4f}，F1分数：{info[k]['F1']:.4f}\n"
                    valNum += con[k, k]

                va = valNum / np.sum(con)
                fileSecond = os.path.join(fileFirst, f'验证集正确率第{j}次创新高.png')
                title = f'验证集正确率为{va:.4f}时的混淆矩阵'
                className = ['0', '1', '2']
                self.plotValConfus(con, title, fileSecond, text, className)
                j += 1
            i += 1

    def plotValConfus(self, confusion, title, resultPath, text, className = None):
        """
        绘制验证集中获得的的混淆矩阵
        :param confusion: 混淆矩阵
        :param title: 标题
        :param resultPath: 保存图片的位置
        :param text: 额外的文本内容
        :param className: 类型列表
        :return:
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        plt.figure(figsize=(8, 6))
        plt.imshow(confusion, interpolation='nearest', cmap='Blues')
        plt.title(title)
        plt.colorbar()
        tickMarks = np.arange(len(className))
        plt.xticks(tickMarks, className, rotation=45)
        plt.yticks(tickMarks, className)

        #为每个格子里添加数字
        thresh = confusion.max() / 2
        for i in range(confusion.shape[0]):
            for j in range(confusion.shape[1]):
                color = 'white' if confusion[i, j] > thresh else 'black'
                plt.text(j, i, format(confusion[i, j], 'd'),
                        ha='center', va='center', color=color,
                        fontsize=8, fontweight='bold')

        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')

        #为混淆矩阵添加一些对应的附加文本信息
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.5, -0.1, text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='center', bbox=props)

        plt.savefig(resultPath, dpi=300, bbox_inches='tight')

    """传入混淆矩阵，计算其他信息，包括精确率、召回率和F1分数，返回列表"""
    def getInfo(self, confusion):
        info = []
        prep = np.sum(confusion, axis=0)  # 沿着列相加，用于计算精确率
        true = np.sum(confusion, axis=1)  # 沿着行相加，用于计算召回率
        for i in range(confusion.shape[0]):
            classInfo = {}
            classInfo['accurate'] = confusion[i, i] / true[i]  # 精确率
            classInfo['recall'] = confusion[i, i] / prep[i]  # 召回率
            classInfo['F1'] = (2 * classInfo['accurate'] * classInfo['recall']) / (
                        classInfo['accurate'] + classInfo['recall'])  # F1分数
            info.append(classInfo)
        return info

"""保存模型的一些信息"""
class ModelSave:
    def __init__(self, modelPath):
        if not os.path.exists(modelPath):
            os.makedirs(modelPath)
        self.modelPath = modelPath

    """保存训练后的相关文件"""
    def saveModelHistory(self, model, trainer, config, experNum, saveConfig):
        filePath = os.path.join(self.modelPath, experNum)

        if not os.path.exists(filePath):
            os.makedirs(filePath)

        if saveConfig['mw']:
            #1.模型权重文件
            torch.save(trainer.bestModel, f'{filePath}/modelWeights.pth')

        if saveConfig['fm']:
            #2.完整模型结构文件
            torch.save(model, f'{filePath}/fullModel.pth')

        if saveConfig['tc']:
            #3.训练配置文件
            with open(f'{filePath}/trainConfig.json', 'w') as f:
                json.dump(config, f, indent=2)

        if saveConfig['th']:
            #4.训练历史文件
            trainHistory = {
                'trainLosses': trainer.history['trainLosses'],
                'valLosses': trainer.history['valLosses'],
                'trainAccs': trainer.history['trainAccs'],
                'valAccs': trainer.history['valAccs']
            }
            with open(f'{filePath}/trainHistory.json', 'w') as f:
                json.dump(trainHistory, f, indent=2)

        if saveConfig['os']:
            #5.优化器状态文件
            torch.save(trainer.optimizer.state_dict(), f'{filePath}/opteimizerState.pth')

        if saveConfig['mm']:
            #6.模型元数据文件
            date = datetime.datetime.now()
            metadata = {
                'modelName': 'SandBoxModel',
                'version': '1.0',
                'createdDate': f'{date.year}年{date.month}月{date.day}日',
                'inputDim': config['model']['nodeDim'],
                'outputDim': config['model']['numClasses'],
                'performance': {
                    'finalTrainAcc': trainer.history['trainAccs'][-1],
                    'finalValAcc': trainer.history['valAccs'][-1]
                }
            }
            with open(f'{filePath}/modelMetadata.yaml', 'w') as f:
                yaml.dump(metadata, f)

        print(f'模型相关文件已经保存在{filePath}中')

