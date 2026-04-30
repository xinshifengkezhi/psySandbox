import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm

class trainer:
    def __init__(self, model, device, modelName):
        self.model = model
        self.device = device
        self.modelName = modelName
        self.criterion = nn.CrossEntropyLoss()
        self.valProbs = []
        self.valLabels = []

    def train(self, epochs, train_loader, val_loader):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        history = {
            'trainLosses': [],
            'trainAccs': [],
            'valLosses': [],
            'valAccs': []
        }

        confusions = []  # 保存所有的混淆矩阵
        baseAcc = 0
        pbar = tqdm(total=epochs, desc='训练进度', unit='epoch')
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_loss = train_loss / total
            train_acc = 100. * correct / total

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            confus = np.zeros((3, 3), dtype=int)
            allProbs = []
            allLabels = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    allProbs.append(probs.cpu().numpy())
                    allLabels.append(targets.cpu().numpy())
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    t = targets.cpu().numpy()
                    p = predicted.cpu().numpy()
                    np.add.at(confus, (t, p), 1)

            val_loss = val_loss / total
            val_acc = 100. * correct / total
            scheduler.step()

            allProbs = np.concatenate(allProbs, axis=0)
            allLabels = np.concatenate(allLabels, axis=0)

            if val_acc > baseAcc:
                baseAcc = val_acc
                confusions.append(confus)
                self.valProbs.append(allProbs)
                self.valLabels.append(allLabels)

            # 记录历史
            history['trainLosses'].append(train_loss)
            history['trainAccs'].append(train_acc)
            history['valLosses'].append(val_loss)
            history['valAccs'].append(val_acc)

            pbar.update(1)

        return history, confusions


    def test(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        confus = np.zeros((3, 3), dtype=int)
        allProbs = []
        allLabels = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                allProbs.append(probs.cpu().numpy())
                allLabels.append(targets.cpu().numpy())
                loss = self.criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                t = targets.cpu().numpy()
                p = predicted.cpu().numpy()
                np.add.at(confus, (t, p), 1)

        allProbs = np.concatenate(allProbs, axis=0)
        allLabels = np.concatenate(allLabels, axis=0)
        test_loss = test_loss / total
        test_acc = 100. * correct / total
        print(f"测试集 Loss: {test_loss:.4f}, 测试集 Acc: {test_acc:.2f}%")
        return confus, test_acc, allProbs, allLabels
