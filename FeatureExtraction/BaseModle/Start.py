import os
import torch
import json
from torch.utils.data import TensorDataset
from FeatureExtraction.BaseModle.TrainEmhance import CombinationOversampler
import logger
from torch.utils.data import DataLoader
from FeatureExtraction.Model.ResultSave import PlotHistory
from FeatureExtraction.BaseModle.CNNModel import CNN1D
from FeatureExtraction.BaseModle.LoadData import LoadData
from FeatureExtraction.BaseModle.training import trainer
from FeatureExtraction.BaseModle.MLPModel import PureMLP
from FeatureExtraction.BaseModle.Transform import Transformer1D
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
import numpy as np
import random
from torch_geometric.graphgym.config import cfg
from registry import register


"""训练过程"""
@register('basetrain')
def startTrain():
    #初始化日志
    logger.setup_logging(cfg.logs.outDir, cfg.logs.backupCount)
    log = logger.get_logger(__name__)
    if cfg.logs.normInfo:
        normConfigs = {
            'datas': cfg.create.dataName,
            'es': cfg.es,
            'nameDim': cfg.create.nameDim,
            'textDim': cfg.create.textDim
        }
        logger.log_config(log, normConfigs, title='其他配置信息')

    #设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    ld = LoadData()
    X, y = ld.loadOriginalData('data/7131个沙盘特征汇总+标注-两位标注全部.xlsx', 'trainData/sandBoxData/label.csv', 5)
    num_classes = len(np.unique(y))
    print(f"类别数: {num_classes}")

    """
        配置k折交叉验证
        n_splits:折数
        shuffle:是否打乱数据集
        random_state:打乱随机数种子
    """
    if cfg.train.randomState is None:
        state = random.randint(1, 1000)
    else:
        state = cfg.train.randomState

    X_train, y_train, X_test, y_test = splitTrain(X, y, random_state=state)
    kf = cfg.train.kfolds
    kFold = StratifiedKFold(n_splits=kf, shuffle=True, random_state=state)

    """
        数据集划分，按照训练集:测试集=8:2
        开启十折交叉验证
        测试集固定
        记录每一折下的测试集准确率和模型，保存最佳的模型
    """
    # 配置需要保存训练配置的文件夹
    modelPath = 'model/CNNModel'

    if cfg.result.expNum == None:
        try:
            # 打开本地保存的文件，读取实验次数，并让次数加一
            with open(f'{modelPath}/experiment.txt', 'r') as f:
                content = f.read().strip()
                num = int(content.split(':')[1])
                newNum = num + 1
        except:
            # 若没有这个文件，直接从次数1开始
            newNum = 1
    else:
        newNum = cfg.result.expNum

    #设置混淆矩阵的路径
    conPath = f'{modelPath}/experiment{str(newNum)}/混淆矩阵'
    if not os.path.exists(conPath):
        os.makedirs(conPath)

    if cfg.emhance.randomSeed is None:
        emState = random.randint(1, 1000)
    else:
        emState = cfg.emhance.randomSeed

    #一些需要记录的信息
    his = []
    foldCon = []
    testCons = []
    testAccs = []
    valProbs = []
    valLabels = []
    testProbs = []
    testLabels = []

    X_test = X_test[:, np.newaxis, :]
    for fold, (train_idx, val_idx) in enumerate(kFold.split(X_train, y_train)):
        print(f'\n====================第{fold + 1}/{kf}折交叉验证====================')

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # 过采样前检查
        print("检查训练集 X_tr 是否包含非数字:")
        print("NaN数量:", np.isnan(X_tr).sum())
        print("Inf数量:", np.isinf(X_tr).sum())

        # 打印第一个样本的特征值（前10个）
        print("第一个样本的前10个特征:", X_tr[0][:10])

        # 检查是否有非数值类型（如字符串）
        if X_tr.dtype.kind not in 'ifc':
            print("警告：X_tr 的数据类型不是数值型，当前类型:", X_tr.dtype)

        inf_cols = np.where(np.isinf(X).any(axis=0))[0]
        print("包含 Inf 的列索引:", inf_cols)
        # 查看这些列的值范围
        for col in inf_cols:
            print(f"列 {col}: min={X[:, col].min()}, max={X[:, col].max()}, 包含 Inf 的样本数={np.isinf(X[:, col]).sum()}")

        X_tr = X_tr[:, np.newaxis, :]
        X_val = X_val[:, np.newaxis, :]


        print("X_tr shape:", X_tr.shape)
        print("X_val shape:", X_val.shape)
        print("X_test shape:", X_test.shape)
        print("y_tr shape:", y_tr.shape)

        oversampler = CombinationOversampler(threshold=0.95, seed=emState)
        X_tr, y_tr = oversampler.fit_resample(X_tr, y_tr)

        train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                      torch.tensor(y_tr, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.long))

        trainLoader = DataLoader(train_dataset, batch_size=cfg.train.batchSize, shuffle=True)
        valLoader = DataLoader(val_dataset, batch_size=cfg.train.batchSize, shuffle=True)

        if cfg.model.modelName == 'CNNModel':
            model = CNN1D(inputDim=X.shape[1], numClasses=num_classes,
                          num_layers=cfg.model.numLayers, use_residual=False,
                          dropout=0.1).to(device)
        elif cfg.model.modelName == 'MLPModel':
            model = PureMLP(X.shape[1], num_classes,
                            [256, 128, 64, 32]).to(device)
        elif cfg.model.modelName == 'Transform':
            model = Transformer1D(cfg.model.hiddenDim, cfg.model.numHeads,
                                  cfg.model.numLayers, num_classes).to(device)
        else:
            raise ValueError(f'{cfg.model.modelName}是不存在的模型名字')
        #输出模型信息
        if cfg.logs.modelInfo:
            logger.log_model_summary(log, fold + 1, model, input_shape=(None, cfg.model.nodeDim))

        #训练模型
        trains = trainer(model, device, cfg.model.modelName)

        print('开始训练……')
        history, confus= trains.train(cfg.train.epochs, trainLoader, valLoader)
        foldCon.append(confus)
        valProbs.append(trains.valProbs)
        valLabels.append(trains.valLabels)

        #测试数据
        testLoader = DataLoader(test_dataset, batch_size=cfg.train.batchSize, shuffle=False)
        testcon, testAcc, probs, labels = trains.test(testLoader)
        testCons.append(testcon)
        testAccs.append(testAcc)
        testProbs.append(probs)
        testLabels.append(labels)

        his.append(history)

    bestFold = np.argmax(testAccs)
    bestProbs = testProbs[bestFold]
    bestLabels = testLabels[bestFold]

    probsList = [batch.tolist() for batch in bestProbs]
    labelsList = [batch.tolist() for batch in bestLabels]

    # 构建字典
    ROCdata = {
        "probs": probsList,
        "labels": labelsList
    }
    ROCResult = 'ROCResult'
    if not os.path.exists(ROCResult):
        os.makedirs(ROCResult)
    modelROC = f'{ROCResult}/{cfg.model.modelName}.json'
    # 保存为 JSON 文件
    with open(modelROC, "w") as f:
        json.dump(ROCdata, f, indent=4)

    ph = PlotHistory()
    if cfg.result.accLoss:
        # 绘制训练历史
        ph.average_history_dicts(his)
        ph.plotTrainHistory(f'{modelPath}/experiment{str(newNum)}/training_history.png')

    path = f'{modelPath}/experiment{str(newNum)}/ROC'
    if cfg.result.valROC:
        print("绘制验证集的ROC曲线中......")
        ph.dealValROC(valProbs, valLabels, path)
    if cfg.result.testROC:
        print("绘制测试集的ROC曲线中......")
        ph.dealTestROC(testProbs, testLabels, path)

    if cfg.result.testConfus:
        testConPath = os.path.join(conPath, '交叉验证下的测试集混淆矩阵')
        os.mkdir(testConPath)
        i = 1
        for con in testCons:
            info = ph.getInfo(con)
            text = ''
            testNum = 0
            for k in range(len(info)):
                text += f"类别{k}的精确率：{info[k]['accurate']:.4f}，" \
                    f"召回率：{info[k]['recall']:.4f}，F1分数：{info[k]['F1']:.4f}\n"
                testNum += con[k, k]
            ta = testNum / np.sum(con)
            testPhoto = os.path.join(testConPath, f'第{i}折的测试集混淆矩阵.png')
            title = f'测试集正确率为{ta:.4f}时的混淆矩阵'
            className = ['0', '1', '2']
            ph.plotValConfus(con, title, testPhoto, text, className)
            i += 1


    #成功保存好全部文件后再将新的次数写入
    with open(f'{modelPath}/experiment.txt', 'w') as f:
        f.write(f'num:{newNum}')

    print("训练完成")


def splitTrain(X, y, majority_ratio=0.8, minority_ratio=0.8, random_state=42):
    """
    根据类别样本数量自动识别多数类和少数类，并按照指定比例分配训练集和测试集。
    多数类保留 majority_ratio 的比例到训练集，每个少数类保留 minority_ratio 的比例到训练集。
    返回打乱顺序后的训练集和测试集。
    """
    np.random.seed(random_state)

    # 统计每个类别的样本索引
    classes = np.unique(y)
    class_indices = {c: np.where(y == c)[0] for c in classes}

    # 找出多数类（样本数最多的类）
    majority_class = max(classes, key=lambda c: len(class_indices[c]))
    minority_classes = [c for c in classes if c != majority_class]

    train_indices = []
    test_indices = []

    # 处理多数类
    idx = class_indices[majority_class]
    n_train = int(len(idx) * majority_ratio)
    # 随机选择训练索引
    selected = np.random.choice(idx, size=n_train, replace=False)
    train_indices.extend(selected)
    test_indices.extend(list(set(idx) - set(selected)))

    # 处理每个少数类
    for c in minority_classes:
        idx = class_indices[c]
        n_train = int(len(idx) * minority_ratio)
        selected = np.random.choice(idx, size=n_train, replace=False)
        train_indices.extend(selected)
        test_indices.extend(list(set(idx) - set(selected)))

    # 转换为 numpy 数组并打乱
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    train_indices = shuffle(train_indices, random_state=random_state)
    test_indices = shuffle(test_indices, random_state=random_state)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print("训练集类别分布：", {c: np.sum(y_train == c) for c in classes})
    print("测试集类别分布：", {c: np.sum(y_test == c) for c in classes})

    return X_train, y_train, X_test, y_test
