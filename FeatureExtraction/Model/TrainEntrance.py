import os
import torch
import logger
import json
from torch_geometric.data import DataLoader
from FeatureExtraction.Model.DataDeal import DataProcessor
from FeatureExtraction.Model.GCNModel import SandBoxModel
from FeatureExtraction.Model.Training import GraphTrainer
from FeatureExtraction.Model.RGCNModel import RelateModel
from FeatureExtraction.Model.GATModel import GATModel
from FeatureExtraction.Model.ResultSave import PlotHistory, ModelSave
from FeatureExtraction.Model.TrainEmhance import trainEmb
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from FeatureExtraction.Model.GTModel import GraphTransFormer
from sklearn.utils import shuffle
import numpy as np
import random
import math
from torch_geometric.graphgym.config import cfg
from registry import register

"""训练过程"""
@register('trains')
def startTrain():
    #初始化日志
    logger.setup_logging(cfg.logs.outDir, cfg.logs.backupCount)
    log = logger.get_logger(__name__)
    if cfg.logs.normInfo:
        normConfigs = {
            'datas': cfg.emhance.dataName,
            'es': cfg.es,
            'nameDim': cfg.create.nameDim,
            'textDim': cfg.create.textDim
        }
        logger.log_config(log, normConfigs, title='其他配置信息')

    #设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备：{device}')

    #设置数据的路径
    trainDatas = os.path.join(cfg.trainDataDir, cfg.emhance.dataName)
    # testDatas = os.path.join(cfg.trainDataDir, cfg.emhance.testData)

    #加载和处理数据
    trainProcessor = DataProcessor(cfg.model.nodeDim, trainDatas, cfg.model.numRelations)
    print("正在加载训练数据……")
    trainGraphs = trainProcessor.loadData(trainDatas, cfg.train.deleteEdge, cfg.es)

    # testProcessor = DataProcessor(cfg.model.nodeDim, testDatas, cfg.model.numRelations)
    # print("正在加载测试数据……")
    # testGraphs = testProcessor.loadData(testDatas, cfg.train.deleteEdge, cfg.es)

    if cfg.train.randomState is None:
        state = random.randint(1, 1000)
    else:
        state = cfg.train.randomState

    traValGraphs, testGraphs, traLabDict, testLabDict = splitTrain(trainGraphs, random_state=state)

    #选择粗糙的复制方法来增加样本数
    # graphs = processor.copyData(graphs)

    #输出数据集的基本信息
    if cfg.logs.dataInfo:
        logger.log_data_info(log, cfg.dataName,
                             num_samples=len(traValGraphs),
                             node_features=cfg.model.nodeDim,
                             num_classes=cfg.model.numClasses)

    """
        配置k折交叉验证
        n_splits:折数
        shuffle:是否打乱数据集
        random_state:打乱随机数种子
    """

    kf = cfg.train.kfolds
    kFold = StratifiedKFold(n_splits=kf, shuffle=True, random_state=state)
    # kFold = KFold(n_splits=kfolds, shuffle=True, random_state=state)

    """
        数据集划分，按照训练集:测试集=8:2
        开启十折交叉验证
        测试集固定
        记录每一折下的测试集准确率和模型，保存最佳的模型
    """
    traValLabels = [graph.y.item() if torch.is_tensor(graph.y) else graph.y for graph in traValGraphs]
    # traValGraphs, testGraphs, traValLabels, testLabels = train_test_split(
    #     traValGraphs, traValLabels, test_size=0.2, random_state=state, stratify=labels
    # )

    # traValGraphs, testGraphs = train_test_split(graphs, test_size=0.2, random_state=state)


    # 配置需要保存训练配置的文件夹
    modelPath = cfg.result.outDir

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
    allTestAcc = []
    allModel = []
    allTrain = []
    his = []
    testAccs = []
    testLosses = []
    foldCon = []
    testCon = []
    valProbs = []
    valLabels = []
    testProbs = []
    testLabels = []
    useSmote = False

    for fold, (train_idx, val_idx) in enumerate(kFold.split(traValGraphs, traValLabels)):
        print(f'\n====================第{fold + 1}/{kf}折交叉验证====================')

        trainGraphs = [traValGraphs[i] for i in train_idx]
        valGraphs = [traValGraphs[i] for i in val_idx]

        #平衡训练集
        if not useSmote:
            print('数据平衡......')
            te = trainEmb()
            trainGraphs = te.combData(trainGraphs, emState, cfg.train.threshold)

        #输出数据划分信息
        if cfg.logs.dataSplitInfo:
            logger.log_data_split(log, fold + 1,
                                  len(trainGraphs), len(valGraphs), len(testGraphs),
                                  train_ratio=0.8*(1-(1/kf)),
                                  val_ratio=0.8/kf,
                                  test_ratio=0.2)

        #创建数据加载器
        trainLoader = DataLoader(trainGraphs, batch_size=cfg.train.batchSize, shuffle=True)
        valLoader = DataLoader(valGraphs, batch_size=cfg.train.batchSize, shuffle=False)

        if cfg.model.modelName == 'GCNModel':
            model = SandBoxModel(
                cfg.model.nodeDim,
                cfg.model.hiddenDim,
                cfg.model.numClasses,
                cfg.model.numLayers,
                cfg.model.numRelations,
                cfg.model.alp,
                cfg.model.probably
            ).to(device)
        elif cfg.model.modelName == 'GraphTransFormer':
            model = GraphTransFormer(
                cfg.model.nodeDim,
                cfg.model.hiddenDim,
                cfg.model.numClasses,
                cfg.model.numLayers,
                cfg.model.probably,
                cfg.model.numHeads,
                cfg.model.edgeDim
            ).to(device)
        elif cfg.model.modelName == 'RGCNModel':
            model = RelateModel(
                cfg.model.nodeDim,
                cfg.model.hiddenDim,
                cfg.model.numRelations,
                cfg.model.numClasses,
                cfg.model.numLayers,
                cfg.model.alp,
                cfg.model.probably,
            ).to(device)
        elif cfg.model.modelName == 'GATModel':
            model = GATModel(
                cfg.model.nodeDim,
                cfg.model.hiddenDim,
                cfg.model.numClasses,
                cfg.model.numLayers,
                cfg.model.numHeads,
                cfg.model.probably,
                cfg.model.alp,
            ).to(device)
        else:
            raise ValueError(f'{cfg.model.modelName}是不存在的模型名字')

        #输出模型信息
        if cfg.logs.modelInfo:
            logger.log_model_summary(log, fold + 1, model, input_shape=(None, cfg.model.nodeDim))

        #训练模型
        trainer = GraphTrainer(model, device, cfg.model.numRelations, cfg.model.modelName)

        # 引入类别权重，计算训练集下各类别的权重
        if cfg.train.classWeights:
            classCounts = trainer.countClasses(trainLoader)
            totalSamples = sum(classCounts.values())
            classWeight = torch.tensor([math.log(1.2 + (totalSamples / (len(classCounts) * count)))
                                         for count in classCounts.values()],
                                        device=device)
        else:
            #将不再使用类别权重式的交叉熵损失
            classWeight = None

        print('开始训练……')
        trainer.train(trainLoader, valLoader, cfg.train.epochs, classWeight, lr=cfg.train.lr)
        foldCon.append(trainer.confusions)

        valProbs.append(trainer.valProbs)
        valLabels.append(trainer.valLabels)
        #测试数据
        testLoader = DataLoader(testGraphs, batch_size=cfg.train.batchSize, shuffle=False)
        testLoss, testAcc, confus, probs, labels = trainer.test(testLoader, classWeight)

        testProbs.append(probs)
        testLabels.append(labels)
        testCon.append(confus)
        testLosses.append(testLoss)
        testAccs.append(testAcc)

        #测试结果输出
        if cfg.logs.testResult:
            metrics = {
                'accuracy': testAcc,
                'loss': testLoss
            }
            logger.log_test_results(log, fold + 1, metrics)
        else:
            print(f'测试集结果 - 损失：{testLoss:.4f}，准确率：{testAcc:.4f}')

        allTestAcc.append(testAcc)
        allModel.append(model)
        allTrain.append(trainer)
        his.append(trainer.history)

    bestFold = np.argmax(allTestAcc)
    bestModel = allModel[bestFold]
    bestTrain = allTrain[bestFold]
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
    modelROC = f'{ROCResult}/{cfg.model.modelName}-{cfg.model.nodeDim}-locate.json'
    # 保存为 JSON 文件
    with open(modelROC, "w") as f:
        json.dump(ROCdata, f, indent=4)

    #文件保存开关
    saveConfig = {
        'mw': cfg.result.modelWeights,
        'fm': cfg.result.fullModel,
        'tc': cfg.result.trainConfig,
        'th': cfg.result.trainHistory,
        'os': cfg.result.optState,
        'mm': cfg.result.modelMetadata
    }
    #训练配置
    config = {
        'model': {
            'modelName':cfg.model.modelName,
            'nodeDim': cfg.model.nodeDim,
            'hiddenDim': cfg.model.hiddenDim,
            'numRelations': cfg.model.numRelations,
            'numClasses': cfg.model.numClasses,
            'numLayers': cfg.model.numLayers,
            'numHeads': cfg.model.numHeads,
            'alp': cfg.model.alp,
            'probably': cfg.model.probably
        },
        'train': {
            'threshold': cfg.train.threshold,
            'emState': emState,
            'datas': cfg.emhance.dataName,
            'kfolds': kf,
            'randomState': state,
            'epochs': cfg.train.epochs,
            'lr': cfg.train.lr,
            'classWeights': cfg.train.classWeights,
            'batchSize': cfg.train.batchSize,
            'deleteEdge': cfg.train.deleteEdge
        }
    }

    config['traLabDict'] = traLabDict
    config['testLabDict'] = testLabDict
    ms = ModelSave(modelPath)


    ms.saveModelHistory(bestModel, bestTrain, config,
                        f'experiment{str(newNum)}', saveConfig)

    ph = PlotHistory()
    if cfg.result.accLoss:
        # 绘制训练历史
        print("绘制训练历史中......")
        ph.average_history_dicts(his)
        ph.plotTrainHistory(f'{modelPath}/experiment{str(newNum)}/training_history.png')

    #绘制ROC曲线
    path = f'{modelPath}/experiment{str(newNum)}/ROC'
    # if cfg.result.valROC:
    #     print("绘制验证集的ROC曲线中......")
    #     ph.dealValROC(valProbs, valLabels, path)
    # if cfg.result.testROC:
    #     print("绘制测试集的ROC曲线中......")
    #     ph.dealTestROC(testProbs, testLabels, path)
    #绘制混淆矩阵
    if cfg.result.valConfus:
        print("绘制验证集的混淆矩阵中......")
        ph.drawConfus(conPath, foldCon)

    if cfg.result.testConfus:
        print("绘制测试集的混淆矩阵中......")
        testConPath = os.path.join(conPath, '交叉验证下的测试集混淆矩阵')
        os.mkdir(testConPath)
        i = 1
        for con in testCon:
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

def splitTrain(graphs, majority_ratio=0.8, minority_ratio=0.8, random_state=42):

    np.random.seed(random_state)

    # 提取所有标签并统计类别
    y = np.array([g.y.item() for g in graphs])
    classes = np.unique(y)
    # 按类别索引图
    class_indices = {c: [i for i, g in enumerate(graphs) if g.y == c] for c in classes}

    # 确定多数类（样本数最多的类）
    majority_class = max(classes, key=lambda c: len(class_indices[c]))
    minority_classes = [c for c in classes if c != majority_class]

    train_indices = []
    test_indices = []

    # 处理多数类
    idx = class_indices[majority_class]
    n_train = int(len(idx) * majority_ratio)
    selected = np.random.choice(idx, size=n_train, replace=False)
    train_indices.extend(selected)
    test_indices.extend(set(idx) - set(selected))

    # 处理每个少数类
    for c in minority_classes:
        idx = class_indices[c]
        n_train = int(len(idx) * minority_ratio)
        selected = np.random.choice(idx, size=n_train, replace=False)
        train_indices.extend(selected)
        test_indices.extend(set(idx) - set(selected))

    # 打乱索引顺序
    train_indices = shuffle(train_indices, random_state=random_state)
    test_indices = shuffle(test_indices, random_state=random_state)

    # 构建训练集和测试集图列表
    train_graphs = [graphs[i] for i in train_indices]
    test_graphs = [graphs[i] for i in test_indices]

    # 打印类别分布（统计标签出现次数）
    train_labels = [g.y for g in train_graphs]
    test_labels = [g.y for g in test_graphs]

    train_dist = {int(c): train_labels.count(c) for c in classes}
    test_dist = {int(c): test_labels.count(c) for c in classes}
    print(f'训练集类别分布：{train_dist}')
    print(f'测试集类别分布：{test_dist}')

    return train_graphs, test_graphs, train_dist, test_dist
