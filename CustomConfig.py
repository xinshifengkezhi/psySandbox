from torch_geometric.graphgym import register_config
from yacs.config import CfgNode as CN

"""
    在FeatureExtraction里有两个Python文件test.py和dealGist.py均采用硬编码，不打算改了
    test.py里面是测试一些小功能或者读取文件获取结构的，还是不删了
    dealGist.py里面是读取两个用反事实解释Gist生成的数据并分配类型的
"""
@register_config('customConfig')
def myConfig(cfg):
    """
        workFlows:执行的模块顺序，用数字1-6表示
        task1-6分别代表：
            将原始的日志文本数据转化为结构化的数据
            按照日志的操作序列一步一步还原最终的图，并生成用于训练模型的数据结构
            图的可视化
            数据平衡
            开始训练
            模型解释
        各模块的配置在后面写明
    """
    cfg.workFlows = []
    cfg.workModule = CN()
    cfg.workModule.task1 = 'OperSeq'
    cfg.workModule.task2 = 'created'
    cfg.workModule.task3 = 'visu'
    cfg.workModule.task4 = 'emhancing'
    cfg.workModule.task5 = 'trains'
    cfg.workModule.task6 = 'expla'
    cfg.workModule.task7 = 'forecast'
    cfg.workModule.task8 = 'basetrain'
    """
        dataDir:原始数据路径
        parmulDir:语义转化模型的存放位置
        structDir:结构化后的数据保存路径
        trainDataDir:训练用的数据路径
        es:进行归一化时扩展的范围，用于所有特征的归一化，若不想归一化或者自己已经处理过数据可以设为0
    """
    cfg.dataName = None
    cfg.dataDir = 'data'
    cfg.parmulDir = 'model/paraphrase-multilingual'
    cfg.structDir = 'struct'
    cfg.trainDataDir = 'trainData'
    cfg.es = 0.01
    """
        日志模块:
            outDir:日志保存位置
            backupCount:最大日志文件数量
            normInfo:输出除了模型信息和训练信息外的其他可能用到的信息
            dataInfo:输出数据集的基本信息
            dataSplitInfo:输出数据集划分的信息
            modelInfo:输出模型的一些信息
            testResult:输出测试集结果信息
    """
    cfg.logs = CN()
    cfg.logs.outDir = 'logs'
    cfg.logs.backupCount = 5
    cfg.logs.normInfo = False
    cfg.logs.dataInfo = False
    cfg.logs.dataSplitInfo = False
    cfg.logs.modelInfo = False
    cfg.logs.testResult = False
    """
        图创建模块:
            dataName:保存的数据文件名称
            nameDim:模具名称映射后的维度
            textDim:模具语义映射后的维度
    """
    cfg.create = CN()
    cfg.create.dataName = 'originalData'
    cfg.create.nameDim = 30
    cfg.create.textDim = 30
    """
        可视化模块:
            graphId:需要可视化的图的id（最好可视化原来的图，别用平衡后的，因为没写）
            tool:可选'plotly','matplotlib'（至于为什么写了两个，可能是当时无聊吧）
    """
    cfg.visual = CN()
    cfg.visual.graphId = None
    cfg.visual.tool = 'plotly'
    """
        平衡模块（平衡采用的是生成随机数的方式来随机选择不重复的列表子集）:
            randomSeed:随机数种子，None就是随机的，会保存随机种子
            dataName:新数据保存的文件夹名称，会和旧数据一起放在trainDataDir下，同时也是训练数据的文件名
                五个已有的数据名字：
                originalData:未平衡的数据，这个里面的label.csv文件里有原始数据打分取平均的结果
                newData1:第一次组合式平衡数据，但没保存组合的图id列表
                newData2:第二次组合式平衡数据，存有组合的图id列表
                castCounterfacturalData:使用反事实解释Gist模型生成的平衡数据，但采用了原来设置的采样方式
                wrtxcounterfacturalData:使用反事实解释Gist模型生成的平衡数据，但使用了类别权重的采样方式
                两个反事实解释所生成的都没有成功生成类别为2的数据，因此类别2是直接复制newData1里面类别2的数据
            firstId:扩展后的图的起始id
    """
    cfg.emhance = CN()
    cfg.emhance.maxCombine = 3
    cfg.emhance.randomSeed = None
    cfg.emhance.dataName = 'newData'
    cfg.emhance.testData = 'testData'
    cfg.emhance.firstId = 20000
    """
        模型参数:
            modelName:模型选择：GCNModel,RGCNModel,GraphTransFormer,GATModel
                    还有执行模块为8下的模型名字:CNNModel,MLPModel,Transform
            nodeDim:节点维度=8+nameDim+textDim
            hiddenDim:中间层维度
            edgeDim:边特征维度
            numRelations:基于RGCN的模型，表示边的关系类型数量，默认设为1
            numClasses:输出维度
            numLayers:RGCN层数
            numHeads:多头注意力头数
            alp:ELU激活函数的超参数
            probably:dropout层丢弃概率
    """
    cfg.model.modelName = 'GCNModel'
    cfg.model.nodeDim = 68
    cfg.model.hiddenDim = 128
    cfg.model.edgeDim = 1
    cfg.model.numRelations = 1
    cfg.model.numClasses = 3
    cfg.model.numLayers = 3
    cfg.model.numHeads = 4
    cfg.model.alp = 1.0
    cfg.model.probably = 0.5
    """
        训练模块:
            kfolds:交叉验证折数
            randomState:划分训练集和测试集的随机种子，None是随机划分，会保存随机种子
            lr:训练初始学习率
            classWeights:是否加入类别权重
            batchSize:批次大小
            deleteEdge:删除边，并使图连通。数字代表保留距离低于此数值的边
            threshold:平衡阈值，少数类平衡到多数类的比例
    """
    cfg.train.kfolds = 10
    cfg.train.threshold = 1.0
    cfg.train.randomState = None
    cfg.train.epochs = 100
    cfg.train.lr = 0.01
    cfg.train.classWeights = False
    cfg.train.batchSize = 32
    cfg.train.deleteEdge = None
    """
        结果保存:
            outDir:输出位置
            expNum:保存的实验编号，None即按照原来的编号自动递增
            之后是保存的内容：
                acc_loss:正确率和损失图像   valConfus:验证集的混淆矩阵    testConfus:测试集的混淆矩阵
                modelWeights:模型权重   fullModel:模型完整结构    trainConfig:训练配置
                trainHistory:训练历史   optState:优化器状态   modelMetadata:模型元数据
    """
    cfg.result = CN()
    cfg.result.outDir = 'model/SandBoxModel'
    cfg.result.expNum = None
    cfg.result.accLoss = False
    cfg.result.valROC = False
    cfg.result.testROC = False
    cfg.result.valConfus = False
    cfg.result.testConfus = False
    cfg.result.modelWeights = False
    cfg.result.fullModel = False
    cfg.result.trainConfig = False
    cfg.result.trainHistory = False
    cfg.result.optState = False
    cfg.result.modelMetadata = False
    """
        解释模块:
            expNum:需要解释的实验编号模型
            resultDir:输出结果文件名
            graphId:需要解释的图列表
            这里重点说明gist生成的图，由于是.pt文件，里面的图已经是经过归一化后的，
            甚至无法确保gist是否修改过节点的特征，因此无法还原节点的名称，因此不解释gistGraph里的图
            topEdges:展示最重要的几条边，None表示展示所有
            topFeat:展示前n个重要的特征
            baseWidth:不使用环形图的线条基础宽度
            minWidth:环形图最小基础线条
            maxWidth:环形图最大基础线条
    """
    cfg.explainer = CN()
    cfg.explainer.expNum = 22
    cfg.explainer.isLoop = True
    cfg.explainer.baseWidth = 3
    cfg.explainer.minWidth = 0.5
    cfg.explainer.maxWidth = 5.0
    cfg.explainer.resultDir = 'explain'
    cfg.explainer.graphId = []
    cfg.explainer.topEdges = None
    cfg.explainer.topFeat = 20
    cfg.explainer.allFeat = False
