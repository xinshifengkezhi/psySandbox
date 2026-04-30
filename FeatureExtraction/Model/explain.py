import torch
import logger
import os
import pandas as pd
import numpy as np
from matplotlib.patches import PathPatch
from torch_geometric.explain import ModelConfig, GNNExplainer, Explainer
from torch_geometric.utils import to_networkx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
import networkx as nx
from FeatureExtraction.Model.GCNModel import SandBoxModel
from FeatureExtraction.Model.DataDeal import DataProcessor
from FeatureExtraction.Model.GTModel import GraphTransFormer
from FeatureExtraction.Model.RGCNModel import RelateModel
from torch_geometric.graphgym.config import cfg
from registry import register
import warnings
from tqdm import tqdm
matplotlib.use('TkAgg')
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric\\.explain\\.explainer")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelExplain:
    """
        传入需要解释的模型实验名称
        experiment:传入的实验序号
    """
    def __init__(self, experiment):
        self.modelPath = f'{cfg.result.outDir}/experiment{str(experiment)}'
        save = f'{self.modelPath}/{cfg.explainer.resultDir}'
        if not os.path.exists(save):
            os.makedirs(save)
        self.savePath = save

        #特征列表
        self.expectedFeatureNames = (
                ['normal'] +
                ['posX', 'posY', 'posZ', 'sentiment', 'pos', 'neu', 'ne'] +
                [f'name_{i}' for i in range(10)] +
                [f'text_{i}' for i in range(10)]
        )

        #保存每张图的张量
        self.allFeatImport = []
        #初始化解释器，这样不用每次计算掩码都初始化一次了
        self.getexplainer()

    def getexplainer(self):
        modelFile = f'{self.modelPath}/modelWeights.pth'
        # 加载模型
        if cfg.model.modelName == 'GCNModel':
            model = SandBoxModel(
                cfg.model.nodeDim,
                cfg.model.hiddenDim,
                cfg.model.numClasses,
                cfg.model.numLayers,
                cfg.model.numRelations,
                cfg.model.alp,
                cfg.model.probably
            )
        if cfg.model.modelName == 'TransFormer':
            model = GraphTransFormer(
                cfg.model.nodeDim,
                cfg.model.hiddenDim,
                cfg.model.numClasses,
                cfg.model.numLayers,
                cfg.model.probably,
                cfg.model.numHeads,
                cfg.model.edgeDim
            )
        if cfg.model.modelName == 'RGCNModel':
            model = RelateModel(
                cfg.model.nodeDim,
                cfg.model.hiddenDim,
                cfg.model.numRelations,
                cfg.model.numClasses,
                cfg.model.numLayers,
                cfg.model.alp,
                cfg.model.probably,
            )
        checkpoint = torch.load(modelFile)
        model.load_state_dict(checkpoint)
        model.eval()

        modelConfig = ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw'
        )

        # 创建解释器算法
        algorithm = GNNExplainer(
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            return_type='raw'
        )
        self.explainer = Explainer(
            model=model,
            algorithm=algorithm,
            explanation_type='model',
            model_config=modelConfig,
            node_mask_type='attributes',
            edge_mask_type='object'
        )

    def explain(self, graphName, topEdges=None, draw='single'):
        """
        rgcn模型的解释
        :param graphId: 图id
        :param topEdges: 要展示的前top个重要边
        :param draw: 绘制单个图还是绘制全局的特征重要性,single是单个，all是全局
        :return: 无
        """

        datas = os.path.join(cfg.trainDataDir, cfg.emhance.dataName)
        nodesfile = os.path.join(datas, 'nodes')
        edgesfile = os.path.join(datas, 'edges')
        labelfile = os.path.join(datas, 'label.csv')


        nodePath = os.path.join(nodesfile, graphName)
        edgePath = os.path.join(edgesfile, graphName)
        processor = DataProcessor(28, datas)

        graphId = graphName[0:-4]


        if os.path.exists(nodePath):
            if draw == 'single':
                print(f"加载图{graphId}的数据")
            graph = processor.getGraph(nodePath, edgePath, labelfile, graphId,
                                       cfg.es, cfg.train.deleteEdge)
        else:
            warnings.warn(f'你可能解释了一个不在节点文件夹内的节点，id{graphId}，我就帮你跳过这个了。\n'
                          f'如果是gistGraph里的，放弃吧，详细情况去看CustomConfig.py里解释模块配置的说明')
            return

        label = graph.y.item()
        explanation = self.explainer(
            x=graph.x,
            edge_index=graph.edge_index,
            batch=None,
            target=label
        )

        #提取掩码
        nodeFeatMask = explanation.node_mask
        edgeMask = explanation.edge_mask

        #累计当前图的平均特征重要性
        featImpMean = nodeFeatMask.mean(dim=0).detach().cpu()
        #节点重要性
        nodeImp = nodeFeatMask.mean(dim=1).detach().cpu()
        self.allFeatImport.append(featImpMean)

        if draw == 'single':
            if cfg.explainer.isLoop:
                self.loopVisual(graph, nodeImp, edgeMask, graphId,
                                nodesfile, label, topEdges=topEdges)
            else:
                self.drawVisual(graph, nodeImp, edgeMask, graphId, label, nodesfile)

            self.plot_feature_importance(featImpMean, graphId, cfg.explainer.topFeat)

    def plotAllFeat(self, topFeat=20):
        """
        绘制所有图的平均重要性
        :param topFeat: 取前n个特征
        :return: 无
        """
        if not self.allFeatImport:
            print('没有累积任何特征重要性，无法绘制整体图')
            return
        print('绘制图重要性特征中......')
        allImp = torch.stack(self.allFeatImport, dim=0)
        overallImp = allImp.mean(dim=0).numpy()

        #按重要性排序
        sortedIdx = np.argsort(overallImp)[::-1]
        if topFeat is not None and topFeat < len(overallImp):
            indices = sortedIdx[:topFeat]
            values = overallImp[indices]
            titleSuffix = f'(Top-{topFeat})'
        else:
            indices = sortedIdx
            values = overallImp[indices]
            titleSuffix = ''

        #获取特征名称
        featureNames = [self.expectedFeatureNames[i] for i in indices]

        #绘图
        plt.figure(figsize=(10, max(6, len(indices) * 0.3)))
        yPos = np.arange(len(indices))
        plt.barh(yPos, values, color='steelblue')
        plt.yticks(yPos, featureNames)
        plt.xlabel('平均重要性（跨所有图）')
        plt.ylabel('特征维度')
        plt.title(f'图  全局特征重要性{titleSuffix}')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        savePath = f'{self.savePath}/overall-feature-importance.png'
        plt.savefig(savePath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'特征重要性图已保存至：{savePath}')

    def plot_feature_importance(self, featImpMean, graphId, topFeat=20):
        """
        绘制全局特征重要性柱状图
        node_feat_mask: [num_nodes, num_features] 的张量
        graph_id: 图ID，用于保存图片
        top_k: 显示重要性最高的前 top_k 个特征
        """
        # 计算每个特征维度的平均重要性
        featImp = featImpMean.numpy()
        numFeatures = len(featImp)

        # 按重要性降序排列，取前 topFeat
        sortedIdx = np.argsort(featImp)[::-1]
        if topFeat is not None and topFeat < numFeatures:
            indices = sortedIdx[:topFeat]
            values = featImp[indices]
            titleSuffix = f' (Top-{topFeat})'
        else:
            indices = sortedIdx
            values = featImp[indices]
            titleSuffix = ''

        featureNames = [self.expectedFeatureNames[i] for i in indices]

        # 绘图
        plt.figure(figsize=(10, max(6, len(indices) * 0.3)))
        yPos = np.arange(len(indices))
        plt.barh(yPos, values, color='steelblue')
        plt.yticks(yPos, featureNames)
        plt.xlabel('平均重要性')
        plt.ylabel('特征维度')
        plt.title(f'全局特征重要性(所有图取平均){titleSuffix}')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        savePath = f'{self.savePath}/graph{graphId}_feature_importance.png'
        plt.savefig(savePath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'特征重要性图已保存至：{savePath}')

    """解释可视化，节点随机分布"""
    def drawVisual(self, graph, nodeImp, edgeMask, graphId, nodePath, label):
        g = to_networkx(graph, to_undirected=True, remove_self_loops=True)

        #建立节点和名称映射，原始图graph里不包含节点名称，因此只能从文件里读取
        nodefile = f'{nodePath}/{graphId}.csv'
        dfnode = pd.read_csv(nodefile)
        customLabels = dfnode.set_index('nodeId')['modelName'].to_dict()

        """重建边与掩码的映射"""
        edgeWeights = {}
        for i, (u, v) in enumerate(graph.edge_index.t().tolist()):
            edgeWeights[(u, v)] = edgeMask[i].item()
            edgeWeights[(v, u)] = edgeMask[i].item()

        #节点重要性：取每个节点上所有特征的最大值作为节点的重要性
        nodeMean = nodeImp.numpy()

        #绘图
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(g, seed=42)

        #节点
        nodes = nx.draw_networkx_nodes(
            g, pos,
            node_color=nodeMean,
            cmap='Reds',
            node_size=600
        )
        nx.draw_networkx_labels(g, pos, labels=customLabels, font_size=10)

        #边根据重要性显示宽度
        edgeColors = [edgeWeights.get((u, v), 0.0) for u, v in g.edges()]
        nx.draw_networkx_edges(g, pos,
                               width=[w*cfg.explainer.baseWidth for w in edgeColors],
                               edge_color='gray')

        plt.colorbar(nodes, label='Node Importance')
        plt.title(f'图{graphId}的边重要性解释(标签：{label})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{self.savePath}/graph{graphId}nor-explanation.png', dpi=150)
        plt.show()

    def loopVisual(self, graph, nodeImp, edgeMask, graphId, nodePath, label, topEdges=None):
        """
        解释可视化，节点环形分布
        :param graph: 图数据
        :param nodeFeatMask: 节点掩码
        :param edgeMask: 边掩码
        :param graphId: 图Id
        :param nodePath: 节点路径
        :param topEdges: 是否显示重要性前k个边，None表示全部显示
        :return: 无
        """
        #获取节点数和边数
        numNodes = graph.x.shape[0]
        numEdges = graph.edge_index.shape[1]

        #读取节点名称
        nodefile = f'{nodePath}/{graphId}.csv'
        dfnode = pd.read_csv(nodefile)
        nodeName = dfnode['modelName'].tolist()

        #计算节点在圆上的坐标
        R = 1.0
        angles = np.linspace(0, 2 * np.pi, numNodes, endpoint=False)
        #这里我们随机打乱节点分布
        np.random.shuffle(angles)
        xCoords = R * np.cos(angles)
        yCoords = R * np.sin(angles)

        #处理边掩码，排序并筛选前k重要的边
        edgeImp = edgeMask.detach().cpu().numpy().flatten()
        if topEdges is not None and topEdges < numEdges:
            sortedIdx = np.argsort(edgeImp)[::-1]
            keepIdx = sortedIdx[:topEdges]
            edgeIndex = graph.edge_index.cpu().numpy()[:, keepIdx]
            edgeImp = edgeImp[keepIdx]
        else:
            #默认展示所有的边
            edgeIndex = graph.edge_index.cpu().numpy()

        #归一化边重要性,将其映射到线宽上
        if edgeImp.max() > edgeImp.min():
            edgeNorm = (edgeImp - edgeImp.min()) / (edgeImp.max() - edgeImp.min())
            minWidth, maxWidth =cfg.explainer.minWidth, cfg.explainer.maxWidth
            lineWidths = minWidth + edgeNorm * (maxWidth - minWidth)
        else:
            lineWidths = np.ones_like(edgeImp) * 2.0

        #绘制边,边采用弧线的方式
        plt.figure(figsize=(15, 15))
        ax = plt.gca()

        """
            使用贝塞尔曲线绘制弧线
            计算弦的中点，曲线的控制点位于环心与中点连线上
            环中点（0,0）这样径向方向单位向量就是弦中点归一化
        """
        for i in range(len(edgeImp)):
            src, tgt = edgeIndex[0, i], edgeIndex[1, i]
            x1, y1 = xCoords[src], yCoords[src]
            x2, y2 = xCoords[tgt], yCoords[tgt]
            midX = (x1 + x2) / 2
            midY = (y1 + y2) / 2
            distMid = np.sqrt(midX ** 2 + midY ** 2)

            if distMid > 1e-6:
                #曲线的控制点沿径向向外偏移
                chordLen = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                offset = -0.2 * chordLen
                ctrlX = midX * (1 + offset / distMid)
                ctrlY = midY * (1 + offset / distMid)
            else:
                #弦过环心，可以直接画成直线
                pathData = [(Path.MOVETO, (x1, y1)),
                            (Path.LINETO, (x2, y2))]
                codes, verts = zip(*pathData)
                path = Path(verts, codes)
                patch = PathPatch(path, facecolor='none', edgecolor='gray',
                                  linewidth=lineWidths[i], alpha=0.7)
                ax.add_patch(patch)
                continue

            #创建贝塞尔曲线
            pathData = [(Path.MOVETO, (x1, y1)),
                        (Path.CURVE3, (ctrlX, ctrlY)),
                        (Path.CURVE3, (x2, y2))]
            codes, verts = zip(*pathData)
            path = Path(verts, codes)
            patch = PathPatch(path, facecolor='none', edgecolor='gray',
                                  linewidth=lineWidths[i], alpha=0.7)
            ax.add_patch(patch)

        #颜色节点映射
        nodeMean = nodeImp.numpy()
        if nodeMean.max() > nodeMean.min():
            nodeNorm = (nodeMean - nodeMean.min()) / (nodeMean.max() - nodeMean.min())
        else:
            nodeNorm = np.zeros_like(nodeMean)

        #绘制节点
        if numNodes > 50:
            textsize = 8
            nodesize = 50
        else:
            textsize = 16
            nodesize = 200
        scatter = ax.scatter(xCoords, yCoords, c=nodeNorm, cmap='Reds',
                             s=nodesize, edgecolors='black', linewidth=1.5, zorder=5)
        plt.colorbar(scatter, ax=ax, label='节点重要性',
                    pad=0.04)

        #标注节点名称，朝着外围书写
        offset = 0.1
        for i, (x, y, name) in enumerate(zip(xCoords, yCoords, nodeName)):
            radVec = np.array([x, y]) / R
            textX = x + offset * radVec[0]
            textY = y + offset * radVec[1]
            angleDeg = np.degrees(angles[i])

            #确定文字方向,设置旋转角度

            if angleDeg > 180:
                angleDeg -= 360
            rotation = angleDeg

            ax.text(textX, textY, name, fontsize=textsize,
                    rotation=rotation, rotation_mode='anchor', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        #图形修饰
        ax.set_aspect('equal')
        ax.axis('off')
        title = f'图{graphId}的边重要性解释解释(标签：{label})'
        if topEdges:
            title += f'(前{topEdges}条边)'
        ax.set_title(title, pad=40, fontsize=30)

        #保存图片
        plt.savefig(f'{self.savePath}/graph{graphId}loop-explanation.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

@register('expla')
def toExplain():
    logger.setup_logging(cfg.logs.outDir, cfg.logs.backupCount)
    log = logger.get_logger(__name__)
    me = ModelExplain(cfg.explainer.expNum)
    datas = os.path.join(cfg.trainDataDir, cfg.emhance.dataName)
    nodesPath = os.path.join(datas, 'nodes')
    if cfg.explainer.allFeat:
        fileList = os.listdir(nodesPath)
        totalNum = len(fileList)
        pbar = tqdm(total=totalNum, desc='解释所有图中...', unit='graph')
        for graphName in fileList:
            me.explain(graphName, cfg.explainer.topEdges, draw='all')
            pbar.update(1)
        pbar.close()
        me.plotAllFeat(cfg.explainer.topFeat)
    else:
        for id in cfg.explainer.graphId:
            graphName = f'{id}.csv'
            me.explain(graphName, cfg.explainer.topEdges)
