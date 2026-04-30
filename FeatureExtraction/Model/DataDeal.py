from torch_geometric.utils import scatter
import pandas as pd
from torch_geometric.data import Data
import os
import torch
import numpy as np
import random
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import copy

"""数据处理类，将保存在训练文件夹里的图转化为PyG列表用于输入图神经网络"""
class DataProcessor:
    def __init__(self, nodeFeaturesDim, dataDir, relations=1):
        """

        :param nodeFeaturesDim: 节点特征维度，不过暂时没用上
        :param dataDir: 需要加载数据的位置
        :param relations: 是否使用关系类型
        """
        self.merged_file_path = os.path.join(dataDir, 'mergedData.pt')
        self.nodeFeaturesDim = nodeFeaturesDim
        self.relations = relations
        codesDir = os.path.join(dataDir, 'sandCode.csv')
        dfCode = pd.read_csv(codesDir)
        names = {}
        texts = {}
        for row in dfCode.itertuples():
            nd = row.nameCode.strip('[]').split()
            nd = [float(x) for x in nd]
            names[row.modelname] = nd
            td = row.textCode.strip('[]').split()
            td = [float(x) for x in td]
            texts[row.modelname] = td
        self.namesData = names
        self.textsData = texts

        # 从dataRange.txt中获取归一化的范围
        rangesDir = os.path.join(dataDir, 'dataRange.csv')

        ranges = []
        """
            这里加了个判断，如果没有dataRange这个文件，
            说明数据应该通过其他方式归一化过了，这里将不进行归一化
        """
        if os.path.exists(rangesDir):
            self.vectors = 360
            dfRanges = pd.read_csv(rangesDir)
            for row in dfRanges.itertuples():
                rang = [row.min, row.max, row.differ]
                ranges.append(rang)
        else:
            self.vectors = 1
            for i in range(7):
                rang = [0, 0, 1]
                ranges.append(rang)
        self.datarange = ranges

    def _collate_graphs(self, graphs):
        """将图列表合并为 (data, slices)，用于保存文件"""
        # 考虑到有些反事实生成的图没有边，这里需确保所有图都有 edge_attr
        for g in graphs:
            if not hasattr(g, 'edge_attr') or g.edge_attr is None:
                # 为无边图添加默认 edge_attr
                # collate 要求所有图有该属性，即使无边也要有形状 [0, feature_dim] 的空张量
                if g.edge_index.numel() == 0:
                    # 无边图：edge_attr 应为空张量，形状 [0, 1]
                    g.edge_attr = torch.empty((0, 1), dtype=torch.float)
                else:
                    # 有边但缺 edge_attr，添加全1特征
                    num_edges = g.edge_index.size(1)
                    g.edge_attr = torch.ones((num_edges, 1), dtype=torch.float)

        return InMemoryDataset.collate(graphs)

    def _save_merged(self, graphs, path):
        """保存合并文件，保存为.pt文件方便读取（使用深拷贝避免修改原图）"""

        graphs_copy = copy.deepcopy(graphs)
        data, slices = InMemoryDataset.collate(graphs_copy)
        torch.save((data, slices), path)
        print(f"合并文件已保存至: {path}")

    def _load_merged(self, path):
        """从合并文件加载图列表，如果添加了新的数据，建议删除这个文件，否则不会读取"""
        data, slices = torch.load(path)
        graphs = []
        num_graphs = len(slices['x']) - 1  # slices 存储的是起始索引，最后一个元素是总长度
        for i in range(num_graphs):
            # 节点特征
            x = data.x[slices['x'][i]:slices['x'][i + 1]]
            # 标签
            y = data.y[slices['y'][i]]
            # 边索引（需要减去节点偏移量，恢复局部索引）
            edge_index = data.edge_index[:, slices['edge_index'][i]:slices['edge_index'][i + 1]]
            node_offset = slices['x'][i]
            edge_index = edge_index - node_offset

            # 边特征
            edge_attr = None
            if 'edge_attr' in slices:
                edge_attr = data.edge_attr[slices['edge_attr'][i]:slices['edge_attr'][i + 1]]
            # 构建 Data 对象
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            graphs.append(graph)
        print(f"从合并文件加载了 {len(graphs)} 个图")
        return graphs

    def loadData(self, datas, deledge, es=0.01):
        """

        :param datas: 加载数据的路径
        :param deledge: 需要删除的边的阈值
        :param es: 范围扩展值，避免出现0,1这样的特殊值
        :return: 图字典，键是图id，值是Data格式的数据
        """
        # 确定合并文件路径
        # if os.path.exists(self.merged_file_path):
        #     print(f'从合并文件加载数据: {self.merged_file_path}')
        #     return self._load_merged(self.merged_file_path)

        nodesfile = os.path.join(datas, 'nodes')
        edgesfile = os.path.join(datas, 'edges')
        labelfile = os.path.join(datas, 'label.csv')
        graphs = []

        nodesList = os.listdir(nodesfile)
        totalNum = len(nodesList)
        gistGraphPath = os.path.join(datas, 'gistGraph')
        if os.path.exists(gistGraphPath):
            gistList = os.listdir(gistGraphPath)
            totalNum += len(gistList)
        else:
            gistList = []

        #初始化进度条
        pbar = tqdm(total=totalNum, desc='加载数据中......', unit='graph')

        for nodeName in nodesList:
            nodePath = os.path.join(nodesfile, nodeName)
            edgePath = os.path.join(edgesfile, nodeName)
            graphId = nodeName[0:-4]
            try:
                graph = self.getGraph(nodePath, edgePath, labelfile, graphId, es, deledge)
            except:
                continue

            if graph is not None:
                graphs.append(graph)
            pbar.update(1)

        if gistList:
            for graph in gistList:
                graphFile = os.path.join(gistGraphPath, graph)
                cf_data = torch.load(graphFile)

                if not hasattr(cf_data, 'edge_attr') or cf_data.edge_attr is None:
                    # 为无边图添加默认 edge_attr
                    # collate 要求所有图有该属性，即使无边也要有形状 [0, feature_dim] 的空张量
                    if cf_data.edge_index.numel() == 0:
                        # 无边图：edge_attr 应为空张量，形状 [0, 1]
                        cf_data.edge_attr = torch.empty((0, 1), dtype=torch.float)
                    else:
                        # 有边但缺 edge_attr，添加全1特征
                        num_edges = cf_data.edge_index.size(1)
                        cf_data.edge_attr = torch.ones((num_edges, 1), dtype=torch.float)

                # 下面这是个添加边特征为一的语句，同样是用于反事实生成的数据，不过前面已经处理过了
                # cf_data.edge_attr = torch.ones(cf_data.edge_index.size(1), 1)
                graphs.append(cf_data)
                pbar.update(1)

        pbar.close()

        # 如果指定了合并文件路径，则保存
        # self._save_merged(graphs, self.merged_file_path)
        #进行简单的随机复制来达到平衡
        # graphs = self.copyData(graphs)

        return graphs

    def getGraph(self, nodePath, edgePath, labelfile, graphId, es, deledge):
        """
        提取图中的所以信息，返回一个Data
        :param nodePath: 节点文件路径
        :param edgePath: 边文件路径
        :param labelfile: 标签文件路径
        :param graphId: 图id
        :return: Data对象
        """
        # 提取节点特征
        nodeFeatures = []
        nodeNum = 0
        dfnode = pd.read_csv(nodePath)

        sema = dfnode['sentiment'].tolist()

        for row in dfnode.itertuples():

            if pd.isna(row.neg):
                neg = 0
            else:
                neg = row.neg

            if pd.isna(row.neu):
                neu = 0
            else:
                neu = row.neu

            if pd.isna(row.currentPosY):
                y = 0
            else:
                y = row.currentPosY

            features = [
                # 角度的值可以不用考虑
                row.normalVectors / self.vectors,
                # 将下列的值根据dataRange.txt里的范围进行归一化(semantic已归一化)
                (row.currentPosX - self.datarange[0][0] + es) / (self.datarange[0][2] + 2 * es),
                (y - self.datarange[1][0] + es) / (self.datarange[1][2] + 2 * es),
                (row.currentPosZ - self.datarange[2][0] + es) / (self.datarange[2][2] + 2 * es),
                # (row.sentiment - self.datarange[3][0] + es) / (self.datarange[3][2] + 2 * es),
                # (row.pos - self.datarange[4][0] + es) / (self.datarange[4][2] + 2 * es),
                # (neu - self.datarange[5][0] + es) / (self.datarange[5][2] + 2 * es),
                # (neg - self.datarange[6][0] + es) / (self.datarange[6][2] + 2 * es)
            ]

            # 检查 features 中每个元素
            for i, val in enumerate(features):
                if np.isnan(val) or np.isinf(val):
                    # print(f"NaN/Inf 出现在特征索引 {i}, 行 {row.Index}, 值: {val}")
                    # # 可以打印该行所有原始数据以便分析
                    # print(row)
                    return

            # 添加名称向量和语义向量
            name = self.namesData[row.modelName]
            text = self.textsData[row.modelName]

            # name = np.mean(name)
            # text = np.mean(text)
            #
            # features.append(name)
            # features.append(text)

            features = features + name + text

            nodeFeatures.append(features)
            nodeNum += 1

        # 提取边特征

        dfedge = pd.read_csv(edgePath)

        # 删除大于某个距离的边，并确保图的连通性
        edgeDict = self.delEdge(dfedge, deledge)
        #进一步随机删除
        # edgeDict = self.random_drop_edges(edgeDict)
        edgeIndex = []
        edgeattr = []

        edgeType = []
        # 从更新后的边中提取边信息
        count = 0
        for info in edgeDict.values():
            dist = info[0]
            sour = info[1]
            tar = info[2]
            count += 1

            #配置里预设的值，如果要启用关系类型，那就将这里的边特征换成关系
            #根据semantic的值，若两个节点,均大于0，则设为0，均小于0则设为1，
            if self.relations == 3:
                if sema[sour] >= 0:
                    if sema[tar] >= 0:
                        type = 0
                    else:
                        type = 2
                else:
                    if sema[tar] >= 0:
                        type = 2
                    else:
                        type = 1
            else:
                type = 0


            # for row in dfedge.itertuples():
            #     sour = row.source
            #     tar = row.target
            #     dist = row.distance

            # 不考虑孤立节点的事，直接跳过所有大于该值的边
            # if dist >= 40.0:
            #     continue

            edgeIndex.append([sour, tar])
            edgeattr.append([dist])
            edgeType.append(type)

            # 对于无向图，添加反向的边
            edgeIndex.append([tar, sour])
            edgeattr.append([dist])
            edgeType.append(type)


        # 将节点和边特征转化为pytorch geometric的data对象
        x = torch.tensor(nodeFeatures, dtype=torch.float)
        edgeIndex = torch.tensor(edgeIndex, dtype=torch.long).t().contiguous()
        edgeattr = torch.tensor(edgeattr, dtype=torch.float)
        edgeType = torch.tensor(edgeType, dtype=torch.long)

        # 检查节点特征
        node_features_array = np.array(nodeFeatures)

        # === 添加详细的数据检查 ===
        if np.isnan(node_features_array).any() or np.isinf(node_features_array).any():
            print(f"警告: 图 {graphId} 的节点特征包含 NaN 或 Inf 值")
            return

        # # 计算边的权重（距离的倒数）
        # epsilon = 1  # 让该权重尽量小于1，避免两个节点过于接近导致权重极大
        # edgeWeights = 1.0 / (edgeattr.squeeze() + epsilon)  # squeeze()将 [n,1] 转为 [n]

        # 获取标签，标签为分数,在0-2内
        df = pd.read_csv(labelfile)
        label = df.loc[df['graphId'] == int(graphId), 'level'].iloc[0]
        label = int(label)

        if label > 2:
            print(f'警告：图：{graphId}的标签异常，异常值为：{label}')
            label = int(label / 10)
        if label < 0:
            print(f'警告：图：{graphId}的标签异常，异常值为：{label}')
            return
        y = torch.tensor([label], dtype=torch.long)

        graph = Data(x=x, edge_index=edgeIndex, edge_attr=edgeattr, edge_type=edgeType, y=y)
        # graph.edgeWeights = edgeWeights

        return graph

    def random_drop_edges(self, edgeDict, drop_ratio=0.6):
        """
        从 edgeDict 中随机删除比例为 drop_ratio 的边，不保证连通性。
        :param edgeDict: 字典 {edge_id: [dist, sour, tar]}
        :param drop_ratio: 删除比例，0~1，例如 0.6 表示删除 60%
        :return: 新的 edgeDict
        """
        if drop_ratio <= 0:
            return edgeDict

        num_edges = len(edgeDict)
        num_to_keep = int(num_edges * (1 - drop_ratio))
        # 确保至少保留一条边
        if num_to_keep < 1:
            num_to_keep = 1

        # 随机选择要保留的边的键
        edge_keys = list(edgeDict.keys())
        keys_to_keep = random.sample(edge_keys, num_to_keep)

        # 构建新的字典
        new_edgeDict = {k: edgeDict[k] for k in keys_to_keep}
        return new_edgeDict

    """查找节点的根父节点"""
    def findParent(self, parent, node):
        if parent[node] != node:
            parent[node] = self.findParent(parent, parent[node])
        return parent[node]

    """合并两个集合，按秩合并"""
    def union(self, parent, rank, node1, node2):
        root1 = self.findParent(parent, node1)
        root2 = self.findParent(parent, node2)

        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1
            return True
        return False

    def delEdge(self, dfedge, threshold):
        """
        运用最小生成树的Kruskal方法，删除边中大于某个值的所有边，并确保图连接
        :param dfedge:需要处理的边列表，DataFrame格式
        :param threshold:删除边阈值
        :return:
        """
        edgeList = []
        nodes = []
        edgedict = {}
        for row in dfedge.itertuples():
            edgeList.append((row.edgeId, row.distance, row.source, row.target))
            edgedict[row.edgeId] = [row.distance, row.source, row.target]
            if row.source not in nodes:
                nodes.append(row.source)
            if row.target not in nodes:
                nodes.append(row.target)

        # 按照升序排序
        edgeList.sort(key=lambda x: x[1])

        # 初始化并查集
        parent = {}
        rank = {}
        for nodeId in nodes:
            parent[nodeId] = nodeId
            rank[nodeId] = 0

        # 存储MST中的边的id
        mstEdges = set()

        # Krusjal算法构建最小生成树
        for edgeId, distance, sour, tar in edgeList:
            # 只考虑小于等于阈值的边
            if distance <= threshold:
                if self.union(parent, rank, sour, tar):
                    mstEdges.add(edgeId)

        # 检查连通性
        roots = set()
        for nodeId in nodes:
            roots.add(self.findParent(parent, nodeId))

        # 如果图不连通，需要添加最小的边来保证连通性
        if len(roots) > 1:
            print()
            # 添加最小的必要边
            for edgeId, distance, sour, tar in edgeList:
                if distance > threshold:
                    if self.union(parent, rank, sour, tar):
                        mstEdges.add(edgeId)

        # 删除不在mst中的边
        edgesRemove = set(edgedict.keys()) - mstEdges
        for edgeId in edgesRemove:
            del edgedict[edgeId]

        return edgedict

    def copyData(self, graphs):
        """对不平衡的类别进行粗糙简单的复制，使类别相等"""
        print('进行简单复制中......')
        # 按标签分组
        labelGraphs = {}
        for graph in graphs:
            label = graph.y.item()
            if label not in labelGraphs:
                labelGraphs[label] = []
            labelGraphs[label].append(graph)

        # 找到最大类别的样本数
        maxCount = max(len(graphs) for graphs in labelGraphs.values())

        # 平衡每个类别的样本
        balancedGraphs = []
        for label, labelGraphs in labelGraphs.items():
            currentCount = len(labelGraphs)

            if currentCount < maxCount:
                # 需要复制的数量
                numadd = maxCount - currentCount
                # 随机选择要复制的样本（有放回）
                indicesCopy = np.random.choice(currentCount, size=numadd, replace=True)

                # 复制图数据
                for idx in indicesCopy:
                    # 创建新的图对象（深拷贝）
                    originalGraph = labelGraphs[idx]
                    newGraph = type(originalGraph)(
                        x=originalGraph.x.clone(),
                        edge_index=originalGraph.edge_index.clone(),
                        edge_attr=originalGraph.edge_attr.clone() if originalGraph.edge_attr is not None else None,
                        y=originalGraph.y.clone()
                    )
                    balancedGraphs.append(newGraph)

                print(f'为标签为{label}随机复制了{len(indicesCopy)}个样本')
            # 添加原始样本
            balancedGraphs.extend(labelGraphs)

        # 打乱顺序
        random.shuffle(balancedGraphs)

        return balancedGraphs
