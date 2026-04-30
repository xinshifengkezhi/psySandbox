import csv
import os
import random
import json
import shutil
import pandas as pd
import numpy as np
from torch_geometric.graphgym.config import cfg
from registry import register

"""进行数据扩充"""
class dataProcess:
    def __init__(self):
        self.originalNode = {}
        self.originalEdge = {}
        #将新增的图id设为全局
        self.graphId = cfg.emhance.firstId

    def getData(self, newNodePath, newEdgePath):
        print('读取原数据中......')
        original = os.path.join(cfg.trainDataDir, cfg.create.dataName)
        nodesPath = os.path.join(original, 'nodes')
        edgesPath = os.path.join(original, 'edges')

        for nodeName in os.listdir(nodesPath):
            nodeFile = os.path.join(nodesPath, nodeName)
            newNodeFile = os.path.join(newNodePath, nodeName)
            #将原来的文件一起复制到新文件夹里
            shutil.copy2(nodeFile, newNodeFile)
            dfnode = pd.read_csv(nodeFile)
            graphId = nodeName[0:-4]
            # 提取节点特征
            nodeFeatures = []

            for row in dfnode.itertuples():
                features = [
                    row.modelId,
                    row.modelName,
                    row.normalVectors,
                    row.currentPosX,
                    row.currentPosY,
                    row.currentPosZ,
                    row.semantic,
                    row.sentiment,
                    row.pos,
                    row.neu,
                    row.neg
                ]
                nodeFeatures.append(features)
            self.originalNode[graphId] = nodeFeatures

        for edgeName in os.listdir(edgesPath):
            edgeFile = os.path.join(edgesPath, edgeName)
            newEdgeFile = os.path.join(newEdgePath, edgeName)
            # 将原来的文件一起复制到新文件夹里
            shutil.copy2(edgeFile, newEdgeFile)
            dfedge = pd.read_csv(edgeFile)
            graphId = edgeName[0:-4]

            # 提取边特征
            edgeFeatures = []

            for row in dfedge.itertuples():
                features = [
                    row.source,
                    row.target,
                    row.distance
                ]
                edgeFeatures.append(features)
            self.originalEdge[graphId] = edgeFeatures
        print('读取完毕')


    """数据扩充"""
    def classify(self):

        #准备用于新的文件保存
        print('创建平衡后的数据存放路径')
        newFile = os.path.join(cfg.trainDataDir, cfg.emhance.dataName)
        if not os.path.exists(newFile):
            os.makedirs(newFile)
        newNode = os.path.join(newFile, 'nodes')
        newEdge = os.path.join(newFile, 'edges')
        if not os.path.exists(newNode):
            os.makedirs(newNode)
        if not os.path.exists(newEdge):
            os.makedirs(newEdge)

        self.getData(newNode, newEdge)

        original = os.path.join(cfg.trainDataDir, cfg.create.dataName)
        #这里先复制两个旧文件
        oldRangeFile = os.path.join(original, 'dataRange.csv')
        oldCodeFile = os.path.join(original, 'sandCode.csv')
        newRangeFile = os.path.join(newFile, 'dataRange.csv')
        newCodeFile = os.path.join(newFile, 'sandCode.csv')
        shutil.copy2(oldRangeFile, newRangeFile)
        shutil.copy2(oldCodeFile, newCodeFile)

        labels = os.path.join(original, 'label.csv')
        df = pd.read_csv(labels)
        label = {}
        for row in df.itertuples():
            if row.level not in label.keys():
                label[row.level] = []
            label[row.level].append(row.graphId)

        print('进行标签分类......')
        maxnum = 0
        i = 0
        listLen = []
        #判断最多的标签
        for value in label.values():
            l = len(value)
            listLen.append(l)
            if l > maxnum:
                maxnum = listLen[i]
            i += 1

        newLabel = os.path.join(newFile, 'label.csv')
        labelfile = open(newLabel, 'w', newline='', encoding='utf-8')
        labelwrite = csv.writer(labelfile)
        labelwrite.writerow(['graphId', 'level'])

        #将原来的数据写入新的标签文件中
        for row in df.itertuples():
            labelwrite.writerow([row.graphId, row.level])

        print('开始数据扩充......')
        #进行数据扩充
        i = 0
        if cfg.emhance.randomSeed is None:
            state = random.randint(1, 1000)
        else:
            state = cfg.emhance.randomSeed

        # 将生成的随机数种子和组合的图id统一写进json里
        emhanceInfoPath = os.path.join(newFile, 'emhanceInfo.json')
        emhanceInfo = {}
        emhanceInfo['random'] = state

        for lab, value in label.items():
            n = maxnum - listLen[i]
            if n != 0:
                newData = self.subset(value, n, state, emhanceInfo)

                #写到csv文件中
                for graphId, gra in newData.items():
                    filename = graphId + '.csv'
                    nodefile = os.path.join(newNode, filename)
                    edgefile = os.path.join(newEdge, filename)
                    labelwrite.writerow([graphId, lab])

                    #写入节点信息
                    j = 0
                    with open(nodefile, 'w', newline='', encoding='utf-8') as file:
                        write = csv.writer(file)
                        write.writerow(['nodeId', 'modelId', 'modelName', 'normalVectors', 'currentPosX',
                                        'currentPosY', 'currentPosZ', 'semantic', 'sentiment', 'pos', 'neu', 'neg'])
                        for node in gra['nodes']:
                            item = [j] + node
                            write.writerow(item)
                            j += 1

                    # 写入边的信息
                    j = 0
                    with open(edgefile, 'w', newline='', encoding='utf-8') as file:
                        write = csv.writer(file)
                        write.writerow(['edgeId', 'source', 'target', 'distance'])
                        for edge in gra['edges']:
                            item = [j] + edge
                            write.writerow(item)
                            j += 1
            i += 1

        labelfile.close()
        with open(emhanceInfoPath, 'w', encoding='utf-8') as f:
            json.dump(emhanceInfo, f, ensure_ascii=False, indent=4)


    """
        获取列表子集，并组合成一个不重复的子集集合
        这里采用二进制的方法来获取
        idList:原列表
        n：需要的子集数量
        返回带图id的新的数据
    """

    # def subset(self, idList, n, newFile, state, emhanceInfo):
    #     # 取得n个随机数
    #     l = len(idList)
    #     print("生成随机数中......")
    #
    #     rng = np.random.default_rng(seed=state)
    #     randomNum = rng.choice(np.arange(100, 100 * n), n, replace=False)
    #     print("生成完成")
    #
    #     newData = {}
    #     # 采用二进制的方法,图的命名从20000开始，避免和原来的重复
    #
    #     for ran in randomNum:
    #         newList = []
    #         for i in range(l):
    #             k = ran % 2
    #             if k == 1:
    #                 newList.append(idList[i])
    #             ran /= 2
    #             ran = int(ran)
    #             if ran == 0:
    #                 break
    #
    #         # 去除列表长度为1的，即不进行任何组合的情况
    #         if len(newList) != 1:
    #             # 返回组合成功的新图
    #             newGraph = self.combinate(newList)
    #             newData[str(self.graphId)] = newGraph
    #             emhanceInfo[self.graphId] = newList
    #         self.graphId += 1
    #
    #     return newData

    def subset(self, idList, n, state, emhanceInfo):

        print("生成随机组合中......")
        rng = np.random.default_rng(seed=state)
        newData = {}

        for _ in range(n):
            # 随机决定组合几个图：至少 2 个，最多 max_k
            k = rng.integers(2, cfg.emhance.maxCombine)
            # 从 idList 中随机选择 k 个不同的图 ID
            selected_ids = rng.choice(idList, size=k, replace=False).tolist()
            newGraph = self.combinate(selected_ids)
            newData[str(self.graphId)] = newGraph
            emhanceInfo[self.graphId] = selected_ids
            self.graphId += 1

        return newData

    """
        将列表中的id所对应的图进行直接组合
        idList:操作的列表
    """
    def combinate(self, newList):
        newGraph = {}
        #n用来表示当前已经组合的节点的数量
        n = 0
        nodes = []
        edges = []
        for graphId in newList:
            graphId = str(graphId)
            nodes = nodes + self.originalNode[graphId]
            for edge in self.originalEdge[graphId]:
                source = int(edge[0])
                source += n
                target = int(edge[1])
                target += n
                newEdge = [str(source), str(target), edge[2]]
                edges.append(newEdge)
            n += len(self.originalNode[graphId])
        newGraph['nodes'] = nodes
        newGraph['edges'] = edges
        return newGraph

@register('emhancing')
def toEmhance():
    dp = dataProcess()
    dp.classify()
