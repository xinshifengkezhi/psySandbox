import os
import pandas as pd
import numpy as np
import csv
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import torch

from FeatureExtraction.Model.ResultSave import PlotHistory
from FeatureExtraction.Precoding import LocalText

plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

"""获取各个特征的范围"""
def getRange():
    filePath = "../nodes"
    poxmin = []
    poxmax = []
    poymin = []
    poymax = []
    pozmin = []
    pozmax = []

    for filename in os.listdir(filePath):
        sourcePath = os.path.join(filePath, filename)
        df = pd.read_csv(sourcePath)
        nx = df['currentPosX'].tolist()
        x = [tx for tx in nx if not np.isnan(tx)]
        ny = df['currentPosY'].tolist()
        y = [ty for ty in ny if not np.isnan(ty)]
        nz = df['currentPosZ'].tolist()
        z = [tz for tz in nz if not np.isnan(tz)]
        poxmin.append(np.min(x))
        poxmax.append(np.max(x))
        poymin.append(np.min(y))
        poymax.append(np.max(y))
        pozmin.append(np.min(z))
        pozmax.append(np.max(z))

    sent = pd.read_csv('../data/items_emotion_score(nltk)-邵明宇.csv')
    sentiment = sent['sentiment'].tolist()
    pos = sent['pos'].tolist()
    neu = sent['neu'].tolist()
    neg = sent['neg'].tolist()

    print(f"x坐标的最小值：{np.min(poxmin)}，最大值：{np.max(poxmax)}，差值：{np.max(poxmax) - np.min(poxmin)}\n"
          f"y坐标的最小值：{np.min(poymin)}，最大值：{np.max(poymax)}，差值：{np.max(poymax) - np.min(poymin)}\n"
          f"z坐标的最小值：{np.min(pozmin)}，最大值：{np.max(pozmax)}，差值：{np.max(pozmax) - np.min(pozmin)}\n"
          f"sentiment的最小值：{np.min(sentiment)}，最大值：{np.max(sentiment)}，差值：{np.max(sentiment) - np.min(sentiment)}\n"
          f"pos的最小值：{np.min(pos)}，最大值：{np.max(pos)}，差值：{np.max(pos) - np.min(pos)}\n"
          f"neu的最小值：{np.min(neu)}，最大值：{np.max(neu)}，差值：{np.max(neu) - np.min(neu)}\n"
          f"neg的最小值：{np.min(neg)}，最大值：{np.max(neg)}，差值：{np.max(neg) - np.min(neg)}\n")

"""按照最简单的方法重新还原图结构"""
class node:
    def __init__(self, nodeId):
        self.nodeId = nodeId

class edge:
    def __init__(self, sour, tar, distance):
        self.sour = sour
        self.tar = tar
        self.distance = distance

class graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def createEdge(self,edgeId, sour, tar, distance):
        if sour not in self.nodes.keys():
            self.nodes[sour] = node(sour)
        if tar not in self.nodes.keys():
            self.nodes[tar] = node(tar)
        self.edges[edgeId] = edge(sour, tar, distance)

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

    """运用最小生成树的原理，删除大于阈值的边"""
    def delEdge(self, threshold):
        edgeList = []
        for edgeId, edgeObj in self.edges.items():
            edgeList.append((edgeId, edgeObj.distance, edgeObj.sour, edgeObj.tar))

        #按照升序排序
        edgeList.sort(key=lambda x: x[1])

        #初始化并查集
        parent = {}
        rank = {}
        for nodeId in self.nodes.keys():
            parent[nodeId] = nodeId
            rank[nodeId] = 0

        mstEdges = set() #存储MST中的边的id

        #Krusjal算法构建最小生成树
        for edgeId, distance, sour, tar in edgeList:
            if distance <= threshold:#只考虑小于等于阈值的边
                if self.union(parent, rank, sour, tar):
                    mstEdges.add(edgeId)

        #检查连通性
        roots = set()
        for nodeId in self.nodes.keys():
            roots.add(self.findParent(parent, nodeId))

        #如果图不连通，需要添加最小的边来保证连通性
        if len(roots) > 1:
            #添加最小的必要边
            for edgeId, distance, sour, tar in edgeList:
                if distance > threshold:
                    if self.union(parent, rank, sour, tar):
                        mstEdges.add(edgeId)

        #删除不在mst中的边
        edgesRemove = set(self.edges.keys()) - mstEdges
        for edgeId in edgesRemove:
            del self.edges[edgeId]

        return mstEdges

def confusionsTest():
    confusions = np.random.randint(0, high=100, size=(3, 3))
    pt = PlotHistory()
    title = '测试矩阵'
    num = 0
    for i in range(3):
        num += confusions[i, i]
    va = num / np.sum(confusions)
    resultPath = '../test.png'
    className = ['0', '1', '2']
    text = '第一行\n第二行'
    pt.plotValConfus(confusions, title, resultPath, text, className)

"""统计标签里面的类别数"""
def count():
    with open('../trainData/originalData/label.csv', 'r') as f:
        dfnode = pd.read_csv(f)
        label = dfnode['score'].tolist()
    ser = pd.Series(label)
    count = ser.value_counts().to_dict()
    print(count)


def plotLabel():
    """
    读取标签文件，统计各标签的数量，并绘制柱状图
    """
    filePath = '../trainData/sandBoxData/label.csv'
    resultPath = '../trainData/sandBoxData/score.png'
    # 读取数据
    df = pd.read_csv(filePath)
    labelCounts = df['score'].value_counts().sort_index()

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labelCounts.index.astype(str), labelCounts.values, color='skyblue')

    # 在柱子上方显示数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom')

    plt.xlabel('标签类别')
    plt.ylabel('数量')
    plt.title('标签类别分布柱状图')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(resultPath, dpi=300, bbox_inches='tight')

    # 可选：打印统计信息
    print("各类别数量：", labelCounts.to_dict())

"""对所有的沙具名称和沙具语义统一进行编码并归一化"""
def encodingText(filePath):
    df = pd.read_csv("../data/items_emotion_score(nltk)-邵明宇.csv")
    names = df['item_name'].tolist()
    texts = df['w_list_ch'].tolist()
    lt = LocalText()
    #将模型名字和语义都转化为30维的向量
    nameCodes = lt.convertTexts(names, 30)
    textCodes = lt.convertTexts(texts, 30)
    nameCodes = normalize(nameCodes)
    textCodes = normalize(textCodes)

    with open(filePath, 'w', newline='', encoding='utf-8') as file:
        write = csv.writer(file)
        write.writerow(['modelname', 'nameCode', 'textCode'])
        for i in range(len(names)):
            write.writerow([names[i], nameCodes[i], textCodes[i]])

"""读取测试"""
def read():
    dfCode = pd.read_csv('../sandCode.csv')
    namesData = {}
    textsData = {}
    for row in dfCode.itertuples():
        nd = row.nameCode.strip('[]').split()
        print(nd)
        nd = [float(x) for x in nd]
        namesData[row.modelname] = nd
        td = row.textCode.strip('[]').split()
        td = [float(x) for x in td]
        textsData[row.modelname] = td


"""将转好的文本向量进行归一化"""
def normalize(codes):
    #确定范围
    maxList = codes[0].copy()
    minList = codes[0].copy()
    dim = len(maxList)
    for row in codes:
        for i in range(dim):
            if row[i] > maxList[i]:
                maxList[i] = row[i]
            if row[i] < minList[i]:
                minList[i] = row[i]

    #进行归一化，es是为了扩大范围
    es = 0.01
    for row in codes:
        for i in range(dim):
            row[i] = (row[i] - minList[i] + es) / (maxList[i] - minList[i] + 2 * es)

    return codes


"""将归一化范围的文件转化为csv文件"""
def toCsv():
    ranges = []
    with open('../dataRange.txt', 'r', encoding='utf-8') as file:
        for line in file:
            rang = []
            stdline = line.strip()
            if not stdline:
                continue
            strline = stdline.split('，')
            rang.append(float(strline[0].split('：')[1]))
            rang.append(float(strline[1].split('：')[1]))
            rang.append(float(strline[2].split('：')[1]))
            ranges.append(rang)

    with open('../dataRange.csv', 'w', newline='', encoding='utf-8') as file:
        write = csv.writer(file)
        write.writerow(['min', 'max', 'differ'])
        for item in ranges:
            write.writerow(item)

"""确定.pt的文件结构"""
def getPt():
    cf_dir = '../counterfactuals'
    cf_file = 'cf_3795.pt'
    cf_path = os.path.join(cf_dir, cf_file)
    cf_data = torch.load(cf_path)
    print(cf_data)

def readcsv():
    csv_path = '../trainData/wrtxcounterfacturalData/transform.csv'
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)  # 自动将第一行作为列名
        for row in reader:
            print(type(row['newId']))
            print(type(row['gistId']))

if __name__ == '__main__':
    # gra = graph()
    # with open('../edges/4319.csv', 'r') as f:
    #     df = pd.read_csv(f)
    #     for row in df.itertuples():
    #         gra.createEdge(row.edgeId, row.source, row.target, row.distance)
    #
    # mstEdges = gra.delEdge(10.0)
    #
    # print("完成")

    # confusionsTest()
    # read()
    # count()
    # encodingText('../sandCode.csv')
    # toCsv()
    # getPt()
    # readcsv()
    # plotLabel()
    ph = PlotHistory()
    ph.drawROCClass()
