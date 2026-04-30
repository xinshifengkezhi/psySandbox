import math

"""定义沙具的一些属性"""
class Exclusive:
    def __init__(self, config=None):
        attribute = {
            'wareId': 0,                 #对于同一种沙具，用该变量区分是第几个
            'modelId': '',               #沙具id
            'modelName': '',             #沙具名字
            'normalVectors': 270,        #沙具的方向角度，默认向前
            'currentPosX': 0.0,          #沙具的x坐标
            'currentPosY': 0.0,          #沙具的y坐标
            'currentPosZ': 0.0,          #沙具的z坐标
            'semantic': 0.0,             #沙具自身的语义象征
            'sentiment': 0,              #情感得分
            'pos': 0,                    #正面得分
            'neu': 0,                    #中性得分
            'neg': 0                     #负面得分
        }
        self.neighbors = []             #与之相邻的沙具id
        self.edges = []                 #与之相连的边的id
        self.config = {**attribute,**(config or {})}

        for key, value in self.config.items():
            setattr(self, key, value)

    """返回节点属性"""
    def toDict(self):
        return {
            'modelName': self.modelName,
            'normalVectors': self.normalVectors,
            'coordinates': f"({self.currentPosX},{self.currentPosY},{self.currentPosZ})",
            'semantic': self.semantic
        }

    def addNeighbor(self, neighbor, edge):
        self.neighbors.append(neighbor)
        self.edges.append(edge)

    def __repr__(self):
        return f"沙具名称：{self.modelName}\n" \
            f"沙具坐标：({self.currentPosX},{self.currentPosY},{self.currentPosZ})"

"""定义沙具之间的关系"""
class Relation:
    def __init__(self, config=None):
        attribute = {
            'vertex': (),                #端点信息id
            'distance': 0.0,             #两节点之间的空间距离
        }

        self.config = {**attribute, **(config or {})}

        for key, value in self.config.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"边所连接的两个沙具：{self.vertex}\n" \
            f"沙具的空间距离：{self.distance}"

"""创建沙盘的图，包括图的一些操作"""
class SandBox:
    def __init__(self, name):
        self.name = name
        self.nexnum = 0                 #节点数量
        self.varnum = 0                 #边的数量
        self.nodes = {}                 #保存所有的节点，格式："节点id:节点对象"
        self.edges = {}                 #保存所有的边,格式："边id:边对象"
        """
            这里采用保留被删除的节点和边的id的值，在新添加的节点中重新使用这些id，
            由于每个节点之间都会添加上边，所以保存的被删除的边和节点数量上一定能对上
            即两个集合不会只有一个为空
            好处是，可以避免删除后id被弃用，导致后续的id越来越大
            坏处是操作复杂
        """
        self.delNodeIds = []            #保留被删除的节点id
        self.delEdgeIds = []            #保留被删除的边的id
        """定义一个节点id列表，用于保存的邻接矩阵、度矩阵"""
        self.nodeList = None
        self.adjMatrix = None
        self.degreeMatrix = None

    """创建沙具，复制沙具：SMHandle_CreateModel操作"""
    def createModel(self, **kwargs):
        if self.delNodeIds:             #如果有未被使用的id，重新获取这些id给新的节点
            id = self.delNodeIds.pop()
        else:                           #如果为空，直接继续使用后续的节点id
            id = self.nexnum
        self.nodes[id] = Exclusive(config=kwargs)
        # print(f"执行的操作：创建沙具\n"
        #       f"添加的沙具id：{self.nodes[id].modelId}-{self.nodes[id].wareId}，沙具名称：{self.nodes[id].modelName}")
        self.createRelat(id)
        self.nexnum += 1
        # print(f"节点id{id}的邻居节点id：{self.nodes[id].neighbors}")
        # print(f"节点id{id}相连的边的id：{self.nodes[id].edges}")


    """在添加节点的同时添加上和其他所有节点的边"""
    def createRelat(self, id):
        if len(self.nodes) == 1:
            return
        count = 0
        for key, value in self.nodes.items():

            if key == id:
                continue
            count += 1
            dist = math.sqrt((self.nodes[id].currentPosX-self.nodes[key].currentPosX)**2+
                             (self.nodes[id].currentPosY-self.nodes[key].currentPosY)**2+
                             (self.nodes[id].currentPosZ-self.nodes[key].currentPosZ)**2)
            """配置边的信息，创建边的对象，并在相应的节点中添加邻居的信息"""
            config = {
                'vertex': (id,key),
                'distance': dist
            }
            if self.delEdgeIds:         #这里的逻辑同节点处的操作
                eid = self.delEdgeIds.pop()
            else:
                eid = self.varnum
            self.edges[eid] = Relation(config)
            self.nodes[id].addNeighbor(key,eid)
            self.nodes[key].addNeighbor(id, eid)
            self.varnum += 1
        #     print(f"\t添加的边：({self.nodes[id].modelName},{self.nodes[key].modelName})")
        # print(f"添加了{count}条边")

    """删除沙具：SMHandle_DeleteModel操作"""
    def deleteModel(self, mId, wid):
        id = self.getNodeId(mId, wid)
        """先删除节点所有的边，因为这里的节点中包含各边的id"""
        # print(self.nodes[id].edges)
        for eid in self.nodes[id].edges:
            """获取邻居的节点，删除对应的邻居id和边id"""
            tul = self.edges[eid].vertex
            if tul[0] == id:
                nid = tul[1]
            else:
                nid = tul[0]
            self.nodes[nid].neighbors.remove(id)
            self.nodes[nid].edges.remove(eid)
            """之后删除字典中边的对象"""
            del self.edges[eid]
            self.varnum -= 1
            self.delEdgeIds.append(eid)
        """最后删除节点对象"""
        # print(f"操作名称：删除沙具，删除的沙具名称：{self.nodes[id].modelName}-{self.nodes[id].wareId}")
        del self.nodes[id]
        self.nexnum -= 1
        self.delNodeIds.append(id)

    """
        根据沙具id获取到对应的沙具节点id
        由于传入的是沙具id，这个方法仅对沙具自身操作时调用来获取沙具节点的id值
    """
    def getNodeId(self, mId, wId):
        for key, value in self.nodes.items():
            if (value.modelId == mId) & (value.wareId == wId):#这里不仅匹配沙具种类id，还匹配同一沙具的id
                return key

    """
        旋转沙具：SMHandle_RotateModel操作
        这里将传入沙具的id和y轴的旋转角度，即只记录绕y轴的旋转，方向按照沙盘视角的逆时针方向
    """
    def rotateModel(self, modelId, wid, currentRotaY):
        id = self.getNodeId(modelId, wid)
        self.nodes[id].normalVectors = (self.nodes[id].normalVectors+currentRotaY) % 360
        # print(f"操作名称：沙具旋转，沙具“{self.nodes[id].modelName}-{self.nodes[id].wareId}”"
        #       f"逆时针旋转了{currentRotaY}度")


    """
        移动沙具：SMHandle_MoveModel操作
        传入沙具id，三个代表新位置的参数
    """
    def moveModel(self, modelId, wid, PosX, PosY, PosZ):
        id = self.getNodeId(modelId,wid)
        self.nodes[id].currentPosX = PosX
        self.nodes[id].currentPosY = PosY
        self.nodes[id].currentPosZ = PosZ

        """这里需要更新所有与之相连的边的空间距离"""
        for eid in self.nodes[id].edges:
            tul = self.edges[eid].vertex
            if tul[0] == id:
                nid = tul[1]
            else:
                nid = tul[0]
            self.edges[eid].distance = math.sqrt((self.nodes[id].currentPosX-self.nodes[nid].currentPosX)**2 +
                             (self.nodes[id].currentPosY-self.nodes[nid].currentPosY) ** 2 +
                             (self.nodes[id].currentPosZ-self.nodes[nid].currentPosZ) ** 2)
        # print(f"操作名称：移动沙具，沙具“{self.nodes[id].modelName}-{self.nodes[id].wareId}”"
        #       f"移动到新位置({PosX},{PosY},{PosZ})")

    """转换为邻接矩阵的形式，以用于后续的GNN"""
    def getMatrex(self):
        #创建节点列表,并建立一个从id到列表索引的字典映射
        num = 0
        list = []
        dict = {}
        for key in self.nodes.keys():
            list.append(key)
            dict[key] = num
            num += 1
        self.nodeList = list
        #创建邻接矩阵
        adjacent = [[0]*self.nexnum for _ in range(self.nexnum)]
        for i in range(self.nexnum):
            for j in self.nodes[list[i]].neighbors:
                adjacent[i][dict[j]] = 1
        self.adjMatrix = adjacent
        #创建度矩阵
        degree = [[0]*self.nexnum for _ in range(self.nexnum)]
        for i in range(self.nexnum):
            degree[i][i] = sum(adjacent[i])
        self.degreeMatrix = degree


