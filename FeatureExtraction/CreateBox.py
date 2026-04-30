import os
import csv
import pandas as pd
import numpy as np
from FeatureExtraction.SandBoxGraph import SandBox
from FeatureExtraction.Precoding import LocalText
from torch_geometric.graphgym.config import cfg
from registry import register

class CreateBox:
    def __init__(self):
        oldData = cfg.dataDir
        self.data = oldData
        item = os.path.join(oldData, 'all_items.csv')
        emotion = os.path.join(oldData, 'items_emotion_score(nltk)-邵明宇.csv')
        self.dfItem = pd.read_csv(item)
        self.dfEmotion = pd.read_csv(emotion)
        oldLabel = os.path.join(self.data, '7131个沙盘特征汇总+标注-两位标注全部.xlsx')
        self.df = pd.read_excel(oldLabel)

    """创建图对象，将数据导入图中"""
    def impGraph(self, start=3795):
        #重定向print的第一种方法
        # dual = printToLog('create.log')
        #第二种方法：设置标准的logging
        # logger = setupLogging('handleError.log')

        # sys.excepthook = handleExcept

        # 遍历structDir下的所有文件
        filePath = cfg.structDir

        # err = []

        """
            遍历文件的开始
        """
        original = os.path.join(cfg.trainDataDir, cfg.create.dataName)
        if not os.path.exists(original):
            os.mkdir(original)
        labels = os.path.join(original, 'label.csv')
        labelfile = open(labels, 'w', newline='', encoding='utf-8')
        labelwrite = csv.writer(labelfile)
        labelwrite.writerow(['graphId', 'score'])
        for filename in os.listdir(filePath):
            graphId = filename[0:-4]

            result = self.df.loc[self.df['sp_id'] == int(graphId)]

            try:
                label = result['zhuan1'].iloc[0] + result['zhuan2'].iloc[0]
            except:
                print('存在字符串格式的评分')
                continue

            if pd.isna(label):
                print('该标签存在问题或者该标签不存在')
                continue

            if int(graphId) < start:
                continue
            gra = self.createGraph(filePath, filename)
            if gra is None:
                continue
            """将所有的沙盘图保存为“沙盘id”：“沙盘图对象”的字典格式"""
            # file2.write(f'{filename[0:-4]}:{len(gra.nodes)}\n')
            # file3.write(f'{filename[0:-4]}:{count}\n')
            self.tocsv(graphId, gra)
            # 将图的标签写入新的csv文件中
            label = int(label / 2)
            labelwrite.writerow([graphId, label])

            # file1.close()
            # file2.close()


            """创建邻接矩阵，度矩阵和节点特征矩阵"""
            # print()
            # gra.getMatrex()
            # print(gra.nodeList)
            # print(gra.adjMatrix)
            # print(gra.degreeMatrix)
            # print()
            # print(err)

        labelfile.close()

    def createGraph(self, filePath, filename):
        sourcePath = os.path.join(filePath, filename)
        # count = 0

        print(f"===============id为{filename[0:-4]}沙盘的图建立===============\n")
        gra = SandBox(filename[0:-4])
        df = pd.read_csv(sourcePath)
        for row in df.itertuples():
            operation = row[4]
            # can = operation[0:6]
            # if(can == 'Cancel'):
            #     file1.write(f'{operation}\n')
            try:
                mid = row[5][:5]
                wid = row[5][12:]
            except:
                pass
            if (operation == "SMHandle_CreateModel") | (operation == "SMHandle_CopyModel"):  # 创建沙具
                mode = int(mid)
                result1 = self.dfItem.loc[self.dfItem['modelId'] == mode, ['modelName', 'normalVectors']].iloc[0]
                result2 = self.dfEmotion.loc[self.dfEmotion['item_id'] == mode,
                                        ['w_list_ch', 'sentiment', 'pos', 'neu', 'neg']].iloc[0]

                # print(result2['w_list_ch'])
                # print(type(result2['w_list_ch']))
                if result1['normalVectors'] == '前':
                    direct = 270
                else:
                    direct = 0
                config = {
                    'wareId': wid,
                    'modelId': mid,
                    'modelName': result1['modelName'],
                    'normalVectors': direct,
                    'currentPosX': float(row[6]),
                    'currentPosY': float(row[7]),
                    'currentPosZ': float(row[8]),
                    'semantic': result2['w_list_ch'],
                    'sentiment': result2['sentiment'],
                    'pos': result2['pos'],
                    'neu': result2['neu'],
                    'neg': result2['neg']
                }
                try:
                    gra.createModel(**config)
                except:
                    print(f"异常操作,沙盘id：{filename[0:-4]}；操作类型：创建沙具；"
                          f"操作id：{row[1]},可能id“{mid}-{wid}”缺失数据")
            elif operation == "SMHandle_DeleteModel":  # 删除沙具
                # count += 1
                try:
                    gra.deleteModel(mid, wid)
                except:
                    # err.append(filename[0:-4])
                    print(f"异常操作,沙盘id：{filename[0:-4]}；操作类型：删除沙具；"
                          f"操作id：{row[1]},可能id“{mid}-{wid}”没有被创建")

            elif operation == "SMHandle_DepthModel":  # 调整沙具深度
                try:
                    gra.moveModel(mid, wid, float(row[6]), float(row[7]), float(row[8]))
                except:
                    print(f"异常操作,沙盘id：{filename[0:-4]}；操作类型：调整深度；"
                          f"操作id：{row[1]},可能移动位置有问题")

            elif operation == "SMHandle_MoveModel":  # 移动沙具
                try:
                    gra.moveModel(mid, wid, float(row[6]), float(row[7]), float(row[8]))
                except:
                    print(f"异常操作,沙盘id：{filename[0:-4]}；操作类型：移动沙具；"
                          f"操作id：{row[1]},可能移动位置有问题")

            elif operation == "SMHandle_RotateModel":  # 旋转沙具
                try:
                    if pd.isna(row[10]):
                        print(f"异常操作，沙盘id：{filename[0:-4]}；操作类型：旋转沙具；"
                              f"操作id{row[1]}的实际记录并不是旋转沙具,或者选择了旋转操作但未进行任何角度的旋转")
                    else:
                        gra.rotateModel(mid, wid, row[10])
                except:
                    print(f"异常操作,沙盘id：{filename[0:-4]}；操作类型：旋转沙具；"
                          f"操作id：{row[1]},问题暂时不知道")

        return gra

    """获取各个特征的范围"""
    def getRange(self, original, filePath):
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

        sentiment = self.dfEmotion['sentiment'].tolist()
        pos = self.dfEmotion['pos'].tolist()
        neu = self.dfEmotion['neu'].tolist()
        neg = self.dfEmotion['neg'].tolist()

        rangeDir = os.path.join(original, 'dataRange.csv')
        with open(rangeDir, 'w', newline='', encoding='utf-8') as file:
            write = csv.writer(file)
            write.writerow(['min', 'max', 'differ'])

            write.writerow([np.min(poxmin), np.max(poxmax), np.max(poxmax) - np.min(poxmin)])
            write.writerow([np.min(poymin), np.max(poymax), np.max(poymax) - np.min(poymin)])
            write.writerow([np.min(pozmin), np.max(pozmax), np.max(pozmax) - np.min(pozmin)])
            write.writerow([np.min(sentiment), np.max(sentiment), np.max(sentiment) - np.min(sentiment)])
            write.writerow([np.min(pos), np.max(pos), np.max(pos) - np.min(pos)])
            write.writerow([np.min(neu), np.max(neu), np.max(neu) - np.min(neu)])
            write.writerow([np.min(neg), np.max(neg), np.max(neg) - np.min(neg)])

        print(f"x坐标的最小值：{np.min(poxmin)}，最大值：{np.max(poxmax)}，差值：{np.max(poxmax) - np.min(poxmin)}\n"
              f"y坐标的最小值：{np.min(poymin)}，最大值：{np.max(poymax)}，差值：{np.max(poymax) - np.min(poymin)}\n"
              f"z坐标的最小值：{np.min(pozmin)}，最大值：{np.max(pozmax)}，差值：{np.max(pozmax) - np.min(pozmin)}\n"
              f"sentiment的最小值：{np.min(sentiment)}，最大值：{np.max(sentiment)}，差值：{np.max(sentiment) - np.min(sentiment)}\n"
              f"pos的最小值：{np.min(pos)}，最大值：{np.max(pos)}，差值：{np.max(pos) - np.min(pos)}\n"
              f"neu的最小值：{np.min(neu)}，最大值：{np.max(neu)}，差值：{np.max(neu) - np.min(neu)}\n"
              f"neg的最小值：{np.min(neg)}，最大值：{np.max(neg)}，差值：{np.max(neg) - np.min(neg)}\n")

    def textMap(self, filePath):
        lt = LocalText(cfg.parmulDir)
        """取出所有的文本向量，转化为数字"""
        names = self.dfEmotion['item_name'].tolist()
        texts = self.dfEmotion['w_list_ch'].tolist()
        # 将模型名字和语义都转化为n维的向量
        nameCodes = lt.convertTexts(names, cfg.create.nameDim)
        textCodes = lt.convertTexts(texts, cfg.create.textDim)
        nameCodes = lt.normalize(nameCodes, cfg.es)
        textCodes = lt.normalize(textCodes, cfg.es)

        with open(filePath, 'w', newline='', encoding='utf-8') as file:
            write = csv.writer(file)
            write.writerow(['modelname', 'nameCode', 'textCode'])
            for i in range(len(names)):
                write.writerow([names[i], nameCodes[i], textCodes[i]])

    """将每一个最终形成的图对象转换为节点文件和边文件,顺带将标签单独写入一个csv文件中"""
    def tocsv(self, graphId, gra):
        original = os.path.join(cfg.trainDataDir, cfg.create.dataName)

        nodes = os.path.join(original, 'nodes')
        edges = os.path.join(original, 'edges')


        if not os.path.exists(nodes):
            os.mkdir(nodes)
        if not os.path.exists(edges):
            os.mkdir(edges)



        filename = graphId + '.csv'
        nodefile = os.path.join(nodes, filename)
        edgefile = os.path.join(edges, filename)



        #考虑到原来建表时节点和边就不连续，这里设置这样的映射使之连续
        nid = 0
        eid = 0
        data = {}
        #写入节点信息
        with open(nodefile, 'w', newline='', encoding='utf-8') as file:
            write = csv.writer(file)
            write.writerow(['nodeId', 'modelId', 'modelName', 'normalVectors', 'currentPosX',
                            'currentPosY', 'currentPosZ', 'semantic', 'sentiment', 'pos', 'neu', 'neg'])
            for nodeId, node in gra.nodes.items():
                data[nodeId] = nid
                write.writerow([nid, node.modelId, node.modelName, node.normalVectors,
                                node.currentPosX, node.currentPosY, node.currentPosZ,
                                node.semantic, node.sentiment, node.pos, node.neu, node.neg])
                nid += 1

        #写入边的信息
        with open(edgefile, 'w', newline='', encoding='utf-8') as file:
            write = csv.writer(file)
            write.writerow(['edgeId', 'source', 'target', 'distance'])
            for edgeId, edge in gra.edges.items():
                write.writerow([eid, data[edge.vertex[0]], data[edge.vertex[1]], edge.distance])
                eid += 1

        print(f'图{graphId}成功写入')

    def others(self):
        original = os.path.join(cfg.trainDataDir, cfg.create.dataName)
        codesPath = os.path.join(original, 'sandCode.csv')
        self.textMap(codesPath)

        # 再写一个节点特征范围的文件
        nodes = os.path.join(original, 'nodes')
        self.getRange(original, nodes)
    """
        为标签文件追加一个列，将抑郁症的情况分为重度：0，轻度：1，正常：2
        level列将0-2归类为重度，3-5归为轻度，6-10为正常
    """
    def addL(self):
        original = os.path.join(cfg.trainDataDir, cfg.create.dataName)
        labels = os.path.join(original, 'label.csv')
        rows = []
        with open(labels, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            field = reader.fieldnames + ['level']

            for row in reader:
                sco = int(row['score'])
                if (sco >= 0) & (sco <= 2):
                    row['level'] = 0
                if (sco >= 3) & (sco <= 5):
                    row['level'] = 0
                if (sco >= 6) & (sco <= 10):
                    row['level'] = 1

                rows.append(row)

        with open(labels, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=field)
            writer.writeheader()
            writer.writerows(rows)

@register('created')
def create():
    cbox = CreateBox()
    cbox.impGraph(3795)
    cbox.addL()
    cbox.others()

