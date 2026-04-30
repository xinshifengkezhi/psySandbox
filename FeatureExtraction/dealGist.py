import torch
import os
import pandas as pd
import shutil
import csv


class gistEmhance:
    def __init__(self):
        self.new_root = "trainData/vrxcounterfacturalData"  # 新数据集目录
        self.cf_dir = "gistData/vrxcounterfactuals/"  # 反事实图目录
        # 读取原始标签
        self.orig_root = "trainData/originalData"  # 原始数据根目录
        self.label_df = pd.read_csv(os.path.join(self.orig_root, "label.csv"))
        self.orig_label_dict = dict(zip(self.label_df['graphId'].astype(str), self.label_df['level']))

    def dealData(self):
        # 配置路径

        old_nodes = os.path.join(self.orig_root, "nodes")
        old_edges = os.path.join(self.orig_root, "edges")

        newNodesPath = os.path.join(self.new_root, 'nodes')
        newEdgesPath = os.path.join(self.new_root, 'edges')
        # 创建新数据集的目录结构（与原始数据类似）
        os.makedirs(newNodesPath, exist_ok=True)
        os.makedirs(newEdgesPath, exist_ok=True)

        # 从dataRange.txt中获取归一化的范围
        rangesDir = os.path.join(self.orig_root, 'dataRange.csv')
        datarange = []
        dfRanges = pd.read_csv(rangesDir)
        for row in dfRanges.itertuples():
            rang = [row.min, row.max, row.differ]
            datarange.append(rang)

        print('开始读取并写入数据......')
        for nodeName in os.listdir(old_nodes):
            nodeFile = os.path.join(old_nodes, nodeName)
            newNodeFile = os.path.join(newNodesPath, nodeName)
            # 将原来的文件一起复制到新文件夹里,但先进行归一化，因为反事实生成的数据本身就是归一化好的
            dfnode = pd.read_csv(nodeFile)

            # 提取节点特征
            nodeFeatures = []

            es = 0.01

            for row in dfnode.itertuples():
                features = [
                    row.nodeId,
                    row.modelId,
                    row.modelName,
                    row.normalVectors / 360,
                    # 将下列的值根据dataRange.txt里的范围进行归一化(semantic已归一化)
                    (row.currentPosX - datarange[0][0] + es) / (datarange[0][2] + 2 * es),
                    (row.currentPosY - datarange[1][0] + es) / (datarange[1][2] + 2 * es),
                    (row.currentPosZ - datarange[2][0] + es) / (datarange[2][2] + 2 * es),
                    row.semantic,
                    (row.sentiment - datarange[3][0] + es) / (datarange[3][2] + 2 * es),
                    (row.pos - datarange[4][0] + es) / (datarange[4][2] + 2 * es),
                    (row.neu - datarange[5][0] + es) / (datarange[5][2] + 2 * es),
                    (row.neg - datarange[6][0] + es) / (datarange[6][2] + 2 * es)
                ]
                nodeFeatures.append(features)
            with open(newNodeFile, 'w', newline='', encoding='utf-8') as file:
                write = csv.writer(file)
                write.writerow(['nodeId', 'modelId', 'modelName', 'normalVectors', 'currentPosX',
                                'currentPosY', 'currentPosZ', 'semantic', 'sentiment', 'pos', 'neu', 'neg'])
                for node in nodeFeatures:
                    write.writerow(node)

        print('读取边数据......')
        for edgeName in os.listdir(old_edges):
            edgeFile = os.path.join(old_edges, edgeName)
            newEdgeFile = os.path.join(newEdgesPath, edgeName)
            # 将原来的文件一起复制到新文件夹里
            shutil.copy2(edgeFile, newEdgeFile)

        newLabel = os.path.join(self.new_root, 'label.csv')
        #写入标签文件
        labelfile = open(newLabel, 'w', newline='', encoding='utf-8')
        labelwrite = csv.writer(labelfile)
        labelwrite.writerow(['graphId', 'level'])

        # 将原来的数据写入新的标签文件中
        for row in self.label_df.itertuples():
            labelwrite.writerow([row.graphId, row.level])

        #复制编码文件
        codesFile = os.path.join(self.orig_root, "sandCode.csv")
        newCodes = os.path.join(self.new_root, "sandCode.csv")
        shutil.copy2(codesFile, newCodes)


        # 准备新标签列表
        new_labels = []

        # 1. 处理原始图：可以选择全部保留，或只保留需要的
        # 这里假设保留所有原始图（您也可以根据需求调整）
        for gid, orig_label in self.orig_label_dict.items():
            # 记录原始图标签
            new_labels.append({'graphId': gid, 'label': orig_label})
            # 复制原始节点和边文件到新目录（如果保留）
            # shutil.copy(os.path.join(orig_root, 'nodes', f"{gid}.csv"), os.path.join(new_root, 'nodes', f"{gid}.csv"))
            # shutil.copy(os.path.join(orig_root, 'edges', f"{gid}.csv"), os.path.join(new_root, 'edges', f"{gid}.csv"))

        labelfile.close()
        self.savePt()


    def savePt(self):

        # 写入标签文件
        newLabel = os.path.join(self.new_root, 'label.csv')
        labelfile = open(newLabel, 'a', newline='', encoding='utf-8')
        labelwrite = csv.writer(labelfile)

        target_class = [0, 2]  # 要平衡的目标类（少数类）
        gistPath = os.path.join(self.new_root, 'gistGraph')
        os.makedirs(gistPath, exist_ok=True)
        # 2. 处理反事实图
        transformFile = os.path.join(self.new_root, 'transform.csv')
        transform = open(transformFile, 'a', newline='', encoding='utf-8')
        transformWrite = csv.writer(transform)
        transformWrite.writerow(['gistId', 'newId', 'origLabel', 'newLabel'])
        newId = 20000
        cf_files = [f for f in os.listdir(self.cf_dir) if f.startswith('cf_') and f.endswith('.pt')]
        classNum = {}
        count = 0
        classNum['0'] = 0
        classNum['1'] = 0
        classNum['2'] = 0
        for cf_file in cf_files:
            count += 1
            cf_path = os.path.join(self.cf_dir, cf_file)
            cf_data = torch.load(cf_path)
            cf_label = cf_data.y.item()
            edge_index = cf_data.edge_index
            num_edges = edge_index.size(1)

            # 从文件名提取原始图 ID，例如 cf_10000.pt -> 10000
            orig_id = cf_file.replace('cf_', '').replace('.pt', '')



            # 筛选逻辑：原始为多数类，反事实为目标类
            cf_label = int(cf_label)
            if cf_label == 0:
                classNum['0'] += 1
            if cf_label == 1:
                classNum['1'] += 1
            if cf_label == 2:
                classNum['2'] += 1
            if cf_label not in [0, 1, 2]:
                print(f'异常标签{cf_label},类型是{type(cf_label)}')
            if cf_label in target_class:
                # 将该反事实图加入新数据集
                new_id = str(newId)  # 新图 ID 可以带前缀避免冲突

                """这段注释掉的代码是给生成的图添加边信息的，用来第二次生成"""
                # num_edges = edge_index.size(1)
                # if num_edges == 0:
                #     print(f'存在无边图: {cf_file}，尝试从原始图 {orig_id} 添加一条边')
                #     origEdge = f'{self.orig_root}/edges/{orig_id}.csv'
                #     df = pd.read_csv(origEdge)
                #
                #     # 筛选距离 <10 的边
                #     candidate_edges = df[df['distance'] < 10]
                #     if not candidate_edges.empty:
                #         # 取第一条符合条件的边（也可以随机选一条）
                #         selected = candidate_edges.iloc[0]
                #         sour = int(selected['source']) - 1  # 转为0-based
                #         tar = int(selected['target']) - 1
                #         dis = float(selected['distance'])
                #
                #         # 获取节点数，验证索引有效性
                #         num_nodes = cf_data.x.size(0)
                #         if sour >= num_nodes or tar >= num_nodes:
                #             print(f"警告: 索引超出范围，节点数 {num_nodes}，源 {sour} 目标 {tar}，跳过")
                #             # 添加自环作为备选
                #             sour = tar = 0
                #             dis = 1.0
                #
                #         # 添加单向边（原图已是双向，这里只需添加一条）
                #         new_edges = torch.tensor([[sour], [tar]], dtype=torch.long)  # 形状 [2, 1]
                #         cf_data.edge_index = new_edges
                #         # 边属性，形状 [1, 1]
                #         edge_attr = torch.tensor([[dis]], dtype=torch.float)
                #         cf_data.edge_attr = edge_attr
                #         print(f"为 {cf_file} 添加了单向边 ({sour}, {tar})，距离 {dis}")
                #     else:
                #         # 无符合条件的边，添加自环
                #         print(f"警告: 原始图 {orig_id} 中无距离 <10 的边，添加自环")
                #         cf_data.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                #         cf_data.edge_attr = torch.tensor([[1.0]], dtype=torch.float)


                # 保存反事实图为 .pt 文件（或者转换为 CSV，但 .pt 更简单）
                torch.save(cf_data, os.path.join(gistPath, f"{new_id}.pt"))
                labelwrite.writerow([new_id, cf_label])
                transformWrite.writerow([orig_id, new_id, None, cf_label])
                print(f"Added counterfactual {new_id} (original {orig_id}:{None} -> {cf_label})")
                newId += 1
        labelfile.close()
        transform.close()
        print(classNum)
        print(count)

    """尝试用之前平衡后的数据平衡反事实生成的数据"""
    def emhance(self):
        count = 0
        kistLabel = 'trainData/newData1/label.csv'
        df = pd.read_csv(kistLabel)

        # 从dataRange.txt中获取归一化的范围
        rangesDir = os.path.join(self.orig_root, 'dataRange.csv')
        datarange = []
        es = 0.01
        dfRanges = pd.read_csv(rangesDir)
        for row in dfRanges.itertuples():
            rang = [row.min, row.max, row.differ]
            datarange.append(rang)

        newLabel = os.path.join(self.new_root, 'label.csv')
        labelfile = open(newLabel, 'a', newline='', encoding='utf-8')
        labelwrite = csv.writer(labelfile)

        print('复制节点数据中......')
        for row in df.itertuples():
            graphId = row.graphId
            label = row.level

            if graphId >= 20000 and label == 2:
                count += 1
                nodeFile = f'trainData/newData1/Nodes/{graphId}.csv'
                newGraphId = graphId + 3000
                labelwrite.writerow([newGraphId, 2])
                newNodeFile = f'{self.new_root}/nodes/{newGraphId}.csv'
                edgeFile = f'trainData/newData1/edges/{graphId}.csv'
                newEdgeFile = f'{self.new_root}/edges/{newGraphId}.csv'
                shutil.copy2(edgeFile, newEdgeFile)
                dfnode = pd.read_csv(nodeFile)
                nodeFeatures = []
                for row in dfnode.itertuples():
                    features = [
                        row.nodeId,
                        row.modelId,
                        row.modelName,
                        row.normalVectors / 360,
                        # 将下列的值根据dataRange.txt里的范围进行归一化(semantic已归一化)
                        (row.currentPosX - datarange[0][0] + es) / (datarange[0][2] + 2 * es),
                        (row.currentPosY - datarange[1][0] + es) / (datarange[1][2] + 2 * es),
                        (row.currentPosZ - datarange[2][0] + es) / (datarange[2][2] + 2 * es),
                        row.semantic,
                        (row.sentiment - datarange[3][0] + es) / (datarange[3][2] + 2 * es),
                        (row.pos - datarange[4][0] + es) / (datarange[4][2] + 2 * es),
                        (row.neu - datarange[5][0] + es) / (datarange[5][2] + 2 * es),
                        (row.neg - datarange[6][0] + es) / (datarange[6][2] + 2 * es)
                    ]
                    nodeFeatures.append(features)
                with open(newNodeFile, 'w', newline='', encoding='utf-8') as file:
                    write = csv.writer(file)
                    write.writerow(['nodeId', 'modelId', 'modelName', 'normalVectors', 'currentPosX',
                                    'currentPosY', 'currentPosZ', 'semantic', 'sentiment', 'pos', 'neu', 'neg'])
                    for node in nodeFeatures:
                        write.writerow(node)
        print(f'为类别2添加了{count}个属于原始平衡的数据')
        labelfile.close()

if __name__ == '__main__':
    ge = gistEmhance()
    # ge.savePt()
    # ge.dealData()
    ge.emhance()