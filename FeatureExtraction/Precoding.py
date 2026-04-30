import os
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

"""使用预训练模型对沙具语义进行预编码"""
class LocalText:
    def __init__(self, modelDir=None):
        self.modelDir = modelDir
        self.model = None

    def loadModel(self):
        # 检查模型是否存在
        if os.path.exists(self.modelDir):
            self.model = SentenceTransformer(self.modelDir)
            print("加载模型")
        else:
            print("未找到模型")

    def convertTexts(self, texts, dim):
        """

        :param texts: 需要处理的文本，列表格式
        :param dim: 生成的向量维度
        :return: 词向量列表（维度固定）
        """
        if self.model is None:
            self.loadModel()
        print(f"正在处理{len(texts)}个文本")

        embeddings = self.model.encode(texts)
        print(f"嵌入向量维度：{embeddings.shape}")

        pca = PCA(n_components=dim)
        reduced = pca.fit_transform(embeddings)

        return reduced

    """将转好的文本向量进行归一化"""
    def normalize(self, codes, es):
        # 确定范围
        maxList = codes[0].copy()
        minList = codes[0].copy()
        dim = len(maxList)
        for row in codes:
            for i in range(dim):
                if row[i] > maxList[i]:
                    maxList[i] = row[i]
                if row[i] < minList[i]:
                    minList[i] = row[i]

        # 进行归一化，es是为了扩大范围
        for row in codes:
            for i in range(dim):
                row[i] = (row[i] - minList[i] + es) / (maxList[i] - minList[i] + 2 * es)

        return codes

