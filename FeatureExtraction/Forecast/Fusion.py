from FeatureExtraction.Forecast.lof import LOFDetector
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import numpy as np

class FusionDBSCANLOF:
    """
    融合改进DBSCAN和LOF的异常检测算法
    """
    def __init__(self, eps=None, min_samples=5, lof_neighbors=20):
        self.eps = eps
        self.min_samples = min_samples
        self.lof_neighbors = lof_neighbors
        self.dbscan = None
        self.lof_detector = None
        self.max_cluster_id = None
        self.max_cluster_center = None
        self.scaler = MinMaxScaler()

    def fit(self, X):
        """
        X: 训练数据 (n_samples, n_features)
        1. 运行DBSCAN
        2. 确定最大簇
        3. 计算每个训练点到最大簇中心的距离（异常分数）
        4. 训练LOF检测器（保存数据以便预测时计算LOF）
        """
        # DBSCAN聚类
        if self.eps is None:
            # 自动估计eps：使用k-distance图，这里简化，取所有样本第min_samples近邻距离的中位数
            neigh = NearestNeighbors(n_neighbors=self.min_samples)
            neigh.fit(X)
            distances, _ = neigh.kneighbors(X)
            k_dist = distances[:, -1]
            self.eps = np.percentile(k_dist, 50)  # 中位数作为eps
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(X)
        self.dbscan = dbscan

        # 找出最大的簇（簇标签不为-1且样本数最多）
        unique_labels = set(labels)
        unique_labels.discard(-1)  # 去掉噪声点
        if len(unique_labels) == 0:
            # 如果没有簇，则所有点都是噪声，最大簇中心无法定义，报错或使用全部数据均值
            self.max_cluster_center = np.mean(X, axis=0)
            self.max_cluster_id = None
        else:
            # 计算每个簇的大小
            cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
            self.max_cluster_id = max(cluster_sizes, key=cluster_sizes.get)
            max_cluster_mask = labels == self.max_cluster_id
            self.max_cluster_center = np.mean(X[max_cluster_mask], axis=0)

        # 计算训练集DBSCAN异常分数：到最大簇中心的欧氏距离
        dbscan_scores = np.linalg.norm(X - self.max_cluster_center, axis=1)

        # 训练LOF检测器
        self.lof_detector = LOFDetector(n_neighbors=self.lof_neighbors)
        self.lof_detector.fit(X)

        # 记录DBSCAN分数的范围，用于后续规范化
        self.dbscan_min = dbscan_scores.min()
        self.dbscan_max = dbscan_scores.max()

        # 保存训练数据（可选，用于预测时规范化）
        self.train_X_ = X

        return self

    def predict(self, X):
        """
        对新样本X进行预测：
        1. 计算DBSCAN异常分数（到最大簇中心的距离）
        2. 计算LOF异常分数
        3. 将DBSCAN分数规范化到LOF分数的大致范围（这里使用min-max映射到[0, max_lof]）
        4. 取两种分数的平均值作为最终异常分数
        返回：异常分数, 中间特征向量（这里直接返回原始特征，或者可返回规范化后的特征？为保持一致，返回原始特征）
        """
        # DBSCAN分数
        dbscan_scores = np.linalg.norm(X - self.max_cluster_center, axis=1)

        # LOF分数
        lof_scores = self.lof_detector.predict(X)

        # 规范化DBSCAN分数到LOF分数的大致范围（使用训练集的min-max映射）
        # 先对训练集DBSCAN分数做min-max到[0,1]，再乘以训练集LOF分数的最大值
        # 但这里简化，直接使用min-max映射到[0, max_lof]
        # 注意：需要先获得训练集LOF分数的最大值
        # 在训练时保存训练集的LOF分数最大值
        if not hasattr(self, 'train_lof_max'):
            # 计算训练集LOF分数
            train_lof = self.lof_detector.predict(self.train_X_)
            self.train_lof_max = train_lof.max()

        # 规范化DBSCAN分数到[0, train_lof_max]
        if self.dbscan_max > self.dbscan_min:
            dbscan_norm = (dbscan_scores - self.dbscan_min) / (self.dbscan_max - self.dbscan_min)
        else:
            dbscan_norm = np.zeros_like(dbscan_scores)
        dbscan_scaled = dbscan_norm * self.train_lof_max

        # 最终分数取平均（或根据论文取交集？但预测时无法取交集，故用平均）
        final_scores = (dbscan_scaled + lof_scores) / 2

        # 中间特征向量：原始特征（也可选择DBSCAN聚类后的表示，但原始特征最直接）
        intermediate_features = X

        return final_scores, intermediate_features
