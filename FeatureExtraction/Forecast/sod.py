import numpy as np
from sklearn.neighbors import NearestNeighbors

class SODDetector:
    """
    轴平行子空间异常检测算法（不包含特征筛选）
    输入：已经过特征筛选的数据
    """
    def __init__(self, k_neighbors=20, l_reference=10, alpha_fss=0.5):
        self.k = k_neighbors          # 最近邻个数 k
        self.l = l_reference           # 共享最近邻参考集大小 l
        self.alpha = alpha_fss         # 阈值 α_FSS
        self.train_X_ = None
        self.neigh_ = None

    def fit(self, X):
        """
        X: 筛选后的训练数据 (n_samples, n_features)
        训练过程：计算每个点的k近邻，并计算共享最近邻相似度（但不需要存储所有）
        实际上SOD的参考集是在预测时动态计算的，因为参考集依赖新点的最近邻。
        这里我们只需要保存训练数据，并建立索引结构。
        """
        self.train_X_ = X
        self.neigh_ = NearestNeighbors(n_neighbors=self.k)
        self.neigh_.fit(X)
        # 预计算训练数据中每个点的k近邻（用于共享最近邻计算）
        self.train_knn_indices_ = self.neigh_.kneighbors(X, n_neighbors=self.k, return_distance=False)
        return self

    def _shared_nearest_neighbors(self, point_idx, candidate_indices):
        """
        计算点point_idx与候选点列表candidate_indices中每个点的共享最近邻数量
        """
        point_knn = set(self.train_knn_indices_[point_idx])
        shared_counts = []
        for cand in candidate_indices:
            cand_knn = set(self.train_knn_indices_[cand])
            shared = len(point_knn.intersection(cand_knn))
            shared_counts.append(shared)
        return shared_counts

    def predict(self, X_new):
        """
        X_new: (n_samples, n_features) 新样本
        返回：异常分数, 中间特征向量（就是输入X_new本身）
        """
        n_samples = len(X_new)
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            x = X_new[i].reshape(1, -1)

            # 1. 找到x在训练集中的k近邻
            distances, knn_indices = self.neigh_.kneighbors(x, n_neighbors=self.k, return_distance=True)
            knn_indices = knn_indices[0]

            x_knn = set(knn_indices)
            shared_counts = []
            for idx in knn_indices:
                neighbor_knn = set(self.train_knn_indices_[idx])
                shared = len(x_knn.intersection(neighbor_knn))
                shared_counts.append(shared)
            # 选择共享最多的l个索引
            top_indices = np.argsort(shared_counts)[::-1][:self.l]
            reference_indices = knn_indices[top_indices]

            # 3. 计算参考集的均值和方差
            ref_points = self.train_X_[reference_indices]
            mu = np.mean(ref_points, axis=0)
            var = np.var(ref_points, axis=0)
            total_var = np.sum(var)

            # 4. 计算子空间定义向量 v
            lambda_dim = X_new.shape[1]  # 特征维度 λ
            threshold = self.alpha * total_var / lambda_dim
            v = (var < threshold).astype(int)

            # 5. 计算异常度
            if np.sum(v) == 0:
                # 如果所有维度都不相关，则距离为0？论文中分母为sum(v)，避免除0
                score = 0
            else:
                diff = x - mu
                weighted_diff = diff * v
                distance = np.sqrt(np.sum(weighted_diff ** 2))
                score = distance / np.sum(v)

            scores[i] = score

        # 中间特征向量：就是X_new本身（因为SOD没有改变特征）
        intermediate_features = X_new
        return scores, intermediate_features
