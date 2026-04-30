from sklearn.neighbors import NearestNeighbors
import numpy as np

class LOFDetector:
    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors
        self.train_data_ = None
        self.neigh_ = None

    def fit(self, X):
        self.train_data_ = X.copy()
        self.neigh_ = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.neigh_.fit(X)
        # 预计算训练点的局部可达密度
        train_lrd = []
        for i in range(len(self.train_data_)):
            dist, idx = self.neigh_.kneighbors([self.train_data_[i]], n_neighbors=self.n_neighbors)
            lrd = self._local_reachability_density(self.train_data_[i], idx[0], dist[0])
            train_lrd.append(lrd)
        self.train_lrd_ = np.array(train_lrd)
        return self

    def _local_reachability_density(self, x, indices, distances):
        k_dist = [self.neigh_.kneighbors([self.train_data_[i]], n_neighbors=self.n_neighbors, return_distance=True)[0][0][-1] for i in indices]
        reach_dist = np.maximum(k_dist, distances)
        lrd = 1.0 / (np.sum(reach_dist) / len(indices))
        return lrd

    def predict(self, X_new):
        distances, indices = self.neigh_.kneighbors(X_new, n_neighbors=self.n_neighbors)
        lrd_new = []
        for i in range(len(X_new)):
            lrd_i = self._local_reachability_density(X_new[i], indices[i], distances[i])
            lrd_new.append(lrd_i)
        lrd_neighbors_mean = [np.mean(self.train_lrd_[idx]) for idx in indices]
        lof_scores = np.array(lrd_neighbors_mean) / np.array(lrd_new)
        return lof_scores
