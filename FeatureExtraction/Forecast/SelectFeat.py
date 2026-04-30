from sklearn.preprocessing import MinMaxScaler
import numpy as np
from collections import Counter

class FeatureSelector:
    def __init__(self, n_bins=10, mode='MRTFS', n_features_to_select=100):
        self.n_bins = n_bins
        self.mode = mode
        self.n_features_to_select = n_features_to_select
        self.selected_indices_ = None
        self.scaler = MinMaxScaler()

    def _grid_partition(self, X):
        X_norm = self.scaler.fit_transform(X)
        bins = np.linspace(0, 1, self.n_bins + 1)
        X_binned = np.digitize(X_norm, bins, right=False) - 1
        X_binned[X_binned == self.n_bins] = self.n_bins - 1
        return X_binned

    def _mutual_info(self, x, y):
        c = np.c_[x, y]
        joint_counts = Counter(map(tuple, c))
        N = len(x)
        joint_prob = np.array(list(joint_counts.values())) / N
        x_counts = Counter(x)
        y_counts = Counter(y)
        x_prob = np.array([x_counts[i] / N for i in sorted(x_counts.keys())])
        y_prob = np.array([y_counts[i] / N for i in sorted(y_counts.keys())])
        mi = 0
        for (i, j), p in zip(joint_counts.keys(), joint_prob):
            mi += p * np.log(p / (x_prob[i] * y_prob[j] + 1e-10))
        return mi

    def fit(self, X):
        n_samples, n_features = X.shape
        X_binned = self._grid_partition(X)
        row_tuples = [tuple(row) for row in X_binned]
        row_counter = Counter(row_tuples)
        D_G = np.array([list(k) for k in row_counter.keys()])
        d_c = np.array(list(row_counter.values()))

        mi_with_target = []
        for j in range(n_features):
            feat_j = D_G[:, j].astype(int)
            mi = self._mutual_info(feat_j, d_c)
            mi_with_target.append(mi)

        if self.mode == 'TFS':
            indices = np.argsort(mi_with_target)[::-1][:self.n_features_to_select]
            self.selected_indices_ = indices
            return
        elif self.mode == 'MRTFS':
            selected = []
            remaining = list(range(n_features))
            mi_matrix = np.zeros((n_features, n_features))
            for i in range(n_features):
                for j in range(i+1, n_features):
                    mi = self._mutual_info(D_G[:, i].astype(int), D_G[:, j].astype(int))
                    mi_matrix[i, j] = mi_matrix[j, i] = mi
            first_idx = np.argmax(mi_with_target)
            selected.append(first_idx)
            remaining.remove(first_idx)
            while len(selected) < self.n_features_to_select:
                best_score = -np.inf
                best_idx = None
                for idx in remaining:
                    redundancy = np.mean([mi_matrix[idx, s] for s in selected])
                    score = mi_with_target[idx] - redundancy
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                selected.append(best_idx)
                remaining.remove(best_idx)
            self.selected_indices_ = np.array(selected)
            return
