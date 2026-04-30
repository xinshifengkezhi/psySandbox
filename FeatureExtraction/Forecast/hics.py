import numpy as np
from scipy.stats import ks_2samp

class HiCSSubspaceSelector:
    def __init__(self, target_dim=3, n_top_subspaces=10, n_mcm=50, alpha_fsh=0.1, random_state=42):
        self.target_dim = target_dim
        self.n_top_subspaces = n_top_subspaces
        self.n_mcm = n_mcm
        self.alpha_fsh = alpha_fsh
        self.random_state = random_state
        self.subspaces_ = []

    def _compute_contrast(self, X_sub, subspace):
        n_samples, dim = X_sub.shape
        if dim < 2:
            return 0
        contrasts = []
        for _ in range(self.n_mcm):
            target_dim_idx = np.random.choice(dim)
            other_dims = [i for i in range(dim) if i != target_dim_idx]
            target_vals = X_sub[:, target_dim_idx]
            n_exclude = int(n_samples * (dim / np.sqrt(self.alpha_fsh)))
            n_exclude = min(n_exclude, n_samples // 2)
            condition_indices = np.random.choice(n_samples, size=n_samples - n_exclude, replace=False)
            condition_vals = target_vals[condition_indices]
            ks_stat, _ = ks_2samp(target_vals, condition_vals)
            contrast = ks_stat
            contrasts.append(contrast)
        return np.mean(contrasts)

    def fit(self, X):
        np.random.seed(self.random_state)
        n_features = X.shape[1]
        # 2维候选
        n_random = min(1000, n_features * (n_features - 1) // 2)
        subspaces_2d = []
        for _ in range(n_random):
            dims = sorted(np.random.choice(n_features, size=2, replace=False))
            subspaces_2d.append(dims)
        subspaces_2d = list(set(tuple(s) for s in subspaces_2d))
        contrast_scores = []
        for sub in subspaces_2d:
            X_sub = X[:, list(sub)]
            contrast = self._compute_contrast(X_sub, sub)
            contrast_scores.append(contrast)
        top_indices = np.argsort(contrast_scores)[::-1][:self.n_top_subspaces]
        current_subspaces = [list(subspaces_2d[i]) for i in top_indices]

        for dim in range(3, self.target_dim + 1):
            candidate_subspaces = []
            for sub in current_subspaces:
                possible_new = list(set(range(n_features)) - set(sub))
                n_candidates = min(100, len(possible_new))
                new_features = np.random.choice(possible_new, size=n_candidates, replace=False)
                for nf in new_features:
                    new_sub = sorted(sub + [nf])
                    candidate_subspaces.append(new_sub)
            candidate_subspaces = list(set(tuple(s) for s in candidate_subspaces))
            scores = []
            for sub in candidate_subspaces:
                X_sub = X[:, list(sub)]
                contrast = self._compute_contrast(X_sub, sub)
                scores.append(contrast)
            top_indices = np.argsort(scores)[::-1][:self.n_top_subspaces]
            current_subspaces = [list(candidate_subspaces[i]) for i in top_indices]
        self.subspaces_ = current_subspaces
        return self

