from FeatureExtraction.Forecast.SelectFeat import FeatureSelector
from FeatureExtraction.Forecast.hics import HiCSSubspaceSelector
from FeatureExtraction.Forecast.lof import LOFDetector
import numpy as np

class FSHiCSDetector:
    def __init__(self,
                 n_bins=10,
                 feature_mode='MRTFS',
                 n_features_to_select=100,
                 hi_target_dim=3,
                 hi_n_top_subspaces=10,
                 hi_n_mcm=50,
                 hi_alpha_fsh=0.1,
                 lof_neighbors=20,
                 random_state=42):
        self.feature_selector = FeatureSelector(n_bins=n_bins,
                                                mode=feature_mode,
                                                n_features_to_select=n_features_to_select)
        self.hi_selector = HiCSSubspaceSelector(target_dim=hi_target_dim,
                                                n_top_subspaces=hi_n_top_subspaces,
                                                n_mcm=hi_n_mcm,
                                                alpha_fsh=hi_alpha_fsh,
                                                random_state=random_state)
        self.lof_detectors = []
        self.lof_neighbors = lof_neighbors
        self.selected_features_ = None
        self.subspaces_ = None

    def fit(self, X):
        self.feature_selector.fit(X)
        self.selected_features_ = self.feature_selector.selected_indices_
        X_selected = X[:, self.selected_features_]
        self.hi_selector.fit(X_selected)
        self.subspaces_ = self.hi_selector.subspaces_
        self.lof_detectors = []
        for sub in self.subspaces_:
            X_sub = X_selected[:, sub]
            lof = LOFDetector(n_neighbors=self.lof_neighbors)
            lof.fit(X_sub)
            self.lof_detectors.append(lof)
        return self

    def predict(self, X):
        X_selected = X[:, self.selected_features_]
        scores = []
        for i, sub in enumerate(self.subspaces_):
            X_sub = X_selected[:, sub]
            lof_scores = self.lof_detectors[i].predict(X_sub)
            scores.append(lof_scores)
        final_scores = np.mean(scores, axis=0)
        return final_scores, X_selected
