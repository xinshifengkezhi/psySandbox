from FeatureExtraction.Forecast.SelectFeat import FeatureSelector
from FeatureExtraction.Forecast.sod import SODDetector

class FSSODDetector:
    def __init__(self,
                 n_bins=10,
                 feature_mode='MRTFS',
                 n_features_to_select=100,
                 sod_k_neighbors=20,
                 sod_l_reference=10,
                 sod_alpha_fss=0.5,
                ):
        self.feature_selector = FeatureSelector(n_bins=n_bins,
                                                mode=feature_mode,
                                                n_features_to_select=n_features_to_select)
        self.sod = SODDetector(k_neighbors=sod_k_neighbors,
                               l_reference=sod_l_reference,
                               alpha_fss=sod_alpha_fss)
        self.selected_features_ = None

    def fit(self, X):
        self.feature_selector.fit(X)
        self.selected_features_ = self.feature_selector.selected_indices_
        X_selected = X[:, self.selected_features_]
        self.sod.fit(X_selected)
        return self

    def predict(self, X):
        X_selected = X[:, self.selected_features_]
        scores, intermediate_features = self.sod.predict(X_selected)
        # 中间特征向量是筛选后的特征
        return scores, X_selected


