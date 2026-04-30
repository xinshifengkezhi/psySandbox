import torch
import numpy as np
from collections import defaultdict
import random

class CombinationOversampler:
    """基于样本组合的过采样方法（适用于表格数据）"""

    def __init__(self, threshold=0.5, seed=42, combine_func='mean'):
        """
        threshold: 需要弥补到最大类样本数的比例，取值范围[0,1]
        seed: 随机种子
        combine_func: 组合方式，'mean' 表示取均值，也可自定义
        """
        self.threshold = threshold
        self.seed = seed
        self.combine_func = combine_func
        torch.manual_seed(seed)
        random.seed(seed)

    def fit_resample(self, X, y):
        """
        X: 特征矩阵，numpy array 或 torch Tensor，形状可以是 (n_samples, n_features) 或 (n_samples, 1, n_features)
        y: 标签，numpy array 或 torch Tensor，形状 (n_samples,)
        返回 (X_resampled, y_resampled) 均为 numpy array，形状固定为 (n_samples, 1, n_features)
        """
        # 转换为 numpy 方便处理
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        if isinstance(y, torch.Tensor):
            y = y.numpy()

        # 记录原始特征数（假设最后一维是特征）
        n_features = X.shape[-1]

        # 如果输入是二维，添加 channel 维度
        if X.ndim == 2:
            X = X[:, np.newaxis, :]  # (n, 1, f)

        # 按类别分组
        y_to_indices = defaultdict(list)
        for idx, label in enumerate(y):
            y_to_indices[label].append(idx)

        # 各类样本数
        counts = {label: len(indices) for label, indices in y_to_indices.items()}
        max_count = max(counts.values())

        new_X_list = []   # 存储一维特征向量
        new_y_list = []

        for label, indices in y_to_indices.items():
            cnt = len(indices)
            # 保留原始样本（展平为一维）
            for idx in indices:
                # X[idx] 形状可能是 (1, n_features) 或 (n_features,)，统一 flatten
                new_X_list.append(X[idx].flatten())
                new_y_list.append(label)

            if cnt == max_count:
                continue

            # 需要生成的样本数
            need = int((max_count - cnt) * self.threshold)
            if need <= 0:
                continue

            print(f"平衡类别 {label}，原始样本数 {cnt}，需要生成 {need} 个新样本")

            generated = 0
            k = 2
            max_k = cnt
            while generated < need and k <= max_k:
                to_generate = need - generated
                for _ in range(to_generate):
                    if k > len(indices):
                        break
                    chosen = random.sample(indices, k)
                    # 组合时也使用一维特征
                    combined_feature = self._combine_features([X[i].flatten() for i in chosen])
                    new_X_list.append(combined_feature)
                    new_y_list.append(label)
                    generated += 1
                    if generated >= need:
                        break
                k += 1

            if generated < need:
                print(f"警告：类别 {label} 仅生成了 {generated} 个样本，不足 {need}")

        # 将所有一维特征向量堆叠，再 reshape 成 (n, 1, n_features)
        X_resampled = np.array(new_X_list).reshape(-1, 1, n_features)
        y_resampled = np.array(new_y_list)
        return X_resampled, y_resampled

    def _combine_features(self, feature_list):
        """组合多个样本的特征，返回新特征向量（一维）"""
        if self.combine_func == 'mean':
            return np.mean(feature_list, axis=0)
        else:
            raise ValueError(f"不支持的 combine_func: {self.combine_func}")