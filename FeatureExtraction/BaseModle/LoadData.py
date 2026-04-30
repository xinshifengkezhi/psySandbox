from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import numpy as np

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LoadData():
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    def loadOriginalData(self, intermediatePath, labelPath, startCol=1, nan_threshold=0.5):
        """
        加载原始Excel数据并清洗
        :param intermediatePath: Excel文件路径
        :param labelPath: 标签CSV文件路径
        :param startCol: 特征起始列索引（从0开始），默认1表示跳过第一列ID
        :param nan_threshold: 列允许的最大NaN比例，超过则删除该列
        :param normalize: 是否将特征归一化到[0,1]区间
        :return: XOrig (np.ndarray, float32), yOrig (np.ndarray, int64)
        """
        # 1. 读取Excel（禁用自动NaN转换，保留原始内容）
        df = pd.read_excel(intermediatePath, na_filter=False)

        # 2. 处理空字符串：将空字符串或纯空格替换为NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)

        # 3. 分离ID列和特征列
        idCol = df.columns[0]
        featureCols = list(df.columns[startCol:])

        print(f"原始数据形状: {df.shape}, 特征列数: {len(featureCols)}")

        # 4. 尝试将特征列转为数值类型（非数值转为NaN）
        for col in featureCols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 5. 构建特征矩阵X
        X = df[featureCols].values.astype(np.float32)

        # 6. 删除包含inf的列
        inf_cols = np.where(np.isinf(X).any(axis=0))[0]
        if len(inf_cols) > 0:
            print(f"删除 {len(inf_cols)} 个包含inf的列: {inf_cols}")
            X = np.delete(X, inf_cols, axis=1)
            featureCols = [col for i, col in enumerate(featureCols) if i not in inf_cols]

        # 7. 处理NaN列（删除NaN比例过高的列）
        nan_ratio_per_col = np.isnan(X).mean(axis=0)
        cols_to_drop = np.where(nan_ratio_per_col > nan_threshold)[0]
        if len(cols_to_drop) > 0:
            print(f"删除 {len(cols_to_drop)} 个NaN比例超过{nan_threshold:.0%}的列")
            X = np.delete(X, cols_to_drop, axis=1)
            featureCols = [col for i, col in enumerate(featureCols) if i not in cols_to_drop]

        # 8. 删除仍包含NaN的行
        rows_with_nan = np.isnan(X).any(axis=1)
        if rows_with_nan.any():
            print(f"删除 {rows_with_nan.sum()} 个包含NaN的行")
            X = X[~rows_with_nan]
            # 注意：ID也需要同步删除，稍后处理
            df = df.iloc[~rows_with_nan]

        # 9. 获取清洗后的ID列表
        ids = df[idCol].astype(str).tolist()

        # 10. 加载标签并匹配
        labelDf = pd.read_csv(labelPath)
        # 假设标签CSV第一列为ID，第三列为标签（索引2）
        labelDict = dict(zip(labelDf.iloc[:, 0].astype(str), labelDf.iloc[:, 2]))

        y = []
        valid_indices = []
        for i, sid in enumerate(ids):
            if sid in labelDict:
                y.append(labelDict[sid])
                valid_indices.append(i)
            else:
                print(f"警告: ID {sid} 在标签文件中未找到，将删除该样本")

        X = X[valid_indices]
        y = np.array(y, dtype=np.int64)

        print(f"最终样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
        print(f"标签分布: {np.bincount(y)}")


        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        print("特征已归一化到[0,1]区间")

        # 12. 最终检查
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("清洗后仍存在NaN或Inf，请检查数据源")

        return X.astype(np.float32), y