from FeatureExtraction.Forecast.Fusion import FusionDBSCANLOF
from FeatureExtraction.Forecast.fssod import FSSODDetector
from FeatureExtraction.Forecast.fshics import FSHiCSDetector
from registry import register

import warnings
warnings.filterwarnings('ignore')
import pandas as pd

def load_all_data(excel_path, id_col=0, feature_start_col=5, header=0):
    """
    读取Excel数据，自动筛选数值列并处理异常值。
    """
    df = pd.read_excel(excel_path, header=header)

    # 提取ID列
    ids = df.iloc[:, id_col].astype(str).tolist()

    # 提取候选特征区域
    features_df = df.iloc[:, feature_start_col:]

    # 筛选数值列：能通过pd.to_numeric转换且至少有一个有效数值的列
    numeric_cols = []
    for col in features_df.columns:
        converted = pd.to_numeric(features_df[col], errors='coerce')
        if not converted.isna().all():
            numeric_cols.append(col)

    if len(numeric_cols) == 0:
        raise ValueError("没有找到数值特征列，请检查数据。")

    # 对保留的列再次强制转换为数值，无效值转为NaN
    features_numeric = features_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # 填充NaN为0
    features_numeric = features_numeric.fillna(0)

    # 转换为float数组
    features = features_numeric.values.astype(float)

    print(f"从Excel读取了 {len(ids)} 个样本，保留了 {features.shape[1]} 个数值特征列。")
    return features, ids

# ---------------------------- 训练与预测 ----------------------------
@register('forecast')
def main():
    excel_path = "data/7131个沙盘特征汇总+标注-两位标注全部.xlsx"
    label_csv_path = "trainData/sandBoxData/label.csv"

    # 读取所有Excel数据
    all_X, all_ids = load_all_data(excel_path, id_col=0, feature_start_col=3)

    # 读取label.csv中的ID
    label_df = pd.read_csv(label_csv_path)
    valid_ids = set(label_df['graphId'].astype(str).tolist())

    # 划分：训练集 = 不在valid_ids中的样本，测试集 = 在valid_ids中的样本
    train_mask = [id_ in valid_ids for id_ in all_ids]

    X_train = all_X[train_mask]
    X_test = X_train
    train_ids = [all_ids[i] for i in range(len(all_ids)) if train_mask[i]]
    test_ids = train_ids

    print(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    # 如果训练集为空，则报错
    if len(X_train) == 0:
        raise ValueError("训练集为空，请检查数据划分条件。")

    # 选择算法类型
    algorithm = 'fusion'

    if algorithm == 'fusion':
        detector = FusionDBSCANLOF(eps=None, min_samples=5, lof_neighbors=15)
    elif algorithm == 'fs_sod':
        detector = FSSODDetector(n_bins=10, feature_mode='MRTFS', n_features_to_select=50,
                                 sod_k_neighbors=20, sod_l_reference=10, sod_alpha_fss=0.5)
    elif algorithm == 'fs_hics':
        detector = FSHiCSDetector(n_bins=10, feature_mode='MRTFS', n_features_to_select=50,
                                  hi_target_dim=3, hi_n_top_subspaces=5, hi_n_mcm=30,
                                  lof_neighbors=15, random_state=42)
    else:
        raise ValueError("Unknown algorithm")

    # 训练
    print("开始训练...")
    detector.fit(X_train)
    print("训练完成。")

    # 预测测试集
    print("开始预测...")
    scores, intermediate_features = detector.predict(X_test)
    print("预测完成。")

    # 保存结果（包含测试集ID、异常分数和中间特征向量）
    feat_cols = [f"selected_feat_{i}" for i in range(intermediate_features.shape[1])]
    df_result = pd.DataFrame(intermediate_features, columns=feat_cols)
    df_result.insert(0, 'id', test_ids)
    df_result.insert(1, 'anomaly_score', scores)

    output_path = "data/intermediate-features.xlsx"
    df_result.to_excel(output_path, index=False)
    print(f"结果已保存至 {output_path}")

if __name__ == "__main__":
    main()