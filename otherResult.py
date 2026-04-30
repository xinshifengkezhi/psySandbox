import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class dealResult:
    def __init__(self, resultDir):
        self.resultDir = resultDir
        self.statisOutput = f'{self.resultDir}/statisInfo'
        os.makedirs(self.statisOutput, exist_ok=True)

    def getInfo(self):
        resultPath = f'{self.resultDir}/results/results_3_0.json'
        with open(resultPath, 'r') as f:
            data = json.load(f)
        metricsResults = data['results']

        #将各指标换个名字
        metricsNames = {
            'src.evaluation.evaluation_metric_correctness.CorrectnessMetric': 'Correctness',
            'src.evaluation.evaluation_metric_fidelity.FidelityMetric': 'Fidelity',
            'src.evaluation.evaluation_metric_ged.GraphEditDistanceMetric': 'Graph Edit Distance',
            'src.evaluation.evaluation_metric_oracle_accuracy.OracleAccuracyMetric': 'Oracle Accuracy',
            'src.evaluation.evaluation_metric_oracle_calls.OracleCallsMetric': 'Oracle Calls',
            'src.evaluation.evaluation_metric_runtime.RuntimeMetric': 'Runtime(s)',
            'src.evaluation.evaluation_metric_sparsity.SparsityMetric': 'Sparsity'
        }

        #提取到字典中
        metricesValues = {}
        for key, displayName in metricsNames.items():
            values = [item['value'] for item in metricsResults[key]]
            metricesValues[displayName] = values
        df = pd.DataFrame(metricesValues)

        #调用各个函数画图
        self.runTime(df['Runtime(s)'])
        self.runTimekde(df['Runtime(s)'])
        self.editDistance(df['Graph Edit Distance'])
        self.editDistanceBox(df['Graph Edit Distance'])
        self.numOracle(df["Oracle Calls"])
        self.correct(df['Correctness'])
        self.sparsity(df['Sparsity'])
        self.sparsityBox(df['Sparsity'])
        self.fidelity(df['Fidelity'])
        self.oracleAcc(df['Oracle Accuracy'])
        self.SparEdit(df)
        self.FideCorrect(df['Fidelity'], df['Correctness'])
        self.otherSave(data)

    def runTime(self, runtimes):
        """========== 1. 运行时分布 =========="""
        plt.figure()
        #直方图
        sns.histplot(runtimes, kde=True, bins=30)

        plt.title('运行时分布直方图')
        plt.xlabel('运行时间 (s)')
        plt.tight_layout()

        output = f'{self.statisOutput}/runtime-hist.png'
        plt.savefig(output, dpi=150)
        plt.close()

    def runTimekde(self, runtimes):
        """========== 1. 运行时分布 =========="""
        plt.figure()
        # 核密度估计图
        sns.kdeplot(runtimes, fill=True)
        plt.title('运行时核密度估计')
        plt.xlabel('运行时间 (s)')
        plt.tight_layout()
        output_kde = f'{self.statisOutput}/runtime-kde.png'
        plt.savefig(output_kde, dpi=150)
        plt.close()

    def editDistance(self, dists):
        """========== 2. 图编辑距离分布（对数坐标） =========="""
        plt.figure()
        sns.histplot(dists, kde=True, bins=30, log_scale=True)
        plt.title('图编辑距离分布（对数坐标）')
        plt.xlabel('图编辑距离')
        plt.tight_layout()
        output = f'{self.statisOutput}/ged-distribution-hist.png'
        plt.savefig(output, dpi=150)
        plt.close()

    def editDistanceBox(self, dists):
        """========== 2. 图编辑距离分布（对数坐标） =========="""
        plt.figure()
        # 使用 seaborn 绘制横向箱线图，并启用对数坐标
        sns.boxplot(x=dists, log_scale=True)
        plt.title('图编辑距离分布(对数坐标）')
        plt.xlabel('图编辑距离')
        plt.tight_layout()
        output = f'{self.statisOutput}/ged-distribution-box.png'  # 保持原文件名，替换原有图片
        plt.savefig(output, dpi=150)
        plt.close()

    def numOracle(self, nums):
        """========== 3. Oracle 调用次数 =========="""
        plt.figure(figsize=(6, 4))
        unique_vals = nums.value_counts().sort_index()
        n_unique = len(unique_vals)
        # 绘制扇形图显示不同调用次数的样本数
        wedges, texts, autotexts = plt.pie(
            unique_vals.values,
            labels=unique_vals.index,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        plt.title('Oracle 调用次数占比')
        plt.axis('equal')
        plt.tight_layout()
        output = f'{self.statisOutput}/oracle-calls.png'
        plt.savefig(output, dpi=150)
        plt.close()

    def correct(self, cors):
        """========== 4. 正确性分布（扇形图） =========="""
        plt.figure()
        counts = cors.value_counts().sort_index()
        # 设置标签
        labels = ['未改变 (0)', '改变 (1)']

        # 绘制扇形图
        wedges, texts, autotexts = plt.pie(
            counts.values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        plt.title('反事实解释后图的标签变化')
        plt.axis('equal')
        plt.tight_layout()
        output = f'{self.statisOutput}/correctness-distribution.png'
        plt.savefig(output, dpi=150)
        plt.close()

    def sparsity(self, spars):
        """========== 5. 稀疏度分布 （直方图）=========="""
        plt.figure()
        sns.histplot(spars, kde=True, bins=30)
        plt.title('稀疏度分布')
        plt.xlabel('稀疏度')
        plt.tight_layout()
        output = f'{self.statisOutput}/sparsity-distribution-hist.png'
        plt.savefig(output, dpi=150)
        plt.close()

    def sparsityBox(self, spars):
        """========== 5. 稀疏度分布（箱线图） =========="""
        plt.figure()
        # 使用纵向箱线图，更直观地展示中位数、四分位数及异常值
        sns.boxplot(y=spars)
        plt.title('稀疏度分布')
        plt.ylabel('稀疏度')
        plt.tight_layout()
        output = f'{self.statisOutput}/sparsity-distribution-box.png'
        plt.savefig(output, dpi=150)
        plt.close()

    def fidelity(self, fides):
        """========== 6. 保真度分布（扇形图） =========="""
        plt.figure()
        counts = fides.value_counts().sort_index()
        # 将类别值转换为字符串作为标签，例如 '-1', '0', '1'
        labels = [str(val) for val in counts.index]

        # 绘制扇形图
        wedges, texts, autotexts = plt.pie(
            counts.values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        plt.title('保真度分布')
        plt.axis('equal')
        plt.tight_layout()
        output = f'{self.statisOutput}/fidelity-distribution.png'
        plt.savefig(output, dpi=150)
        plt.close()

    def oracleAcc(self, accs):
        """========== 7. Oracle 准确率分布（扇形图） =========="""
        plt.figure()
        counts = accs.value_counts().sort_index()
        # 将类别值转换为字符串作为标签（例如 "0", "1"）
        labels = [str(val) for val in counts.index]

        wedges, texts, autotexts = plt.pie(
            counts.values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        plt.title('Oracle 准确率分布')
        plt.axis('equal')
        plt.tight_layout()
        output = f'{self.statisOutput}/oracle-accuracy-distribution.png'
        plt.savefig(output, dpi=150)
        plt.close()

    def SparEdit(self, df):
        """========== 8. 对比图：稀疏度 vs 图编辑距离（按保真度着色） =========="""
        plt.figure(figsize=(10, 6))
        scatter = sns.scatterplot(data=df, x='Sparsity', y='Graph Edit Distance',
                                  hue='Fidelity', palette='viridis', alpha=0.6)
        plt.title('稀疏度 和 图编辑距离（按保真度着色）')
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        output = f'{self.statisOutput}/sparsity-ged.png'
        plt.savefig(output, dpi=150)
        plt.close()

    def FideCorrect(self, fides, cors):
        """========== 9. 对比图：保真度与正确性的堆叠条形图 =========="""
        plt.figure(figsize=(8, 6))
        cross = pd.crosstab(fides, cors)
        cross.plot(kind='bar', stacked=True, colormap='Set2')
        plt.title('保真度 和 正确性的堆叠条形图')
        plt.xlabel('保真度')
        plt.ylabel('图数量')
        plt.legend(title='Correctness')
        plt.tight_layout()
        output = f'{self.statisOutput}/fidelity-correctness.png'
        plt.savefig(output, dpi=150)
        plt.close()

    def otherSave(self, data):
        # ========== 打印实验配置摘要 ==========
        output = f'{self.statisOutput}/otherInfo.txt'
        with open(output, 'w', encoding='utf-8') as f:
            config = data['config']
            f.write('=============== 实验配置摘要 ===============\n')
            f.write(f"数据集: {config['dataset']['parameters']['name']}\n")
            f.write(f"节点特征维度: {config['dataset']['parameters']['node_features_dim']}\n")
            f.write(f"解释器: {config['explainer']['class']}\n")
            f.write(f"解释器参数: alpha={config['explainer']['parameters']['alpha']}，"
                    f"heads={config['explainer']['parameters']['heads']}，"
                    f"hidden_dim={config['explainer']['parameters']['hidden_dim']}\n")
            f.write(f"Oracle 模型: {config['oracle']['parameters']['model']['class']}\n")
            f.write(f"Oracle 参数: hiddenDim={config['oracle']['parameters']['model']['parameters']['hiddenDim']}, "
                    f"numLayers={config['oracle']['parameters']['model']['parameters']['numLayers']}\n")
            f.write(f"运行 ID: {data['config']['run_id']}, 折数: {data['config']['fold_id']}\n")

        print("\n所有图表已保存为 PNG 文件。")

if __name__ == '__main__':
    dr = dealResult('gistOutput/GCN-output')
    dr.getInfo()