import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATModel(nn.Module):
    """
    基于 GAT 的图分类模型，结构与 SandBoxModel (GCN) 类似
    :param nodeDim: 输入节点特征维度
    :param hiddenDim: 隐藏层维度（每个注意力头的输出维度）
    :param numClasses: 分类类别数
    :param numLayers: GAT 层数（包括输出层）
    :param heads: 多头注意力的头数（除最后一层外，每层使用 heads 个头）
    :param dropout: Dropout 概率
    :param alp: ELU 激活函数的 alpha 参数
    :param convLayers: 池化后的一维卷积层数（可选）
    :param convKernel: 一维卷积核大小
    :param convPadding: 一维卷积填充
    """
    def __init__(self, nodeDim, hiddenDim, numClasses, numLayers, heads,
                 dropout, alp, convLayers=0, convKernel=5, convPadding=2):
        super().__init__()
        self.numLayers = numLayers
        self.convLayers = convLayers
        self.dropout = dropout

        # 存储 GAT 层和对应的批归一化层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # 输入层：多头输出，concat=True 输出维度 = hiddenDim * heads
        self.convs.append(GATConv(nodeDim, hiddenDim, heads=heads, dropout=dropout, concat=True))
        if numLayers > 1:
            self.norms.append(nn.BatchNorm1d(hiddenDim * heads))

        # 中间隐藏层
        for _ in range(numLayers - 2):
            self.convs.append(GATConv(hiddenDim * heads, hiddenDim, heads=heads, dropout=dropout, concat=True))
            self.norms.append(nn.BatchNorm1d(hiddenDim * heads))

        # 输出层（最后一层）：使用单头，concat=False，输出维度 = hiddenDim
        if numLayers > 1:
            self.convs.append(GATConv(hiddenDim * heads, hiddenDim, heads=1, dropout=dropout, concat=False))

        self.activation = nn.ELU(alpha=alp)

        # 可选：池化后的一维卷积
        if convLayers > 0:
            self.postConvs = nn.ModuleList()
            self.postNorms = nn.ModuleList()
            inCh = 1
            for i in range(convLayers):
                outCh = hiddenDim if i < convLayers - 1 else hiddenDim // 2
                conv = nn.Conv1d(inCh, outCh, kernel_size=convKernel, padding=convPadding)
                self.postConvs.append(conv)
                if i < convLayers - 1:
                    self.postNorms.append(nn.BatchNorm1d(outCh))
                inCh = outCh
            self.adaptivePool = nn.AdaptiveAvgPool1d(1)
            classifier_input_dim = hiddenDim // 2
        else:
            classifier_input_dim = hiddenDim

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, classifier_input_dim // 2),
            nn.ELU(alpha=alp),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_dim // 2, numClasses)
        )

    def forward(self, x, edgeIndex, batch=None):
        """
        x: 节点特征矩阵 (N, nodeDim)
        edgeIndex: 边索引 (2, E)
        batch: 批次向量 (N,)，用于全局池化，若为 None 则返回节点级输出
        """
        x = x.float()
        prev_x = None

        for i, conv in enumerate(self.convs):
            # GAT 卷积
            x_new = conv(x, edgeIndex)

            # 残差连接（维度匹配时）
            if prev_x is not None and x_new.shape == prev_x.shape:
                x_new = x_new + prev_x

            # 如果不是最后一层，进行归一化、激活、dropout
            if i != self.numLayers - 1:
                x_new = self.norms[i](x_new)
                x_new = self.activation(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            prev_x = x_new
            x = x_new

        # 图级别池化
        if batch is not None:
            x = global_mean_pool(x, batch)

            # 可选：池化后的一维卷积处理
            if self.convLayers > 0:
                x = x.unsqueeze(1)  # (batch, 1, hiddenDim)
                for i, conv in enumerate(self.postConvs):
                    x = conv(x)
                    if i < self.convLayers - 1:
                        x = self.postNorms[i](x)
                        x = self.activation(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.adaptivePool(x)  # (batch, outCh, 1)
                x = x.squeeze(-1)

        # 保存嵌入
        embed = x.clone()

        # 分类
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1), embed