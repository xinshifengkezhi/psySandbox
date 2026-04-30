import torch.nn as nn
import torch
from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn.functional as F

"""
    模型类，默认包含三个RGCN层,分类器定义了两个全连接层和一个正则化层
    nodeDim表示输入层的特征维度，
    hiddenDim表示隐藏层的维度
    numRelations表示边的关系类型，由于目前先做分类，因此先做一种关系类型
    numClasses为最终的输出维度，将结果映射到0-2的分数中

"""
class RelateModel(nn.Module):
    def __init__(self, nodeDim, hiddenDim, numRelations, numClasses, numLayers=3,
                 alp=1.0, probably=0.5):
        super().__init__()
        self.numLayers = numLayers
        # 初始化 ModuleList
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.prob = probably

        # 输入层
        self.convs.append(RGCNConv(nodeDim, hiddenDim, numRelations))
        # 如果总层数大于1，输入层之后需要归一化
        if numLayers > 1:
            self.norms.append(nn.BatchNorm1d(hiddenDim))

        # 中间隐藏层
        for _ in range(numLayers - 2):
            self.convs.append(RGCNConv(hiddenDim, hiddenDim, numRelations))
            self.norms.append(nn.BatchNorm1d(hiddenDim))

        # 输出层（最后一层 RGCN，无归一化）
        if numLayers > 1:
            self.convs.append(RGCNConv(hiddenDim, hiddenDim, numRelations))
        # self.norms 数量 = (1 if numLayers>1 else 0) + (numLayers-2) = numLayers-1

        self.activation = nn.ELU(alpha=alp)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hiddenDim, hiddenDim // 2),
            nn.ELU(alpha=alp),
            nn.Dropout(probably),
            nn.Linear(hiddenDim // 2, numClasses)
        )

    """
        前向传播函数
        这里的edgeType默认为None，自动处理该情况，将所有的边设置为同一种关系类型
    """
    def forward(self, x, edgeIndex, edgeType=None, batch=None):

        if edgeType is None:
            edgeType = torch.zeros(edgeIndex.size(1), dtype=torch.long, device=edgeIndex.device)

        # 用于残差连接的变量（前一层的输出）
        prev_x = None

        for i, conv in enumerate(self.convs):
            # RGCN 卷积
            x_new = conv(x, edgeIndex, edgeType)

            # 残差连接：如果维度匹配，则加上前一层的输出（跳过连接）
            if prev_x is not None and x_new.shape == prev_x.shape:
                x_new = x_new + prev_x

            # 如果不是最后一层，则进行归一化、激活和 dropout
            if i != self.numLayers - 1:
                # 使用对应的 norm 层（注意 norms 列表顺序与 convs 前 numLayers-1 层对应）
                x_new = self.norms[i](x_new)
                x_new = self.activation(x_new)
                x_new = F.dropout(x_new, p=self.prob, training=self.training)

            # 更新 prev_x 为当前层的输出（用于下一层的残差）
            prev_x = x_new
            x = x_new   # 更新 x 继续下一层

        # 全局池化（图分类时使用）
        if batch is not None:
            x = global_mean_pool(x, batch)

        # 分类
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)