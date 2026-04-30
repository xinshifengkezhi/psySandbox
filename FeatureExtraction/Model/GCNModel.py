import torch.nn as nn
import torch
# from torch_geometric.nn import RGCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SandBoxModel(nn.Module):
    def __init__(self, nodeDim, hiddenDim, numClasses, numLayers=3, numRelations=1 ,
                 alp=1.0, probably=0.5,
                 convLayers=0, convKernel=5, convPadding=2):
        """
        模型类，默认包含三个GCN层,分类器定义了两个全连接层和一个正则化层，还有残差连接
        :param nodeDim: 输入层的特征维度
        :param hiddenDim: 隐藏层的维度
        :param numRelations: 边的关系类型，由于目前先做分类，因此先做一种关系类型
        :param numClasses: 最终的输出维度，将结果映射到0-2的分数中
        :param numLayers: 隐藏层的层数
        :param alp: ELU激活函数的超参数
        :param probably: Dropout丢弃概率
        """
        super().__init__()
        self.numLayers = numLayers
        self.convLayers = convLayers
        # 初始化 ModuleList
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.prob = probably

        # 输入层
        # self.convs.append(RGCNConv(nodeDim, hiddenDim, numRelations))
        self.convs.append(GCNConv(nodeDim, hiddenDim))
        # 如果总层数大于1，输入层之后需要归一化
        if numLayers > 1:
            self.norms.append(nn.BatchNorm1d(hiddenDim))

        # 中间隐藏层
        for _ in range(numLayers - 2):
            # self.convs.append(RGCNConv(hiddenDim, hiddenDim, numRelations))
            self.convs.append(GCNConv(hiddenDim, hiddenDim))
            self.norms.append(nn.BatchNorm1d(hiddenDim))

        # 输出层（最后一层，无归一化）
        if numLayers > 1:
            # self.convs.append(RGCNConv(hiddenDim, hiddenDim, numRelations))
            self.convs.append(GCNConv(hiddenDim, hiddenDim))
        # self.norms 数量 = (1 if numLayers>1 else 0) + (numLayers-2) = numLa
        # yers-1

        self.activation = nn.ELU(alpha=alp)
        # self.activation = nn.ReLU(inplace=False)
        # self.activation = nn.Sigmoid()

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
            # 自适应池化到长度 1
            self.adaptivePool = nn.AdaptiveAvgPool1d(1)
            classifier = hiddenDim // 2
        else:
            # 如果没有卷积层，分类器输入不变
            classifier = hiddenDim

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(classifier, classifier // 2),
            nn.ELU(alpha=alp),
            # nn.ReLU(inplace=False),
            # nn.Sigmoid(),
            nn.Dropout(probably),
            nn.Linear(classifier // 2, numClasses)
        )

    """
        前向传播函数
        这里的edgeType默认为None，自动处理该情况，将所有的边设置为同一种关系类型
    """
    def forward(self, x, edgeIndex, batch=None):
        x = x.float()

        # if edgeType is None:
        #     edgeType = torch.zeros(edgeIndex.size(1), dtype=torch.long, device=edgeIndex.device)

        # 用于残差连接的变量（前一层的输出）
        prev_x = None

        for i, conv in enumerate(self.convs):
            # GCN 卷积
            x_new = conv(x, edgeIndex)

            # 残差连接：如果维度匹配，则加上前一层的输出（跳过连接）
            if prev_x is not None and x_new.shape == prev_x.shape:
                x_new = x_new + prev_x

            # 如果不是最后一层，则进行归一化、激活和 dropout
            if i != self.numLayers - 1:
                # 使用对应的 norm 层
                x_new = self.norms[i](x_new)
                x_new = self.activation(x_new)
                x_new = F.dropout(x_new, p=self.prob, training=self.training)

            # 更新 prev_x 为当前层的输出（用于下一层的残差）
            prev_x = x_new
            x = x_new   # 更新 x 继续下一层

        # 全局池化（图分类时使用）
        if batch is not None:
            x = global_mean_pool(x, batch)

            # ==================== 新增：池化后的卷积处理 ====================
            if self.convLayers > 0:
                x = x.unsqueeze(1)
                for i, conv in enumerate(self.postConvs):
                    x = conv(x)
                    if i < self.convLayers - 1:
                        x = self.postNorms[i](x)
                        x = self.activation(x)
                        x = F.dropout(x, p=self.prob, training=self.training)
                x = self.adaptivePool(x)
                x = x.squeeze(-1)

        # 分类
        embed = x.clone()

        x = self.classifier(x)
        return F.log_softmax(x, dim=-1), embed
