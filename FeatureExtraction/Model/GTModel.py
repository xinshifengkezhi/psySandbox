import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool

class GraphTransFormer(nn.Module):
    def __init__(self, nodeDim, hiddenDim, numClasses,
                 numLayers=3,
                 dropout=0.1,
                 numHeads=4,
                 edgeDim=1
                 ):

        """
        模型类，默认包含三个GCN层,分类器定义了两个全连接层和一个正则化层，还有残差连接
        :param nodeDim: 输入层的特征维度
        :param hiddenDim: 隐藏层的维度
        :param numRelations: 边的关系类型，由于目前先做分类，因此先做一种关系类型
        :param numClasses: 最终的输出维度，将结果映射到0-2的分数中
        :param numLayers: 隐藏层的层数
        :param alp: ELU激活函数的超参数
        :param probably: Dropout丢弃概率
        :param numHeads:多头注意力层数
        :param edgeDim:边特征维度
        """
        super(GraphTransFormer, self).__init__()

        #节点特征投影
        self.nodeEncoder = nn.Linear(nodeDim, hiddenDim)

        #边特征编码
        self.edgeEncoder = nn.Sequential(
            nn.Linear(edgeDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim)
        )

        #图TransFormer层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(numLayers):
            conv = TransformerConv(
                in_channels=hiddenDim,
                out_channels=hiddenDim,
                heads=numHeads,
                concat=False,
                dropout=dropout,
                edge_dim=hiddenDim,
                bias=True
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hiddenDim))

        self.dropout = nn.Dropout(dropout)

        #全局平均池化
        self.pool = global_mean_pool

        #分类头
        self.classifier = nn.Linear(hiddenDim, numClasses)

    def forward(self, x, edgeIndex, edgeAttr, batch=None):
        """
        :param x: 节点特征
        :param edgeIndex: 边索引
        :param edgeAttr: 边特征
        :param batch: 批次索引
        :return: 每个图的log_softmax概率
        """
        #特征编码
        x = self.nodeEncoder(x)
        if edgeAttr.dim() == 1:
            edgeAttr = edgeAttr.unsqueeze(-1)
        edgeEmbed = self.edgeEncoder(edgeAttr)

        #多层图TransFormer
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, edgeIndex, edge_attr=edgeEmbed)
            x = norm(x + residual)
            x = F.relu(x)
            x = self.dropout(x)

        #图池化
        if batch is None:
            x = x.mean(dim=0, keepdim=True)
        else:
            x = self.pool(x, batch)

        #图分类
        out = self.classifier(x)
        out = F.log_softmax(out, dim=-1)
        return out
