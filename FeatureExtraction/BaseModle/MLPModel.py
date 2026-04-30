import torch.nn as nn

class PureMLP(nn.Module):
    """
    纯线性层模型，输入形状 (batch, 1, inputLength)
    """

    def __init__(self, inputLength, numClasses, hiddenDims, dropout=0.1):
        super().__init__()
        self.flatten = nn.Flatten()

        # 构建全连接层序列
        layers = []
        prevDim = inputLength  # 展平后的维度
        for hdim in hiddenDims:
            layers.append(nn.Linear(prevDim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prevDim = hdim

        layers.append(nn.Linear(prevDim, numClasses))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)