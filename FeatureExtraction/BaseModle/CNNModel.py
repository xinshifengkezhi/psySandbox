import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, inputDim, numClasses, num_layers=1, use_residual=True, dropout=0.1):
        super().__init__()
        self.use_residual = use_residual
        # 第一层固定
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2, 2)

        if use_residual:
            # 残差模式：每个残差块包含2层
            num_blocks = max(1, num_layers // 2)
            self.res_blocks = nn.ModuleList()
            in_ch = 64
            out_ch = 64
            for i in range(num_blocks):
                if i > 0 and i % 2 == 0:  # 每2个块翻倍通道
                    out_ch = min(256, out_ch * 2)
                self.res_blocks.append(ResidualBlock1D(in_ch, out_ch, stride=1 if i==0 else 1))
                in_ch = out_ch
        else:
            # 普通卷积模式
            self.conv_layers = nn.ModuleList()
            in_ch = 64
            out_ch = 64
            for i in range(num_layers - 1):  # 减去第一层
                if i > 0 and i % 2 == 0:
                    out_ch = min(256, out_ch * 2)
                self.conv_layers.append(nn.Conv1d(in_ch, out_ch, 3, padding=1))
                self.conv_layers.append(nn.BatchNorm1d(out_ch))
                self.conv_layers.append(nn.ReLU())
                if (i+1) % 2 == 0:  # 每2层池化一次
                    self.conv_layers.append(nn.MaxPool1d(2, 2))
                in_ch = out_ch

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, numClasses)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        if self.use_residual:
            for block in self.res_blocks:
                x = block(x)
        else:
            for layer in self.conv_layers:
                x = layer(x)

        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResidualBlock1D(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(inChannels, outChannels, kernelSize, stride, padding)
        self.bn1 = nn.BatchNorm1d(outChannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(outChannels, outChannels, kernelSize, 1, padding)
        self.bn2 = nn.BatchNorm1d(outChannels)
        self.downsample = None
        if stride != 1 or inChannels != outChannels:
            self.downsample = nn.Sequential(
                nn.Conv1d(inChannels, outChannels, 1, stride),
                nn.BatchNorm1d(outChannels)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out
