import torch.nn as nn

class Transformer1D(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)  # 从1维映射到d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, 1, seq_len) → (batch, seq_len, 1)
        x = x.transpose(1, 2)  # 交换第1维和第2维
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.transformer(x)  # (batch, seq_len, d_model)
        x = x.mean(dim=1)  # 全局平均池化
        return self.fc(x)