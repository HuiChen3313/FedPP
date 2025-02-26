import torch
import torch.nn as nn
import torch.nn.functional as F

class OneLayerMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=1):
        super(OneLayerMLP, self).__init__()
        # Elevate dimensions from [B, 1] to [B, d]
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        # Reduce dimensions back from [B, d] to [B, 1]
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        # 1D normalization layer
        # self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        # Forward pass through the network
        x = self.layer1(x)
        # x = F.tanh(x)
        x = self.layer2(x)
        return x