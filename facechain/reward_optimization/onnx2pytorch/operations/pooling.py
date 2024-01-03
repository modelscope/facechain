import torch
from torch import nn


class GlobalAveragePool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, input: torch.Tensor):
        return self.avgpool(input)
