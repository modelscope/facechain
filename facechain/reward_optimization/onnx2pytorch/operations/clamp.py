import torch
from torch import nn


class Clamp(nn.Module):
    def __init__(self, min, max):
        self.min = min
        self.max = max
        super().__init__()

    def forward(self, input):
        return torch.clamp(input, self.min, self.max)
