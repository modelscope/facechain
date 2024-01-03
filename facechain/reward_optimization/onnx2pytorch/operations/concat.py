import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.axis = dim

    def forward(self, *input):
        return torch.cat(input, axis=self.axis)
