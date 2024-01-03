import torch
from torch import nn


class Gather(nn.Module):
    def __init__(self, dim=0):
        self.dim = dim
        super().__init__()

    def forward(self, input: torch.Tensor, index: torch.Tensor):
        return torch.gather(input, self.dim, index)
