import torch
from torch import nn


class Mul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        return torch.mul(input1, input2)
