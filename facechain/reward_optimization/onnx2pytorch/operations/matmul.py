import torch
from torch import nn

class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        return torch.matmul(input1, input2)
