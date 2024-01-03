import torch
from torch import nn


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        return torch.add(input1, input2)
