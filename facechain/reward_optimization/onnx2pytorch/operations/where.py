import torch
from torch import nn


class Where(nn.Module):
    def forward(self, *input):
        return torch.where(input[0], input[1], input[2])
