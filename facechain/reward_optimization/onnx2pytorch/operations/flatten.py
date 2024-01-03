import torch
from torch import nn


class Flatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: torch.Tensor):
        return torch.flatten(input, start_dim=self.start_dim, end_dim=self.end_dim)
