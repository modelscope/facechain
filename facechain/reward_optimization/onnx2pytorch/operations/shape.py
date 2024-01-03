import torch
from torch import nn

class Shape(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input: torch.Tensor):
        return torch.tensor(input.shape)
