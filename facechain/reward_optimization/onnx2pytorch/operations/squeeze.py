import torch
from torch import nn
from facechain.reward_optimization.onnx2pytorch.utils import get_selection


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        self.dim = dim
        super().__init__()

    def forward(self, input):
        if self.dim is None:
            return torch.squeeze(input)
        elif isinstance(self.dim, int):
            return torch.squeeze(input, dim=self.dim)
        else:
            for dim in sorted(self.dim, reverse=True):
                input = torch.squeeze(input, dim=dim)
            return input
