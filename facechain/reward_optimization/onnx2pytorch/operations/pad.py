import torch.nn.functional as F
from torch import nn

class Pad(nn.Module):
    def __init__(self, mode="constant", padding=None):
        self.mode = mode
        self.padding = padding
        super().__init__()

    def forward(self, input, pads=None, value=0):
        if self.padding is not None:
            pads = self.padding
        elif pads is None:
            raise TypeError("pad forward() missing 1 required positional argument: 'pads'")
        return F.pad(input, list(pads), mode=self.mode, value=value)
