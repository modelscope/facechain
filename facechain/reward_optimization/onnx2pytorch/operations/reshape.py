import torch
from torch import nn


class Reshape(nn.Module):
    def __init__(self, shape=None):
        super().__init__()
        self.shape = shape
        self.initial_input_shape = None
    def forward(self, input: torch.Tensor, shape=None):
        shape = shape if shape is not None else self.shape
        shape = [x if x != 0 else input.size(i) for i, x in enumerate(shape)]
        inp_shape = torch.tensor(input.shape)
        if self.initial_input_shape is None:
            self.initial_input_shape = inp_shape
        elif len(shape) == 2 and shape[-1] == -1:
            pass
        elif torch.equal(self.initial_input_shape, inp_shape):
            pass

        return torch.reshape(input, tuple(shape))
