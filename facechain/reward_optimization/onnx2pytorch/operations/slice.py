import torch
from torch import nn


class Slice(nn.Module):
    def __init__(self, dim=None, starts=None, ends=None, steps=None):
        self.dim = [dim] if isinstance(dim, int) else dim
        self.starts = starts
        self.ends = ends
        self.steps = steps
        super().__init__()

    def forward(
        self, input: torch.Tensor, starts=None, ends=None, axes=None, steps=None
    ):
        if axes is None:
            axes = self.dim
        if starts is None:
            starts = self.starts
        if ends is None:
            ends = self.ends
        if steps is None:
            steps = self.steps

        # If axes=None set them to (0, 1, 2, ...)
        if axes is None:
            axes = tuple(range(len(starts)))
        if steps is None:
            steps = tuple(1 for _ in axes)

        selection = [slice(None) for _ in range(max(axes) + 1)]
        for i, axis in enumerate(axes):
            selection[axis] = slice(starts[i], ends[i], steps[i])
        return input.__getitem__(selection)
