from itertools import accumulate
import torch
from torch import nn
from facechain.reward_optimization.onnx2pytorch.utils import assign_values_to_dim


class Split(nn.Module):
    def __init__(
        self, split_size_or_sections=None, number_of_splits=None, dim=0, keep_size=True
    ):
        """
        Parameters
        ----------
        split_size_or_sections: tuple[int]
        number_of_splits: int
            The number of equal splits along dim.
        dim: int
            Split dimension. Tensor is split over this axis.
        keep_size: bool
            If True it keeps the size of the split the same as in initial pass.
            Else it splits it accordingly to the pruned input.
        """
        assert (
            split_size_or_sections is not None or number_of_splits is not None
        ), "One of the parameters needs to be set."
        self.dim = dim
        self.split_size_or_sections = split_size_or_sections
        self.number_of_splits = number_of_splits
        self.keep_size = keep_size
        self.input_indices = None
        self.placeholder = None
        super().__init__()

    def _get_sections(self, input):
        """Calculate sections from number of splits."""
        dim_size = input[0].shape[self.dim]
        assert (
            dim_size % self.number_of_splits == 0
        ), "Dimension size {} not equally divisible by {}.".format(
            dim_size, self.number_of_splits
        )
        s = dim_size // self.number_of_splits
        sections = tuple(s for _ in range(self.number_of_splits))
        return sections

    def forward(self, *input):
        if self.split_size_or_sections is None:
            self.split_size_or_sections = self._get_sections(input)

        if self.input_indices is not None:
            self.placeholder *= 0
            assign_values_to_dim(
                self.placeholder, input[0], self.input_indices, self.dim
            )
            split = torch.split(self.placeholder, self.split_size_or_sections, self.dim)
        else:
            split = torch.split(*input, self.split_size_or_sections, dim=self.dim)
        return split

    def set_input_indices(self, input: tuple):
        assert isinstance(input, (tuple, list))

        inp = input[0]
        # We assume that aggregation dimensions correspond to split dimension
        axis = self.get_axis(inp.shape, self.dim)

        # Mask shows where features are non zero in the whole axis.
        mask = inp != 0
        if len(inp.shape) > 1:
            mask = mask.sum(axis=tuple(axis)) != 0

        if not self.keep_size:
            # Read docstrings
            if isinstance(self.split_size_or_sections, tuple):
                indices = list(accumulate(self.split_size_or_sections))
                indices = torch.tensor(indices) - 1
            else:
                raise NotImplementedError("Not implemented for split size.")
            cs = torch.cumsum(mask, 0)
            ind = [0] + cs[indices].tolist()
            sec = [ind[i + 1] - ind[i] for i in range(len(ind) - 1)]
            self.split_size_or_sections = sec
        else:
            (self.input_indices,) = torch.where(mask)
            self.placeholder = torch.zeros(inp.shape)

    def __str__(self):
        return "Split(dim={})".format(self.dim)
