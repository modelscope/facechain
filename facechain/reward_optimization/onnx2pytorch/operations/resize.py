import warnings
import torch
from torch import nn
from torch.nn import functional as F

empty_tensor = torch.Tensor([])


class Resize(nn.Module):
    def __init__(self, mode="nearest", align_corners=None, **kwargs):
        self.mode = mode
        self.align_corners = align_corners
        for key in kwargs.keys():
            warnings.warn(
                "Pytorch's interpolate uses no {}. " "Result might differ.".format(key)
            )
        super().__init__()

    def forward(self, inp, roi=empty_tensor, scales=empty_tensor, sizes=empty_tensor):
        if roi.nelement() > 0:
            warnings.warn("Pytorch's interpolate uses no roi. Result might differ.")

        scales = list(scales)
        sizes = list(sizes)
        shape = list(inp.shape)
        if shape[:2] == sizes[:2]:
            sizes = sizes[2:]  # Pytorch's interpolate takes only H and W params
        elif scales[:2] == [1, 1]:
            scales = scales[2:]
        elif len(scales) == 0 and len(sizes) == 0:
            raise ValueError("One of the two, scales or sizes, needs to be defined.")
        else:
            raise NotImplementedError(
                "Pytorch's interpolate does not scale batch and channel dimensions."
            )

        if len(scales) == 0:
            scales = None
        elif len(sizes) == 0:
            sizes = None
        else:
            raise ValueError(
                "Only one of the two, scales or sizes, needs to be defined."
            )

        return F.interpolate(
            inp,
            scale_factor=2,
            size=sizes,
            mode=self.mode,
            align_corners=self.align_corners,
        )


class Upsample(Resize):
    """Deprecated onnx operator."""

    def forward(self, inp, scales):
        return super().forward(inp, torch.tensor([]), scales, torch.tensor([]))
