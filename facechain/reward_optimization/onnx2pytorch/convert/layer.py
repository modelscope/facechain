import torch
from torch import nn
from onnx import numpy_helper

from facechain.reward_optimization.onnx2pytorch.convert.attribute import extract_attributes, extract_attr_values


def extract_params(params):
    """Extract weights and biases."""
    param_length = len(params)
    if param_length == 1:
        weight = params[0]
        bias = None
    elif param_length == 2:
        weight = params[0]
        bias = params[1]
    else:
        raise ValueError("Unexpected number of parameters: {}".format(param_length))
    return weight, bias


def load_params(layer, weight, bias):
    """Load weight and bias to a given layer from onnx format."""
    layer.weight.data = torch.from_numpy(numpy_helper.to_array(weight))
    if bias is not None:
        layer.bias.data = torch.from_numpy(numpy_helper.to_array(bias))

# 卷积/反卷积/池化层的转换
def convert_layer(node, layer_type, params=None):
    """Use to convert Conv, MaxPool, AvgPool layers."""
    assert layer_type in [
        "Conv",
        "ConvTranspose",
        "MaxPool",
        "AvgPool",
    ], "Incorrect layer type: {}".format(layer_type)
    kwargs = extract_attributes(node)
    kernel_size_length = len(kwargs["kernel_size"])
    try:
        layer = getattr(nn, "{}{}d".format(layer_type, kernel_size_length))
    except AttributeError:
        raise ValueError(
            "Unexpected length of kernel_size dimension: {}".format(kernel_size_length)
        )

    if params:
        pad_layer = None
        weight, bias = extract_params(params)
        kwargs["bias"] = bias is not None
        kwargs["in_channels"] = weight.dims[1] * kwargs.get("groups", 1)
        kwargs["out_channels"] = weight.dims[0]

        if layer_type == "ConvTranspose":
            kwargs["in_channels"], kwargs["out_channels"] = (
                kwargs["out_channels"],
                kwargs["in_channels"],
            )

        # if padding is a layer, remove from kwargs and prepend later
        if isinstance(kwargs["padding"], nn.Module):
            pad_layer = kwargs.pop("padding")

        # initialize layer and load weights
        layer = layer(**kwargs)
        load_params(layer, weight, bias)
        if pad_layer is not None:
            layer = nn.Sequential(pad_layer, layer)
    else:
        # initialize operations without parameters (MaxPool, AvgPool, etc.)
        if(len(kwargs["padding"]) == 2):
            if node.op_type == "AveragePool":
                kwargs["count_include_pad"] = False
            layer = layer(**kwargs)
        else:
            if node.op_type == "AveragePool":
                kernel_size_x = kwargs["kernel_size"][0]
                kernel_size_y = kwargs["kernel_size"][1]
                pad_x = kwargs["padding"][1]
                pad_y = kwargs["padding"][3]
                kernel_size_x -= pad_x
                kernel_size_y -= pad_y
                kwargs["padding"] = (0, 0)
                kwargs["kernel_size"] = (kernel_size_x, kernel_size_y)
                layer = layer(**kwargs)
            else:
                pad_layer = nn.ConstantPad2d(kwargs["padding"], 0.0)
                kwargs["padding"] = (0, 0)
                layer = layer(**kwargs)
                layer = nn.Sequential(pad_layer, layer)
    return layer

# BN层的转换
def convert_batch_norm_layer(node, params):
    kwargs = extract_attributes(node)
    layer = nn.BatchNorm2d(params[0].dims[0])

    # kwargs["num_features"] = params[0].dims[0]
    # print(kwargs)
    # initialize layer and load weights
    # layer = layer(**kwargs)
    key = ["weight", "bias", "running_mean", "running_var"]
    for key, value in zip(key, params):
        getattr(layer, key).data = torch.from_numpy(numpy_helper.to_array(value))

    return layer

# InstanceNorm层的转换
def convert_instance_norm_layer(node, params):
    kwargs = extract_attributes(node)
    # Skips input dimension check, not possible before forward pass
    layer = nn.InstanceNorm2d()

    kwargs["num_features"] = params[0].dims[0]
    # initialize layer and load weights
    layer = layer(**kwargs)
    key = ["weight", "bias"]
    for key, value in zip(key, params):
        getattr(layer, key).data = torch.from_numpy(numpy_helper.to_array(value))

    return layer

# 全连接层的转换
def convert_linear_layer(node, params):
    """Convert linear layer from onnx node and params."""
    # Default Gemm attributes
    dc = dict(
        transpose_weight=True,
        transpose_activation=False,
        weight_multiplier=1,
        bias_multiplier=1,
    )
    dc.update(extract_attributes(node))
    for attr in node.attribute:
        if attr.name in ["transA"] and extract_attr_values(attr) != 0:
            raise NotImplementedError(
                "Not implemented for attr.name={} and value!=0.".format(attr.name)
            )

    kwargs = {}
    weight, bias = extract_params(params)
    kwargs["bias"] = bias is not None
    kwargs["in_features"] = weight.dims[1]
    kwargs["out_features"] = weight.dims[0]

    # initialize layer and load weights
    layer = nn.Linear(**kwargs)
    load_params(layer, weight, bias)

    # apply onnx gemm attributes
    if dc.get("transpose_weight"):
        layer.weight.data = layer.weight.data.t()

    layer.weight.data *= dc.get("weight_multiplier")
    if layer.bias is not None:
        layer.bias.data *= dc.get("bias_multiplier")

    return layer
