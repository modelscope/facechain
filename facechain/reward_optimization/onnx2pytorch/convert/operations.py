from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from onnx import numpy_helper

from facechain.reward_optimization.onnx2pytorch.convert.attribute import extract_attributes
from facechain.reward_optimization.onnx2pytorch.convert.layer import (
    convert_layer,
    convert_linear_layer,
    convert_batch_norm_layer,
    convert_instance_norm_layer,
)
from facechain.reward_optimization.onnx2pytorch.operations import *
from facechain.reward_optimization.onnx2pytorch.operations import Resize, Upsample
from facechain.reward_optimization.onnx2pytorch.utils import value_wrapper


def convert_operations(onnx_model, batch_dim=0):
    """
    Convert onnx model operations. Yields onnx's operator_id, opeartor_name and
    converted pytorch operator.

    Parameters
    ----------
    onnx_model: onnx.ModelProto
        Loaded onnx model.
    batch_dim: int
        Usually 0 for computer vision models and 1 for NLP models.

    Returns
    -------
    iterator: (op_id, op_name, op)
    """
    weights = {tensor.name: tensor for tensor in onnx_model.graph.initializer}

    for i, node in enumerate(onnx_model.graph.node):
        params = [weights[par_name] for par_name in node.input if par_name in weights]

        if node.op_type == "Conv":
            op = convert_layer(node, "Conv", params)
        elif node.op_type == "Relu":
            op = nn.ReLU()
        elif node.op_type == "LeakyRelu":
            op = nn.LeakyReLU(**extract_attributes(node))
        elif node.op_type == "Sigmoid":
            op = nn.Sigmoid()
        elif node.op_type == "MaxPool":
            op = convert_layer(node, "MaxPool")
        elif node.op_type == "AveragePool":
            op = convert_layer(node, "AvgPool")
        elif node.op_type == "Flatten":
            op = Flatten(**extract_attributes(node))
        elif node.op_type == "Gemm":
            op = convert_linear_layer(node, params)
            op.feature_dim = batch_dim + 1  # Necessary for transformers
        elif node.op_type == "BatchNormalization":
            op = convert_batch_norm_layer(node, params=params)
        elif node.op_type == "InstanceNormalization":
            op = convert_instance_norm_layer(node, params=params)
        elif node.op_type == "Concat":
            op = Concat(**extract_attributes(node))
        elif node.op_type == "Constant":
            # 常量OP如何解决的问题，先过一遍ONNX-Simplifier
            op = value_wrapper(torch.from_numpy(extract_attributes(node)["constant"]))
        elif node.op_type == "Reshape":
            shape = list(
                filter(lambda x: x.name == node.input[1], onnx_model.graph.initializer)
            )
            shape = numpy_helper.to_array(shape[0]) if shape else None
            op = Reshape(tuple(shape))
        elif node.op_type == "Shape":
            op = Shape()
        elif node.op_type == "Gather":
            op = Gather(**extract_attributes(node))
        elif node.op_type == "Squeeze":
            op = Squeeze(**extract_attributes(node))
        elif node.op_type == "Unsqueeze":
            op = partial(torch.unsqueeze, **extract_attributes(node))
        elif node.op_type == "ConstantOfShape":
            op = ConstantOfShape(**extract_attributes(node))
        elif node.op_type == "Slice":
            op = Slice(**extract_attributes(node))
        elif node.op_type == "Cast":
            op = Cast(**extract_attributes(node))
        elif node.op_type == "Where":
            op = Where()
        elif node.op_type == "Equal":
            op = torch.eq
        elif node.op_type == "Mul":
            op = Mul(**extract_attributes(node))
        elif node.op_type == "Div":
            op = torch.true_divide
        elif node.op_type == "MatMul":
            if params:
                weight = torch.from_numpy(numpy_helper.to_array(params[0]))
                op = nn.Linear(weight.shape[0], weight.shape[1], bias=False)
                op.weight.data = weight.t()

                # check if next node Add to add bias
                next_node = onnx_model.graph.node[i + 1]
                next_params = [
                    weights[par_name]
                    for par_name in next_node.input
                    if par_name in weights
                ]
                if next_params and next_node.op_type == "Add":
                    bias = torch.from_numpy(numpy_helper.to_array(next_params[0]))
                    op.bias = nn.Parameter(bias)
                    node.output.pop()
                    node.output.extend(next_node.output)
                    onnx_model.graph.node.pop(i + 1)  # remove next node
            else:
                op = Matmul()
        elif node.op_type == "Sub":
            op = torch.sub
        elif node.op_type == "Pow":
            op = torch.pow
        elif node.op_type == "Sqrt":
            op = torch.sqrt
        elif node.op_type == "Softmax":
            op = nn.Softmax(dim = 1)
        elif node.op_type == "Transpose":
            op = partial(torch.Tensor.permute, **extract_attributes(node))
        elif node.op_type == "Split":
            kwargs = extract_attributes(node)
            # if the split_size_or_sections is not in node attributes,
            # the number_of_splits becomes the number of node outputs
            if "split_size_or_sections" not in kwargs:
                kwargs["number_of_splits"] = len(node.output)
            op = Split(**kwargs)
        elif node.op_type == "ReduceMean":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.mean, **kwargs)
        elif node.op_type == "Add":
            op = Add()
        elif node.op_type == "GlobalAveragePool":
            op = GlobalAveragePool()
        elif node.op_type == "ConvTranspose":
            op = convert_layer(node, "ConvTranspose", params)
        elif node.op_type == "Identity":
            op = nn.Identity()
        elif node.op_type == "Resize":
            op = Resize(**extract_attributes(node))
        elif node.op_type == "Upsample":
            op = Upsample(**extract_attributes(node))
        elif node.op_type == "OneHot":
            op = OneHot(**extract_attributes(node))
        elif node.op_type == "Pad":
            op = Pad(**extract_attributes(node))
        elif node.op_type == "Clip":
            op = Clamp(**extract_attributes(node))
        elif node.op_type == "Tanh":
            op = torch.tanh
        elif node.op_type == "Erf":
            op = torch.erf
        elif node.op_type == "Log":
            op = torch.log
        elif node.op_type == "Exp":
            op = torch.exp
        elif node.op_type == "LRN":
            op = nn.LocalResponseNorm(**extract_attributes(node))
        elif node.op_type == "Dropout":
            op = nn.Dropout(p=1.0)
        else:
            op = getattr(torch, node.op_type.lower(), None)
            if op is None:
                raise NotImplementedError(
                    "Conversion not implemented for op_type={}.".format(node.op_type)
                )
            else:
                print(
                    "Automatic inference of operator: {}".format(node.op_type.lower())
                )

        op_name = "{}_{}".format(node.op_type, node.output[0])
        op_id = node.output[0]
        yield op_id, op_name, op
