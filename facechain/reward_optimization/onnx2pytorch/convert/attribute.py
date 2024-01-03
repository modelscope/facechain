import warnings

import onnx
from onnx import numpy_helper


from facechain.reward_optimization.onnx2pytorch.utils import (
    extract_padding_params_for_conv_layer,
    extract_padding_params,
)

TENSOR_PROTO_MAPPING = dict([i[::-1] for i in onnx.TensorProto.DataType.items()])

AttributeType = dict(
    UNDEFINED=0,
    FLOAT=1,
    INT=2,
    STRING=3,
    TENSOR=4,
    GRAPH=5,
    SPARSE_TENSOR=11,
    FLOATS=6,
    INTS=7,
    STRINGS=8,
    TENSORS=9,
    GRAPHS=10,
    SPARSE_TENSORS=12,
)

# 获取ONNX节点属性的具体值
def extract_attr_values(attr):
    """Extract onnx attribute values."""
    if attr.type == AttributeType["INT"]:
        value = attr.i
    elif attr.type == AttributeType["FLOAT"]:
        value = attr.f
    elif attr.type == AttributeType["INTS"]:
        value = tuple(attr.ints)
    elif attr.type == AttributeType["FLOATS"]:
        value = tuple(attr.floats)
    elif attr.type == AttributeType["TENSOR"]:
        value = numpy_helper.to_array(attr.t)
    elif attr.type == AttributeType["STRING"]:
        value = attr.s.decode()
    else:
        raise NotImplementedError(
            "Extraction of attribute type {} not implemented.".format(attr.type)
        )
    return value

# 提取ONNX节点的各个属性
def extract_attributes(node):
    """Extract onnx attributes. Map onnx feature naming to pytorch."""
    kwargs = {}
    for attr in node.attribute:
        if attr.name == "dilations":
            kwargs["dilation"] = extract_attr_values(attr)
        elif attr.name == "group":
            kwargs["groups"] = extract_attr_values(attr)
        elif attr.name == "kernel_shape":
            kwargs["kernel_size"] = extract_attr_values(attr)
        elif attr.name == "pads":
            params = extract_attr_values(attr)
            if node.op_type == "Pad":
                kwargs["padding"] = extract_padding_params(params)
            else:
                # Works for Conv, MaxPooling and other layers from convert_layer func
                kwargs["padding"] = extract_padding_params_for_conv_layer(params)
        elif attr.name == "strides":
            kwargs["stride"] = extract_attr_values(attr)
        elif attr.name == "axis" and node.op_type == "Flatten":
            kwargs["start_dim"] = extract_attr_values(attr)
        elif attr.name == "axis" or attr.name == "axes":
            v = extract_attr_values(attr)
            if isinstance(v, (tuple, list)) and len(v) == 1:
                kwargs["dim"] = v[0]
            else:
                kwargs["dim"] = v
        elif attr.name == "keepdims":
            kwargs["keepdim"] = bool(extract_attr_values(attr))
        elif attr.name == "epsilon":
            kwargs["eps"] = extract_attr_values(attr)
        elif attr.name == "momentum":
            kwargs["momentum"] = extract_attr_values(attr)
        elif attr.name == "ceil_mode":
            kwargs["ceil_mode"] = bool(extract_attr_values(attr))
        elif attr.name == "value":
            kwargs["constant"] = extract_attr_values(attr)
        elif attr.name == "perm":
            kwargs["dims"] = extract_attr_values(attr)
        elif attr.name == "split":
            kwargs["split_size_or_sections"] = extract_attr_values(attr)
        elif attr.name == "spatial":
            kwargs["spatial"] = extract_attr_values(attr)  # Batch norm parameter
        elif attr.name == "to":
            kwargs["dtype"] = TENSOR_PROTO_MAPPING[extract_attr_values(attr)].lower()
        elif attr.name == "mode":
            kwargs["mode"] = extract_attr_values(attr)
        elif attr.name == "transB":
            kwargs["transpose_weight"] = not extract_attr_values(attr)
        elif attr.name == "transA":
            kwargs["transpose_activation"] = bool(extract_attr_values(attr))
        elif attr.name == "alpha" and node.op_type == "LeakyRelu":
            kwargs["negative_slope"] = extract_attr_values(attr)
        elif attr.name == "alpha" and node.op_type != "LRN":
            kwargs["weight_multiplier"] = extract_attr_values(attr)
        elif attr.name == "beta" and node.op_type != "LRN":
            kwargs["bias_multiplier"] = extract_attr_values(attr)
        elif attr.name == "starts":
            kwargs["starts"] = extract_attr_values(attr)
        elif attr.name == "ends":
            kwargs["ends"] = extract_attr_values(attr)
        elif attr.name == "coordinate_transformation_mode":
            arg = extract_attr_values(attr)
            if arg == "align_corners":
                kwargs["align_corners"] = True
            else:
                warnings.warn(
                    "Pytorch's interpolate uses no coordinate_transformation_mode={}. "
                    "Result might differ.".format(arg)
                )
        elif node.op_type == "Resize":
            # These parameters are not used, warn in Resize operator
            kwargs[attr.name] = extract_attr_values(attr)
        elif attr.name == "auto_pad":
            value = extract_attr_values(attr)
            if value == "NOTSET":
                pass
            else:
                raise NotImplementedError(
                    "auto_pad={} functionality not implemented.".format(value)
                )
        elif attr.name == "alpha" and node.op_type == "LRN":
            kwargs["alpha"] = extract_attr_values(attr)
        elif attr.name == "beta" and node.op_type == "LRN":
            kwargs["beta"] = extract_attr_values(attr)
        elif attr.name == "bias" and node.op_type == "LRN":
            kwargs["k"] = extract_attr_values(attr)
        elif attr.name == "size" and node.op_type == "LRN":
            kwargs["size"] = extract_attr_values(attr)
        elif attr.name == "min":
            kwargs["min"] = extract_attr_values(attr)
        elif attr.name == "max":
            kwargs["max"] = extract_attr_values(attr)
        else:
            raise NotImplementedError(
                "Extraction of attribute {} not implemented.".format(attr.name)
            )
    if 'eps' in kwargs.keys() or 'momentum' in kwargs.keys():
        kwargs.pop('eps', None)
        kwargs.pop('momentum', None)
    return kwargs
