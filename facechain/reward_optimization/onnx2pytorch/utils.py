import io

import torch
import numpy as np
import onnx

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def value_wrapper(value):
    def callback(*args, **kwargs):
        return value

    return callback

def is_constant(value):
    return value.ndim == 0 or value.shape == torch.Size([1])


def is_symmetric(params):
    """
    Check if parameters are symmetric, all values [2,2,2,2].
    Then we can use only [2,2].
    """
    assert len(params) // 2 == len(params) / 2, "Non even number of parameters."
    idx = len(params) // 2
    for i in range(0, idx):
        if params[i] != params[idx + i]:
            return False
    return True

# 为ConstantPad2D这种OP提取padding参数
def extract_padding_params(params):
    """Extract padding parameters fod Pad layers."""
    pad_dim = len(params) // 2
    pads = np.array(params).reshape(-1, pad_dim).T.flatten()  # .tolist()

    # Some padding modes do not support padding in batch and channel dimension.
    # If batch and channel dimension have no padding, discard.
    if (pads[:4] == 0).all():
        pads = pads[4:]
    pads = pads.tolist()
    # Reverse, because for pytorch first two numbers correspond to last dimension, etc.
    pads.reverse()
    return pads

# 为卷积OP提取padding参数
# >>> import torch
# >>> x = [1, 2,3, 4]
# >>> import numpy as np
# >>> y = np.array(x).reshape(-1,2).T.flatten()
# >>> print(y)
# [1 3 2 4]
# >>> print(y[:4])
# [1 3 2 4]
# >>> print(y[4:])
# []

def extract_padding_params_for_conv_layer(params):
    """
    Padding params in onnx are different than in pytorch. That is why we need to
    check if they are symmetric and cut half or return a padding layer.
    """
    # 参数是否堆成
    if is_symmetric(params):
        return params[: len(params) // 2]
    else:
        pads = extract_padding_params(params)[::-1]
        return pads


def get_selection(indices, dim):
    """
    Give selection to assign values to specific indices at given dimension.
    Enables dimension to be dynamic:
        tensor[get_selection(indices, dim=2)] = values
    Alternatively the dimension is fixed in code syntax:
        tensor[:, :, indices] = values
    """
    assert dim >= 0, "Negative dimension not supported."
    # Behaviour with python lists is unfortunately not working the same.
    if isinstance(indices, list):
        indices = torch.tensor(indices)
    assert isinstance(indices, (torch.Tensor, np.ndarray))
    selection = [slice(None) for _ in range(dim + 1)]
    selection[dim] = indices
    return selection


def assign_values_to_dim(tensor, values, indices, dim, inplace=True):
    """
    Inplace tensor operation that assigns values to corresponding indices
    at given dimension.
    """
    if dim < 0:
        dim = dim + len(tensor.shape)
    selection = get_selection(indices, dim)
    if not inplace:
        tensor = tensor.clone()
    tensor[selection] = values
    return tensor


def get_type(x):
    """
    Extract type from onnxruntime input.

    Parameters
    ----------
    x: onnxruntime.capi.onnxruntime_pybind11_state.NodeArg
    """
    if x.type.startswith("tensor"):
        typ = x.type[7:-1]
    else:
        raise NotImplementedError("For type: {}".format(x.type))

    if typ == "float":
        typ = "float32"
    elif typ == "double":
        typ = "float64"
    return typ


def get_shape(x, unknown_dim_size=1):
    """
    Extract shape from onnxruntime input.
    Replace unknown dimension by default with 1.

    Parameters
    ----------
    x: onnxruntime.capi.onnxruntime_pybind11_state.NodeArg
    unknown_dim_size: int
        Default: 1
    """
    shape = x.shape
    # replace unknown dimensions by default with 1
    shape = [i if isinstance(i, int) else unknown_dim_size for i in shape]
    return shape


def get_activation_value(onnx_model, inputs, activation_names):
    """
    Get activation value from an onnx model.

    Parameters
    ----------
    onnx_model: onnx.ModelProto
    inputs: list[np.ndarray]
    activation_names: list[str]
        Can be retrieved from onnx node: list(node.output)

    Returns
    -------
    value: list[np.ndarray]
        Value of the activation with activation_name.
    """
    assert ort is not None, "onnxruntime needed. pip install onnxruntime"
    assert all(isinstance(x, np.ndarray) for x in inputs)

    if not isinstance(activation_names, (list, tuple)):
        activation_names = [activation_names]

    # clear output
    while len(onnx_model.graph.output):
        onnx_model.graph.output.pop()

    for activation_name in activation_names:
        activation_value = onnx.helper.ValueInfoProto()
        activation_value.name = activation_name
        onnx_model.graph.output.append(activation_value)

    buffer = io.BytesIO()
    onnx.save(onnx_model, buffer)
    buffer.seek(0)
    onnx_model_new = onnx.load(buffer)
    sess = ort.InferenceSession(onnx_model_new.SerializeToString())

    input_names = [x.name for x in sess.get_inputs()]
    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = dict(zip(input_names, inputs))

    return sess.run(None, inputs)

# 获取网络所有的输入节点名字
def get_inputs_names(onnx_model):
    param_names = set([x.name for x in onnx_model.graph.initializer])
    input_names = [x.name for x in onnx_model.graph.input]
    input_names = [x for x in input_names if x not in param_names]
    return input_names


def get_inputs_sample(onnx_model, to_torch=False):
    """Get inputs sample from onnx model."""
    assert ort is not None, "onnxruntime needed. pip install onnxruntime"

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    inputs = sess.get_inputs()
    input_names = get_inputs_names(onnx_model)
    input_tensors = [
        np.abs(np.random.rand(*get_shape(x)).astype(get_type(x))) for x in inputs
    ]
    if to_torch:
        input_tensors = [torch.from_numpy(x) for x in input_tensors]
    return dict(zip(input_names, input_tensors))
