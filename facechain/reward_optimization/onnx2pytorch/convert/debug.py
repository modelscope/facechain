import torch
import numpy as np

from facechain.reward_optimization.onnx2pytorch.utils import get_activation_value


def debug_model_conversion(onnx_model, inputs, pred_act, node, rtol=1e-3, atol=1e-4):
    """Compare if the activations of pytorch are the same as from onnxruntime."""
    if not isinstance(inputs, list):
        raise TypeError("inputs should be in a list.")

    if not all(isinstance(x, np.ndarray) for x in inputs):
        inputs = [x.detach().numpy() for x in inputs]

    exp_act = get_activation_value(onnx_model, inputs, list(node.output))
    if isinstance(pred_act, list):
        for a, b in zip(exp_act, pred_act):
            assert torch.allclose(torch.from_numpy(a), b, rtol=rtol, atol=atol)
    else:
        a = torch.from_numpy(exp_act[0])
        b = pred_act
        if torch.allclose(a, b, rtol=rtol, atol=atol) == False:
            print(node.input[0])
            print(node.op_type)
        assert torch.allclose(a, b, rtol=rtol, atol=atol)
