# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import os
import re
from collections import defaultdict
from safetensors.torch import load_file


def merge_lora(pipeline, lora_path, multiplier, from_safetensor=False, device='cpu', dtype=torch.float32):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    if from_safetensor:
        state_dict = load_file(lora_path, device=device)
    else:
        if os.path.exists(os.path.join(lora_path, 'pytorch_lora_weights.bin')):
            checkpoint = torch.load(os.path.join(lora_path, 'pytorch_lora_weights.bin'), map_location=torch.device(device))
        elif os.path.exists(os.path.join(lora_path, 'pytorch_lora_weights.safetensors')):
            checkpoint= load_file(os.path.join(lora_path,'pytorch_lora_weights.safetensors'), device=device)
        new_dict = dict()
        for idx, key in enumerate(checkpoint):
            new_key = re.sub(r'\.processor\.', '_', key)
            new_key = re.sub(r'mid_block\.', 'mid_block_', new_key)
            new_key = re.sub('_lora.up.', '.lora_up.', new_key)
            new_key = re.sub('_lora.down.', '.lora_down.', new_key)
            new_key = re.sub(r'\.(\d+)\.', '_\\1_', new_key)
            new_key = re.sub('to_out', 'to_out_0', new_key)
            new_key = 'lora_unet_' + new_key
            new_dict[new_key] = checkpoint[key]
            state_dict = new_dict
    updates = defaultdict(dict)
    for key, value in state_dict.items():
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(layer_infos) == 0:
                    print('Error loading layer')
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        curr_layer.weight.data = curr_layer.weight.data.to(device)
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2),
                                                                    weight_down.squeeze(3).squeeze(2)).unsqueeze(
                2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline
