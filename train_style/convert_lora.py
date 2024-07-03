from safetensors.torch import load_file, save_file
import re

def convert_lora(src, dst):
    checkpoint = load_file(src, device='cpu')
    new_dict = dict()
    for idx, key in enumerate(checkpoint):
        new_key = re.sub(r'\.processor\.', '_', key)
        new_key = re.sub(r'mid_block\.', 'mid_block_', new_key)
        new_key = re.sub('.lora.up.', '.lora_up.', new_key)
        new_key = re.sub('.lora.down.', '.lora_down.', new_key)
        new_key = re.sub(r'\.(\d+)\.', '_\\1_', new_key)
        new_key = re.sub('.to_out.0_', '_to_out_0.', new_key)
        new_key = re.sub('.to_q', '_to_q', new_key)
        new_key = re.sub('.to_k', '_to_k', new_key)
        new_key = re.sub('.to_v', '_to_v', new_key)
        new_key = 'lora_unet_' + new_key[5:]
        new_dict[new_key] = checkpoint[key]
    save_file(new_dict, dst)

