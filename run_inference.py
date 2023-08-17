# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from facechain.inference import GenPortrait
import cv2
from modelscope import snapshot_download


examples = {
    'prompt_male': [
        ['wearing silver armor'],
        ['wearing T-shirt']
    ],
    'prompt_female': [
        ['wearing beautiful traditional hanfu, upper_body'],
        ['wearing an elegant evening gown']
    ],
}

example_styles = [
    {'name': '默认风格(default_style_model_path)'},
    {'name': '凤冠霞帔(Chinese traditional gorgeous suit)',
     'model_id': 'ly261666/civitai_xiapei_lora',
     'revision': 'v1.0.0',
     'bin_file': 'xiapei.safetensors',
     'multiplier_style': 0.35,
     'add_prompt_style': 'red, hanfu, tiara, crown, '},
]

use_main_model = True
use_face_swap = True
use_post_process = True
use_stylization = False
processed_dir = './processed'
num_generate = 5
base_model = 'ly261666/cv_portrait_model'
revision = 'v2.0'
base_model_sub_dir = 'film/film'
train_output_dir = './output'
output_dir = './generated'
use_cloth_prompt = True  # Cloth prompt and style model cannot be used at the same time.

if use_cloth_prompt:
    cloth_prompt = 'wearing high-class business/working suit'  # examples['prompt_male'][0]
    style_model_path = None
    multiplier_style = None
    add_prompt_style = None
else:
    model_dir = snapshot_download(example_styles[1]['model_id'], revision=example_styles[1]['revision'])
    cloth_prompt = None
    style_model_path = os.path.join(model_dir, example_styles[1]['bin_file'])
    multiplier_style = example_styles[1]['multiplier_style']
    add_prompt_style = example_styles[1]['add_prompt_style']

gen_portrait = GenPortrait(cloth_prompt, style_model_path, multiplier_style, add_prompt_style,
                           use_main_model, use_face_swap, use_post_process,
                           use_stylization)
outputs = gen_portrait(processed_dir, num_generate, base_model,
                       train_output_dir, base_model_sub_dir, revision)

os.makedirs(output_dir, exist_ok=True)

for i, out_tmp in enumerate(outputs):
    cv2.imwrite(os.path.join(output_dir, f'{i}.png'), out_tmp)
