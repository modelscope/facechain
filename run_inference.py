# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import json
from facechain.inference_fact import GenPortrait
import cv2
from facechain.utils import snapshot_download
from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, base_models


def generate_pos_prompt(style_model, prompt_cloth):
    if style_model is not None:
        matched = list(filter(lambda style: style_model == style['name'], styles))
        if len(matched) == 0:
            raise ValueError(f'styles not found: {style_model}')
        matched = matched[0]
        if matched['model_id'] is None:
            pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
        else:
            pos_prompt = pos_prompt_with_style.format(matched['add_prompt_style'])
    else:
        pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
    return pos_prompt

styles = []
for base_model in base_models:
    style_in_base = []
    folder_path = f"styles/{base_model['name']}"
    files = os.listdir(folder_path)
    files.sort()
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            style_in_base.append(data['name'])
            styles.append(data)
    base_model['style_list'] = style_in_base

use_pose_model = False
input_img_path = 'poses/man/pose2.png'
pose_image = 'poses/man/pose1.png'
num_generate = 5
multiplier_style = 0.25
output_dir = './generated'
base_model_idx = 0
style_idx = 0

base_model = base_models[base_model_idx]
style = styles[style_idx]
model_id = style['model_id']

if model_id == None:
    style_model_path = None
    pos_prompt = generate_pos_prompt(style['name'], style['add_prompt_style'])
else:
    if os.path.exists(model_id):
        model_dir = model_id
    else:
        model_dir = snapshot_download(model_id, revision=style['revision'])
    style_model_path = os.path.join(model_dir, style['bin_file'])
    pos_prompt = generate_pos_prompt(style['name'], style['add_prompt_style'])  # style has its own prompt

if not use_pose_model:
    pose_image = None

gen_portrait = GenPortrait()

outputs = gen_portrait(num_generate, base_model_idx, style_model_path, pos_prompt, neg_prompt, input_img_path, pose_image, multiplier_style)

os.makedirs(output_dir, exist_ok=True)

for i, out_tmp in enumerate(outputs):
    cv2.imwrite(os.path.join(output_dir, f'{i}.png'), out_tmp)

