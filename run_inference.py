# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import json
from facechain.inference import GenPortrait
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
    folder_path = f"/mnt/workspace/new_facechain/facechain/styles/{base_model['name']}"
    files = os.listdir(folder_path)
    files.sort()
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            style_in_base.append(data['name'])
            styles.append(data)
    base_model['style_list'] = style_in_base

use_main_model = True
use_face_swap = True
use_post_process = True
use_stylization = False
use_depth_control = False
use_pose_model = False
pose_image = 'poses/man/pose1.png'
processed_dir = './processed'
num_generate = 5
multiplier_style = 0.25
multiplier_human = 0.85
train_output_dir = './output'
output_dir = './generated'
base_model = base_models[0]
style = styles[0]
model_id = style['model_id']
sr_img_size = 1
portrait_stylization_idx = None

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
    pose_model_path = None
    use_depth_control = False
    pose_image = None
else:
    model_dir = snapshot_download('damo/face_chain_control_model', revision='v1.0.1')
    pose_model_path = os.path.join(model_dir, 'model_controlnet/control_v11p_sd15_openpose')

gen_portrait = GenPortrait(pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt, style_model_path,
                           multiplier_style, multiplier_human, use_main_model,
                           use_face_swap, use_post_process,
                           use_stylization)

outputs = gen_portrait(processed_dir, num_generate, base_model['model_id'],
                       train_output_dir, base_model['sub_path'], base_model['revision'], sr_img_size=sr_img_size, portrait_stylization_idx=portrait_stylization_idx)

os.makedirs(output_dir, exist_ok=True)

for i, out_tmp in enumerate(outputs):
    cv2.imwrite(os.path.join(output_dir, f'{i}.png'), out_tmp)

