# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from facechain.inference import GenPortrait
import cv2
from facechain.utils import snapshot_download
from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, styles, base_models
import argparse


def generate_pos_prompt(style_model, prompt_cloth):
    if style_model in base_models[0]['style_list'][:-1] or style_model is None:
        pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
    else:
        matched = list(filter(lambda style: style_model == style['name'], styles))
        if len(matched) == 0:
            raise ValueError(f'styles not found: {style_model}')
        matched = matched[0]
        pos_prompt = pos_prompt_with_style.format(matched['add_prompt_style'])
    return pos_prompt


# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='run_inference for different base model')

use_main_model = True
use_face_swap = True
use_post_process = True
use_stylization = False
use_depth_control = False
use_pose_model = False
pose_image = 'poses/man/pose1.png'
processed_dir = './processed'
num_generate = 5
base_model = 'ly261666/cv_portrait_model'
revision = 'v2.0'
multiplier_style = 0.25
multiplier_human = 0.85
base_model_sub_dir = 'film/film'
train_output_dir = './output'
output_dir = './generated'
input_dir = './imgs'
style = styles[0]
model_id = style['model_id']


parser.add_argument('--base_model', type=str, default=base_model, help='base_model value')
parser.add_argument('--revision', type=str, default=revision, help='revision value')
parser.add_argument('--base_model_sub_dir', type=str, default=base_model_sub_dir, help='base_model_sub_dir value')
parser.add_argument('--input_dir', type=str, default=input_dir, help='input_dir value')
parser.add_argument('--processed_dir', type=str, default=processed_dir, help='processed_dir value')
parser.add_argument('--output_dir', type=str, default=output_dir, help='output_dir value')
# 解析命令行参数
args = parser.parse_args()
base_model = args.base_model
revision = args.revision
base_model_sub_dir = args.base_model_sub_dir
input_dir = args.input_dir
processed_dir = args.processed_dir
output_dir = args.output_dir


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

outputs = gen_portrait(processed_dir, num_generate, base_model,
                       train_output_dir, base_model_sub_dir, revision)

os.makedirs(output_dir, exist_ok=True)

for i, out_tmp in enumerate(outputs):
    cv2.imwrite(os.path.join(output_dir, f'{i}.png'), out_tmp)
