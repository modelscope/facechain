# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from facechain.inference import GenPortrait
import cv2
from modelscope import snapshot_download
from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, styles, cloth_prompt


def generate_pos_prompt(style_model, prompt_cloth):
    if style_model == styles[0]['name'] or style_model is None:
        pos_prompt = pos_prompt_with_cloth.format(prompt_cloth)
    else:
        matched = list(filter(lambda style: style_model == style['name'], styles))
        if len(matched) == 0:
            raise ValueError(f'styles not found: {style_model}')
        matched = matched[0]
        pos_prompt = pos_prompt_with_style.format(matched['add_prompt_style'])
    return pos_prompt


use_main_model = True
use_face_swap = True
use_post_process = True
use_stylization = False
processed_dir = './processed'
num_generate = 5
base_model = 'ly261666/cv_portrait_model'
revision = 'v2.0'
multiplier_style = 0.25
base_model_sub_dir = 'film/film'
train_output_dir = './output'
output_dir = './generated'
use_style = False

if not use_style:
    style_model_path = None
    pos_prompt = generate_pos_prompt(styles[0]['name'], cloth_prompt[0]['prompt'])
else:
    model_dir = snapshot_download(styles[1]['model_id'], revision=styles[1]['revision'])
    style_model_path = os.path.join(model_dir, styles[1]['bin_file'])
    pos_prompt = generate_pos_prompt(styles[1]['name'], styles[1]['add_prompt_style'])  # style has its own prompt

gen_portrait = GenPortrait(pos_prompt, neg_prompt, style_model_path, multiplier_style, use_main_model,
                           use_face_swap, use_post_process,
                           use_stylization)

outputs = gen_portrait(processed_dir, num_generate, base_model,
                       train_output_dir, base_model_sub_dir, revision)

os.makedirs(output_dir, exist_ok=True)

for i, out_tmp in enumerate(outputs):
    cv2.imwrite(os.path.join(output_dir, f'{i}.png'), out_tmp)

