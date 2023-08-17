# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from facechain.inference import GenPortrait
import cv2

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
cloth_prompt = 'wearing high-class business/working suit'
style_model_path = None
multiplier_style = None
add_prompt_style = None

gen_portrait = GenPortrait(cloth_prompt, style_model_path, multiplier_style, add_prompt_style,
                           use_main_model, use_face_swap, use_post_process,
                           use_stylization)
outputs = gen_portrait(processed_dir, num_generate, base_model,
                       train_output_dir, base_model_sub_dir, revision)

os.makedirs(output_dir, exist_ok=True)

for i, out_tmp in enumerate(outputs):
    cv2.imwrite(os.path.join(output_dir, f'{i}.png'), out_tmp)
