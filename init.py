# Copyright (c) Alibaba, Inc. and its affiliates.
import enum
import os
import json
import shutil
import slugify
import time
import cv2
import gradio as gr
import numpy as np
import torch
from glob import glob
import platform
from facechain.inference_fact import GenPortrait
from facechain.inference_inpaint_fact import GenPortrait_inpaint
from facechain.utils import snapshot_download, check_ffmpeg, project_dir, join_worker_data_dir
from facechain.constants import neg_prompt as neg, pos_prompt_with_cloth, pos_prompt_with_style, \
    pose_examples, base_models, tts_speakers_map


inference_done_count = 0
character_model = 'ly261666/cv_portrait_model'
BASE_MODEL_MAP = {
    "leosamsMoonfilm_filmGrain20": "写实模型(Realistic sd_1.5 model)",
    "MajicmixRealistic_v6": "\N{fire}写真模型(Photorealistic sd_1.5 model)",
}

styles = []
style_list = []
base_models_reverse = [base_models[1], base_models[0]]
for base_model in base_models_reverse:
    folder_path = f"{os.path.dirname(os.path.abspath(__file__))}/styles/{base_model['name']}"
    files = os.listdir(folder_path)
    files.sort()
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            if data['img'][:2] == './':
                data['img'] = f"{project_dir}/{data['img'][2:]}"
                if base_model['name'] == 'leosamsMoonfilm_filmGrain20':
                    data['base_model_index'] = 0
                else:
                    data['base_model_index'] = 1
            style_list.append(data['name'])
            styles.append(data)

for style in styles:
    print(style['name'])
    if style['model_id'] is not None:
        model_dir = snapshot_download(style['model_id'], revision=style['revision'])

gen_portrait = GenPortrait()
gen_portrait_inpaint = GenPortrait_inpaint()
