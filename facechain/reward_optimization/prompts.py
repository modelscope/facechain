"""This file defines customize prompt funtions. The prompt function which takes no arguments
generates a random prompt from the given prompt distribution each time it is called.
"""
import random
from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, base_models
import os
import json
import cv2
from facechain.utils import snapshot_download, project_dir
import numpy as np

def generate_pos_prompt(style_model, prompt_cloth, styles):
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

def facechain(style_model=None, processed_dir='') -> str:
    # TODO:
    # this code snippet need to be updated when the corresponding code snippet in app.py is updated
    styles = []
    for base_model in base_models:
        style_in_base = []
        folder_path = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}/styles/{base_model['name']}"
        files = os.listdir(folder_path)
        files.sort()
        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                if data['img'][:2] == './':
                    data['img'] = f"{project_dir}/{data['img'][2:]}"
                style_in_base.append(data['name'])
                styles.append(data)
    try:
        if style_model is None:
            style = random.choice(styles)
        else:
            for i_style in styles:
                if i_style['name'] == style_model:
                    style = i_style
       
    except:
        raise ValueError(f"Style '{style_model}' does not exist.")
    
    assert style != None, f"Style '{style_model}' does not exist."

    model_id = style['model_id']
    if model_id == None:
        style_model_path = None
        pos_prompt = generate_pos_prompt(style['name'], style['add_prompt_style'], styles)
    else:
        if os.path.exists(model_id):
            model_dir = model_id
        else:
            model_dir = snapshot_download(model_id, revision=style['revision'])
        style_model_path = os.path.join(model_dir, style['bin_file'])
        pos_prompt = generate_pos_prompt(style['name'], style['add_prompt_style'], styles)  # style has its own prompt

    if style_model_path is None:
        model_dir = snapshot_download('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
        style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_fromleo.safetensors')

    if processed_dir == '':
        raise ValueError('processed_dir is empty.')
    train_dir = str(processed_dir)
    add_prompt_style = []
    f = open(os.path.join(train_dir, 'metadata.jsonl'), 'r')
    tags_all = []
    cnt = 0
    cnts_trigger = np.zeros(6)
    for line in f:
        cnt += 1
        data = json.loads(line)['text'].split(', ')
        tags_all.extend(data)
        if data[1] == 'a boy':
            cnts_trigger[0] += 1
        elif data[1] == 'a girl':
            cnts_trigger[1] += 1
        elif data[1] == 'a handsome man':
            cnts_trigger[2] += 1
        elif data[1] == 'a beautiful woman':
            cnts_trigger[3] += 1
        elif data[1] == 'a mature man':
            cnts_trigger[4] += 1
        elif data[1] == 'a mature woman':
            cnts_trigger[5] += 1
        else:
            print('Error.')
    f.close()

    attr_idx = np.argmax(cnts_trigger)
    trigger_styles = ['a boy, children, ', 'a girl, children, ', 'a handsome man, ', 'a beautiful woman, ',
                      'a mature man, ', 'a mature woman, ']
    trigger_style = '(<fcsks>:10), ' + trigger_styles[attr_idx]
    # if attr_idx == 2 or attr_idx == 4:
    #     neg_prompt += ', children'

    for tag in tags_all:
        if tags_all.count(tag) > 0.5 * cnt:
            if ('hair' in tag or 'face' in tag or 'mouth' in tag or 'skin' in tag or 'smile' in tag):
                if not tag in add_prompt_style:
                    add_prompt_style.append(tag)


    
    if len(add_prompt_style) > 0:
        add_prompt_style = ", ".join(add_prompt_style) + ', '
    else:
        add_prompt_style = ''

    final_pos_prompt = trigger_style + add_prompt_style + pos_prompt

    return final_pos_prompt, style_model_path