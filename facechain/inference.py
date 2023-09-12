# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, \
    UniPCMultistepScheduler
from facechain.utils import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from torch import multiprocessing
from transformers import pipeline as tpipeline

from facechain.data_process.preprocessing import Blipv2
from facechain.merge_lora import merge_lora


def _data_process_fn_process(input_img_dir):
    Blipv2()(input_img_dir)


def data_process_fn(input_img_dir, use_data_process):
    ## TODO add face quality filter
    if use_data_process:
        ## TODO

        _process = multiprocessing.Process(target=_data_process_fn_process, args=(input_img_dir,))
        _process.start()
        _process.join()

    return os.path.join(str(input_img_dir) + '_labeled', "metadata.jsonl")

def txt2img(pipe, pos_prompt, neg_prompt, num_images=10):
    batch_size = 5
    images_out = []
    for i in range(int(num_images / batch_size)):
        images_style = pipe(prompt=pos_prompt, height=512, width=512, guidance_scale=7, negative_prompt=neg_prompt,
                            num_inference_steps=40, num_images_per_prompt=batch_size).images
        images_out.extend(images_style)
    return images_out

def img_pad(pil_file, fixed_height=512, fixed_width=512):
    w, h = pil_file.size

    if h / float(fixed_height) >= w / float(fixed_width):
        factor = h / float(fixed_height)
        new_w = int(w / factor)
        pil_file.thumbnail(size=(new_w, fixed_height))
        pad_w = int((fixed_width - new_w) / 2)
        pad_w1 = (fixed_width - new_w) - pad_w
        array_file = np.array(pil_file)
        array_file = np.pad(array_file, ((0, 0), (pad_w, pad_w1), (0, 0)), 'constant')
    else:
        factor = w / float(fixed_width)
        new_h = int(h / factor)
        pil_file.thumbnail(size=(fixed_width, new_h))
        pad_h = fixed_height - new_h
        pad_h1 = 0
        array_file = np.array(pil_file)
        array_file = np.pad(array_file, ((pad_h, pad_h1), (0, 0), (0, 0)), 'constant')

    output_file = Image.fromarray(array_file)
    return output_file

def preprocess_pose(origin_img) -> Image:
    img = Image.open(origin_img)
    img = img_pad(img)
    model_dir = snapshot_download('damo/face_chain_control_model',revision='v1.0.1')
    openpose = OpenposeDetector.from_pretrained(os.path.join(model_dir, 'model_controlnet/ControlNet'))
    result = openpose(img, include_hand=True, output_type='np')
    # resize to original size
    h, w = img.size
    result = cv2.resize(result, (w, h))
    return result

def txt2img_pose(pipe, pose_im, pos_prompt, neg_prompt, num_images=10):
    batch_size = 2
    images_out = []
    for i in range(int(num_images / batch_size)):
        images_style = pipe(prompt=pos_prompt, image=pose_im, height=512, width=512, guidance_scale=7, negative_prompt=neg_prompt,
                            num_inference_steps=40, num_images_per_prompt=batch_size).images
        images_out.extend(images_style)
    return images_out

def txt2img_multi(pipe, images, pos_prompt, neg_prompt, num_images=10):
    batch_size = 2
    images_out = []
    for i in range(int(num_images / batch_size)):
        images_style = pipe(pos_prompt, images, height=512, width=512, guidance_scale=7, negative_prompt=neg_prompt, controlnet_conditioning_scale=[1.0, 0.5],
                            num_inference_steps=40, num_images_per_prompt=batch_size).images
        images_out.extend(images_style)
    return images_out

def get_mask(result):
    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    h, w = masks[0].shape
    mask_hair = np.zeros((h, w))
    mask_face = np.zeros((h, w))
    mask_human = np.zeros((h, w))
    for i in range(len(labels)):
        if scores[i] > 0.8:
            if labels[i] == 'Face':
                if np.sum(masks[i]) > np.sum(mask_face):
                    mask_face = masks[i]
            elif labels[i] == 'Human':
                if np.sum(masks[i]) > np.sum(mask_human):
                    mask_human = masks[i]
            elif labels[i] == 'Hair':
                if np.sum(masks[i]) > np.sum(mask_hair):
                    mask_hair = masks[i]
    mask_rst = np.clip(mask_human - mask_hair - mask_face, 0, 1)
    mask_rst = np.expand_dims(mask_rst, 2)
    mask_rst = np.concatenate([mask_rst, mask_rst, mask_rst], axis=2)
    return mask_rst

def main_diffusion_inference(pos_prompt, neg_prompt,
                             input_img_dir, base_model_path, style_model_path, lora_model_path,
                             multiplier_style=0.25,
                             multiplier_human=0.85):
    if style_model_path is None:
        model_dir = snapshot_download('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
        style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_fromleo.safetensors')

    pipe = StableDiffusionPipeline.from_pretrained(base_model_path, safety_checker=None, torch_dtype=torch.float32)
    lora_style_path = style_model_path
    lora_human_path = lora_model_path
    pipe = merge_lora(pipe, lora_style_path, multiplier_style, from_safetensor=True)
    pipe = merge_lora(pipe, lora_human_path, multiplier_human, from_safetensor=lora_human_path.endswith('safetensors'))
    print(f'multiplier_style:{multiplier_style}, multiplier_human:{multiplier_human}')
    
    train_dir = str(input_img_dir) + '_labeled'
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
    trigger_style = '<fcsks>, ' + trigger_styles[attr_idx]
    if attr_idx == 2 or attr_idx == 4:
        neg_prompt += ', children'

    for tag in tags_all:
        if tags_all.count(tag) > 0.5 * cnt:
            if ('hair' in tag or 'face' in tag or 'mouth' in tag or 'skin' in tag or 'smile' in tag):
                if not tag in add_prompt_style:
                    add_prompt_style.append(tag)


    
    if len(add_prompt_style) > 0:
        add_prompt_style = ", ".join(add_prompt_style) + ', '
    else:
        add_prompt_style = ''

    pipe = pipe.to("cuda")
    images_style = txt2img(pipe, trigger_style + add_prompt_style + pos_prompt, neg_prompt, num_images=10)
    return images_style

def main_diffusion_inference_pose(pose_model_path, pose_image,
                                  pos_prompt, neg_prompt,
                                  input_img_dir, base_model_path, style_model_path, lora_model_path,
                                  multiplier_style=0.25,
                                  multiplier_human=0.85):
    if style_model_path is None:
        model_dir = snapshot_download('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
        style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_fromleo.safetensors')

    controlnet = ControlNetModel.from_pretrained(pose_model_path, torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, safety_checker=None, controlnet=controlnet, torch_dtype=torch.float32)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pose_im = Image.open(pose_image)
    pose_im = img_pad(pose_im)
    model_dir = snapshot_download('damo/face_chain_control_model',revision='v1.0.1')
    openpose = OpenposeDetector.from_pretrained(os.path.join(model_dir, 'model_controlnet/ControlNet'))
    pose_im = openpose(pose_im, include_hand=True)

    lora_style_path = style_model_path
    lora_human_path = lora_model_path
    pipe = merge_lora(pipe, lora_style_path, multiplier_style, from_safetensor=True)
    pipe = merge_lora(pipe, lora_human_path, multiplier_human, from_safetensor=False)
    print(f'multiplier_style:{multiplier_style}, multiplier_human:{multiplier_human}')
    
    train_dir = str(input_img_dir) + '_labeled'
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
    trigger_style = '<fcsks>, ' + trigger_styles[attr_idx]
    if attr_idx == 2 or attr_idx == 4:
        neg_prompt += ', children'

    for tag in tags_all:
        if tags_all.count(tag) > 0.5 * cnt:
            if ('hair' in tag or 'face' in tag or 'mouth' in tag or 'skin' in tag or 'smile' in tag):
                if not tag in add_prompt_style:
                    add_prompt_style.append(tag)

    if len(add_prompt_style) > 0:
        add_prompt_style = ", ".join(add_prompt_style) + ', '
    else:
        add_prompt_style = ''
    # trigger_style = trigger_style + 'with <input_id> face, '
    # pos_prompt = 'Generate a standard ID photo of a chinese {}, solo, wearing high-class business/working suit, beautiful smooth face, with high-class/simple pure color background, looking straight into the camera with shoulders parallel to the frame, smile, high detail face, best quality, photorealistic'.format(gender)
    pipe = pipe.to("cuda")
    # print(trigger_style + add_prompt_style + pos_prompt)
    images_style = txt2img_pose(pipe, pose_im, trigger_style + add_prompt_style + pos_prompt, neg_prompt, num_images=10)
    return images_style


def main_diffusion_inference_multi(pose_model_path, pose_image,
                                  pos_prompt, neg_prompt,
                                  input_img_dir, base_model_path, style_model_path, lora_model_path,
                                  multiplier_style=0.25,
                                  multiplier_human=0.85):
    if style_model_path is None:
        model_dir = snapshot_download('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
        style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_fromleo.safetensors')

    model_dir = snapshot_download('damo/face_chain_control_model', revision='v1.0.1')
    controlnet = [
        ControlNetModel.from_pretrained(pose_model_path, torch_dtype=torch.float32),
        ControlNetModel.from_pretrained(os.path.join(model_dir, 'model_controlnet/control_v11p_sd15_depth'), torch_dtype=torch.float32)
    ]
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, safety_checker=None, controlnet=controlnet, torch_dtype=torch.float32)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pose_image = Image.open(pose_image)
    pose_image = img_pad(pose_image)
    openpose = OpenposeDetector.from_pretrained(os.path.join(model_dir, 'model_controlnet/ControlNet'))
    pose_im = openpose(pose_image, include_hand=True)
    segmentation_pipeline = pipeline(Tasks.image_segmentation,
                                     'damo/cv_resnet101_image-multiple-human-parsing')
    result = segmentation_pipeline(pose_image)
    mask_rst = get_mask(result)
    pose_image = np.array(pose_image)
    pose_image = (pose_image * mask_rst).astype(np.uint8)
    pose_image = Image.fromarray(pose_image)
    depth_estimator = tpipeline('depth-estimation', os.path.join(model_dir, 'model_controlnet/dpt-large'))
    depth_im = depth_estimator(pose_image)['depth']
    depth_im = np.array(depth_im)
    depth_im = depth_im[:, :, None]
    depth_im = np.concatenate([depth_im, depth_im, depth_im], axis=2)
    depth_im = Image.fromarray(depth_im)
    control_im = [pose_im, depth_im]

    lora_style_path = style_model_path
    lora_human_path = lora_model_path
    pipe = merge_lora(pipe, lora_style_path, multiplier_style, from_safetensor=True)
    pipe = merge_lora(pipe, lora_human_path, multiplier_human, from_safetensor=False)
    print(f'multiplier_style:{multiplier_style}, multiplier_human:{multiplier_human}')
    
    train_dir = str(input_img_dir) + '_labeled'
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
    trigger_style = '<fcsks>, ' + trigger_styles[attr_idx]
    if attr_idx == 2 or attr_idx == 4:
        neg_prompt += ', children'

    for tag in tags_all:
        if tags_all.count(tag) > 0.5 * cnt:
            if ('hair' in tag or 'face' in tag or 'mouth' in tag or 'skin' in tag or 'smile' in tag):
                if not tag in add_prompt_style:
                    add_prompt_style.append(tag)

    if len(add_prompt_style) > 0:
        add_prompt_style = ", ".join(add_prompt_style) + ', '
    else:
        add_prompt_style = ''
    # trigger_style = trigger_style + 'with <input_id> face, '
    # pos_prompt = 'Generate a standard ID photo of a chinese {}, solo, wearing high-class business/working suit, beautiful smooth face, with high-class/simple pure color background, looking straight into the camera with shoulders parallel to the frame, smile, high detail face, best quality, photorealistic'.format(gender)
    pipe = pipe.to("cuda")
    # print(trigger_style + add_prompt_style + pos_prompt)
    images_style = txt2img_multi(pipe, control_im, trigger_style + add_prompt_style + pos_prompt, neg_prompt, num_images=10)
    return images_style

def stylization_fn(use_stylization, rank_results):
    if use_stylization:
        ## TODO
        pass
    else:
        return rank_results


def main_model_inference(pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt, style_model_path, multiplier_style, multiplier_human, use_main_model,
                         input_img_dir=None, base_model_path=None, lora_model_path=None):
    if use_main_model:
        multiplier_style_kwargs = {'multiplier_style': multiplier_style} if multiplier_style is not None else {}
        multiplier_human_kwargs = {'multiplier_human': multiplier_human} if multiplier_human is not None else {}
        if pose_image is None:
            return main_diffusion_inference(pos_prompt, neg_prompt, input_img_dir, base_model_path,
                                            style_model_path, lora_model_path,
                                            **multiplier_style_kwargs, **multiplier_human_kwargs)
        else:
            pose_image = compress_image(pose_image, 1024 * 1024)
            if use_depth_control:
                return main_diffusion_inference_multi(pose_model_path, pose_image, pos_prompt,
                                                      neg_prompt, input_img_dir, base_model_path, style_model_path,
                                                      lora_model_path,
                                                      **multiplier_style_kwargs, **multiplier_human_kwargs)
            else:
                return main_diffusion_inference_pose(pose_model_path, pose_image, pos_prompt, neg_prompt,
                                                     input_img_dir, base_model_path, style_model_path, lora_model_path,
                                                     **multiplier_style_kwargs, **multiplier_human_kwargs)


def select_high_quality_face(input_img_dir):
    input_img_dir = str(input_img_dir) + '_labeled'
    quality_score_list = []
    abs_img_path_list = []
    ## TODO
    face_quality_func = pipeline(Tasks.face_quality_assessment, 'damo/cv_manual_face-quality-assessment_fqa', model_revision='v2.0')

    for img_name in os.listdir(input_img_dir):
        if img_name.endswith('jsonl') or img_name.startswith('.ipynb') or img_name.startswith('.safetensors'):
            continue
        
        if img_name.endswith('jpg') or img_name.endswith('png'):
            abs_img_name = os.path.join(input_img_dir, img_name)
            face_quality_score = face_quality_func(abs_img_name)[OutputKeys.SCORES]
            if face_quality_score is None:
                quality_score_list.append(0)
            else:
                quality_score_list.append(face_quality_score[0])
            abs_img_path_list.append(abs_img_name)

    sort_idx = np.argsort(quality_score_list)[::-1]
    print('Selected face: ' + abs_img_path_list[sort_idx[0]])

    return Image.open(abs_img_path_list[sort_idx[0]])


def face_swap_fn(use_face_swap, gen_results, template_face):
    if use_face_swap:
        ## TODO
        out_img_list = []
        image_face_fusion = pipeline('face_fusion_torch',
                                     model='damo/cv_unet_face_fusion_torch', model_revision='v1.0.3')
        for img in gen_results:
            result = image_face_fusion(dict(template=img, user=template_face))[OutputKeys.OUTPUT_IMG]
            out_img_list.append(result)

        return out_img_list
    else:
        ret_results = []
        for img in gen_results:
            ret_results.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        return ret_results


def post_process_fn(use_post_process, swap_results_ori, selected_face, num_gen_images):
    if use_post_process:
        sim_list = []
        ## TODO
        face_recognition_func = pipeline(Tasks.face_recognition, 'damo/cv_ir_face-recognition-ood_rts', model_revision='v2.5')
        face_det_func = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd', model_revision='v1.1')
        swap_results = []
        for img in swap_results_ori:
            result_det = face_det_func(img)
            bboxes = result_det['boxes']
            if len(bboxes) == 1:
                bbox = bboxes[0]
                lenface = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                if 120 < lenface < 300:
                    swap_results.append(img)

        select_face_emb = face_recognition_func(selected_face)[OutputKeys.IMG_EMBEDDING][0]

        for img in swap_results:
            emb = face_recognition_func(img)[OutputKeys.IMG_EMBEDDING]
            if emb is None or select_face_emb is None:
                sim_list.append(0)
            else:
                sim = np.dot(emb, select_face_emb)
                sim_list.append(sim.item())
        sort_idx = np.argsort(sim_list)[::-1]

        return np.array(swap_results)[sort_idx[:min(int(num_gen_images), len(swap_results))]]
    else:
        return np.array(swap_results_ori)


class GenPortrait:
    def __init__(self, pose_model_path, pose_image, use_depth_control, pos_prompt, neg_prompt, style_model_path, multiplier_style, multiplier_human,
                 use_main_model=True, use_face_swap=True,
                 use_post_process=True, use_stylization=True):
        self.use_main_model = use_main_model
        self.use_face_swap = use_face_swap
        self.use_post_process = use_post_process
        self.use_stylization = use_stylization
        self.multiplier_style = multiplier_style
        self.multiplier_human = multiplier_human
        self.style_model_path = style_model_path
        self.pos_prompt = pos_prompt
        self.neg_prompt = neg_prompt
        self.pose_model_path = pose_model_path
        self.pose_image = pose_image
        self.use_depth_control = use_depth_control

    def __call__(self, input_img_dir, num_gen_images=6, base_model_path=None,
                 lora_model_path=None, sub_path=None, revision=None):
        base_model_path = snapshot_download(base_model_path, revision=revision)
        if sub_path is not None and len(sub_path) > 0:
            base_model_path = os.path.join(base_model_path, sub_path)

        # main_model_inference PIL
        gen_results = main_model_inference(self.pose_model_path, self.pose_image, self.use_depth_control,
                                           self.pos_prompt, self.neg_prompt,
                                           self.style_model_path, self.multiplier_style, self.multiplier_human,
                                           self.use_main_model, input_img_dir=input_img_dir,
                                           lora_model_path=lora_model_path, base_model_path=base_model_path)

        # select_high_quality_face PIL
        selected_face = select_high_quality_face(input_img_dir)
        # face_swap cv2
        swap_results = face_swap_fn(self.use_face_swap, gen_results, selected_face)
        # pose_process
        rank_results = post_process_fn(self.use_post_process, swap_results, selected_face,
                                       num_gen_images=num_gen_images)
        # stylization
        final_gen_results = stylization_fn(self.use_stylization, rank_results)


        return final_gen_results

def compress_image(input_path, target_size):
    output_path = change_extension_to_jpg(input_path)

    image = cv2.imread(input_path)

    quality = 95
    try:
        while cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].size > target_size:
            quality -= 5
    except:
        import pdb;pdb.set_trace()

    compressed_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tostring()

    with open(output_path, 'wb') as f:
        f.write(compressed_image)
    return output_path


def change_extension_to_jpg(image_path):

    base_name = os.path.basename(image_path)
    new_base_name = os.path.splitext(base_name)[0] + ".jpg"

    directory = os.path.dirname(image_path)

    new_image_path = os.path.join(directory, new_base_name)
    return new_image_path
