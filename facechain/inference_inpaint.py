# Copyright (c) Alibaba, Inc. and its affiliates.
# Modified from the original implementation at https://github.com/modelscope/facechain/pull/104.
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import transform
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, \
    StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
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


def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0], x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image


def data_process_fn(input_img_dir, use_data_process):
    ## TODO add face quality filter
    if use_data_process:
        ## TODO

        _process = multiprocessing.Process(target=_data_process_fn_process, args=(input_img_dir,))
        _process.start()
        _process.join()

    return os.path.join(str(input_img_dir) + '_labeled', "metadata.jsonl")


def call_face_crop(det_pipeline, image, crop_ratio):
    det_result = det_pipeline(image)
    bboxes = det_result['boxes']
    keypoints = det_result['keypoints']
    area = 0
    idx = 0
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        area_tmp = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area_tmp > area:
            area = area_tmp
            idx = i
    bbox = bboxes[idx]
    keypoint = keypoints[idx]
    points_array = np.zeros((5, 2))
    for k in range(5):
        points_array[k, 0] = keypoint[2 * k]
        points_array[k, 1] = keypoint[2 * k + 1]
    w, h = image.size
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    bbox[0] = np.clip(np.array(bbox[0], np.int32) - face_w * (crop_ratio - 1) / 2, 0, w - 1)
    bbox[1] = np.clip(np.array(bbox[1], np.int32) - face_h * (crop_ratio - 1) / 2, 0, h - 1)
    bbox[2] = np.clip(np.array(bbox[2], np.int32) + face_w * (crop_ratio - 1) / 2, 0, w - 1)
    bbox[3] = np.clip(np.array(bbox[3], np.int32) + face_h * (crop_ratio - 1) / 2, 0, h - 1)
    bbox = np.array(bbox, np.int32)
    return bbox, points_array


def crop_and_paste(Source_image, Source_image_mask, Target_image, Source_Five_Point, Target_Five_Point, Source_box, use_warp=True):
    if use_warp:
        Source_Five_Point = np.reshape(Source_Five_Point, [5, 2]) - np.array(Source_box[:2])
        Target_Five_Point = np.reshape(Target_Five_Point, [5, 2])

        Crop_Source_image = Source_image.crop(np.int32(Source_box))
        Crop_Source_image_mask = Source_image_mask.crop(np.int32(Source_box))
        Source_Five_Point, Target_Five_Point = np.array(Source_Five_Point), np.array(Target_Five_Point)

        tform = transform.SimilarityTransform()
        tform.estimate(Source_Five_Point, Target_Five_Point)
        M = tform.params[0:2, :]

        warped = cv2.warpAffine(np.array(Crop_Source_image), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)
        warped_mask = cv2.warpAffine(np.array(Crop_Source_image_mask), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)

        mask = np.float32(warped_mask == 0)
        output = mask * np.float32(Target_image) + (1 - mask) * np.float32(warped)
    else:
        mask = np.float32(np.array(Source_image_mask) == 0)
        output = mask * np.float32(Target_image) + (1 - mask) * np.float32(Source_image)
    return output, mask


def segment(segmentation_pipeline, img, ksize=0, eyeh=0, ksize1=0, include_neck=False, warp_mask=None, return_human=False):
    if True:
        result = segmentation_pipeline(img)
        masks = result['masks']
        scores = result['scores']
        labels = result['labels']
        if len(masks) == 0:
            return
        h, w = masks[0].shape
        mask_face = np.zeros((h, w))
        mask_hair = np.zeros((h, w))
        mask_neck = np.zeros((h, w))
        mask_cloth = np.zeros((h, w))
        mask_human = np.zeros((h, w))
        for i in range(len(labels)):
            if scores[i] > 0.8:
                if labels[i] == 'Torso-skin':
                    mask_neck += masks[i]
                elif labels[i] == 'Face':
                    mask_face += masks[i]
                elif labels[i] == 'Human':
                    mask_human += masks[i]
                elif labels[i] == 'Hair':
                    mask_hair += masks[i]
                elif labels[i] == 'UpperClothes' or labels[i] == 'Coat':
                    mask_cloth += masks[i]
        mask_face = np.clip(mask_face, 0, 1)
        mask_hair = np.clip(mask_hair, 0, 1)
        mask_neck = np.clip(mask_neck, 0, 1)
        mask_cloth = np.clip(mask_cloth, 0, 1)
        mask_human = np.clip(mask_human, 0, 1)
        if np.sum(mask_face) > 0:
            soft_mask = np.clip(mask_face, 0, 1)
            if ksize1 > 0:
                kernel_size1 = int(np.sqrt(np.sum(soft_mask)) * ksize1)
                kernel1 = np.ones((kernel_size1, kernel_size1))
                soft_mask = cv2.dilate(soft_mask, kernel1, iterations=1)
            if ksize > 0:
                kernel_size = int(np.sqrt(np.sum(soft_mask)) * ksize)
                kernel = np.ones((kernel_size, kernel_size))
                soft_mask_dilate = cv2.dilate(soft_mask, kernel, iterations=1)
                if warp_mask is not None:
                    soft_mask_dilate = soft_mask_dilate * (np.clip(soft_mask + warp_mask[:, :, 0], 0, 1))
                if eyeh > 0:
                    soft_mask = np.concatenate((soft_mask[:eyeh], soft_mask_dilate[eyeh:]), axis=0)
                else:
                    soft_mask = soft_mask_dilate
        else:
            if ksize1 > 0:
                kernel_size1 = int(np.sqrt(np.sum(soft_mask)) * ksize1)
                kernel1 = np.ones((kernel_size1, kernel_size1))
                soft_mask = cv2.dilate(mask_face, kernel1, iterations=1)
            else:
                soft_mask = mask_face
        if include_neck:
            soft_mask = np.clip(soft_mask + mask_neck, 0, 1)

    if return_human:  
        mask_human = cv2.GaussianBlur(mask_human, (21, 21), 0) * mask_human
        return soft_mask, mask_human
    else:
        return soft_mask


def crop_bottom(pil_file, width):
    if width == 512:
        height = 768
    else:
        height = 1152
    w, h = pil_file.size
    factor = w / width
    new_h = int(h / factor)
    pil_file = pil_file.resize((width, new_h))
    crop_h = min(int(new_h / 32) * 32, height)
    array_file = np.array(pil_file)
    array_file = array_file[:crop_h, :, :]
    output_file = Image.fromarray(array_file)
    return output_file


def img2img_multicontrol(img, control_image, controlnet_conditioning_scale, pipe, mask, pos_prompt, neg_prompt,
                         strength, num=1, use_ori=False):
    image_mask = Image.fromarray(np.uint8(mask * 255))
    image_human = []
    for i in range(num):
        image_human.append(pipe(image=img, mask_image=image_mask, control_image=control_image, prompt=pos_prompt,
                                negative_prompt=neg_prompt, guidance_scale=7, strength=strength, num_inference_steps=40,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                num_images_per_prompt=1).images[0])
        if use_ori:
            image_human[i] = Image.fromarray((np.array(image_human[i]) * mask[:,:,None] + np.array(img) * (1 - mask[:,:,None])).astype(np.uint8))
    return image_human


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


def main_diffusion_inference_inpaint(inpaint_image, strength, output_img_size, pos_prompt, neg_prompt,
                                     input_img_dir, base_model_path, style_model_path, lora_model_path,
                                     multiplier_style=0.05,
                                     multiplier_human=1.0):
    if style_model_path is None:
        model_dir = snapshot_download('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
        style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_fromleo.safetensors')

    segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')
    det_pipeline = pipeline(Tasks.face_detection, 'damo/cv_ddsar_face-detection_iclr23-damofd')
    model_dir = snapshot_download('damo/face_chain_control_model', revision='v1.0.1')
    model_dir1 = snapshot_download('ly261666/cv_wanx_style_model',revision='v1.0.3')

    if output_img_size == 512:
        dtype = torch.float32
    else:
        dtype = torch.float16

    train_dir = str(input_img_dir) + '_labeled'
    add_prompt_style = []
    f = open(os.path.join(train_dir, 'metadata.jsonl'), 'r')
    tags_all = []
    cnt = 0
    cnts_trigger = np.zeros(6)
    is_old = False
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
            is_old = True
        elif data[1] == 'a mature woman':
            cnts_trigger[5] += 1
            is_old = True
        else:
            print('Error.')
    f.close()

    attr_idx = np.argmax(cnts_trigger)
    trigger_styles = ['a boy, children, ', 'a girl, children, ', 'a handsome man, ', 'a beautiful woman, ',
                      'a mature man, ', 'a mature woman, ']
    trigger_style = '(<fcsks>:10), ' + trigger_styles[attr_idx]
    if attr_idx == 2 or attr_idx == 4:
        neg_prompt += ', children'

    for tag in tags_all:
        if tags_all.count(tag) > 0.5 * cnt:
            if ('glasses' in tag or 'smile' in tag):
                if not tag in add_prompt_style:
                    add_prompt_style.append(tag)

    if len(add_prompt_style) > 0:
        add_prompt_style = ", ".join(add_prompt_style) + ', '
    else:
        add_prompt_style = ''

    if isinstance(inpaint_image, str): 
        inpaint_im = Image.open(inpaint_image)
    else:
        inpaint_im = inpaint_image
    inpaint_im = crop_bottom(inpaint_im, output_img_size)
    # return [inpaint_im, inpaint_im, inpaint_im]
    openpose = OpenposeDetector.from_pretrained(os.path.join(model_dir, "model_controlnet/ControlNet"))
    controlnet = ControlNetModel.from_pretrained(os.path.join(model_dir, "model_controlnet/control_v11p_sd15_openpose"), torch_dtype=dtype)
    openpose_image = openpose(np.array(inpaint_im, np.uint8), include_hand=True, include_face=False)
    w, h = inpaint_im.size

    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=dtype)
    lora_style_path = style_model_path
    lora_human_path = lora_model_path
    pipe = merge_lora(pipe, lora_style_path, multiplier_style, from_safetensor=True)
    pipe = merge_lora(pipe, lora_human_path, multiplier_human, from_safetensor=False)
    pipe = pipe.to("cuda")
    image_faces = []
    for i in range(1):
        image_face = pipe(prompt=trigger_style + add_prompt_style + pos_prompt, image=openpose_image, height=h, width=w,
                          guidance_scale=7, negative_prompt=neg_prompt, 
                          num_inference_steps=40, num_images_per_prompt=1).images[0]
        image_faces.append(image_face)
    selected_face = select_high_quality_face(input_img_dir)
    swap_results = face_swap_fn(True, image_faces, selected_face)

    controlnet = [
        ControlNetModel.from_pretrained(os.path.join(model_dir, "model_controlnet/control_v11p_sd15_openpose"), torch_dtype=dtype),
        ControlNetModel.from_pretrained(os.path.join(model_dir1, "contronet-canny"), torch_dtype=dtype)
    ]
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(base_model_path, controlnet=controlnet,
                                                                    torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = merge_lora(pipe, style_model_path, multiplier_style, from_safetensor=True)
    pipe = merge_lora(pipe, lora_model_path, multiplier_human, from_safetensor=False)
    pipe = pipe.to("cuda")

    images_human = []
    images_auto = []
    inpaint_bbox, inpaint_keypoints = call_face_crop(det_pipeline, inpaint_im, 1.1)
    eye_height = int((inpaint_keypoints[0, 1] + inpaint_keypoints[1, 1]) / 2)
    canny_image = cv2.Canny(np.array(inpaint_im, np.uint8), 100, 200)[:, :, None]
    mask = segment(segmentation_pipeline, inpaint_im, ksize=0.05, eyeh=eye_height)
    canny_image = (canny_image * (1.0 - mask[:, :, None])).astype(np.uint8)
    canny_image = Image.fromarray(np.concatenate([canny_image, canny_image, canny_image], axis=2))
    # canny_image.save('canny.png')
    for i in range(1):
        image_face = swap_results[i]
        image_face = Image.fromarray(image_face[:, :, ::-1])

        face_bbox, face_keypoints = call_face_crop(det_pipeline, image_face, 1.5)
        face_mask = segment(segmentation_pipeline, image_face)
        face_mask = np.expand_dims((face_mask * 255).astype(np.uint8), axis=2)
        face_mask = np.concatenate([face_mask, face_mask, face_mask], axis=2)
        face_mask = Image.fromarray(face_mask)
        replaced_input_image, warp_mask = crop_and_paste(image_face, face_mask, inpaint_im, face_keypoints,
                                                         inpaint_keypoints, face_bbox)
        warp_mask = 1.0 - warp_mask
        # cv2.imwrite('tmp_{}.png'.format(i), replaced_input_image[:, :, ::-1])

        openpose_image = openpose(np.array(replaced_input_image * warp_mask, np.uint8), include_hand=True,
                                  include_body=False, include_face=True)
        # openpose_image.save('openpose_{}.png'.format(i))
        read_control = [openpose_image, canny_image]
        inpaint_mask, human_mask = segment(segmentation_pipeline, inpaint_im, ksize=0.1, ksize1=0.06, eyeh=eye_height, include_neck=False,
                               warp_mask=warp_mask, return_human=True)
        inpaint_with_mask = ((1.0 - inpaint_mask[:,:,None]) * np.array(inpaint_im))[:,:,::-1]
        # cv2.imwrite('inpaint_with_mask_{}.png'.format(i), inpaint_with_mask)
        print('Finishing segmenting images.')
        images_human.extend(img2img_multicontrol(inpaint_im, read_control, [1.0, 0.2], pipe, inpaint_mask,
                                                 trigger_style + add_prompt_style + pos_prompt, neg_prompt,
                                                 strength=strength))
        images_auto.extend(img2img_multicontrol(inpaint_im, read_control, [1.0, 0.2], pipe, np.zeros_like(inpaint_mask),
                                                 trigger_style + add_prompt_style + pos_prompt, neg_prompt,
                                                 strength=0.025))
    
        edge_add = np.array(inpaint_im).astype(np.int16) - np.array(images_auto[i]).astype(np.int16)
        edge_add = edge_add * (1 - human_mask[:,:,None])
        images_human[i] = Image.fromarray((np.clip(np.array(images_human[i]).astype(np.int16) + edge_add.astype(np.int16), 0, 255)).astype(np.uint8))
    
    images_rst = []
    
    for i in range(len(images_human)):
        im = images_human[i]
        canny_image = cv2.Canny(np.array(im, np.uint8), 100, 200)[:, :, None]
        canny_image = Image.fromarray(np.concatenate([canny_image, canny_image, canny_image], axis=2))
        openpose_image = openpose(np.array(im, np.uint8), include_hand=True, include_face=True)
        read_control = [openpose_image, canny_image]
        inpaint_mask, human_mask = segment(segmentation_pipeline, images_human[i], ksize=0.02, return_human=True)
        print('Finishing segmenting images.')
        image_rst = img2img_multicontrol(im, read_control, [0.8, 0.8], pipe, inpaint_mask,
                                         trigger_style + add_prompt_style + pos_prompt, neg_prompt, strength=0.1,
                                         num=1)[0]
        image_auto = img2img_multicontrol(im, read_control, [0.8, 0.8], pipe, np.zeros_like(inpaint_mask),
                                         trigger_style + add_prompt_style + pos_prompt, neg_prompt, strength=0.025,
                                         num=1)[0]
        edge_add = np.array(im).astype(np.int16) - np.array(image_auto).astype(np.int16)
        edge_add = edge_add * (1 - human_mask[:,:,None])
        image_rst = Image.fromarray((np.clip(np.array(image_rst).astype(np.int16) + edge_add.astype(np.int16), 0, 255)).astype(np.uint8))
        
        images_rst.append(image_rst)

    for i in range(1):
        images_rst[i].save('inference_{}.png'.format(i))

    return images_rst

def main_diffusion_inference_inpaint_multi(inpaint_images, strength, output_img_size, pos_prompt, neg_prompt,
                                     input_img_dir, base_model_path, style_model_path, lora_model_path,
                                     multiplier_style=0.05,
                                     multiplier_human=1.0):
    if style_model_path is None:
        model_dir = snapshot_download('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
        style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_fromleo.safetensors')

    segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')
    det_pipeline = pipeline(Tasks.face_detection, 'damo/cv_ddsar_face-detection_iclr23-damofd')
    model_dir = snapshot_download('damo/face_chain_control_model', revision='v1.0.1')
    model_dir1 = snapshot_download('ly261666/cv_wanx_style_model',revision='v1.0.3')

    if output_img_size == 512:
        dtype = torch.float32
    else:
        dtype = torch.float16

    train_dir = str(input_img_dir) + '_labeled'
    add_prompt_style = []
    f = open(os.path.join(train_dir, 'metadata.jsonl'), 'r')
    tags_all = []
    cnt = 0
    cnts_trigger = np.zeros(6)
    is_old = False
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
            is_old = True
        elif data[1] == 'a mature woman':
            cnts_trigger[5] += 1
            is_old = True
        else:
            print('Error.')
    f.close()

    attr_idx = np.argmax(cnts_trigger)
    trigger_styles = ['a boy, children, ', 'a girl, children, ', 'a handsome man, ', 'a beautiful woman, ',
                      'a mature man, ', 'a mature woman, ']
    trigger_style = '(<fcsks>:10), ' + trigger_styles[attr_idx]
    if attr_idx == 2 or attr_idx == 4:
        neg_prompt += ', children'

    for tag in tags_all:
        if tags_all.count(tag) > 0.5 * cnt:
            if ('glasses' in tag or 'smile' in tag):
                if not tag in add_prompt_style:
                    add_prompt_style.append(tag)

    if len(add_prompt_style) > 0:
        add_prompt_style = ", ".join(add_prompt_style) + ', '
    else:
        add_prompt_style = ''
        
    openpose = OpenposeDetector.from_pretrained(os.path.join(model_dir, "model_controlnet/ControlNet"))
    controlnet = ControlNetModel.from_pretrained(os.path.join(model_dir, "model_controlnet/control_v11p_sd15_openpose"), torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=dtype)
    lora_style_path = style_model_path
    lora_human_path = lora_model_path
    pipe = merge_lora(pipe, lora_style_path, multiplier_style, from_safetensor=True)
    pipe = merge_lora(pipe, lora_human_path, multiplier_human, from_safetensor=False)
    pipe = pipe.to("cuda")
    image_faces = []

    for i in range(1):
        inpaint_im = inpaint_images[i]
        inpaint_im = crop_bottom(inpaint_im, output_img_size)

        openpose_image = openpose(np.array(inpaint_im, np.uint8), include_hand=True, include_face=False)
        w, h = inpaint_im.size
        image_face = pipe(prompt=trigger_style + add_prompt_style + pos_prompt, image=openpose_image, height=h, width=w,
                          guidance_scale=7, negative_prompt=neg_prompt, 
                          num_inference_steps=40, num_images_per_prompt=1).images[0]
        image_faces.append(image_face)
        
    selected_face = select_high_quality_face(input_img_dir)
    swap_results = face_swap_fn(True, image_faces, selected_face)

    controlnet = [
        ControlNetModel.from_pretrained(os.path.join(model_dir, "model_controlnet/control_v11p_sd15_openpose"), torch_dtype=dtype),
        ControlNetModel.from_pretrained(os.path.join(model_dir1, "contronet-canny"), torch_dtype=dtype)
    ]
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(base_model_path, controlnet=controlnet,
                                                                    torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = merge_lora(pipe, style_model_path, multiplier_style, from_safetensor=True)
    pipe = merge_lora(pipe, lora_model_path, multiplier_human, from_safetensor=False)
    pipe = pipe.to("cuda")

    images_human = []  
    images_auto = []
    for i in range(1):
        inpaint_im = inpaint_images[i]
        inpaint_bbox, inpaint_keypoints = call_face_crop(det_pipeline, inpaint_im, 1.1)
        eye_height = int((inpaint_keypoints[0, 1] + inpaint_keypoints[1, 1]) / 2)
        canny_image = cv2.Canny(np.array(inpaint_im, np.uint8), 100, 200)[:, :, None]
        mask = segment(segmentation_pipeline, inpaint_im, ksize=0.05, eyeh=eye_height)
        canny_image = (canny_image * (1.0 - mask[:, :, None])).astype(np.uint8)
        canny_image = Image.fromarray(np.concatenate([canny_image, canny_image, canny_image], axis=2))
        
        image_face = swap_results[i]
        image_face = Image.fromarray(image_face[:, :, ::-1])

        face_bbox, face_keypoints = call_face_crop(det_pipeline, image_face, 1.5)
        face_mask = segment(segmentation_pipeline, image_face)
        face_mask = np.expand_dims((face_mask * 255).astype(np.uint8), axis=2)
        face_mask = np.concatenate([face_mask, face_mask, face_mask], axis=2)
        face_mask = Image.fromarray(face_mask)
        replaced_input_image, warp_mask = crop_and_paste(image_face, face_mask, inpaint_im, face_keypoints,
                                                         inpaint_keypoints, face_bbox)
        warp_mask = 1.0 - warp_mask
        # cv2.imwrite('tmp_{}.png'.format(i), replaced_input_image[:, :, ::-1])

        openpose_image = openpose(np.array(replaced_input_image * warp_mask, np.uint8), include_hand=True,
                                  include_body=False, include_face=True)
        # openpose_image.save('openpose_{}.png'.format(i))
        read_control = [openpose_image, canny_image]
        inpaint_mask, human_mask = segment(segmentation_pipeline, inpaint_im, ksize=0.1, ksize1=0.06, eyeh=eye_height, include_neck=False,
                               warp_mask=warp_mask, return_human=True)
        inpaint_with_mask = ((1.0 - inpaint_mask[:,:,None]) * np.array(inpaint_im))[:,:,::-1]
        # cv2.imwrite('inpaint_with_mask_{}.png'.format(i), inpaint_with_mask)
        print('Finishing segmenting images.')
        images_human.extend(img2img_multicontrol(inpaint_im, read_control, [1.0, 0.2], pipe, inpaint_mask,
                                                 trigger_style + add_prompt_style + pos_prompt, neg_prompt,
                                                 strength=strength))
        images_auto.extend(img2img_multicontrol(inpaint_im, read_control, [1.0, 0.2], pipe, np.zeros_like(inpaint_mask),
                                                 trigger_style + add_prompt_style + pos_prompt, neg_prompt,
                                                 strength=0.025))
    
        edge_add = np.array(inpaint_im).astype(np.int16) - np.array(images_auto[i]).astype(np.int16)
        edge_add = edge_add * (1 - human_mask[:,:,None])
        images_human[i] = Image.fromarray((np.clip(np.array(images_human[i]).astype(np.int16) + edge_add.astype(np.int16), 0, 255)).astype(np.uint8))
        
    
    images_rst = []
    for i in range(len(images_human)):
        im = images_human[i]
        canny_image = cv2.Canny(np.array(im, np.uint8), 100, 200)[:, :, None]
        canny_image = Image.fromarray(np.concatenate([canny_image, canny_image, canny_image], axis=2))
        openpose_image = openpose(np.array(im, np.uint8), include_hand=True, include_face=True)
        read_control = [openpose_image, canny_image]
        inpaint_mask, human_mask = segment(segmentation_pipeline, images_human[i], ksize=0.02, return_human=True)
        print('Finishing segmenting images.')
        image_rst = img2img_multicontrol(im, read_control, [0.8, 0.8], pipe, np.zeros_like(inpaint_mask),
                                         trigger_style + add_prompt_style + pos_prompt, neg_prompt, strength=0.1,
                                         num=1)[0]
        image_auto = img2img_multicontrol(im, read_control, [0.8, 0.8], pipe, np.zeros_like(inpaint_mask),
                                         trigger_style + add_prompt_style + pos_prompt, neg_prompt, strength=0.025,
                                         num=1)[0]
        edge_add = np.array(im).astype(np.int16) - np.array(image_auto).astype(np.int16)
        edge_add = edge_add * (1 - human_mask[:,:,None])
        image_rst = Image.fromarray((np.clip(np.array(image_rst).astype(np.int16) + edge_add.astype(np.int16), 0, 255)).astype(np.uint8))
        
        images_rst.append(image_rst)

    for i in range(1):
        images_rst[i].save('inference_{}.png'.format(i))

    return images_rst

def stylization_fn(use_stylization, rank_results):
    if use_stylization:
        ## TODO
        pass
    else:
        return rank_results


def main_model_inference(inpaint_image, strength, output_img_size,
                         pos_prompt, neg_prompt, style_model_path, multiplier_style, multiplier_human, use_main_model,
                         input_img_dir=None, base_model_path=None, lora_model_path=None):

    if use_main_model:
        multiplier_style_kwargs = {'multiplier_style': multiplier_style} if multiplier_style is not None else {}
        multiplier_human_kwargs = {'multiplier_human': multiplier_human} if multiplier_human is not None else {}
        return main_diffusion_inference_inpaint(inpaint_image, strength, output_img_size, pos_prompt, neg_prompt,
                                                input_img_dir, base_model_path, style_model_path, lora_model_path,
                                                **multiplier_style_kwargs, **multiplier_human_kwargs)

def main_model_inference_multi(inpaint_image, strength, output_img_size,
                         pos_prompt, neg_prompt, style_model_path, multiplier_style, multiplier_human, use_main_model,
                         input_img_dir=None, base_model_path=None, lora_model_path=None):

    if use_main_model:
        multiplier_style_kwargs = {'multiplier_style': multiplier_style} if multiplier_style is not None else {}
        multiplier_human_kwargs = {'multiplier_human': multiplier_human} if multiplier_human is not None else {}
        return main_diffusion_inference_inpaint_multi(inpaint_image, strength, output_img_size, pos_prompt, neg_prompt,
                                                input_img_dir, base_model_path, style_model_path, lora_model_path,
                                                **multiplier_style_kwargs, **multiplier_human_kwargs)


def select_high_quality_face(input_img_dir):
    input_img_dir = str(input_img_dir) + '_labeled'
    quality_score_list = []
    abs_img_path_list = []
    ## TODO
    face_quality_func = pipeline(Tasks.face_quality_assessment, 'damo/cv_manual_face-quality-assessment_fqa',
                                 model_revision='v2.0')

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
                                     model='damo/cv_unet_face_fusion_torch', model_revision='v1.0.5')
        segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')
        for img in gen_results:
            result = image_face_fusion(dict(template=img, user=template_face))[OutputKeys.OUTPUT_IMG]
            face_mask = segment(segmentation_pipeline, img, ksize=0.1)
            result = (result * face_mask[:,:,None] + np.array(img)[:,:,::-1] * (1 - face_mask[:,:,None])).astype(np.uint8)
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
        face_recognition_func = pipeline(Tasks.face_recognition, 'damo/cv_ir_face-recognition-ood_rts',
                                         model_revision='v2.5')
        face_det_func = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd',
                                 model_revision='v1.1')
        swap_results = swap_results_ori

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


class GenPortrait_inpaint:
    def __init__(self, inpaint_img, strength, num_faces,
                 pos_prompt, neg_prompt, style_model_path, multiplier_style, multiplier_human,
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
        self.inpaint_img = inpaint_img
        self.strength = strength
        self.num_faces = num_faces

    def __call__(self, input_img_dir1=None, input_img_dir2=None, base_model_path=None,
                 lora_model_path1=None, lora_model_path2=None, sub_path=None, revision=None):
        base_model_path = snapshot_download(base_model_path, revision=revision)
        if sub_path is not None and len(sub_path) > 0:
            base_model_path = os.path.join(base_model_path, sub_path)

        face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd',
                                  model_revision='v1.1')
        result_det = face_detection(self.inpaint_img)
        bboxes = result_det['boxes']
        assert(len(bboxes)) == self.num_faces
        bboxes = np.array(bboxes).astype(np.int16)
        lefts = []
        for bbox in bboxes:
            lefts.append(bbox[0])
        idxs = np.argsort(lefts)
        
        if lora_model_path1 != None:
            face_box = bboxes[idxs[0]]
            inpaint_img_large = cv2.imread(self.inpaint_img)
            mask_large = np.ones_like(inpaint_img_large)
            mask_large1 = np.zeros_like(inpaint_img_large)
            h,w,_ = inpaint_img_large.shape
            for i in range(len(bboxes)):
                if i != idxs[0]:
                    bbox = bboxes[i]
                    inpaint_img_large[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0
                    mask_large[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

            face_ratio = 0.45
            cropl = int(max(face_box[3] - face_box[1], face_box[2] - face_box[0]) / face_ratio / 2)
            cx = int((face_box[2] + face_box[0])/2)
            cy = int((face_box[1] + face_box[3])/2)
            cropup = min(cy, cropl)
            cropbo = min(h-cy, cropl)
            crople = min(cx, cropl)
            cropri = min(w-cx, cropl)
            inpaint_img = np.pad(inpaint_img_large[cy-cropup:cy+cropbo, cx-crople:cx+cropri], ((cropl-cropup, cropl-cropbo), (cropl-crople, cropl-cropri), (0, 0)), 'constant')
            inpaint_img = cv2.resize(inpaint_img, (512, 512)) 
            inpaint_img = Image.fromarray(inpaint_img[:,:,::-1])
            mask_large1[cy-cropup:cy+cropbo, cx-crople:cx+cropri] = 1
            mask_large = mask_large * mask_large1

            gen_results = main_model_inference(inpaint_img, self.strength, 512,
                                               self.pos_prompt, self.neg_prompt,
                                               self.style_model_path, self.multiplier_style, self.multiplier_human,
                                               self.use_main_model, input_img_dir=input_img_dir1,
                                               lora_model_path=lora_model_path1, base_model_path=base_model_path)

            # select_high_quality_face PIL
            selected_face = select_high_quality_face(input_img_dir1)
            # face_swap cv2
            swap_results = face_swap_fn(self.use_face_swap, gen_results, selected_face)
            # stylization
            final_gen_results = swap_results

            print(len(final_gen_results))

            final_gen_results_new = []
            inpaint_img_large = cv2.imread(self.inpaint_img)
            ksize = int(10 * cropl / 256)
            for i in range(len(final_gen_results)):
                print('Start cropping.')
                rst_gen = cv2.resize(final_gen_results[i], (cropl * 2, cropl * 2))
                rst_crop = rst_gen[cropl-cropup:cropl+cropbo, cropl-crople:cropl+cropri]
                print(rst_crop.shape)
                inpaint_img_rst = np.zeros_like(inpaint_img_large)
                print('Start pasting.')
                inpaint_img_rst[cy-cropup:cy+cropbo, cx-crople:cx+cropri] = rst_crop
                print('Fininsh pasting.')
                print(inpaint_img_rst.shape, mask_large.shape, inpaint_img_large.shape)
                mask_large = mask_large.astype(np.float32)
                kernel = np.ones((ksize * 2, ksize * 2))
                mask_large1 = cv2.erode(mask_large, kernel, iterations=1)
                mask_large1 = cv2.GaussianBlur(mask_large1, (int(ksize * 1.8) * 2 + 1, int(ksize * 1.8) * 2 + 1), 0)
                mask_large1[face_box[1]:face_box[3], face_box[0]:face_box[2]] = 1
                mask_large = mask_large * mask_large1
                final_inpaint_rst = (inpaint_img_rst.astype(np.float32) * mask_large.astype(np.float32) + inpaint_img_large.astype(np.float32) * (1.0 - mask_large.astype(np.float32))).astype(np.uint8)
                print('Finish masking.')
                final_gen_results_new.append(final_inpaint_rst)
                print('Finish generating.')
        else:
            inpaint_img_large = cv2.imread(self.inpaint_img)
            inpaint_img_le = cv2.imread(self.inpaint_img)
            final_gen_results_new = [inpaint_img_le, inpaint_img_le, inpaint_img_le]
        
        for i in range(1):
            cv2.imwrite('tmp_inpaint_left_{}.png'.format(i), final_gen_results_new[i])
        
        if lora_model_path2 != None and self.num_faces == 2:
            face_box = bboxes[idxs[1]]
            mask_large = np.ones_like(inpaint_img_large)
            mask_large1 = np.zeros_like(inpaint_img_large)
            h,w,_ = inpaint_img_large.shape
            for i in range(len(bboxes)):
                if i != idxs[1]:
                    bbox = bboxes[i]
                    inpaint_img_large[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0
                    mask_large[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

            face_ratio = 0.45
            cropl = int(max(face_box[3] - face_box[1], face_box[2] - face_box[0]) / face_ratio / 2)
            cx = int((face_box[2] + face_box[0])/2)
            cy = int((face_box[1] + face_box[3])/2)
            cropup = min(cy, cropl)
            cropbo = min(h-cy, cropl)
            crople = min(cx, cropl)
            cropri = min(w-cx, cropl)
            mask_large1[cy-cropup:cy+cropbo, cx-crople:cx+cropri] = 1
            mask_large = mask_large * mask_large1
            
            inpaint_imgs = []
            for i in range(1):
                inpaint_img_large = final_gen_results_new[i] * mask_large
                inpaint_img = np.pad(inpaint_img_large[cy-cropup:cy+cropbo, cx-crople:cx+cropri], ((cropl-cropup, cropl-cropbo), (cropl-crople, cropl-cropri), (0, 0)), 'constant')
                inpaint_img = cv2.resize(inpaint_img, (512, 512)) 
                inpaint_img = Image.fromarray(inpaint_img[:,:,::-1])
                inpaint_imgs.append(inpaint_img)
            

            gen_results = main_model_inference_multi(inpaint_imgs, self.strength, 512,
                                               self.pos_prompt, self.neg_prompt,
                                               self.style_model_path, self.multiplier_style, self.multiplier_human,
                                               self.use_main_model, input_img_dir=input_img_dir2,
                                               lora_model_path=lora_model_path2, base_model_path=base_model_path)

            # select_high_quality_face PIL
            selected_face = select_high_quality_face(input_img_dir2)
            # face_swap cv2
            swap_results = face_swap_fn(self.use_face_swap, gen_results, selected_face)
            # stylization
            final_gen_results = swap_results

            print(len(final_gen_results))

            final_gen_results_final = []
            inpaint_img_large = cv2.imread(self.inpaint_img)
            ksize = int(10 * cropl / 256)
            for i in range(len(final_gen_results)):
                print('Start cropping.')
                rst_gen = cv2.resize(final_gen_results[i], (cropl * 2, cropl * 2))
                rst_crop = rst_gen[cropl-cropup:cropl+cropbo, cropl-crople:cropl+cropri]
                print(rst_crop.shape)
                inpaint_img_rst = np.zeros_like(inpaint_img_large)
                print('Start pasting.')
                inpaint_img_rst[cy-cropup:cy+cropbo, cx-crople:cx+cropri] = rst_crop
                print('Fininsh pasting.')
                print(inpaint_img_rst.shape, mask_large.shape, inpaint_img_large.shape)
                mask_large = mask_large.astype(np.float32)
                kernel = np.ones((ksize * 2, ksize * 2))
                mask_large1 = cv2.erode(mask_large, kernel, iterations=1)
                mask_large1 = cv2.GaussianBlur(mask_large1, (int(ksize * 1.8) * 2 + 1, int(ksize * 1.8) * 2 + 1), 0)
                mask_large1[face_box[1]:face_box[3], face_box[0]:face_box[2]] = 1
                mask_large = mask_large * mask_large1
                final_inpaint_rst = (inpaint_img_rst.astype(np.float32) * mask_large.astype(np.float32) + final_gen_results_new[i].astype(np.float32) * (1.0 - mask_large.astype(np.float32))).astype(np.uint8)
                print('Finish masking.')
                final_gen_results_final.append(final_inpaint_rst)
                print('Finish generating.')
        else:
            final_gen_results_final = final_gen_results_new

        outputs = final_gen_results_final
        outputs_RGB = []
        for out_tmp in outputs:
            outputs_RGB.append(cv2.cvtColor(out_tmp, cv2.COLOR_BGR2RGB))
        image_path = './lora_result.png'
        if len(outputs) > 0:
            result = concatenate_images(outputs)
            cv2.imwrite(image_path, result)

        return final_gen_results_final

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
