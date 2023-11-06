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
# from controlnet_aux import OpenposeDetector
from dwpose import DWposeDetector
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


def segment(segmentation_pipeline, img, ksize=0, return_human=False, return_cloth=False, return_hand=False):
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
        mask_hands = np.zeros((h, w))
        for i in range(len(labels)):
            if scores[i] > 0.8:
                if labels[i] == 'Torso-skin':
                    mask_neck += masks[i]
                elif labels[i] == 'Face':
                    mask_face += masks[i]
                elif labels[i] == 'Human':
                    if np.sum(masks[i]) > np.sum(mask_human):
                        mask_human = masks[i]
                elif labels[i] == 'Hair':
                    mask_hair += masks[i]
                elif labels[i] == 'UpperClothes' or labels[i] == 'Coat' or labels[i] == 'Dress' or labels[i] == 'Pants' or labels[i] == 'Skirt':
                    mask_cloth += masks[i]
                elif labels[i] == 'Left-arm' or labels[i] == 'Right-arm':
                    mask_hands += masks[i]
        mask_face = np.clip(mask_face * mask_human, 0, 1)
        mask_hair = np.clip(mask_hair * mask_human, 0, 1)
        mask_neck = np.clip(mask_neck * mask_human, 0, 1)
        mask_cloth = np.clip(mask_cloth * mask_human, 0, 1)
        mask_human = np.clip(mask_human, 0, 1)
        mask_hands = np.clip(mask_hands * mask_human, 0, 1)

        if return_cloth:
            if ksize > 0:
                kernel = np.ones((ksize, ksize))
                soft_mask = cv2.erode(mask_cloth, kernel, iterations=1) 
                return soft_mask
            else:
                return mask_cloth
        if return_hand:
            return mask_hands
        if return_human:
            mask_head = np.clip(mask_face + mask_hair + mask_neck, 0, 1)
            kernel = np.ones((ksize, ksize))
            dilated_head = cv2.dilate(mask_head, kernel, iterations=1)
            mask_human = np.clip(mask_human - dilated_head + mask_cloth, 0, 1)
            return mask_human
        if np.sum(mask_face) > 0:
            soft_mask = np.clip(mask_face, 0, 1)
            if ksize > 0:
                # kernel_size = int(np.sqrt(np.sum(soft_mask)) * ksize)
                kernel = np.ones((ksize, ksize))
                soft_mask = cv2.dilate(soft_mask, kernel, iterations=1)               
        else:
            soft_mask = mask_face

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


def main_diffusion_inference_tryon(inpaint_image, strength, output_img_size, pos_prompt, neg_prompt,
                                     input_img_dir, base_model_path, style_model_path, lora_model_path,
                                     multiplier_style=0.05,
                                     multiplier_human=1.0):
    if style_model_path is None:
        model_dir = snapshot_download('Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
        style_model_path = os.path.join(model_dir, 'zjz_mj_jiyi_small_addtxt_fromleo.safetensors')

    segmentation_pipeline = pipeline(Tasks.image_segmentation, 'damo/cv_resnet101_image-multiple-human-parsing')
    det_pipeline = pipeline(Tasks.face_detection, 'damo/cv_ddsar_face-detection_iclr23-damofd')
    model_dir = snapshot_download('damo/face_chain_control_model', revision='v1.0.1')
    model_dir0 = snapshot_download('damo/face_chain_control_model', revision='v1.0.2')
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
    if attr_idx == 2 or attr_idx == 4:
        neg_prompt += ', children'
    
    neg_prompt += ', blurry, blurry background'

    for tag in tags_all:
        if tags_all.count(tag) > 0.5 * cnt:
            if ('glasses' in tag or 'smile' in tag or 'hair' in tag):
                if not tag in add_prompt_style:
                    add_prompt_style.append(tag)

    if len(add_prompt_style) > 0:
        add_prompt_style = ", ".join(add_prompt_style) + ', '
    else:
        add_prompt_style = ''
    
    print(add_prompt_style)

    if isinstance(inpaint_image, str): 
        inpaint_im = Image.open(inpaint_image)
    else:
        inpaint_im = inpaint_image
    inpaint_im = crop_bottom(inpaint_im, output_img_size)
    w, h = inpaint_im.size
    
    dwprocessor = DWposeDetector(os.path.join(model_dir0, 'dwpose_models'))
    openpose_image, handbox = dwprocessor(np.array(inpaint_im, np.uint8), include_body=True, include_hand=True, include_face=False, return_handbox=True)
    openpose_image = Image.fromarray(openpose_image)
    openpose_image.save('openpose.png')

    controlnet = [
        ControlNetModel.from_pretrained(os.path.join(model_dir, "model_controlnet/control_v11p_sd15_openpose"), torch_dtype=dtype),
        ControlNetModel.from_pretrained(os.path.join(model_dir, 'model_controlnet/control_v11p_sd15_depth'),
                                        torch_dtype=dtype),
        ControlNetModel.from_pretrained(os.path.join(model_dir1, "contronet-canny"), torch_dtype=dtype)
    ]
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(base_model_path, controlnet=controlnet,
                                                                    torch_dtype=dtype, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = merge_lora(pipe, style_model_path, multiplier_style, from_safetensor=True)
    pipe = merge_lora(pipe, lora_model_path, multiplier_human, from_safetensor=False)
    pipe = pipe.to("cuda")

    images_human = []
    mask = segment(segmentation_pipeline, inpaint_im, return_hand=True)
    mask1 = segment(segmentation_pipeline, inpaint_im, ksize=5, return_human=True)

    canny_image = cv2.Canny(np.array(inpaint_im, np.uint8), 80, 200)[:, :, None]
    canny_image = (canny_image * mask1[:, :, None]).astype(np.uint8)
    canny_image = Image.fromarray(np.concatenate([canny_image, canny_image, canny_image], axis=2))
    canny_image.save('canny.png')

    depth_estimator = tpipeline('depth-estimation', os.path.join(model_dir, 'model_controlnet/dpt-large'))
    depth_im = np.zeros((h, w))
    for hbox in handbox:
        depth_input = Image.fromarray(np.array(inpaint_im)[hbox[1]:hbox[3], hbox[0]:hbox[2]])
        depth_rst = depth_estimator(depth_input)['depth']
        depth_rst = np.array(depth_rst)
        depth_im[hbox[1]:hbox[3], hbox[0]:hbox[2]] = depth_rst
    depth_im = depth_im[:, :, None]
    depth_im = np.concatenate([depth_im, depth_im, depth_im], axis=2)
    depth_im = (depth_im * mask[:, :, None]).astype(np.uint8)
    depth_im = Image.fromarray(depth_im)
    depth_im.save('depth.png')

    for i in range(1):
        read_control = [openpose_image, depth_im, canny_image]
        cloth_mask_warp = segment(segmentation_pipeline, inpaint_im, return_cloth=True, ksize=5)
        cloth_mask = segment(segmentation_pipeline, inpaint_im, return_cloth=True, ksize=15)
        inpaint_with_mask = (cloth_mask_warp[:,:,None] * np.array(inpaint_im))[:,:,::-1]
        inpaint_mask = 1.0 - cloth_mask
        cv2.imwrite('inpaint_with_mask_{}.png'.format(i), inpaint_with_mask)
        print('Finishing segmenting images.')
        images_human.extend(img2img_multicontrol(inpaint_im, read_control, [1.0, 0.2, 0.4], pipe, inpaint_mask,
                                                 trigger_style + add_prompt_style + pos_prompt, neg_prompt,
                                                 strength=strength))

    for i in range(1):
        soft_cloth_mask_warp = cv2.GaussianBlur(cloth_mask_warp, (5, 5), 0, 0)
        image_human = (np.array(images_human[i]) * (1.0 - soft_cloth_mask_warp[:,:,None]) + np.array(inpaint_im) * soft_cloth_mask_warp[:,:,None]).astype(np.uint8)
        images_human[i] = Image.fromarray(image_human)
        images_human[i].save('inference_{}.png'.format(i))

    return images_human


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
        return main_diffusion_inference_tryon(inpaint_image, strength, output_img_size, pos_prompt, neg_prompt,
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
            # face_mask = segment(segmentation_pipeline, img, ksize=10)
            # result = (result * face_mask[:,:,None] + np.array(img)[:,:,::-1] * (1 - face_mask[:,:,None])).astype(np.uint8)
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


class GenPortrait_tryon:
    def __init__(self, inpaint_img, strength,
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

    def __call__(self, input_img_dir=None, base_model_path=None,
                 lora_model_path=None, sub_path=None, revision=None):
        base_model_path = snapshot_download(base_model_path, revision=revision)
        if sub_path is not None and len(sub_path) > 0:
            base_model_path = os.path.join(base_model_path, sub_path)

        gen_results = main_model_inference(self.inpaint_img, self.strength, 768,
                                           self.pos_prompt, self.neg_prompt, 
                                           self.style_model_path, self.multiplier_style, self.multiplier_human,
                                           self.use_main_model, input_img_dir=input_img_dir,
                                           lora_model_path=lora_model_path, base_model_path=base_model_path)
        # select_high_quality_face PIL
        selected_face = select_high_quality_face(input_img_dir)
        # face_swap cv2
        swap_results = face_swap_fn(self.use_face_swap, gen_results, selected_face)
        # pose_process
        final_gen_results_final = swap_results

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
