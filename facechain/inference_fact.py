# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
import sys

import cv2
import json
import numpy as np
import torch
from controlnet_aux import OpenposeDetector
from diffusers import (ControlNetModel, PNDMScheduler, DDIMScheduler, AutoencoderKL, StableDiffusionControlNetPipeline, StableDiffusionPipeline)
from facechain.merge_lora import merge_lora, restore_lora

from PIL import Image
from torch import multiprocessing
from transformers import pipeline as tpipeline

from modelscope import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from face_adapter import FaceAdapter_v1, Face_Extracter_v1


def txt2img(pipe, face_image, pos_prompt, neg_prompt, num_images=10):
    batch_size = 1
    images_out = []
    for i in range(int(num_images / batch_size)):
        images_style = pipe.generate(
            prompt=pos_prompt,
            face_image=face_image,
            height=512,
            width=512,
            guidance_scale=5,
            negative_prompt=neg_prompt,
            num_inference_steps=50,
            num_images_per_prompt=batch_size)
        images_out.extend(images_style)
    return images_out


def img_pad(pil_file, fixed_height=512, fixed_width=512):
    w, h = pil_file.size

    if h / float(fixed_height) >= w / float(fixed_width):
        factor = h / float(fixed_height)
        new_w = int(w / factor)
        pil_file = pil_file.resize((new_w, fixed_height))
        pad_w = int((fixed_width - new_w) / 2)
        pad_w1 = (fixed_width - new_w) - pad_w
        array_file = np.array(pil_file)
        array_file = np.pad(array_file, ((0, 0), (pad_w, pad_w1), (0, 0)),
                            'constant')
    else:
        factor = w / float(fixed_width)
        new_h = int(h / factor)
        pil_file = pil_file.resize((fixed_width, new_h))
        pad_h = fixed_height - new_h
        pad_h1 = 0
        array_file = np.array(pil_file)
        array_file = np.pad(array_file, ((pad_h, pad_h1), (0, 0), (0, 0)),
                            'constant')

    output_file = Image.fromarray(array_file)
    print(output_file.size)
    return output_file


def txt2img_multi(pipe,
                  face_image,
                  images,
                  pos_prompt,
                  neg_prompt,
                  num_images=10,
                  pose_control_weight=1.0,
                  depth_control_weight=0.5,
                  cfg_scale=7):
    batch_size = 1
    images_out = []
    for i in range(int(num_images / batch_size)):
        images_style = pipe.generate(
            prompt=pos_prompt,
            face_image=face_image,
            image=images,
            height=512,
            width=512,
            guidance_scale=cfg_scale,
            negative_prompt=neg_prompt,
            controlnet_conditioning_scale=pose_control_weight,
            num_inference_steps=50,
            num_images_per_prompt=batch_size)
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
    mask_rm = np.clip(mask_face, 0, 1)
    kernel = np.ones((7, 7))
    mask_rm = cv2.dilate(mask_rm, kernel, iterations=1)
    mask_rst = np.clip(mask_human - mask_rm, 0, 1)
    mask_rst = np.expand_dims(mask_rst, 2)
    mask_rst = np.concatenate([mask_rst, mask_rst, mask_rst], axis=2)
    return mask_rst


def main_diffusion_inference_multi(num_gen_images,
                                   pose_control_weight,
                                   depth_control_weight,
                                   pose_image,
                                   use_face_pose,
                                   pos_prompt,
                                   neg_prompt,
                                   input_img,
                                   segmentation_pipeline=None,
                                   image_face_fusion=None,
                                   openpose=None,
                                   controlnet=None,
                                   depth_estimator=None,
                                   pipe=None,
                                   cfg_scale=7):

    add_prompt_style = ''

    pose_image = Image.open(pose_image)
    pose_image = img_pad(pose_image)
    if use_face_pose:
        pose_im = openpose(pose_image, include_hand=True)
        pose_im = pose_im.resize((512, 512))
    else:
        pose_im = openpose(pose_image, include_hand=True)
        pose_im = pose_im.resize((512, 512))
        result = segmentation_pipeline(pose_im)
        mask_rst = get_mask(result)
        pose_im = np.array(pose_im)
        pose_im = (pose_im * mask_rst).astype(np.uint8)
        pose_im = Image.fromarray(pose_im)
        pose_im.save('pose.png')
    
    control_im = pose_im

    images_style = txt2img_multi(
            pipe,
            input_img,
            control_im,
            add_prompt_style + pos_prompt,
            neg_prompt,
            num_images=num_gen_images,
            pose_control_weight=pose_control_weight,
            depth_control_weight=depth_control_weight,
            cfg_scale=cfg_scale)

    return images_style, False


def stylization_fn(use_stylization, rank_results):
    if use_stylization:
        #  TODO
        pass
    else:
        return rank_results


def main_model_inference(num_gen_images,
                         pose_control_weight,
                         depth_control_weight,
                         pose_image,
                         use_depth_control,
                         use_face_pose,
                         pos_prompt,
                         neg_prompt,
                         use_main_model,
                         input_img=None,
                         segmentation_pipeline=None,
                         image_face_fusion=None,
                         openpose=None,
                         controlnet=None,
                         depth_estimator=None,
                         pipe=None,
                         cfg_scale=7):
    if use_main_model:
        if pose_image is None:
            pass
        else:
            #pose_image = compress_image(pose_image, 1024 * 1024)
            if use_depth_control:
                print('pose_control_weight', pose_control_weight)
                print('depth_control_weight', depth_control_weight)
                print('pos_prompt', pos_prompt)
                print('use_face_pose', use_face_pose)
                return main_diffusion_inference_multi(
                    num_gen_images,
                    pose_control_weight, depth_control_weight, pose_image,
                    use_face_pose, pos_prompt, neg_prompt, input_img,
                    segmentation_pipeline, image_face_fusion, openpose,
                    controlnet, depth_estimator, pipe, cfg_scale)
            else:
                pass


def face_swap_fn(use_face_swap, gen_results, template_face, image_face_fusion):
    if use_face_swap:
        #  TODO
        out_img_list = []
        for img in gen_results:
            result = image_face_fusion(dict(
                template=img, user=template_face))[OutputKeys.OUTPUT_IMG]
            # result = image_face_fusion.inference(np.array(img), np.array(template_face))
            out_img_list.append(result)

        return out_img_list
    else:
        ret_results = []
        for img in gen_results:
            ret_results.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        return ret_results


def post_process_fn(use_post_process, swap_results_ori, selected_face,
                    num_gen_images):
    if use_post_process:
        sim_list = []
        #  TODO
        face_recognition_func = pipeline(
            Tasks.face_recognition,
            'damo/cv_ir_face-recognition-ood_rts',
            model_revision='v2.5')
        face_det_func = pipeline(
            task=Tasks.face_detection,
            model='damo/cv_ddsar_face-detection_iclr23-damofd',
            model_revision='v1.1')
        swap_results = []
        for img in swap_results_ori:
            result_det = face_det_func(img)
            bboxes = result_det['boxes']
            if len(bboxes) == 1:
                bbox = bboxes[0]
                lenface = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
                if lenface > 100:
                    swap_results.append(img)

        select_face_emb = face_recognition_func(selected_face)[
            OutputKeys.IMG_EMBEDDING][0]

        for img in swap_results:
            emb = face_recognition_func(img)[OutputKeys.IMG_EMBEDDING]
            if emb is None or select_face_emb is None:
                sim_list.append(0)
            else:
                sim = np.dot(emb, select_face_emb)
                sim_list.append(sim.item())
        sort_idx = np.argsort(sim_list)[::-1]

        return np.array(swap_results)[
            sort_idx[:min(int(num_gen_images), len(swap_results))]]
    else:
        return np.array(swap_results_ori)


class GenPortrait:

    def __init__(self):
        dtype = torch.float16
        
        pose_model_id = 'damo/face_chain_control_model'
        pose_revision = 'v1.0.1'
        pose_file = 'model_controlnet/control_v11p_sd15_openpose'
        pose_model_path = os.path.join(
            snapshot_download(pose_model_id, revision=pose_revision),
            pose_file)
        
        self.controlnet = ControlNetModel.from_pretrained(pose_model_path, torch_dtype=torch.float16)

        self.segmentation_pipeline = pipeline(
            Tasks.image_segmentation,
            'damo/cv_resnet101_image-multiple-human-parsing',
            model_revision='v1.0.1')

        self.image_face_fusion = pipeline('face_fusion_torch',
                                     model='damo/cv_unet_face_fusion_torch', model_revision='v1.0.3')

        model_dir = snapshot_download(
            'damo/face_chain_control_model', revision='v1.0.1')
        self.openpose = OpenposeDetector.from_pretrained(
            os.path.join(model_dir, 'model_controlnet/ControlNet')).to('cuda')

        self.face_quality_func = pipeline(
            Tasks.face_quality_assessment,
            'damo/cv_manual_face-quality-assessment_fqa',
            model_revision='v2.0')

        model_dir = snapshot_download(
            'ly261666/cv_wanx_style_model', revision='v1.0.2')
        
        fr_weight_path = snapshot_download('yucheng1996/FaceChain-FACT', revision='v1.0.0')
        fr_weight_path = os.path.join(fr_weight_path, 'ms1mv2_model_TransFace_S.pt')
        
        fact_model_path = snapshot_download('yucheng1996/FaceChain-FACT', revision='v1.0.0')
        self.face_adapter_path_maj = os.path.join(fact_model_path, 'adapter_maj_mask_large_new_reg001_faceshuffle_00290001.ckpt')
        self.face_adapter_path_film = os.path.join(fact_model_path, 'adapter_film_mask_large_new_reg001_faceshuffle_00290001.ckpt')
        
        self.face_extracter_maj = Face_Extracter_v1(fr_weight_path=fr_weight_path, fc_weight_path=self.face_adapter_path_maj)
        self.face_extracter_film = Face_Extracter_v1(fr_weight_path=fr_weight_path, fc_weight_path=self.face_adapter_path_film)
        
        self.face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_resnet50_face-detection_retinaface')
        self.skin_retouching = pipeline(
            'skin-retouching-torch',
            model='damo/cv_unet_skin_retouching_torch',
            model_revision='v1.0.1')
        self.fair_face_attribute_func = pipeline(Tasks.face_attribute_recognition,
            snapshot_download('damo/cv_resnet34_face-attribute-recognition_fairface', revision='v2.0.2'))
        
        base_model_path_maj = snapshot_download('MAILAND/majicmixRealistic_v6', revision='v1.0.0')
        base_model_path_maj = os.path.join(base_model_path_maj, 'realistic')
        
        base_model_path_film = snapshot_download('ly261666/cv_portrait_model', revision='v2.0')
        base_model_path_film = os.path.join(base_model_path_film, 'film/film')
        
        self.pipe_maj = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_path_maj,
                safety_checker=None,
                controlnet=self.controlnet,
                torch_dtype=dtype)
        self.pipe_maj.scheduler = PNDMScheduler.from_config(
                self.pipe_maj.scheduler.config)
        
        self.pipe_film = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_path_film,
                safety_checker=None,
                controlnet=self.controlnet,
                torch_dtype=dtype)
        self.pipe_film.scheduler = PNDMScheduler.from_config(
                self.pipe_film.scheduler.config)
        
        self.cfg_scale = 7.0
        
        self.face_adapter_maj = FaceAdapter_v1(self.pipe_maj, self.face_detection, self.segmentation_pipeline, self.face_extracter_maj, self.face_adapter_path_maj, 'cuda', True)
        self.face_adapter_maj.set_scale(0.5)
        self.face_adapter_maj.delayed_face_condition = 0.0
        self.face_adapter_maj.pipe.to('cpu')
        
        self.face_adapter_film = FaceAdapter_v1(self.pipe_film, self.face_detection, self.segmentation_pipeline, self.face_extracter_film, self.face_adapter_path_film, 'cuda', True)
        self.face_adapter_film.set_scale(0.5)
        self.face_adapter_film.delayed_face_condition = 0.0
        self.face_adapter_film.pipe.to('cpu')
        
        self.use_main_model = True
        self.use_post_process = False
        self.use_stylization = False        
        self.use_depth_control = True
        self.use_face_pose = True
        self.use_face_swap = True
        

    def __call__(self,
                 use_face_swap,
                 num_gen_images=1, 
                 base_model_idx=0, 
                 style_model_path=None,
                 pos_prompt='', 
                 neg_prompt='', 
                 input_img_path=None, 
                 pose_image=None, 
                 multiplier_style=0):
        
        self.use_face_swap = (use_face_swap > 0)
        st = time.time()
        if pose_image is not None:
            self.pose_image = pose_image
            self.pose_control_weight = 1.0
            self.depth_control_weight = 0.0
        else:
            self.pose_image = input_img_path
            self.pose_control_weight = 0.0
            self.depth_control_weight = 0.0

        self.out_img_size = 512
        
        input_img = Image.open(input_img_path).convert('RGB')
        w, h = input_img.size
        if max(w, h) > 1000:
            scale = 1000 / max(w, h)
            input_img = input_img.resize((int(w * scale), int(h * scale)))
        
        if True:
            result = self.skin_retouching(input_img)
            input_img = Image.fromarray(result[OutputKeys.OUTPUT_IMG][:,:,::-1])
            self.pos_prompt = pos_prompt
            self.neg_prompt = neg_prompt
            
            attribute_result = self.fair_face_attribute_func(input_img)
            score_gender = np.array(attribute_result['scores'][0])
            score_age = np.array(attribute_result['scores'][1])
            gender = np.argmax(score_gender)
            age = np.argmax(score_age)
            if age < 2:
                if gender == 0:
                    attr_idx = 0
                else:
                    attr_idx = 1
            elif age > 4:
                if gender == 0:
                    attr_idx = 4
                else:
                    attr_idx = 5
            else:
                if gender == 0:
                    attr_idx = 2
                else:
                    attr_idx = 3
            use_age_prompt = True
            if attr_idx == 3 or attr_idx == 5:
                use_age_prompt = False

            age_prompts = ['20-year-old, ', '25-year-old, ', '35-year-old, ']

            if age > 1 and age < 5 and use_age_prompt:
                self.pos_prompt = age_prompts[age - 2] + self.pos_prompt
            
            trigger_styles = [
                'a boy, children, ', 'a girl, children, ',
                'a handsome man, ', 'a beautiful woman, ',
                'a mature man, ', 'a mature woman, '
            ]
            trigger_style = trigger_styles[attr_idx]
            if attr_idx == 2 or attr_idx == 4:
                self.neg_prompt += ', children'
            
            self.pos_prompt = trigger_style + self.pos_prompt
        
        if base_model_idx == 0:
            self.pipe = self.pipe_film
            self.face_adapter = self.face_adapter_film
        else:
            self.pipe = self.pipe_maj
            self.face_adapter = self.face_adapter_maj
        
        if style_model_path is None:
            model_dir = snapshot_download(
                'Cherrytest/zjz_mj_jiyi_small_addtxt_fromleo', revision='v1.0.0')
            style_model_path = os.path.join(
                model_dir, 'zjz_mj_jiyi_small_addtxt_fromleo.safetensors')
                
        # main_model_inference PIL
        self.pipe.to('cuda')
        
        self.pipe = merge_lora(
            self.pipe,
            style_model_path,
            multiplier_style,
            device='cuda',
            from_safetensor=True)
        
        gen_results, is_old = main_model_inference(
            num_gen_images,
            self.pose_control_weight,
            self.depth_control_weight,
            self.pose_image,
            self.use_depth_control,
            self.use_face_pose,
            self.pos_prompt,
            self.neg_prompt,
            self.use_main_model,
            input_img=input_img,
            segmentation_pipeline=self.segmentation_pipeline,
            image_face_fusion=self.image_face_fusion,
            openpose=self.openpose,
            controlnet=self.controlnet,
            depth_estimator=None,
            pipe=self.face_adapter,
            cfg_scale=self.cfg_scale)
        mt = time.time()
        
        self.pipe = restore_lora(
            self.pipe,
            style_model_path,
            multiplier_style,
            device='cuda',
            from_safetensor=True)
        
        self.pipe.to('cpu')

        # select_high_quality_face PIL
        selected_face = input_img
        # face_swap cv2
        swap_results = face_swap_fn(self.use_face_swap, gen_results,
                                    selected_face, self.image_face_fusion)
        # stylization
        final_gen_results = stylization_fn(self.use_stylization, swap_results)

        return final_gen_results


def compress_image(input_path, target_size):
    output_path = change_extension_to_jpg(input_path)

    image = cv2.imread(input_path)

    quality = 95
    while cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality
                                       ])[1].size > target_size:
        quality -= 5

    compressed_image = cv2.imencode(
        '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tostring()

    with open(output_path, 'wb') as f:
        f.write(compressed_image)
    return output_path


def change_extension_to_jpg(image_path):

    base_name = os.path.basename(image_path)
    new_base_name = os.path.splitext(base_name)[0] + '.jpg'

    directory = os.path.dirname(image_path)

    new_image_path = os.path.join(directory, new_base_name)
    return new_image_path
