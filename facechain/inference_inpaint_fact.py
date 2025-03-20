# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import sys
import time

import cv2
import json
import numpy as np
import torch

from controlnet_aux import OpenposeDetector
from diffusers import (ControlNetModel, DPMSolverMultistepScheduler,
                       DPMSolverSinglestepScheduler,
                       StableDiffusionControlNetInpaintPipeline,
                       StableDiffusionControlNetPipeline,
                       StableDiffusionImg2ImgPipeline, StableDiffusionPipeline,
                       PNDMScheduler,
                       UniPCMultistepScheduler)
from facechain.merge_lora import merge_lora

from PIL import Image
from skimage import transform
from torch import multiprocessing
from transformers import pipeline as tpipeline

from modelscope import snapshot_download
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from face_adapter import FaceAdapter_v1, Face_Extracter_v1


def concatenate_images(images):
    heights = [img.shape[0] for img in images]
    max_width = sum([img.shape[1] for img in images])

    concatenated_image = np.zeros((max(heights), max_width, 3), dtype=np.uint8)
    x_offset = 0
    for img in images:
        concatenated_image[0:img.shape[0],
                           x_offset:x_offset + img.shape[1], :] = img
        x_offset += img.shape[1]
    return concatenated_image


def call_face_crop(det_pipeline, image, crop_ratio):
    det_result = det_pipeline(image)
    bboxes = det_result['boxes']
    keypoints = det_result['keypoints']
    area = 0
    idx = 0
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        area_tmp = (float(bbox[2]) - float(bbox[0])) * (
            float(bbox[3]) - float(bbox[1]))
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
    bbox[0] = np.clip(
        np.array(bbox[0], np.int32) - face_w * (crop_ratio - 1) / 2, 0, w - 1)
    bbox[1] = np.clip(
        np.array(bbox[1], np.int32) - face_h * (crop_ratio - 1) / 2, 0, h - 1)
    bbox[2] = np.clip(
        np.array(bbox[2], np.int32) + face_w * (crop_ratio - 1) / 2, 0, w - 1)
    bbox[3] = np.clip(
        np.array(bbox[3], np.int32) + face_h * (crop_ratio - 1) / 2, 0, h - 1)
    bbox = np.array(bbox, np.int32)
    return bbox, points_array


def crop_and_paste(Source_image,
                   Source_image_mask,
                   Target_image,
                   Source_Five_Point,
                   Target_Five_Point,
                   Source_box,
                   use_warp=True):
    if use_warp:
        Source_Five_Point = np.reshape(Source_Five_Point, [5, 2]) - np.array(
            Source_box[:2])
        Target_Five_Point = np.reshape(Target_Five_Point, [5, 2])

        Crop_Source_image = Source_image.crop(np.int32(Source_box))
        Crop_Source_image_mask = Source_image_mask.crop(np.int32(Source_box))
        Source_Five_Point, Target_Five_Point = np.array(
            Source_Five_Point), np.array(Target_Five_Point)

        tform = transform.SimilarityTransform()
        tform.estimate(Source_Five_Point, Target_Five_Point)
        M = tform.params[0:2, :]

        warped = cv2.warpAffine(
            np.array(Crop_Source_image),
            M,
            np.shape(Target_image)[:2][::-1],
            borderValue=0.0)
        warped_mask = cv2.warpAffine(
            np.array(Crop_Source_image_mask),
            M,
            np.shape(Target_image)[:2][::-1],
            borderValue=0.0)

        mask = np.float16(warped_mask == 0)
        output = mask * np.float16(Target_image) + (
            1 - mask) * np.float16(warped)
    else:
        mask = np.float16(np.array(Source_image_mask) == 0)
        output = mask * np.float16(Target_image) + (
            1 - mask) * np.float16(Source_image)
    return output, mask


def segment(segmentation_pipeline,
            img,
            ksize=0,
            eyeh=0,
            ksize1=0,
            include_neck=False,
            warp_mask=None,
            return_human=False):
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
        mask_human = np.zeros((h, w))
        for i in range(len(labels)):
            if scores[i] > 0.8:
                if labels[i] == 'Torso-skin':
                    if np.sum(masks[i]) > np.sum(mask_neck):
                        mask_neck = masks[i]
                elif labels[i] == 'Face':
                    if np.sum(masks[i]) > np.sum(mask_face):
                        mask_face = masks[i]
                elif labels[i] == 'Human':
                    if np.sum(masks[i]) > np.sum(mask_human):
                        mask_human = masks[i]
                elif labels[i] == 'Hair':
                    if np.sum(masks[i]) > np.sum(mask_hair):
                        mask_hair = masks[i]
        mask_face = np.clip(mask_face, 0, 1)
        mask_hair = np.clip(mask_hair, 0, 1)
        mask_neck = np.clip(mask_neck, 0, 1)
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
                    soft_mask_dilate = soft_mask_dilate * (
                        np.clip(soft_mask + warp_mask[:, :, 0], 0, 1))
                if eyeh > 0:
                    soft_mask = np.concatenate(
                        (soft_mask[:eyeh], soft_mask_dilate[eyeh:]), axis=0)
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


def img2img_multicontrol(img,
                         face_image,
                         control_image,
                         controlnet_conditioning_scale,
                         pipe,
                         mask,
                         pos_prompt,
                         neg_prompt,
                         strength,
                         num=1,
                         use_ori=False):
    image_mask = Image.fromarray(np.uint8(mask * 255))
    image_human = []
    for i in range(num):
        image_human.append(
            pipe.generate(
                image=img,
                face_image=face_image,
                mask_image=image_mask,
                control_image=control_image,
                prompt=pos_prompt,
                negative_prompt=neg_prompt,
                guidance_scale=5.0,
                strength=strength,
                num_inference_steps=50,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_images_per_prompt=1)[0])
        if use_ori:
            image_human[i] = Image.fromarray(
                (np.array(image_human[i]) * mask[:, :, None] + np.array(img) *
                 (1 - mask[:, :, None])).astype(np.uint8))
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


def main_diffusion_inference_inpaint(num_gen_images,
                                     inpaint_image,
                                     strength,
                                     output_img_size,
                                     times,
                                     pos_prompt,
                                     neg_prompt,
                                     input_img,
                                     segmentation_pipeline=None,
                                     image_face_fusion=None,
                                     openpose=None,
                                     controlnet=None,
                                     det_pipeline=None,
                                     pipe_pose=None,
                                     pipe_all=None,
                                     face_quality_func=None):

    dtype = torch.float16

    add_prompt_style = ''

    if isinstance(inpaint_image, str):
        inpaint_im = Image.open(inpaint_image)
    else:
        inpaint_im = inpaint_image
    inpaint_im = crop_bottom(inpaint_im, 512)

    st = time.time()
    openpose_image = openpose(
        np.array(inpaint_im, np.uint8), include_hand=True, include_face=False)
    w, h = inpaint_im.size
    et = time.time()
    print('inference_0 time: {:.4f}s'.format(et - st))


    st = time.time()

    image_faces = []
    for i in range(num_gen_images):
        image_face = pipe_pose.generate(
            prompt=add_prompt_style + pos_prompt,
            image=openpose_image,
            face_image=input_img,
            height=h,
            width=w,
            guidance_scale=5.0,
            negative_prompt=neg_prompt,
            num_inference_steps=50,
            num_images_per_prompt=1)[0]
        image_faces.append(image_face)
    et = time.time()
    print('inference_1 time: {:.4f}s'.format(et - st))

    st = time.time()
    selected_face = input_img
    swap_results = face_swap_fn(True, image_faces, selected_face,
                                image_face_fusion, segmentation_pipeline)
    torch.cuda.empty_cache()

    et = time.time()
    print('inference_2 time: {:.4f}s'.format(et - st))

    st = time.time()
    images_human = []
    images_auto = []
    inpaint_bbox, inpaint_keypoints = call_face_crop(det_pipeline, inpaint_im,
                                                     1.1)
    eye_height = int((inpaint_keypoints[0, 1] + inpaint_keypoints[1, 1]) / 2)
    canny_image = cv2.Canny(np.array(inpaint_im, np.uint8), 100, 200)[:, :,
                                                                      None]
    mask = segment(
        segmentation_pipeline, inpaint_im, ksize=0.05, eyeh=eye_height)
    canny_image = (canny_image * (1.0 - mask[:, :, None])).astype(np.uint8)
    canny_image = Image.fromarray(
        np.concatenate([canny_image, canny_image, canny_image], axis=2))
    et = time.time()
    print('inference_4 time: {:.4f}s'.format(et - st))
    st = time.time()
    # canny_image.save('canny.png')
    for i in range(num_gen_images):
        image_face = swap_results[i]
        image_face = Image.fromarray(image_face[:, :, ::-1])

        face_bbox, face_keypoints = call_face_crop(det_pipeline, image_face,
                                                   1.5)
        face_mask = segment(segmentation_pipeline, image_face)
        face_mask = np.expand_dims((face_mask * 255).astype(np.uint8), axis=2)
        face_mask = np.concatenate([face_mask, face_mask, face_mask], axis=2)
        face_mask = Image.fromarray(face_mask)
        replaced_input_image, warp_mask = crop_and_paste(
            image_face, face_mask, inpaint_im, face_keypoints,
            inpaint_keypoints, face_bbox)
        warp_mask = 1.0 - warp_mask
        # cv2.imwrite('tmp_{}.png'.format(i), replaced_input_image[:, :, ::-1])

        st = time.time()
        openpose_image = openpose(
            np.array(replaced_input_image * warp_mask, np.uint8),
            include_hand=True,
            include_body=False,
            include_face=True)
        et = time.time()
        print('inference_5 time: {:.4f}s'.format(et - st))
        # openpose_image.save('openpose_{}.png'.format(i))
        read_control = [openpose_image, canny_image]
        inpaint_mask, human_mask = segment(
            segmentation_pipeline,
            inpaint_im,
            ksize=0.1,
            ksize1=0.04,
            eyeh=eye_height,
            include_neck=False,
            warp_mask=warp_mask,
            return_human=True)
        inpaint_with_mask = ((1.0 - inpaint_mask[:, :, None])
                             * np.array(inpaint_im))[:, :, ::-1]
        # cv2.imwrite('inpaint_with_mask_{}.png'.format(i), inpaint_with_mask)
        print('Finishing segmenting images.')
        images_human.extend(
            img2img_multicontrol(
                inpaint_im,
                input_img,
                read_control, [1.0, 0.2],
                pipe_all,
                inpaint_mask,
                add_prompt_style + pos_prompt,
                neg_prompt,
                strength=strength))
        images_auto.extend(
            img2img_multicontrol(
                inpaint_im,
                input_img,
                read_control, [1.0, 0.2],
                pipe_all,
                np.zeros_like(inpaint_mask),
                add_prompt_style + pos_prompt,
                neg_prompt,
                strength=0.025))

        edge_add = np.array(inpaint_im).astype(np.int16) - np.array(
            images_auto[i]).astype(np.int16)
        edge_add = edge_add * (1 - human_mask[:, :, None])
        images_human[i] = Image.fromarray((np.clip(
            np.array(images_human[i]).astype(np.int16)
            + edge_add.astype(np.int16), 0, 255)).astype(np.uint8))

    st = time.time()
    images_rst = []
    for i in range(len(images_human)):
        im = images_human[i]
        canny_image = cv2.Canny(np.array(im, np.uint8), 100, 200)[:, :, None]
        canny_image = Image.fromarray(
            np.concatenate([canny_image, canny_image, canny_image], axis=2))
        st = time.time()
        openpose_image = openpose(
            np.array(im, np.uint8),
            include_hand=True,
            include_face=True,
            include_body=False)
        et = time.time()
        print('inference_6 time: {:.4f}s'.format(et - st))
        read_control = [openpose_image, canny_image]
        inpaint_mask, human_mask = segment(
            segmentation_pipeline,
            images_human[i],
            ksize=0.02,
            return_human=True)
        print('Finishing segmenting images.')
        image_rst = img2img_multicontrol(
            im,
            input_img,
            read_control, [0.8, 0.8],
            pipe_all,
            inpaint_mask,
            add_prompt_style + pos_prompt,
            neg_prompt,
            strength=0.1,
            num=1)[0]
        image_auto = img2img_multicontrol(
            im,
            input_img,
            read_control, [0.8, 0.8],
            pipe_all,
            np.zeros_like(inpaint_mask),
            add_prompt_style + pos_prompt,
            neg_prompt,
            strength=0.025,
            num=1)[0]
        edge_add = np.array(im).astype(np.int16) - np.array(image_auto).astype(
            np.int16)
        edge_add = edge_add * (1 - human_mask[:, :, None])
        image_rst = Image.fromarray((np.clip(
            np.array(image_rst).astype(np.int16) + edge_add.astype(np.int16),
            0, 255)).astype(np.uint8))
        images_rst.append(image_rst)

    return images_rst, False


def stylization_fn(use_stylization, rank_results):
    if use_stylization:
        #  TODO
        pass
    else:
        return rank_results


def main_model_inference(num_gen_images,
                         inpaint_image,
                         strength,
                         output_img_size,
                         pos_prompt,
                         neg_prompt,
                         use_main_model,
                         input_img=None,
                         segmentation_pipeline=None,
                         image_face_fusion=None,
                         openpose=None,
                         controlnet=None,
                         det_pipeline=None,
                         pipe_pose=None,
                         pipe_all=None,
                         face_quality_func=None):
    # inpaint_image = compress_image(inpaint_image, 1024 * 1024)
    if use_main_model:
        return main_diffusion_inference_inpaint(
            num_gen_images,
            inpaint_image,
            strength,
            output_img_size,
            1,
            pos_prompt,
            neg_prompt,
            input_img,
            segmentation_pipeline=segmentation_pipeline,
            image_face_fusion=image_face_fusion,
            openpose=openpose,
            controlnet=controlnet,
            det_pipeline=det_pipeline,
            pipe_pose=pipe_pose,
            pipe_all=pipe_all,
            face_quality_func=face_quality_func)


def select_high_quality_face(input_img_dir, face_quality_func):
    input_img_dir = str(input_img_dir) + '_labeled'
    quality_score_list = []
    abs_img_path_list = []
    #  TODO
    for img_name in os.listdir(input_img_dir):
        if img_name.endswith('jsonl') or img_name.startswith(
                '.ipynb') or img_name.startswith('.safetensors'):
            continue

        if img_name.endswith('jpg') or img_name.endswith('png'):
            abs_img_name = os.path.join(input_img_dir, img_name)
            face_quality_score = face_quality_func(abs_img_name)[
                OutputKeys.SCORES]
            if face_quality_score is None:
                quality_score_list.append(0)
            else:
                quality_score_list.append(face_quality_score[0])
            abs_img_path_list.append(abs_img_name)

    sort_idx = np.argsort(quality_score_list)[::-1]
    print('Selected face: ' + abs_img_path_list[sort_idx[0]])

    return Image.open(abs_img_path_list[sort_idx[0]])


def face_swap_fn(use_face_swap, gen_results, template_face, image_face_fusion,
                 segmentation_pipeline):
    if use_face_swap:
        #  TODO
        out_img_list = []
        for img in gen_results:
            result = image_face_fusion(dict(
                template=img, user=template_face))[OutputKeys.OUTPUT_IMG]
            face_mask = segment(segmentation_pipeline, img, ksize=0.1)
            result = (result * face_mask[:, :, None]
                      + np.array(img)[:, :, ::-1] *
                      (1 - face_mask[:, :, None])).astype(np.uint8)
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
        swap_results = swap_results_ori

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


def process_inpaint_img(inpaint_img, resize_size=(1024, 1024)):
    if isinstance(inpaint_img, str):
        #inpaint_img = compress_image(inpaint_img, 1024 * 1024)
        inpaint_img = Image.open(inpaint_img)
    else:
        inpaint_img = Image.fromarray(inpaint_img[:, :, ::-1])
    ori_img = np.array(inpaint_img)

    h, w, _ = ori_img.shape
    ns = max(h, w)
    ori_img_square = cv2.copyMakeBorder(ori_img, int((ns - h) / 2),
                                        (ns - h) - int((ns - h) / 2),
                                        int((ns - w) / 2), (ns - w) - int(
                                            (ns - w) / 2), cv2.BORDER_DEFAULT)
    ori_img_square_resized = cv2.resize(ori_img_square, resize_size)
    return Image.fromarray(ori_img_square_resized)


def postprocess_inpaint_img(img2img_res, output_size=(768, 1024)):
    resized = cv2.resize(
        np.array(img2img_res), (output_size[1], output_size[1]))
    croped = resized[:, (output_size[1] - output_size[0])
                     // 2:(output_size[1] - output_size[0]) // 2
                     + output_size[0], :]
    return Image.fromarray(croped)


class GenPortrait_inpaint:

    def __init__(self):
        cfg_face = True
        
        fact_model_path = snapshot_download('yucheng1996/FaceChain-FACT', revision='v1.0.0')
        adapter_path = os.path.join(fact_model_path, 'adapter_maj_mask_large_new_reg001_faceshuffle_00290001.ckpt')

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
        self.depth_estimator = tpipeline(
            'depth-estimation',
            os.path.join(model_dir, 'model_controlnet/dpt-large'))

        self.face_quality_func = pipeline(
            Tasks.face_quality_assessment,
            'damo/cv_manual_face-quality-assessment_fqa',
            model_revision='v2.0')
        self.face_detection = pipeline(
            task=Tasks.face_detection,
            model='damo/cv_ddsar_face-detection_iclr23-damofd',
            model_revision='v1.1')

        dtype = torch.float16
        model_dir1 = snapshot_download(
            'ly261666/cv_wanx_style_model', revision='v1.0.3')
        self.controlnet = [
            ControlNetModel.from_pretrained(
                os.path.join(model_dir,
                             'model_controlnet/control_v11p_sd15_openpose'),
                torch_dtype=dtype),
            ControlNetModel.from_pretrained(
                os.path.join(model_dir1, 'contronet-canny'), torch_dtype=dtype)
        ]

        model_dir = snapshot_download(
            'ly261666/cv_wanx_style_model', revision='v1.0.2')

        self.face_adapter_path = adapter_path
        self.cfg_face = cfg_face
        
        fr_weight_path = snapshot_download('yucheng1996/FaceChain-FACT', revision='v1.0.0')
        fr_weight_path = os.path.join(fr_weight_path, 'ms1mv2_model_TransFace_S.pt')
        
        self.face_extracter = Face_Extracter_v1(fr_weight_path=fr_weight_path, fc_weight_path=self.face_adapter_path)
        self.face_detection0 = pipeline(task=Tasks.face_detection, model='damo/cv_resnet50_face-detection_retinaface')
        self.skin_retouching = pipeline(
            'skin-retouching-torch',
            model=snapshot_download('damo/cv_unet_skin_retouching_torch', revision='v1.0.1.1'))
        self.fair_face_attribute_func = pipeline(Tasks.face_attribute_recognition,
            snapshot_download('damo/cv_resnet34_face-attribute-recognition_fairface', revision='v2.0.2'))
        
        base_model_path = snapshot_download('MAILAND/majicmixRealistic_v6', revision='v1.0.0')
        base_model_path = os.path.join(base_model_path, 'realistic')
        
        pipe_pose = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path,
            safety_checker=None,
            controlnet=self.controlnet[0],
            torch_dtype=torch.float16)
        pipe_all = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            base_model_path,
            safety_checker=None,
            controlnet=self.controlnet,
            torch_dtype=torch.float16)
        pipe_pose.scheduler = PNDMScheduler.from_config(
                pipe_pose.scheduler.config)
        pipe_all.scheduler = PNDMScheduler.from_config(
                pipe_all.scheduler.config)
        
        face_adapter_path = self.face_adapter_path
        self.face_adapter_pose = FaceAdapter_v1(pipe_pose, self.face_detection0, self.segmentation_pipeline, self.face_extracter, face_adapter_path, 'cuda', self.cfg_face) 
        self.face_adapter_all = FaceAdapter_v1(pipe_all, self.face_detection0, self.segmentation_pipeline, self.face_extracter, face_adapter_path, 'cuda', self.cfg_face) 
        self.face_adapter_pose.set_scale(0.5)
        self.face_adapter_all.set_scale(0.55)
        
        self.face_adapter_pose.pipe.to('cpu')
        self.face_adapter_all.pipe.to('cpu')
                   

    def __call__(self,
                 use_face_swap,
                 inpaint_img,
                 strength,
                 output_img_size,
                 num_faces,
                 selected_face,
                 pos_prompt,
                 neg_prompt,
                 input_img_path=None,
                 num_gen_images=1):
        
        st = time.time()
        self.use_main_model = True
        self.use_face_swap = (use_face_swap > 0)
        self.use_post_process = False
        self.use_stylization = False
        self.neg_prompt = neg_prompt
        self.inpaint_img = inpaint_img
        self.strength = strength
        self.num_faces = num_faces
        self.selected_face = selected_face
        self.output_img_size = output_img_size
        self.pos_prompt = pos_prompt

        if isinstance(self.inpaint_img, str):
            self.inpaint_img = Image.open(self.inpaint_img)
        else:
            self.inpaint_img = Image.fromarray(self.inpaint_img[:, :, ::-1])
        result_det = self.face_detection(self.inpaint_img)
        bboxes = result_det['boxes']
        assert len(bboxes) > self.num_faces - 1
        bboxes = np.array(bboxes).astype(np.int16)
        if len(bboxes) > self.num_faces:
            areas = np.zeros(len(bboxes))
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                areas[i] = (float(bbox[2]) - float(bbox[0])) * (
                    float(bbox[3]) - float(bbox[1]))
            top_idxs = np.argsort(areas)[::-1][:self.num_faces]
            bboxes = bboxes[top_idxs]
            assert len(bboxes) == self.num_faces
        lefts = []
        for bbox in bboxes:
            lefts.append(bbox[0])
        idxs = np.argsort(lefts)

        if input_img_path != None:
            face_box = bboxes[idxs[self.selected_face - 1]]
            inpaint_img_large = np.copy(np.array(self.inpaint_img)[:, :, ::-1])
            mask_large = np.ones_like(inpaint_img_large)
            mask_large1 = np.zeros_like(inpaint_img_large)
            h, w, _ = inpaint_img_large.shape
            for i in range(len(bboxes)):
                if i != idxs[self.selected_face - 1]:
                    bbox = bboxes[i]
                    inpaint_img_large[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0
                    mask_large[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 0

            face_ratio = 0.7
            cropl = int(
                max(face_box[3] - face_box[1], face_box[2] - face_box[0])
                / face_ratio / 2)
            cx = int((face_box[2] + face_box[0]) / 2)
            cy = int((face_box[1] + face_box[3]) / 2)
            cropup = min(cy, cropl)
            cropbo = min(h - cy, cropl)
            crople = min(cx, cropl)
            cropri = min(w - cx, cropl)
            inpaint_img = np.pad(
                inpaint_img_large[cy - cropup:cy + cropbo,
                                  cx - crople:cx + cropri],
                ((cropl - cropup, cropl - cropbo),
                 (cropl - crople, cropl - cropri), (0, 0)), 'constant')
            inpaint_img = cv2.resize(inpaint_img, (512, 512))
            inpaint_img = Image.fromarray(inpaint_img[:, :, ::-1])
            mask_large1[cy - cropup:cy + cropbo, cx - crople:cx + cropri] = 1
            mask_large = mask_large * mask_large1
            
            input_img = Image.open(input_img_path).convert('RGB')
            w, h = input_img.size
            if max(w, h) > 2000:
                scale = 2000 / max(w, h)
                input_img = input_img.resize((int(w * scale), int(h * scale)))

            result = self.skin_retouching(np.array(input_img)[:,:,::-1])
            input_img = result[OutputKeys.OUTPUT_IMG]

            input_img = Image.fromarray(input_img[:, :, ::-1])
            
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
            
            self.face_adapter_pose.pipe.to('cuda')
            self.face_adapter_all.pipe.to('cuda')

            gen_results, is_old = main_model_inference(
                num_gen_images,
                inpaint_img,
                self.strength,
                self.output_img_size,
                self.pos_prompt,
                self.neg_prompt,
                self.use_main_model,
                input_img=input_img,
                segmentation_pipeline=self.segmentation_pipeline,
                image_face_fusion=self.image_face_fusion,
                openpose=self.openpose,
                controlnet=self.controlnet,
                det_pipeline=self.face_detection,
                pipe_pose=self.face_adapter_pose,
                pipe_all=self.face_adapter_all,
                face_quality_func=self.face_quality_func)
            mt = time.time()
            self.face_adapter_pose.pipe.to('cpu')
            self.face_adapter_all.pipe.to('cpu')

            # select_high_quality_face PIL
            selected_face = input_img
            # face_swap cv2
            swap_results = face_swap_fn(self.use_face_swap, gen_results,
                                        selected_face, self.image_face_fusion,
                                        self.segmentation_pipeline)
            # stylization
            final_gen_results = swap_results
            
            final_gen_results_new = []
            inpaint_img_large = np.copy(np.array(self.inpaint_img)[:, :, ::-1])
            ksize = int(10 * cropl / 256)
            for i in range(len(final_gen_results)):
                print('Start cropping.')
                rst_gen = cv2.resize(final_gen_results[i],
                                     (cropl * 2, cropl * 2))
                rst_crop = rst_gen[cropl - cropup:cropl + cropbo,
                                   cropl - crople:cropl + cropri]
                print(rst_crop.shape)
                inpaint_img_rst = np.zeros_like(inpaint_img_large)
                print('Start pasting.')
                inpaint_img_rst[cy - cropup:cy + cropbo,
                                cx - crople:cx + cropri] = rst_crop
                print('Fininsh pasting.')
                print(inpaint_img_rst.shape, mask_large.shape,
                      inpaint_img_large.shape)
                mask_large = mask_large.astype(np.float32)
                kernel = np.ones((ksize * 2, ksize * 2))
                mask_large1 = cv2.erode(mask_large, kernel, iterations=1)
                mask_large1 = cv2.GaussianBlur(
                    mask_large1,
                    (int(ksize * 1.8) * 2 + 1, int(ksize * 1.8) * 2 + 1), 0)
                mask_large1[face_box[1]:face_box[3],
                            face_box[0]:face_box[2]] = 1
                mask_large = mask_large * mask_large1
                final_inpaint_rst = (
                    inpaint_img_rst.astype(np.float32)
                    * mask_large.astype(np.float32)
                    + inpaint_img_large.astype(np.float32) *
                    (1.0 - mask_large.astype(np.float32))).astype(np.uint8)
                print('Finish masking.')
                final_gen_results_new.append(final_inpaint_rst)
                print('Finish generating.')

        et = time.time()
        print('Inference Time: {:.4f}s'.format(et - st))
        print('Inference Time Process: {:.4f}s'.format(et - mt))
        torch.cuda.empty_cache()
        return final_gen_results_new


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
