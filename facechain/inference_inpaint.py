
import argparse
import copy
import cv2
import gc
import math
import numpy as np
import os
import torch
from glob import glob

from PIL import Image
from skimage import transform
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetInpaintPipeline,
)
from controlnet_aux import OpenposeDetector
from facechain.constants import paiya_default_positive, paiya_default_negative
from facechain.merge_lora import merge_lora


def safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, face_seg, mask_type):
    """
    Get expanded box, keypoints, and face segmentation mask.

    Args:
        image (np.ndarray): The input image.
        retinaface_result (dict): The detection result from RetinaFace.
        crop_ratio (float): The ratio for expanding the crop of the face part.
        face_seg (function): The face segmentation model.
        mask_type (str): The method of face segmentation, either 'crop' or 'skin'.

    Returns:
        np.ndarray: The box relative to the original image, expanded.
        np.ndarray: The keypoints relative to the original image.
        Image: The face segmentation result.
    """
    h, w, c = np.shape(image)
    if len(retinaface_result['boxes']) != 0:
        # Get the RetinaFace box and expand it
        retinaface_box = np.array(retinaface_result['boxes'][0])
        face_width = retinaface_box[2] - retinaface_box[0]
        face_height = retinaface_box[3] - retinaface_box[1]
        retinaface_box[0] = np.clip(retinaface_box[0] - face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[1] = np.clip(retinaface_box[1] - face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box[2] = np.clip(retinaface_box[2] + face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[3] = np.clip(retinaface_box[3] + face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box = np.array(retinaface_box, np.int32)

        # Detect keypoints
        retinaface_keypoints = np.reshape(retinaface_result['keypoints'][0], [5, 2])
        retinaface_keypoints = np.array(retinaface_keypoints, np.float32)

        # Mask part
        # retinaface_crop = Image.fromarray(image).crop(tuple(np.int32(retinaface_box)))
        retinaface_crop = image.crop(tuple(np.int32(retinaface_box)))
        retinaface_mask = np.zeros_like(np.array(image, np.uint8))
        if mask_type == "skin":
            retinaface_sub_mask = face_seg(retinaface_crop)
            retinaface_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = np.expand_dims(retinaface_sub_mask, -1)
        else:
            retinaface_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
        retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))
    else:
        retinaface_box = np.array([])
        retinaface_keypoints = np.array([])
        retinaface_mask = np.zeros_like(np.array(image, np.uint8))
        retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))

    return retinaface_box, retinaface_keypoints, retinaface_mask_pil


def crop_and_paste(Source_image, Source_image_mask, Target_image, Source_Five_Point, Target_Five_Point, Source_box):
    """
    Crop and paste a face from the source image to the target image.

    Args:
        Source_image (Image): The source image.
        Source_image_mask (Image): The mask of the face in the source image.
        Target_image (Image): The target template image.
        Source_Five_Point (np.ndarray): Five facial keypoints in the source image.
        Target_Five_Point (np.ndarray): Five facial keypoints in the target image.
        Source_box (list): The coordinates of the face box in the source image.

    Returns:
        np.ndarray: The output image with the face pasted.
    """
    Source_Five_Point = np.reshape(Source_Five_Point, [5, 2]) - np.array(Source_box[:2])
    Target_Five_Point = np.reshape(Target_Five_Point, [5, 2])

    Crop_Source_image                       = Source_image.crop(np.int32(Source_box))
    Crop_Source_image_mask                  = Source_image_mask.crop(np.int32(Source_box))
    Source_Five_Point, Target_Five_Point    = np.array(Source_Five_Point), np.array(Target_Five_Point)

    tform = transform.SimilarityTransform()
    tform.estimate(Source_Five_Point, Target_Five_Point)
    M = tform.params[0:2, :]

    warped      = cv2.warpAffine(np.array(Crop_Source_image), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)
    warped_mask = cv2.warpAffine(np.array(Crop_Source_image_mask), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)

    mask        = np.float32(warped_mask == 0)
    output      = mask * np.float32(Target_image) + (1 - mask) * np.float32(warped)
    return output


def call_face_crop(retinaface_detection, image, crop_ratio, prefix="tmp"):
    """
    Perform face detection, mask, and keypoint extraction using RetinaFace.

    Args:
        retinaface_detection (function): The RetinaFace detection function.
        image (Image): The input image.
        crop_ratio (float): The crop ratio for face expansion.
        prefix (str): Prefix for temporary files (default is "tmp").

    Returns:
        np.ndarray: Detected face bounding box.
        np.ndarray: Detected face keypoints.
        Image: Extracted face mask.
    """
    # Perform RetinaFace detection
    retinaface_result = retinaface_detection(image)
    
    # Get mask and keypoints
    retinaface_box, retinaface_keypoints, retinaface_mask_pil = safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, None, "crop")
    
    return retinaface_box, retinaface_keypoints, retinaface_mask_pil


def build_pipeline_facechain(baseline_model_path, lora_model_path, cache_model_dir, from_safetensor=False):
    """
    Build and configure a facechain inpaint pipeline.

    Args:
        baseline_model_path (str): Path to the baseline model.
        lora_model_path (str): Path to the LoRA model.
        cache_model_dir (str): Directory to cache models.
        from_safetensor (bool): Use safe tensor for LoRA.

    Returns:
        pipeline: Built pipeline.
        generator: Random number generator with manual seed.
    """
    # Apply to FP16
    weight_dtype = torch.float32

    # Build ControlNet
    controlnet = [
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=weight_dtype, \
            cache_dir=os.path.join(cache_model_dir, "controlnet")),
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, \
            cache_dir=os.path.join(cache_model_dir, "controlnet")),
    ]

    # Build SDInpaint Pipeline
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        baseline_model_path,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
    ).to("cuda")
    # Merge LoRA into pipeline
    pipe = merge_lora(pipeline, lora_model_path, 1.0, from_safetensor=from_safetensor)

    # to fit some env lack of xformers
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except:
        pass
    pipeline.enable_sequential_cpu_offload()

    # Set Pipeline Scheduler
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    
    # Set manual seed
    generator = torch.Generator("cuda").manual_seed(42)
    
    return pipeline, generator

class GenPortraitInpaint:
    def __init__(self, 
                 cache_model_dir=None,
                 use_main_model=True, 
                 crop_template=True, 
                 short_side_resize=768):
        self.crop_template = crop_template 
        self.short_side_resize = short_side_resize


    def __call__(self, base_model_path, lora_model_path, instance_data_dir,
                 input_template_list, cache_model_dir,
                 first_controlnet_strength=0.45, second_controlnet_strength=0.1, final_fusion_ratio=0.5,
                 use_fusion_before=True, use_fusion_after=True,
                 first_controlnet_conditioning_scale=[0.5, 0.3], second_controlnet_conditioning_scale=[0.75, 0.75]):
        """
        Generate portrait inpaintings.

        Args:
            base_model_path (str): Base model path.
            lora_model_path (str): LoRA model path.
            face_id_image_path (str): Face ID image path.
            input_template_list (list): List of input template paths.
            input_roop_image_list (list): List of input roop image paths.
            input_prompt: Input prompt.
            cache_model_dir: Cache model directory.
            first_controlnet_strength (float): First controlnet strength.
            second_controlnet_strength (float): Second controlnet strength.
            final_fusion_ratio (float): Final fusion ratio.
            use_fusion_before (bool): Flag to use fusion before.
            use_fusion_after (bool): Flag to use fusion after.
            first_controlnet_conditioning_scale (list): First controlnet conditioning scale.
            second_controlnet_conditioning_scale (list): Second controlnet conditioning scale.

        Returns:
            final_res (list): List of generated images.
        """
        print(f'lora_model_path            :', lora_model_path)
        print(f'first_controlnet_strength  :', first_controlnet_strength)
        print(f'second_controlnet_strength :', second_controlnet_strength)
        print(f'final_fusion_ratio         :', final_fusion_ratio)
        print(f'use_fusion_before          :', use_fusion_before)
        print(f'use_fusion_after           :', use_fusion_after)

        pos_prompt = 'Generate a standard photo of a chinese , beautiful smooth face, smile, high detail face, best quality, photorealistic' + paiya_default_positive
        neg_prompt = paiya_default_negative

        # facechain original lora prompt engineering
        train_dir = str(input_img_dir) + '_labeled'
        face_id_image_path = os.path.join(train_dir, 'faceid.jpg')
        input_roop_image_list = glob(os.path.join(train_dir, '*.png'))[:2] # debug for 2

        add_prompt_style = ''
        trigger_style = '<sks>'
        if 1: 
            # train_dir = str(input_img_dir) + '_labeled'
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
            trigger_style = '<sks>, ' + trigger_styles[attr_idx]
    

            if attr_idx == 2 or attr_idx == 4:
                neg_prompt += ', children'

            add_prompt_style_list = []
            for tag in tags_all:
                if tags_all.count(tag) > 0.5 * cnt:
                    if ('hair' in tag or 'face' in tag or 'mouth' in tag or 'skin' in tag or 'smile' in tag):
                        if not tag in add_prompt_style_list:
                            add_prompt_style_list.append(tag)
            
            if len(add_prompt_style_list) > 0:
                add_prompt_style = ", ".join(add_prompt_style_list) + ', '
            else:
                add_prompt_style = ''

        input_prompt = add_prompt_style + trigger_style  + paiya_default_positive
        
        # build pipeline
        sd_inpaint_pipeline, generator = build_pipeline_facechain(
            base_model_path, lora_model_path, cache_model_dir, from_safetensor=lora_model_path.endswith('safetensors')
        )    
        retinaface_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
        image_face_fusion = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo')
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir=os.path.join(cache_model_dir, "controlnet_detector"))
        face_id_image = Image.open(face_id_image_path) 

        final_res = []
        for roop_idx, input_roop_image in enumerate(input_roop_image_list):
            for template_idx, input_template in enumerate(input_template_list):

                template_image =  Image.open(input_template)
                roop_image = Image.open(input_roop_image) 

                # crop template to fit sd 
                if self.crop_template:
                    crop_safe_box, _, _ = call_face_crop(retinaface_detection, template_image, 3, "crop")
                    input_image = copy.deepcopy(template_image).crop(crop_safe_box)
                else:
                    input_image = template_image
                
                if 1:
                    # fit template to shortside 768
                    short_side  = min(input_image.width, input_image.height)
                    resize      = float(short_side / int(self.short_side_resize))
                    new_size    = (int(input_image.width//resize), int(input_image.height//resize))
                    input_image = input_image.resize(new_size)

                    new_width   = int(np.shape(input_image)[1] // 32 * 32)
                    new_height  = int(np.shape(input_image)[0] // 32 * 32)
                    input_image = input_image.resize([new_width, new_height])

                roop_face_retinaface_box, roop_face_retinaface_keypoints, roop_face_retinaface_mask = call_face_crop(retinaface_detection, face_id_image, 1.5, "roop")
                retinaface_box, retinaface_keypoints, input_mask = call_face_crop(retinaface_detection, input_image, 1.1, "template")
                
                # crop and paste original input as OpenPose input
                use_replace_before = True
                if use_replace_before:
                    replaced_input_image = crop_and_paste(face_id_image, roop_face_retinaface_mask, input_image, roop_face_retinaface_keypoints, retinaface_keypoints, roop_face_retinaface_box)
                else:
                    replaced_input_image = input_image
                
                # face fusion input as Canny Input
                use_fusion_before = True
                if use_fusion_before:
                    result          = image_face_fusion(dict(template=input_image, user=roop_image))[OutputKeys.OUTPUT_IMG]
                    result          = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                else:
                    result = input_image
                
                # Prepare for ControlNet
                openpose_image  = self.openpose(np.array(replaced_input_image, np.uint8))
                canny_image     = cv2.Canny(np.array(result, np.uint8), 100, 200)[:, :, None]
                canny_image     = Image.fromarray(np.concatenate([canny_image, canny_image, canny_image], axis=2))
                read_control    = [openpose_image, canny_image]
                
                #  Fusion as Input, and mask inpaint with ControlNet
                generate_image_old = sd_inpaint_pipeline(
                    input_prompt, image=result, mask_image=input_mask, control_image=read_control, strength=first_controlnet_strength, negative_prompt=paiya_default_negative, 
                    guidance_scale=9, num_inference_steps=30, generator=generator, height=np.shape(input_image)[0], width=np.shape(input_image)[1], \
                    controlnet_conditioning_scale=first_controlnet_conditioning_scale
                ).images[0]
                
                # Image fusion with roop
                use_fusion_after = True
                if use_fusion_after:
                    generate_image = image_face_fusion(dict(template=generate_image_old, user=roop_image))[OutputKeys.OUTPUT_IMG]
                    generate_image = cv2.cvtColor(generate_image, cv2.COLOR_BGR2RGB)
                else:
                    generate_image = generate_image_old
                
                # Prepare for ControlNet Input
                openpose_image  = self.openpose(generate_image)
                canny_image     = cv2.Canny(np.array(generate_image, np.uint8), 100, 200)[:, :, None]
                canny_image     = Image.fromarray(np.concatenate([canny_image, canny_image, canny_image], axis=2))
                read_control    = [openpose_image, canny_image]

                # Fusion ensemble before final SD
                input_image_2   = Image.fromarray(np.uint8((np.array(generate_image_old, np.float32) * (1-final_fusion_ratio) + np.array(generate_image, np.float32) * final_fusion_ratio)))
                generate_image = input_image_2
                
                if self.crop_template:
                    origin_image    = np.array(copy.deepcopy(template_image))
                    x1,y1,x2,y2     = crop_safe_box
                    generate_image  = generate_image.resize([x2-x1, y2-y1])
                    origin_image[y1:y2,x1:x2] = np.array(generate_image)
                    origin_image = Image.fromarray(np.uint8(origin_image))
                    # origin_image = Image.fromarray(codeformer_helper.infer(codeFormer_net, face_helper, bg_upsampler, np.array(origin_image)))
                    origin_image.save(f'debug_{roop_idx}_{template_idx}.jpg')
                    generate_image = origin_image
                
                res = cv2.cvtColor(np.array(generate_image), cv2.COLOR_BGR2RGB)
                final_res.append(res)
        
        return final_res


if __name__=="__main__":

    import sys

    input_template = sys.argv[1]
    input_roop_image = sys.argv[2]

    cache_model_dir = '/mnt/zhoulou.wzh/AIGC/model_data/'
    lora_model_path = './pai_ya_tmp/zhoumo.safetensors'
    input_prompt = f"zhoumo_face, zhoumo, 1girl,"
    base_model_path = '/mnt/workspace/.cache/modelscope/ly261666/cv_portrait_model/realistic'

    inpaiter = GenPortraitInpaint(crop_template=True, short_side_resize=512)
    res = inpaiter(base_model_path=base_model_path, lora_model_path=lora_model_path, input_template=[input_template],
        input_roop_image=[input_roop_image], input_prompt=input_prompt, cache_model_dir=cache_model_dir)