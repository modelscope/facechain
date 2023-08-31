
import argparse
import copy
import cv2
import gc
import math
import numpy as np
import os
import torch
import json
from glob import glob

from PIL import Image
from skimage import transform
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetInpaintPipeline,
)
from controlnet_aux import OpenposeDetector
from facechain.data_process.preprocessing import get_popular_prompts
from facechain.constants import inpaint_default_positive, inpaint_default_negative
from facechain.merge_lora import merge_lora
from facechain.data_process.face_process_utils import call_face_crop, crop_and_paste


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
    # Apply to FP32
    weight_dtype = torch.float32

    # Build ControlNet
    controlnet = [
        ControlNetModel.from_pretrained(os.path.join(cache_model_dir, "controlnet", "sd-controlnet-openpose"), torch_dtype=weight_dtype),
        ControlNetModel.from_pretrained(os.path.join(cache_model_dir, "controlnet", "sd-controlnet-canny"), torch_dtype=weight_dtype),
    ]

    # Build SDInpaint Pipeline
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        baseline_model_path,
        safety_checker=None,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
    ).to("cuda")
    # Merge LoRA into pipeline
    pipe = merge_lora(pipeline, lora_model_path, 1.2, from_safetensor=from_safetensor)

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
                 input_template_list, cache_model_dir, select_face_num=2,
                 first_controlnet_strength=0.45, second_controlnet_strength=0.1, final_fusion_ratio=0.5,
                 use_fusion_before=True, use_fusion_after=True,
                 first_controlnet_conditioning_scale=[0.5, 0.3], 
                 sub_path=None, revision=None):
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
        base_model_path = snapshot_download(base_model_path, revision=revision)
        if sub_path is not None and len(sub_path) > 0:
            base_model_path = os.path.join(base_model_path, sub_path)
        print(f'lora_model_path            :', lora_model_path)
        print(f'select_face_num            :', select_face_num)
        print(f'first_controlnet_strength  :', first_controlnet_strength)
        print(f'second_controlnet_strength :', second_controlnet_strength)
        print(f'final_fusion_ratio         :', final_fusion_ratio)
        print(f'use_fusion_before          :', use_fusion_before)
        print(f'use_fusion_after           :', use_fusion_after)
        print(f'input_template_list        :', input_template_list)

        # hack for gr.Text input
        if isinstance(input_template_list, str):
            input_template_list = [input_template_list[2:-2]]

        # setting inpaint used faceid image & roop image with preprocessed output in xx_ensemble dir, if not exists fallback to original FC traindata dir
        reference_dir = str(instance_data_dir) + '_ensemble'
        reference_dir_faceid = os.path.join(reference_dir, 'face_id.jpg')
        if os.path.exists(reference_dir) and os.path.exists(reference_dir_faceid):
            face_id_image_path = glob(os.path.join(lora_model_path, 'face_id.jpg'))[0]
            input_roop_image_list = glob(os.path.join(reference_dir, 'best_roop_image_*.jpg'))[:select_face_num] # debug for 2
        # not exists means no PR104 training ensemble
        else:
            reference_dir = str(instance_data_dir) + '_labeled'
            face_id_image_path = glob(os.path.join(reference_dir, '*.png'))[0]
            input_roop_image_list = glob(os.path.join(reference_dir, '*.png'))[:select_face_num] # debug for 2

        
        # setting prompt with original FaceChain training prompt engineering
        pos_prompt = 'Generate a standard photo of a chinese , beautiful smooth face, smile, high detail face, best quality,'
        add_pos_prompt, add_neg_prompt = get_popular_prompts(instance_data_dir)
        input_prompt = pos_prompt + add_pos_prompt + inpaint_default_positive
        neg_prompt = add_neg_prompt + inpaint_default_negative 
        print("input_prompt :", input_prompt)

        # build inpaint pipeline && some other pilot pipeline
        sd_inpaint_pipeline, generator = build_pipeline_facechain(
            base_model_path, lora_model_path, cache_model_dir, from_safetensor=lora_model_path.endswith('safetensors')
        )    
        print("inpaint pipeline load end")
        retinaface_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
        image_face_fusion = pipeline('face_fusion_torch', model='damo/cv_unet_face_fusion_torch', model_revision='v1.0.3')
        self.openpose = OpenposeDetector.from_pretrained(os.path.join(cache_model_dir, "controlnet_detector"))
        print("preprocess model loaded")

        face_id_image = Image.open(face_id_image_path) 
        # generate in roop
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
                    input_prompt, image=result, mask_image=input_mask, control_image=read_control, strength=first_controlnet_strength, negative_prompt=inpaint_default_negative, 
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

