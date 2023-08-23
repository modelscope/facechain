
import math
import os
import argparse
import copy
import cv2
import gc
import numpy as np
import torch

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from PIL import Image





# call_face_crop/crop_and_paste used for inpaint operator to combine template with input image
from PIL import Image
from skimage import transform
import numpy as np

DEFAULT_POSITIVE = 'beautiful, cool, finely detail, light smile, extremely detailed CG unity 8k wallpaper, huge filesize, best quality, realistic, photo-realistic, ultra high res, raw phot, put on makeup'
DEFAULT_NEGATIVE = 'hair, teeth, sketch, duplicate, ugly, huge eyes, text, logo, worst face, strange mouth, nsfw, NSFW, low quality, worst quality, worst quality, low quality, normal quality, lowres, watermark, lowres, monochrome, naked, nude, nsfw, bad anatomy, bad hands, normal quality, grayscale, mural,'


def safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, face_seg, mask_type):
    '''
    Inputs:
        image                   输入图片；
        retinaface_result       retinaface的检测结果；
        crop_ratio              人脸部分裁剪扩充比例；
        face_seg                人脸分割模型；
        mask_type               人脸分割的方式，一个是crop，一个是skin，人脸分割结果是人脸皮肤或者人脸框
    
    Outputs:
        retinaface_box          扩增后相对于原图的box
        retinaface_keypoints    相对于原图的keypoints
        retinaface_mask_pil     人脸分割结果
    '''
    h, w, c = np.shape(image)
    if len(retinaface_result['boxes']) != 0:
        # 获得retinaface的box并且做一手扩增
        retinaface_box      = np.array(retinaface_result['boxes'][0])
        face_width          = retinaface_box[2] - retinaface_box[0]
        face_height         = retinaface_box[3] - retinaface_box[1]
        retinaface_box[0]   = np.clip(np.array(retinaface_box[0], np.int32) - face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[1]   = np.clip(np.array(retinaface_box[1], np.int32) - face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box[2]   = np.clip(np.array(retinaface_box[2], np.int32) + face_width * (crop_ratio - 1) / 2, 0, w - 1)
        retinaface_box[3]   = np.clip(np.array(retinaface_box[3], np.int32) + face_height * (crop_ratio - 1) / 2, 0, h - 1)
        retinaface_box      = np.array(retinaface_box, np.int32)

        # 检测关键点
        retinaface_keypoints = np.reshape(retinaface_result['keypoints'][0], [5, 2])
        retinaface_keypoints = np.array(retinaface_keypoints, np.float32)

        # mask部分
        retinaface_crop     = image.crop(np.int32(retinaface_box))
        retinaface_mask     = np.zeros_like(np.array(image, np.uint8))
        if mask_type == "skin":
            retinaface_sub_mask = face_seg(retinaface_crop)
            retinaface_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = np.expand_dims(retinaface_sub_mask, -1)
        else:
            retinaface_mask[retinaface_box[1]:retinaface_box[3], retinaface_box[0]:retinaface_box[2]] = 255
        retinaface_mask_pil = Image.fromarray(np.uint8(retinaface_mask))
    else:
        retinaface_box          = np.array([])
        retinaface_keypoints    = np.array([])
        retinaface_mask         = np.zeros_like(np.array(image, np.uint8))
        retinaface_mask_pil     = Image.fromarray(np.uint8(retinaface_mask))
        
    return retinaface_box, retinaface_keypoints, retinaface_mask_pil

def crop_and_paste(Source_image, Source_image_mask, Target_image, Source_Five_Point, Target_Five_Point, Source_box):
    '''
    Inputs:
        Source_image            原图像；
        Source_image_mask       原图像人脸的mask比例；
        Target_image            目标模板图像；
        Source_Five_Point       原图像五个人脸关键点；
        Target_Five_Point       目标图像五个人脸关键点；
        Source_box              原图像人脸的坐标；
    
    Outputs:
        output                  贴脸后的人像
    '''
    Source_Five_Point = np.reshape(Source_Five_Point, [5, 2]) - np.array(Source_box[:2])
    Target_Five_Point = np.reshape(Target_Five_Point, [5, 2])

    Crop_Source_image                       = Source_image.crop(np.int32(Source_box))
    Crop_Source_image_mask                  = Source_image_mask.crop(np.int32(Source_box))
    Source_Five_Point, Target_Five_Point    = np.array(Source_Five_Point), np.array(Target_Five_Point)

    tform = transform.SimilarityTransform()
    # 程序直接估算出转换矩阵M
    tform.estimate(Source_Five_Point, Target_Five_Point)
    M = tform.params[0:2, :]

    warped      = cv2.warpAffine(np.array(Crop_Source_image), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)
    warped_mask = cv2.warpAffine(np.array(Crop_Source_image_mask), M, np.shape(Target_image)[:2][::-1], borderValue=0.0)

    mask        = np.float32(warped_mask == 0)
    output      = mask * np.float32(Target_image) + (1 - mask) * np.float32(warped)
    return output

def call_face_crop(retinaface_detection, image, crop_ratio, prefix="tmp"):
    # retinaface检测部分
    # 检测人脸框
    retinaface_result                                           = retinaface_detection(image) 
    # 获取mask与关键点
    retinaface_box, retinaface_keypoints, retinaface_mask_pil   = safe_get_box_mask_keypoints(image, retinaface_result, crop_ratio, None, "crop")

    return retinaface_box, retinaface_keypoints, retinaface_mask_pil


from diffusers import (ControlNetModel,
                    DPMSolverMultistepScheduler,
                    StableDiffusionControlNetInpaintPipeline,
                    )
from controlnet_aux import OpenposeDetector
def build_pipeline_facechain(baseline_model_path, lora_model_path, cache_model_dir, from_safetensor=False):
    from facechain.merge_lora import merge_lora
    
    # Apply to FP16
    weight_dtype = torch.float32

    # Build ControlNet
    controlnet = [
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=weight_dtype, cache_dir=os.path.join(cache_model_dir, "controlnet")),
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=weight_dtype, cache_dir=os.path.join(cache_model_dir, "controlnet")),
    ]
    # openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir=os.path.join(cache_model_dir, "controlnet_detector"))
    
    # Build SDInpaint Pipeline
    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        baseline_model_path,
        controlnet = controlnet, 
        torch_dtype=weight_dtype,
    ).to("cuda")
    pipe = merge_lora(pipeline, lora_model_path, 1.0, from_safetensor=from_safetensor)

    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.enable_sequential_cpu_offload()
    
    # Set Pipeline Scheduler
    pipeline.scheduler  = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    # Set manual seed
    generator           = torch.Generator("cuda").manual_seed(42) 
    return pipeline, generator


class GenPortraitInpaint:
    def __init__(self, 
                 cache_model_dir=None,
                 use_main_model=True, 
                 crop_template = True, 
                 short_side_resize = 768):
        self.crop_template = crop_template 
        self.short_side_resize = short_side_resize


    def __call__(self, base_model_path, 
        lora_model_path, 
        input_template, 
        input_roop_image, 
        input_prompt, 
        cache_model_dir,
        first_controlnet_strength = 0.45,
        second_controlnet_strength = 0.1,
        first_controlnet_conditioning_scale = [0.5,0.3], 
        second_controlnet_conditioning_scale = [0.75,0.75], 
        final_fusion_ratio=0.5,
        use_fusion_before=True, use_fusion_after=True):

        sd_inpaint_pipeline, generator = build_pipeline_facechain(base_model_path, lora_model_path, cache_model_dir, from_safetensor=lora_model_path.endswith('safetensors'))    
        retinaface_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
        image_face_fusion = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo')
        self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir=os.path.join(cache_model_dir, "controlnet_detector"))
        
        #
        template_image =  Image.open(input_template)
        roop_image = Image.open(input_roop_image) 
        face_id_image =  Image.open(input_roop_image) 
        
        

        # crop template to fit sd 
        if self.crop_template:
            # 获取人像坐标并且截取
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
            input_prompt, image=result, mask_image=input_mask, control_image=read_control, strength=first_controlnet_strength, negative_prompt=DEFAULT_NEGATIVE, 
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
        
        # HERE IS THE FINAL OUTPUT
        generate_image = sd_inpaint_pipeline(
            input_prompt, image=input_image_2, mask_image=input_mask, control_image=read_control, strength=second_controlnet_strength, negative_prompt=DEFAULT_NEGATIVE, 
            guidance_scale=9, num_inference_steps=30, generator=generator, height=np.shape(input_image)[0], width=np.shape(input_image)[1], \
            controlnet_conditioning_scale=second_controlnet_conditioning_scale
        ).images[0]

        generate_image.save('debug_result_1.jpg')

        if self.crop_template:
            origin_image    = np.array(copy.deepcopy(template_image))
            x1,y1,x2,y2     = crop_safe_box
            generate_image  = generate_image.resize([x2-x1, y2-y1])
            origin_image[y1:y2,x1:x2] = np.array(generate_image)
            origin_image = Image.fromarray(np.uint8(origin_image))
            # origin_image = Image.fromarray(codeformer_helper.infer(codeFormer_net, face_helper, bg_upsampler, np.array(origin_image)))
            origin_image.save('debug_result_1.jpg')
            generate_image = origin_image

        return generate_image


if __name__=="__main__":


    import  sys

    input_template = sys.argv[1]
    input_roop_image = sys.argv[2]

    cache_model_dir = '/root/photog_dsw/model_data/'

    # paiya
    # base_model_path = '/root/photog_dsw/model_data/ChilloutMix-ni-fp16/'
    lora_model_path = './pai_ya_tmp/zhoumo.safetensors'
    input_prompt = f"zhoumo_face, zhoumo, 1girl," + DEFAULT_POSITIVE


    # facechain 
    # base_model_path = '/mnt/workspace/.cache/modelscope/ly261666/cv_portrait_model/film/film'
    base_model_path = '/mnt/workspace/.cache/modelscope/ly261666/cv_portrait_model/realistic'
    # lora_model_path = './pai_ya_tmp/'
    # input_prompt = f"<sks>, 1girl," + DEFAULT_POSITIVE


    inpaiter = GenPortraitInpaint()
    res = inpaiter(base_model_path=base_model_path, lora_model_path=lora_model_path,  input_template=input_template,
         input_roop_image=input_roop_image, input_prompt=input_prompt, cache_model_dir=cache_model_dir)

    if 0:

        final_fusion_ratio  = 0.5

        # build pipeline openpose/face_detection/image_face_fusion
        sd_inpaint_pipeline, generator = build_pipeline_facechain(base_model_path, lora_model_path, cache_model_dir, from_safetensor=lora_model_path.endswith('safetensors'))    
        openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet", cache_dir=os.path.join(cache_model_dir, "controlnet_detector"))
        retinaface_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
        image_face_fusion = pipeline(Tasks.image_face_fusion, model='damo/cv_unet-image-face-fusion_damo')
        
        #
        template_image =  Image.open(input_template)
        roop_image = Image.open(input_roop_image) 
        face_id_image =  Image.open(input_roop_image) 
        

        # crop template to fit sd 
        crop_template = False
        if crop_template:
            # 获取人像坐标并且截取
            crop_safe_box, _, _ = call_face_crop(retinaface_detection, template_image, 3, "crop")
            input_image = copy.deepcopy(template_image).crop(crop_safe_box)
        else:
            input_image = template_image
        
        if 1:
            # fit template to shortside 768
            short_side  = min(input_image.width, input_image.height)
            resize      = float(short_side / 768)
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
        openpose_image  = openpose(np.array(replaced_input_image, np.uint8))
        canny_image     = cv2.Canny(np.array(result, np.uint8), 100, 200)[:, :, None]
        canny_image     = Image.fromarray(np.concatenate([canny_image, canny_image, canny_image], axis=2))
        read_control    = [openpose_image, canny_image]
        
        #  Fusion as Input, and mask inpaint with ControlNet
        generate_image_old = sd_inpaint_pipeline(
            input_prompt, image=result, mask_image=input_mask, control_image=read_control, strength=0.45, negative_prompt=DEFAULT_NEGATIVE, 
            guidance_scale=9, num_inference_steps=30, generator=generator, height=np.shape(input_image)[0], width=np.shape(input_image)[1], \
            controlnet_conditioning_scale=[0.50, 0.30]
        ).images[0]
        
        use_fusion_after = True
        if use_fusion_after:
            generate_image = image_face_fusion(dict(template=generate_image_old, user=roop_image))[OutputKeys.OUTPUT_IMG]
            generate_image = cv2.cvtColor(generate_image, cv2.COLOR_BGR2RGB)
        else:
            generate_image = generate_image_old
        
        # Prepare for ControlNet
        openpose_image  = openpose(generate_image)
        canny_image     = cv2.Canny(np.array(generate_image, np.uint8), 100, 200)[:, :, None]
        canny_image     = Image.fromarray(np.concatenate([canny_image, canny_image, canny_image], axis=2))
        read_control    = [openpose_image, canny_image]

        # kind of fusion ensemble
        input_image_2   = Image.fromarray(np.uint8((np.array(generate_image_old, np.float32) * (1-final_fusion_ratio) + np.array(generate_image, np.float32) * final_fusion_ratio)))
        
        # HERE IS THE FINAL OUTPUT
        generate_image = sd_inpaint_pipeline(
            input_prompt, image=input_image_2, mask_image=input_mask, control_image=read_control, strength=0.10, negative_prompt=DEFAULT_NEGATIVE, 
            guidance_scale=9, num_inference_steps=30, generator=generator, height=np.shape(input_image)[0], width=np.shape(input_image)[1], \
            controlnet_conditioning_scale=[0.75, 0.75]
        ).images[0]

        generate_image.save('result_1.jpg')

        if crop_template:
            origin_image    = np.array(copy.deepcopy(template_image))
            x1,y1,x2,y2     = crop_safe_box
            generate_image  = generate_image.resize([x2-x1, y2-y1])
            origin_image[y1:y2,x1:x2] = np.array(generate_image)
            origin_image = Image.fromarray(np.uint8(origin_image))
            # origin_image = Image.fromarray(codeformer_helper.infer(codeFormer_net, face_helper, bg_upsampler, np.array(origin_image)))
            origin_image.save('result_2.jpg')

    
