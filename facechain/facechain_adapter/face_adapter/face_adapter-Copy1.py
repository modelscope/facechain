import os
from typing import List

import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image

from torch.distributed import get_rank
import torch.nn as nn
import numpy as np
from torchvision.transforms import transforms

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2

from .face_attention_processor import FaceAttnProcessor, AttnProcessor, CNAttnProcessor
from . import vit
from . import face_preprocess

def detect(image, face_detection):
    result_det = face_detection(image)
    confs = result_det['scores']
    idx = np.argmax(confs)
    pts = result_det['keypoints'][idx]
    points_vec = np.array(pts)
    points_vec = points_vec.reshape(5,2)
    return points_vec

def get_mask_head(result):
    masks = result['masks']
    scores = result['scores']
    labels = result['labels']
    img_shape = masks[0].shape
    mask_hair = np.zeros(img_shape)
    mask_face = np.zeros(img_shape)
    mask_human = np.zeros(img_shape)
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
    mask_head = np.clip(mask_hair + mask_face, 0, 1)
    ksize = max(int(np.sqrt(np.sum(mask_face)) / 20), 1)
    kernel = np.ones((ksize, ksize))
    mask_head = cv2.dilate(mask_head, kernel, iterations=1) * mask_human
    _, mask_head = cv2.threshold((mask_head * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask_head, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    mask_head = np.zeros(img_shape).astype(np.uint8)
    cv2.fillPoly(mask_head, [contours[max_idx]], 255)
    mask_head = mask_head.astype(np.float32) / 255 
    mask_head = np.clip(mask_head + mask_face, 0, 1)
    mask_head = np.expand_dims(mask_head, 2)
    return mask_head

def align(image, points_vec):
    warped = face_preprocess.preprocess(np.array(image)[:,:,::-1], bbox=None, landmark=points_vec, image_size='112, 112')
    return Image.fromarray(warped[:,:,::-1])

def face_image_preprocess(image, segmentation_pipeline, face_detection):
    result = segmentation_pipeline(image)
    mask_head = get_mask_head(result)
    image = Image.fromarray((np.array(image) * mask_head).astype(np.uint8))
    points_vec = detect(image, face_detection)
    image = align(image, points_vec)    
    return image

class Face_Transformer(nn.Module):
    def __init__(self, name='vits', weight='./ms1mv2_model_TransFace_S.pt'):
        super().__init__()

        # FR transformer
        self.net = vit.VisionTransformer(img_size=112, 
                                         patch_size=9, 
                                         num_classes=512, 
                                         embed_dim=512, 
                                         depth=12,
                                         num_heads=8, 
                                         drop_path_rate=0.1, 
                                         norm_layer="ln", 
                                         mask_ratio=0.1)
        self.net.load_state_dict(torch.load(weight, map_location='cpu'))
        self.net.eval()



    @torch.no_grad()                                    
    def forward(self, img):
        local_fac_rep, global_fac_rep = self.net(img)
        global_fac_rep = global_fac_rep.unsqueeze(1)
        return local_fac_rep, global_fac_rep


    
class Face_Prj_wofc(nn.Module):
    def __init__(self, in_channel, out_channel, in_dim, out_dim, mult, layer_num=4, glu=True, dropout=0.0):
        super().__init__()

        layers = nn.Sequential()
        inner_channel = in_channel * mult
        for i in range(layer_num):
            layers.append(nn.Sequential(
                nn.Linear(in_channel if i==0 else inner_channel, inner_channel if i!=(layer_num-1) else out_channel),
                nn.ReLU() if not glu else nn.GELU()
            ))
        layers = nn.Linear(in_channel, out_channel)
        self.net = layers
        self.feature = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, avr_face_rep):


        avr_face_rep = avr_face_rep.permute(0, 2, 1)
        face_g_embed = self.net(avr_face_rep)
        face_g_embed = face_g_embed.permute(0, 2, 1)
        face_g_embed = self.feature(face_g_embed)
        return face_g_embed

    
    
class Face_Extracter(nn.Module):
    def __init__(self, fr_weight_path, fc_weight_path):
        super().__init__()

        self.face_transformer = Face_Transformer(weight=fr_weight_path)
        self.face_prj_wofc = Face_Prj_wofc(in_channel=144, out_channel=196, in_dim=512, out_dim=768, mult=1, glu=True, dropout=0.0)
        # now the weight is parameters of both adapter and fc
        weights = torch.load(fc_weight_path)
        # keep fc parameter
        weights_fc = {key.replace('local_fac_prj.', ''): value for key, value in weights.items() if key.startswith('local_fac_prj')}
        self.face_prj_wofc.load_state_dict(weights_fc, strict=True)
        self.face_prj_wofc.eval()

    def forward(self, face_img):

        avr_face_rep, _ = self.face_transformer(face_img)
        face_g_embed = self.face_prj_wofc(avr_face_rep)
        return face_g_embed


        
class FaceAdapter:
    
    def __init__(self, sd_pipe, face_detection, segmentation_pipeline, face_extracter, ckpt, device):
        
        self.device = device
        self.ckpt = ckpt
        
        # self.face_extracter = Face_Extracter(fr_weight_path='face_adapter/ms1mv2_model_TransFace_S.pt', fc_weight_path='face_adapter/mirror_adapter_20_film.ckpt').to(self.device)
        # self.face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_resnet50_face-detection_retinaface')
        # self.segmentation_pipeline = pipeline(task=Tasks.image_segmentation, model='damo/cv_resnet101_image-multiple-human-parsing')
        self.face_extracter = face_extracter.to(self.device)
        self.face_detection = face_detection
        self.segmentation_pipeline = segmentation_pipeline
        
        self.pipe = sd_pipe.to(self.device)
        
        self.set_adapter()        
        self.load_adapter()
        
        
    def set_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = FaceAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                scale=1.0).to(self.device, dtype=torch.float32)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor())
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor())
        
    def load_adapter(self):
        state_dict = torch.load(self.ckpt, map_location="cpu")
        # crossattn_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        self.pipe.unet.load_state_dict(state_dict, strict=False)
        
    
    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, FaceAttnProcessor):
                attn_processor.scale = scale
        
    def generate(
        self,
        face_image=None,
        prompt=None,
        negative_prompt=None,
        num_samples=1,
        seed=None,
        guidance_scale=5.0,
        num_inference_steps=50,
        **kwargs,
    ):
        # self.set_scale(scale)
        
        num_prompts = 1
        
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        
        # img = torch.rand(1, 3, 112, 112) 
        face_image = face_image_preprocess(face_image, self.segmentation_pipeline, self.face_detection)
        face_image.save('face_image.png')
        face_image = (transforms.PILToTensor()(face_image).float()/255 - 0.5) / 0.5
        face_image = face_image.unsqueeze(0).to(self.device)
        image_prompt_embeds = self.face_extracter(face_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        assert(seq_len == 196)
        

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, image_prompt_embeds], dim=1)
            
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
        
        return images
    
    
    

