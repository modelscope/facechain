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
import math

from .utils import is_torch2_available
if is_torch2_available:
    from .face_attention_processor_v1 import FaceAttnProcessor2_0 as FaceAttnProcessor, AttnProcessor2_0 as AttnProcessor, CNAttnProcessor2_0 as CNAttnProcessor
else:
    from .face_attention_processor_v1 import FaceAttnProcessor, AttnProcessor, CNAttnProcessor
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


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )
    
    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

# 1x144x1280 -> 1x16x768
class Face_Prj_Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=16,
        embedding_dim=512,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        
        latents = self.latents.repeat(x.size(0), 1, 1)
        
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        
        # print(latents.shape) # 16,1024
        latents = self.proj_out(latents)
        return self.norm_out(latents)

    
    
class Face_Extracter_v1(nn.Module):
    def __init__(self, fr_weight_path, fc_weight_path):
        super().__init__()

        self.face_transformer = Face_Transformer(weight=fr_weight_path)
        self.face_prj_wofc = Face_Prj_Resampler(dim=1024, depth=4, dim_head=64, heads=12, num_queries=16, embedding_dim=512, output_dim=768, ff_mult=4)
        # now the weight is parameters of both adapter and fc
        weights = torch.load(fc_weight_path)
        # keep fc parameter
        weights_fc = {key.replace('local_fac_prj.', ''): value for key, value in weights.items() if key.startswith('local_fac_prj')}
        self.face_prj_wofc.load_state_dict(weights_fc, strict=True)
        self.face_prj_wofc.eval()

    def forward(self, face_img):

        avr_face_rep, _ = self.face_transformer(face_img)
        face_g_embed = self.face_prj_wofc(avr_face_rep)
        neg_face_g_embed = self.face_prj_wofc(torch.zeros_like(avr_face_rep))
        
        num_ims, seq_len, _ = face_g_embed.shape
        face_g_embed = face_g_embed.view(1, num_ims * seq_len, -1)
        neg_face_g_embed = neg_face_g_embed.view(1, num_ims * seq_len, -1)
        
        return face_g_embed, neg_face_g_embed

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

        
class FaceAdapter_v1:
    
    def __init__(self, sd_pipe, face_detection, segmentation_pipeline, face_extracter, ckpt, device, cfg_face=False):
        
        self.device = device
        self.ckpt = ckpt
        
        # self.face_extracter = Face_Extracter(fr_weight_path='face_adapter/ms1mv2_model_TransFace_S.pt', fc_weight_path='face_adapter/mirror_adapter_20_film.ckpt').to(self.device)
        # self.face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_resnet50_face-detection_retinaface')
        # self.segmentation_pipeline = pipeline(task=Tasks.image_segmentation, model='damo/cv_resnet101_image-multiple-human-parsing')
        self.face_extracter = face_extracter.to(self.device)
        self.face_detection = face_detection
        self.segmentation_pipeline = segmentation_pipeline
        
        self.pipe = sd_pipe.to(self.device)
        self.scale = 1.0
        self.delayed_face_condition = 0.0
        self.cfg_face = cfg_face
        
        self.set_adapter()        
        self.load_adapter()
        
        
    def set_adapter(self):
        unet = self.pipe.unet
        
        layer_norms = {}
        for i in range(3):
            for j in range(2):
                ln = unet.down_blocks[i].attentions[j].transformer_blocks[0].norm2
                layer_norms['down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.processor'.format(i,j)] = ln
                unet.down_blocks[i].attentions[j].transformer_blocks[0].norm2 = Identity()
        
        for i in range(3):
            for j in range(3):
                ln = unet.up_blocks[i+1].attentions[j].transformer_blocks[0].norm2
                layer_norms['up_blocks.{}.attentions.{}.transformer_blocks.0.attn2.processor'.format(i+1,j)] = ln
                unet.up_blocks[i+1].attentions[j].transformer_blocks[0].norm2 = Identity()
        
        ln = unet.mid_block.attentions[0].transformer_blocks[0].norm2
        layer_norms['mid_block.attentions.0.transformer_blocks.0.attn2.processor'] = ln
        unet.mid_block.attentions[0].transformer_blocks[0].norm2 = Identity()
        
        
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
                scale=1.0, ln=layer_norms[name]).to(self.device, dtype=torch.float16)
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
        new_state_dict = {}
        for k in state_dict.keys():
            if 'processor' in k:
                new_state_dict[k] = state_dict[k]
        self.pipe.unet.load_state_dict(new_state_dict, strict=False)
        
    
    def set_scale(self, scale):
        self.scale = scale
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, FaceAttnProcessor):
                attn_processor.scale = scale
    
    def set_num_ims(self, num_ims):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, FaceAttnProcessor):
                attn_processor.num_ims = num_ims
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    for attn_processor in controlnet.attn_processors.values():
                        if isinstance(attn_processor, CNAttnProcessor):
                            attn_processor.num_ims = num_ims
            else:
                for attn_processor in self.pipe.controlnet.attn_processors.values():
                    if isinstance(attn_processor, CNAttnProcessor):
                        attn_processor.num_ims = num_ims
        
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
        delayed_face_condition = self.delayed_face_condition
        
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        print(prompt, negative_prompt, self.scale)
        
        
        # img = torch.rand(1, 3, 112, 112) 
        if isinstance(face_image, Image.Image):
            face_image = [face_image]
            
        num_ims = len(face_image)
        
        self.set_num_ims(num_ims)
        
        resize_images = []
        for face_image_ori in face_image:
            face_image_ori = face_image_preprocess(face_image_ori, self.segmentation_pipeline, self.face_detection)
            # face_image.save('face_image.png')
            face_image_ori = (transforms.PILToTensor()(face_image_ori).float()/255 - 0.5) / 0.5
            resize_images.append(face_image_ori.unsqueeze(0).to(self.device))
        
        
        resize_images = torch.cat(resize_images, dim=0)
        image_prompt_embeds, neg_image_prompt_embeds = self.face_extracter(resize_images)
        
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        
        neg_image_prompt_embeds = neg_image_prompt_embeds.repeat(1, num_samples, 1)
        neg_image_prompt_embeds = neg_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        assert(seq_len == 16 * num_ims)
        

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            if self.cfg_face:
                negative_prompt_embeds = torch.cat([negative_prompt_embeds_, neg_image_prompt_embeds], dim=1)
            else:
                negative_prompt_embeds = torch.cat([negative_prompt_embeds_, image_prompt_embeds], dim=1)
            
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        if delayed_face_condition > 0:
            scale = self.scale
            self.set_scale(0.0)
            latents = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                end_time=delayed_face_condition,
                **kwargs,
            )
            self.set_scale(scale)
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                start_time=delayed_face_condition,
                start_latent=latents,
                **kwargs,
            ).images
        else:
            images = self.pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                **kwargs,
            ).images
        
        return images
    
    
    

