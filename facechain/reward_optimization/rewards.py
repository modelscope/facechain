"""This file defines customize reward funtions. The reward function which takes in a batch of images
and corresponding prompts returns a batch of rewards each time it is called.
"""

from pathlib import Path
from typing import Callable, List, Union, Tuple
import math
import numpy as np
import torch
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import torchvision
import os
from torch.nn import DataParallel
import cv2
from facechain.reward_optimization.backbones import get_model
import torch.nn.functional as F
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import matplotlib.pyplot as plt
import torchvision.transforms.functional as Ft
from torch import Tensor
from kornia.geometry.transform import crop_and_resize
import facechain.reward_optimization.onnx2pytorch as onnx2pytorch

import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from modelscope import snapshot_download


class FaceCrop(torch.nn.Module):
    def __init__(self, target_size, face_detection_pipeline):
        super().__init__()
        self.target_size = target_size
        self.face_detection_pipeline = face_detection_pipeline

    def forward(self, img):
        '''
        input should be: C, H, W
        '''
        img_pil = _convert_images(img.unsqueeze(0))
        result = self.face_detection_pipeline(img_pil)
        try:
            bbox = result[0]['boxes'][0]
        except:
            bbox = [0,0,img.shape[2],img.shape[1]]
        bbox = list(map(int, bbox))
        j,i,w,h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3] - bbox[1]

        boxes = torch.tensor([[j,i],[j+w,i],[j+w,i+h],[j,i+h]]).unsqueeze(0)
        
        padding = False
        if len(img.shape) == 3: 
            padding = True
            img = img.unsqueeze(0)
        output = crop_and_resize(img.float(), boxes, [self.target_size, self.target_size]).to(img.device, dtype=img.dtype)
        if padding:
            output = output.squeeze(0)

        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
        

def _convert_images(images: Union[List[Image.Image], np.array, torch.Tensor]) -> List[Image.Image]:
    
    if isinstance(images, List) and isinstance(images[0], Image.Image):
        return images
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    images = [Image.fromarray(image) for image in images]

    return images


def face_similarity(target_image_dir: str,
            grad_scale=1,
            device=None,
            accelerator=None,
            torch_dtype=None):
    target_size = 112
    def load_model(model_path, name='r100'):
        net = get_model(name, fp16=True)
        net.load_state_dict(torch.load(model_path))
        net.eval()
        return net.to(device)
    
    retina_face_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')

    face_transform = torchvision.transforms.Compose(
        [FaceCrop(target_size=target_size, face_detection_pipeline=retina_face_detection)])
    def load_image(scorer, img_paths):
        imgs = []
        for idx, img_path in enumerate(img_paths):
            image = cv2.imread(str(img_path))
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image).to(device, dtype=torch.uint8)
            
            img = torch.permute(image, (2, 0, 1))
            try:
                img = face_transform(img).to(torch_dtype)
                img.div_(255.)
                imgs.append(img)
            except:
                pass

        imgs = torch.stack(imgs, dim=0).squeeze().to(torch_dtype)
        feat = scorer(imgs)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat

    # get the path of the current dir
    model_dir = snapshot_download('eavesy/recognition_arcface')
    model_path = os.path.join(model_dir, 'glint360k_cosface_r100_fp16_0.1.pth')
    try:
        scorer = load_model(model_path)
    except:
        raise ImportError("Please download the model first")

    scorer.requires_grad_(False)

    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((target_size,target_size))])

    # get all image files including png format and jpg format in `target_image_dir`
    target_image_files_png = [each for each in Path(target_image_dir).glob("*.png")]
    target_image_files_jpg = [each for each in Path(target_image_dir).glob("*.jpg")]
    target_image_files = target_image_files_png + target_image_files_jpg
    if len(target_image_files) == 0:
        raise ValueError("No image files found in `target_image_dir`")
    target_embs = load_image(scorer, target_image_files)
    assert len(target_embs) != 0, "image dir does not include face images!"
        
    def __call__(im_pix_un, prompts, train=False):
        if train:
            im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        else:
            im_pix = im_pix_un
        imgs = []
        for i in range(im_pix.shape[0]):
            img = face_transform(im_pix[i])
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        feat = scorer(imgs)
        feat = torch.nn.functional.normalize(feat, p=2)
        rewards = torch.matmul(feat, target_embs.T)
        # gumbel softmax trick to make it differentiable
        max_idx = F.gumbel_softmax(rewards, tau=1, hard=True)
        rewards = (rewards*max_idx).sum(dim=1)
        return grad_scale * rewards
    return __call__





class MLPDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )


    def forward(self, embed):
        return self.layers(embed)


class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLPDiff()
        model_dir = snapshot_download('eavesy/aesthetic_mlp')
        model_path = os.path.join(model_dir, 'linear.pth')
        state_dict = torch.load(model_path)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    def __call__(self, images):
        device = next(self.parameters()).device
        embed = self.clip.get_image_features(pixel_values=images)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

def aesthetic(target_image_dir,
            grad_scale=1,
            device=None,
            accelerator=None,
            torch_dtype=None):
    
    target_size = 224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
    scorer = AestheticScorerDiff(dtype=torch_dtype).to(device, dtype=torch_dtype)
    scorer.eval()
    scorer.requires_grad_(False)
    def __call__(im_pix_un, prompts, train=False):
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        im_pix = torchvision.transforms.Resize((target_size,target_size))(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)
        rewards = scorer(im_pix)
        return grad_scale * rewards
    return __call__


def hpsv2(target_image_dir,
            grad_scale=1,
            device=None,
            accelerator=None,
            torch_dtype=None):
    import subprocess
    import sys
    try:
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'hpsv2'])
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        
    def load_model(model_path, name="ViT-H-14"):
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            name,
            'laion2B-s32B-b79K',
            precision=torch_dtype,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )    
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device, dtype=torch_dtype)
        model.eval()
        return model

    target_size = 224

    model_dir = snapshot_download('eavesy/HPSv2_score')
    model_path = os.path.join(model_dir, "HPS_v2_compressed.pt")
    scorer = load_model(model_path)
    tokenizer = get_tokenizer("ViT-H-14")
    scorer.requires_grad_(False)

    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def __call__(im_pix_un, prompts, train=False):
        if train:
            im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        else:
            im_pix = im_pix_un

        im_pix = torchvision.transforms.Resize((target_size,target_size))(im_pix)
        im_pix = normalize(im_pix).to(im_pix_un.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = scorer(im_pix, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)

        rewards = scores
        return grad_scale * rewards

    return __call__