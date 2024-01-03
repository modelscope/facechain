import torch
from backbones import get_model
import numpy as np
from PIL import Image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import torchvision
from typing import Callable, List, Union, Tuple
from pathlib import Path


def _convert_images(images: Union[List[Image.Image], np.array, torch.Tensor]) -> List[Image.Image]:
    
    if isinstance(images, List) and isinstance(images[0], Image.Image):
        return images
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    images = [Image.fromarray(image) for image in images]

    return images

class RewardModel(torch.nn.Module):

    def __init__(self, target_image_dir: str,
                grad_scale=1,
                device=None,
                accelerator=None,
                torch_dtype=None):

        model_path='/mnt/workspace/zanghongyu.zhy/aigc/face/insightface/recognition/arcface_torch/glint360k_cosface_r100_fp16_0.1/backbone.pth'
        self.feat_extractor = get_model('r100',  fp16=True)
        self.feat_extractor.load_state_dict(torch.load(model_path))
        self.feat_extractor.eval()
        self.feat_extractor.requires_grad_(False)

        self.grad_scale = grad_scale
        self.device = device
        self.accelerator = accelerator
        self.torch_dtype = torch_dtype

        retina_face_detection = pipeline(Tasks.face_detection, 'damo/cv_resnet50_face-detection_retinaface')
        self.face_transform = torchvision.transforms.Compose(
            [FaceCrop(target_size=112, face_detection_pipeline=retina_face_detection)])

        target_image_files = Path(target_image_dir).glob("*.png")
        self.target_embs = self.load_image(target_image_files)
        assert len(self.target_embs) != 0, "image dir is empty!"
        
        emb_dim = self.target_embs.shape[1]
        self.linear = torch.nn.Linear(emb_dim*2, 1, bias=False)


    def load_image(self, img_paths):
        imgs = []
        for idx, img_path in enumerate(img_paths):
            image = cv2.imread(str(img_path))
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image).to(self.device, dtype=torch.uint8)
            
            img = torch.permute(image, (2, 0, 1))
            try:
                img = face_transform(img).to(self.torch_dtype)
                img.div_(255.)
                imgs.append(img)
            except:
                pass

        imgs = torch.stack(imgs, dim=0).squeeze().to(self.torch_dtype)
        feat = self.feature_extractor(imgs)
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat


        
    def forward(self, im_pix_un, prompts, train=False):
        if train:
            im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1) 
        else:
            im_pix = im_pix_un
        imgs = []
        for i in range(im_pix.shape[0]):
            img = face_transform(im_pix[i])
            img_pil = _convert_images(img.unsqueeze(0))
            img_pil[0].save(f'compare/{i}.png')

            imgs.append(img)
        imgs = torch.stack(imgs, dim=0)
        feat = scorer(imgs)
        feat = torch.nn.functional.normalize(feat, p=2)
        rewards = torch.matmul(feat, self.target_embs.T)
        max_idx = F.gumbel_softmax(rewards, tau=1, hard=True)
        rewards = (rewards*max_idx).sum(dim=1)
        return self.grad_scale * rewards



class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 1)
    def forward(self, x):
        return self.linear(x)

if __name__=='__main__':
    a = A()
    inp = torch.randn(5, 10,256)
    out = a(inp)
    print(out.shape)