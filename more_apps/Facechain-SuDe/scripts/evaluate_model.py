import argparse, os, sys, glob
from pytorch_lightning import seed_everything
sys.path.append(os.path.join(sys.path[0], '..'))

import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.personalized import PersonalizedBase
from evaluation.clip_eval import LDMCLIPEvaluator
import subprocess
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model



def evaluate_model_func(temp_prompt_adj, prompt, ckpt_path, data_dir, output_path, not_gen=False):
    # model.embedding_manager.load(opt.embedding_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    evaluator = LDMCLIPEvaluator(device)
    # prompt = opt.prompt
    data_loader = PersonalizedBase(data_dir, size=512, flip_p=0.0)
    images = [torch.from_numpy(data_loader[i]["image"]).permute(2, 0, 1) for i in range(data_loader.num_images)]
    images = torch.stack(images, axis=0)

    if not_gen:
        data_loader_gen = PersonalizedBase(os.path.join(output_path, prompt), size=512, flip_p=0.0)
        images_gen = [torch.from_numpy(data_loader_gen[i]["image"]).permute(2, 0, 1) for i in range(data_loader_gen.num_images)]
        images_gen = torch.stack(images_gen, axis=0)
        sim_img, sim_text, sim_samples_to_text_adj = evaluator.evaluate_not_gen(images, prompt, temp_prompt_adj, images_gen)
    else:
        seed_everything(2023)
        config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval_with_tokens.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
        model = load_model_from_config(config, ckpt_path)  # TODO: check path
        model = model.to(device)
        sim_img, sim_text, sim_samples_to_text_adj = evaluator.evaluate(model, images, prompt, temp_prompt_adj, output_path, n_samples=4)
    
    
    
    return sim_img, sim_text, sim_samples_to_text_adj



def extract_all_images(images, model, datasetclass, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        datasetclass(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['image'].to(device)
            if hasattr(model, 'encode_image'):
                if device == 'cuda':
                    b = b.to(torch.float16)
                all_image_features.append(model.encode_image(b).cpu().numpy())
            else:
                all_image_features.append(model(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features



def dinoeval_image(image_dir, image_dir_ref, device):
    image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir)
                   if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]
    image_paths_ref = [os.path.join(image_dir_ref, path) for path in os.listdir(image_dir_ref)
                       if path.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.JPG'))]

    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, DINOImageDataset, device, batch_size=64, num_workers=8)

    image_feats_ref = extract_all_images(
        image_paths_ref, model, DINOImageDataset, device, batch_size=64, num_workers=8)

    image_feats = image_feats / \
        np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / \
        np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T
    return np.mean(res)


def Convert(image):
    return image.convert("RGB")


class DINOImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(256, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            Convert,
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)








# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--prompt",
#         type=str,
#         nargs="?",
#         default="a painting of a virus monster playing guitar",
#         help="the prompt to render"
#     )

#     parser.add_argument(
#         "--ckpt_path", 
#         type=str, 
#         default="/data/pretrained_models/ldm/text2img-large/model.ckpt", 
#         help="Path to pretrained ldm text2img model")

#     # parser.add_argument(
#     #     "--embedding_path", 
#     #     type=str, 
#     #     help="Path to a pre-trained embedding manager checkpoint")

#     parser.add_argument(
#         "--data_dir",
#         type=str,
#         help="Path to directory with images used to train the embedding vectors"
#     )

#     opt = parser.parse_args()


#     config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval_with_tokens.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
#     model = load_model_from_config(config, opt.ckpt_path)  # TODO: check path
#     # model.embedding_manager.load(opt.embedding_path)

#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     model = model.to(device)

#     evaluator = LDMCLIPEvaluator(device)

#     prompt = opt.prompt

#     data_loader = PersonalizedBase(opt.data_dir, size=512, flip_p=0.0)

#     images = [torch.from_numpy(data_loader[i]["image"]).permute(2, 0, 1) for i in range(data_loader.num_images)]
#     images = torch.stack(images, axis=0)

#     sim_img, sim_text = evaluator.evaluate(model, images, opt.prompt, n_samples=8)
    
#     print("Image similarity: ", sim_img)
#     print("Text similarity: ", sim_text)

    