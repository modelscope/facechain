import argparse

import cv2
import numpy as np
import torch

from backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5) ## Pre-processing
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat, weight, local_patch_entropy = net(img).numpy()
    print(feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    #parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--network', type=str, default='vit_s_dp005_mask_0', help='backbone network')
    parser.add_argument('--weight', type=str, default='') ## load pre-trained model
    parser.add_argument('--img', type=str, default=None)  ## load img path
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)
