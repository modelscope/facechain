import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np





class FaceDownsampler(nn.Module):
    def __init__(self, resize_input=112):
        super().__init__()
        self.resize_input = resize_input

    def forward(self, grounding_extra_input):

        out = torch.nn.functional.interpolate(grounding_extra_input, (self.resize_input, self.resize_input), mode='nearest')

        return out


if __name__ == '__main__':
    img_dir = '/mnt/workspace/haoyu/data/mirror_dataset/sl_data/aligned/919_5.jpg'
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = FaceDownsampler()
    
    feat = net(img)
    print(feat.shape)