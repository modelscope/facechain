import cv2
import numpy as np
import torch
from torch import nn
from backbones import vit
from torch.distributed import get_rank
class Face_Extractor(nn.Module):
    def __init__(self, name='vits', weight='./pretrained_model/ms1mv2_model_TransFace_S.pt'):
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

class Face_Transformer(nn.Module):
    def __init__(self, name='vits', weight='./pretrained_model/ms1mv2_model_TransFace_S.pt'):
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
    def __init__(self, in_channel, out_channel, in_dim, out_dim, mult, layer_num=4, glu=True, dropout=0.):
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
    def __init__(self, fc_weight_path):
        super().__init__()

        self.face_transformer = Face_Transformer()
        self.face_prj_wofc = Face_Prj_wofc(in_channel=144, out_channel=196, in_dim=512, out_dim=768, mult=1, )
        # now the weight is parameters of both adapter and fc, so set strict=False
        weights = torch.load(fc_weight_path)
        weights_fc = {key.replace('local_fac_prj.', ''): value for key, value in weights.items() if key.startswith('local_fac_prj')}
        self.face_prj_wofc.load_state_dict(weights_fc, strict=True)
        self.face_prj_wofc.eval()

    def forward(self, face_img):

        avr_face_rep, _ = self.face_transformer(face_img)
        face_g_embed = self.face_prj_wofc(avr_face_rep)
        # avr_face_rep = avr_face_rep.permute(0, 2, 1)
        # face_g_embed = self.net(avr_face_rep)
        # face_g_embed = face_g_embed.permute(0, 2, 1)
        # face_g_embed = self.feature(face_g_embed)
        return face_g_embed





if __name__ == '__main__':
    
    face_extracter = Face_Extracter(fc_weight_path='./mirror_adapter_20_film.ckpt') 
    img = torch.rand(1, 3, 112, 112) 
    face_emb = face_extracter(img)
    print(face_emb.shape)
