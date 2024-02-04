import torch
from torch import nn
import torch.nn.functional as F
from face_inference import Face_Extractor

class Face_Prj(nn.Module):
    def __init__(self, in_channel, out_channel, in_dim, out_dim, mult, id_batch, face_batch, layer_num=4, glu=True, dropout=0.):
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
        self.channel_merge = nn.Linear(face_batch, face_batch) # 3 3
        self.feature = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_channel)
        )
        self.id_batch = id_batch
        self.face_batch = face_batch

    def forward(self, avr_face_rep): 
        # batch cmt 6[2 * 3] 50 768
        # avr_face_rep = avr_face_rep.reshape(2, 3, 50, 768)
        # permute(0, 2, 3, 1)
        # FC
        # Action: 1. batch speed 2. Atten show, prompt
        # print(avr_face_rep) [3, 144, 512]
        # avr_face_rep = avr_face_rep.view(self.id_batch, self.face_batch, 144, 512)
        # avr_face_rep = avr_face_rep.permute(0, 2, 3, 1)
        # avr_face_rep = self.channel_merge(avr_face_rep)
        # avr_face_rep = avr_face_rep.permute(0, 3, 1, 2)
        # avr_face_rep = avr_face_rep.view(self.id_batch * self.face_batch, 144, 512)

        avr_face_rep = avr_face_rep.permute(2, 1, 0) # [3, 144, 512]
        # print(avr_face_rep.shape)
        # print(self.channel_merge.weight.shape)
        # import pdb
        # pdb.set_trace()
        avr_face_rep = self.channel_merge(avr_face_rep)
        avr_face_rep = avr_face_rep.permute(2, 1, 0)

        avr_face_rep = avr_face_rep.permute(0, 2, 1)
        face_g_embed = self.net(avr_face_rep)
        face_g_embed = face_g_embed.permute(0, 2, 1)
        face_g_embed = self.feature(face_g_embed)
        return face_g_embed

class Face_Prj_wofc(nn.Module):
    def __init__(self, in_channel, out_channel, in_dim, out_dim, mult, id_batch, face_batch, layer_num=4, glu=True, dropout=0.):
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
        self.id_batch = id_batch
        self.face_batch = face_batch

    def forward(self, avr_face_rep): 

        avr_face_rep = avr_face_rep.permute(0, 2, 1)
        face_g_embed = self.net(avr_face_rep)
        face_g_embed = face_g_embed.permute(0, 2, 1)
        face_g_embed = self.feature(face_g_embed)
        return face_g_embed


def face_merge(txt_embed, face_g_embed):
    assert txt_embed.shape[-1] == face_g_embed.shape[-1], 'dim of text embedding is {}, and dim of global face embeddimg is {}, they are not aligned'.format(txt_embed.shape, face_g_embed.shape)
    txt_embed[:, txt_embed.shape[1] - face_g_embed.shape[1] : , :] = face_g_embed

if __name__ == '__main__':
    fac_ext = Face_Extractor()
    global_prj = Face_Prj(1, 4, 512, 768, 1)

    img = torch.rand(3, 3, 112, 112)
    local_rep, global_rep = fac_ext(img) # [3, 144, 512] [3, 512]

    avr_face_rep = torch.mean(global_rep, dim=0, keepdim=True).repeat(3, 1).unsqueeze(1) # [3, 1, 512]

    txt_embed = torch.zeros(3, 77, 768)
    face_embed = global_prj(avr_face_rep) # [3, 4, 768]
    face_merge(txt_embed, face_embed)

    local_prj = Face_Prj(144, 196, 512, 768, 1)
    rich_rep = local_prj(local_rep)

