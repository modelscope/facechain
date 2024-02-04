from abc import abstractmethod
from functools import partial
import math

import numpy as np
import random
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.attention import SpatialTransformer
# from .positionnet  import PositionNet
from torch.utils import checkpoint
from ldm.util import instantiate_from_config
from copy import deepcopy

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context, objs):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, objs)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x




class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # return checkpoint(
        #     self._forward, (x, emb), self.parameters(), self.use_checkpoint
        # )
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, emb )
        else:
            return self._forward(x, emb) 


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h




class UNetModel(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=8,
        use_scale_shift_norm=False,
        transformer_depth=1,           
        context_dim=None,  
        fuser_type = None,
        inpaint_mode = False,
        grounding_downsampler = None,
        local_fac_prj = None,
        global_fac_prj = None,
        face_prob = 0.1
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.context_dim = context_dim
        self.fuser_type = fuser_type
        self.inpaint_mode = inpaint_mode
        self.face_prob = face_prob
        assert fuser_type in ["gatedSA","gatedSA2","gatedCA"]

        self.local_face_tokenizer_input = None # set externally 


        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )



        self.downsample_net = None 
        self.additional_channel_from_downsampler = 0
        self.first_conv_type = "SD"
        self.first_conv_restorable = True 
        if grounding_downsampler is not None:
            self.downsample_net = instantiate_from_config(grounding_downsampler)  
            self.additional_channel_from_downsampler = self.downsample_net.out_dim
            self.first_conv_type = "GLIGEN"

        if inpaint_mode:
            # The new added channels are: masked image (encoded image) and mask, which is 4+1
            in_c = in_channels+self.additional_channel_from_downsampler+in_channels+1
            self.first_conv_restorable = False # in inpaint; You must use extra channels to take in masked real image  
        else:
            in_c = in_channels+self.additional_channel_from_downsampler
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_c, model_channels, 3, padding=1))])


        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        # = = = = = = = = = = = = = = = = = = = = Down Branch = = = = = = = = = = = = = = = = = = = = #
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ ResBlock(ch,
                                    time_embed_dim,
                                    dropout,
                                    out_channels=mult * model_channels,
                                    dims=dims,
                                    use_checkpoint=use_checkpoint,
                                    use_scale_shift_norm=use_scale_shift_norm,) ]

                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(SpatialTransformer(ch, key_dim=context_dim, value_dim=context_dim, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, fuser_type=fuser_type, use_checkpoint=use_checkpoint))
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1: # will not go to this downsample branch in the last feature
                out_ch = ch
                self.input_blocks.append( TimestepEmbedSequential( Downsample(ch, conv_resample, dims=dims, out_channels=out_ch ) ) )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
        dim_head = ch // num_heads

        # self.input_blocks = [ C |  RT  RT  D  |  RT  RT  D  |  RT  RT  D  |   R  R   ]


        # = = = = = = = = = = = = = = = = = = = = BottleNeck = = = = = = = = = = = = = = = = = = = = #
        
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            SpatialTransformer(ch, key_dim=context_dim, value_dim=context_dim, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, fuser_type=fuser_type, use_checkpoint=use_checkpoint),
            ResBlock(ch,
                     time_embed_dim,
                     dropout,
                     dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm))



        # = = = = = = = = = = = = = = = = = = = = Up Branch = = = = = = = = = = = = = = = = = = = = #

        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ ResBlock(ch + ich,
                                    time_embed_dim,
                                    dropout,
                                    out_channels=model_channels * mult,
                                    dims=dims,
                                    use_checkpoint=use_checkpoint,
                                    use_scale_shift_norm=use_scale_shift_norm) ]
                ch = model_channels * mult
                
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append( SpatialTransformer(ch, key_dim=context_dim, value_dim=context_dim, n_heads=num_heads, d_head=dim_head, depth=transformer_depth, fuser_type=fuser_type, use_checkpoint=use_checkpoint) )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append( Upsample(ch, conv_resample, dims=dims, out_channels=out_ch) )
                    ds //= 2
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))


        # self.output_blocks = [ R  R  RU | RT  RT  RTU |  RT  RT  RTU  |  RT  RT  RT  ]


        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        self.device = self.input_blocks[0][0].weight.device
        self.local_fac_prj = instantiate_from_config(local_fac_prj).to(self.device)
        # self.global_fac_prj = instantiate_from_config(global_fac_prj).to(self.device)
        

    def restore_first_conv_from_SD(self):
        if self.first_conv_restorable:
            device = self.input_blocks[0][0].weight.device

            SD_weights = th.load("SD_input_conv_weight_bias.pth")
            self.GLIGEN_first_conv_state_dict = deepcopy(self.input_blocks[0][0].state_dict())

            self.input_blocks[0][0] = conv_nd(2, 4, 320, 3, padding=1)
            self.input_blocks[0][0].load_state_dict(SD_weights)
            self.input_blocks[0][0].to(device)

            self.first_conv_type = "SD"
        else:
            print("First conv layer is not restorable and skipped this process, probably because this is an inpainting model?")


    def restore_first_conv_from_GLIGEN(self):
        breakpoint() # TODO 


    def forward(self, input):

        from torch.distributed import get_rank
        if ('local_fac_rep' in input):
            local_fac_rep = input['local_fac_rep']
        else:
            # Guidance null case
            local_fac_rep = self.local_face_tokenizer_input.get_null_input() # [3, 144, 512]
        if self.training and random.random() < self.face_prob and self.local_face_tokenizer_input.set: # random drop for guidance
            local_fac_rep = self.local_face_tokenizer_input.get_null_input()


        
        objs = self.local_fac_prj(local_fac_rep) # [3, 196, 768]
        
        # Time embedding 
        t_emb = timestep_embedding(input["timesteps"], self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # input tensor  
        h = input["x"]
        
        # Text input 
        context = input["context"]
        from rep_merge import face_merge

        # Start forwarding 
        hs = []
        for module in self.input_blocks:
            h = module(h, emb, context, objs)
            hs.append(h)

        h = self.middle_block(h, emb, context, objs)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, objs)

        return self.out(h)










