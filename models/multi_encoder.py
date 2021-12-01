#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch
from .GDN import GDN
from .basics import FeatureEncoder
from models.SFT_layer import SFT_layer
from .swin_transformer import SwinTransformerAlignment as TAlign

class RGBEncoder(nn.Module):
    def __init__(self, in_channel1=3, out_channel_N=192, out_channel_M=320):
        super().__init__()
        # rgb:
        self.feat_encoder = FeatureEncoder(in_channel1, 64, 2)
        self.sft_layer = SFT_layer(64)
        self.rgb_conv1 = nn.Conv2d(128, out_channel_N, 5, stride=1, padding=2)
        self.rgb_gdn1 = GDN(out_channel_N)
        self.rgb_conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.rgb_gdn2 = GDN(out_channel_N)
        self.rgb_conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.rgb_gdn3 = GDN(out_channel_N)
        self.rgb_conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)

        self.align1 = TAlign(in_chans=out_channel_N, layers_cfg=[2, 3, 2, 8, False])
        self.fusion_conv1 = nn.Conv2d(out_channel_N*2, out_channel_N, 1, 1, 0)
        self.align2 = TAlign(in_chans=out_channel_N, layers_cfg=[2, 3, 2, 8, False])
        self.fusion_conv2 = nn.Conv2d(out_channel_N*2, out_channel_N, 1, 1, 0)
        self.align3 = TAlign(in_chans=out_channel_N, layers_cfg=[2, 3, 2, 8, False])
        self.fusion_conv3 = nn.Conv2d(out_channel_N*2, out_channel_N, 1, 1, 0)

    def forward(self, rgb, irs, _ir):
        rgb = self.feat_encoder(rgb)
        _ir = self.sft_layer(rgb, _ir)
        rgb = torch.cat([rgb, _ir], dim=1)
        
        rgb = self.rgb_gdn1(self.rgb_conv1(rgb))
        rgb = self.fusion_conv1(torch.cat([rgb, self.align1(irs[0], rgb)], dim=1))

        rgb = self.rgb_gdn2(self.rgb_conv2(rgb))
        rgb = self.fusion_conv2(torch.cat([rgb, self.align2(irs[1], rgb)], dim=1))

        rgb = self.rgb_gdn3(self.rgb_conv3(rgb))
        rgb = self.fusion_conv3(torch.cat([rgb, self.align3(irs[2], rgb)], dim=1))

        rgb = self.rgb_conv4(rgb)
        return rgb, _ir