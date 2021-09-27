#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch
from .GDN import GDN


class RGBEncoder(nn.Module):
    def __init__(self, in_channel1=3, out_channel_N=192, out_channel_M=320):
        super().__init__()
        # rgb:
        self.feat_conv = nn.Conv2d(in_channel1, 64, 3, stride=2, padding=1)
        self.rgb_conv1 = nn.Conv2d(64, out_channel_N, 5, stride=1, padding=2)
        self.rgb_gdn1 = GDN(out_channel_N)
        self.rgb_conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.rgb_gdn2 = GDN(out_channel_N)
        self.rgb_conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.rgb_gdn3 = GDN(out_channel_N)
        self.rgb_conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)

    def forward(self, rgb, _ir):
        rgb = self.feat_conv(rgb)
        rgb = rgb - _ir
        rgb = self.rgb_gdn1(self.rgb_conv1(rgb))
        rgb = self.rgb_gdn2(self.rgb_conv2(rgb))
        rgb = self.rgb_gdn3(self.rgb_conv3(rgb))
        rgb = self.rgb_conv4(rgb)
        return rgb