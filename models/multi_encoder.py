#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import torch.nn as nn
import torch
from .GDN import GDN


class MultiEncoder(nn.Module):
    def __init__(self, in_channel1=3, in_channel2=1, out_channel_N=192, out_channel_M=320):
        super().__init__()
        # rgb:
        self.rgb_conv1 = nn.Conv2d(in_channel1, out_channel_N, 5, stride=2, padding=2)
        self.rgb_gdn1 = GDN(out_channel_N)
        self.rgb_conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.rgb_gdn2 = GDN(out_channel_N)
        self.rgb_conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.rgb_gdn3 = GDN(out_channel_N)
        self.rgb_conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
            
        # ir:
        self.ir_conv1 = nn.Conv2d(in_channel2, out_channel_N, 5, stride=1, padding=2)
        self.ir_gdn1 = GDN(out_channel_N)
        self.ir_conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.ir_gdn2 = GDN(out_channel_N)
        self.ir_conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.ir_gdn3 = GDN(out_channel_N)
        self.ir_conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)

        # ir -> rgb
        self.fusion_conv1 = nn.Conv2d(out_channel_N*2, out_channel_N, 5, 1, 2)
        self.fusion_conv2 = nn.Conv2d(out_channel_N*2, out_channel_N, 5, 1, 2)
        self.fusion_conv3 = nn.Conv2d(out_channel_N*2, out_channel_N, 5, 1, 2)

    def forward(self, rgb, ir):
        rgb = self.rgb_gdn1(self.rgb_conv1(rgb))
        ir = self.ir_gdn1(self.ir_conv1(ir))
        rgb = self.fusion_conv1(torch.cat([rgb, ir], dim=1))

        rgb = self.rgb_gdn2(self.rgb_conv2(rgb))
        ir = self.ir_gdn2(self.ir_conv2(ir))
        rgb = self.fusion_conv2(torch.cat([rgb, ir], dim=1))

        rgb = self.rgb_gdn3(self.rgb_conv3(rgb))
        ir = self.ir_gdn3(self.ir_conv3(ir))
        rgb = self.fusion_conv3(torch.cat([rgb, ir], dim=1))

        rgb = self.rgb_conv4(rgb)
        ir = self.ir_conv4(ir)
        return rgb, ir