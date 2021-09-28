#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch
from .GDN import GDN

class MultiEncoder(nn.Module):
    def __init__(self, in_channel2=1, out_channel_N=192, out_channel_M=320):
        super().__init__()
        # ir:
        self.ir_conv1 = nn.Conv2d(in_channel2, out_channel_N, 5, stride=1, padding=2)
        self.ir_gdn1 = GDN(out_channel_N)
        self.ir_conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.ir_gdn2 = GDN(out_channel_N)
        self.ir_conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.ir_gdn3 = GDN(out_channel_N)
        self.ir_conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)

    def forward(self, ir):
        ir = self.ir_gdn1(self.ir_conv1(ir))
        ir = self.ir_gdn2(self.ir_conv2(ir))
        ir = self.ir_gdn3(self.ir_conv3(ir))
        ir = self.ir_conv4(ir)
        return ir