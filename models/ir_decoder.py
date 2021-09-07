#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch
from .GDN import GDN

class MultiDecoder(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel2=1, out_channel_N=192, out_channel_M=320):
        super().__init__()
        # ir
        self.ir_deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.ir_igdn1 = GDN(out_channel_N, inverse=True)
        self.ir_deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.ir_igdn2 = GDN(out_channel_N, inverse=True)
        self.ir_deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.ir_igdn3 = GDN(out_channel_N, inverse=True)
        self.ir_deconv4 = nn.ConvTranspose2d(out_channel_N, out_channel2, 5, stride=1, padding=2, output_padding=0)

    def forward(self, ir):
        ir = self.ir_igdn1(self.ir_deconv1(ir))
        ir = self.ir_igdn2(self.ir_deconv2(ir))
        ir = self.ir_igdn3(self.ir_deconv3(ir))
        ir = self.ir_deconv4(ir) 
        return ir
