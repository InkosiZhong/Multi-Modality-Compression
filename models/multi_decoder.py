#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import torch.nn as nn
import torch
from .GDN import GDN
from .swin_transformer import SwinTransformerAlignment as TAlign

class MultiDecoder(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel1=3, out_channel2=1, out_channel_N=192, out_channel_M=320):
        super().__init__()
        # rgb
        self.rgb_deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.rgb_igdn1 = GDN(out_channel_N, inverse=True)
        self.rgb_deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.rgb_igdn2 = GDN(out_channel_N, inverse=True)
        self.rgb_deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.rgb_igdn3 = GDN(out_channel_N, inverse=True)
        self.rgb_deconv4 = nn.ConvTranspose2d(out_channel_N, out_channel1, 5, stride=2, padding=2, output_padding=1)

        # ir
        self.ir_deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.ir_igdn1 = GDN(out_channel_N, inverse=True)
        self.ir_deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.ir_igdn2 = GDN(out_channel_N, inverse=True)
        self.ir_deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.ir_igdn3 = GDN(out_channel_N, inverse=True)
        self.ir_deconv4 = nn.ConvTranspose2d(out_channel_N, out_channel2, 5, stride=1, padding=2, output_padding=0)

        # ir -> rgb
        self.align1 = TAlign(in_chans=out_channel_N, layers_cfg={2, 3, 2, 8, False})
        self.fusion_conv1 = nn.Conv2d(out_channel_N*2, out_channel_N, 5, 1, 2)
        self.align2 = TAlign(in_chans=out_channel_N, layers_cfg={2, 3, 2, 8, False})
        self.fusion_conv2 = nn.Conv2d(out_channel_N*2, out_channel_N, 5, 1, 2)
        self.align2 = TAlign(in_chans=out_channel_N, layers_cfg={2, 3, 2, 8, False})
        self.fusion_conv3 = nn.Conv2d(out_channel_N*2, out_channel_N, 5, 1, 2)

    def forward(self, rgb, ir):
        rgb = self.rgb_igdn1(self.rgb_deconv1(rgb))
        ir = self.ir_igdn1(self.ir_deconv1(ir))
        ir = self.align1(ir, rgb) # ir -> rgb
        rgb = self.fusion_conv1(torch.cat([rgb, ir], dim=1))

        rgb = self.rgb_igdn2(self.rgb_deconv2(rgb))
        ir = self.ir_igdn2(self.ir_deconv2(ir))
        ir = self.align1(ir, rgb) 
        rgb = self.fusion_conv2(torch.cat([rgb, ir], dim=1))

        rgb = self.rgb_igdn3(self.rgb_deconv3(rgb))
        ir = self.ir_igdn3(self.ir_deconv3(ir))
        ir = self.align1(ir, rgb)
        rgb = self.fusion_conv3(torch.cat([rgb, ir], dim=1))

        rgb = self.rgb_deconv4(rgb)
        ir = self.ir_deconv4(ir) 
        return rgb, ir
