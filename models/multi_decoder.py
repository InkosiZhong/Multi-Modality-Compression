#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import torch.nn as nn
import torch
from .GDN import GDN
from .basics import FeatureDecoder
from .swin_transformer import SwinTransformerAlignment as TAlign

class RGBDecoder(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel1=3, out_channel_N=192, out_channel_M=320):
        super().__init__()
        # rgb
        self.rgb_deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.rgb_igdn1 = GDN(out_channel_N, inverse=True)
        self.rgb_deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.rgb_igdn2 = GDN(out_channel_N, inverse=True)
        self.rgb_deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.rgb_igdn3 = GDN(out_channel_N, inverse=True)
        self.rgb_deconv4 = nn.ConvTranspose2d(out_channel_N, 64, 5, stride=1, padding=2, output_padding=0)
        self.feat_decoder = FeatureDecoder(out_channel1, 128, 2)

        self.align1 = TAlign(in_chans=out_channel_N, layers_cfg=[2, 3, 2, 8, False])
        self.fusion_conv1 = nn.Conv2d(out_channel_N*2, out_channel_N, 1, 1, 0)
        self.align2 = TAlign(in_chans=out_channel_N, layers_cfg=[2, 3, 2, 8, False])
        self.fusion_conv2 = nn.Conv2d(out_channel_N*2, out_channel_N, 1, 1, 0)
        self.align3 = TAlign(in_chans=out_channel_N, layers_cfg=[2, 3, 2, 8, False])
        self.fusion_conv3 = nn.Conv2d(out_channel_N*2, out_channel_N, 1, 1, 0)

    def forward(self, rgb, irs, _ir):
        rgb = self.rgb_igdn1(self.rgb_deconv1(rgb))
        rgb = self.fusion_conv1(torch.cat([rgb, self.align1(irs[0], rgb)], dim=1))

        rgb = self.rgb_igdn2(self.rgb_deconv2(rgb))
        rgb = self.fusion_conv2(torch.cat([rgb, self.align2(irs[1], rgb)], dim=1))

        rgb = self.rgb_igdn3(self.rgb_deconv3(rgb))
        rgb = self.fusion_conv3(torch.cat([rgb, self.align3(irs[2], rgb)], dim=1))

        rgb = self.rgb_deconv4(rgb)
        rgb = torch.cat([rgb, _ir], dim=1)
        rgb = self.feat_decoder(rgb)
        return rgb
