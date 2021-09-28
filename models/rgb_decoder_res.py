#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import torch.nn as nn
from .GDN import GDN
from .basics import FeatureDecoder

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
        self.feat_decoder = FeatureDecoder(out_channel1, 64, 2)

    def forward(self, rgb, _ir):
        rgb = self.rgb_igdn1(self.rgb_deconv1(rgb))
        rgb = self.rgb_igdn2(self.rgb_deconv2(rgb))
        rgb = self.rgb_igdn3(self.rgb_deconv3(rgb))
        rgb = self.rgb_deconv4(rgb)
        rgb = self.feat_decoder(_ir + rgb)
        return rgb
