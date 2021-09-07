#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from .analysis_prior_nips import Analysis_prior_net_nips
import math
import torch.nn as nn
import torch

class Synthesis_prior_net_nips(nn.Module):
    '''
    Decode synthesis prior
    '''
    def __init__(self, out_channel_N=192):
        super(Synthesis_prior_net_nips, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N + out_channel_N // 2, 5,
                                          stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data,
                                     math.sqrt(2 * (2*out_channel_N+out_channel_N // 2) / (out_channel_N * 2)))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N + out_channel_N // 2, out_channel_N * 2, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data,
                                     (math.sqrt(2 * 1 * (3*out_channel_N+out_channel_N // 2) / (3*out_channel_N))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return self.deconv3(x)
