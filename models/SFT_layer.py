import torch.nn as nn
import torch

class SFT_layer(nn.modules):
    '''
    (rgb, _ir) -> conv -> conv -> (gamma, beta)
           |                         | *    | +
            ---------------------> _ir1 -> _ir2
    '''
    def __init__(self, in_channel = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(2*in_channel, 2*in_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(2*in_channel, 2*in_channel, 3, 1, 1)
        self.in_channel = in_channel

    def forward(self, rgb, _ir): # (b, c, h, w)
        gamma_beta = self.conv2(self.conv1(torch.cat([rgb, _ir], dim=1))) # (b, 2c, h, w)
        gamma = gamma_beta[:, :self.in_channel, :, :]
        beta = gamma_beta[:, self.in_channel: 2*self.in_channel, :, :]
        return _ir * gamma + beta