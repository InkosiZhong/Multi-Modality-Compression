import torch.nn as nn
import torch

class SFT_layer(nn.Module):
    def __init__(self, in_channel = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel*2, in_channel*2, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channel*2, in_channel*2, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channel*2, in_channel*2, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channel*2, in_channel*2, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.gamma_conv = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)
        self.beta_conv = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)
        self.in_channel = in_channel

    def forward(self, rgb, _ir): # (b, c, h, w)
        b, _, h, w = rgb.shape
        x = self.lrelu(self.conv1(torch.cat([rgb, _ir], dim=1)))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        gamma = self.gamma_conv(x)
        beta = self.beta_conv(x)
        gamma = torch.mean(gamma.view(b, self.in_channel, -1), dim=2, keepdim=True) # (b, c, 1)
        beta = torch.mean(beta.view(b, self.in_channel, -1), dim=2, keepdim=True) # (b, c, 1)

        return (_ir.view(b, self.in_channel, -1) * gamma + beta).view(b, -1, h, w)