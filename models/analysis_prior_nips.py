#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
# from .basics import *
import math
import torch.nn as nn
import torch


class Analysis_prior_net_nips(nn.Module):
    '''
    Analysis prior net
    '''
    def __init__(self, out_channel_N=192, out_channel_M=192):
        super(Analysis_prior_net_nips, self).__init__()
        self.conv1 = nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 =  nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)