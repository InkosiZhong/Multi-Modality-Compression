import math
from models.context_module import Context_prediction_net, Entropy_parameter_net
from models.bitEstimator import BitEstimator
from models.analysis_prior_nips import Analysis_prior_net_nips
from models.synthesis_prior_nips import Synthesis_prior_net_nips
from models.ir_encoder import MultiEncoder
from models.ir_decoder import MultiDecoder
import torch.nn as nn
import torch

class SFT_layer(nn.Module):
    def __init__(self, in_channel = 64, out_channel_N=192, out_channel_M=192):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel*2, in_channel*2, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channel*2, in_channel*2, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channel*2, in_channel*2, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channel*2, in_channel*2, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.compress = SFTCompression(in_channel*2, out_channel_N, out_channel_M)
        self.gamma_conv = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)
        self.beta_conv = nn.Conv2d(in_channel*2, in_channel, 3, 1, 1)
        self.in_channel = in_channel

    def forward(self, rgb, _ir): # (b, c, h, w)
        b, _, h, w = rgb.shape
        x = self.lrelu(self.conv1(torch.cat([rgb, _ir], dim=1)))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x, mse_loss, bpp_feature, bpp_z, bpp = self.compress(x)
        gamma = self.gamma_conv(x)
        beta = self.beta_conv(x)
        return _ir * gamma + beta, mse_loss, bpp
        #gamma = torch.mean(gamma.view(b, self.in_channel, -1), dim=2, keepdim=True) # (b, c, 1)
        #beta = torch.mean(beta.view(b, self.in_channel, -1), dim=2, keepdim=True) # (b, c, 1)

        #return (_ir.view(b, self.in_channel, -1) * gamma + beta).view(b, -1, h, w)


class SFTCompression(nn.Module):
    def __init__(self, in_channel=128, out_channel_N=192, out_channel_M=192):
        super().__init__()
        self.encoder = MultiEncoder(in_channel2=in_channel, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.decoder = MultiDecoder(out_channel2=in_channel, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net_nips(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net_nips(out_channel_N=out_channel_N)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.contextPrediction = Context_prediction_net(out_channel_M=out_channel_M)
        self.entropyParameters = Entropy_parameter_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)

        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input):
        # encoder
        feature = self.encoder(input)
        batch_size = feature.size()[0]

        quant_noise_feature = torch.zeros(input.size(0), self.out_channel_M, input.size(2) // 8, input.size(3) // 8).cuda()
        quant_noise_z = torch.zeros(input.size(0), self.out_channel_N, input.size(2) // 32, input.size(3) // 32).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        z = self.priorEncoder(feature)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        recon_sigma = self.priorDecoder(compressed_z)
        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        predict_context = self.contextPrediction(compressed_feature_renorm)
        entropy_params = self.entropyParameters(torch.cat((recon_sigma, predict_context), 1))

        # decoder
        recon = self.decoder(compressed_feature_renorm)

        #entropy_params = recon_sigma
        mu = entropy_params[:, 0: self.out_channel_M, :, :]
        sigma = entropy_params[:, self.out_channel_M: self.out_channel_M * 2, :, :]
        # recon = prediction + recon_res
        clipped_recon = recon.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon - input).pow(2))

        def feature_probs_based_sigma_nips(feature, mu, sigma):
            sigma = sigma.pow(2)
            sigma = sigma.clamp(1e-5, 1e5)
            gaussian = torch.distributions.normal.Normal(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
            return total_bits, prob

        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        
        total_bits_feature, _ = feature_probs_based_sigma_nips(compressed_feature_renorm, mu, sigma)
        shape = input.size()
        bpp_feature = total_bits_feature / (batch_size * shape[2] * shape[3])
        bpp_z = total_bits_z / (batch_size * shape[2] * shape[3])
        bpp = bpp_feature + bpp_z

        return clipped_recon, mse_loss, bpp_feature, bpp_z, bpp