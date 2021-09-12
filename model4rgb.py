import torch
import torch.nn as nn
import math
from models import *
from models.rgb_encoder import *
from models.rgb_decoder import *


class MultiCompression(nn.Module):
    def __init__(self, in_channel1=3, out_channel_N=192, out_channel_M=192):
        super().__init__()
        self.encoder = MultiEncoder(in_channel1=in_channel1, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.decoder = MultiDecoder(out_channel1=in_channel1, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        # rgb
        self.rgbPriorEncoder = Analysis_prior_net_nips(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.rgbPriorDecoder = Synthesis_prior_net_nips(out_channel_N=out_channel_N)
        self.rgbBitEstimator_z = BitEstimator(out_channel_N)
        self.rgbContextPrediction = Context_prediction_net(out_channel_M=out_channel_M)
        self.rgbEntropyParameters = Entropy_parameter_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_rgb):
        # encoder
        rgb_feature = self.encoder(input_rgb)
        batch_size = rgb_feature.size()[0]

        # rgb
        rgb_quant_noise_feature = torch.zeros(input_rgb.size(0), self.out_channel_M, input_rgb.size(2) // 16, input_rgb.size(3) // 16).cuda()
        rgb_quant_noise_z = torch.zeros(input_rgb.size(0), self.out_channel_N, input_rgb.size(2) // 64, input_rgb.size(3) // 64).cuda()
        rgb_quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(rgb_quant_noise_feature), -0.5, 0.5)
        rgb_quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(rgb_quant_noise_z), -0.5, 0.5)
        rgb_z = self.rgbPriorEncoder(rgb_feature)
        if self.training:
            rgb_compressed_z = rgb_z + rgb_quant_noise_z
        else:
            rgb_compressed_z = torch.round(rgb_z)
        rgb_recon_sigma = self.rgbPriorDecoder(rgb_compressed_z)
        rgb_feature_renorm = rgb_feature
        if self.training:
            rgb_compressed_feature_renorm = rgb_feature_renorm + rgb_quant_noise_feature
        else:
            rgb_compressed_feature_renorm = torch.round(rgb_feature_renorm)
        rgb_predict_context = self.rgbContextPrediction(rgb_compressed_feature_renorm)
        rgb_entropy_params = self.rgbEntropyParameters(torch.cat((rgb_recon_sigma, rgb_predict_context), 1))

        # decoder
        rgb_recon_image = self.decoder(rgb_compressed_feature_renorm)

        # rgb
        #rgb_entropy_params = rgb_recon_sigma
        rgb_mu = rgb_entropy_params[:, 0: self.out_channel_M, :, :]
        rgb_sigma = rgb_entropy_params[:, self.out_channel_M: self.out_channel_M * 2, :, :]
        # recon_image = prediction + recon_res
        rgb_clipped_recon_image = rgb_recon_image.clamp(0., 1.)
        # distortion
        rgb_mse_loss = torch.mean((rgb_recon_image - input_rgb).pow(2))

        def feature_probs_based_sigma_nips(feature, mu, sigma):
            sigma = sigma.pow(2)
            sigma = sigma.clamp(1e-5, 1e5)
            gaussian = torch.distributions.normal.Normal(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            return total_bits, probs
            
        def iclr18_estimate_bits_z(rgb_z):
            rgb_prob = self.rgbBitEstimator_z(rgb_z + 0.5) - self.rgbBitEstimator_z(rgb_z - 0.5)
            rgb_total_bits = torch.sum(torch.clamp(-1.0 * torch.log(rgb_prob + 1e-5) / math.log(2.0), 0, 50))
            return rgb_total_bits, rgb_prob

        rgb_total_bits_z, _ = iclr18_estimate_bits_z(rgb_compressed_z)
        
        # rgb
        rgb_total_bits_feature, _ = feature_probs_based_sigma_nips(rgb_compressed_feature_renorm, rgb_mu, rgb_sigma)
        rgb_shape = input_rgb.size()
        rgb_bpp_feature = rgb_total_bits_feature / (batch_size * rgb_shape[2] * rgb_shape[3])
        rgb_bpp_z = rgb_total_bits_z / (batch_size * rgb_shape[2] * rgb_shape[3])
        rgb_bpp = rgb_bpp_feature + rgb_bpp_z

        return rgb_clipped_recon_image, rgb_mse_loss, \
                rgb_bpp_feature, rgb_bpp_z, rgb_bpp