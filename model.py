import torch
import torch.nn as nn
import math
from models import *
from models.multi_encoder import *
from models.multi_decoder import *


class MultiCompression(nn.Module):
    def __init__(self, in_channel1=3, in_channel2=1, out_channel_N=192, out_channel_M=192):
        super().__init__()
        self.encoder = MultiEncoder(in_channel1=in_channel1, in_channel2=in_channel2, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.decoder = MultiDecoder(out_channel1=in_channel1, out_channel2=in_channel2, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        # rgb
        self.rgbPriorEncoder = Analysis_prior_net_nips(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.rgbPriorDecoder = Synthesis_prior_net_nips(out_channel_N=out_channel_N)
        self.rgbBitEstimator_z = BitEstimator(out_channel_N)
        self.rgbContextPrediction = Context_prediction_net(out_channel_M=out_channel_M)
        self.rgbEntropyParameters = Entropy_parameter_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        # ir
        self.irPriorEncoder = Analysis_prior_net_nips(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.irPriorDecoder = Synthesis_prior_net_nips(out_channel_N=out_channel_N)
        self.irBitEstimator_z = BitEstimator(out_channel_N)
        self.irContextPrediction = Context_prediction_net(out_channel_M=out_channel_M)
        self.irEntropyParameters = Entropy_parameter_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)

        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_rgb, input_ir):
        # encoder
        rgb_feature, ir_feature = self.encoder(input_rgb, input_ir)
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

        #ir 
        ir_quant_noise_feature = torch.zeros(input_ir.size(0), self.out_channel_M, input_ir.size(2) // 8, input_ir.size(3) // 8).cuda()
        ir_quant_noise_z = torch.zeros(input_ir.size(0), self.out_channel_N, input_ir.size(2) // 32, input_ir.size(3) // 32).cuda()
        ir_quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(ir_quant_noise_feature), -0.5, 0.5)
        ir_quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(ir_quant_noise_z), -0.5, 0.5)
        ir_z = self.irPriorEncoder(ir_feature)
        if self.training:
            ir_compressed_z = ir_z + ir_quant_noise_z
        else:
            ir_compressed_z = torch.round(ir_z)
        ir_recon_sigma = self.irPriorDecoder(ir_compressed_z)
        ir_feature_renorm = ir_feature
        if self.training:
            ir_compressed_feature_renorm = ir_feature_renorm + ir_quant_noise_feature
        else:
            ir_compressed_feature_renorm = torch.round(ir_feature_renorm)
        ir_predict_context = self.irContextPrediction(ir_compressed_feature_renorm)
        ir_entropy_params = self.irEntropyParameters(torch.cat((ir_recon_sigma, ir_predict_context), 1))

        # decoder
        rgb_recon_image, ir_recon_image = self.decoder(rgb_compressed_feature_renorm, ir_compressed_feature_renorm)

        # rgb
        #rgb_entropy_params = rgb_recon_sigma
        rgb_mu = rgb_entropy_params[:, 0: self.out_channel_M, :, :]
        rgb_sigma = rgb_entropy_params[:, self.out_channel_M: self.out_channel_M * 2, :, :]
        # recon_image = prediction + recon_res
        rgb_clipped_recon_image = rgb_recon_image.clamp(0., 1.)
        # distortion
        rgb_mse_loss = torch.mean((rgb_recon_image - input_rgb).pow(2))

        # ir
        #ir_entropy_params = ir_recon_sigma
        ir_mu = ir_entropy_params[:, 0: self.out_channel_M, :, :]
        ir_sigma = ir_entropy_params[:, self.out_channel_M: self.out_channel_M * 2, :, :]
        # recon_image = prediction + recon_res
        ir_clipped_recon_image = ir_recon_image.clamp(0., 1.)
        # distortion
        ir_mse_loss = torch.mean((ir_recon_image - input_ir).pow(2))

        def feature_probs_based_sigma_nips(feature, mu, sigma):
            sigma = sigma.pow(2)
            sigma = sigma.clamp(1e-5, 1e5)
            gaussian = torch.distributions.normal.Normal(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(rgb_z, ir_z):
            rgb_prob = self.rgbBitEstimator_z(rgb_z + 0.5) - self.rgbBitEstimator_z(rgb_z - 0.5)
            rgb_total_bits = torch.sum(torch.clamp(-1.0 * torch.log(rgb_prob + 1e-5) / math.log(2.0), 0, 50))
            ir_prob = self.irBitEstimator_z(ir_z + 0.5) - self.irBitEstimator_z(ir_z - 0.5)
            ir_total_bits = torch.sum(torch.clamp(-1.0 * torch.log(ir_prob + 1e-5) / math.log(2.0), 0, 50))
            return rgb_total_bits, ir_total_bits, rgb_prob, ir_prob

        rgb_total_bits_z, ir_total_bits_z, _, _ = iclr18_estimate_bits_z(rgb_compressed_z, ir_compressed_z)
        # rgb
        rgb_total_bits_feature, _ = feature_probs_based_sigma_nips(rgb_compressed_feature_renorm, rgb_mu, rgb_sigma)
        rgb_shape = input_rgb.size()
        rgb_bpp_feature = rgb_total_bits_feature / (batch_size * rgb_shape[2] * rgb_shape[3])
        rgb_bpp_z = rgb_total_bits_z / (batch_size * rgb_shape[2] * rgb_shape[3])
        rgb_bpp = rgb_bpp_feature + rgb_bpp_z

        # ir
        ir_total_bits_feature, _ = feature_probs_based_sigma_nips(ir_compressed_feature_renorm, ir_mu, ir_sigma)
        ir_shape = input_ir.size()
        ir_bpp_feature = ir_total_bits_feature / (batch_size * ir_shape[2] * ir_shape[3])
        ir_bpp_z = ir_total_bits_z / (batch_size * ir_shape[2] * ir_shape[3])
        ir_bpp = ir_bpp_feature + ir_bpp_z

        return rgb_clipped_recon_image, ir_clipped_recon_image, rgb_mse_loss, ir_mse_loss, \
                rgb_bpp_feature, ir_bpp_feature, rgb_bpp_z, ir_bpp_z, rgb_bpp, ir_bpp