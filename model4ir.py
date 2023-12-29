import torch
import torch.nn as nn
import math
from models import *
from models.ir_encoder import *
from models.ir_decoder import *


class MultiCompression(nn.Module):
    def __init__(self, in_channel2=1, out_channel_N=192, out_channel_M=192):
        super().__init__()
        self.encoder = IREncoder(in_channel2=in_channel2, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.decoder = IRDecoder(out_channel2=in_channel2, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        # ir
        self.irPriorEncoder = Analysis_prior_net_nips(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.irPriorDecoder = Synthesis_prior_net_nips(out_channel_N=out_channel_N)
        self.irBitEstimator_z = BitEstimator(out_channel_N)
        self.irContextPrediction = Context_prediction_net(out_channel_M=out_channel_M)
        self.irEntropyParameters = Entropy_parameter_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)

        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_ir):
        # encoder
        ir_feature, _ = self.encoder(input_ir)
        batch_size = ir_feature.size()[0]

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
        ir_recon_image, _ = self.decoder(ir_compressed_feature_renorm)

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

        def iclr18_estimate_bits_z(ir_z):
            ir_prob = self.irBitEstimator_z(ir_z + 0.5) - self.irBitEstimator_z(ir_z - 0.5)
            ir_total_bits = torch.sum(torch.clamp(-1.0 * torch.log(ir_prob + 1e-5) / math.log(2.0), 0, 50))
            return ir_total_bits, ir_prob

        ir_total_bits_z, _ = iclr18_estimate_bits_z(ir_compressed_z)
        
        # ir
        ir_total_bits_feature, _ = feature_probs_based_sigma_nips(ir_compressed_feature_renorm, ir_mu, ir_sigma)
        ir_shape = input_ir.size()
        ir_bpp_feature = ir_total_bits_feature / (batch_size * ir_shape[2] * ir_shape[3])
        ir_bpp_z = ir_total_bits_z / (batch_size * ir_shape[2] * ir_shape[3])
        ir_bpp = ir_bpp_feature + ir_bpp_z

        return ir_clipped_recon_image, ir_mse_loss, \
                ir_bpp_feature, ir_bpp_z, ir_bpp