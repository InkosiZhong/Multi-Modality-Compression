import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models import *


def save_model(model, iter, home):
    print("save at " + home + "/snapshot/iter{}.model")
    torch.save(model.state_dict(), home+"/snapshot/iter{}.model".format(iter))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class ImageCompressor(nn.Module):
    def __init__(self, img_channel=3, out_channel_N=192, out_channel_M=192):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net(in_channel=img_channel, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel=img_channel, out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net_nips(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net_nips(out_channel_N=out_channel_N)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.context_prediction = Context_prediction_net(out_channel_M=out_channel_M)
        self.entropy_parameters = Entropy_parameter_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64, input_image.size(3) // 64).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]

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
        predict_context = self.context_prediction(compressed_feature_renorm)
        entropy_params = self.entropy_parameters(torch.cat((recon_sigma, predict_context), 1))
        recon_image = self.Decoder(compressed_feature_renorm)
        entropy_params = recon_sigma
        mu = entropy_params[:, 0: self.out_channel_M, :, :]
        sigma = entropy_params[:, self.out_channel_M: self.out_channel_M * 2, :, :]
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        #mse_loss = torch.mean((recon_image - input_image).pow(2))
        mse_loss = torch.mean((clipped_recon_image - input_image).pow(2))

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

        total_bits_feature, _ = feature_probs_based_sigma_nips(compressed_feature_renorm, mu, sigma)
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z

        return clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp


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
        ir_quant_noise_feature = torch.zeros(input_ir.size(0), self.out_channel_M, input_ir.size(2) // 16, input_ir.size(3) // 16).cuda()
        ir_quant_noise_z = torch.zeros(input_ir.size(0), self.out_channel_N, input_ir.size(2) // 64, input_ir.size(3) // 64).cuda()
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
        rgb_entropy_params = rgb_recon_sigma
        rgb_mu = rgb_entropy_params[:, 0: self.out_channel_M, :, :]
        rgb_sigma = rgb_entropy_params[:, self.out_channel_M: self.out_channel_M * 2, :, :]
        # recon_image = prediction + recon_res
        rgb_clipped_recon_image = rgb_recon_image.clamp(0., 1.)
        # distortion
        #mse_loss = torch.mean((recon_image - input_image).pow(2))
        rgb_mse_loss = torch.mean((rgb_clipped_recon_image - input_rgb).pow(2))

        # ir
        ir_entropy_params = ir_recon_sigma
        ir_mu = ir_entropy_params[:, 0: self.out_channel_M, :, :]
        ir_sigma = ir_entropy_params[:, self.out_channel_M: self.out_channel_M * 2, :, :]
        # recon_image = prediction + recon_res
        ir_clipped_recon_image = ir_recon_image.clamp(0., 1.)
        # distortion
        #mse_loss = torch.mean((recon_image - input_image).pow(2))
        ir_mse_loss = torch.mean((ir_clipped_recon_image - input_ir).pow(2))

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