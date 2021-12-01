import os 
import sys
import numpy as np
import math
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

input_dir = "./input"
encode_dir = "./encode"
decode_dir = "./decode"
log_dir = "./result.txt"


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blured tensors
    """

    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=255, size_average=True, full=False):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, weights=None):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels
    Returns:
        torch.Tensor: ms-ssim results
    """
    h, w, c = X.shape
    transform = transforms.ToTensor()
    X = transform(X).view(1,c,h,w)
    Y = transform(Y).view(1,c,h,w)
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


def psnr(img1, img2):
    mse = np.mean((img1/1. - img2/1.) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.**2/mse)

bpps = []
psnrs = []
ms_ssims = []
ms_ssim_dbs = []

def bpg_test(img_name, q):
    img_dir = input_dir + '/' + img_name
    input_img = Image.open(img_dir)
    input_w, input_h = input_img.size
    input_c = len(input_img.split())
    if input_c == 3:
        input_img = input_img.convert('RGB')
    elif input_c == 1:
        input_img = input_img.convert('L')
    input_img = np.array(input_img)

    os.system(f'bpgenc -q {q} -f 444 {img_dir} -o {encode_dir}/{img_name.split(".")[0]}.bpg')
    os.system(f'bpgdec -o {decode_dir}/{img_name} {encode_dir}/{img_name.split(".")[0]}.bpg')

    size = os.path.getsize(f'{encode_dir}/{img_name.split(".")[0]}.bpg') * 8
    output_img = Image.open(f"{decode_dir}/{img_name}").convert('RGB')
    if input_c == 3:
        output_img = output_img.convert('RGB')
    elif input_c == 1:
        output_img = output_img.convert('L')
    output_img = np.array(output_img)
    bpps.append(size / input_h / input_w)
    psnrs.append(psnr(input_img, output_img))
    msssim = ms_ssim(input_img, output_img, data_range=1., size_average=True)
    ms_ssims.append(msssim)
    ms_ssim_dbs.append(-10*(torch.log(1-ms_ssims)/np.log(10)))

if __name__ == '__main__': 
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    encode_dir = input_dir + '/encoder'
    decode_dir = input_dir + '/decoder'

    if not os.path.exists(encode_dir):
        os.mkdir(encode_dir)
    if not os.path.exists(decode_dir):
        os.mkdir(decode_dir)

    img_names = [f for f in os.listdir(input_dir) if f[-3:] == 'png']、
    for q in range(0, 52, 4):
        bpps.clear()
        psnrs.clear()
        ms_ssims.clear()
        ms_ssim_dbs.clear()

        for img_name in img_names:
            bpg_test(img_name, q)

        print(f'mean(qp={q}) 	bpp: {np.mean(bpps):.5f} 	psnr: {np.mean(psnrs):.3f} 	ms-ssim: {np.mean(ms_ssims):.4f}    ms-ssim-db: {np.mean(ms_ssim_dbs):.3f}')
        with open(log_dir, 'a') as log:
            log.write(f'{np.mean(bpps):.5f} {np.mean(psnrs):.3f} {np.mean(ms_ssims):.4f} {np.mean(ms_ssim_dbs):.3f}\n')