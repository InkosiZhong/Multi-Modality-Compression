import torch
import torch.optim as optim
import time
from datasets import build_dataset
from tensorboardX import SummaryWriter
from Meter import AverageMeter
import numpy as np
from model4rgb import *
from common import *

torch.backends.cudnn.enabled = True

def train(epoch, global_step):
    logger.info("Epoch {} begin".format(epoch))
    net.train()
    global optimizer
    elapsed, losses, rgb_psnrs, ir_psnrs, bpps, rgb_bpps, ir_bpps, \
        rgb_bpp_features, ir_bpp_features, rgb_bpp_zs, ir_bpp_zs, \
        rgb_mse_losses, ir_mse_losses = [AverageMeter(print_freq) for _ in range(13)]

    for _, input in enumerate(zip(train_rgb_loader, train_ir_loader)):
        rgb_input, ir_input = input
        rgb_input = rgb_input.cuda()
        ir_input = ir_input.cuda()
        start_time = time.time()
        global_step += 1
        _, _, rgb_mse_loss, ir_mse_loss, \
                rgb_bpp_feature, ir_bpp_feature, rgb_bpp_z, ir_bpp_z, rgb_bpp, ir_bpp = net(rgb_input, ir_input)
        bpp = rgb_bpp + ir_bpp
        distribution_loss = bpp
        mse_loss = rgb_mse_loss + ir_mse_loss
        distortion = mse_loss
        rd_loss = train_lambda * distortion + distribution_loss
        optimizer.zero_grad()
        rd_loss.backward()
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 5)
        optimizer.step()
        if (global_step % cal_step) == 0:
            if rgb_mse_loss.item() > 0:
                rgb_psnr = 10 * (torch.log(1 * 1 / rgb_mse_loss) / np.log(10))
                rgb_psnrs.update(rgb_psnr.item())
                ir_psnr = 10 * (torch.log(1 * 1 / ir_mse_loss) / np.log(10))
                ir_psnrs.update(ir_psnr.item())
            else:
                rgb_psnrs.update(100)
                ir_psnrs.update(100)
            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item())
            bpps.update(bpp.item())
            rgb_bpps.update(rgb_bpp.item())
            ir_bpps.update(ir_bpp.item())
            rgb_bpp_features.update(rgb_bpp_feature.item())
            ir_bpp_features.update(ir_bpp_feature.item())
            rgb_bpp_zs.update(rgb_bpp_z.item())
            ir_bpp_zs.update(ir_bpp_z.item())
            rgb_mse_losses.update(rgb_mse_loss.item())
            ir_mse_losses.update(ir_mse_loss.item())

        if (global_step % print_freq) == 0:
            train_logger(global_step, epoch, cur_lr, losses, elapsed, bpps, \
                rgb_psnrs, rgb_mse_losses, rgb_bpps, rgb_bpp_zs, rgb_bpp_features, \
                ir_psnrs, ir_mse_losses, ir_bpps, ir_bpp_zs, ir_bpp_features)

        if (global_step % save_model_freq) == 0:
            save_model(model, global_step, home)
            test(global_step)
            net.train()

    return global_step


def test(step):
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        rgb_sumBpp = 0
        ir_sumBpp = 0
        rgb_sumPsnr = 0
        ir_sumPsnr = 0
        rgb_sumMsssim = 0
        ir_sumMsssim = 0
        rgb_sumMsssimDB = 0
        ir_sumMsssimDB = 0
        cnt = 0
        for _, input in enumerate(zip(test_rgb_loader, test_ir_loader)):
            rgb_input, ir_input = input
            rgb_input = rgb_input.cuda()
            ir_input = ir_input.cuda()
            rgb_clipped_recon_image, ir_clipped_recon_image, rgb_mse_loss, ir_mse_loss, \
                rgb_bpp_feature, ir_bpp_feature, rgb_bpp_z, ir_bpp_z, rgb_bpp, ir_bpp = net(rgb_input, ir_input)
            rgb_mse_loss, ir_mse_loss, rgb_bpp_feature, ir_bpp_feature, rgb_bpp_z, ir_bpp_z, rgb_bpp, ir_bpp, bpp = \
                torch.mean(rgb_mse_loss), torch.mean(ir_mse_loss), torch.mean(rgb_bpp_feature), torch.mean(ir_bpp_feature), \
                    torch.mean(rgb_bpp_z), torch.mean(ir_bpp_z), torch.mean(rgb_bpp), torch.mean(ir_bpp), torch.mean(bpp)
            sumBpp += bpp
            rgb_psnr = 10 * (torch.log(1. / rgb_mse_loss) / np.log(10))
            rgb_sumBpp += rgb_bpp
            rgb_sumPsnr += rgb_psnr
            ir_psnr = 10 * (torch.log(1. / ir_mse_loss) / np.log(10))
            ir_sumBpp += ir_bpp
            ir_sumPsnr += ir_psnr

            def cal_msssim(recon):
                msssim = ms_ssim(recon.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
                msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
                return msssim, msssimDB

            rgb_msssim, rgb_msssimDB = cal_msssim(rgb_clipped_recon_image)
            rgb_sumMsssimDB += rgb_msssimDB
            rgb_sumMsssim += rgb_msssim
            
            ir_msssim, ir_msssimDB = cal_msssim(ir_clipped_recon_image)
            ir_sumMsssimDB += ir_msssimDB
            ir_sumMsssim += ir_msssim
            logger.info("Bpp:{:.6f}, Bpp(rgb):{:.6f}, Bpp(ir):{:.6f}, \
                PSNR(rgb):{:.6f}, MS-SSIM(rgb):{:.6f}, MS-SSIM-DB(rgb):{:.6f}, \
                PSNR(ir):{:.6f}, MS-SSIM(ir):{:.6f}, MS-SSIM-DB(ir):{:.6f}".format(\
                    bpp, rgb_bpp, ir_bpp, rgb_psnr, rgb_msssim, rgb_msssimDB, ir_psnr, ir_msssim, ir_msssimDB))
            cnt += 1
        
        test_logger(step, cnt, sumBpp,
            rgb_sumPsnr, rgb_sumMsssim, rgb_sumMsssimDB, rgb_sumBpp,
            ir_sumPsnr, ir_sumMsssim, ir_sumMsssimDB, ir_sumBpp)


if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(seed=args.seed)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    tb_logger = None
    
    filehandler = logging.FileHandler(home + '/log.txt')
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    logger.setLevel(logging.INFO)
    logger.info("image compression training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    parse_config(args)
    logger.info("out_channel_N:{}, out_channel_M:{}".format(out_channel_N, out_channel_M))
    logger.info('Branch: Master')

    model = MultiCompression(in_channel1=3, in_channel2=1, out_channel_N=out_channel_N, out_channel_M=out_channel_M)

    if args.pretrain_rgb != '':
        logger.info("loading model:{}".format(args.pretrain_rgb))
        load_model(model, args.pretrain_rgb)
    if args.pretrain_ir != '':
        logger.info("loading model:{}".format(args.pretrain_ir))
        load_model(model, args.pretrain_ir)
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)

    net = model.cuda()
    logger.info(net)
    net = torch.nn.DataParallel(net, list(range(gpu_num)))
    parameters = net.parameters()

    global test_rgb_loader, test_ir_loader
    test_rgb_loader, test_ir_loader, _ = build_dataset(test_rgb_dir, test_ir_dir, 1, 1)
    if args.test:
        test(global_step)
        exit(-1)
    optimizer = optim.Adam(parameters, lr=base_lr)

    global train_rgb_loader, train_ir_loader
    tb_logger = SummaryWriter(home + '/events')

    train_rgb_loader, train_ir_loader, n = build_dataset(train_rgb_dir, train_ir_dir, batch_size, 2)

    steps_epoch = global_step // n
    save_model(model, global_step, home)
    for epoch in range(steps_epoch, tot_epoch):
        adjust_learning_rate(optimizer, global_step)
        if global_step > tot_step:
            save_model(model, global_step, home)
            break
        global_step = train(epoch, global_step)
        save_model(model, global_step, home)
