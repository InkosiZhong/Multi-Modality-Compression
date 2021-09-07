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
    elapsed, losses, psnrs, bpps, bpp_features, bpp_zs, mse_losses = [AverageMeter(print_freq) for _ in range(7)]

    for _, input in enumerate(train_rgb_loader):
        input = input.cuda()
        start_time = time.time()
        global_step += 1
        _, mse_loss, bpp_feature, bpp_z, bpp = net(input)
        distribution_loss = bpp
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
            if mse_loss.item() > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                psnrs.update(psnr.item())
            else:
                psnrs.update(100)
            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item())
            bpps.update(bpp.item())
            bpp_features.update(bpp_feature.item())
            bpp_zs.update(bpp_z.item())
            mse_losses.update(mse_loss.item())

        if (global_step % print_freq) == 0:
            train_logger(global_step, epoch, cur_lr, losses, elapsed, None, \
                psnrs, mse_losses, bpps, bpp_zs, bpp_features, \
                None, None, None, None, None)
            
        if (global_step % save_model_freq) == 0:
            save_model(model, global_step, home)
            test(global_step)
            net.train()

    return global_step


def test(step):
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for _, input in enumerate(test_rgb_loader):
            input = input.cuda()
            clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(input)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = ms_ssim(clipped_recon_image.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            logger.info("Bpp(rgb):{:.6f}, PSNR(rgb):{:.6f}, MS-SSIM(rgb):{:.6f}, MS-SSIM-DB(rgb):{:.6f}".format(\
                bpp, psnr, msssim, msssimDB))
            cnt += 1

        test_logger(step, cnt, None,
            sumPsnr, sumMsssim, sumMsssimDB, sumBpp,
            None, None, None, None)


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

    model = MultiCompression(in_channel1=3, out_channel_N=out_channel_N, out_channel_M=out_channel_M)

    if args.pretrain_rgb != '':
        logger.info("loading model:{}".format(args.pretrain_rgb))
        global_step = load_model(model, args.pretrain_rgb)
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)
    net = model.cuda()
    logger.info(net)
    net = torch.nn.DataParallel(net, list(range(gpu_num)))
    parameters = net.parameters()

    global test_rgb_loader, test_ir_loader
    test_rgb_loader, _, _ = build_dataset(test_rgb_dir, None, 1, 1)
    if args.test:
        test(global_step)
        exit(-1)
    optimizer = optim.Adam(parameters, lr=base_lr)

    global train_rgb_loader
    tb_logger = SummaryWriter(home + '/events')

    train_rgb_loader, _, n = build_dataset(train_rgb_dir, None, batch_size, 2)

    steps_epoch = global_step // n
    save_model(model, global_step, home)
    for epoch in range(steps_epoch, tot_epoch):
        adjust_learning_rate(optimizer, global_step)
        if global_step > tot_step:
            save_model(model, global_step, home)
            break
        global_step = train(epoch, global_step)
        save_model(model, global_step, home)