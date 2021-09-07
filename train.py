import os
import argparse
from model import *
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
from datasets import Datasets, TestKodakDataset, build_dataset
from tensorboardX import SummaryWriter
from Meter import AverageMeter
torch.backends.cudnn.enabled = True

import wandb

home = "./experiments/"
enable_wandb = True
wandb_project = "TAVC"
wandb_recovery = ""

train_data_dir = "../datasets/FLIR/train/"
test_data_dir = "../datasets/FLIR/val/"
train_rgb_dir = train_data_dir + "RGB/"
train_ir_dir = train_data_dir + "IR/"
test_rgb_dir = test_data_dir + "RGB/"
test_ir_dir = test_data_dir + "IR/"

# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 8192
print_freq = 100
cal_step = 40
warmup_step = 0#  // gpu_num
batch_size = 4
tot_epoch = 1000000
tot_step = 2500000
decay_interval = 2200000
lr_decay = 0.1
rgb_size = 256
ir_size = 128
logger = logging.getLogger("ImageCompression")
tb_logger = None
global_step = 0
save_model_freq = 50000
out_channel_N = 192
out_channel_M = 320

parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')

'''parser.add_argument('-n', '--name', default='',
        help='output training details')'''
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('--test', action='store_true')
parser.add_argument('--config', dest='config', required=False,
        help = 'hyperparameter in json format')
parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')

def parse_config(config):
    config = json.load(open(args.config))
    global tot_epoch, tot_step,  base_lr, cur_lr, lr_decay, decay_interval, train_lambda, batch_size, print_freq, \
        out_channel_M, out_channel_N, save_model_freq, cal_step
    if 'tot_epoch' in config:
        tot_epoch = config['tot_epoch']
    if 'tot_step' in config:
        tot_step = config['tot_step']
    if 'train_lambda' in config:
        train_lambda = config['train_lambda']
        if train_lambda < 4096:
            out_channel_N = 128
            out_channel_M = 192
        else:
            out_channel_N = 192
            out_channel_M = 320
    if 'batch_size' in config:
        batch_size = config['batch_size']
    if "print_freq" in config:
        print_freq = config['print_freq']
    if "cal_step" in config:
        cal_step = config['cal_step']
    if "save_model_freq" in config:
        save_model_freq = config['save_model_freq']
    if 'lr' in config:
        if 'base' in config['lr']:
            base_lr = config['lr']['base']
            cur_lr = base_lr
        if 'decay' in config['lr']:
            lr_decay = config['lr']['decay']
        if 'decay_interval' in config['lr']:
            decay_interval = config['lr']['decay_interval']
    if 'out_channel_N' in config:
        out_channel_N = config['out_channel_N']
    if 'out_channel_M' in config:
        out_channel_M = config['out_channel_M']

    global train_data_dir, test_data_dir
    if "train_dataset" in config:
        train_data_dir = config['train_dataset']
    if "test_data_dir" in config:
        test_data_dir = config['test_dataset']

    global train_rgb_dir, train_ir_dir, test_rgb_dir, test_ir_dir
    train_rgb_dir = train_data_dir + "RGB/"
    train_ir_dir = train_data_dir + "IR/"
    test_rgb_dir = test_data_dir + "RGB/"
    test_ir_dir = test_data_dir + "IR/"
    
    global home, enable_wandb, wandb_project, wandb_recovery
    run_name = config['run_name'] if 'run_name' in config else 'unknown'
    home += run_name
    if not args.test:
        if not os.path.exists(home):
            os.mkdir(home)
        else:
            print("home dir is already exists.")
        if not os.path.exists(home + "/snapshot"):
            os.mkdir(home + "/snapshot") # to save model
        else:
            print("snapshot dir is already exists.")
        if 'wandb' in config and 'enable' in config['wandb'] and 'project' in config['wandb']:
            enable_wandb = config['wandb']['enable']
            if not enable_wandb:
                return
            wandb_project = config['wandb']['project']
            if 'dryrun' in config['wandb'] and config['wandb']['dryrun']:
                os.environ["WANDB_MODE"] = "dryrun"
            if 'recovery' in config['wandb']:
                wandb_recovery = config['wandb']['recovery']
                print('recovery wandb to task: ', wandb_recovery)
            if wandb_recovery == "":
                wandb.init(sync_tensorboard=True, project=wandb_project, name=run_name)
            else:
                wandb.init(sync_tensorboard=True, project=wandb_project, name=run_name, resume=wandb_recovery)


def adjust_learning_rate(optimizer, global_step):
    global cur_lr
    global warmup_step
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:#  // gpu_num:
        lr = base_lr
    else:
        # lr = base_lr * (lr_decay ** (global_step // decay_interval))
        lr = base_lr * lr_decay
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch, global_step):
    logger.info("Epoch {} begin".format(epoch))
    net.train()
    global optimizer
    elapsed, losses, rgb_psnrs, ir_psnrs, bpps, rgb_bpps, ir_bpps, \
        rgb_bpp_features, ir_bpp_features, rgb_bpp_zs, ir_bpp_zs, \
        rgb_mse_losses, ir_mse_losses = [AverageMeter(print_freq) for _ in range(13)]
    # model_time = 0
    # compute_time = 0
    # log_time = 0

    # AMP
    #scaler = torch.cuda.amp.GradScaler()

    for batch_idx, input in enumerate(zip(train_rgb_loader, train_ir_loader)):
        #with torch.cuda.amp.autocast():
        #torch.autograd.set_detect_anomaly(True) #debug
        rgb_input, ir_input = input
        rgb_input = rgb_input.cuda()
        ir_input = ir_input.cuda()
        start_time = time.time()
        global_step += 1
        # print("debug", torch.max(input), torch.min(input))
        rgb_clipped_recon_image, ir_clipped_recon_image, rgb_mse_loss, ir_mse_loss, \
                rgb_bpp_feature, ir_bpp_feature, rgb_bpp_z, ir_bpp_z, rgb_bpp, ir_bpp = net(rgb_input, ir_input)
        # print("debug", clipped_recon_image.shape, " ", mse_loss.shape, " ", bpp.shape)
        # print("debug", mse_loss, " ", bpp_feature, " ", bpp_z, " ", bpp)
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
        # model_time += (time.time()-start_time)
        #optimizer.zero_grad()
        #clip_gradient(optimizer, 5)
        #with torch.autograd.detect_anomaly(): # debug
        #scaler.scale(rd_loss).backward()
        #scaler.step(optimizer)
        #scaler.update()
        if (global_step % cal_step) == 0:
            # t0 = time.time()
            if rgb_mse_loss.item() > 0:
                rgb_psnr = 10 * (torch.log(1 * 1 / rgb_mse_loss) / np.log(10))
                rgb_psnrs.update(rgb_psnr.item())
                ir_psnr = 10 * (torch.log(1 * 1 / ir_mse_loss) / np.log(10))
                ir_psnrs.update(ir_psnr.item())
            else:
                rgb_psnrs.update(100)
                ir_psnrs.update(100)
            # t1 = time.time()
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
            # t2 = time.time()
            # compute_time += (t2 - t0)

        if (global_step % print_freq) == 0:
            # begin = time.time()
            tb_logger.add_scalar('lr', cur_lr, global_step)
            tb_logger.add_scalar('rd_loss', losses.avg, global_step)
            tb_logger.add_scalar('rgb_psnr', rgb_psnrs.avg, global_step)
            tb_logger.add_scalar('ir_psnr', ir_psnrs.avg, global_step)
            tb_logger.add_scalar('bpp', bpps.avg, global_step)
            tb_logger.add_scalar('rgb_bpp', rgb_bpps.avg, global_step)
            tb_logger.add_scalar('ir_bpp', ir_bpps.avg, global_step)
            tb_logger.add_scalar('rgb_bpp_feature', rgb_bpp_features.avg, global_step)
            tb_logger.add_scalar('ir_bpp_feature', ir_bpp_features.avg, global_step)
            tb_logger.add_scalar('rgb_bpp_z', rgb_bpp_zs.avg, global_step)
            tb_logger.add_scalar('ir_bpp_z', ir_bpp_zs.avg, global_step)
            process = global_step / tot_step * 100.0
            log = (' | '.join([
                f'Step [{global_step}/{tot_step}={process:.2f}%]',
                f'Epoch {epoch}',
                f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                f'Lr {cur_lr}',
                f'Total Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'PSNR(rgb) {rgb_psnrs.val:.3f} ({rgb_psnrs.avg:.3f})',
                f'PSNR(ir) {ir_psnrs.val:.3f} ({ir_psnrs.avg:.3f})',
                f'Bpp {bpps.val:.5f} ({bpps.avg:.5f})',
                f'Bpp(rgb) {rgb_bpps.val:.5f} ({rgb_bpps.avg:.5f})',
                f'Bpp(ir) {ir_bpps.val:.5f} ({ir_bpps.avg:.5f})',
                f'Bpp_feature(rgb) {rgb_bpp_features.val:.5f} ({rgb_bpp_features.avg:.5f})',
                f'Bpp_feature(ir) {ir_bpp_features.val:.5f} ({ir_bpp_features.avg:.5f})',
                f'Bpp_z(rgb) {rgb_bpp_zs.val:.5f} ({rgb_bpp_zs.avg:.5f})',
                f'Bpp_z(ir) {ir_bpp_zs.val:.5f} ({ir_bpp_zs.avg:.5f})',
                f'MSE(rgb) {rgb_mse_losses.val:.5f} ({rgb_mse_losses.avg:.5f})',
                f'MSE(ir) {ir_mse_losses.val:.5f} ({ir_mse_losses.avg:.5f})',
            ]))
            logger.info(log)
            # log_time = time.time() - begin
            # print("Log time", log_time)
            # print("Compute time", compute_time)
            # print("Model time", model_time)
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
        for batch_idx, input in enumerate(zip(test_rgb_loader, test_ir_loader)):
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
            logger.info("Bpp:{:.6f}, PSNR(rgb):{:.6f}, PSNR(ir):{:.6f}, \
                MS-SSIM(rgb):{:.6f}, MS-SSIM(rgb):{:.6f}, MS-SSIM-DB(rgb):{:.6f}, MS-SSIM-DB(ir):{:.6f}".format(\
                    bpp, rgb_psnr, ir_psnr, rgb_msssim, ir_msssim, rgb_msssimDB, ir_msssimDB))
            cnt += 1

        logger.info("Test on Kodak dataset: model-{}".format(step))
        sumBpp /= cnt
        rgb_sumPsnr /= cnt
        rgb_sumMsssim /= cnt
        rgb_sumMsssimDB /= cnt
        ir_sumPsnr /= cnt
        ir_sumMsssim /= cnt
        ir_sumMsssimDB /= cnt
        logger.info("Dataset Average result---Bpp:{:.6f}, PSNR(rgb):{:.6f}, PSNR(ir):{:.6f}, \
                MS-SSIM(rgb):{:.6f}, MS-SSIM(rgb):{:.6f}, MS-SSIM-DB(rgb):{:.6f}, MS-SSIM-DB(ir):{:.6f}".format(\
                    bpp, rgb_psnr, ir_psnr, rgb_msssim, ir_msssim, rgb_msssimDB, ir_msssimDB))
        if tb_logger !=None:
            logger.info("Add tensorboard---Step:{}".format(step))
            tb_logger.add_scalar("BPP_Test", sumBpp, step)
            tb_logger.add_scalar("PSNR_Test(rgb)", rgb_sumPsnr, step)
            tb_logger.add_scalar("MS-SSIM_Test(rgb)", rgb_sumMsssim, step)
            tb_logger.add_scalar("MS-SSIM_DB_Test(ir)", ir_sumMsssimDB, step)
            tb_logger.add_scalar("PSNR_Test(ir)", ir_sumPsnr, step)
            tb_logger.add_scalar("MS-SSIM_Test(ir)", ir_sumMsssim, step)
            tb_logger.add_scalar("MS-SSIM_DB_Test(ir)", ir_sumMsssimDB, step)
        else:
            logger.info("No need to add tensorboard")


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
    parse_config(args.config)
    logger.info("out_channel_N:{}, out_channel_M:{}".format(out_channel_N, out_channel_M))
    logger.info('Branch: Master')
    model = MultiCompression(3, 1, out_channel_N, out_channel_M)
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
    # save_model(model, 0)
    global train_rgb_loader, train_ir_loader
    #tb_logger = SummaryWriter(os.path.join(home, 'events'))
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
