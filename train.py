import torch
import torch.optim as optim
import time
from datasets import build_dataset
from tensorboardX import SummaryWriter
from Meter import AverageMeter
import numpy as np
from model import *
from common import *

torch.backends.cudnn.enabled = True

home = "./experiments/"
enable_wandb = True
wandb_project = "TAVC"
wandb_recovery = ""
tb_logger = None

train_data_dir = "../datasets/FLIR/train/"
test_data_dir = "../datasets/FLIR/val/"
train_rgb_dir = train_data_dir + "/RGB/"
train_ir_dir = train_data_dir + "/IR/"
test_rgb_dir = test_data_dir + "/RGB/"
test_ir_dir = test_data_dir + "/IR/"

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
global_step = 0
save_model_freq = 50000
out_channel_N = 192
out_channel_M = 320

parser = argparse.ArgumentParser(description='Pytorch reimplement for variational image compression with a scale hyperprior')

parser.add_argument('--finetune', action='store_true')
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('-r', '--pretrain_rgb', default = '',
        help='load rgb pretrain model')
parser.add_argument('-i', '--pretrain_ir', default = '',
        help='load ir pretrain model')
parser.add_argument('-m', '--mode', help='MDMC mode: train_ir, train_rgb')
parser.add_argument('--test', action='store_true')
parser.add_argument('--config', dest='config', required=False,
        help = 'hyperparameter in json format')
parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')
parser.add_argument('-f', '--freeze', default=0, type=int, help='freeze steps for ir and rgb pretrain parameters')

def parse_config(args):
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
    if "test_dataset" in config:
        test_data_dir = config['test_dataset']

    global train_rgb_dir, train_ir_dir, test_rgb_dir, test_ir_dir
    train_rgb_dir = train_data_dir + "/RGB/"
    train_ir_dir = train_data_dir + "/IR/"
    test_rgb_dir = test_data_dir + "/RGB/"
    test_ir_dir = test_data_dir + "/IR/"
    
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
                wandb.init(sync_tensorboard=True, project=wandb_project, name=run_name, dir=home)
            else:
                wandb.init(sync_tensorboard=True, project=wandb_project, name=run_name, dir=home, resume=wandb_recovery)


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

    log_ready = False
    for _, input in enumerate(zip(train_rgb_loader, train_ir_loader)):
        if args.freeze and global_step == args.freeze: # finish freezing
            logger.info(f"Unfreeze paramaters at step {global_step}")
            for name, param in net.named_parameters():
                if (args.mode == 'train_ir' and 'rgb' not in name) or \
                   (args.mode == 'train_rgb' and 'ir' not in name):
                    param.requires_grad = True
            optimizer = optim.Adam(net.parameters(), lr=cur_lr)

        rgb_input, ir_input = input
        rgb_input = rgb_input.cuda()
        ir_input = ir_input.cuda()
        start_time = time.time()
        global_step += 1
        _, _, rgb_mse_loss, ir_mse_loss, \
                rgb_bpp_feature, ir_bpp_feature, rgb_bpp_z, ir_bpp_z, rgb_bpp, ir_bpp = net(rgb_input, ir_input)
        bpp = rgb_bpp if args.mode == 'train_rgb' else ir_bpp
        distribution_loss = bpp
        mse_loss = rgb_mse_loss if args.mode == 'train_rgb' else ir_mse_loss
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
            log_ready = True

        if log_ready and (global_step % print_freq) == 0:
            train_logger(tb_logger, global_step, tot_step, epoch, cur_lr, losses, elapsed, bpps, \
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
                    torch.mean(rgb_bpp_z), torch.mean(ir_bpp_z), torch.mean(rgb_bpp), torch.mean(ir_bpp), torch.mean(rgb_bpp + ir_bpp)
            sumBpp += bpp
            rgb_psnr = 10 * (torch.log(1. / rgb_mse_loss) / np.log(10))
            rgb_sumBpp += rgb_bpp
            rgb_sumPsnr += rgb_psnr
            ir_psnr = 10 * (torch.log(1. / ir_mse_loss) / np.log(10))
            ir_sumBpp += ir_bpp
            ir_sumPsnr += ir_psnr

            def cal_msssim(recon, input):
                msssim = ms_ssim(recon.cpu().detach(), input.cpu(), data_range=1.0, size_average=True)
                msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
                return msssim, msssimDB

            rgb_msssim, rgb_msssimDB = cal_msssim(rgb_clipped_recon_image, rgb_input)
            rgb_sumMsssimDB += rgb_msssimDB
            rgb_sumMsssim += rgb_msssim
            
            ir_msssim, ir_msssimDB = cal_msssim(ir_clipped_recon_image, ir_input)
            ir_sumMsssimDB += ir_msssimDB
            ir_sumMsssim += ir_msssim
            logger.info("Bpp:{:.6f}, Bpp(rgb):{:.6f}, Bpp(ir):{:.6f}, \
                PSNR(rgb):{:.6f}, MS-SSIM(rgb):{:.6f}, MS-SSIM-DB(rgb):{:.6f}, \
                PSNR(ir):{:.6f}, MS-SSIM(ir):{:.6f}, MS-SSIM-DB(ir):{:.6f}".format(\
                    bpp, rgb_bpp, ir_bpp, rgb_psnr, rgb_msssim, rgb_msssimDB, ir_psnr, ir_msssim, ir_msssimDB))
            cnt += 1
        
        test_logger(tb_logger, step, cnt, sumBpp,
            rgb_sumPsnr, rgb_sumMsssim, rgb_sumMsssimDB, rgb_sumBpp,
            ir_sumPsnr, ir_sumMsssim, ir_sumMsssimDB, ir_sumBpp)


if __name__ == "__main__":
    args = parser.parse_args()
    parse_config(args)
    torch.manual_seed(seed=args.seed)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    
    filehandler = logging.FileHandler(home + '/log.txt')
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)

    logger.setLevel(logging.INFO)
    logger.info("image compression training")
    logger.info("config : ")
    logger.info(open(args.config).read())
    logger.info("out_channel_N:{}, out_channel_M:{}".format(out_channel_N, out_channel_M))
    logger.info('Branch: Master')

    if args.mode not in ['train_rgb', 'train_ir']:
        logger.info(f'unknown mode \'{args.mode}\', use \'train_rgb\' or \'train_ir\'')
        exit(-1)
    
    model = MultiCompression(in_channel1=3, 
                             in_channel2=1, 
                             out_channel_N=out_channel_N, 
                             out_channel_M=out_channel_M,
                             mode=args.mode)

    if args.pretrain_rgb != '':
        logger.info("loading model:{}".format(args.pretrain_rgb))
        load_model(model, args.pretrain_rgb)
    if args.pretrain_ir != '':
        logger.info("loading model:{}".format(args.pretrain_ir))
        load_model(model, args.pretrain_ir)
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)
        if args.finetune:
            global_step = 0

    net = model.cuda()
    for name, param in net.named_parameters():
        if (args.mode == 'train_ir' and 'rgb' in name) or \
           (args.mode == 'train_rgb' and 'ir' in name):
            param.requires_grad = False
    # freeze
    if args.freeze != 0:
        logger.info(f"Freeze parameters for {args.freeze} steps")
        for name, param in net.named_parameters():
            if "align" not in name and "fusion" not in name:
                param.requires_grad = False
    
    logger.info(net)
    net = torch.nn.DataParallel(net, list(range(gpu_num)))
    parameters = net.parameters()

    test_rgb_loader, test_ir_loader, _ = build_dataset(test_rgb_dir, test_ir_dir, 1, 1)
    if args.test:
        test(global_step)
        exit(-1)
    optimizer = optim.Adam(parameters, lr=base_lr)

    tb_logger = SummaryWriter(home + '/events/')

    train_rgb_loader, train_ir_loader, n = build_dataset(train_rgb_dir, train_ir_dir, batch_size, 2, train_data_dir+'/FLIR.txt')

    steps_epoch = global_step // n
    save_model(model, global_step, home)
    for epoch in range(steps_epoch, tot_epoch):
        adjust_learning_rate(optimizer, global_step)
        if global_step > tot_step:
            save_model(model, global_step, home)
            break
        global_step = train(epoch, global_step)
        save_model(model, global_step, home)