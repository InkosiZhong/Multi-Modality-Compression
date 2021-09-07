import os
import argparse
import torch
import json
import logging
import wandb

home = "./experiments/"
enable_wandb = True
wandb_project = "TAVC"
wandb_recovery = ""

train_data_dir = "../datasets/FLIR/train/"
test_data_dir = "../datasets/FLIR/val/"
train_rgb_dir = train_data_dir + "/RGB/"
train_ir_dir = train_data_dir + "/IR/"
test_rgb_dir = test_data_dir + "/RGB/"
test_ir_dir = test_data_dir + "/IR/"

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
logger = logging.getLogger("MDMC")
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
    if "test_data_dir" in config:
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


def create_train_log(log, prefix, value, k=5):
    if value is not None:
        tb_logger.add_scalar(prefix, value.avg, global_step)
        if k == 3:
            log += f' | {prefix} {value.val:.3f} ({value.avg:.3f})'
        else:
            log += f' | {prefix} {value.val:.5f} ({value.avg:.5f})'
    return log


def train_logger(global_step, epoch, cur_lr, losses, elapsed, bpps, # both
    rgb_psnrs, rgb_mse_losses, rgb_bpps, rgb_bpp_zs, rgb_bpp_features, # rgb
    ir_psnrs, ir_mse_losses, ir_bpps, ir_bpp_zs, ir_bpp_features): # ir

    process = global_step / tot_step * 100.0
    log = f'Step [{global_step}/{tot_step}={process:.2f}%]'
    log += f' | Epoch {epoch}'
    if elapsed is not None:
        log += f' | time {elapsed.val:.5f} ({elapsed.avg:.3f})'
    if cur_lr is not None:
        tb_logger.add_scalar('lr', cur_lr, global_step) 
        log += f' | lr {cur_lr}'
    log = create_train_log(log, 'rd_loss', losses, 3)
    log = create_train_log(log, 'psnr(rgb)', rgb_psnrs, 3)
    log = create_train_log(log, 'psnr(ir)', ir_psnrs, 3)
    log = create_train_log(log, 'bpp', bpps)
    log = create_train_log(log, 'bpp(rgb)', rgb_bpps)
    log = create_train_log(log, 'bpp(ir)', ir_bpps)
    log = create_train_log(log, 'bpp_feature(rgb)', rgb_bpp_features)
    log = create_train_log(log, 'bpp_feature(ir)', ir_bpp_features)
    log = create_train_log(log, 'bpp_z(rgb)', rgb_bpp_zs)
    log = create_train_log(log, 'bpp_z(ir)', ir_bpp_zs)
    if rgb_mse_losses is not None:
        log += f' | rgb_mse_loss {rgb_mse_losses.val:.5f} ({rgb_mse_losses.avg:.5f})'
    if ir_mse_losses is not None:
        log += f' | ir_mse_loss {ir_mse_losses.val:.5f} ({ir_mse_losses.avg:.5f})'

    logger.info(log)

def create_test_log(log, step, prefix, value, cnt):
    if value is not None:
        value /= cnt
        if tb_logger != None:
            tb_logger.add_scalar(prefix, value, step)
        log += f', {prefix}:{value:.6f}'
    return log

def test_logger(step, cnt, bpps, # both
    rgb_psnrs, rgb_msssims, rgb_msssimsDB, rgb_bpps, # rgb
    ir_psnrs, ir_msssims, ir_msssimsDB, ir_bpps): # ir

    logger.info("Test: model-{}".format(step))
    log = "Dataset Average result"
    log = create_test_log(log, step, 'bpp-test', bpps, cnt)
    log = create_test_log(log, step, 'bpp-test(rgb)', rgb_bpps, cnt)
    log = create_test_log(log, step, 'psnr-test(rgb)', rgb_psnrs, cnt)
    log = create_test_log(log, step, 'ms-ssim-test(rgb)', rgb_msssims, cnt)
    log = create_test_log(log, step, 'ms-ssim-db-test(ir)', ir_msssimsDB, cnt)
    log = create_test_log(log, step, 'bpp-test(ir)', ir_bpps, cnt)
    log = create_test_log(log, step, 'psnr-test(ir)', ir_psnrs, cnt)
    log = create_test_log(log, step, 'ms-ssim-test(ir)', ir_msssims, cnt)
    log = create_test_log(log, step, 'ms-ssim-db-test(ir)', ir_msssimsDB, cnt)
    
    logger.info(log)