from genericpath import exists
import os
import argparse
from posixpath import basename
import torch
import json
import logging
import wandb
import torchvision.utils as vutil
from functools import partial

gpu_num = torch.cuda.device_count()
logger = logging.getLogger("MDMC")


def create_train_log(tb_logger, log, global_step, prefix, value, k=5):
    if value is not None:
        tb_logger.add_scalar(prefix, value.avg, global_step)
        if k == 3:
            log += f' | {prefix} {value.val:.3f} ({value.avg:.3f})'
        else:
            log += f' | {prefix} {value.val:.5f} ({value.avg:.5f})'
    return log


def train_logger(tb_logger, global_step, tot_step, epoch, cur_lr, losses, elapsed, bpps, # both
    rgb_psnrs, rgb_mse_losses, rgb_bpps, rgb_bpp_zs, rgb_bpp_features, # rgb
    ir_psnrs, ir_mse_losses, ir_bpps, ir_bpp_zs, ir_bpp_features): # ir

    process = global_step / tot_step * 100.0
    log = f'Step [{global_step}/{tot_step}={process:.2f}%]'
    log += f' | Epoch {epoch}'
    if elapsed is not None:
        log += f' | time {elapsed.val:.3f} ({elapsed.avg:.3f})'
    if cur_lr is not None:
        tb_logger.add_scalar('lr', cur_lr, global_step) 
        log += f' | lr {cur_lr}'
    log = create_train_log(tb_logger, log, global_step, 'rd_loss', losses, 3)
    log = create_train_log(tb_logger, log, global_step, 'psnr(rgb)', rgb_psnrs, 3)
    log = create_train_log(tb_logger, log, global_step, 'psnr(ir)', ir_psnrs, 3)
    log = create_train_log(tb_logger, log, global_step, 'bpp', bpps)
    log = create_train_log(tb_logger, log, global_step, 'bpp(rgb)', rgb_bpps)
    log = create_train_log(tb_logger, log, global_step, 'bpp(ir)', ir_bpps)
    log = create_train_log(tb_logger, log, global_step, 'bpp_feature(rgb)', rgb_bpp_features)
    log = create_train_log(tb_logger, log, global_step, 'bpp_feature(ir)', ir_bpp_features)
    log = create_train_log(tb_logger, log, global_step, 'bpp_z(rgb)', rgb_bpp_zs)
    log = create_train_log(tb_logger, log, global_step, 'bpp_z(ir)', ir_bpp_zs)
    if rgb_mse_losses is not None:
        log += f' | rgb_mse_loss {rgb_mse_losses.val:.5f} ({rgb_mse_losses.avg:.5f})'
    if ir_mse_losses is not None:
        log += f' | ir_mse_loss {ir_mse_losses.val:.5f} ({ir_mse_losses.avg:.5f})'

    logger.info(log)

def create_test_log(tb_logger, log, step, prefix, value, cnt):
    if value is not None:
        value /= cnt
        if tb_logger != None:
            tb_logger.add_scalar(prefix, value, step)
        log += f', {prefix}:{value:.6f}'
    return log

def test_logger(tb_logger, step, cnt, bpps, # both
    rgb_psnrs, rgb_msssims, rgb_msssimsDB, rgb_bpps, # rgb
    ir_psnrs, ir_msssims, ir_msssimsDB, ir_bpps): # ir

    logger.info("Test: model-{}".format(step))
    log = "Dataset Average result"
    log = create_test_log(tb_logger, log, step, 'bpp-test', bpps, cnt)
    log = create_test_log(tb_logger, log, step, 'bpp-test(rgb)', rgb_bpps, cnt)
    log = create_test_log(tb_logger, log, step, 'psnr-test(rgb)', rgb_psnrs, cnt)
    log = create_test_log(tb_logger, log, step, 'ms-ssim-test(rgb)', rgb_msssims, cnt)
    log = create_test_log(tb_logger, log, step, 'ms-ssim-db-test(rgb)', rgb_msssimsDB, cnt)
    log = create_test_log(tb_logger, log, step, 'bpp-test(ir)', ir_bpps, cnt)
    log = create_test_log(tb_logger, log, step, 'psnr-test(ir)', ir_psnrs, cnt)
    log = create_test_log(tb_logger, log, step, 'ms-ssim-test(ir)', ir_msssims, cnt)
    log = create_test_log(tb_logger, log, step, 'ms-ssim-db-test(ir)', ir_msssimsDB, cnt)
    
    logger.info(log)


def save_model(model, iter, home):
    print("save at " + home + "/snapshot/iter_{}.pth.tar".format(iter))
    torch.save(model.state_dict(), home+"/snapshot/iter_{}.pth.tar".format(iter))


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


vis_idx = ''
log_dir = './visualize/log.txt'
def _vis_hook(module, input, output, name):
    def _mkdir():
        if not os.path.exists(path):
            os.mkdir(path)
    path = f'./visualize/{vis_idx}'
    _mkdir()
    path = f'./visualize/{vis_idx}/{name}'
    _mkdir()
    data = output.clone().detach()
    data = data.permute(1, 0, 2, 3)
    with open(log_dir, 'a') as f:
        f.write(f'{vis_idx}_{name}: {torch.mean(torch.abs(data)):.6f}\n')
    for i, d in enumerate(data):
        vutil.save_image(d, os.path.join(path, f'{i}.jpg'))


def _build_vis_hook(name):
    return partial(_vis_hook, name=name)


def build_vis_hook(model, vis_layers: dict):
    for n, m in model.named_modules():
        if n in vis_layers:
            m.register_forward_hook(_build_vis_hook(vis_layers[n]))


def set_vis_idx(idx):
    global vis_idx
    vis_idx = idx