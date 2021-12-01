import os.path
import os
import sys
from glob import glob
import random
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import shutil

seq = {
    'train':
        [f'set00/V00{k}' for k in range(9)] + \
        [f'set01/V00{k}' for k in range(6)] + \
        [f'set02/V00{k}' for k in range(5)] + \
        [f'set03/V00{k}' for k in range(2)] + \
        [f'set04/V00{k}' for k in range(2)] + \
        [f'set05'] + \
        [f'set07/V00{k}' for k in range(3)] + \
        [f'set08/V00{k}' for k in range(3)] + \
        [f'set09/V00{k}' for k in range(1)] + \
        [f'set11/V00{k}' for k in range(2)],
    'test':
        [f'set06/V00{k}' for k in range(5)] + \
        [f'set10/V00{k}' for k in range(2)]
}

def parse_dir(ori, out):
    if not os.path.exists(ori):
        print("input path not exists.")
        sys.exit()

    if not os.path.exists(out):
        os.mkdir(out)
    else:
        print("output path already exists.")
        sys.exit()

    ori_dir = {
        'train': [os.path.join(ori, s) for s in seq['train']],
        'test': [os.path.join(ori, s) for s in seq['test']]
    }

    os.mkdir(out + '/train/')
    os.mkdir(out + '/test/')
    os.mkdir(out + '/train/RGB')
    os.mkdir(out + '/train/IR')
    os.mkdir(out + '/test/RGB')
    os.mkdir(out + '/test/IR')
    out_dir = {
        'train': {
            'rgb': [os.path.join(out, 'train/RGB', s) for s in seq['train']],
            'ir': [os.path.join(out, 'train/IR', s) for s in seq['train']]
        },
        'test': {
            'rgb': [os.path.join(out, 'test/RGB', s) for s in seq['test']],
            'ir': [os.path.join(out, 'test/IR', s) for s in seq['test']]
        },
    }

    for dir in out_dir['train']['rgb']:
        os.makedirs(dir)
    for dir in out_dir['train']['ir']:
        os.makedirs(dir)
    for dir in out_dir['test']['rgb']:
        os.makedirs(dir)
        os.makedirs(dir + '/ref')
    for dir in out_dir['test']['ir']:
        os.makedirs(dir)
        os.makedirs(dir + '/ref')

    return ori_dir, out_dir


def match_rgb_ir(home) -> list:
    rgb_names = os.listdir(os.path.join(home, 'visible'))
    ir_names = os.listdir(os.path.join(home, 'lwir'))
    return list(set(rgb_names).intersection(set(ir_names)))


def process_image(name_with_idx, home, rgb_out_dir, ir_out_dir, t=0): 
    name, idx = name_with_idx
    rgb_ori = os.path.join(home, 'visible', name)
    ir_ori = os.path.join(home, 'lwir', name)
    shutil.copyfile(rgb_ori, os.path.join(rgb_out_dir, f'{idx:0>5d}.jpg'))
    shutil.copyfile(ir_ori, os.path.join(ir_out_dir, f'{idx:0>5d}.jpg'))
    if t and idx % t == 1:
        shutil.copyfile(rgb_ori, os.path.join(rgb_out_dir, f'ref/{idx:0>5d}.jpg'))
        shutil.copyfile(ir_ori, os.path.join(ir_out_dir, f'ref/{idx:0>5d}.jpg'))


def shuffle(output_path):
    image_path = []
    for folder in seq['train']:
        image_path += [os.path.join(folder, os.path.basename(p)) for p in glob(os.path.join(output_path, 'train/RGB', folder, '*.*'))]
    random.shuffle(image_path)
    out_file = os.path.join(output_path, 'train/KAIST.txt')
    with open(out_file, 'w') as f:
        f.write('\n'.join(image_path))


if __name__ == '__main__': 
    ori_data_dir = sys.argv[1]
    output_path = sys.argv[2]
    k_train = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    k_test = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    len_test = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    ori_dir, out_dir = parse_dir(ori_data_dir, output_path)

    # train
    for in_fd, rgb_fd, ir_fd in zip(ori_dir['train'], out_dir['train']['rgb'], out_dir['train']['ir']):
        names = sorted(match_rgb_ir(in_fd))[::k_train]
        name_with_idx = [(name, i + 1) for i, name in enumerate(names)]
        p_process = partial(process_image, home=in_fd, rgb_out_dir=rgb_fd, ir_out_dir=ir_fd)
        with Pool() as p:
            res = list(tqdm(p.imap(p_process, name_with_idx), total=len(name_with_idx), desc=f'copying (train) from {in_fd}'))
    shuffle(output_path)

    # test
    for in_fd, rgb_fd, ir_fd in zip(ori_dir['test'], out_dir['test']['rgb'], out_dir['test']['ir']):
        names = sorted(match_rgb_ir(in_fd))[:len_test*k_test:k_test]
        name_with_idx = [(name, i + 1) for i, name in enumerate(names)]
        p_process = partial(process_image, home=in_fd, rgb_out_dir=rgb_fd, ir_out_dir=ir_fd, t=10)
        with Pool() as p:
            res = list(tqdm(p.imap(p_process, name_with_idx), total=len(name_with_idx), desc=f'sampling (test) from {in_fd}'))

    out_file = os.path.join(output_path, 'test/KAIST.txt')
    with open(out_file, 'w') as f:
        f.write('\n'.join(seq['test']))