import os.path
import os
import sys
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import shutil

def parse_dir(ori, out):
    if not os.path.exists(out):
        os.mkdir(out)
    else:
        print("output path already exists.")
        sys.exit()

    ori_dir = {
        'train':
            [os.path.join(ori, f'set00/V00{k}') for k in range(9)] + \
            [os.path.join(ori, f'set01/V00{k}') for k in range(6)] + \
            [os.path.join(ori, f'set02/V00{k}') for k in range(5)] + \
            [os.path.join(ori, f'set03/V00{k}') for k in range(2)] + \
            [os.path.join(ori, f'set04/V00{k}') for k in range(2)] + \
            [os.path.join(ori, f'set05')] + \
            [os.path.join(ori, f'set07/V00{k}') for k in range(3)] + \
            [os.path.join(ori, f'set08/V00{k}') for k in range(3)] + \
            [os.path.join(ori, f'set09/V00{k}') for k in range(1)] + \
            [os.path.join(ori, f'set11/V00{k}') for k in range(2)],
        'test':
            [os.path.join(ori, f'set06/V00{k}') for k in range(5)] + \
            [os.path.join(ori, f'set10/V00{k}') for k in range(2)]
    }

    os.mkdir(out + '/train/')
    os.mkdir(out + '/test/')
    out_dir = {
        'train':{
            'rgb':out + '/train/RGB/', 
            'ir': out + '/train/IR/'},
        'test':{
            'rgb':out + '/test/RGB/', 
            'ir': out + '/test/IR/'}}
    os.mkdir(out_dir['train']['rgb'])
    os.mkdir(out_dir['train']['ir'])
    os.mkdir(out_dir['test']['rgb'])
    os.mkdir(out_dir['test']['ir'])
    return ori_dir, out_dir


def match_rgb_ir(home) -> list:
    rgb_names = os.listdir(os.path.join(home, 'visible'))
    ir_names = os.listdir(os.path.join(home, 'lwir'))
    return list(set(rgb_names).intersection(set(ir_names)))


def process_image(name_with_idx, home, out_dir): 
    name, idx = name_with_idx
    rgb_ori = os.path.join(home, 'visible', name)
    ir_ori = os.path.join(home, 'lwir', name)
    shutil.copyfile(rgb_ori, os.path.join(out_dir['rgb'], f'{idx}.jpg'))
    shutil.copyfile(ir_ori, os.path.join(out_dir['ir'], f'{idx}.jpg'))


if __name__ == '__main__': 
    ori_data_dir = sys.argv[1]
    output_path = sys.argv[2]
    k_train = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    k_test = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    ori_dir, out_dir = parse_dir(ori_data_dir, output_path)

    # train
    idx = 0
    for folder in ori_dir['train']:
        names = sorted(match_rgb_ir(folder)[::k_train])
        name_with_idx = [(name, idx+i) for i, name in enumerate(names)]
        idx += len(names)
        p_process = partial(process_image, home=folder, out_dir=out_dir['train'])
        with Pool() as p:
            res = list(tqdm(p.imap(p_process, name_with_idx), total=len(name_with_idx), desc=f'sampling {folder}'))

    # test
    idx = 0
    for folder in ori_dir['test']:
        names = sorted(match_rgb_ir(folder)[::k_test])
        name_with_idx = [(name, idx+i) for i, name in enumerate(names)]
        idx += len(names)
        p_process = partial(process_image, home=folder, out_dir=out_dir['test'])
        with Pool() as p:
            res = list(tqdm(p.imap(p_process, name_with_idx), total=len(name_with_idx), desc=f'sampling {folder}'))