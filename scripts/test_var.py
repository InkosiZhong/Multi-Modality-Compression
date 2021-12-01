import os.path
import os
import sys
from PIL import Image, ImageStat
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import matplotlib.pyplot as plt

blacklist = ['.DS_Store']

def parse_dir(ori):
    ori_dir = {
        'train':{
            'rgb':ori + '/train/RGB/', 
            'ir': ori + '/train/IR/'}}
    return ori_dir


def match_rgb_ir(root: dict) -> list:
    rgb_names = os.listdir(root['rgb'])
    ir_names = os.listdir(root['ir'])
    rgb_names = [i[:-4] for i in rgb_names if i not in blacklist] # png
    ir_names = [i[:-4] for i in ir_names if i not in blacklist] # png
    return list(set(rgb_names).intersection(set(ir_names)))


def process_image(name, ori_dir):
    ir_img = Image.open(ori_dir['train']['ir']+name+'.png')
    stat = ImageStat.Stat(ir_img)
    return stat.var[0]


if __name__ == '__main__': 
    ori_data_dir = sys.argv[1]
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 200

    ori_dir = parse_dir(ori_data_dir)

    train_data = match_rgb_ir(ori_dir['train'])

    # train
    p_process = partial(process_image, ori_dir=ori_dir)
    with Pool() as p:
        res = list(tqdm(p.imap(p_process, train_data), total=len(train_data), desc='testing var'))
    
    low = len([i for i in res if i < k])
    print(f'for k = {k}, {low}/{len(train_data)}, ({low / len(train_data)})')
    #plt.hist(res, bins=20)
    #plt.show()