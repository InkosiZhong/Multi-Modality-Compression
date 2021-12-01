import os.path
import os
import sys
from PIL import Image, ImageStat
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

blacklist = ['.DS_Store']

def parse_dir(ori, out):
    if not os.path.exists(out):
        os.mkdir(out)
    else:
        print("output path already exists.")
        sys.exit()

    ori_dir = {
        'train':{
            'rgb':ori + '/train/RGB/', 
            'ir': ori + '/train/IR/'}}

    os.mkdir(out + '/train/')
    out_dir = {
        'train':{
            'rgb':out + '/train/RGB/', 
            'ir': out + '/train/IR/'}}
    os.mkdir(out_dir['train']['rgb'])
    os.mkdir(out_dir['train']['ir'])
    return ori_dir, out_dir


def match_rgb_ir(root: dict) -> list:
    rgb_names = os.listdir(root['rgb'])
    ir_names = os.listdir(root['ir'])
    rgb_names = [i[:-4] for i in rgb_names if i not in blacklist] # png
    ir_names = [i[:-4] for i in ir_names if i not in blacklist] # png
    return list(set(rgb_names).intersection(set(ir_names)))


def process_image(name, ori_dir, out_dir, k):
    rgb_img = Image.open(ori_dir['train']['rgb']+name+'.png')
    ir_img = Image.open(ori_dir['train']['ir']+name+'.png')

    stat = ImageStat.Stat(ir_img)
    if stat.var[0] > k: # has context
        ir_patch_path = out_dir['train']['ir']+name+".png"
        ir_img.save(ir_patch_path)
        rgb_patch_path = out_dir['train']['rgb']+name+".png"
        rgb_img.save(rgb_patch_path)
    
    return stat.var[0] > k


if __name__ == '__main__': 
    ori_data_dir = sys.argv[1]
    output_path = sys.argv[2]
    k = float(sys.argv[3]) if len(sys.argv) > 3 else 200
    ori_dir, out_dir = parse_dir(ori_data_dir, output_path)

    train_data = match_rgb_ir(ori_dir['train'])

    # train
    p_process = partial(process_image, ori_dir=ori_dir, out_dir=out_dir, k=k)
    with Pool() as p:
        res = list(tqdm(p.imap(p_process, train_data), total=len(train_data), desc='cleaning'))

    low = len([i for i in res if not i])
    print(f'for k = {k}, {low}/{len(train_data)}, ({low / len(train_data)})')