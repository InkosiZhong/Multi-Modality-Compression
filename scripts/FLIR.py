import os.path
import os
import random
import sys
from PIL import Image
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

def parse_dir(ori, out):
    if not os.path.exists(out):
        os.mkdir(out)
    else:
        print("output path already exists.")
        sys.exit()

    ori_dir = {
        'train':{
            'rgb':ori + '/train/RGB/', 
            'ir': ori + '/train/thermal_8_bit/'}, 
        'val':{
            'rgb':ori + '/val/RGB/', 
            'ir': ori + '/val/thermal_8_bit/'}}

    os.mkdir(out + '/train/')
    os.mkdir(out + '/val/')
    out_dir = {
        'train':{
            'rgb':out + '/train/RGB/', 
            'ir': out + '/train/IR/'}, 
        'val':{
            'rgb':out + '/val/RGB/', 
            'ir': out + '/val/IR/'}}
    os.mkdir(out_dir['train']['rgb'])
    os.mkdir(out_dir['train']['ir'])
    os.mkdir(out_dir['val']['rgb'])
    os.mkdir(out_dir['val']['ir'])
    return ori_dir, out_dir


def match_rgb_ir(root: dict) -> list:
    rgb_names = os.listdir(root['rgb'])
    ir_names = os.listdir(root['ir'])
    rgb_names = [i[:-4] for i in rgb_names] # jpg
    ir_names = [i[:-5] for i in ir_names] # jpeg
    return list(set(rgb_names).intersection(set(ir_names)))


def process_image(name, ori_dir, out_dir, ir_patch_size, size_k, patch_k, train=True): # set train = False to forbidden random_crop
    key = 'train' if train else 'val'
    rgb_img = Image.open(ori_dir[key]['rgb']+name+'.jpg')
    ir_img = Image.open(ori_dir[key]['ir']+name+'.jpeg')

    # resize
    rgb_img_size = [ir_img.width*size_k, ir_img.height*size_k]
    rgb_img = rgb_img.resize(rgb_img_size, Image.BICUBIC)
    if not train:
        ir_patch_path = out_dir[key]['ir']+name+".png"
        ir_img.save(ir_patch_path)
        rgb_patch_path = out_dir[key]['rgb']+name+".png"
        rgb_img.save(rgb_patch_path)
        return

    # random crop
    rgb_patch_size = [size * 2 for size in ir_patch_size]
    sample_num= int(ir_img.height * ir_img.width / ir_patch_size[0] / ir_patch_size[1]) * patch_k
    for sample_id in range(sample_num):
        random_resize_factor = random.random() * 0.4 + 0.6  #random 0.6 - 1.0 resize
        crop_size = [round(ir_patch_size[0] / random_resize_factor), round(ir_patch_size[1] / random_resize_factor)]

        random_crop_x1 = int(random.random() * (ir_img.width - crop_size[1] - 2))
        random_crop_y1 = int(random.random() * (ir_img.height - crop_size[0] - 2))
        random_crop_x2 = random_crop_x1 + crop_size[1]
        random_crop_y2 = random_crop_y1 + crop_size[0]

        random_ir_box = (random_crop_x1, random_crop_y1, random_crop_x2, random_crop_y2)

        ir_crop_patch = ir_img.crop(random_ir_box)
        ir_crop_patch = ir_crop_patch.resize(ir_patch_size, Image.BICUBIC)
        ir_patch_path = out_dir[key]['ir'] + name + "_%04d.png" % (sample_id)
        ir_crop_patch.save(ir_patch_path)

        random_rgb_box = (random_crop_x1*size_k, random_crop_y1*size_k, random_crop_x2*size_k, random_crop_y2*size_k)

        rgb_crop_patch = rgb_img.crop(random_rgb_box)
        rgb_crop_patch = rgb_crop_patch.resize(rgb_patch_size, Image.BICUBIC)
        rgb_patch_path = out_dir[key]['rgb'] + name + "_%04d.png" % (sample_id)
        rgb_crop_patch.save(rgb_patch_path)

if __name__ == '__main__': 
    ori_data_dir = sys.argv[1]
    output_path = sys.argv[2]
    ori_dir, out_dir = parse_dir(ori_data_dir, output_path)

    train_data = match_rgb_ir(ori_dir['train'])
    val_data = match_rgb_ir(ori_dir['val'])

    ir_patch_size = [128, 128]
    size_k = 2
    patch_k = 4

    # val
    p_process = partial(process_image, ori_dir=ori_dir, out_dir=out_dir, \
        ir_patch_size=ir_patch_size, size_k=size_k, patch_k=patch_k, train=False)
    with Pool() as p:
        res = list(tqdm(p.imap(p_process, val_data), total=len(val_data), desc='  val'))

    # train
    p_process = partial(process_image, ori_dir=ori_dir, out_dir=out_dir, \
        ir_patch_size=ir_patch_size, size_k=size_k, patch_k=patch_k)
    with Pool() as p:
        res = list(tqdm(p.imap(p_process, train_data), total=len(train_data), desc='train'))