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
            'ir': ori + '/train/IR/'}}

    os.mkdir(out + '/train/')
    out_dir = {
        'train':{
            'rgb':out + '/train/RGB/', 
            'ir': out + '/train/IR/'}}
    os.mkdir(out_dir['train']['rgb'])
    os.mkdir(out_dir['train']['ir'])
    return ori_dir, out_dir


def process_image(name, ori_dir, out_dir, ir_patch_size, size_k, patch_k): 
    key = 'train'
    rgb_img = Image.open(ori_dir[key]['rgb']+name+'.jpg')
    ir_img = Image.open(ori_dir[key]['ir']+name+'.jpg')

    # resize
    rgb_img_size = [ir_img.width*size_k, ir_img.height*size_k]
    rgb_img = rgb_img.resize(rgb_img_size, Image.BICUBIC)

    # random crop
    rgb_patch_size = [size * size_k for size in ir_patch_size]
    sample_num= int(ir_img.height * ir_img.width / ir_patch_size[0] / ir_patch_size[1]) * patch_k
    for sample_id in range(sample_num):
        # resize
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

        random_rgb_box = (random_crop_x1*size_k, random_crop_y1*size_k, random_crop_x2*size_k, random_crop_y2*size_k)

        rgb_crop_patch = rgb_img.crop(random_rgb_box)
        rgb_crop_patch = rgb_crop_patch.resize(rgb_patch_size, Image.BICUBIC)
        rgb_patch_path = out_dir[key]['rgb'] + name + "_%04d.png" % (sample_id)

        #flip
        random_flip = random.randint(0, 3)
        if random_flip in [1, 3]:
            ir_crop_patch = ir_crop_patch.transpose(Image.FLIP_LEFT_RIGHT)
            rgb_crop_patch = rgb_crop_patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random_flip in [2, 3]:
            ir_crop_patch = ir_crop_patch.transpose(Image.FLIP_TOP_BOTTOM)
            rgb_crop_patch = rgb_crop_patch.transpose(Image.FLIP_TOP_BOTTOM)

        ir_crop_patch.save(ir_patch_path)
        rgb_crop_patch.save(rgb_patch_path)


if __name__ == '__main__': 
    ori_data_dir = sys.argv[1]
    output_path = sys.argv[2]
    ori_dir, out_dir = parse_dir(ori_data_dir, output_path)

    train_data = [name[:-4] for name in os.listdir(ori_dir['train']['rgb'])]

    ir_patch_size = [256, 256]
    size_k = 1
    patch_k = 4

    # train
    p_process = partial(process_image, ori_dir=ori_dir, out_dir=out_dir, \
        ir_patch_size=ir_patch_size, size_k=size_k, patch_k=patch_k)
    with Pool() as p:
        res = list(tqdm(p.imap(p_process, train_data), total=len(train_data), desc='train'))