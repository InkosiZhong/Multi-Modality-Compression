import os.path
import os
import sys
from glob import glob
import random
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import shutil
from PIL import Image

seq = \
    [f'set06/V00{k}' for k in range(5)] + \
    [f'set10/V00{k}' for k in range(2)]


def process_image(name, home):
    img = Image.open(os.path.join(home, name))
    img.save(os.path.join(home, f'{name[:-4]}.png'))

if __name__ == '__main__': 
    home = 'KAIST_video/test_x265/RGB'
    for s in seq:
        ori = os.path.join(home, s)
        names = sorted([f for f in os.listdir(ori) if f[-3:] == 'jpg'])
        p_process = partial(process_image, home=ori)
        with Pool() as p:
            res = list(tqdm(p.imap(p_process, names), total=len(names), desc=f'converting {ori}'))