import os
from glob import glob
import random
import sys

def shuffle(src_path: str, out_path: str):
    rgb_path = src_path + '/RGB/'
    if not os.path.exists(rgb_path):
        raise Exception(f"[!] {rgb_path} not exitd")
    
    image_path = sorted(glob(os.path.join(rgb_path, "*.*")))
    image_path = [os.path.basename(p) for p in image_path]
    random.shuffle(image_path)
    with open(out_path, 'w') as f:
        f.write('\n'.join(image_path))


if __name__ == '__main__': 
    src_path = sys.argv[1]
    out_path = sys.argv[2]
    shuffle(src_path, out_path)