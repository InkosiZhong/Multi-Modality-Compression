import os 
import numpy as np
import imageio
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

seq = ['C_BD', 'C_BQ', 'C_PS', 'C_RH']

def crop(name, input_s):
	img = imageio.imread(os.path.join(input_s, name))
	h = (img.shape[0] // 64) * 64
	w = (img.shape[1] // 64) * 64
	img = np.array(img)[:h, :w, :]
	imageio.imsave(os.path.join(input_s, 'crop', name), img)

if __name__ == '__main__': 
	home = './HEVC'
	N = 100
	for s in seq:
		ori = os.path.join(home, s)
		out = os.path.join(ori, 'crop')
		if not os.path.exists(out):
			os.mkdir(out)
		names = sorted([f for f in os.listdir(ori) if f[-3:] == 'png'])[:N]
		p_process = partial(crop, input_s=ori)
		with Pool() as p:
			res = list(tqdm(p.imap(p_process, names), total=len(names), desc=f'crop {ori}'))