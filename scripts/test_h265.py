import os 
import sys
import numpy as np
import math
from PIL import Image

input_dir = "./KAIST_video/test_x265/RGB"
log_dir = "./result.txt"

seq = \
    [f'set06/V00{k}' for k in range(5)] + \
    [f'set10/V00{k}' for k in range(2)]

def psnr(img1, img2):
	mse = np.mean((img1/1. - img2/1.) ** 2)
	if mse < 1.0e-10:
		return 100
	return 10 * math.log10(255.**2/mse)


def x265_test(input_s, qp):
	psnrs = []
	encode_dir = input_s + '/encoder/output.mkv'
	if os.path.exists(encode_dir):
		os.remove(encode_dir)
	decode_dir = input_s + '/decoder'
	names = sorted([f for f in os.listdir(input_s) if f[-3:] == 'png'])
	src = Image.open(os.path.join(input_s, names[0]))
	w, h = src.size
	c = len(src.split())
	os.system(f'ffmpeg -pix_fmt yuv420p -s {w}x{h} -r 10 -i {input_s}/%05d.png -vframes 100 -c:v libx265 -tune zerolatency -x265-params "qp={qp}:keyint=10" {encode_dir}')
	os.system(f'ffmpeg -i {encode_dir} {decode_dir}/%05d.png')
	size = os.path.getsize(encode_dir) * 8
	for name in names:
		src = Image.open(os.path.join(input_s, name))
		dec = Image.open(os.path.join(decode_dir, name))
		if c == 3:
			src = src.convert('RGB')
			dec = dec.convert('RGB')
		elif c == 1:
			src = src.convert('L')
			dec = dec.convert('L')
		src = np.array(src)
		dec = np.array(dec)
		psnrs.append(psnr(src, dec))

	bpp = size / h / w / len(names)
	return np.mean(psnrs), bpp

if __name__ == '__main__': 
	if len(sys.argv) > 1:
		input_dir = sys.argv[1]

	all_psnrs = []
	all_bpps = []
	for s in seq:
		input_s = os.path.join(input_dir, s)
		encode_dir = input_s + '/encoder'
		decode_dir = input_s + '/decoder'
		log_dir = input_s + '/result.txt'
		
		if not os.path.exists(encode_dir):
			os.mkdir(encode_dir)
		if not os.path.exists(decode_dir):
			os.mkdir(decode_dir)
		if os.path.exists(log_dir):
			os.remove(log_dir)

		psnrs = []
		bpps = []
		for qp in range(22, 42, 5):
			mean_psnr, bpp = x265_test(input_s, qp)
			psnrs.append(mean_psnr)
			bpps.append(bpp)
			print(f'mean(qp={qp}) 	bpp: {bpp:.5f} 	psnr: {mean_psnr:.3f}')
			with open(log_dir, 'a') as log:
				log.write(f'{bpp:.5f},{mean_psnr:.3f}\n')
				
		all_psnrs.append(psnrs)
		bpps.append(bpps)
	
	all_psnrs = np.mean(all_psnrs, 0)
	all_bpps = np.mean(all_bpps, 0)
	print(all_psnrs)
	print(all_bpps)