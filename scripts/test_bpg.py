import os 
import sys
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

input_dir = "./kodak/input"
encode_dir = "./kodak/encode"
decode_dir = "./kodak/decode"

def psnr(img1, img2):
	mse = np.mean((img1/1. - img2/1.) ** 2)
	if mse < 1.0e-10:
		return 100
	return 10 * math.log10(255.**2/mse)

bpps = []
psnrs = []

def bpg_test(img_name, q):
	img_dir = input_dir + '/' + img_name
	input_img = Image.open(img_dir)
	input_w, input_h = input_img.size
	input_c = len(input_img.split())
	if input_c == 3:
		input_img = input_img.convert('RGB')
	elif input_c == 1:
		input_img = input_img.convert('L')
	input_img = np.array(input_img)

	os.system(f'bpgenc -q {q} -f 444 {img_dir} -o {encode_dir}/{img_name.split(".")[0]}.bpg')
	os.system(f'bpgdec -o {decode_dir}/{img_name} {encode_dir}/{img_name.split(".")[0]}.bpg')

	size = os.path.getsize(f'{encode_dir}/{img_name.split(".")[0]}.bpg') * 8
	output_img = Image.open(f"{decode_dir}/{img_name}").convert('RGB')
	if input_c == 3:
		output_img = output_img.convert('RGB')
	elif input_c == 1:
		output_img = output_img.convert('L')
	output_img = np.array(output_img)
	bpps.append(size / input_h / input_w)
	psnrs.append(psnr(input_img, output_img))

if __name__ == '__main__': 
	if len(sys.argv) > 1:
		input_dir = sys.argv[1]
	if len(sys.argv) > 2:
		encode_dir = sys.argv[2]
	else:
		encode_dir = input_dir + '/encoder'
	if len(sys.argv) > 3:
		decode_dir = sys.argv[3] + '/decoder'
	if len(sys.argv) > 4:
		the_bpp, the_psnr = sys.argv[4].split(',')
		plt.plot(float(the_bpp), float(the_psnr))
	
	if not os.path.exists(encode_dir):
		os.mkdir(encode_dir)
	if not os.path.exists(decode_dir):
		os.mkdir(decode_dir)

	img_names = os.listdir(input_dir)
	all_bpps = []
	all_psnrs = []
	for q in range(0, 52, 4):
		bpps.clear()
		psnrs.clear()

		for img_name in img_names:
			bpg_test(img_name, q)
		
		all_bpps.append(np.mean(bpps))
		all_psnrs.append(np.mean(psnrs))
		print(f'mean(qp={q}) 	bpp: {np.mean(bpps):.5f} 	psnr: {np.mean(psnrs):.3f}')
	
	plt.plot(all_bpps, all_psnrs)
	plt.savefig(f'bpg_test.png', dpi=120)
	plt.show()