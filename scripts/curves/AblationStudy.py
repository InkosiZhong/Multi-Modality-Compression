import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
LineWidth = 2
	
'''bpp = [0.1247, 0.178, 0.2557, 0.3494]
psnr = [34.212, 36.109, 37.606, 38.983]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.1275, 0.1845, 0.2614, 0.3544]
psnr = [33.637, 35.772, 37.162, 38.267]
ours_ca, = plt.plot(bpp, psnr, "g-s", linewidth=LineWidth, label='Ours(CA)')

bpp = [0.1313, 0.1853, 0.2619, 0.3620]
psnr = [33.867, 35.476, 36.967, 38.228]
ours_diff, = plt.plot(bpp, psnr, "c-p", linewidth=LineWidth, label='Ours(Res)')

bpp = [0.1181, 0.178, 0.2557, 0.3602]
psnr = [33.671, 35.219, 36.576, 37.82]
ours_sa, = plt.plot(bpp, psnr, "m-s", linewidth=LineWidth, label='Ours(SA)')

bpp = [0.1232, 0.1807, 0.2638, 0.3676]
psnr = [33.323, 34.947, 36.454, 37.928]
ours_ct, = plt.plot(bpp, psnr, "y-p", linewidth=LineWidth, label='Ours(Cat)')

bpp = [0.123, 0.179, 0.2617, 0.3657]
psnr = [33.128, 34.58, 36.029, 37.631]
Minnen, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='Minnen')'''

bpp = [0.12, 0.1699, 0.2389, 0.336]
psnr = [35.344, 36.939, 38.359, 39.838]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.1226, 0.1765, 0.2502, 0.3473]
psnr = [35.125, 36.615, 38.243, 39.629]
ours_ca, = plt.plot(bpp, psnr, "c-s", linewidth=LineWidth, label='Ours(CA)')

bpp = [0.1252, 0.1781, 0.2513, 0.3539]
psnr = [34.996, 36.48, 38.19, 39.505]
ours_diff, = plt.plot(bpp, psnr, "g-p", linewidth=LineWidth, label='Ours(Res)')

bpp = [0.1087, 0.1605, 0.2326, 0.3301]
psnr = [34.741, 36.147, 37.703, 38.901]
ours_sa, = plt.plot(bpp, psnr, "m-s", linewidth=LineWidth, label='Ours(SA)')

bpp = [0.1149, 0.1674, 0.2454, 0.3485]
psnr = [34.632, 36.074, 37.649, 38.987]
ours_ct, = plt.plot(bpp, psnr, "y-p", linewidth=LineWidth, label='Ours(Cat)')

bpp = [0.1162, 0.1694, 0.2469, 0.351]
psnr = [34.574, 36.07, 37.728, 38.988]
Minnen, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='Minnen')

'''bpp = [0.2576, 0.3562,]
psnr = [29.948, 30.879]
ours_elem, = plt.plot(bpp, psnr, "b--s", linewidth=LineWidth, label='Ours(EA)')'''

plt.legend(handles=[ours, ours_ca, ours_diff, ours_sa, ours_ct, Minnen], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('KAIST dataset')
plt.savefig('Ablation.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('Ablation.png', format='png', dpi=300, bbox_inches='tight')