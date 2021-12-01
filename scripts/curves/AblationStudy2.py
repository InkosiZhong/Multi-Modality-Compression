import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
LineWidth = 3
	
bpp = [0.1275, 0.1845, 0.2614]
psnr = [33.637, 35.772, 37.162]
ours_ca, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Ours(CA)')

bpp = [0.1853, 0.2619]
psnr = [35.476, 36.967]
ours_diff, = plt.plot(bpp, psnr, "g--p", linewidth=LineWidth, label='Ours(CR)')

bpp = [0.2601]
psnr = [29.314]
ours_elem, = plt.plot(bpp, psnr, "m--s", linewidth=LineWidth, label='Ours(EA)')

bpp = [0.0854, 0.123, 0.179, 0.2617]
psnr = [31.726, 33.128, 34.58, 36.029]
Minnen, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[ours_ca, ours_diff, ours_elem, Minnen], loc=2)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('KAIST dataset')
plt.savefig('Ablation2.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('Ablation2.png', format='png', dpi=300, bbox_inches='tight')