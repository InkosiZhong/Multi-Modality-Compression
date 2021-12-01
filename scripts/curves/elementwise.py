import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
LineWidth = 2
	
'''bpp = [0.1247, 0.178, 0.2557, 0.3494]
psnr = [34.212, 36.109, 37.606, 38.983]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Ours')'''

bpp = [0.1275, 0.1845, 0.2614, 0.3544]
psnr = [33.637, 35.772, 37.162, 38.267]
ours_ca, = plt.plot(bpp, psnr, "g-s", linewidth=LineWidth, label='Ours(CA)')

'''bpp = [0.123, 0.179, 0.2617, 0.3657]
psnr = [33.128, 34.58, 36.029, 37.631]
Minnen, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='Minnen')'''

bpp = [0.2576, 0.3562,]
psnr = [29.948, 30.879]
ours_elem, = plt.plot(bpp, psnr, "b--s", linewidth=LineWidth, label='Ours(EA)')

plt.legend(handles=[ours_ca, ours_elem], loc=3)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('KAIST dataset')
plt.savefig('elementwise.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('elementwise.png', format='png', dpi=300, bbox_inches='tight')