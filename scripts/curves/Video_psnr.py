import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
LineWidth = 3

bpp = [0.0779, 0.1165, 0.1877, 0.2879]
psnr = [35.522, 37.115, 39.219, 41.444]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Ours')
	
bpp = [0.065, 0.1083, 0.1774, 0.2974]
psnr = [34.791, 36.443, 38.034, 40.594]
FVC, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='FVC')

bpp = [0.1111, 0.176, 0.2777]
psnr = [34.712, 35.989, 37.367]
h265, = plt.plot(bpp, psnr, "m-s", linewidth=LineWidth, label='H265')

'''bpp = [0.0734, 0.1201, 0.2157]
psnr = [35.148, 37.104, 39.135]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.065, 0.1083, 0.1774, 0.2974]
psnr = [34.791, 36.443, 38.034, 40.594]
FVC, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='FVC')'''


plt.legend(handles=[ours, FVC, h265], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('KAIST dataset (Video)')
plt.savefig('Video.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('Video.png', format='png', dpi=300, bbox_inches='tight')