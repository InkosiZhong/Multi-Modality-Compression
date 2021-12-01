import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 20}
matplotlib.rc('font', **font)
LineWidth = 3

# rgb
bpp = [0.0547, 0.0826, 0.1238, 0.1792, 0.2513, 0.3421]
psnr = [35.092, 36.626, 38.103, 39.471, 40.704, 41.734]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.45305, 0.30056, 0.19486, 0.12622, 0.08254] #, 0.05418]
psnr = [41.725, 40.032, 38.412, 36.877, 35.377] #, 33.904]
bpg, = plt.plot(bpp, psnr, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.0576, 0.0869, 0.131, 0.1929, 0.2689, 0.3708]
psnr = [34.972, 36.435, 37.919, 39.299, 40.606, 41.747]
Minnen, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('FLIR dataset')
plt.savefig('FLIR_psnr.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('FLIR_psnr.png', format='png', dpi=300, bbox_inches='tight')

# ir
plt.close()
bpp = [0.0923, 0.1384, 0.2013, 0.3097]
psnr = [32.283, 33.382, 34.378, 35.39]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.1188, 0.155, 0.2048, 0.301]
psnr = [32.272, 33.034, 33.799, 34.74]
bpg, = plt.plot(bpp, psnr, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.0986, 0.1436, 0.2117, 0.318]
psnr = [32.148, 33.262, 34.323, 35.323]
Minnen, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('FLIR dataset')
plt.savefig('FLIR_psnr_ir.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('FLIR_psnr_ir.png', format='png', dpi=300, bbox_inches='tight')