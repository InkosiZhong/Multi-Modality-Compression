import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 20}
matplotlib.rc('font', **font)
LineWidth = 3

# rgb
bpp = [0.0864, 0.12, 0.1699, 0.2389, 0.336]
psnr = [33.751, 35.344, 36.939, 38.359, 39.838]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.33392, 0.23219, 0.15530, 0.10642] #, 0.07362]
psnr = [39.302, 37.374, 35.452, 33.746] #, 32.101]
bpg, = plt.plot(bpp, psnr, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.0811, 0.1162, 0.1694, 0.2469, 0.351]
psnr = [33.222, 34.574, 36.07, 37.728, 38.988]
Minnen, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('KAIST dataset')
plt.savefig('KAIST_psnr_rgb.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('KAIST_psnr_rgb.png', format='png', dpi=300, bbox_inches='tight')

# ir
plt.close()
bpp = [0.0635, 0.0883, 0.1254, 0.1635, 0.2354]
psnr = [39.813, 41.556, 42.994, 44.42, 45.915]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.26553, 0.19053, 0.13779, 0.10004, 0.07084] #, 0.05099]
psnr = [44.907, 43.753, 42.487, 41.107, 39.593] #, 38.028]
bpg, = plt.plot(bpp, psnr, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.0657, 0.0927, 0.1339, 0.1891, 0.2551]
psnr = [39.256, 41.009, 42.779, 44.304, 45.604]
Minnen, = plt.plot(bpp, psnr, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('KAIST dataset')
plt.savefig('KAIST_psnr_ir.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('KAIST_psnr_ir.png', format='png', dpi=300, bbox_inches='tight')