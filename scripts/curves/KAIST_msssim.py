import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 20}
matplotlib.rc('font', **font)
LineWidth = 3

# rgb
bpp = [0.1754, 0.2494, 0.3503, 0.4829] # 0.0825, 0.1209, 
msssim = [17.861, 19.632, 21.601, 23.854] # 14.59, 16.164, 
ours, = plt.plot(bpp, msssim, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.46721, 0.33392, 0.23219] #, 0.15530] #, 0.10642] #, 0.07362]
msssim = [20.307, 18.324, 16.321] #, 14.573] #, 13.113] #, 11.852]
bpg, = plt.plot(bpp, msssim, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.1668, 0.2456, 0.354, 0.4984] # 0.0738, 0.1117, 
msssim = [17.476, 19.231, 21.218, 23.261] # 14.1794, 15.756, 
Minnen, = plt.plot(bpp, msssim, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM(dB)')
plt.title('KAIST dataset')
plt.savefig('KAIST_msssim_rgb.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('KAIST_msssim_rgb.png', format='png', dpi=300, bbox_inches='tight')

# ir
plt.close()
bpp = [0.0635, 0.0883, 0.1254, 0.1635, 0.2354]
msssim = [16.791, 18.466, 19.988, 21.342, 23.004]
ours, = plt.plot(bpp, msssim, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.26553, 0.19053, 0.13779, 0.10004, 0.07084] #, 0.05099]
msssim = [23.001, 21.471, 19.973, 18.424, 16.853] #, 15.264]
bpg, = plt.plot(bpp, msssim, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.0657, 0.0927, 0.1339, 0.1891, 0.2551]
msssim = [16.297, 18.044, 19.812, 21.147, 22.325]
Minnen, = plt.plot(bpp, msssim, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM(dB)')
plt.title('KAIST dataset')
plt.savefig('KAIST_msssim_ir.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('KAIST_msssim_ir.png', format='png', dpi=300, bbox_inches='tight')