import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 20}
matplotlib.rc('font', **font)
LineWidth = 3

# rgb
bpp = [0.0547, 0.0826, 0.1238, 0.1792, 0.2513, 0.3421]
msssim = [14.29, 15.599, 16.975, 18.377, 19.864, 21.334]
ours, = plt.plot(bpp, msssim, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.45305, 0.30056, 0.19486, 0.12622, 0.08254] #, 0.05418]
msssim = [20.554, 18.824, 17.210, 15.785, 14.430] #, 13.126]
bpg, = plt.plot(bpp, msssim, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.0576, 0.0869, 0.131, 0.1929, 0.2689, 0.3708]
msssim = [14.09, 15.437, 16.865, 18.243, 19.796, 21.451]
Minnen, = plt.plot(bpp, msssim, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM(dB)')
plt.title('FLIR dataset')
plt.savefig('FLIR_msssim.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('FLIR_msssim.png', format='png', dpi=300, bbox_inches='tight')

# ir
plt.close()
bpp = [0.1384, 0.2013, 0.3097, 0.5069]
msssim = [14.225, 15.526, 17.02, 18.981]
ours, = plt.plot(bpp, msssim, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.155, 0.2048, 0.301, 0.4979] #, 0.05418]
msssim = [13.984, 14.899, 15.952, 17.558] #, 13.126]
bpg, = plt.plot(bpp, msssim, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.1436, 0.2117, 0.318, 0.5016]
msssim = [14.225, 15.567, 17.037, 18.811]
Minnen, = plt.plot(bpp, msssim, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('MS-SSIM(dB)')
plt.title('FLIR dataset')
plt.savefig('FLIR_msssim_ir.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('FLIR_msssim_ir.png', format='png', dpi=300, bbox_inches='tight')