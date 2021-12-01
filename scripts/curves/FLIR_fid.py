import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 20}
matplotlib.rc('font', **font)
LineWidth = 3

# rgb
bpp = [0.0547, 0.0826, 0.1238, 0.1792, 0.2513, 0.3421]
fid = [67.071, 46.400, 31.831, 21.812, 12.605, 8.419]
ours, = plt.plot(bpp, fid, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.45305, 0.30056, 0.19486, 0.12622, 0.08254] #, 0.05418]
fid = [8.921, 13.894, 24.744, 38.622, 60.827] #, 89.247]
bpg, = plt.plot(bpp, fid, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.0576, 0.0869, 0.131, 0.1929, 0.2689, 0.3708]
fid = [70.570, 46.647, 31.925, 22.985, 14.331, 8.19]
Minnen, = plt.plot(bpp, fid, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=1)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('FID')
plt.title('FLIR dataset')
plt.savefig('FLIR_fid.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('FLIR_fid.png', format='png', dpi=300, bbox_inches='tight')


# ir
plt.close()
bpp = [0.1384, 0.2013, 0.3097, 0.5069] #, 0.7547]
fid = [134.097, 101.776, 61.384, 36.365] #, 20.686]
ours, = plt.plot(bpp, fid, "k-o", linewidth=LineWidth, label='Ours')

#bpp = [0.155, 0.2048, 0.301, 0.4979] #, 0.7298]
#fid = [117.863, 99.479, 76.871, 48.912] #, 32.739]
bpp = [0.2048, 0.3010, 0.3879, 0.4979]
fid = [99.479, 76.871, 61.379, 48.912] #, 32.739]
bpg, = plt.plot(bpp, fid, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.1436, 0.2117, 0.318, 0.5016] #, 0.6840]
fid = [134.899, 99.943, 62.755, 41.151] #, 22.004]
Minnen, = plt.plot(bpp, fid, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=1)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('FID')
plt.title('FLIR dataset')
plt.savefig('FLIR_fid_ir.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('FLIR_fid_ir.png', format='png', dpi=300, bbox_inches='tight')