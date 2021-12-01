import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 20}
matplotlib.rc('font', **font)
LineWidth = 3

# rgb
plt.close()
bpp = [0.12, 0.1699, 0.2389, 0.336] # 0.0864, 
fid = [75.248, 50.473, 33.183, 18.178] # 106.411, 
ours, = plt.plot(bpp, fid, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.33392, 0.23219, 0.15530, 0.10642] #, 0.07362]
fid = [24.596, 40.899, 63.993, 92.383] #, 127.683]
bpg, = plt.plot(bpp, fid, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.1162, 0.1694, 0.2469, 0.351] # 0.0811, 
fid = [82.515, 56.349, 33.522, 19.292] # 103.656, 
Minnen, = plt.plot(bpp, fid, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=1)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('FID')
plt.title('KAIST dataset')
plt.savefig('KAIST_fid_rgb.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('KAIST_fid_rgb.png', format='png', dpi=300, bbox_inches='tight')

# ir
plt.close()
bpp = [0.0883, 0.1254, 0.1635, 0.2354] # 0.0635
fid = [152.169, 107.52, 93.4741, 74.353] # 190.453
ours, = plt.plot(bpp, fid, "k-o", linewidth=LineWidth, label='Ours')

bpp = [0.26553, 0.19053, 0.13779, 0.10004] #, 0.07084] #, 0.05099]
fid = [68.027, 88.157, 120.818, 180.515] #, 223.923] #, 282.361]
bpg, = plt.plot(bpp, fid, "m-s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.0927, 0.1339, 0.1891, 0.2551] # 0.0657
fid = [146.215, 115.344, 90.268, 82.466] # 172.570
Minnen, = plt.plot(bpp, fid, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[bpg, Minnen, ours], loc=1)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('FID')
plt.title('KAIST dataset')
plt.savefig('KAIST_fid_ir.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('KAIST_fid_ir.png', format='png', dpi=300, bbox_inches='tight')