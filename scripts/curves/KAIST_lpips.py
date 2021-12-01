import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 20}
matplotlib.rc('font', **font)
LineWidth = 3

# ir
plt.close()
bpp = [0.0635, 0.0883, 0.1254, 0.1635, 0.2354]
lpips = [0.172, 0.146, 0.122, 0.110, 0.097]
ours, = plt.plot(bpp, lpips, "k-o", linewidth=LineWidth, label='Ours')

#bpp = [0.26553, 0.19053, 0.13779, 0.10004, 0.07084] #, 0.05099]
#lpips = [] #, 282.361]
#bpg, = plt.plot(bpp, lpips, "m--s", linewidth=LineWidth, label='BPG (4:4:4)')

bpp = [0.0657, 0.0927, 0.1339, 0.1891, 0.2551]
lpips = [0.169, 0.139, 0.115, 0.113, 0.105]
Minnen, = plt.plot(bpp, lpips, "r-v", linewidth=LineWidth, label='Minnen')


plt.legend(handles=[Minnen, ours], loc=1)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('LPIPS')
plt.title('KAIST dataset')
plt.savefig('KAIST_lpips_ir.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('KAIST_lpips_ir.png', format='png', dpi=300, bbox_inches='tight')