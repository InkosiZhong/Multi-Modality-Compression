import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Arial', 'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
LineWidth = 2

'''bpp = [0.12, 0.1699, 0.2389, 0.336]
psnr = [35.344, 36.939, 38.359, 39.838]
ours, = plt.plot(bpp, psnr, "k-o", linewidth=LineWidth, label='Ours')'''

bpp = [0.1135, 0.1645, 0.2358, 0.3282]
psnr = [34.933, 36.485, 37.805, 38.958]
ours_sa6, = plt.plot(bpp, psnr, "g-p", linewidth=LineWidth, label='Ours(SA-6)')

bpp = [0.1087, 0.1605, 0.2326, 0.3301]
psnr = [34.741, 36.147, 37.703, 38.901]
ours_sa3, = plt.plot(bpp, psnr, "b-s", linewidth=LineWidth, label='Ours(SA)')

bpp = [0.1157, 0.1679, 0.2416, 0.3352]
psnr = [34.918, 36.271, 37.674, 38.91]
ours_sa1, = plt.plot(bpp, psnr, "y-p", linewidth=LineWidth, label='Ours(SA-1)')

bpp = [0.1187, 0.1711, 0.2585, 0.3496]
psnr = [34.742, 36.119, 37.97, 39.034]
ours_cat, = plt.plot(bpp, psnr, "m-s", linewidth=LineWidth, label='Ours(FeatCat)')

bpp = [0.1162, 0.1694, 0.2469, 0.351]
psnr = [34.574, 36.07, 37.728, 38.988]
Minnen, = plt.plot(bpp, psnr, "r--v", linewidth=LineWidth, label='Minnen')

plt.legend(handles=[ours_sa6, ours_sa3, ours_sa1, ours_cat, Minnen], loc=4)
plt.grid()
plt.xlabel('Bpp')
plt.ylabel('PSNR(dB)')
plt.title('KAIST dataset')
plt.savefig('Transformer.eps', format='eps', dpi=300, bbox_inches='tight')
plt.savefig('Transformer.png', format='png', dpi=300, bbox_inches='tight')