import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family': 'Arial', 'weight': 'normal', 'size': 16}
norm = matplotlib.colors.Normalize(vmin=0.6, vmax=1.)
matplotlib.rc('font', **font)
def draw(corr):
    label = ['-2', '-1', '0', '1', '2']
    fig = plt.figure(figsize=(20, 5))
    l = len(data.keys())
    i = 1
    for k, v in corr.items():
        ax = fig.add_subplot(int(f'1{l}{i}'))
        ax.set_yticks(range(len(label)))
        ax.set_yticklabels(label)
        ax.set_xticks(range(len(label)))
        ax.set_xticklabels(label)
        im = plt.imshow(v, cmap='Oranges', norm=norm)
        ax.set_title(k)
        i += 1
    fig.subplots_adjust(right=0.8)
    cbar_x = fig.add_axes([0.85, 0.2, 0.015, 0.6])
    plt.colorbar(im, cax=cbar_x)
    plt.savefig('correlation.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig('correlation.png', format='png', dpi=300, bbox_inches='tight')


def parse(c):
    corr = []
    for l in c:
        l = l.replace('\n', '').split(' ')
        corr.append(l)
    return np.array(corr).astype(np.float32)


if __name__ == '__main__':
    data = {
        'FLIR (Minnen)' : 'LatentCorrelation/FLIR_s2048.txt',
        'FLIR (Ours)' : 'LatentCorrelation/FLIR_m2048.txt',
        'KAIST (Minnen)' : 'LatentCorrelation/KAIST_s2048.txt',
        'KAIST (Ours)' : 'LatentCorrelation/KAIST_m2048.txt'
    }
    for k, v in data.items():
        with open(v) as f:
            lines = f.readlines()
            data[k] = parse(lines)
    draw(data)