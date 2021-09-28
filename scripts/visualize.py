import matplotlib.pyplot as plt
import matplotlib.image as mp_img
import numpy as np
import os
from tqdm import tqdm

channels = [8, 14, 30, 44, 52]

if __name__ == '__main__':
    home = './visualize'
    folders = [f for f in os.listdir(home) if not os.path.isfile(os.path.join(home, f))]
    pbar = tqdm(list(folders))
    for folder in pbar:
        path = os.path.join(home, folder)
        sub_folders = [f for f in os.listdir(path) if not os.path.isfile(os.path.join(path, f))]
        n, m = len(channels), len(sub_folders)
        for i, ch in enumerate(channels):
            for j, f in enumerate(sub_folders):
                img = mp_img.imread(os.path.join(path, f, f'{ch}.jpg'))
                plt.subplot(n, m, i * m + j + 1)
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                if i == n - 1:
                    plt.xlabel(f)
                if j == 0:
                    plt.ylabel(ch)
        plt.savefig(f'{path}.jpg')
        pbar.update(1)