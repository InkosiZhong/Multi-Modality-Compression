import sys
import numpy as np
'''import torch
from torch import cosine_similarity as cosine

def spatial_correlation(x: torch.Tensor): # input [1,192,H/16,W/16]
    b, c, h, w = x.shape
    assert b == 1, 'only support batch=1'
    x = x.view(c, h, w)
    mean = torch.mean(x)
    std = torch.std(x)
    x = (x - mean) / std # normalization
    corr = torch.zeros([25], device=x.device)
    cnt = 0
    for i in range(2, h-2, 5):
        for j in range(2, w-2, 5):
            ctr = x[:,i,j].reshape(1, c)
            win = x[:,i-2:i+3,j-2:j+3].reshape(c, 25).permute(1,0) # 5x5 space
            corr += cosine(ctr, win, -1)
            cnt += 1
    corr /= cnt
    corr = corr.view(5, 5)
    return corr # 5x5
'''


def parse(c):
    corr = []
    for l in c:
        l = l.replace('\n', '').split('  ')
        corr.append(l)
    return corr


def mean_correlation(path):
    lines = None
    with open(path) as f:
        lines = f.readlines()
    corrs = []
    for i in range(0, len(lines), 6):
        c = lines[i:i+5]
        corrs.append(np.array(parse(c)).astype(np.float32))
    corrs = np.array(corrs).mean(0)
    print(corrs)
    np.savetxt('mean.txt', corrs)


if __name__ == '__main__':
    mean_correlation(sys.argv[1])