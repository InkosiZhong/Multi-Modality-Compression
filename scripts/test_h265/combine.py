import numpy as np
with open('result.txt') as f:
    lines = f.readlines()
    psnrs = [float(x[5:-2]) for x in lines[1::3]]
    bpps = [float(x[4:-2]) for x in lines[2::3]]
    print(np.mean(psnrs), np.mean(bpps))