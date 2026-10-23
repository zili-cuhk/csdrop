"""Write data/tcga_boot_idx.csv, the bootstrap resampling indices used in the paper.

The 200 bootstrap samples of the 433 patients are drawn with numpy.random.RandomState(20261013),
as in boot_path.py. Column b contains the 1-based row indices of bootstrap sample b.
Usage: python make_boot_idx.py
"""
import os
import numpy as np

n, B = 433, 200
rng = np.random.RandomState(20261013)
idx = np.array([rng.randint(0, n, n) for _ in range(B)]).T + 1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'tcga_boot_idx.csv')
np.savetxt(out, idx, delimiter=',', fmt='%d')
print('wrote', out, idx.shape)
