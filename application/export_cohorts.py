"""Write the analysis cohorts used by the MATLAB checks and the bootstrap.

Usage: python export_cohorts.py
Reads data/clinical.csv and data/LUAD_clinicalMatrix.tsv (see README) and writes
data/tcga_<rule>.csv (columns: Y, status, six covariates; status 1 = recorded PFS
event, 2 = death without an earlier PFS event, 3 = dropout, 0 = administrative censoring)
and data/tcga_<rule>_cuts.csv (internal cutpoints of the baseline partition).
"""
import os, sys
import numpy as np
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _H)
from prep import prepare
VARS = ['age10', 'male', 'stage2', 'stage34', 'nodepos', 't34']
for rule in ('exclude_unknown', 'unknown_as_admin', 'unknown_as_dropout'):
    d = prepare(rule)
    st = d['e1'] * 1 + d['e2'] * 2 + d['dr'] * 3
    M = np.column_stack([d['Y'], st, d['Z']])
    out = os.path.join(_H, 'data', f'tcga_{rule}.csv')
    np.savetxt(out, M, delimiter=',', header=','.join(['Y', 'status'] + VARS), comments='', fmt='%.10g')
    np.savetxt(os.path.join(_H, 'data', f'tcga_{rule}_cuts.csv'), np.atleast_2d(d['cut'][1:-1]), delimiter=',', fmt='%.10g')
    print(rule, M.shape[0], np.bincount(st.astype(int), minlength=4))
