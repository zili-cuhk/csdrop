import sys, json, numpy as np
import os
_H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(_H, '..', 'python')); import csm
D = np.loadtxt(os.path.join(_H, 'data', 'tcga_exclude_unknown.csv'), delimiter=',', skiprows=1)
c = np.atleast_1d(np.loadtxt(os.path.join(_H, 'data', 'tcga_exclude_unknown_cuts.csv'), delimiter=','))
cuts = np.concatenate([[0], c, [np.inf]]); grid = [-0.3, 0.0, 0.2, 0.5]; times = np.array([12., 24., 36.])
b0, b1 = int(sys.argv[1]), int(sys.argv[2]); rng = np.random.RandomState(20261013); n = len(D)
idx_all = [rng.randint(0, n, n) for _ in range(200)]
out = []
for b in range(b0, b1):
    idx = idx_all[b]; Y, st, Z = D[idx, 0], D[idx, 1].astype(int), D[idx, 2:]
    rec = dict(b=b, fits={})
    for t in grid:
        try:
            r = csm.gselo_select(Y, st, Z, cuts, 2, 'frank', t)
            if r['status'] != 'ok': rec['fits'][str(t)] = None; continue
            s = r['sel'][0.25]; F = csm.cif(r['P'], s['X'], s['xi'], Z, times).mean(1)
            rec['fits'][str(t)] = dict(S=s['S'].astype(int).tolist(), F=F.tolist())
        except Exception as e:
            rec['fits'][str(t)] = None
    out.append(rec); print(b, flush=True)
    with open(os.path.join(_H, 'out', f'boot_{b0}_{b1}.jsonl'), 'a') as fh: fh.write(json.dumps(rec) + '\n')
json.dump(out, open(os.path.join(_H, 'out', f'boot_{b0}_{b1}.json'), 'w'))
