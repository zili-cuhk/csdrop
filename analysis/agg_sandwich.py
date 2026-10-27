"""Support sandwich of Corollary 2: when the sensitivity set contains the true
association tau0, the estimated robust support contains only true effects and the
estimated possible support contains every true effect above the threshold, with
probability tending to one.

Usage: python3 analysis/agg_sandwich.py path_all.csv.gz [expD_all.csv.gz expD_targets.json] [outdir]

Inputs
  path_all.csv.gz      concatenated output of run_path_fits.m (Experiment C,
                       cells A01-A24, grid {0, 0.1, ..., 0.5})
  expD_all.csv.gz      concatenated output of run_expD_fits.m (Experiment E,
                       weak dropout effect 0.15 for Z3, tau0 = 0.5, grid {-0.3, ..., 0.5})
  expD_targets.json    pseudo-true targets of pseudotrue/expD_targets.py (for Btrue)
For each cell and association grid the summary gives, over replications with
admissible fits at every grid value (criterion EBIC_0.25),
  P_rob_in_S0    P(robust support is a subset of S0_eta)
  P_S0_in_pos    P(S0_eta is a subset of the possible support)
  P_both         both inclusions
  FP_rob         average number of inactive coefficients in the robust support
  FN_pos         average number of true effects missing from the possible support
where S0_eta = {(d, j) in S0 : |x0_dj| > eta}, eta = 0.05, equals the true
support in both designs. Grids that do not contain tau0 are reported for contrast.
Output: sandwich_summary.csv."""
import sys, os, numpy as np, pandas as pd

args = sys.argv[1:]
fC = args[0]
fE, fT = (args[1], args[2]) if len(args) >= 3 and args[1].endswith('.gz') else (None, None)
outdir = args[3] if fE and len(args) > 3 else (args[1] if not fE and len(args) > 1 else '.')
CRIT = 'EBIC0.25'


def supmat(r, p):
    S = np.zeros((3, p), bool)
    for m, c in enumerate(('sup_event1', 'sup_event2', 'sup_dropout')):
        v = r[c]
        if isinstance(v, str) and v.strip():
            S[m, [int(x) - 1 for x in v.split(';')]] = True
    return S


def summarize(D, cells, grids, rows, exp):
    D = D[D.criterion == CRIT]
    for sid, g in D.groupby('sid'):
        if sid not in cells:
            continue
        tau0, S0 = cells[sid]
        p = S0.shape[1]
        reps = {}
        for _, r in g.iterrows():
            if r.status == 'ok':
                reps.setdefault(int(r.rep), {})[round(float(r.tau), 2)] = supmat(r, p)
        for name, grid in grids.items():
            grid = [round(t, 2) for t in grid]
            A = [np.array([b[t] for t in grid]) for b in reps.values() if all(t in b for t in grid)]
            if not A:
                continue
            rob = np.array([a.all(0) for a in A]); pos = np.array([a.any(0) for a in A])
            in1 = np.array([not (s & ~S0).any() for s in rob])
            in2 = np.array([not (S0 & ~s).any() for s in pos])
            rows.append(dict(exp=exp, sid=sid, tau0=tau0, grid=name, contains_tau0=tau0 in grid,
                             reps=len(A), P_rob_in_S0=in1.mean(), P_S0_in_pos=in2.mean(),
                             P_both=(in1 & in2).mean(),
                             FP_rob=np.mean([(s & ~S0).sum() for s in rob]),
                             FN_pos=np.mean([(S0 & ~s).sum() for s in pos])))


def true_support(signal, p):
    B = np.zeros((3, p), bool)
    for m, idx in enumerate(([0, 1, 4], [0, 2, 5], [1, 3, 6])):
        B[m, idx] = True
    return B


rows = []
# Experiment C: same cell order as R/scenarios.R (tau fastest, then signal, p, n)
SC = [dict(tau0=t, signal=s, p=p, n=n) for n in (400, 600) for p in (10, 40)
      for s in ('moderate', 'strong') for t in (0, 0.2, 0.5)]
cellsC = {f'A{i + 1:02d}': (c['tau0'], true_support(c['signal'], c['p'])) for i, c in enumerate(SC)}
gridsC = {'{0,...,0.5}': [0, .1, .2, .3, .4, .5], '{0,0.2,0.5}': [0, .2, .5], '{0,0.5}': [0, .5],
          '{0.1,0.3,0.5}': [.1, .3, .5], '{0,0.1,0.3}': [0, .1, .3]}
DC = pd.read_csv(fC, dtype={'sup_event1': str, 'sup_event2': str, 'sup_dropout': str})
summarize(DC, cellsC, gridsC, rows, 'C')

if fE:
    import json
    B0 = np.array(json.load(open(fT))['Btrue'])
    S0 = np.abs(B0) > 0.05
    DE = pd.read_csv(fE, dtype={'sup_event1': str, 'sup_event2': str, 'sup_dropout': str})
    cellsE = {sid: (0.5, S0) for sid in DE.sid.unique()}
    gridsE = {'[-0.3,0.5]': [-.3, -.2, -.1, 0, .1, .2, .3, .4, .5], '[0,0.5]': [0, .1, .2, .3, .4, .5],
              '[0.3,0.5]': [.3, .4, .5], '{-0.3,0.1,0.5}': [-.3, .1, .5], '[-0.3,0.3]': [-.3, -.2, -.1, 0, .1, .2, .3]}
    summarize(DE, cellsE, gridsE, rows, 'E')

R = pd.DataFrame(rows)
os.makedirs(outdir, exist_ok=True)
R.to_csv(os.path.join(outdir, 'sandwich_summary.csv'), index=False)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 400)
print(R.round(3).to_string(index=False))
for exp, h in R.groupby('exp'):
    a = h[h.contains_tau0]; b = h[~h.contains_tau0]
    print(f'\nExperiment {exp}, grids containing tau0: P(rob in S0) {a.P_rob_in_S0.min():.3f}-{a.P_rob_in_S0.max():.3f}, '
          f'P(S0 in pos) {a.P_S0_in_pos.min():.3f}-{a.P_S0_in_pos.max():.3f}')
    if len(b):
        print(f'Experiment {exp}, grids without tau0: P(rob in S0) {b.P_rob_in_S0.min():.3f}-{b.P_rob_in_S0.max():.3f}, '
              f'P(S0 in pos) {b.P_S0_in_pos.min():.3f}-{b.P_S0_in_pos.max():.3f}')
