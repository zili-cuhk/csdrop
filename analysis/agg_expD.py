"""Summaries for Exp D (tau-dependent signal). Usage: python3 agg_expD.py expD_all.csv.gz targets.json [outdir]"""
import sys, os, json, numpy as np, pandas as pd
f, tf = sys.argv[1], sys.argv[2]; outdir = sys.argv[3] if len(sys.argv) > 3 else '.'
D = pd.read_csv(f, dtype={'sup_event1': str, 'sup_event2': str, 'sup_dropout': str})
D = D[D.criterion == 'EBIC0.25'].copy(); D['tau'] = D.tau.round(1); D['n'] = D.sid.str.extract(r'(\d+)$')[0].astype(int)
T = json.load(open(tf)); B0 = np.array(T['Btrue']) != 0; p = B0.shape[1]
tgt = {round(float(t), 1): np.abs(np.array(r['X'])) > 0.05 for t, r in T['res'].items()}
def supmat(r):
    S = np.zeros((3, p), bool)
    for m, c in enumerate(('sup_event1', 'sup_event2', 'sup_dropout')):
        v = r[c]
        if isinstance(v, str) and v.strip(): S[m, [int(x) - 1 for x in v.split(';')]] = True
    return S
Z3 = (2, 2)                                   # dropout effect of Z3: the tau-dependent signal
SETS = {'wide [-0.3,0.5]': [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5], 'mid [0,0.5]': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'narrow [0.3,0.5]': [0.3, 0.4, 0.5], 'wide-coarse {-0.3,0.1,0.5}': [-0.3, 0.1, 0.5]}
rows, freq = [], []
for n, g in D.groupby('n'):
    reps = {}
    for _, r in g.iterrows():
        if r.status == 'ok': reps.setdefault(int(r.rep), {})[float(r.tau)] = r
    for t in sorted(tgt):
        A = np.array([supmat(b[t]) for b in reps.values() if t in b])
        freq.append(dict(n=n, tau=t, **{f'{["e1", "e2", "d"][m]}Z{j+1}': A[:, m, j].mean() for m in range(3) for j in range(p)}))
        rows.append(dict(n=n, method=f'single tau={t}', reps=len(A), P_exact_S0=np.mean([(s == B0).all() for s in A]),
                         P_eq_target=np.mean([(s == tgt[t]).all() for s in A]), P_Z3=A[:, 2, 2].mean(),
                         FP_vs_S0=np.mean([(s & ~B0).sum() for s in A]), FN_vs_S0=np.mean([(~s & B0).sum() for s in A])))
    full = [b for b in reps.values() if len(b) == 9]
    TH = [max(b, key=lambda t: b[t].ell_full) for b in full]
    A = np.array([supmat(b[t]) for b, t in zip(full, TH)])
    rows.append(dict(n=n, method='profile tau-hat', reps=len(A), P_exact_S0=np.mean([(s == B0).all() for s in A]), P_Z3=A[:, 2, 2].mean(),
                     FP_vs_S0=np.mean([(s & ~B0).sum() for s in A]), FN_vs_S0=np.mean([(~s & B0).sum() for s in A]),
                     tauhat_mean=np.mean(TH), P_tauhat_true=np.mean(np.isclose(TH, 0.5))))
    for name, grid in SETS.items():
        SR_t = np.logical_and.reduce([tgt[t] for t in grid]); SP_t = np.logical_or.reduce([tgt[t] for t in grid])
        Ss = [np.array([supmat(b[t]) for t in grid]) for b in full]
        SR = np.array([s.all(0) for s in Ss]); SP = np.array([s.any(0) for s in Ss])
        rows.append(dict(n=n, method=f'robust {name}', reps=len(SR), target_has_Z3=bool(SR_t[Z3]),
                         P_eq_target=np.mean([(s == SR_t).all() for s in SR]), P_Z3=SR[:, 2, 2].mean(),
                         FP_vs_target=np.mean([(s & ~SR_t).sum() for s in SR]), FN_vs_target=np.mean([(~s & SR_t).sum() for s in SR])))
        rows.append(dict(n=n, method=f'possible {name}', reps=len(SP), target_has_Z3=bool(SP_t[Z3]),
                         P_eq_target=np.mean([(s == SP_t).all() for s in SP]), P_Z3=SP[:, 2, 2].mean(),
                         FP_vs_target=np.mean([(s & ~SP_t).sum() for s in SP]), FN_vs_target=np.mean([(~s & SP_t).sum() for s in SP]),
                         leak_recall=np.mean([(s & SP_t & ~B0).sum() / max((SP_t & ~B0).sum(), 1) for s in SP])))
R = pd.DataFrame(rows); Fq = pd.DataFrame(freq)
os.makedirs(outdir, exist_ok=True); R.to_csv(os.path.join(outdir, 'expD_summary.csv'), index=False); Fq.to_csv(os.path.join(outdir, 'expD_selfreq.csv'), index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200); pd.set_option('display.max_columns', 20)
print(R.round(3).to_string(index=False))
print('\nselection frequency of the tau-dependent signal (dropout Z3) and leaks by tau')
print(Fq[['n', 'tau', 'dZ3', 'dZ5', 'dZ6', 'e2Z2', 'e2Z4', 'e2Z7', 'e1Z7']].round(3).to_string(index=False))
