"""Summaries for Exp C (tier 2) from the concatenated output of run_path_fits.m.
Usage: python3 agg_path.py path_all.csv.gz [outdir]
Writes support_summary.csv and cif_summary.csv and prints the main tables."""
import sys, os, itertools, numpy as np, pandas as pd

f = sys.argv[1]; outdir = sys.argv[2] if len(sys.argv) > 2 else '.'
D = pd.read_csv(f, dtype={'sup_event1': str, 'sup_event2': str, 'sup_dropout': str})
D['tau'] = D['tau'].round(2)

# scenario map: same order as build_scenarios() in R/scenarios.R (tau fastest, then signal, p, n)
rows = [dict(tau0=t, signal=s, p=p, n=n) for n in (400, 600) for p in (10, 40) for s in ('moderate', 'strong') for t in (0, 0.2, 0.5)]
SC = pd.DataFrame(rows); SC['sid'] = [f'A{i+1:02d}' for i in range(len(SC))]
D = D.merge(SC, on='sid', how='left')

def coefs(signal, p):
    B = np.zeros((3, p), bool)
    for m, idx in enumerate(([0, 1, 4], [0, 2, 5], [1, 3, 6])): B[m, idx] = True
    return B

def supmat(r, p):
    S = np.zeros((3, p), bool)
    for m, c in enumerate(('sup_event1', 'sup_event2', 'sup_dropout')):
        v = r[c]
        if isinstance(v, str) and v.strip():
            S[m, [int(x) - 1 for x in v.split(';')]] = True
    return S

GRIDS = {'G6 {0,...,0.5}': [0, .1, .2, .3, .4, .5], 'G3 {0,0.2,0.5}': [0, .2, .5], 'G2 {0,0.5}': [0, .5],
         'G3off {0.1,0.3,0.5}': [.1, .3, .5], 'G3low {0,0.1,0.3}': [0, .1, .3]}
nt = len([c for c in D.columns if c.startswith('F1_t')])
supp_rows, cif_rows = [], []
for crit in ('EBIC0.25', 'BIC', 'EBIC0.50'):
    d = D[D.criterion == crit]
    for sid, g in d.groupby('sid'):
        sc = SC[SC.sid == sid].iloc[0]; p = int(sc.p); S0 = coefs(sc.signal, p)
        reps = {}
        for _, r in g.iterrows():
            reps.setdefault(int(r.rep), {})[float(r.tau)] = r
        single = {t: [] for t in GRIDS['G6 {0,...,0.5}']}
        for rep, byt in reps.items():
            for t, r in byt.items():
                if r.status == 'ok': single[t].append(supmat(r, p))
        base = dict(sid=sid, criterion=crit, n=int(sc.n), p=p, signal=sc.signal, tau0=sc.tau0)
        # estimate-tau-then-select: tau-hat maximizes the full-model log-likelihood over the grid
        if 'ell_full' in g.columns:
            Th, TH = [], []
            for rep, byt in reps.items():
                ok = {t: r for t, r in byt.items() if r.status == 'ok' and np.isfinite(r.ell_full)}
                if len(ok) < len(byt): continue
                th = max(ok, key=lambda t: ok[t].ell_full); TH.append(th); Th.append(supmat(ok[th], p))
            if Th:
                A = np.array(Th)
                supp_rows.append(dict(base, method='profile tau-hat', reps=len(A),
                                      P_exact=np.mean([(s == S0).all() for s in A]),
                                      FP=np.mean([(s & ~S0).sum() for s in A]), FN=np.mean([(~s & S0).sum() for s in A]),
                                      Pcorr_dropout=np.mean([(s[2] == S0[2]).all() for s in A]),
                                      tauhat_mean=np.mean(TH), tauhat_sd=np.std(TH), P_tauhat_true=np.mean(np.isclose(TH, sc.tau0))))
        for t, L in single.items():
            if not L: continue
            A = np.array(L)
            supp_rows.append(dict(base, method=f'single tau={t}', reps=len(A),
                                  P_exact=np.mean([(s == S0).all() for s in A]),
                                  FP=np.mean([(s & ~S0).sum() for s in A]), FN=np.mean([(~s & S0).sum() for s in A]),
                                  Pcorr_dropout=np.mean([(s[2] == S0[2]).all() for s in A])))
        for gname, grid in GRIDS.items():
            SRl, SPl, Lh, Uh, Tt = [], [], [], [], []
            PLl, PUl = [], []
            for rep, byt in reps.items():
                rs = [byt.get(round(t, 2)) for t in grid]
                if any(r is None or r.status != 'ok' for r in rs): continue
                Ss = np.array([supmat(r, p) for r in rs])
                SRl.append(Ss.all(0)); SPl.append(Ss.any(0))
                F = np.array([[[r[f'F{k}_t{j}'] for j in range(1, nt + 1)] for k in (1, 2)] for r in rs])  # G x 2 x nt
                Lh.append(F.min(0)); Uh.append(F.max(0))
                Tt.append(np.array([[rs[0][f'T{k}_t{j}'] for j in range(1, nt + 1)] for k in (1, 2)]))
                if 'PL1_t1' in rs[0]:
                    PLl.append(np.array([[rs[0][f'PL{k}_t{j}'] for j in range(1, nt + 1)] for k in (1, 2)]))
                    PUl.append(np.array([[rs[0][f'PU{k}_t{j}'] for j in range(1, nt + 1)] for k in (1, 2)]))
            if not SRl: continue
            SR, SP = np.array(SRl), np.array(SPl)
            supp_rows.append(dict(base, method=f'robust {gname}', reps=len(SR),
                                  P_exact=np.mean([(s == S0).all() for s in SR]),
                                  FP=np.mean([(s & ~S0).sum() for s in SR]), FN=np.mean([(~s & S0).sum() for s in SR]),
                                  Pcorr_dropout=np.mean([(s[2] == S0[2]).all() for s in SR])))
            supp_rows.append(dict(base, method=f'possible {gname}', reps=len(SP),
                                  P_exact=np.mean([(s == S0).all() for s in SP]),
                                  FP=np.mean([(s & ~S0).sum() for s in SP]), FN=np.mean([(~s & S0).sum() for s in SP]),
                                  Pcorr_dropout=np.mean([(s[2] == S0[2]).all() for s in SP])))
            if crit != 'EBIC0.25': continue
            Lh, Uh, Tt = np.array(Lh), np.array(Uh), np.array(Tt)
            sL, sU = Lh.std(0, ddof=1), Uh.std(0, ddof=1)
            from scipy.stats import norm
            from scipy.optimize import brentq
            for k in range(2):
                for j in range(nt):
                    s = max(sL[k, j], sU[k, j]); delta = max(np.mean(Uh[:, k, j] - Lh[:, k, j]), 0)
                    c = brentq(lambda c: norm.cdf(c + delta / s) - norm.cdf(-c) - 0.95, 0, 4) if s > 0 else 0
                    lo, hi = Lh[:, k, j] - c * sL[k, j], Uh[:, k, j] + c * sU[k, j]
                    cif_rows.append(dict(base, grid=gname, cause=k + 1, time_index=j + 1, reps=len(Lh),
                                         true_mean=Tt[:, k, j].mean(), L_mean=Lh[:, k, j].mean(), U_mean=Uh[:, k, j].mean(),
                                         width=np.mean(Uh[:, k, j] - Lh[:, k, j]),
                                         cover_bounds=np.mean((Lh[:, k, j] <= Tt[:, k, j]) & (Tt[:, k, j] <= Uh[:, k, j])),
                                         cover_IM_mcSE=np.mean((lo <= Tt[:, k, j]) & (Tt[:, k, j] <= hi)), IM_c=c,
                                         width_IM=np.mean(hi - lo),
                                         width_peterson=(np.mean(np.array(PUl)[:, k, j] - np.array(PLl)[:, k, j]) if PLl else np.nan),
                                         cover_peterson=(np.mean((np.array(PLl)[:, k, j] <= Tt[:, k, j]) & (Tt[:, k, j] <= np.array(PUl)[:, k, j])) if PLl else np.nan)))
SS = pd.DataFrame(supp_rows); CC = pd.DataFrame(cif_rows)
os.makedirs(outdir, exist_ok=True)
SS.to_csv(os.path.join(outdir, 'support_summary.csv'), index=False); CC.to_csv(os.path.join(outdir, 'cif_summary.csv'), index=False)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 400)
e = SS[SS.criterion == 'EBIC0.25']
piv = e.pivot_table(index=['n', 'p', 'signal', 'tau0'], columns='method', values='P_exact')
keep = [c for c in piv.columns if c.startswith('robust') or c in ('single tau=0', 'single tau=0.2', 'single tau=0.5', 'profile tau-hat')]
print('P(exact joint support), EBIC0.25\n', piv[keep].round(3))
print('\nCIF coverage (EBIC0.25), averaged over scenarios\n', CC.groupby(['grid', 'cause'])[[c for c in ('cover_bounds', 'cover_IM_mcSE', 'width', 'width_IM', 'width_peterson', 'cover_peterson') if c in CC.columns]].mean().round(3))
