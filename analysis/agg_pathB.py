"""Exp C-B summaries. Usage: python3 agg_pathB.py pathB_all.csv.gz [outdir]"""
import sys, os, numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
f = sys.argv[1]; outdir = sys.argv[2] if len(sys.argv) > 2 else '.'
D = pd.read_csv(f, dtype={'sup_event1': str, 'sup_event2': str, 'sup_dropout': str})
D = D[D.criterion == 'EBIC0.25'].copy(); D['tau'] = D.tau.round(1)
truth = {1: ('clayton', 0.5, 0.0), 2: ('gumbel', 0.5, 0.0), 3: ('frank', 0.2, 0.5), 4: ('frank', 0.2, 1.0)}
SC = {f'B0{i}': dict(signal=('moderate' if i <= 4 else 'strong'), fam=truth[(i - 1) % 4 + 1][0], tau0=truth[(i - 1) % 4 + 1][1],
                     delta=truth[(i - 1) % 4 + 1][2]) for i in range(1, 9)}
p = 10
S0 = np.zeros((3, p), bool)
for m, idx in enumerate(([0, 1, 4], [0, 2, 5], [1, 3, 6])): S0[m, idx] = True
def supmat(r):
    S = np.zeros((3, p), bool)
    for m, c in enumerate(('sup_event1', 'sup_event2', 'sup_dropout')):
        v = r[c]
        if isinstance(v, str) and v.strip(): S[m, [int(x) - 1 for x in v.split(';')]] = True
    return S
nt = 4
rows, cifrows = [], []
for sid, g in D.groupby('sid'):
    sc = SC[sid]
    reps = {}
    for _, r in g.iterrows():
        fam = 'frank' if r.tau == 0 else r.family          # tau = 0 is the independence copula for every family
        reps.setdefault(int(r.rep), {})[(fam, float(r.tau))] = r
    keys_frank = [('frank', t) for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
    keys_all = keys_frank + [(fm, t) for fm in ('clayton', 'gumbel') for t in (0.1, 0.2, 0.3, 0.4, 0.5)]
    true_key = (sc['fam'], sc['tau0']) if sc['fam'] != 'frank' else ('frank', sc['tau0'])
    methods = {}
    def add(name, S):
        methods.setdefault(name, []).append(S)
    Lf, Uf, La, Ua, Tt = [], [], [], [], []
    TH = {'frank-profile': [], 'joint-profile': []}
    for rep, b in reps.items():
        if not all(k in b for k in keys_all): continue
        Sm = {k: supmat(b[k]) for k in keys_all}
        add('oracle (true family, true tau)', Sm[true_key])
        add('Frank at true tau', Sm[('frank', sc['tau0'])])
        add('working independence', Sm[('frank', 0.0)])
        kf = max(keys_frank, key=lambda k: b[k].ell_full); add('Frank tau-hat', Sm[kf]); TH['frank-profile'].append(kf)
        ka = max(keys_all, key=lambda k: b[k].ell_full); add('(family, tau)-hat', Sm[ka]); TH['joint-profile'].append(ka)
        add('robust Frank [0,0.5]', np.logical_and.reduce([Sm[k] for k in keys_frank]))
        add('robust families x [0,0.5]', np.logical_and.reduce([Sm[k] for k in keys_all]))
        add('robust families at true tau', np.logical_and.reduce([Sm[(fm, sc['tau0'])] for fm in ('frank', 'clayton', 'gumbel')]))
        F = lambda k: np.array([[b[k][f'F{c}_t{j}'] for j in range(1, nt + 1)] for c in (1, 2)])
        Ff = np.array([F(k) for k in keys_frank]); Fa = np.array([F(k) for k in keys_all])
        Lf.append(Ff.min(0)); Uf.append(Ff.max(0)); La.append(Fa.min(0)); Ua.append(Fa.max(0))
        Tt.append(np.array([[b[keys_all[0]][f'T{c}_t{j}'] for j in range(1, nt + 1)] for c in (1, 2)]))
    for name, L in methods.items():
        A = np.array(L)
        rows.append(dict(sid=sid, **sc, method=name, reps=len(A), P_exact=np.mean([(s == S0).all() for s in A]),
                         FP=np.mean([(s & ~S0).sum() for s in A]), FN=np.mean([(~s & S0).sum() for s in A]),
                         Pcorr_E1=np.mean([(s[0] == S0[0]).all() for s in A]), Pcorr_E2=np.mean([(s[1] == S0[1]).all() for s in A]),
                         Pcorr_D=np.mean([(s[2] == S0[2]).all() for s in A])))
    for nm, ks in TH.items():
        fams = [k[0] for k in ks]; taus = np.array([k[1] for k in ks])
        rows.append(dict(sid=sid, **sc, method=f'[{nm}]', reps=len(ks), tauhat_mean=taus.mean(), P_tau_true=np.mean(np.isclose(taus, sc['tau0'])),
                         P_family_true=np.mean([fm == sc['fam'] for fm in fams]),
                         family_share=str(pd.Series(fams).value_counts(normalize=True).round(2).to_dict())))
    for lab, L, U in (('Frank [0,0.5]', Lf, Uf), ('families x [0,0.5]', La, Ua)):
        L, U, T = np.array(L), np.array(U), np.array(Tt)
        sL, sU = L.std(0, ddof=1), U.std(0, ddof=1)
        for c in range(2):
            for j in range(nt):
                s = max(sL[c, j], sU[c, j]); dl = max(np.mean(U[:, c, j] - L[:, c, j]), 0)
                cc = brentq(lambda x: norm.cdf(x + dl / s) - norm.cdf(-x) - 0.95, 0, 4)
                lo, hi = L[:, c, j] - cc * sL[c, j], U[:, c, j] + cc * sU[c, j]
                cifrows.append(dict(sid=sid, **sc, set=lab, cause=c + 1, t=j + 1, width=np.mean(U[:, c, j] - L[:, c, j]),
                                    cover_bounds=np.mean((L[:, c, j] <= T[:, c, j]) & (T[:, c, j] <= U[:, c, j])),
                                    cover_IM=np.mean((lo <= T[:, c, j]) & (T[:, c, j] <= hi)), width_IM=np.mean(hi - lo)))
R = pd.DataFrame(rows); C = pd.DataFrame(cifrows)
os.makedirs(outdir, exist_ok=True); R.to_csv(os.path.join(outdir, 'pathB_selection.csv'), index=False); C.to_csv(os.path.join(outdir, 'pathB_cif.csv'), index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300); pd.set_option('display.max_columns', 30)
order = ['oracle (true family, true tau)', 'Frank at true tau', 'working independence', 'Frank tau-hat', '(family, tau)-hat',
         'robust Frank [0,0.5]', 'robust families at true tau', 'robust families x [0,0.5]']
R['case'] = R.fam + ' tau=' + R.tau0.astype(str) + ' d=' + R.delta.astype(str) + ' ' + R.signal
for v in ('P_exact', 'FP', 'FN', 'Pcorr_D'):
    print(f'\n{v}'); print(R[R.method.isin(order)].pivot_table(index='case', columns='method', values=v)[order].round(3))
print('\ntau-hat / family-hat'); print(R[R.method.str.startswith('[')][['case', 'method', 'tauhat_mean', 'P_tau_true', 'P_family_true', 'family_share']].to_string(index=False))
print('\nCIF'); print(C.groupby(['set', 'case' if 'case' in C else 'sid'])[['width', 'cover_bounds', 'cover_IM', 'width_IM']].mean().round(3))
