"""Larger-dimension cells P01-P36: model-space adjustment of EBIC.

Usage: python3 analysis/agg_largep_ebic.py [root] [propdir]
  propdir  results subfolder of the copHCR/HDCR refits (default prop_gamma,
           written by lsf/propPg.lsf with gamma_E in {0, 0.25, 0.5, 1}, the
           Chen-Chen value 1 - 1/(2 kappa) (criterion EBICcc) and the sparsity
           grid extended to gamma_P = 0.5)

For copHCR (PROP_TRUE) and HDCR (PROP_T00) under every EBIC criterion found,
and for Lasso (CS_LASSO, results/comp) under BIC and EBIC_0.5, computes per
scenario the exact joint-support recovery (Pcorr), false positives (FP) and
false negatives (FN) summed over the margins, the MSE (sum over margins of
(b - b0)' Sigma (b - b0), Sigma_jk = 0.3^|j-k|), and the integrated squared CIF
error summed over the two causes. Writes results/summary/largep/ebic_<propdir>.csv
and the LaTeX table tab_largep_ebic.tex (tau0 = 0.5).
"""
import os, sys, glob
import numpy as np, pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else '.'
PD = sys.argv[2] if len(sys.argv) > 2 else 'prop_gamma'
O = os.path.join(D, 'results', 'summary', 'largep'); os.makedirs(O, exist_ok=True)
S = pd.read_csv(os.path.join(D, 'scenarios.csv')); S = S[S.exp == 'P'].set_index('sid')
MI = {'event1': 0, 'event2': 1, 'dropout': 2}


def coefs(signal, p):
    B = np.zeros((3, p))
    if signal == 'moderate':
        B[0, [0, 1, 4]] = [.5, -.4, .3]; B[1, [0, 2, 5]] = [-.35, .45, -.3]; B[2, [1, 3, 6]] = [.4, -.35, .3]
    else:
        B[0, [0, 1, 4]] = [.7, -.7, .6]; B[1, [0, 2, 5]] = [-.7, .7, -.6]; B[2, [1, 3, 6]] = [.7, -.7, .6]
    return B


def vec(s, p):
    if not isinstance(s, str) or s == '':
        return np.zeros(p)
    return np.nan_to_num(np.array([float(x) for x in s.split(';')]))


rows = []
srcs = [('PROP_TRUE', PD, None), ('PROP_T00', PD, None), ('CS_LASSO', 'comp', ['bic', 'ebic0.5'])]
for fit, sub, crits in srcs:
    for f in sorted(glob.glob(os.path.join(D, 'results', sub, f'sel_{fit}_P*.csv'))):
        X = pd.read_csv(f, dtype={'coef': str, 'support': str})
        X = X[X.status == 'ok']
        if crits: X = X[X.criterion.isin(crits)]
        for (sid, cr, rep), g in X.groupby(['sid', 'criterion', 'rep']):
            if len(g) != 3: continue
            sc = S.loc[sid]; p = int(sc.p); B0 = coefs(sc.signal, p)
            Sig = 0.3 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
            mse = fp = fn = 0; ok = True
            for _, r in g.iterrows():
                b = vec(r.coef, p); b0 = B0[MI[r.margin]]
                sel, act = b != 0, b0 != 0
                d = b - b0; mse += d @ Sig @ d
                fp += np.sum(sel & ~act); fn += np.sum(~sel & act); ok &= np.array_equal(sel, act)
            rows.append(dict(sid=sid, fit=fit, crit=cr, rep=rep, corr=ok, fp=fp, fn=fn, mse=mse))
R = pd.DataFrame(rows).groupby(['sid', 'fit', 'crit']).agg(
    reps=('rep', 'nunique'), Pcorr=('corr', 'mean'), FP=('fp', 'mean'), FN=('fn', 'mean'),
    MSE=('mse', 'mean')).reset_index()
C = []
for fit, sub, _ in srcs:
    F = glob.glob(os.path.join(D, 'results', sub, f'cif_{fit}_P*.csv'))
    if not F: continue
    X = pd.concat([pd.read_csv(f) for f in F])
    X = (X.groupby(['sid', 'criterion', 'rep']).ise.sum().groupby(['sid', 'criterion']).mean()
         .reset_index().rename(columns={'criterion': 'crit', 'ise': 'ISE'}))
    X['fit'] = fit; C.append(X)
R = R.merge(pd.concat(C), on=['sid', 'fit', 'crit'], how='left')
R = R.merge(S[['n', 'p', 'signal', 'tau']], left_on='sid', right_index=True)
R.to_csv(os.path.join(O, f'ebic_{PD}.csv'), index=False)
print('replications per cell and criterion: min', R.reps.min())

# ---------------------------------------------------------------- LaTeX, tau0 = 0.5
NAME = {'PROP_TRUE': 'copHCR', 'PROP_T00': 'HDCR', 'CS_LASSO': 'Lasso'}
CL = {'BIC': 'BIC', 'bic': 'BIC', 'EBIC0.25': r'EBIC$_{0.25}$', 'EBIC0.50': r'EBIC$_{0.5}$',
      'ebic0.5': r'EBIC$_{0.5}$', 'EBIC1.00': r'EBIC$_{1}$', 'EBICcc': r'EBIC$_{\mathrm{cc}}$'}
cols = [(f, c) for f in ('PROP_TRUE', 'PROP_T00') for c in ('EBIC0.25', 'EBIC0.50', 'EBICcc', 'EBIC1.00')
        if ((R.fit == f) & (R.crit == c)).any()] + [('CS_LASSO', 'bic'), ('CS_LASSO', 'ebic0.5')]
t = R[np.isclose(R.tau, 0.5)]
L = [r'\begin{tabular}{lrr' + 'r' * len(cols) + '}', r'\toprule']
groups = [(NAME[f], sum(1 for c in cols if c[0] == f)) for f in ('PROP_TRUE', 'PROP_T00', 'CS_LASSO')]
mc, k, cm = [], 4, ''
for nm, w in groups:
    mc.append(r'\multicolumn{%d}{c}{%s}' % (w, nm)); cm += r'\cmidrule(lr){%d-%d}' % (k, k + w - 1); k += w
L += [' & & & ' + ' & '.join(mc) + r'\\', cm,
      r'Signal & $n$ & $p$ & ' + ' & '.join(CL[c] for _, c in cols) + r'\\', r'\midrule']
for metric, name, fmt, sc in [('Pcorr', 'Pcorr', '%.3f', 1), ('FP', 'FP', '%.2f', 1), ('FN', 'FN', '%.2f', 1),
                              ('MSE', 'MSE', '%.3f', 1), ('ISE', r'CIF error ($\times10^3$)', '%.2f', 1e3)]:
    L.append(r'\multicolumn{%d}{l}{\textit{%s}}\\' % (3 + len(cols), name))
    for sg in ('moderate', 'strong'):
        for i, (n, p) in enumerate([(n, p) for n in (400, 600, 1000) for p in (100, 200)]):
            v = []
            for f, c in cols:
                r = t[(t.signal == sg) & (t.n == n) & (t.p == p) & (t.fit == f) & (t.crit == c)]
                v.append('--' if len(r) == 0 or not np.isfinite(r.iloc[0][metric]) else fmt % (r.iloc[0][metric] * sc))
            L.append((sg.capitalize() if i == 0 else '') + f' & {n} & {p} & ' + ' & '.join(v) + r'\\')
    if metric != 'ISE': L.append(r'\addlinespace')
L += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(O, 'tab_largep_ebic.tex'), 'w').write('\n'.join(L) + '\n')
pd.set_option('display.width', 250)
for m in ('Pcorr', 'FP', 'MSE', 'ISE'):
    print('\n' + m)
    print(t.pivot_table(index=['signal', 'n', 'p'], columns=['fit', 'crit'], values=m).round(3).to_string())
print('\ntable written to', os.path.join(O, 'tab_largep_ebic.tex'))
