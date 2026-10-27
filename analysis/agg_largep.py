"""Selection, estimation and prediction summaries for the larger-dimension cells
P01-P36 (n in {400, 600, 1000}, p in {100, 200}, moderate and strong signal,
tau0 in {0, 0.2, 0.5}).

Usage:  python3 analysis/agg_largep.py [root] [outdir]
        root   project root containing scenarios.csv and results/{liu,prop,comp}
        outdir default results/summary/largep

Methods
  copHCR  proposed method at the true tau, EBIC_0.25 support, refitted (PROP in
          results/liu; CIF error from PROP_TRUE in results/prop)
  HDCR    proposed method under working independence, tau = 0 (WI in results/liu;
          CIF error from PROP_T00 in results/prop)
  Lasso   cause-specific Cox lasso with BIC refits, independent censoring
          (CS_LASSO, criterion bic, results/comp)
  Oracle  unpenalized joint MLE at the true tau on the true support (ORACLE)

Selection metrics for the joint model (sums over the three margins)
  Pcorr  proportion of replications with exactly the true support in all margins
  FP     average number of inactive covariates selected
  FN     average number of active covariates missed
  Size   average number of selected covariates
  MSE    sum over margins of (b - b0)' Sigma (b - b0), Sigma_jk = 0.3^|j-k|
CIF error: integrated squared error over t in [0, 6] for each cause, mean over
replications (copHCR, HDCR, Lasso).
Estimation: Bias, SE, ASE and CP of the active coefficients for copHCR, HDCR and
Oracle, conditional on selection, as in agg_liu.py.

Outputs: largep_selection.csv, largep_cif.csv, largep_estimation.csv,
largep_admissible.csv and the LaTeX tables
  tab_largep_sel_<signal>.tex  Pcorr, FP, FN, MSE by (n, p) and tau0
  tab_largep_cif_<signal>.tex  CIF ISE for both causes by (n, p) and tau0
"""
import sys, os, glob, numpy as np, pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else '.'
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, 'results', 'summary', 'largep')
os.makedirs(out, exist_ok=True)
MARG = ['event1', 'event2', 'dropout']
METHODS = ['copHCR', 'HDCR', 'Lasso', 'Oracle']
FITMAP = {'PROP': 'copHCR', 'WI': 'HDCR', 'ORACLE': 'Oracle'}
CIFMAP = {'PROP_TRUE': 'copHCR', 'PROP_T00': 'HDCR', 'CS_LASSO': 'Lasso'}
CIFCRIT = {'PROP_TRUE': 'EBIC0.25', 'PROP_T00': 'EBIC0.25', 'CS_LASSO': 'bic'}
NS, PS, TAUS = (400, 600, 1000), (100, 200), (0.0, 0.2, 0.5)


def coefs(signal, p):
    B = np.zeros((3, p))
    if signal == 'moderate':
        B[0, [0, 1, 4]] = [0.50, -0.40, 0.30]
        B[1, [0, 2, 5]] = [-0.35, 0.45, -0.30]
        B[2, [1, 3, 6]] = [0.40, -0.35, 0.30]
    else:
        B[0, [0, 1, 4]] = [0.70, -0.70, 0.60]
        B[1, [0, 2, 5]] = [-0.70, 0.70, -0.60]
        B[2, [1, 3, 6]] = [0.70, -0.70, 0.60]
    return B


def vec(s, p, fill=0.0):
    if not isinstance(s, str) or s == '':
        return np.full(p, fill)
    v = np.array([float(x) for x in s.split(';')])
    if v.size != p:
        return np.full(p, np.nan)
    return np.nan_to_num(v, nan=fill) if fill == 0.0 else v


def read(pattern, **kw):
    F = glob.glob(os.path.join(root, 'results', *pattern))
    if not F:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, **kw) for f in F], ignore_index=True)


S = pd.read_csv(os.path.join(root, 'scenarios.csv'))
S = S[S.exp == 'P'].set_index('sid')

L = read(('liu', 'liu_P*.csv'), dtype={'support': str, 'coef': str, 'se': str})
if len(L):
    L['method'] = L.fit.map(FITMAP)
C = read(('comp', 'sel_CS_LASSO_P*.csv'), dtype={'support': str, 'coef': str})
if len(C):
    C = C[C.criterion == 'bic'].copy(); C['method'] = 'Lasso'; C['se'] = ''
cols = ['sid', 'rep', 'method', 'margin', 'status', 'coef', 'se']
D = pd.concat([x[cols] for x in (L, C) if len(x)], ignore_index=True)
D = D.drop_duplicates(['sid', 'rep', 'method', 'margin'], keep='last')

# ---------------------------------------------------------------- selection
sel_rows, est_rows, fail = [], [], []
for sid, sc in S.iterrows():
    p = int(sc.p); B0 = coefs(sc.signal, p)
    Sig = 0.3 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    for meth in METHODS:
        g = D[(D.sid == sid) & (D.method == meth)]
        if len(g) == 0:
            continue
        ok = g.groupby('rep').status.apply(lambda s: (s == 'ok').all() and len(s) == 3)
        reps = ok.index[ok.values]
        fail.append(dict(sid=sid, method=meth, reps=len(ok), ok=len(reps)))
        g = g[g.rep.isin(reps)]
        if len(g) == 0:
            continue
        rec = []
        for rep, gr in g.groupby('rep'):
            corr, mse, fp, fn, size = True, 0.0, 0, 0, 0
            for m_i, m in enumerate(MARG):
                b = vec(gr[gr.margin == m].coef.iloc[0], p)
                sel = np.abs(b) > 0; act = B0[m_i] != 0; d = b - B0[m_i]
                corr &= np.array_equal(sel, act); mse += d @ Sig @ d
                fp += np.sum(sel & ~act); fn += np.sum(~sel & act); size += np.sum(sel)
            rec.append(dict(corr=corr, mse=mse, fp=fp, fn=fn, size=size))
        f = pd.DataFrame(rec)
        sel_rows.append(dict(sid=sid, n=sc.n, p=p, signal=sc.signal, tau=sc.tau, method=meth,
                             reps=len(f), Pcorr=f['corr'].mean(), FP=f.fp.mean(), FN=f.fn.mean(),
                             Size=f['size'].mean(), MSE=f.mse.mean(),
                             MSE_mcse=f.mse.std(ddof=1) / np.sqrt(len(f))))
        # ------------------------------------------------------------ estimation
        if meth != 'Lasso':
            for m_i, m in enumerate(MARG):
                gm = g[g.margin == m]
                Bh = np.array([vec(c, p) for c in gm.coef])
                SE = np.array([vec(s, p, fill=np.nan) for s in gm.se])
                for j in np.flatnonzero(B0[m_i]):
                    b0 = B0[m_i, j]; est = Bh[:, j]; se = SE[:, j]
                    s = est != 0
                    cov = np.where(s & np.isfinite(se), np.abs(est - b0) <= 1.96 * se, False)
                    est_rows.append(dict(
                        sid=sid, n=sc.n, p=p, signal=sc.signal, tau=sc.tau, method=meth, margin=m,
                        var=j + 1, true=b0, reps=len(est), SelRate=s.mean(),
                        Bias=(est[s] - b0).mean(), SE=est[s].std(ddof=1),
                        ASE=np.nanmean(se[s]), CP=cov[s].mean()))

SEL = pd.DataFrame(sel_rows); EST = pd.DataFrame(est_rows); FAIL = pd.DataFrame(fail)

# ---------------------------------------------------------------- CIF error
cif = []
for fit, sub in (('PROP_TRUE', 'prop'), ('PROP_T00', 'prop'), ('CS_LASSO', 'comp')):
    X = read((sub, f'cif_{fit}_P*.csv'))
    if len(X) == 0:
        continue
    X = X[X.criterion == CIFCRIT[fit]].drop_duplicates(['sid', 'rep', 'cause'], keep='last')
    X['method'] = CIFMAP[fit]
    cif.append(X)
CIF = pd.DataFrame()
if cif:
    X = pd.concat(cif, ignore_index=True)
    CIF = X.groupby(['sid', 'method', 'cause']).ise.agg(['mean', 'count']).reset_index()
    CIF = CIF.pivot_table(index=['sid', 'method'], columns='cause', values=['mean', 'count'])
    CIF.columns = [f'{a}_{b}' for a, b in CIF.columns]
    CIF = CIF.reset_index().rename(columns={'mean_1': 'ISE1', 'mean_2': 'ISE2',
                                            'count_1': 'reps1', 'count_2': 'reps2'})
    CIF = CIF.merge(S[['n', 'p', 'signal', 'tau']], left_on='sid', right_index=True)

SEL.to_csv(os.path.join(out, 'largep_selection.csv'), index=False)
EST.to_csv(os.path.join(out, 'largep_estimation.csv'), index=False)
CIF.to_csv(os.path.join(out, 'largep_cif.csv'), index=False)
FAIL.to_csv(os.path.join(out, 'largep_admissible.csv'), index=False)
if len(FAIL):
    print('admissible replications (min over cells):')
    print(FAIL.groupby('method').ok.min().to_string())


# ---------------------------------------------------------------- LaTeX
def f3(x): return '--' if not np.isfinite(x) else f'{x:.3f}'
def f2(x): return '--' if not np.isfinite(x) else f'{x:.2f}'
def f4(x): return '--' if not np.isfinite(x) else f'{x:.4f}'


def table(T, signal, metrics, fmts, path, methods):
    T = T[T.signal == signal]
    k = len(metrics)
    lines = [r'\begin{tabular}{ll' + 'r' * (k * len(TAUS)) + '}', r'\toprule',
             ' & & ' + ' & '.join(r'\multicolumn{%d}{c}{$\tau_0=%g$}' % (k, t) for t in TAUS) + r'\\',
             ''.join(r'\cmidrule(lr){%d-%d}' % (3 + k * i, 2 + k * (i + 1)) for i in range(len(TAUS))),
             r'$(n,p)$ & Method & ' + ' & '.join([' & '.join(metrics)] * len(TAUS)) + r'\\',
             r'\midrule']
    first = True
    for n in NS:
        for p in PS:
            if len(T[(T.n == n) & (T.p == p)]) == 0:
                continue
            if not first:
                lines.append(r'\addlinespace')
            first = False
            for i, meth in enumerate(methods):
                cells = []
                for t in TAUS:
                    r = T[(T.n == n) & (T.p == p) & (T.method == meth) & np.isclose(T.tau, t)]
                    if len(r) == 0:
                        cells += ['--'] * k; continue
                    r = r.iloc[0]
                    cells += [fm(r[c]) for c, fm in zip(metrics, fmts)]
                lab = f'$({n},{p})$' if i == 0 else ''
                lines.append(f'{lab} & {meth} & ' + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(path, 'w').write('\n'.join(lines) + '\n')


for sg in ('moderate', 'strong'):
    if len(SEL):
        table(SEL, sg, ['Pcorr', 'FP', 'FN', 'MSE'], [f3, f2, f2, f3],
              os.path.join(out, f'tab_largep_sel_{sg}.tex'), METHODS)
    if len(CIF):
        table(CIF, sg, ['ISE1', 'ISE2'], [f4, f4],
              os.path.join(out, f'tab_largep_cif_{sg}.tex'), ['copHCR', 'HDCR', 'Lasso'])

pd.set_option('display.width', 220)
if len(SEL):
    print(SEL.pivot_table(index=['signal', 'n', 'p', 'method'], columns='tau',
                          values=['Pcorr', 'MSE']).round(3).to_string())
if len(CIF):
    print(CIF.pivot_table(index=['signal', 'n', 'p', 'method'], columns='tau',
                          values=['ISE1', 'ISE2']).round(4).to_string())
print('\ntables written to', out)
