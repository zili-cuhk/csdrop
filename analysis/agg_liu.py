"""Selection and estimation tables in the format of Liu, Wang, Wang and Song
(Statistica Sinica 36, 999-1022), built from the Experiment A datasets.

Usage:  python3 analysis/agg_liu.py [root] [outdir]
        root   project root containing scenarios.csv, results/liu, results/comp
        outdir default results/summary/liu

Methods (rows of the selection tables)
  Full      unpenalized joint MLE at the true tau with all covariates (FULL)
  Lasso     cause-specific Cox lasso with BIC refits, independent censoring (CS_LASSO, bic)
  Liu       two-margin model of Liu et al. (2026) per cause, EBIC_0.25 (LIU2026, results/prop)
  WI        proposed method under working independence, EBIC_0.25 (WI)
  Proposed  proposed method at the true tau, EBIC_0.25 (PROP)
  Oracle    unpenalized joint MLE at the true tau on the true support (ORACLE)

Selection metrics, per margin and for the joint model (sum over the three margins)
  Pcorr  proportion of replications with exactly the true support
  MSE    (b - b0)' Sigma (b - b0) with Sigma_jk = 0.3^|j-k|, mean over replications
  FP     average number of inactive covariates selected
  FN     average number of active covariates missed
  Size   average number of selected covariates

Estimation metrics for each active coefficient, for Proposed, WI and Oracle
  cond   over the replications that select the coefficient (SelRate given)
  uncond over all replications, an unselected coefficient estimated by zero and
         its interval counted as not covering
  Bias, SE (Monte Carlo SD), ASE (mean Wald SE), CP (95% Wald interval)

Outputs: liu_selection.csv, liu_estimation.csv, and LaTeX tables
  tab_liu_sel_<signal>.tex          joint model, tau0 in {0.2, 0.5}  (main text)
  tab_liu_sel_<signal>_margins.tex  per margin, tau0 in {0, 0.2, 0.5} (supplement)
  tab_liu_est_<signal>_<tau>.tex    estimation, Proposed / WI / Oracle
"""
import sys, os, glob, numpy as np, pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else '.'
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, 'results', 'summary', 'liu')
os.makedirs(out, exist_ok=True)
MARG = ['event1', 'event2', 'dropout']
MLAB = {'event1': 'Event 1', 'event2': 'Event 2', 'dropout': 'Dropout'}
METHODS = ['Full', 'Lasso', 'Liu', 'WI', 'Proposed', 'Oracle']
NS = (400, 600, 1000)
FITMAP = {'FULL': 'Full', 'PROP': 'Proposed', 'WI': 'WI', 'ORACLE': 'Oracle'}


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


def vec(s, p):
    if not isinstance(s, str) or s == '':
        return np.zeros(p)
    v = np.array([float(x) for x in s.split(';')])
    return np.nan_to_num(v, nan=0.0) if v.size == p else np.full(p, np.nan)


def vec_nan(s, p):
    if not isinstance(s, str) or s == '':
        return np.full(p, np.nan)
    return np.array([float(x) for x in s.split(';')])


S = pd.read_csv(os.path.join(root, 'scenarios.csv'))
S = S[S.exp == 'A'].set_index('sid')

L = pd.concat([pd.read_csv(f, dtype={'support': str, 'coef': str, 'se': str})
               for f in glob.glob(os.path.join(root, 'results', 'liu', 'liu_*.csv'))], ignore_index=True)
L['method'] = L.fit.map(FITMAP)
C = pd.concat([pd.read_csv(f, dtype={'support': str, 'coef': str})
               for f in glob.glob(os.path.join(root, 'results', 'comp', 'sel_CS_LASSO_A*.csv'))], ignore_index=True)
C = C[(C.criterion == 'bic')].copy()
C['method'] = 'Lasso'; C['se'] = ''
Lf = glob.glob(os.path.join(root, 'results', 'prop', 'sel_LIU2026_A*.csv'))
if Lf:
    W = pd.concat([pd.read_csv(f, dtype={'support': str, 'coef': str}) for f in Lf], ignore_index=True)
    W = W[W.criterion == 'EBIC0.25'].copy(); W['method'] = 'Liu'; W['se'] = ''
else:
    W = pd.DataFrame(columns=['sid', 'rep', 'method', 'margin', 'status', 'coef', 'se'])
cols = ['sid', 'rep', 'method', 'margin', 'status', 'coef', 'se']
D = pd.concat([L[cols], C[cols], W[cols]], ignore_index=True)

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
        per = {m: [] for m in MARG}
        for m_i, m in enumerate(MARG):
            for _, r in g[g.margin == m].iterrows():
                b = vec(r.coef, p); sel = np.abs(b) > 0; act = B0[m_i] != 0
                d = b - B0[m_i]
                per[m].append(dict(rep=r.rep, corr=np.array_equal(sel, act), mse=d @ Sig @ d,
                                   fp=np.sum(sel & ~act), fn=np.sum(~sel & act), size=np.sum(sel)))
        frames = {m: pd.DataFrame(per[m]).set_index('rep').sort_index() for m in MARG}
        joint = frames['event1'].copy()
        joint['corr'] = frames['event1']['corr'] & frames['event2']['corr'] & frames['dropout']['corr']
        for k in ['mse', 'fp', 'fn', 'size']:
            joint[k] = sum(frames[m][k] for m in MARG)
        frames['joint'] = joint
        for m, f in frames.items():
            sel_rows.append(dict(sid=sid, n=sc.n, p=p, signal=sc.signal, tau=sc.tau, method=meth, margin=m,
                                 reps=len(f), Pcorr=f['corr'].mean(), MSE=f.mse.mean(),
                                 MSE_mcse=f.mse.std(ddof=1) / np.sqrt(len(f)),
                                 FP=f.fp.mean(), FN=f.fn.mean(), Size=f['size'].mean()))
        # ------------------------------------------------------------ estimation
        if meth in ('Proposed', 'WI', 'Oracle'):
            for m_i, m in enumerate(MARG):
                gm = g[g.margin == m]
                Bh = np.array([vec(c, p) for c in gm.coef])
                SE = np.array([vec_nan(s, p) for s in gm.se])
                for j in np.flatnonzero(B0[m_i]):
                    b0 = B0[m_i, j]; est = Bh[:, j]; se = SE[:, j]
                    s = est != 0
                    cov = np.abs(est - b0) <= 1.96 * se
                    cov = np.where(s & np.isfinite(se), cov, False)
                    est_rows.append(dict(
                        sid=sid, n=sc.n, p=p, signal=sc.signal, tau=sc.tau, method=meth, margin=m,
                        var=j + 1, true=b0, reps=len(est), SelRate=s.mean(),
                        Bias_cond=(est[s] - b0).mean(), SE_cond=est[s].std(ddof=1),
                        ASE_cond=np.nanmean(se[s]), CP_cond=cov[s].mean(),
                        Bias_uncond=(est - b0).mean(), SE_uncond=est.std(ddof=1),
                        ASE_uncond=np.nanmean(se[s]), CP_uncond=cov.mean()))

SEL = pd.DataFrame(sel_rows); EST = pd.DataFrame(est_rows); FAIL = pd.DataFrame(fail)
SEL.to_csv(os.path.join(out, 'liu_selection.csv'), index=False)
EST.to_csv(os.path.join(out, 'liu_estimation.csv'), index=False)
FAIL.to_csv(os.path.join(out, 'liu_admissible.csv'), index=False)
print('admissible replications (min over scenarios):')
print(FAIL.groupby('method').ok.min().to_string())


# ---------------------------------------------------------------- LaTeX
def f3(x): return '--' if not np.isfinite(x) else f'{x:.3f}'
def f2(x): return '--' if not np.isfinite(x) else f'{x:.2f}'
def fb(x): return '--' if not np.isfinite(x) else (f'{x:.3f}'.replace('-', '$-$'))
MLATEX = {'Full': 'Full', 'Lasso': 'Lasso', 'Liu': 'Liu2026', 'WI': 'WI', 'Proposed': 'Proposed', 'Oracle': 'Oracle'}


def sel_table(signal, margin, taus, path):
    T = SEL[(SEL.signal == signal) & (SEL.margin == margin)]
    nc = 5 * len(taus)
    lines = [r'\begin{tabular}{ll' + 'r' * nc + '}', r'\toprule',
             ' & & ' + ' & '.join(r'\multicolumn{5}{c}{$\tau_0=%g$}' % t for t in taus) + r'\\',
             ''.join(r'\cmidrule(lr){%d-%d}' % (3 + 5 * k, 7 + 5 * k) for k in range(len(taus))),
             r'$(n,p)$ & Method & ' + ' & '.join(['Pcorr & MSE & FP & FN & Size'] * len(taus)) + r'\\',
             r'\midrule']
    first = True
    for n in NS:
        for p in (10, 40):
            if len(T[(T.n == n) & (T.p == p)]) == 0: continue
            if not first: lines.append(r'\addlinespace')
            first = False
            for k, meth in enumerate(METHODS):
                cells = []
                for t in taus:
                    r = T[(T.n == n) & (T.p == p) & (T.method == meth) & (np.isclose(T.tau, t))]
                    if len(r) == 0: cells += ['--'] * 5; continue
                    r = r.iloc[0]
                    cells += [f3(r.Pcorr), f3(r.MSE), f3(r.FP), f3(r.FN), f3(r.Size)]
                lab = f'$({n},{p})$' if k == 0 else ''
                lines.append(f'{lab} & {MLATEX[meth]} & ' + ' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(path, 'w').write('\n'.join(lines) + '\n')


def sel_margins(signal, path):
    parts = []
    for m in MARG:
        sel_table(signal, m, (0, 0.2, 0.5), path + '.tmp')
        parts.append(r'\multicolumn{17}{l}{\textit{%s}}\\' % MLAB[m])
        body = open(path + '.tmp').read().splitlines()
        parts.append('\n'.join(body[6:-2]))
    os.remove(path + '.tmp')
    sel_table(signal, 'joint', (0, 0.2, 0.5), path + '.tmp')
    head = open(path + '.tmp').read().splitlines()[:6]; os.remove(path + '.tmp')
    open(path, 'w').write('\n'.join(head + ['\n'.join(parts).replace(r'\midrule' + '\n', '')] +
                                    [r'\bottomrule', r'\end{tabular}']) + '\n')


VLAB = {'event1': r'\beta_{1%d}', 'event2': r'\beta_{2%d}', 'dropout': r'\gamma_{%d}'}


def est_table(signal, tau, kind, path):
    T = EST[(EST.signal == signal) & np.isclose(EST.tau, tau)]
    meths = ['Proposed', 'WI', 'Oracle']
    lines = [r'\begin{tabular}{llrr' + 'rrrr' * len(meths) + '}', r'\toprule',
             ' & & & & ' + ' & '.join(r'\multicolumn{4}{c}{%s}' % m for m in meths) + r'\\',
             ''.join(r'\cmidrule(lr){%d-%d}' % (5 + 4 * k, 8 + 4 * k) for k in range(len(meths))),
             r'$(n,p)$ & Para & True & SelRate & ' + ' & '.join(['Bias & SE & ASE & CP'] * len(meths)) + r'\\',
             r'\midrule']
    first = True
    for n in NS:
        for p in (10, 40):
            Tp = T[(T.n == n) & (T.p == p)]
            if len(Tp) == 0: continue
            if not first: lines.append(r'\addlinespace')
            first = False
            k = 0
            for m in MARG:
                for j in sorted(Tp[Tp.margin == m]['var'].unique()):
                    cells = []
                    for meth in meths:
                        r = Tp[(Tp.margin == m) & (Tp['var'] == j) & (Tp.method == meth)]
                        if len(r) == 0: cells += ['--'] * 4; continue
                        r = r.iloc[0]
                        cells += [fb(r['Bias_' + kind]), f3(r['SE_' + kind]), f3(r['ASE_' + kind]),
                                  f3(r['CP_' + kind])]
                    tr = Tp[(Tp.margin == m) & (Tp['var'] == j)].iloc[0].true
                    sr = Tp[(Tp.margin == m) & (Tp['var'] == j) & (Tp.method == 'Proposed')]
                    sr = f3(sr.iloc[0].SelRate) if len(sr) else '--'
                    lab = f'$({n},{p})$' if k == 0 else ''
                    lines.append(f'{lab} & ${VLAB[m] % j}$ & {fb(tr)} & {sr} & ' + ' & '.join(cells) + r'\\')
                    k += 1
    lines += [r'\bottomrule', r'\end{tabular}']
    open(path, 'w').write('\n'.join(lines) + '\n')


for sg in ('moderate', 'strong'):
    sel_table(sg, 'joint', (0.2, 0.5), os.path.join(out, f'tab_liu_sel_{sg}.tex'))
    sel_margins(sg, os.path.join(out, f'tab_liu_sel_{sg}_margins.tex'))
    for t in (0.2, 0.5):
        for kind in ('cond', 'uncond'):
            est_table(sg, t, kind, os.path.join(out, f'tab_liu_est_{sg}_{int(10 * t):02d}_{kind}.tex'))

pd.set_option('display.width', 220)
J = SEL[SEL.margin == 'joint'].pivot_table(index=['signal', 'n', 'p', 'method'], columns='tau',
                                            values=['Pcorr', 'MSE'])
print(J.round(3).to_string())
print('\nestimation (moderate, tau0 = 0.5, conditional on selection):')
E = EST[(EST.signal == 'moderate') & np.isclose(EST.tau, 0.5)]
print(E[['n', 'p', 'method', 'margin', 'var', 'true', 'SelRate', 'Bias_cond', 'SE_cond', 'ASE_cond',
         'CP_cond', 'CP_uncond']].round(3).to_string(index=False))
print('\ntables written to', out)
