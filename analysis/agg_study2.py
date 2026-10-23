"""Study 2 table: one factor at a time from the base case n = 600, p = 10,
moderate signals, Frank tau0 = 0.5, plus the larger-dimension cell (n = 1000, p = 100).

Usage:  python3 analysis/agg_study2.py [root] [outdir]

For every setting and method: Pcorr of each margin, joint FP and FN, and the
integrated squared CIF error (x 1000) of each cause, with Monte Carlo SEs in
the CSV. Methods: Proposed (EBIC_0.25), WI (working independence), Liu2026
(two-margin model per cause), Lasso (cause-specific lasso with BIC refits).
Outputs: study2.csv and tab_study2.tex.
"""
import sys, os, glob, numpy as np, pandas as pd
root = sys.argv[1] if len(sys.argv) > 1 else '.'
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, 'results', 'summary', 'study2')
os.makedirs(out, exist_ok=True)
MARG = ['event1', 'event2', 'dropout']
SC = pd.read_csv(os.path.join(root, 'scenarios.csv')).set_index('sid')

def find(cond):
    q = SC.query(cond)
    return q.index[0] if len(q) else None

base = find("exp == 'A' and n == 600 and p == 10 and signal == 'moderate' and abs(tau - 0.5) < 1e-9")
bmod = SC[(SC.exp == 'B') & (SC.signal == 'moderate')]
def bsid(fam, delta):
    q = bmod[(bmod.family == fam) & (abs(bmod.delta - delta) < 1e-9)]
    return q.index[0] if len(q) else None
# (label, sid, fit_id of the proposed method)
SETTINGS = [('Base case', base, 'PROP_T05'),
            ('Dropout 10\\%', 'S01', 'PROP_TRUE'), ('Dropout 35\\%', 'S02', 'PROP_TRUE'),
            ('Staggered entry, 15\\%', 'S03', 'PROP_TRUE'), ('Staggered entry, 30\\%', 'S04', 'PROP_TRUE'),
            ('$\\tau_0=-0.3$', 'S05', 'PROP_TRUE'), ('Rare event 2', 'S06', 'PROP_TRUE'),
            ('Weibull, 3 intervals', 'S07', 'PROP_TRUE'), ('Weibull, 5 intervals', 'S07', 'PROP_Q5'),
            ('Clayton', bsid('clayton', 0), 'PROP_FRANK'), ('Gumbel--Hougaard', bsid('gumbel', 0), 'PROP_FRANK'),
            ('$\\delta=0.5$', bsid('frank', 0.5), 'PROP_ORACLE'), ('$\\delta=1$', bsid('frank', 1.0), 'PROP_ORACLE'),
            ('$n=1000$, $p=100$', 'H01', 'PROP_TRUE')]

def load(kind, sub, fit, sid):
    fs = glob.glob(os.path.join(root, 'results', sub, f'{kind}_{fit}_{sid}_*.csv'))
    if not fs: return None
    return pd.concat([pd.read_csv(f, dtype={'support': str, 'coef': str}) for f in fs], ignore_index=True)

def coefs(signal, p):
    B = np.zeros((3, p))
    if signal == 'moderate':
        B[0, [0, 1, 4]] = [.5, -.4, .3]; B[1, [0, 2, 5]] = [-.35, .45, -.3]; B[2, [1, 3, 6]] = [.4, -.35, .3]
    else:
        B[0, [0, 1, 4]] = [.7, -.7, .6]; B[1, [0, 2, 5]] = [-.7, .7, -.6]; B[2, [1, 3, 6]] = [.7, -.7, .6]
    return B

rows = []
for lab, sid, pfit in SETTINGS:
    if sid is None or sid not in SC.index: continue
    sc = SC.loc[sid]; p = int(sc.p); B0 = coefs(sc.signal, p)
    meths = [('Proposed', 'prop', pfit, 'EBIC0.25'), ('WI', 'prop', 'PROP_T00', 'EBIC0.25'),
                                 ('Liu2026', 'prop', 'LIU2026', 'EBIC0.25'), ('Lasso', 'comp', 'CS_LASSO', 'bic')]
    if pfit == 'PROP_Q5':
        meths = meths[:1]                     # comparators do not depend on the partition
    for meth, sub, fit, crit in meths:
        sel = load('sel', sub, fit, sid); cif = load('cif', sub, fit, sid)
        r = dict(setting=lab, sid=sid, method=meth)
        if sel is not None:
            sel = sel[(sel.criterion == crit) & (sel.status == 'ok')]
            fp = fn = 0
            reps = None
            joint = None
            jev = None
            for m_i, m in enumerate(MARG):
                g = sel[sel.margin == m]
                act = B0[m_i] != 0
                S = np.array([[False] * p if not isinstance(x, str) or x == '' else
                              np.isin(np.arange(1, p + 1), [int(v) for v in x.split(';')]) for x in g.support])
                if S.size == 0: continue
                corr = (S == act).all(1)
                cs = pd.Series(corr, index=g.rep.values)
                joint = cs if joint is None else (joint & cs.reindex(joint.index, fill_value=False))
                if m != 'dropout':
                    jev = cs if jev is None else (jev & cs.reindex(jev.index, fill_value=False))
                r[f'Pcorr_{m}'] = corr.mean(); r[f'Pcorr_{m}_se'] = corr.std(ddof=1) / np.sqrt(len(corr))
                fp = fp + (S & ~act).sum(1).mean(); fn = fn + (~S & act).sum(1).mean()
                reps = len(g)
            r.update(FP=fp, FN=fn, reps=reps)
            if joint is not None:
                r['Pcorr_joint'] = joint.mean(); r['Pcorr_joint_se'] = joint.std(ddof=1) / np.sqrt(len(joint))
                r['Pcorr_events'] = jev.mean()
        if cif is not None:
            cif = cif[cif.criterion == crit]
            for k in (1, 2):
                e = 1e3 * cif[cif.cause == k].ise.values
                if len(e):
                    r[f'CIF{k}'] = e.mean(); r[f'CIF{k}_se'] = e.std(ddof=1) / np.sqrt(len(e))
        rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(out, 'study2.csv'), index=False)
pd.set_option('display.width', 220)
show = [c for c in ['setting', 'method', 'reps', 'Pcorr_event1', 'Pcorr_event2', 'Pcorr_dropout', 'FP', 'FN', 'CIF1', 'CIF2'] if c in R]
print(R[show].round(3).to_string(index=False))

def f(x, d=3):
    return '--' if x is None or not np.isfinite(x) else f'{x:.{d}f}'
L = [r'\begin{tabular}{llrrrrrrr}', r'\toprule',
     r'Setting & Method & \multicolumn{3}{c}{Pcorr} & FP & FN & \multicolumn{2}{c}{CIF error ($\times10^3$)}\\',
     r'\cmidrule(lr){3-5}\cmidrule(lr){8-9}',
     r' & & E1 & E2 & D & & & Cause 1 & Cause 2\\', r'\midrule']
first = True
for lab in dict.fromkeys(R.setting):
    if not first: L.append(r'\addlinespace')
    first = False
    for k, (_, r) in enumerate(R[R.setting == lab].iterrows()):
        g = lambda c, d=3: f(r.get(c, np.nan), d)
        L.append(f"{lab if k == 0 else ''} & {r.method} & {g('Pcorr_event1')} & {g('Pcorr_event2')} & "
                 f"{g('Pcorr_dropout')} & {g('FP', 2)} & {g('FN', 2)} & {g('CIF1', 2)} & {g('CIF2', 2)}\\\\")
L += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(out, 'tab_study2.tex'), 'w').write('\n'.join(L) + '\n')

# compact version for the main text: joint Pcorr and the CIF error averaged over the two causes
MS = ['Proposed', 'WI', 'Liu2026', 'Lasso']
C = [r'\begin{tabular}{l' + 'rrr' * len(MS) + '}', r'\toprule',
     ' & ' + ' & '.join(r'\multicolumn{3}{c}{%s}' % m for m in MS) + r'\\',
     ''.join(r'\cmidrule(lr){%d-%d}' % (2 + 3 * k, 4 + 3 * k) for k in range(len(MS))),
     'Setting & ' + ' & '.join(['E & D & CIF'] * len(MS)) + r'\\', r'\midrule']
for lab in dict.fromkeys(R.setting):
    cells = []
    for m in MS:
        q = R[(R.setting == lab) & (R.method == m)]
        if len(q) == 0:
            cells += ['--', '--', '--']; continue
        q = q.iloc[0]
        cif = np.nanmean([q.get('CIF1', np.nan), q.get('CIF2', np.nan)])
        cells += [f(q.get('Pcorr_events', np.nan), 2), f(q.get('Pcorr_dropout', np.nan), 2), f(cif, 2)]
    C.append(lab + ' & ' + ' & '.join(cells) + r'\\')
C += [r'\bottomrule', r'\end{tabular}']
open(os.path.join(out, 'tab_study2_compact.tex'), 'w').write('\n'.join(C) + '\n')
print('written to', out)
