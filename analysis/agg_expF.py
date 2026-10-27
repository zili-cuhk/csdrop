"""Exp F and Exp G summaries: coverage of the 95% Imbens-Manski intervals with
bootstrap standard errors and of the envelope interval.

Usage: python3 analysis/agg_expF.py results/expF [outdir] [scenarios.csv]
       python3 analysis/agg_expF.py results/expG [outdir] [scenarios.csv]

The input folder holds the files <tag>_<sid>_<rep>.csv of run_expF_fits.m
(tag = expF for the grid {0, 0.25, 0.5}, expG for the near-tie grid
{0.4, 0.45, 0.5}). For each scenario, cause and time the summary reports
coverage of the true cohort-averaged CIF by
 (a) the IM interval with bootstrap standard errors (Section 3.6, Theorem 4(iv)),
 (b) the envelope interval E_n of Theorem 4(v), the minimum of the pointwise
     lower 95% limits to the maximum of the pointwise upper 95% limits,
 (c) the IM interval with the Monte Carlo standard deviations of the bounds,
the average widths, the ratio of the envelope width to the IM width, the median
of (U - L) / max(seL, seU) over replications, which is small in the near-tie
regime, and the ratio of the mean bootstrap SE to the Monte Carlo SD.
Outputs <tag>_summary.csv and the LaTeX table tab_<tag>.tex."""
import sys, os, glob, numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

d = sys.argv[1].rstrip('/')
tag = os.path.basename(d)
out = sys.argv[2] if len(sys.argv) > 2 else '.'
scen = sys.argv[3] if len(sys.argv) > 3 else os.path.join(d, '..', '..', 'scenarios.csv')
D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, tag + '_*.csv'))], ignore_index=True)
D = D.drop_duplicates(['sid', 'rep', 'cause', 't'], keep='last')
G = len([c for c in D.columns if c.startswith('F_g')])


def cn(width, s):
    if not s > 0: return np.nan
    f = lambda c: norm.cdf(c + width / s) - norm.cdf(-c) - 0.95
    return brentq(f, 0, 5)


rows = []
for (sid, k, t), g in D.groupby(['sid', 'cause', 't']):
    L, U, tr = g.L.values, g.U.values, g.truth.values
    sL, sU = g.seL.values, g.seU.values
    c = np.array([cn(u - l, max(a, b)) for l, u, a, b in zip(L, U, sL, sU)])
    lo, hi = L - c * sL, U + c * sU
    F = g[[f'F_g{j}' for j in range(1, G + 1)]].values
    S = g[[f'se_g{j}' for j in range(1, G + 1)]].values
    elo, ehi = (F - 1.96 * S).min(1), (F + 1.96 * S).max(1)
    sdL, sdU = L.std(ddof=1), U.std(ddof=1)
    cm = np.array([cn(u - l, max(sdL, sdU)) for l, u in zip(L, U)])
    mlo, mhi = L - cm * sdL, U + cm * sdU
    rows.append(dict(sid=sid, cause=k, t=t, reps=len(g), B_ok=g.B_ok.mean(),
                     cover_boot=np.mean((lo <= tr) & (tr <= hi)), width_boot=np.mean(hi - lo),
                     cover_env=np.mean((elo <= tr) & (tr <= ehi)), width_env=np.mean(ehi - elo),
                     cover_mc=np.mean((mlo <= tr) & (tr <= mhi)), width_mc=np.mean(mhi - mlo),
                     width_ratio=np.mean(ehi - elo) / np.mean(hi - lo),
                     width_over_se=np.median((U - L) / np.maximum(sL, sU)),
                     seL_over_sd=sL.mean() / sdL, seU_over_sd=sU.mean() / sdU,
                     argmin_end=np.mean(g.gL == 1), argmax_end=np.mean(g.gU == G)))
R = pd.DataFrame(rows)
if os.path.exists(scen):
    S = pd.read_csv(scen)[['sid', 'n', 'signal', 'tau']]
    R = S.merge(R, on='sid')
os.makedirs(out, exist_ok=True)
R.to_csv(os.path.join(out, f'{tag}_summary.csv'), index=False)
pd.set_option('display.width', 200)
print(R.round(3).to_string(index=False))
print('\nOverall: coverage IM %.3f (range %.3f-%.3f), envelope %.3f (range %.3f-%.3f), MC %.3f' % (
    R.cover_boot.mean(), R.cover_boot.min(), R.cover_boot.max(),
    R.cover_env.mean(), R.cover_env.min(), R.cover_env.max(), R.cover_mc.mean()))
print('envelope / IM width %.2f-%.2f; median width/SE %.2f-%.2f; SE/SD %.2f-%.2f' % (
    R.width_ratio.min(), R.width_ratio.max(), R.width_over_se.min(), R.width_over_se.max(),
    min(R.seL_over_sd.min(), R.seU_over_sd.min()), max(R.seL_over_sd.max(), R.seU_over_sd.max())))

# LaTeX: one row per (signal, tau0, cause, t)
if 'signal' in R.columns:
    f3 = lambda x: f'{x:.3f}'
    f2 = lambda x: f'{x:.2f}'
    lines = [r'\begin{tabular}{llrrrrrrr}', r'\toprule',
             r' & & & \multicolumn{2}{c}{IM} & \multicolumn{2}{c}{Envelope} & & \\',
             r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
             r'Signal & $\tau_0$ & $(k,t)$ & CP & Width & CP & Width & Width/SE & SE/SD\\', r'\midrule']
    first = True
    for (sg, tau), h in R.sort_values(['signal', 'tau', 'cause', 't']).groupby(['signal', 'tau'], sort=False):
        if not first: lines.append(r'\addlinespace')
        first = False
        for i, (_, r) in enumerate(h.iterrows()):
            lab = f'{sg} & {tau:g}' if i == 0 else ' & '
            sesd = 0.5 * (r.seL_over_sd + r.seU_over_sd)
            lines.append(f'{lab} & $({int(r.cause)},{r.t:g})$ & {f3(r.cover_boot)} & {f3(r.width_boot)} & '
                         f'{f3(r.cover_env)} & {f3(r.width_env)} & {f2(r.width_over_se)} & {f2(sesd)}\\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, f'tab_{tag}.tex'), 'w').write('\n'.join(lines) + '\n')
    print('table written to', os.path.join(out, f'tab_{tag}.tex'))
