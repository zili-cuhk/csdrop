"""Exp F summaries: coverage of the 95% Imbens-Manski intervals with bootstrap standard errors.
Usage: python3 agg_expF.py results/expF [outdir]
For each scenario, cause and time: coverage of the true cohort-averaged CIF by
 (a) the IM interval with bootstrap standard errors (Section 3.6, Theorem 4(iv)),
 (b) the conservative interval of Section S10.9 (min/max of pointwise 95% limits),
 (c) the IM interval with the Monte Carlo standard deviations of the bounds (as in Exp C),
the average interval widths, and the ratio of the mean bootstrap SE to the Monte Carlo SD."""
import sys, os, glob, numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
d = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else '.'
D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, 'expF_*.csv'))], ignore_index=True)
G = len([c for c in D.columns if c.startswith('F_g')])
def cn(width, s):
    if not s > 0: return np.nan
    f = lambda c: norm.cdf(c + width / s) - norm.cdf(-c) - 0.95
    return brentq(f, 0, 5)
rows = []
for (sid, k, t), g in D.groupby(['sid', 'cause', 't']):
    L, U, tr = g.L.values, g.U.values, g.truth.values
    c = np.array([cn(u - l, max(a, b)) for l, u, a, b in zip(L, U, g.seL, g.seU)])
    lo, hi = L - c * g.seL.values, U + c * g.seU.values
    F = g[[f'F_g{j}' for j in range(1, G + 1)]].values; S = g[[f'se_g{j}' for j in range(1, G + 1)]].values
    clo, chi = (F - 1.96 * S).min(1), (F + 1.96 * S).max(1)
    sdL, sdU = L.std(ddof=1), U.std(ddof=1)
    cm = np.array([cn(u - l, max(sdL, sdU)) for l, u in zip(L, U)])
    mlo, mhi = L - cm * sdL, U + cm * sdU
    rows.append(dict(sid=sid, cause=k, t=t, reps=len(g), B_ok=g.B_ok.mean(),
                     cover_boot=np.mean((lo <= tr) & (tr <= hi)), width_boot=np.mean(hi - lo),
                     cover_cons=np.mean((clo <= tr) & (tr <= chi)), width_cons=np.mean(chi - clo),
                     cover_mc=np.mean((mlo <= tr) & (tr <= mhi)), width_mc=np.mean(mhi - mlo),
                     seL_over_sd=g.seL.mean() / sdL, seU_over_sd=g.seU.mean() / sdU,
                     argmin_end=np.mean(g.gL == 1), argmax_end=np.mean(g.gU == G)))
R = pd.DataFrame(rows)
os.makedirs(out, exist_ok=True); R.to_csv(os.path.join(out, 'expF_summary.csv'), index=False)
pd.set_option('display.width', 200)
print(R.round(3).to_string(index=False))
print('\nOverall: coverage boot %.3f (range %.3f-%.3f), conservative %.3f, MC %.3f; SE/SD %.2f-%.2f' % (
    R.cover_boot.mean(), R.cover_boot.min(), R.cover_boot.max(), R.cover_cons.mean(), R.cover_mc.mean(),
    min(R.seL_over_sd.min(), R.seU_over_sd.min()), max(R.seL_over_sd.max(), R.seU_over_sd.max())))
