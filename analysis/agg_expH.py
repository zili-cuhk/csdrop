"""Exp H summaries: discretization error of the grid bounds (Theorem 5).

Usage: python3 analysis/agg_expH.py results/expH [outdir] [scenarios.csv]

Input: files expH_<sid>_<rep>.csv of run_expF_fits.m with tag expH, the fine
grid 0:0.025:0.5 (21 values) and B = 0. For every replication, cause and time,
the bounds over the subgrids with G = 3, 6, 11 equally spaced values
(mesh 0.25, 0.1, 0.05) are compared with the bounds over the fine grid
(G = 21, mesh 0.025). The discretization errors
    eL = L_G - L_21 >= 0,   eU = U_21 - U_G >= 0
are reported relative to the Monte Carlo standard deviation of the fine-grid
bound across replications, as mean and 95th percentile. Theorem 5 requires
the mesh to be o(n^{-1/2}), so the ratios should be small when mesh * sqrt(n)
is small. The proportion of replications whose fine-grid bound is attained at
an interior grid value is also given (an interior bound is where a coarse grid
can miss the continuum bound).
Outputs expH_summary.csv and tab_expH.tex (maximum over cause and time)."""
import sys, os, glob, numpy as np, pandas as pd

d = sys.argv[1].rstrip('/')
out = sys.argv[2] if len(sys.argv) > 2 else '.'
scen = sys.argv[3] if len(sys.argv) > 3 else os.path.join(d, '..', '..', 'scenarios.csv')
D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(d, 'expH_*.csv'))], ignore_index=True)
D = D.drop_duplicates(['sid', 'rep', 'cause', 't'], keep='last')
GF = len([c for c in D.columns if c.startswith('F_g')])
if GF != 21:
    sys.exit(f'expected the 21-point grid 0:0.025:0.5, found {GF} grid values')
SUB = {3: np.arange(0, 21, 10), 6: np.arange(0, 21, 4), 11: np.arange(0, 21, 2), 21: np.arange(21)}

rows = []
for (sid, k, t), g in D.groupby(['sid', 'cause', 't']):
    F = g[[f'F_g{j}' for j in range(1, GF + 1)]].values
    F = F[np.all(np.isfinite(F), 1)]
    L21, U21 = F.min(1), F.max(1)
    sdL, sdU = L21.std(ddof=1), U21.std(ddof=1)
    iL, iU = F.argmin(1), F.argmax(1)
    for G, idx in SUB.items():
        eL = F[:, idx].min(1) - L21
        eU = U21 - F[:, idx].max(1)
        rows.append(dict(sid=sid, cause=k, t=t, G=G, mesh=0.5 / (G - 1), reps=len(F),
                         eL_mean=eL.mean() / sdL, eL_q95=np.quantile(eL, 0.95) / sdL,
                         eU_mean=eU.mean() / sdU, eU_q95=np.quantile(eU, 0.95) / sdU,
                         sdL=sdL, sdU=sdU,
                         Lint=np.mean((iL > 0) & (iL < GF - 1)), Uint=np.mean((iU > 0) & (iU < GF - 1))))
R = pd.DataFrame(rows)
if os.path.exists(scen):
    S = pd.read_csv(scen)[['sid', 'n', 'signal', 'tau']]
    R = S.merge(R, on='sid')
    R['mesh_sqrt_n'] = R.mesh * np.sqrt(R.n)
os.makedirs(out, exist_ok=True)
R.to_csv(os.path.join(out, 'expH_summary.csv'), index=False)
pd.set_option('display.width', 200)
print(R[R.G < 21].round(3).to_string(index=False))

if 'signal' in R.columns:
    M = (R[R.G < 21].groupby(['n', 'signal', 'tau', 'G'])
         [['eL_mean', 'eL_q95', 'eU_mean', 'eU_q95', 'Lint', 'Uint']].max().reset_index())
    f3 = lambda x: f'{x:.3f}'
    lines = [r'\begin{tabular}{rllrrrrrr}', r'\toprule',
             r' & & & & & \multicolumn{2}{c}{Lower bound} & \multicolumn{2}{c}{Upper bound}\\',
             r'\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
             r'$n$ & Signal & $\tau_0$ & $G$ & Mesh$\times n^{1/2}$ & Mean & 95\% & Mean & 95\%\\',
             r'\midrule']
    first = True
    for (n, sg, tau), h in M.groupby(['n', 'signal', 'tau']):
        if not first: lines.append(r'\addlinespace')
        first = False
        for i, (_, r) in enumerate(h.sort_values('G').iterrows()):
            lab = f'{n} & {sg} & {tau:g}' if i == 0 else ' & & '
            lines.append(f'{lab} & {int(r.G)} & {0.5 / (r.G - 1) * np.sqrt(n):.2f} & {f3(r.eL_mean)} & '
                         f'{f3(r.eL_q95)} & {f3(r.eU_mean)} & {f3(r.eU_q95)}\\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(os.path.join(out, 'tab_expH.tex'), 'w').write('\n'.join(lines) + '\n')
    print('table written to', os.path.join(out, 'tab_expH.tex'))
