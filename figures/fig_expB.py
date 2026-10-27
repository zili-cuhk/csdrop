"""Figure: Experiment B dot plot (Pcorr of the event margins, Pcorr of dropout, and CIF error)."""
import os as _os, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parents[1]
RES = _pl.Path(_os.environ.get('CSDROP_RESULTS', ROOT / 'results'))
OUTFIG = ROOT / 'output' / 'figures'; OUTFIG.mkdir(parents=True, exist_ok=True)
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': ['cmr10', 'STIXGeneral'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'font.size': 8.5, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
D = pd.read_csv(RES / 'summary' / 'study2' / 'study2.csv')
D['method'] = D.method.replace({'Liu2026': 'Two-margin'})
D['CIF'] = (D.CIF1 + D.CIF2) / 2
order = ['Base case', 'Dropout 10\\%', 'Dropout 35\\%', 'Staggered entry, 15\\%', 'Staggered entry, 30\\%',
         '$\\tau_0=-0.3$', 'Rare event 2', 'Weibull, 3 intervals', 'Weibull, 5 intervals', 'Clayton',
         'Gumbel--Hougaard', '$\\delta=0.5$', '$\\delta=1$', '$n=1000$, $p=100$']
lab = {o: o.replace('\\%', '%').replace('--', '\u2013') for o in order}
lab['$\\tau_0=-0.3$'] = r'$\tau_0=-0.3$'; lab['Weibull, 3 intervals'] = 'Weibull, 3 intervals'
lab['$n=1000$, $p=100$'] = r'$n=1000,\ p=100$'
STY = {'Lasso': ('#1b9e77', 's'), 'Two-margin': ('#7570b3', '^'), 'WI': ('#d95f02', 'D'), 'Proposed': ('#000000', 'o')}
offs = {'Lasso': 0.27, 'Two-margin': 0.09, 'WI': -0.09, 'Proposed': -0.27}
panels = [('Pcorr_events', 'Pcorr, both event margins'), ('Pcorr_dropout', 'Pcorr, dropout'), ('CIF', r'CIF error ($\times10^3$)')]
fig, ax = plt.subplots(1, 3, figsize=(6.6, 4.6), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1.15]})
ypos = {o: len(order) - 1 - i for i, o in enumerate(order)}
for j, (col, title) in enumerate(panels):
    for i, o in enumerate(order):
        if i % 2 == 0: ax[j].axhspan(ypos[o] - 0.5, ypos[o] + 0.5, color='0.95', lw=0)
    for m, (c, mk) in STY.items():
        T = D[D.method == m].set_index('setting')
        ys, xs = [], []
        for o in order:
            if o in T.index and np.isfinite(T.loc[o, col]):
                ys.append(ypos[o] + offs[m]); xs.append(T.loc[o, col])
        ax[j].plot(xs, ys, ls='none', marker=mk, color=c, ms=3.6, mfc=c if m == 'Proposed' else 'white', mew=0.9, label={'WI': 'HDCR', 'Proposed': 'copHCR'}.get(m, m))
    ax[j].set_title(title, fontsize=8.5)
    ax[j].set_ylim(-0.6, len(order) - 0.4)
    ax[j].grid(axis='x', color='0.88', lw=0.5)
    ax[j].spines[['top', 'right']].set_visible(False)
    ax[j].tick_params(axis='y', length=0)
ax[0].set_xlim(-0.03, 1.0); ax[1].set_xlim(-0.03, 1.0)
ax[2].set_xscale('log'); ax[2].set_xlim(0.4, 8); ax[2].set_xticks([0.5, 1, 2, 4, 8]); ax[2].set_xticklabels(['0.5', '1', '2', '4', '8']); ax[2].minorticks_off()
ax[0].set_yticks([ypos[o] for o in order]); ax[0].set_yticklabels([lab[o] for o in order])
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.58, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.8)
fig.savefig(OUTFIG / 'figure_expB.pdf')
