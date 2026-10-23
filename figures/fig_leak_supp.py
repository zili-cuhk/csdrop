"""Web Figure: pseudo-true values of the zero coefficients in each margin (Experiment A design, p = 10, tau0 = 0.5)."""
import os as _os, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parents[1]
RES = _pl.Path(_os.environ.get('CSDROP_RESULTS', ROOT / 'results'))
OUTFIG = ROOT / 'output' / 'figures'; OUTFIG.mkdir(parents=True, exist_ok=True)
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({'font.family': ['cmr10', 'STIXGeneral'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
                     'font.size': 8.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})
BLUE, GRAY, BAND = '#2a78d6', '#9b9a96', '#ecebe7'
ETA = 0.05
fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4), sharex=True, sharey='row')
titles = ['Event 1 margin', 'Event 2 margin', 'Dropout margin']
for row, sig in enumerate(['strong', 'moderate']):
    D = json.load(open(RES / 'pseudotrue' / f'path_{sig}_0.5_p10.json'))
    Bt = np.array(D['Btrue']); taus = np.array([r['tau'] for r in D['res']]); B = np.array([r['B'] for r in D['res']])
    active_any = (Bt != 0).any(0)
    for d in range(3):
        ax = axes[row, d]
        ax.axhspan(-ETA, ETA, color=BAND, zorder=0, lw=0)
        ax.axvline(0.5, color='0.55', lw=0.7, zorder=1)
        ax.axhline(0, color='0.75', lw=0.5, zorder=1)
        labs = []
        for j in range(Bt.shape[1]):
            if Bt[d, j] != 0: continue
            y = B[:, d, j]
            if active_any[j]:
                ax.plot(taus, y, color=BLUE, lw=1.3, zorder=3); labs.append([y[0], f'$Z_{{{j+1}}}$'])
            else:
                ax.plot(taus, y, color=GRAY, lw=0.9, ls=(0, (1.5, 1.5)), zorder=2)
        labs.sort(key=lambda t: t[0])
        for i in range(1, len(labs)):
            if labs[i][0] - labs[i - 1][0] < 0.032: labs[i][0] = labs[i - 1][0] + 0.032
        for yv, txt in labs:
            ax.text(taus[0] - 0.02, yv, txt, ha='right', va='center', fontsize=7.5)
        ax.set_xlim(-0.45, 0.68); ax.set_ylim(-0.22, 0.22); ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6])
        ax.spines[['top', 'right']].set_visible(False)
        if row == 0: ax.set_title(titles[d], fontsize=8.5)
        if d == 0: ax.set_ylabel(f'{sig.capitalize()} signals\npseudo-true coefficient')
        if row == 1: ax.set_xlabel(r"Assumed Kendall's $\tau$")
h = [Line2D([], [], color=BLUE, lw=1.3, label='Active in another margin'),
     Line2D([], [], color=GRAY, lw=0.9, ls=(0, (1.5, 1.5)), label=r'Inactive in every margin ($Z_8$ to $Z_{10}$)'),
     Patch(color=BAND, label=r'Absolute value below $\eta=0.05$'),
     Line2D([], [], color='0.55', lw=0.7, label=r'True value $\tau_0=0.5$')]
fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(OUTFIG / 'figure_support_leakage.pdf')
