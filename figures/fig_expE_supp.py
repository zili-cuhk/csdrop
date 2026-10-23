"""Web Figure: Experiment E selection frequencies of the single analyses for each n (strong signals)."""
import os as _os, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parents[1]
RES = _pl.Path(_os.environ.get('CSDROP_RESULTS', ROOT / 'results'))
OUTFIG = ROOT / 'output' / 'figures'; OUTFIG.mkdir(parents=True, exist_ok=True)
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': ['cmr10', 'STIXGeneral'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True,
                     'font.size': 8.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})
BLUE, ORANGE, INK = '#2a78d6', '#eb6834', '#0b0b0b'
F = pd.read_csv(RES / 'summary' / 'expD' / 'expD_selfreq.csv')
B0 = np.array(json.load(open(RES / 'pseudotrue' / 'expD_targets.json'))['Btrue']) != 0
true_cols = [f'{m}Z{j+1}' for i, m in enumerate(['e1', 'e2', 'd']) for j in range(10) if B0[i, j] and not (i == 2 and j == 2)]
leaks = ['e1Z7', 'e2Z2', 'e2Z4', 'e2Z7', 'dZ5', 'dZ6']
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6), sharey=True)
for ax, n in zip(axes, (600, 2000, 5000)):
    g = F[F.n == n].sort_values('tau'); t = g.tau.values
    ax.axvline(0.5, color='0.55', lw=0.7)
    ax.plot(t, g[true_cols].min(axis=1), color=INK, lw=1.2)
    for c in leaks: ax.plot(t, g[c], color=BLUE, lw=0.8)
    ax.plot(t, g['dZ3'], color=ORANGE, lw=1.8)
    ax.set_title(f'$n={n}$', fontsize=8.5); ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks([-0.3, 0, 0.3, 0.5]); ax.set_xticklabels(['$-0.3$', '0', '0.3', '0.5'])
    ax.grid(axis='y', color='0.92', lw=0.5); ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel(r"Assumed Kendall's $\tau$")
axes[0].set_ylabel('Selection frequency')
h = [Line2D([], [], color=INK, lw=1.2, label='True effects (smallest)'), Line2D([], [], color=BLUE, lw=0.8, label='Leaking zero coefficients'),
     Line2D([], [], color=ORANGE, lw=1.8, label=r'$\tau$-dependent dropout effect')]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(OUTFIG / 'figure_expE_selfreq.pdf')
