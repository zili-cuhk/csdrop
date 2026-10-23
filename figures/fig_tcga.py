"""Figure: cohort-averaged CIFs in the lung adenocarcinoma analysis, with the Aalen-Johansen estimate."""
import os as _os, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parents[1]
RES = _pl.Path(_os.environ.get('CSDROP_RESULTS', ROOT / 'results'))
OUTFIG = ROOT / 'output' / 'figures'; OUTFIG.mkdir(parents=True, exist_ok=True)
import sys, json, numpy as np, matplotlib
sys.path.insert(0, str(ROOT / 'application'))
from prep import prepare
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': ['cmr10', 'STIXGeneral'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'font.size': 8.5, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
R = {a: json.load(open(ROOT / 'application' / 'out' / f'orig_a{a:02d}.json')) for a in range(1, 7)}
d = prepare('exclude_unknown')
Y = d['Y']; st = d['e1'] * 1 + d['e2'] * 2
ts = np.unique(Y[st > 0]); S = 1.0; F = np.zeros((2, len(ts) + 1)); tt = [0.0]
for i, t in enumerate(ts):
    nr = (Y >= t).sum(); d1 = ((Y == t) & (st == 1)).sum(); d2 = ((Y == t) & (st == 2)).sum()
    F[0, i + 1] = F[0, i] + S * d1 / nr; F[1, i + 1] = F[1, i] + S * d2 / nr; S *= 1 - (d1 + d2) / nr; tt.append(t)
tt = np.array(tt)
blues = {1: '#c6dbef', 2: '#6baed6', 3: '#2171b5', 4: '#08306b'}
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9))
titles = ['Recorded PFS event', 'Death without an earlier PFS event']
from matplotlib.patches import Patch
for k in range(2):
    a = ax[k]
    keep = tt <= 50
    g = np.array(R[1]['grid']); Fs = np.array([np.array(R[aid]['cif'])[k] for aid in (1, 2, 3, 4)])
    a.fill_between(g, Fs.min(0), Fs.max(0), color='#9ecae1', lw=0, alpha=0.9, zorder=1)
    a.plot(g, np.array(R[3]['cif'])[k], color='#08519c', lw=1.3, zorder=3)
    a.plot(R[5]['grid'], np.array(R[5]['cif'])[k], color='#d95f02', lw=1.1, ls='--', zorder=3)
    a.plot(R[6]['grid'], np.array(R[6]['cif'])[k], color='#d95f02', lw=1.1, ls=':', zorder=3)
    a.step(tt[keep], F[k, keep], where='post', color='0.35', lw=0.8, zorder=2)
    a.set_title(titles[k], fontsize=8.5); a.set_xlabel('Months'); a.set_xlim(0, 50); a.set_xticks([0, 12, 24, 36, 48])
    a.set_ylim(0, 0.52 if k == 0 else 0.23)
    a.grid(color='0.92', lw=0.5); a.spines[['top', 'right']].set_visible(False)
ax[0].set_ylabel('Cohort-averaged CIF')
from matplotlib.lines import Line2D
h = [Patch(color='#9ecae1', label=r'Frank, $\tau\in[-0.3,0.5]$'), Line2D([], [], color='#08519c', lw=1.3, label=r'Frank, $\tau=0.2$'),
     Line2D([], [], color='0.35', lw=0.8, label='Aalen\u2013Johansen'),
     Line2D([], [], color='#d95f02', lw=1.1, ls='--', label=r'Unknown as administrative, $\tau=0.2$'),
     Line2D([], [], color='#d95f02', lw=1.1, ls=':', label=r'Unknown as dropout, $\tau=0.2$')]
fig.legend(handles=[h[0], h[3], h[1], h[4], h[2]], loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=8)
fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.2)
fig.savefig(OUTFIG / 'figure_tcga_cif_sensitivity.pdf')
for t in (12, 24, 36):
    i = np.searchsorted(tt, t, side='right') - 1; print(t, F[:, i].round(3))
