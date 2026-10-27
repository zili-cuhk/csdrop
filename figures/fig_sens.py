"""Figure 3: population pseudo-true values, single-analysis selection, and the sensitivity-set summaries (Experiment E, strong signals)."""
import os as _os, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parents[1]
RES = _pl.Path(_os.environ.get('CSDROP_RESULTS', ROOT / 'results'))
OUTFIG = ROOT / 'output' / 'figures'; OUTFIG.mkdir(parents=True, exist_ok=True)
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family': ['cmr10', 'STIXGeneral'], 'mathtext.fontset': 'cm',
                     'axes.formatter.use_mathtext': True, 'font.size': 10, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'axes.edgecolor': '#52514e', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'pdf.fonttype': 42})
ETA, TAU0 = 0.05, 0.5
INK, MUTED, GRID = '#0b0b0b', '#9b9a96', '#e6e5e1'
BLUE, ORANGE = '#2a78d6', '#eb6834'
XT = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5]; XTL = ['$-0.3$', '', '', '0', '', '', '0.3', '', '0.5']

# (a) pseudo-true values in the design of Experiment E (sample of 40,000)
T = json.load(open(RES / 'pseudotrue' / 'expD_targets.json'))
Bt = np.array(T['Btrue']); tg = sorted(T['res'], key=float); taus = np.array([float(t) for t in tg])
X = np.abs(np.array([T['res'][t]['X'] for t in tg]))
dep = (2, 2)                                       # dropout coefficient of covariate 3
truem = (Bt != 0).copy(); truem[dep] = False
zero = Bt == 0
true_min = X[:, truem].min(1); leak_max = X[:, zero].max(1); dep_val = X[:, dep[0], dep[1]]

# (b) single-analysis selection frequencies, n = 5000
F = pd.read_csv(RES / 'summary' / 'expD' / 'expD_selfreq.csv')
true_cols = [f'{m}Z{j+1}' for i, m in enumerate(['e1', 'e2', 'd']) for j in range(10) if truem[i, j]]
leaks = ['e1Z7', 'e2Z2', 'e2Z4', 'e2Z7', 'dZ5', 'dZ6']
g = F[F.n == 5000].sort_values('tau')

# (c) robust and possible supports over [-0.3, 0.5]
S = pd.read_csv(RES / 'summary' / 'expD' / 'expD_summary.csv')
NS = [600, 2000, 5000]
rob = S[S.method == 'robust wide [-0.3,0.5]'].set_index('n'); pos = S[S.method == 'possible wide [-0.3,0.5]'].set_index('n')
rob_target = rob.loc[NS, 'P_eq_target'].values; rob_dep = rob.loc[NS, 'P_Z3'].values; pos_dep = pos.loc[NS, 'P_Z3'].values
pos_leak = pos.loc[NS, 'leak_recall'].values

fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.9))
def base(a):
    a.grid(axis='y', color=GRID, lw=0.5); a.set_axisbelow(True); a.spines[['top', 'right']].set_visible(False)
def taux(a):
    a.set_xlim(-0.33, 0.53); a.set_xticks(XT); a.set_xticklabels(XTL); a.set_xlabel(r"Assumed Kendall's $\tau$")
    a.axvline(TAU0, color=MUTED, lw=0.7, zorder=1)
    a.text(TAU0, a.get_ylim()[1], r' $\tau_0$', color='#52514e', fontsize=9.5, ha='left', va='top')

a = ax[0]; base(a)
a.set_ylim(0, 0.2); a.set_yticks([0, 0.05, 0.1, 0.15, 0.2])
a.axhline(ETA, color=INK, lw=0.6, zorder=1); a.text(-0.32, ETA + 0.004, r'$\eta$', fontsize=10, va='bottom')
a.plot(taus, leak_max, color=BLUE, lw=1.5, marker='v', ms=3.2, zorder=3)
a.plot(taus, dep_val, color=ORANGE, lw=1.5, marker='o', ms=3.2, zorder=3)
taux(a); a.set_ylabel('Absolute pseudo-true value')
a.set_title('(a) Pseudo-true values', fontsize=10, loc='left')

a = ax[1]; base(a)
a.set_ylim(-0.03, 1.03); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
a.plot(g.tau, g[true_cols].min(axis=1), color=INK, lw=1.5, zorder=3)
a.plot(g.tau, g[leaks].max(axis=1), color=BLUE, lw=1.5, marker='v', ms=3.2, zorder=3)
a.plot(g.tau, g['dZ3'], color=ORANGE, lw=1.5, marker='o', ms=3.2, zorder=4)
taux(a); a.set_ylabel('Selection frequency')
a.set_title('(b) Single analysis, $n=5000$', fontsize=10, loc='left')

a = ax[2]; base(a)
a.set_ylim(-0.03, 1.03); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
xs = np.arange(3)
a.plot(xs, rob_target, color='0.5', lw=1.5, marker='s', ms=4, zorder=3)
a.plot(xs, pos_dep, color=ORANGE, lw=1.5, marker='o', ms=4, zorder=3)
a.plot(xs, rob_dep, color=ORANGE, lw=1.5, ls=(0, (3, 2)), marker='o', ms=4, mfc='white', zorder=3)
a.set_xticks(xs); a.set_xticklabels([str(n) for n in NS]); a.set_xlim(-0.3, 2.3); a.set_xlabel('$n$')
a.set_ylabel('Proportion of replications')
a.set_title('(c) Sensitivity-set summaries', fontsize=10, loc='left')
a.text(-0.2, 0.9, 'robust support\nequals its target', fontsize=8.5, ha='left', va='top', linespacing=1.05, color=INK)
a.text(1.2, 0.55, 'weak effect\nin possible support', fontsize=8.5, ha='left', va='top', linespacing=1.05, color=INK)
a.text(1.1, 0.06, 'weak effect in robust support', fontsize=8.5, ha='center', va='bottom', color=INK)

h = [Line2D([], [], color=INK, lw=1.5, label='True effects (smallest)'),
     Line2D([], [], color=BLUE, lw=1.5, marker='v', ms=3.2, label='Leaking zero coefficients (largest)'),
     Line2D([], [], color=ORANGE, lw=1.5, marker='o', ms=3.2, label='Weak dropout effect')]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.015),
           handlelength=1.8, columnspacing=1.8)
fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.0)
fig.savefig(OUTFIG / 'figure_sensitivity.pdf')
print('true_min', true_min.min().round(3), 'dep', dep_val.round(3), 'leak', leak_max.round(3))
print(rob_target, pos_dep, rob_dep, pos_leak)
