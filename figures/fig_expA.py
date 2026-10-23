"""Figure: Experiment A selection under moderate signals (joint Pcorr and MSE relative to the oracle)."""
import os as _os, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parents[1]
RES = _pl.Path(_os.environ.get('CSDROP_RESULTS', ROOT / 'results'))
OUTFIG = ROOT / 'output' / 'figures'; OUTFIG.mkdir(parents=True, exist_ok=True)
import pandas as pd, numpy as np, matplotlib, glob
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': ['cmr10', 'STIXGeneral'], 'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True, 'font.size': 8.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'pdf.fonttype': 42})
S = pd.read_csv(RES / 'summary' / 'liu' / 'liu_selection.csv')
S = S[(S.margin == 'joint') & (S.signal == 'moderate')]
NS = [400, 600, 1000]
STY = {  # label, color, marker, linestyle
    'Full': ('Full', '#7f7f7f', 'x', ':'),
    'Lasso': ('Lasso', '#1b9e77', 's', '--'),
    'Liu': ('Two-margin', '#7570b3', '^', '-.'),
    'WI': ('WI', '#d95f02', 'D', '-'),
    'Proposed': ('Proposed', '#000000', 'o', '-'),
}
cols = [(10, 0.2), (10, 0.5), (40, 0.2), (40, 0.5)]
R = str(RES)
sc = pd.read_csv(ROOT / 'scenarios.csv'); A = sc[sc.sid.str.match(r'A\d\d')][['sid', 'n', 'p', 'signal', 'tau']]
def load(pat, crit):
    d = pd.concat(pd.read_csv(f) for f in glob.glob(pat)); d = d[d.criterion == crit]
    return d.groupby('sid').ise.mean()
CIF = {'Lasso': load(f'{R}/comp/cif_CS_LASSO_A*.csv', 'bic'), 'Liu': load(f'{R}/prop/cif_LIU2026_A*.csv', 'EBIC0.25'),
       'WI': load(f'{R}/prop/cif_PROP_T00_A*.csv', 'EBIC0.25'),
       'Proposed': pd.concat([load(f'{R}/prop/cif_PROP_T02_A*.csv', 'EBIC0.25')[A[np.isclose(A.tau, 0.2)].sid.values],
                              load(f'{R}/prop/cif_PROP_T05_A*.csv', 'EBIC0.25')[A[np.isclose(A.tau, 0.5)].sid.values]])}
A = A.set_index('sid')
fig, ax = plt.subplots(3, 4, figsize=(6.6, 5.6), sharex=True)
for c, (p, tau) in enumerate(cols):
    T = S[(S.p == p) & np.isclose(S.tau, tau)].set_index(['method', 'n'])
    orc = T.loc['Oracle']['MSE']
    for m in ['Full', 'Lasso', 'Liu', 'WI', 'Proposed']:
        lab, col, mk, ls = STY[m]
        y1 = [T.loc[(m, n), 'Pcorr'] for n in NS]
        y2 = [T.loc[(m, n), 'MSE'] / orc.loc[n] for n in NS]
        if m != 'Full':
            ax[0, c].plot(NS, y1, color=col, marker=mk, ls=ls, lw=(1.4 if m == 'Proposed' else 1.0), ms=4.0, mew=0.9, mfc=(col if m in ('Proposed', 'Full') else 'white'), label=lab)
        ax[1, c].plot(NS, y2, color=col, marker=mk, ls=ls, lw=(1.4 if m == 'Proposed' else 1.0), ms=4.0, mew=0.9, mfc=(col if m in ('Proposed', 'Full') else 'white'), label=lab)
        if m != 'Full':
            sid = {n: A[(A.n == n) & (A.p == p) & (A.signal == 'moderate') & np.isclose(A.tau, tau)].index[0] for n in NS}
            y3 = [CIF[m][sid[n]] / CIF['Proposed'][sid[n]] for n in NS]
            ax[2, c].plot(NS, y3, color=col, marker=mk, ls=ls, lw=(1.4 if m == 'Proposed' else 1.0), ms=4.0, mew=0.9, mfc=(col if m in ('Proposed', 'Full') else 'white'))
    ax[0, c].set_title(f'$p={p}$, $\\tau_0={tau}$', fontsize=8.5)
    ax[0, c].set_ylim(-0.03, 1.03); ax[0, c].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1, c].set_yscale('log'); ax[1, c].set_ylim(0.8, 40)
    ax[1, c].axhline(1, color='0.6', lw=0.6, ls='--')
    ax[1, c].set_yticks([1, 2, 5, 10, 20]); ax[1, c].set_yticklabels(['1', '2', '5', '10', '20'])
    ax[1, c].minorticks_off()
    ax[2, c].set_yscale('log'); ax[2, c].set_ylim(0.8, 14); ax[2, c].axhline(1, color='0.6', lw=0.6, ls='--')
    ax[2, c].set_yticks([1, 2, 5, 10]); ax[2, c].set_yticklabels(['1', '2', '5', '10']); ax[2, c].minorticks_off()
    for r in range(3):
        ax[r, c].set_xticks(NS); ax[r, c].set_xlim(330, 1070)
        ax[r, c].grid(axis='y', color='0.9', lw=0.5)
        ax[r, c].spines[['top', 'right']].set_visible(False)
        if c > 0: ax[r, c].set_yticklabels([])
    ax[2, c].set_xlabel('$n$')
ax[0, 0].set_ylabel('Pcorr of the\njoint support')
ax[1, 0].set_ylabel('MSE relative\nto the oracle')
ax[2, 0].set_ylabel('CIF error relative\nto the proposed method')
h, l = ax[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.045, 1, 1), w_pad=0.6, h_pad=0.8)
fig.savefig(OUTFIG / 'figure_expA_selection.pdf')
