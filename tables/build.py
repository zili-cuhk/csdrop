"""Build the main-text and supplementary tables of the revised simulation section
from the outputs of analysis/agg_liu.py and analysis/agg_study2.py."""
import os, re, numpy as np, pandas as pd
import pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parents[1]
RES = _pl.Path(os.environ.get('CSDROP_RESULTS', ROOT / 'results'))
SRC_L = str(RES / 'summary' / 'liu'); SRC_S = str(RES / 'summary' / 'study2')
OUT = str(ROOT / 'output' / 'tables'); os.makedirs(OUT, exist_ok=True)
E = pd.read_csv(f'{SRC_L}/liu_estimation.csv')
MARG = ['event1', 'event2', 'dropout']
VLAB = {'event1': r'\beta_{1%d}', 'event2': r'\beta_{2%d}', 'dropout': r'\beta_{C%d}'}
def f3(x): return '--' if not np.isfinite(x) else f'{x:.3f}'
def fb(x): return '--' if not np.isfinite(x) else f'{x:.3f}'.replace('-', '$-$')

def est_rows(signal, tau, NS, kind='cond'):
    T = E[(E.signal == signal) & np.isclose(E.tau, tau)]
    meths = ['Proposed', 'WI', 'Oracle']
    head = [r'\toprule',
            ' & & & & ' + ' & '.join(r'\multicolumn{4}{c}{%s}' % m for m in meths) + r'\\',
            ''.join(r'\cmidrule(lr){%d-%d}' % (5 + 4 * k, 8 + 4 * k) for k in range(3)),
            r'$(n,p)$ & Para & True & SelRate & ' + ' & '.join(['Bias & SE & ASE & CP'] * 3) + r'\\', r'\midrule']
    body = []; first = True
    for n in NS:
        for p in (10, 40):
            Tp = T[(T.n == n) & (T.p == p)]
            if len(Tp) == 0: continue
            if not first: body.append(r'\addlinespace')
            first = False; k = 0
            for m in MARG:
                for j in sorted(Tp[Tp.margin == m]['var'].unique()):
                    cells = []
                    for meth in meths:
                        r = Tp[(Tp.margin == m) & (Tp['var'] == j) & (Tp.method == meth)].iloc[0]
                        cells += [fb(r['Bias_' + kind]), f3(r['SE_' + kind]), f3(r['ASE_' + kind]), f3(r['CP_' + kind])]
                    q = Tp[(Tp.margin == m) & (Tp['var'] == j) & (Tp.method == 'Proposed')].iloc[0]
                    lab = f'$({n},{p})$' if k == 0 else ''
                    body.append(f'{lab} & ${VLAB[m] % j}$ & {fb(q.true)} & {f3(q.SelRate)} & ' + ' & '.join(cells) + r'\\')
                    k += 1
    return 'llrr' + 'rrrr' * 3, head, body

def tabular(spec, head, body):
    return '\n'.join([r'\begin{tabular}{%s}' % spec] + head + body + [r'\bottomrule', r'\end{tabular}']) + '\n'

def longtable(spec, head, body, caption, label, ncol):
    L = [r'\begin{longtable}{%s}' % spec, r'\caption{%s}\label{%s}\\' % (caption, label)] + head + [r'\endfirsthead',
         r'\multicolumn{%d}{c}{Web Table \ref{%s} continued}\\' % (ncol, label)] + head + [r'\endhead',
         r'\midrule', r'\multicolumn{%d}{r}{Continued on next page}\\' % ncol, r'\endfoot', r'\bottomrule', r'\endlastfoot'] + body + [r'\end{longtable}']
    return '\n'.join(L) + '\n'

def split_tabular(path):
    L = open(path).read().strip().splitlines()
    spec = re.match(r'\\begin\{tabular\}\{(.*)\}', L[0]).group(1)
    i = L.index(r'\midrule')
    head = L[1:i + 1]; body = [x for x in L[i + 1:-2]]
    return spec, head, body

# main text
spec, head, body = est_rows('moderate', 0.5, (400, 1000))
open(f'{OUT}/tab_main_est.tex', 'w').write(tabular(spec, head, body))
os.system(f'cp {SRC_L}/tab_liu_sel_moderate.tex {OUT}/tab_main_sel.tex; cp {SRC_S}/tab_study2_compact.tex {OUT}/tab_main_expB.tex')

# supplement
s = []
s.append(r'\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}' '\n'
         r'\caption{Experiment A. Variable selection for the joint model of the three margins under moderate signals. The true model size is 9, and each entry is based on 500 replications.}\label{tab:supp-A-moderate}' '\n'
         + open(f'{SRC_L}/tab_liu_sel_moderate.tex').read() + r'\end{table}' '\n')
s.append(r'\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}' '\n'
         r'\caption{Experiment A. Variable selection for the joint model of the three margins under strong signals. The true model size is 9, and each entry is based on 500 replications.}\label{tab:supp-A-strong}' '\n'
         + open(f'{SRC_L}/tab_liu_sel_strong.tex').read() + r'\end{table}' '\n')
for sg, lab in (('moderate', 'tab:supp-A-margin-mod'), ('strong', 'tab:supp-A-margin-str')):
    spec, head, body = split_tabular(f'{SRC_L}/tab_liu_sel_{sg}_margins.tex')
    s.append(r'\begin{landscape}\footnotesize\setlength{\tabcolsep}{3pt}' '\n' +
             longtable(spec, head, body, f'Experiment A. Variable selection for each margin under {sg} signals (500 replications).', lab, 17) +
             r'\end{landscape}' '\n')
for sg, tau, lab in (('moderate', 0.5, 'tab:supp-A-est-mod05'), ('moderate', 0.2, 'tab:supp-A-est-mod02'), ('strong', 0.5, 'tab:supp-A-est-str05')):
    spec, head, body = est_rows(sg, tau, (400, 600, 1000))
    s.append(r'\begin{landscape}\footnotesize\setlength{\tabcolsep}{3pt}' '\n' +
             longtable(spec, head, body, f'Experiment A. Estimation of the active coefficients under {sg} signals and $\\tau_0={tau}$. Bias, SE, ASE, and CP are computed over the replications that select the coefficient, and SelRate is the selection rate of the proposed method.', lab, 16) +
             r'\end{landscape}' '\n')
spec, head, body = split_tabular(f'{SRC_S}/tab_study2.tex')
s.append(r'\begin{footnotesize}\setlength{\tabcolsep}{4pt}' '\n' +
         longtable(spec, head, body, 'Experiment B. Pcorr for each margin, joint numbers of false positives and false negatives, and integrated squared CIF error ($\\times10^3$) for each cause (500 replications).', 'tab:supp-B-full', 9) +
         r'\end{footnotesize}' '\n')
open(f'{OUT}/supp_tables.tex', 'w').write('\n'.join(s))
print('done')
# method labels used in the manuscript: Two-margin (Liu et al., 2026), HDCR (the
# proposed method under working independence) and copHCR (the proposed method)
def relabel(t):
    t = t.replace('Liu2026', 'Two-margin')
    t = re.sub(r'(?<![A-Za-z])WI(?![A-Za-z])', 'HDCR', t)
    return re.sub(r'(?<![A-Za-z])Proposed(?![A-Za-z])', 'copHCR', t)
for fn in ('tab_main_sel.tex', 'tab_main_expB.tex', 'supp_tables.tex'):
    t = relabel(open(f'{OUT}/{fn}').read()); open(f'{OUT}/{fn}', 'w').write(t)
