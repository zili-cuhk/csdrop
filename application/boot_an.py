import os
_H = os.path.dirname(os.path.abspath(__file__))
import json, glob, numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

recs = []
for f in sorted(glob.glob(os.path.join(_H, 'out', 'boot_*.jsonl'))):
    recs += [json.loads(l) for l in open(f)]
grid = ['-0.3', '0.0', '0.2', '0.5']
ok = [r for r in recs if all(r['fits'].get(t) for t in grid)]
nb, nok = len(recs), len(ok)
S = np.array([[r['fits'][t]['S'] for t in grid] for r in ok])        # b x g x 3 x 6
F = np.array([[r['fits'][t]['F'] for t in grid] for r in ok])        # b x g x 2 x 3
full_S = np.zeros((3, 6), int); full_S[0, 2] = 1; full_S[1, 3] = 1
Lf = np.array([[0.1502, 0.2856, 0.3878], [0.0793, 0.1291, 0.1660]])
Uf = np.array([[0.1674, 0.3098, 0.4142], [0.0858, 0.1378, 0.1755]])
freq = S.mean(0)                       # g x 3 x 6
rob = S.min(1); pos = S.max(1)         # b x 3 x 6
frob, fpos = rob.mean(0), pos.mean(0)
eq_rob = np.mean([(rob[b] == full_S).all() for b in range(nok)])
eq_pos = np.mean([(pos[b] == full_S).all() for b in range(nok)])
eq_g = [np.mean([(S[b, g] == full_S).all() for b in range(nok)]) for g in range(4)]
Lb, Ub = F.min(1), F.max(1)            # b x 2 x 3
argmin = F.argmin(1); argmax = F.argmax(1)
sL, sU = Lb.std(0, ddof=1), Ub.std(0, ddof=1)
sg = F.std(0, ddof=1)                  # g x 2 x 3

def cn(width, s):
    f = lambda c: norm.cdf(c + width / s) - norm.cdf(-c) - 0.95
    return brentq(f, 0, 5)

full_F = np.array([[[0.1502, 0.2856, 0.3878], [0.0793, 0.1291, 0.1660]],
                   [[0.1543, 0.2935, 0.3973], [0.0808, 0.1319, 0.1694]],
                   [[0.1584, 0.2995, 0.4040], [0.0824, 0.1342, 0.1719]],
                   [[0.1674, 0.3098, 0.4142], [0.0858, 0.1378, 0.1755]]])
def iqr_se(A, axis=0):
    # normalized interquartile range of the bootstrap replicates (Web Appendix S10)
    q75, q25 = np.quantile(A, [0.75, 0.25], axis=axis)
    return (q75 - q25) / (2 * norm.ppf(0.75))

def intervals(sL, sU, sg):
    # Imbens-Manski interval (Section 3.6) and envelope interval (Theorem 4(v))
    IM = np.zeros((2, 3, 2)); ENV = np.zeros((2, 3, 2)); C = np.zeros((2, 3))
    for k in range(2):
        for t in range(3):
            c = cn(Uf[k, t] - Lf[k, t], max(sL[k, t], sU[k, t])); C[k, t] = c
            IM[k, t] = [Lf[k, t] - c * sL[k, t], Uf[k, t] + c * sU[k, t]]
            ENV[k, t] = [min(full_F[:, k, t] - 1.96 * sg[:, k, t]), max(full_F[:, k, t] + 1.96 * sg[:, k, t])]
    return IM, ENV, C

IM, ENV, C = intervals(sL, sU, sg)
qL, qU, qg = iqr_se(Lb), iqr_se(Ub), iqr_se(F)
IMq, ENVq, _ = intervals(qL, qU, qg)
width_over_se = (Uf - Lf) / np.maximum(sL, sU)
names = ['Age per 10 years', 'Male sex', 'Stage II', 'Stage III--IV', 'Node-positive', 'T3--T4']
margins = ['Recorded PFS event', 'Death without earlier PFS', 'Dropout']
print('records', nb, 'ok', nok)
print('eq joint per tau', np.round(eq_g, 3), 'robust', round(eq_rob, 3), 'possible', round(eq_pos, 3))
print('argmin at -0.3 share', (argmin == 0).mean(), 'argmax at 0.5 share', (argmax == 3).mean())
for m in range(3):
    for j in range(6):
        print(margins[m][:10], names[j][:12], np.round(freq[:, m, j], 3), round(frob[m, j], 3), round(fpos[m, j], 3))
print('sL', np.round(sL, 4)); print('sU', np.round(sU, 4)); print('c', np.round(C, 3))
print('IM', np.round(IM, 3)); print('envelope', np.round(ENV, 3))
print('IQR se: sL', np.round(qL, 4), 'sU', np.round(qU, 4))
print('IM (IQR se)', np.round(IMq, 3)); print('envelope (IQR se)', np.round(ENVq, 3))
print('width of the bounds / max(sL, sU)', np.round(width_over_se, 2))

# LaTeX outputs
rows = []
for m in range(3):
    for j in range(6):
        lab = margins[m] if j == 0 else ''
        vals = ' & '.join(f'{v:.2f}' for v in freq[:, m, j])
        rows.append(f'{lab} & {names[j]} & {vals} & {frob[m, j]:.2f} & {fpos[m, j]:.2f}\\\\')
    if m < 2: rows.append('\\addlinespace')
open(os.path.join(_H, 'out', 'tab_app_freq.tex'), 'w').write('\n'.join(rows) + '\n')
causes = ['Recorded PFS event', 'Death without earlier PFS']
pet = {(0, 0): (0.162, 0.231), (0, 1): (0.312, 0.582), (0, 2): (0.356, 0.688),
       (1, 0): (0.076, 0.145), (1, 1): (0.109, 0.379), (1, 2): (0.125, 0.457)}
fam = {(0, 0): (0.150, 0.170), (0, 1): (0.286, 0.310), (0, 2): (0.388, 0.414),
       (1, 0): (0.079, 0.087), (1, 1): (0.129, 0.138), (1, 2): (0.166, 0.176)}
rows = []
for k in range(2):
    for t, mo in enumerate([12, 24, 36]):
        lab = causes[k] if t == 0 else ''
        rows.append(f'{lab} & {mo} & [{Lf[k,t]:.3f}, {Uf[k,t]:.3f}] & [{fam[(k,t)][0]:.3f}, {fam[(k,t)][1]:.3f}] & '
                    f'[{IM[k,t,0]:.3f}, {IM[k,t,1]:.3f}] & [{ENV[k,t,0]:.3f}, {ENV[k,t,1]:.3f}] & '
                    f'[{pet[(k,t)][0]:.3f}, {pet[(k,t)][1]:.3f}]\\\\')
    if k == 0: rows.append('\\addlinespace')
open(os.path.join(_H, 'out', 'tab_app_cif.tex'), 'w').write('\n'.join(rows) + '\n')
mac = {'appB': nb, 'appBok': nok, 'appEqRob': f'{100*eq_rob:.1f}', 'appEqPos': f'{100*eq_pos:.1f}',
       'appEqMin': f'{100*min(eq_g):.1f}', 'appEqMax': f'{100*max(eq_g):.1f}',
       'appIMpfs': f'{IM[0,2,0]:.3f} to {IM[0,2,1]:.3f}', 'appENVpfs': f'{ENV[0,2,0]:.3f} to {ENV[0,2,1]:.3f}', 'appArgEnd': f'{100*min((argmin==0).mean(), (argmax==3).mean()):.0f}'}
open(os.path.join(_H, 'out', 'app_numbers.tex'), 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in mac.items()))
json.dump(dict(freq=freq.tolist(), frob=frob.tolist(), fpos=fpos.tolist(), IM=IM.tolist(), ENV=ENV.tolist(),
               IM_iqr=IMq.tolist(), ENV_iqr=ENVq.tolist(), sL_iqr=qL.tolist(), sU_iqr=qU.tolist(),
               width_over_se=width_over_se.tolist(), sL=sL.tolist(), sU=sU.tolist(), nb=nb, nok=nok, eq_g=eq_g, eq_rob=eq_rob, eq_pos=eq_pos),
          open(os.path.join(_H, 'out', 'boot_summary.json'), 'w'), indent=1)
