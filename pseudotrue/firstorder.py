"""Check of the leakage formula d psi*/d theta = I^{-1} E[s_psi s_theta] at theta0 (Frank).
Compares (a) OPG projection, (b) Hessian implicit-function slope, (c) finite-difference slope of refits."""
import numpy as np, sys, json
from dgp import *
signal, tau0, n, p = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(11)
Y, st, Z, B = gen(n, p, signal, tau0, rng)
P = csm.prep(Y, st, Z, CUT, 2)
D = json.load(open(f'path_{signal}_{tau0}_p{p}.json'))
R = {r['tau']: r for r in D['res']}
X0, xi0 = np.array(R[tau0]['X']), np.array(R[tau0]['xi'])
th0 = csm.tau2theta('frank', tau0)
K, M = 2, P['M']

def ell_i(X, xi, th):
    a = np.exp(xi); eta = P['Zs'] @ X.T; H = np.exp(eta) * (P['E'] @ a.T)
    lu, lv = -H[:, :K].sum(1), -H[:, K]; u, v = np.exp(lu), np.exp(lv)
    q = csm.cop_eval('frank', th, u, v); s = P['st']
    out = np.zeros(P['n'])
    for d in range(K + 1):
        idx = s == d + 1; out[idx] += xi[d, P['iv'][idx]] + eta[idx, d]
    isE, isC, isA = (s >= 1) & (s <= K), s == K + 1, s == 0
    out[isE] += lu[isE] + q['logC1'][isE]; out[isC] += lv[isC] + q['logC2'][isC]; out[isA] += q['logC'][isA]
    return out

def scores_i(X, xi, th):
    a = np.exp(xi); eta = P['Zs'] @ X.T; H = np.exp(eta) * (P['E'] @ a.T)
    u, v = np.exp(-H[:, :K].sum(1)), np.exp(-H[:, K])
    q = csm.cop_eval('frank', th, u, v); s = P['st']
    isE, isC = (s >= 1) & (s <= K), s == K + 1
    rT = np.where(isE, q['tE'], np.where(isC, q['tD'], q['tA']))
    rC = np.where(isE, q['cE'], np.where(isC, q['cD'], q['cA']))
    r = np.column_stack([rT, rT, rC])
    Dl = np.column_stack([(s == d + 1).astype(float) for d in range(K + 1)])
    gX = ((Dl - r * H)[:, :, None] * P['Zs'][:, None, :]).reshape(P['n'], -1)        # n x 3p
    ind = np.zeros((P['n'], K + 1, M))
    for d in range(K + 1):
        idx = np.where(s == d + 1)[0]; ind[idx, d, P['iv'][idx]] = 1
    gxi = ind - a[None] * (r * np.exp(eta))[:, :, None] * P['E'][:, None, :]
    return np.hstack([gX, gxi.reshape(P['n'], -1)])

S = scores_i(X0, xi0, th0)
print('max |mean score| at fitted point:', np.abs(S.sum(0)).max())
h = 1e-4
st_ = (ell_i(X0, xi0, th0 + h) - ell_i(X0, xi0, th0 - h)) / (2 * h)
Pi_opg = np.linalg.solve(S.T @ S, S.T @ st_)
# Hessian slope: -H_pp^{-1} H_ptheta
k = S.shape[1]; vec = np.concatenate([X0.ravel(), xi0.ravel()])
def unpack(v): return v[:3 * p].reshape(3, p), v[3 * p:].reshape(3, M)
Hpp = np.zeros((k, k))
for j in range(k):
    e = np.zeros(k); e[j] = 1e-5
    Hpp[:, j] = (scores_i(*unpack(vec + e), th0).sum(0) - scores_i(*unpack(vec - e), th0).sum(0)) / 2e-5
Hpt = (scores_i(X0, xi0, th0 + h).sum(0) - scores_i(X0, xi0, th0 - h).sum(0)) / (2 * h)
Pi_hess = -np.linalg.solve((Hpp + Hpp.T) / 2, Hpt)
# convert theta-slope to tau-slope and to raw coefficient scale
dth = (csm.tau2theta('frank', tau0 + 1e-3) - csm.tau2theta('frank', tau0 - 1e-3)) / 2e-3
sd = P['sd']
toB = lambda Pi: (Pi[:3 * p].reshape(3, p) / sd[None, :]) * dth
fd = (np.array(R[round(tau0 + 0.05, 2)]['B']) - np.array(R[round(tau0 - 0.05, 2)]['B'])) / 0.1
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print('slope dB/dtau: OPG projection\n', toB(Pi_opg)); print('Hessian\n', toB(Pi_hess)); print('finite difference of refits\n', fd)
Bt = np.array(D['Btrue'])
print('corr OPG vs FD over inactive coords:', np.corrcoef(toB(Pi_opg)[Bt == 0], fd[Bt == 0])[0, 1],
      ' Hess vs FD:', np.corrcoef(toB(Pi_hess)[Bt == 0], fd[Bt == 0])[0, 1])
json.dump(dict(opg=toB(Pi_opg).tolist(), hess=toB(Pi_hess).tolist(), fd=fd.tolist()), open(f'firstorder_{signal}_{tau0}.json', 'w'))
