import numpy as np, dgp, json
from dgp import *
orig = dgp.coefs
def c2(s, p):
    B = orig(s, p); B[2, 2] = 0.15; return B
dgp.coefs = c2
rng = np.random.default_rng(51)
Y, st, Z, B = dgp.gen(40000, 10, 'strong', 0.5, rng)
P = csm.prep(Y, st, Z, CUT, 2)
opt = dict(csm.OPT); opt['refit_max_iter'] = 4000; opt['refit_tol'] = 1e-6; opt['coef_bound'] = 10
X = np.zeros((3, 10)); xi = csm.breslow_init(P, X); out = {}
for t in [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]:
    f = csm.fit(P, ('frank', csm.tau2theta('frank', t)), X, xi, dict(kind='none'), opt=opt, unpen=True); X, xi = f['X'], f['xi']
    out[t] = dict(status=f['status'], X=f['X'].tolist())
    S = np.abs(f['X']) > 0.05
    print(t, f['status'], 'S_eta:', {m: (np.where(S[m])[0] + 1).tolist() for m in range(3)}, flush=True)
json.dump(dict(Btrue=B.tolist(), res=out), open('expD_targets.json', 'w'))
