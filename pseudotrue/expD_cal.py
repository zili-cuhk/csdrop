import numpy as np, dgp, sys
from dgp import *
c = float(sys.argv[1]); orig = dgp.coefs
dgp.coefs = lambda s, p: (lambda B: (B.__setitem__((2, 2), c), B)[1])(orig(s, p))
rng = np.random.default_rng(41)
Y, st, Z, B = dgp.gen(40000, 10, 'strong', 0.5, rng)
print('c', c, 'rates e1 e2 drop admin', [round(float(np.mean(st == k)), 3) for k in (1, 2, 3, 0)], flush=True)
P = csm.prep(Y, st, Z, CUT, 2)
opt = dict(csm.OPT); opt['refit_max_iter'] = 4000; opt['refit_tol'] = 1e-6; opt['coef_bound'] = 10
X = np.zeros((3, 10)); xi = csm.breslow_init(P, X); out = []
for t in [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]:
    f = csm.fit(P, ('frank', csm.tau2theta('frank', t)), X, xi, dict(kind='none'), opt=opt, unpen=True); X, xi = f['X'], f['xi']
    leak = np.abs(np.where(B == 0, f['X'], 0)); i = np.unravel_index(leak.argmax(), leak.shape)
    print('%5.1f %s dropZ3 %.3f  max other inactive %.3f at %s' % (t, f['status'], f['X'][2, 2], leak.max(), i), flush=True)
