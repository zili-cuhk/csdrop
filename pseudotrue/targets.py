"""Population targets for Exp C: pseudo-true and restricted pseudo-true cohort CIFs on the tau grid."""
import numpy as np, sys, json
from dgp import *
signal, tau0 = sys.argv[1], float(sys.argv[2]); n, p = 40000, 10
rng = np.random.default_rng(21)
Y, st, Z, B = gen(n, p, signal, tau0, rng)
P = csm.prep(Y, st, Z, CUT, 2)
times = np.array([1., 2., 4., 6.]); grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
opt = dict(csm.OPT); opt['refit_max_iter'] = 4000; opt['refit_tol'] = 1e-6; opt['coef_bound'] = 10
X = np.zeros((3, p)); xi = csm.breslow_init(P, X); out = []
for t in sorted(grid, key=lambda t: abs(t - tau0)):
    cop = ('frank', csm.tau2theta('frank', t))
    f = csm.fit(P, cop, X, xi, dict(kind='none'), opt=opt, unpen=True)
    S = np.abs(f['X']) > 0.05
    fr = csm.fit(P, cop, f['X'], f['xi'], dict(kind='none'), mask=S, opt=opt, unpen=True)
    Ff = csm.cif(P, f['X'], f['xi'], Z, times).mean(1); Fr = csm.cif(P, fr['X'], fr['xi'], Z, times).mean(1)
    out.append(dict(tau=t, status=[f['status'], fr['status']], X=f['X'].tolist(), S=S.astype(int).tolist(), F_full=Ff.tolist(), F_restr=Fr.tolist()))
    X, xi = f['X'], f['xi']
    print(signal, tau0, t, f['status'], fr['status'], np.round(Fr, 4).tolist(), flush=True)
from scipy import stats
json.dump(dict(signal=signal, tau0=tau0, grid=grid, times=times.tolist(), res=sorted(out, key=lambda r: r['tau'])), open(f'targets_{signal}_{tau0}.json', 'w'))
