import numpy as np, sys, json, time
from dgp import *
signal, tau0, n = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
rng = np.random.default_rng(7)
p = 7
Y, st, Z, B = gen(n, p, signal, tau0, rng)
print('rates', [float(np.mean(st == k)) for k in (1, 2, 3, 0)], flush=True)
P = csm.prep(Y, st, Z, CUT, 2)
taus = [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
grid = np.array([0.5, 1, 2, 4, 6])
Zev = Z[:2000]
X = np.zeros((3, p)); xi = csm.breslow_init(P, X)
opt = dict(csm.OPT); opt['refit_max_iter'] = 5000; opt['refit_tol'] = 1e-6; opt['coef_bound'] = 10
res = []
# start from tau0, then walk outwards with warm starts
order = sorted(taus, key=lambda t: (abs(t - tau0), t))
start = {}
for t in order:
    nb = min(start, key=lambda s: abs(s - t)) if start else None
    X0, xi0 = (start[nb] if nb is not None else (X, xi))
    t0 = time.time()
    f = csm.fit(P, ('frank', csm.tau2theta('frank', t)), X0, xi0, dict(kind='none'), opt=opt, unpen=True)
    start[t] = (f['X'], f['xi'])
    F = csm.cif(P, f['X'], f['xi'], Zev, grid)          # K x nev x G
    B_raw, a_raw = csm.to_raw(P, f['X'], f['xi'])
    res.append(dict(tau=t, status=f['status'], ell=float(f['ell']), B=B_raw.tolist(), a=a_raw.tolist(),
                    Fbar=F.mean(1).tolist(), Fi=F[:, :300, :].tolist(), sec=time.time() - t0, it=f['iters']))
    print(t, f['status'], f['iters'], round(f['ell'], 3), np.round(F.mean(1), 4).tolist(), round(time.time() - t0, 1), flush=True)
res.sort(key=lambda r: r['tau'])
json.dump(dict(signal=signal, tau0=tau0, n=n, grid=grid.tolist(), B=B.tolist(), res=res), open(f'mono_{signal}_{tau0}_{n}.json', 'w'))
