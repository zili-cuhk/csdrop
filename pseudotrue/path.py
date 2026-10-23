import numpy as np, sys, json, time
from dgp import *
signal, tau0, n, p = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(11)
Y, st, Z, B = gen(n, p, signal, tau0, rng)
P = csm.prep(Y, st, Z, CUT, 2)
taus = [round(x, 2) for x in np.arange(-0.3, 0.651, 0.05)]
opt = dict(csm.OPT); opt['refit_max_iter'] = 4000; opt['refit_tol'] = 1e-6; opt['coef_bound'] = 10
start = {}; res = []
for t in sorted(taus, key=lambda t: (abs(t - tau0), t)):
    nb = min(start, key=lambda s: abs(s - t)) if start else None
    X0, xi0 = start[nb] if nb is not None else (np.zeros((3, p)), csm.breslow_init(P, np.zeros((3, p))))
    t0 = time.time()
    f = csm.fit(P, ('frank', csm.tau2theta('frank', t)), X0, xi0, dict(kind='none'), opt=opt, unpen=True)
    start[t] = (f['X'], f['xi'])
    Br, _ = csm.to_raw(P, f['X'], f['xi'])
    res.append(dict(tau=t, status=f['status'], ell=float(f['ell']), B=Br.tolist(), X=f['X'].tolist(), xi=f['xi'].tolist()))
    print(t, f['status'], f['iters'], round(time.time() - t0, 1), flush=True)
res.sort(key=lambda r: r['tau'])
json.dump(dict(signal=signal, tau0=tau0, n=n, p=p, Btrue=B.tolist(), res=res), open(f'path_{signal}_{tau0}_p{p}.json', 'w'))
