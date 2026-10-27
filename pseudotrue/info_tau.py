"""Curvature of the profile log-likelihood in the association (Web Table tab:info,
Proposition 1 of the main text).

Usage:
  python3 pseudotrue/info_tau.py fit <signal> <n> <Mfac> <mode> [outdir]
      signal  moderate | strong
      n       sample size (40000 in the paper)
      Mfac    each true baseline interval [0,1), [1,2), [2,inf) is split into
              Mfac pieces at quantiles of the observed event times, so the
              partition has M = 3 * Mfac intervals and contains the true one
      mode    cov    the ten covariates of the baseline design (p = 10)
              nocov  the covariates replaced by one independent N(0, 1) covariate
      Writes <outdir>/lb_<mode>_<signal>_<n>_M<M>.json.
  python3 pseudotrue/info_tau.py table [outdir]
      Collects the json files into the LaTeX body of Web Table tab:info.

The paper used n = 40000, Mfac in {1, 2, 4, 8}, both signals and both modes
(16 fits, a few minutes each). The model is fitted without penalization by
L-BFGS at tau in {0.30, 0.35, ..., 0.70} for the Frank copula with tau0 = 0.5,
and a quadratic in tau is fitted to the maximized log-likelihood per observation
over tau in [0.40, 0.60] (curv5, argmax5) and over the full range (curv9, argmax9).
The curvature at the maximizer estimates the efficient information for tau under
correct specification (mode cov).
"""
import os, sys, json, glob
import numpy as np
from scipy.optimize import minimize

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'python'))
sys.path.insert(0, here)                    # pseudotrue/dgp.py before python/dgp.py
from dgp import gen
import csm

TAUS = [0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65, 0.3, 0.7]


def fit(signal, n, Mfac, mode, outdir):
    rng = np.random.default_rng(11)
    Y, st, Z, _ = gen(n, 10, signal, 0.5, rng)
    if mode == 'nocov':
        Z = rng.standard_normal((n, 1))
    p = Z.shape[1]
    base = [0, 1, 2]; ymax = Y.max(); cuts = [0.0]
    for a, b in zip(base, base[1:] + [ymax]):
        yy = np.sort(Y[(Y > a) & (Y <= b) & (st > 0)])      # event times only
        for q in range(1, Mfac):
            cuts.append(float(np.quantile(yy, q / Mfac)))
        if b < ymax:
            cuts.append(float(b))
    cuts = np.array(sorted(set(cuts)) + [np.inf]); M = len(cuts) - 1
    P = csm.prep(Y, st, Z, cuts, 2)

    def negll(w, cop):
        X = w[:3 * p].reshape(3, p); xi = w[3 * p:].reshape(3, M)
        ell, gX, gxi, ok, _ = csm.loglik(P, cop, X, xi)
        if not ok:
            return 1e300, np.zeros_like(w)
        return -ell / n, -np.concatenate([gX.ravel(), gxi.ravel()]) / n

    X0 = np.zeros((3, p)); xi0 = csm.breslow_init(P, X0)
    w = np.concatenate([X0.ravel(), xi0.ravel()])
    out = {}
    for t in TAUS:
        cop = ('frank', csm.tau2theta('frank', t))
        r = minimize(negll, w, args=(cop,), jac=True, method='L-BFGS-B',
                     options=dict(maxiter=20000, gtol=1e-10, ftol=1e-15, maxcor=50))
        if t == 0.5:
            w = r.x                                         # warm start for the other tau
        out[t] = dict(ell=-r.fun, ok=bool(r.success), nit=int(r.nit), g=float(np.abs(r.jac).max()))
    tt = np.array(sorted(out)); ee = np.array([out[t]['ell'] for t in tt])
    m5 = (tt >= 0.4) & (tt <= 0.6)
    c5 = np.polyfit(tt[m5] - 0.5, ee[m5], 2); c9 = np.polyfit(tt - 0.5, ee, 2)
    res = dict(signal=signal, n=n, M=M, mode=mode,
               curv5=float(-2 * c5[0]), argmax5=float(0.5 - c5[1] / (2 * c5[0])),
               curv9=float(-2 * c9[0]), argmax9=float(0.5 - c9[1] / (2 * c9[0])),
               fits={str(k): v for k, v in out.items()})
    os.makedirs(outdir, exist_ok=True)
    json.dump(res, open(os.path.join(outdir, f'lb_{mode}_{signal}_{n}_M{M}.json'), 'w'))
    print(json.dumps({k: v for k, v in res.items() if k != 'fits'}))


def table(outdir):
    R = [json.load(open(f)) for f in glob.glob(os.path.join(outdir, 'lb_*.json'))]
    get = lambda sg, mode, M: next((r for r in R if r['signal'] == sg and r['mode'] == mode
                                    and r['M'] == M), None)
    lines = []
    for sg in ('strong', 'moderate'):
        Ms = sorted({r['M'] for r in R if r['signal'] == sg})
        for i, M in enumerate(Ms):
            a, b = get(sg, 'cov', M), get(sg, 'nocov', M)
            ca = f"{a['curv5']:.3f} & {a['argmax5']:.2f}" if a else '-- & --'
            cb = f"{b['curv5']:.4f} & {b['argmax5']:.2f}" if b else '-- & --'
            lines.append(f"{sg.capitalize() if i == 0 else ''} & {M} & {ca} & {cb}\\\\")
        lines.append(r'\addlinespace')
    print('\n'.join(lines[:-1]))
    print('\nimplied standard error of the profile estimate at n = 600 (cov, M = 24):')
    for sg in ('strong', 'moderate'):
        a = get(sg, 'cov', 24)
        if a:
            print(f"  {sg}: {1 / np.sqrt(600 * a['curv5']):.3f}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'fit':
        fit(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5],
            sys.argv[6] if len(sys.argv) > 6 else 'results/info')
    elif len(sys.argv) > 1 and sys.argv[1] == 'table':
        table(sys.argv[2] if len(sys.argv) > 2 else 'results/info')
    else:
        sys.exit(__doc__)
