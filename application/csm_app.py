"""Python prototype of the v13 estimator. Structure mirrors the MATLAB files 1:1.
Margins are indexed d = 0..K-1 (events) and d = K (dropout).
Status codes: 1..K event, K+1 dropout, 0 administrative (internal); the data
files use 3 for dropout with K = 2, which is the same thing.
"""
import numpy as np
from math import lgamma

# ---------------------------------------------------------------- copulas
def cop_eval(fam, th, u, v):
    """Return dict with logC, logC1, logC2 and the six log-derivative ratios:
    tE = 1 + u C11/C1, tD = u C12/C2, tA = u C1/C,
    cE = v C12/C1,     cD = 1 + v C22/C2, cA = v C2/C."""
    lu, lv = np.log(u), np.log(v)
    if fam == 'indep' or (fam == 'frank' and abs(th) < 1e-10) or \
       (fam == 'clayton' and abs(th) < 1e-10) or (fam == 'gumbel' and abs(th - 1) < 1e-10):
        one = np.ones_like(u)
        return dict(logC=lu + lv, logC1=lv.copy(), logC2=lu.copy(),
                    tE=one, tD=one.copy(), tA=one.copy(), cE=one.copy(), cD=one.copy(), cA=one.copy())
    if fam == 'frank':
        eu, ev = np.exp(-th * u), np.exp(-th * v)
        a, b, c = np.expm1(-th * u), np.expm1(-th * v), np.expm1(-th)
        D = c + a * b
        C = -np.log1p(a * b / c) / th
        logC1 = -th * u + np.log(b / D)
        logC2 = -th * v + np.log(a / D)
        tE = 1 - th * u * (c - b) / D
        tD = -th * u * eu * c / (a * D)
        tA = u * eu * b / (D * C)
        cE = -th * v * ev * c / (b * D)
        cD = 1 - th * v * (c - a) / D
        cA = v * ev * a / (D * C)
        return dict(logC=np.log(C), logC1=logC1, logC2=logC2, tE=tE, tD=tD, tA=tA, cE=cE, cD=cD, cA=cA)
    if fam == 'clayton':
        U, V = np.exp(-th * lu), np.exp(-th * lv)          # u^-th, v^-th
        S = U + V - 1
        lS = np.log(S)
        return dict(logC=-lS / th,
                    logC1=-(th + 1) * lu - (1 / th + 1) * lS,
                    logC2=-(th + 1) * lv - (1 / th + 1) * lS,
                    tE=-th + (th + 1) * U / S, tD=(th + 1) * U / S, tA=U / S,
                    cE=(th + 1) * V / S, cD=-th + (th + 1) * V / S, cA=V / S)
    if fam == 'gumbel':
        x, y = -lu, -lv
        A = (x ** th + y ** th) ** (1 / th)
        lA = np.log(A)
        Ax = np.exp((1 - th) * lA + (th - 1) * np.log(x))
        Ay = np.exp((1 - th) * lA + (th - 1) * np.log(y))
        f = 1 + (th - 1) / A
        return dict(logC=-A,
                    logC1=-A + (1 - th) * lA + (th - 1) * np.log(x) + x,
                    logC2=-A + (1 - th) * lA + (th - 1) * np.log(y) + y,
                    tE=Ax * f - (th - 1) / x, tD=Ax * f, tA=Ax,
                    cE=Ay * f, cD=Ay * f - (th - 1) / y, cA=Ay)
    raise ValueError(fam)

def tau2theta(fam, tau):
    if fam == 'frank':
        if tau == 0: return 0.0
        from scipy import integrate, optimize
        d1 = lambda t: integrate.quad(lambda s: s / np.expm1(s) if s != 0 else 1.0, 0, t)[0] / t
        br = (1e-6, 200) if tau > 0 else (-200, -1e-6)
        return optimize.brentq(lambda t: 1 - 4 / t + 4 * d1(t) / t - tau, *br, xtol=1e-14)
    if fam == 'clayton': return 2 * tau / (1 - tau)
    if fam == 'gumbel': return 1 / (1 - tau)
    if fam == 'indep': return 0.0

# ---------------------------------------------------------------- data prep
def prep(Y, status, Z, cuts, K):
    """status: 1..K events, K+1 dropout, 0 admin. cuts includes 0 and possibly inf."""
    n, p = Z.shape
    M = len(cuts) - 1
    E = np.clip(np.minimum(Y[:, None], cuts[None, 1:]) - cuts[None, :-1], 0, None)   # n x M
    iv = np.searchsorted(cuts, Y, side='left') - 1      # Y in (k_{m-1}, k_m]
    iv = np.clip(iv, 0, M - 1)
    Dm = np.zeros((K + 1, M))
    for d in range(K + 1):
        np.add.at(Dm[d], iv[status == d + 1], 1)
    mu, sd = Z.mean(0), Z.std(0, ddof=1)
    return dict(n=n, p=p, K=K, M=M, Y=Y, st=status, E=E, iv=iv, Dm=Dm,
                Zs=(Z - mu) / sd, mu=mu, sd=sd, cuts=np.asarray(cuts, float),
                n0=np.array([(status == d + 1).sum() for d in range(K + 1)]))

def estimable(P, min_total=5):
    return bool(np.all(P['n0'] >= min_total) and np.all(P['Dm'] >= 1))

# ---------------------------------------------------------------- likelihood
def loglik(P, cop, X, xi, need_grad=True):
    """X: (K+1) x p coefficients (std scale); xi: (K+1) x M log heights.
    Returns ell, grad_X (of ell), grad_xi (of ell), ok."""
    K, n = P['K'], P['n']
    a = np.exp(xi)
    eta = P['Zs'] @ X.T                                   # n x (K+1)
    H0 = P['E'] @ a.T                                     # n x (K+1)
    H = np.exp(eta) * H0
    lu, lv = -H[:, :K].sum(1), -H[:, K]
    u, v = np.exp(lu), np.exp(lv)
    st = P['st']
    isE, isC, isA = (st >= 1) & (st <= K), st == K + 1, st == 0
    if np.any(u <= 0) or np.any(v <= 0):
        return -np.inf, None, None, False, None
    q = cop_eval(cop[0], cop[1], u, v)
    ell_i = np.zeros(n)
    for d in range(K + 1):
        idx = st == d + 1
        ell_i[idx] += xi[d, P['iv'][idx]] + eta[idx, d]
    ell_i[isE] += lu[isE] + q['logC1'][isE]
    ell_i[isC] += lv[isC] + q['logC2'][isC]
    ell_i[isA] += q['logC'][isA]
    ell = ell_i.sum()
    if not np.isfinite(ell):
        return -np.inf, None, None, False, None
    if not need_grad:
        return ell, None, None, True, None
    rT = np.where(isE, q['tE'], np.where(isC, q['tD'], q['tA']))
    rC = np.where(isE, q['cE'], np.where(isC, q['cD'], q['cA']))
    r = np.column_stack([np.repeat(rT[:, None], K, 1), rC])     # n x (K+1)
    Delta = np.column_stack([(st == d + 1).astype(float) for d in range(K + 1)])
    gX = (Delta - r * H).T @ P['Zs']                              # (K+1) x p   eq (14)-(15)
    R = (r * np.exp(eta)).T @ P['E']                              # (K+1) x M   R_dm
    gxi = P['Dm'] - a * R                                         # dell/dxi
    ok = np.all(np.isfinite(gX)) and np.all(np.isfinite(gxi))
    return ell, gX, gxi, ok, R

# ---------------------------------------------------------------- objective
def rho_vec(P, gP):
    Pdim = (P['K'] + 1) * P['p']
    return np.log(np.maximum(P['n0'], 2)) + 2 * gP * np.log(Pdim)

def objective(P, cop, X, xi, pen):
    """pen: dict(kind='none'|'gselo'|'ridge', rho, theta, lam, mask)"""
    ell, *_ = loglik(P, cop, X, xi, need_grad=False)
    return -2 * ell / P['n'] + penalty(P, X, pen), ell

def penalty(P, X, pen):
    if pen['kind'] == 'gselo':
        return (pen['rho'][:, None] * (1 - np.exp(-pen['theta'] * X ** 2))).sum() / P['n']
    if pen['kind'] == 'ridge':
        return pen['lam'] * (X ** 2).sum() / P['n']
    return 0.0

def pen_weight(P, X, pen):
    """w in (21): coefficient of x^2 in the quadratic majorizer of the penalty."""
    if pen['kind'] == 'gselo':
        return pen['rho'][:, None] * pen['theta'] * np.exp(-pen['theta'] * X ** 2) / P['n']
    if pen['kind'] == 'ridge':
        return np.full_like(X, pen['lam'] / P['n'])
    return np.zeros_like(X)

def pen_grad(P, X, pen):
    return 2 * pen_weight(P, X, pen) * X

def breslow_init(P, X):
    eta = P['Zs'] @ X.T
    den = (np.exp(eta)).T @ P['E']
    return np.log(np.maximum(P['Dm'], 0.5) / den)

# ---------------------------------------------------------------- solver
OPT = dict(max_iter=600, n_consec=3, tol_obj=1e-7, tol_coef=1e-5, tol_grad=5e-3,
           max_bt=50, mono=1e-10, L_floor=1e-6, coef_bound=4.0, a_lo=1e-5, a_hi=2.0,
           eps_num=1e-8, refit_max_iter=1000, refit_tol=5e-4)

def fit(P, cop, X0, xi0, pen, mask=None, opt=OPT, unpen=False):
    """Alternating MM/backtracking solver (eqs 20-24). mask: (K+1) x p boolean of free
    coefficients (others held at 0). unpen=True uses the refit stopping rule."""
    n = P['n']
    X = X0.copy() if mask is None else np.where(mask, X0, 0.0)
    if mask is None: mask = np.ones_like(X, bool)
    xi = xi0.copy()
    Q, ell = objective(P, cop, X, xi, pen)
    if not np.isfinite(Q):
        return dict(status='nonfinite_start', X=X, xi=xi, Q=Q, iters=0)
    L = 1.0
    consec = 0
    max_iter = opt['refit_max_iter'] if unpen else opt['max_iter']
    status = 'max_iter'
    for it in range(1, max_iter + 1):
        ell, gX, gxi, ok, R = loglik(P, cop, X, xi)
        if not ok: status = 'nonfinite_grad'; break
        gg = -2 * gX / n                                  # gradient of g
        gg = np.where(mask, gg, 0.0)
        g0 = -2 * ell / n
        w = pen_weight(P, X, pen)
        # ---- coefficient block: (22) with backtracking on L
        L = max(L / opt.get('relax', 2), opt['L_floor'])
        acc = False
        for bt in range(opt['max_bt'] + 1):
            Xn = np.where(mask, (L * X - gg) / (L + 2 * w), 0.0)
            Qn, elln = objective(P, cop, Xn, xi, pen)
            gn = -2 * elln / n
            dX = Xn - X
            if np.isfinite(Qn) and gn <= g0 + (gg * dX).sum() + L / 2 * (dX ** 2).sum() + opt['mono'] \
               and Qn <= Q + opt['mono']:
                acc = True; break
            L *= 2
        if not acc: status = 'coef_linesearch'; break
        Qx = Qn
        dXmax = np.abs(Xn - X).max(); X = Xn
        if np.abs(X).max() > opt['coef_bound']: status = 'coef_bound'; break
        # ---- baseline block: (24)
        ell, gX, gxi, ok, R = loglik(P, cop, X, xi)
        if not ok: status = 'nonfinite_grad'; break
        a = np.exp(xi)
        qd = -2 * gxi / n                                  # dg/dxi = 2(aR - D)/n
        hd = 2 * (P['Dm'] + np.abs(a * R)) / n + opt['eps_num']
        s = 1.0; acc = False
        for bt in range(opt['max_bt'] + 1):
            xin = xi - s * qd / hd
            Qn, _ = objective(P, cop, X, xin, pen)
            if np.isfinite(Qn) and Qn <= Qx + opt['mono']:
                acc = True; break
            s /= 2
        if not acc: status = 'base_linesearch'; break
        xi = xin
        a = np.exp(xi)
        if np.any(a < opt['a_lo']) or np.any(a > opt['a_hi']): status = 'base_bound'; break
        Qold, Q = Q, Qn
        # ---- convergence diagnostics at the new point
        ell, gX, gxi, ok, R = loglik(P, cop, X, xi)
        if not ok: status = 'nonfinite_grad'; break
        gcoef = np.abs(np.where(mask, -2 * gX / n + pen_grad(P, X, pen), 0)).max()
        gbase = np.abs(-2 * gxi / n).max()
        if unpen:
            if gcoef <= opt['refit_tol'] and gbase <= opt['refit_tol']:
                status = 'ok'; break
        else:
            c_ok = (abs(Q - Qold) / max(1, abs(Qold)) <= opt['tol_obj'] and
                    dXmax / max(1, np.abs(X).max()) <= opt['tol_coef'] and
                    gcoef <= opt['tol_grad'] and gbase <= opt['tol_grad'])
            consec = consec + 1 if c_ok else 0
            if consec >= opt['n_consec']:
                status = 'ok'; break
    ell, *_ = loglik(P, cop, X, xi, need_grad=False)
    return dict(status=status, X=X, xi=xi, Q=Q, ell=ell, iters=it)

# ---------------------------------------------------------------- Algorithm 1
def gselo_select(Y, status, Z, cuts, K, fam, tau, gP_grid=(0, .025, .05, .075, .10, .125),
                 V=(50, 200, 800), eta=0.05, gE=(0, .25, .5), ridge_lam=1.0, opt=OPT, direct_seeds=False):
    P = prep(Y, status, Z, cuts, K)
    if not estimable(P):
        return dict(status='nonestimable')
    cop = (fam, tau2theta(fam, tau))
    D1, p = K + 1, P['p']
    nopen = dict(kind='none')
    # seeds: zero, marginal Cox (= independence-copula joint MLE), weak ridge
    Z0 = np.zeros((D1, p))
    marg = fit(P, ('indep', 0), Z0, breslow_init(P, Z0), nopen, opt=opt, unpen=True)
    ridge = fit(P, ('indep', 0), Z0, breslow_init(P, Z0), dict(kind='ridge', lam=ridge_lam), opt=opt, unpen=True)
    seeds = [('zero', Z0), ('marginal', marg['X']), ('ridge', ridge['X'])]
    branches = []
    for name, Xs in seeds:
        f = fit(P, cop, Xs, breslow_init(P, Xs), nopen, opt=opt, unpen=True)
        if f['status'] == 'ok':
            branches.append((name, f))
    if direct_seeds:   # seeds also used directly as penalized starting values
        for name, Xs in seeds:
            branches.append((name + '_direct', dict(X=Xs, xi=breslow_init(P, Xs))))
    if not branches:
        return dict(status='no_admissible_branch')
    # penalized paths
    cand = {}
    for gP in gP_grid:
        rho = rho_vec(P, gP)
        for bname, bf in branches:
            for direction, order in (('inc', V), ('dec', V[::-1])):
                Xc, xic = bf['X'], bf['xi']
                for th in order:
                    f = fit(P, cop, Xc, xic, dict(kind='gselo', rho=rho, theta=th), opt=opt)
                    if f['status'] == 'ok':
                        key = (gP, th)
                        f['branch'] = bname + ':' + direction
                        if key not in cand or f['Q'] < cand[key]['Q']:
                            cand[key] = f
                        Xc, xic = f['X'], f['xi']
                    # failed fits do not update the warm start
    # threshold + refit (cached by support)
    rows, cache = [], {}
    for (gP, th) in sorted(cand):              # same order as the MATLAB (g, v) loops
        f = cand[(gP, th)]
        S = np.abs(f['X']) > eta
        key = S.tobytes()
        if key not in cache:
            cache[key] = fit(P, cop, f['X'], f['xi'], nopen, mask=S, opt=opt, unpen=True)
        rf = cache[key]
        if rf['status'] != 'ok':
            continue
        rows.append(dict(gP=gP, th=th, S=S, branch=f['branch'], Qpen=f['Q'], qR=int(S.sum()), ell=rf['ell'], X=rf['X'], xi=rf['xi']))
    if not rows:
        return dict(status='no_admissible_refit')
    n, Pdim, M = P['n'], D1 * p, P['M']
    out = {}
    for ge in gE:
        crit = np.array([-2 * r['ell'] + (r['qR'] + D1 * M) * np.log(n) +
                         2 * ge * (lgamma(Pdim + 1) - lgamma(r['qR'] + 1) - lgamma(Pdim - r['qR'] + 1))
                         for r in rows])
        best = 0
        for j in range(1, len(rows)):
            rb, rj = rows[best], rows[j]
            if crit[j] < crit[best] - 1e-10 * abs(crit[best]):
                best = j
            elif abs(crit[j] - crit[best]) <= 1e-10 * abs(crit[best]):
                if (rj['qR'], rj['gP'], rj['th'], crit[j]) < (rb['qR'], rb['gP'], rb['th'], crit[best]):
                    best = j
        out[ge] = rows[best]
    return dict(status='ok', P=P, sel=out, branches=[b for b, _ in branches], ncand=len(cand))

def to_raw(P, X, xi):
    B = X / P['sd'][None, :]
    a = np.exp(xi) * np.exp(-(B @ P['mu']))[:, None]
    return B, a

def cif(P, X, xi, Zraw, grid):
    """Exact intervalwise CIFs (Section 3.5) for raw covariates Zraw."""
    K, cuts = P['K'], P['cuts']
    Zs = (Zraw - P['mu']) / P['sd']
    a = np.exp(xi)
    lam = np.exp(Zs @ X[:K].T)                              # n x K
    n, G = Zs.shape[0], len(grid)
    F = np.zeros((K, n, G))
    Sprev = np.ones(n)
    for m in range(P['M']):
        lo, hi = cuts[m], cuts[m + 1]
        rk = lam * a[:K, m][None, :]
        r = rk.sum(1)
        d = np.clip(np.minimum(grid, hi) - lo, 0, None)
        inc = np.where(r[:, None] > 0, -np.expm1(-np.outer(r, d)) / np.where(r > 0, r, 1)[:, None],
                       np.outer(np.ones(n), d))
        for k in range(K):
            F[k] += (Sprev * rk[:, k])[:, None] * inc
        if np.isfinite(hi):
            Sprev = Sprev * np.exp(-r * (hi - lo))
    return F
