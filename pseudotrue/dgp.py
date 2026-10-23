import numpy as np, sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import csm
CUT = np.array([0, 1, 2, np.inf]); A1 = np.array([.12,.09,.07]); A2 = np.array([.09,.075,.06]); AC0 = np.array([.055,.075,.10])
FROZEN = {('moderate',0):(0.550843413027010,8.76565595085491),('moderate',0.2):(0.679796401819543,9.39301248954755),
          ('moderate',0.5):(0.919451899640470,10.10206933249560),('strong',0):(0.521141378895718,7.36516463793110),
          ('strong',0.2):(0.612753275222009,7.84234197037593),('strong',0.5):(0.770079873784685,8.39305286737204)}
def coefs(signal, p):
    B = np.zeros((3, p))
    if signal == 'moderate':
        B[0,[0,1,4]] = [.5,-.4,.3]; B[1,[0,2,5]] = [-.35,.45,-.3]; B[2,[1,3,6]] = [.4,-.35,.3]
    else:
        B[0,[0,1,4]] = [.7,-.7,.6]; B[1,[0,2,5]] = [-.7,.7,-.6]; B[2,[1,3,6]] = [.7,-.7,.6]
    return B
def inv_cumhaz(h, R):
    c1 = R[:,0]; c2 = c1 + R[:,1]
    return np.where(h <= c1, h/R[:,0], np.where(h <= c2, 1 + (h-c1)/R[:,1], 2 + (h-c2)/R[:,2]))
def rfrank(th, n, rng):
    u = rng.random(n); w = rng.random(n)
    if th == 0: return u, w
    v = -np.log1p(w*np.expm1(-th)/(w + (1-w)*np.exp(-th*u)))/th
    return u, v
def gen(n, p, signal, tau, rng, delta=0.0):
    mC, a = FROZEN[(signal, tau)]
    L = np.linalg.cholesky(0.3**np.abs(np.subtract.outer(np.arange(p), np.arange(p))))
    Z = rng.standard_normal((n, p)) @ L.T
    B = coefs(signal, p)
    e1, e2, eC = np.exp(Z@B[0]), np.exp(Z@B[1]), np.exp(Z@B[2])
    u, v = rfrank(csm.tau2theta('frank', tau), n, rng)
    T = inv_cumhaz(-np.log(u), np.outer(e1, A1) + np.outer(e2, A2))
    m = np.searchsorted(CUT, T, side='right') - 1
    p1 = e1*A1[m]/(e1*A1[m] + e2*A2[m])
    J = np.where(rng.random(n) < p1, 1, 2)
    RC = np.outer(eC*np.exp(delta*(J == 1)), AC0)
    C = inv_cumhaz(-np.log(v)/mC, RC)
    Y = np.minimum(np.minimum(T, C), a)
    st = np.where((T <= C) & (T <= a), J, np.where(C <= a, 3, 0))
    return Y, st, Z, B
