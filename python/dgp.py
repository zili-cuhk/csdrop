import numpy as np
from scipy.optimize import brentq
from scipy import integrate
CUT=np.array([0.,1.,2.,np.inf])
A1=np.array([.12,.09,.07]); A2=np.array([.09,.075,.06]); AC=np.array([.055,.075,.10])
def coefs(signal,p):
    B=np.zeros((3,p))
    if signal=='moderate':
        B[0,[0,1,4]]=[.5,-.4,.3]; B[1,[0,2,5]]=[-.35,.45,-.3]; B[2,[1,3,6]]=[.4,-.35,.3]
    else:
        B[0,[0,1,4]]=[.7,-.7,.6]; B[1,[0,2,5]]=[-.7,.7,-.6]; B[2,[1,3,6]]=[.7,-.7,.6]
    return B
def debye1(x):
    return integrate.quad(lambda t: t/np.expm1(t) if t>0 else 1.0,0,x)[0]/x
def tau2theta(fam,tau):
    if fam=='frank':
        if tau==0: return 0.0
        return brentq(lambda th: 1-4/th+4*debye1(th)/th - tau, 1e-6, 200)
    if fam=='clayton': return 2*tau/(1-tau)
    if fam=='gumbel': return 1/(1-tau)
def rcopula(fam,th,n,rng):
    u=rng.random(n); w=rng.random(n)
    if fam=='frank':
        if th==0: return u,w
        a=np.exp(-th*u)

        # conditional inverse: v = -1/th * log(1 + w*(e^{-th}-1)/(w + (1-w) e^{-th u}))
        v=-np.log1p(w*np.expm1(-th)/(w+(1-w)*a))/th
        return u,v
    if fam=='clayton':
        if th==0: return u,w
        v=((w**(-th/(1+th))-1)*u**(-th)+1)**(-1/th); return u,v
    if fam=='gumbel':
        al=1/th
        if al==1: return u,w
        W=rng.random(n)*np.pi; E=rng.exponential(size=n)
        S=np.sin(al*W)/np.sin(W)**(1/al)*(np.sin((1-al)*W)/E)**((1-al)/al)
        E1=rng.exponential(size=n);E2=rng.exponential(size=n)
        return np.exp(-(E1/S)**al), np.exp(-(E2/S)**al)
def inv_cumhaz(h,r):  # r: n x 3 rates
    c1=r[:,0]; c2=c1+r[:,1]
    return np.where(h<=c1,h/r[:,0],np.where(h<=c2,1+(h-c1)/r[:,1],2+(h-c2)/r[:,2]))
def gen(n,p,signal,fam,tau,mC,a,rng,delta=0.0):
    idx=np.arange(p); S=0.3**np.abs(idx[:,None]-idx[None,:])
    Z=rng.standard_normal((n,p))@np.linalg.cholesky(S).T
    B=coefs(signal,p); e1=np.exp(Z@B[0]); e2=np.exp(Z@B[1]); eC=np.exp(Z@B[2])
    U,V=rcopula(fam,tau2theta(fam,tau),n,rng)
    rT=np.outer(e1,A1)+np.outer(e2,A2)
    T=inv_cumhaz(-np.log(U),rT)
    m=np.searchsorted(CUT,T,side='right')-1
    p1=e1*A1[m]/(e1*A1[m]+e2*A2[m]); J=np.where(rng.random(n)<p1,1,2)
    eCJ=eC*np.exp(delta*(J==1))
    C=inv_cumhaz(-np.log(V),np.outer(eCJ,mC*AC))
    Y=np.minimum(np.minimum(T,C),a)
    st=np.where(Y==T,J,np.where(Y==C,3,0))
    return Z,Y,st
FROZEN={('moderate',0.0):(0.550843413027010,8.76565595085491),('moderate',0.2):(0.679796401819543,9.39301248954755),
('moderate',0.5):(0.919451899640470,10.10206933249560),('strong',0.0):(0.521141378895718,7.36516463793110),
('strong',0.2):(0.612753275222009,7.84234197037593),('strong',0.5):(0.770079873784685,8.39305286737204)}
