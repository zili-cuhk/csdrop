"""Original (v13) candidate structure used for the published simulation and TCGA results:
per gamma_P, an increasing and a decreasing continuation path that both start from zero
coefficients with crude occurrence/exposure baselines, plus a fixed start at the full-model
unpenalized joint MLE for each shape (3 x 3 = 9 direction-by-shape candidates); per shape the
admissible candidate with the smallest penalized objective is retained; L relaxation 1.2."""
import numpy as np
from math import lgamma
import csm_app as csm
OPT=dict(csm.OPT, relax=1.2)
def crude_init(P):
    xi=np.zeros((P['K']+1,P['M'])); expo=P['E'].sum(0)
    for d in range(P['K']+1): xi[d]=np.log(np.maximum(P['Dm'][d]/np.maximum(expo,1e-12),1e-3))
    return xi
def select(P, cop, gP_grid=(0,.025,.05,.075,.10,.125), V=(50,200,800), eta=0.05, gE=(0,.25,.5), opt=OPT, fixed=None):
    D1,p=P['K']+1,P['p']; Z0=np.zeros((D1,p)); nop=dict(kind='none')
    mle=csm.fit(P,cop,Z0,csm.breslow_init(P,Z0),nop,opt=opt,unpen=True)
    if mle['status']!='ok': return dict(status='initializer_failed')
    rows=[]; cache={}
    grid=[(g,th) for g in gP_grid for th in V] if fixed is None else [fixed]
    for gP in sorted(set(g for g,_ in grid)):
        rho=csm.rho_vec(P,gP); cands={th:[] for th in V}
        paths=[('inc',list(V)),('dec',list(V)[::-1])]
        for dname,order in paths:
            X,xi=Z0,crude_init(P)
            for th in order:
                f=csm.fit(P,cop,X,xi,dict(kind='gselo',rho=rho,theta=th),opt=opt)
                f['src']=dname; cands[th].append(f); X,xi=f['X'],f['xi']   # warm start even after failure
        for th in V:
            f=csm.fit(P,cop,mle['X'],mle['xi'],dict(kind='gselo',rho=rho,theta=th),opt=opt); f['src']='mle'; cands[th].append(f)
        for th in V:
            if (gP,th) not in grid: continue
            ok=[]
            for f in cands[th]:
                if f['status']!='ok': continue
                S=np.abs(f['X'])>eta; key=S.tobytes()
                if key not in cache: cache[key]=csm.fit(P,cop,f['X'],f['xi'],nop,mask=S,opt=opt,unpen=True)
                if cache[key]['status']=='ok': ok.append((f,S,cache[key]))
            if not ok: continue
            f,S,rf=min(ok,key=lambda t:t[0]['Q'])
            rows.append(dict(gP=gP,th=th,S=S,qR=int(S.sum()),ell=rf['ell'],X=rf['X'],xi=rf['xi'],branch=f['src'],Qpen=f['Q'],
                             nfail=sum(c['status']!='ok' for c in cands[th])))
    n,Pdim,M=P['n'],D1*p,P['M']; out={}
    for ge in gE:
        crit=np.array([-2*r['ell']+(r['qR']+D1*M)*np.log(n)+2*ge*(lgamma(Pdim+1)-lgamma(r['qR']+1)-lgamma(Pdim-r['qR']+1)) for r in rows])
        best=0
        for j in range(1,len(rows)):
            tol=1e-10*abs(crit[best])
            if crit[j]<crit[best]-tol or (abs(crit[j]-crit[best])<=tol and (rows[j]['qR'],rows[j]['gP'],rows[j]['th'],crit[j])<(rows[best]['qR'],rows[best]['gP'],rows[best]['th'],crit[best])): best=j
        out[ge]=dict(rows[best],crit=float(crit[best]))
    return dict(status='ok',sel=out,rows=rows,mle=mle)
