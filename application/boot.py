import os as _os
import sys,json,numpy as np
sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import csm_app as csm, orig_alg
from prep import prepare
from multiprocessing import Pool
d=prepare('exclude_unknown'); n=len(d['Y']); cut=d['cut']; st0=d['e1']+2*d['e2']+3*d['dr']
cop=('frank',csm.tau2theta('frank',0.2)); opt=orig_alg.OPT
REF=np.zeros((3,6),bool); REF[0,2]=REF[1,3]=True
def one(rep):
    u=np.random.RandomState(829200000+rep).random_sample(n); idx=np.ceil(n*u).astype(int)-1
    P=csm.prep(d['Y'][idx],st0[idx],d['Z'][idx],cut,2)
    if not (np.all(P['n0']>=5) and np.all(P['Dm']>=1)): return dict(rep=rep,status='empty_required_interval')
    Z0=np.zeros((3,6)); nop=dict(kind='none')
    mle=csm.fit(P,cop,Z0,csm.breslow_init(P,Z0),nop,opt=opt,unpen=True)
    pen=dict(kind='gselo',rho=csm.rho_vec(P,0),theta=200); cands=[]
    f=csm.fit(P,cop,Z0,orig_alg.crude_init(P),pen,opt=opt); f['src']='zero'; cands.append(f)
    if np.all(np.isfinite(mle['X'])):
        f=csm.fit(P,cop,mle['X'],mle['xi'],pen,opt=opt); f['src']='mle'; cands.append(f)
    ok=[c for c in cands if c['status']=='ok']
    if not ok: return dict(rep=rep,status='no_admissible',mle=mle['status'])
    b=min(ok,key=lambda c:c['Q']); S=np.abs(b['X'])>0.05
    rf=csm.fit(P,cop,b['X'],b['xi'],nop,mask=S,opt=opt,unpen=True)
    if rf['status']!='ok': return dict(rep=rep,status='refit_failed',mle=mle['status'])
    F=csm.cif(P,rf['X'],rf['xi'],d['Z'],np.array([12.,24.,36.])).mean(1)
    return dict(rep=rep,status='ok',mle=mle['status'],src=b['src'],S=S.astype(int).tolist(),cif=F.tolist())
if __name__=='__main__':
    with Pool(2) as p: R=p.map(one,range(1,201))
    json.dump(R,open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'out','boot.json'),'w'))
    ok=[r for r in R if r['status']=='ok']; print('ok',len(ok),'other',[(r['rep'],r['status']) for r in R if r['status']!='ok'])
    for lab,sub in [('all',ok),('qualified',[r for r in ok if r['mle']=='ok']),('guard',[r for r in ok if r['mle']!='ok'])]:
        if not sub: continue
        S=np.array([r['S'] for r in sub]).astype(bool)
        m=[np.mean([np.array_equal(s[k],REF[k]) for s in S]) for k in range(3)]
        j=np.mean([np.array_equal(s,REF) for s in S])
        print(lab,len(sub),'margin exact',np.round(m,3),'joint exact %.3f (%d)'%(j,round(j*len(sub))),'mean sizes',S.sum(2).mean(0).round(3))
    C=np.array([r['cif'] for r in ok]); print('CIF mean',C.mean(0).round(3).tolist()); print('2.5%',np.percentile(C,2.5,axis=0).round(3).tolist()); print('97.5%',np.percentile(C,97.5,axis=0).round(3).tolist())
