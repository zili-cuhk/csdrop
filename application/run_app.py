import os as _os
import sys, json, time, numpy as np
_H=_os.path.dirname(_os.path.abspath(__file__)); sys.path.insert(0,_os.path.join(_H,'..','python')); sys.path.insert(0,_H)
import csm_app as csm
from prep import prepare
DIRECT=False
VARS=['age_per_10_years','male','stage_II','stage_III_IV','node_positive','T3_T4']
ANALYSES={1:('exclude_unknown','frank',-0.3),2:('exclude_unknown','frank',0.0),3:('exclude_unknown','frank',0.2),
 4:('exclude_unknown','frank',0.5),5:('unknown_as_admin','frank',0.2),6:('unknown_as_dropout','frank',0.2),
 7:('exclude_unknown','clayton',0.2),8:('exclude_unknown','clayton',0.5),9:('exclude_unknown','gumbel',0.2),10:('exclude_unknown','gumbel',0.5)}
def status_of(d): return d['e1']*1+d['e2']*2+d['dr']*3
def info_wald(P,cop,X,xi,S):
    """observed information over free (X[S], xi) by central differences of the analytic score."""
    D1,M=xi.shape; idx=np.argwhere(S)
    def score(th):
        Xc=np.zeros_like(X); Xc[S]=th[:len(idx)]; xc=th[len(idx):].reshape(D1,M)
        _,gX,gxi,ok,_=csm.loglik(P,cop,Xc,xc); return np.concatenate([gX[S],gxi.ravel()])
    th0=np.concatenate([X[S],xi.ravel()]); k=len(th0)
    def hess(th):
        H=np.zeros((k,k)); h=3e-5
        for j in range(k):
            e=np.zeros(k); e[j]=h*max(1,abs(th[j])); H[:,j]=(score(th+e)-score(th-e))/(2*e[j])
        return (H+H.T)/2
    for it in range(20):                      # Newton polish of the refit to |score|<1e-8
        g=score(th0)
        if np.abs(g).max()<1e-8: break
        th0=th0-np.linalg.solve(hess(th0),g)
    I=-hess(th0); C=np.linalg.inv(I); ev=np.linalg.eigvalsh(I)
    Xp=np.zeros_like(X); Xp[S]=th0[:len(idx)]; xp=th0[len(idx):].reshape(D1,M)
    return np.sqrt(np.diag(C))[:len(idx)], 1/np.linalg.cond(I), np.abs(score(th0)).max(), Xp, xp
def run(aid):
    rule,fam,tau=ANALYSES[aid]; t0=time.time()
    d=prepare(rule); st=status_of(d)
    r=csm.gselo_select(d['Y'],st,d['Z'],d['cut'],2,fam,tau,direct_seeds=DIRECT)
    P=r['P']; cop=(fam,csm.tau2theta(fam,tau)); out=dict(aid=aid,rule=rule,fam=fam,tau=tau,n=int(P['n']),
        cut=d['cut'][1:3].tolist(),branches=r['branches'],ncand=r['ncand'],sel={})
    grid=np.array([12.,24.,36.])
    for ge,row in r['sel'].items():
        S=row['S']; se,rc,gmax,Xp,xp=info_wald(P,cop,row['X'],row['xi'],S)
        row['X'],row['xi']=Xp,xp; row['ell']=csm.loglik(P,cop,Xp,xp,need_grad=False)[0]
        B,_=csm.to_raw(P,row['X'],row['xi']); seraw=se/P['sd'][np.argwhere(S)[:,1]]
        terms=[]
        for (dd,j),b,s in zip(np.argwhere(S),B[S],seraw):
            terms.append(dict(margin=int(dd),var=VARS[j],beta=float(b),se=float(s),HR=float(np.exp(b)),lo=float(np.exp(b-1.96*s)),hi=float(np.exp(b+1.96*s))))
        F=csm.cif(P,row['X'],row['xi'],d['Z'],grid).mean(1)
        out['sel'][str(ge)]=dict(gP=row['gP'],th=row['th'],branch=row['branch'],Qpen=row['Qpen'],qR=row['qR'],ell=float(row['ell']),rcond=float(rc),grad=float(gmax),terms=terms,cif=F.tolist())
    out['sec']=time.time()-t0
    _os.makedirs(_os.path.join(_H,'out'),exist_ok=True); json.dump(out,open(_os.path.join(_H,'out',f'a{aid:02d}.json'),'w'),indent=1)
    return out
if __name__=='__main__':
    from multiprocessing import Pool
    ids=[int(a) for a in sys.argv[1:]] or list(ANALYSES)
    with Pool(min(len(ids),10)) as p:
        for o in p.imap_unordered(run,ids):
            print(o['aid'],o['fam'],o['tau'],o['rule'],'n',o['n'],'%.0fs'%o['sec'],flush=True)
            for ge,s in o['sel'].items(): print('   gE',ge,'gP',s['gP'],'th',s['th'],'ell %.3f'%s['ell'],[(t['margin'],t['var'],round(t['HR'],2),round(t['lo'],2),round(t['hi'],2)) for t in s['terms']],np.round(s['cif'],3).tolist(),flush=True)
