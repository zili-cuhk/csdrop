import os as _os
import sys, json, numpy as np
sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import csm_app as csm, orig_alg
from prep import prepare
from run_app import ANALYSES, VARS, status_of, info_wald
res={}
for aid,(rule,fam,tau) in ANALYSES.items():
    d=prepare(rule); P=csm.prep(d['Y'],status_of(d),d['Z'],d['cut'],2); cop=(fam,csm.tau2theta(fam,tau))
    r=orig_alg.select(P,cop); o=dict(rule=rule,fam=fam,tau=tau,n=int(P['n']),sel={})
    for ge,row in r['sel'].items():
        S=row['S']; se,rc,g,Xp,xp=info_wald(P,cop,row['X'],row['xi'],S)
        B,_=csm.to_raw(P,Xp,xp); idx=np.argwhere(S); seraw=se/P['sd'][idx[:,1]]
        terms=[dict(margin=int(m),var=VARS[j],HR=float(np.exp(b)),lo=float(np.exp(b-1.96*s)),hi=float(np.exp(b+1.96*s))) for (m,j),b,s in zip(idx,B[S],seraw)]
        F=csm.cif(P,Xp,xp,d['Z'],np.array([12.,24.,36.])).mean(1)
        o['sel'][ge]=dict(gP=row['gP'],th=row['th'],qR=row['qR'],ell=float(csm.loglik(P,cop,Xp,xp,need_grad=False)[0]),crit=row['crit'],rcond=rc,grad=g,terms=terms,cif=F.round(4).tolist())
    res[aid]=o
    s=o['sel'][0.25]
    print(aid,fam,tau,rule,'n',o['n'],'| EBIC.25',[(t['margin'],t['var'],'%.2f (%.2f, %.2f)'%(t['HR'],t['lo'],t['hi'])) for t in s['terms']],'ell %.3f rcond %.4f grad %.1e'%(s['ell'],s['rcond'],s['grad']),'CIF',s['cif'],flush=True)
    for ge in (0,0.5):
        if [t['var']+str(t['margin']) for t in o['sel'][ge]['terms']]!=[t['var']+str(t['margin']) for t in s['terms']]:
            print('     gE',ge,'differs:',[(t['margin'],t['var'],round(t['HR'],2)) for t in o['sel'][ge]['terms']],'gP',o['sel'][ge]['gP'],'th',o['sel'][ge]['th'])
json.dump({str(k):v for k,v in res.items()},open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'out','orig_all.json'),'w'),indent=1,default=float)
