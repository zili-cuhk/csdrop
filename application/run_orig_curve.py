import os as _os
import sys, json, numpy as np
sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import csm_app as csm, orig_alg
from prep import prepare
from run_app import ANALYSES, VARS, status_of, info_wald
from multiprocessing import Pool
GRID=np.linspace(0,50,201)
def one(aid):
    rule,fam,tau=ANALYSES[aid]
    d=prepare(rule); P=csm.prep(d['Y'],status_of(d),d['Z'],d['cut'],2); cop=(fam,csm.tau2theta(fam,tau))
    r=orig_alg.select(P,cop); row=r['sel'][0.25]; S=row['S']
    se,rc,g,Xp,xp=info_wald(P,cop,row['X'],row['xi'],S)
    F=csm.cif(P,Xp,xp,d['Z'],GRID).mean(1)
    terms=[(int(m),VARS[j]) for (m,j) in np.argwhere(S)]
    out=dict(aid=aid,rule=rule,fam=fam,tau=tau,terms=terms,grid=GRID.tolist(),cif=np.asarray(F).tolist())
    json.dump(out,open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'out',f'orig_a{aid:02d}.json'),'w'))
    return aid,terms,np.round(np.asarray(F)[:,[48,96,144]],3).tolist()
if __name__=='__main__':
    with Pool(6) as p:
        for o in p.imap_unordered(one,[1,2,3,4,5,6]): print(o,flush=True)
