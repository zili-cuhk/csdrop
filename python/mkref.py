import numpy as np, json
from csm import *
from dgp import gen, FROZEN
cuts=np.array([0,1,2,np.inf]); mC,a=FROZEN[('moderate',0.5)]
Z,Y,st=gen(400,10,'moderate','frank',0.5,mC,a,np.random.default_rng(11))
import os
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','matlab','csm','test')+os.sep
np.savetxt(out+'ref_data.csv',np.column_stack([Y,st,Z]),delimiter=',',fmt='%.17g',header='Y,status,'+','.join(f'Z{j+1}' for j in range(10)),comments='')
P=prep(Y,st,Z,cuts,2)
d=np.arange(1,4)[:,None]; j=np.arange(1,11)[None,:]
X=0.3*np.sin(d*j); xi=np.log(np.array([[.12,.09,.07],[.09,.075,.06],[.05,.07,.09]]))
ref={'lik':[]}
for fam,tau in [('frank',.5),('clayton',.5),('gumbel',.5),('frank',-.3),('indep',0)]:
    cop=(fam,tau2theta(fam,tau)); ell,gX,gxi,ok,R=loglik(P,cop,X,xi)
    ref['lik'].append(dict(family=fam,tau=tau,theta=cop[1],ell=ell,gX=gX.tolist(),gxi=gxi.tolist()))
res=gselo_select(Y,st,Z,cuts,2,'frank',0.5)
ref['select']=[dict(gE=ge,gP=r['gP'],theta=r['th'],S=r['S'].astype(int).tolist(),ell=r['ell'],B=to_raw(res['P'],r['X'],r['xi'])[0].tolist()) for ge,r in res['sel'].items()]
F=cif(res['P'],res['sel'][0.25]['X'],res['sel'][0.25]['xi'],Z[:5],np.array([0.5,1,3,6]))
ref['cif_first5']=F.tolist()
json.dump(ref,open(out+'ref_values.json','w'),indent=1)
print('theta',[r['theta'] for r in ref['lik']]); print([ (s['gE'],s['gP'],s['theta'],round(s['ell'],6)) for s in ref['select']])
