import numpy as np, pandas as pd
import os
DATA=os.environ.get("CSDROP_TCGA_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
CLIN=os.path.join(DATA, "clinical.csv")
LEG=os.path.join(DATA, "LUAD_clinicalMatrix.tsv")
def prepare(rule):
    c=pd.read_csv(CLIN, dtype={'patient_id':str})
    l=pd.read_csv(LEG, sep='\t', dtype=str, usecols=['_PATIENT','lost_follow_up'])
    l['lf']=l.lost_follow_up.fillna('').str.strip().str.upper()
    lost=[]
    for p in c.patient_id:
        v=l.lf[l._PATIENT==p].values
        lost.append('YES' if (v=='YES').any() else ('NO' if (v=='NO').any() else 'UNKNOWN'))
    lost=np.array(lost)
    os_t=c.os_months.astype(float).values; pfs_t=c.pfs_months.astype(float).values
    ose=c.os_event.values==1; pfe=c.pfs_event.values==1
    pos=np.isfinite(os_t)&(os_t>0)&np.isfinite(pfs_t)&(pfs_t>=0)
    tol=1e-8
    prog=pfe&(~ose|(pfs_t<os_t-tol)); death=ose&~prog; noev=~(prog|death)
    drop=noev&(lost=='YES'); adm=noev&(lost=='NO'); unk=noev&(lost=='UNKNOWN')
    flow=dict(linked=len(c),nonpos=int((~pos).sum()),prog=int((prog&pos).sum()),death=int((death&pos).sum()),
              drop=int((drop&pos).sum()),adm=int((adm&pos).sum()),unk=int((unk&pos).sum()))
    inc=pos.copy()
    if rule=='exclude_unknown': inc&=~unk
    elif rule=='unknown_as_admin': adm=adm|unk
    elif rule=='unknown_as_dropout': drop=drop|unk
    s=lambda x: np.array(c[x].fillna('').astype(str).str.strip().str.upper().tolist(),dtype=str)
    age=c.age.astype(float).values; sex=s('sex'); st=s('stage'); n=s('n_stage'); t=s('t_stage')
    comp=np.isfinite(age)&np.isin(sex,['MALE','FEMALE'])&(np.char.str_len(st.astype(str))>0)&np.isin(n,['N0','N1','N2','N3'])&np.isin(t,['T1','T1A','T1B','T2','T2A','T2B','T3','T4'])
    st=st.astype(str)
    s2=np.char.startswith(st,'STAGE II')&~np.char.startswith(st,'STAGE III')
    s34=np.char.startswith(st,'STAGE III')|np.char.startswith(st,'STAGE IV')
    Z=np.column_stack([age/10,(sex=='MALE'),s2,s34,np.isin(n,['N1','N2','N3']),np.isin(t,['T3','T4'])]).astype(float)
    Y=np.maximum(os_t,pfs_t); Y[prog]=pfs_t[prog]; Y[death]=os_t[death]
    keep=inc&comp
    d=dict(Z=Z[keep],Y=Y[keep],e1=prog[keep].astype(int),e2=death[keep].astype(int),dr=drop[keep].astype(int),ad=adm[keep].astype(int),id=c.patient_id.values[keep])
    assert np.all(d['e1']+d['e2']+d['dr']+d['ad']==1)
    d['cut']=cutpoints(d); d['flow']=flow; d['n_incomplete']=int((inc&~comp).sum())
    return d
def mquantile(x,p):
    # MATLAB quantile: midpoint interpolation (p in [0.5/n,(n-0.5)/n])
    x=np.sort(x); n=len(x); pos=n*np.asarray(p)-0.5
    pos=np.clip(pos,0,n-1); lo=np.floor(pos).astype(int); hi=np.minimum(lo+1,n-1)
    return x[lo]+(pos-lo)*(x[hi]-x[lo])
def cutpoints(d):
    Y=d['Y']; occ=np.column_stack([d['e1'],d['e2'],d['dr']])
    pooled=Y[occ.sum(1)==1]
    cand=np.unique(mquantile(pooled,np.arange(1,10)/10)); target=mquantile(pooled,[1/3,2/3])
    best=(-np.inf,-np.inf); bp=None
    for a in range(len(cand)-1):
        for b in range(a+1,len(cand)):
            pair=cand[[a,b]]; itv=(Y[:,None]>pair).sum(1)
            cnt=np.array([[np.sum(occ[itv==j,k]) for j in range(3)] for k in range(3)])
            sc=(cnt.min(),-np.abs(pair-target).sum())
            if sc[0]>best[0] or (sc[0]==best[0] and sc[1]>best[1]): best=sc; bp=pair
    return np.array([0,bp[0],bp[1],np.inf])
if __name__=='__main__':
    for r in ['exclude_unknown','unknown_as_admin','unknown_as_dropout']:
        d=prepare(r); Y=d['Y']; itv=np.searchsorted(d['cut'][1:-1],Y,side='left')
        print(r,d['flow'],'n',len(Y),'counts',d['e1'].sum(),d['e2'].sum(),d['dr'].sum(),d['ad'].sum(),'incomplete',d['n_incomplete'],'cut',np.round(d['cut'][1:3],3))
        for k in ['e1','e2','dr']: print('  ',k,[int(d[k][itv==j].sum()) for j in range(3)])
        Zs=(d['Z']-d['Z'].mean(0))/d['Z'].std(0,ddof=1); print('  cond',np.round(np.linalg.cond(Zs),2),'VIF max',np.round(np.diag(np.linalg.inv(np.corrcoef(d['Z'].T))).max(),2))
