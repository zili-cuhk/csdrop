"""Larger-dimension cells P01-P36: comparison of model-space adjustments.

Usage: python3 analysis/agg_largep_gamma.py [root]
For copHCR (PROP_TRUE) and HDCR (PROP_T00) with EBIC_0.25 and EBIC_0.5, and for
Lasso (CS_LASSO) with BIC and EBIC_0.5, computes the exact joint-support
recovery, the false positives, the MSE (sum over margins of (b - b0)' Sigma
(b - b0)) and the integrated squared CIF error summed over the two causes.
Writes results/summary/largep/{crit_compare.csv, largep_crit_full.csv,
tab_largep_gamma.tex, tab_largep_cif2.tex}.
"""
import os, sys, glob
import numpy as np, pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else '.'
O = os.path.join(D, 'results', 'summary', 'largep'); os.makedirs(O, exist_ok=True)
S=pd.read_csv(f'{D}/scenarios.csv'); S=S[S.exp=='P'].set_index('sid')
def coefs(signal,p):
    B=np.zeros((3,p))
    if signal=='moderate': B[0,[0,1,4]]=[.5,-.4,.3];B[1,[0,2,5]]=[-.35,.45,-.3];B[2,[1,3,6]]=[.4,-.35,.3]
    else: B[0,[0,1,4]]=[.7,-.7,.6];B[1,[0,2,5]]=[-.7,.7,-.6];B[2,[1,3,6]]=[.7,-.7,.6]
    return B
M={'event1':0,'event2':1,'dropout':2}
rows=[]
for fit,sub,crits in [('PROP_TRUE','prop',['EBIC0.25','EBIC0.50']),('PROP_T00','prop',['EBIC0.25','EBIC0.50']),('CS_LASSO','comp',['bic','ebic0.5'])]:
    for f in glob.glob(f'{D}/results/{sub}/sel_{fit}_P*.csv'):
        X=pd.read_csv(f,dtype={'coef':str}); X=X[X.criterion.isin(crits)&(X.status=='ok')]
        for (sid,cr,rep),g in X.groupby(['sid','criterion','rep']):
            sc=S.loc[sid]; p=int(sc.p); B0=coefs(sc.signal,p); Sig=0.3**np.abs(np.subtract.outer(np.arange(p),np.arange(p)))
            mse=0
            for _,r in g.iterrows():
                b=np.array([float(v) for v in r.coef.split(';')]) if isinstance(r.coef,str) and r.coef else np.zeros(p)
                d=np.nan_to_num(b)-B0[M[r.margin]]; mse+=d@Sig@d
            rows.append(dict(sid=sid,fit=fit,crit=cr,rep=rep,mse=mse))
R=pd.DataFrame(rows).groupby(['sid','fit','crit']).mse.mean().reset_index()
C=[]
for fit,sub in [('PROP_TRUE','prop'),('PROP_T00','prop'),('CS_LASSO','comp')]:
    X=pd.concat([pd.read_csv(f) for f in glob.glob(f'{D}/results/{sub}/cif_{fit}_P*.csv')])
    X=X.groupby(['sid','criterion']).ise.mean().reset_index().rename(columns={'criterion':'crit'}); X['fit']=fit; C.append(X)
C=pd.concat(C)
T=R.merge(C,on=['sid','fit','crit'],how='left').merge(S[['n','p','signal','tau']],left_on='sid',right_index=True)
T=T[T.crit.isin(['EBIC0.25','EBIC0.50','bic','ebic0.5'])]
T.to_csv(os.path.join(O,'crit_compare.csv'),index=False)

act={'event1':{1,2,5},'event2':{1,3,6},'dropout':{2,4,7}}
def sup(s): return set() if not isinstance(s,str) or s=='' else set(int(x) for x in s.split(';'))
out=[]
for fit,sub,crits in [('PROP_TRUE','prop',['EBIC0.25','EBIC0.50']),('PROP_T00','prop',['EBIC0.25','EBIC0.50']),('CS_LASSO','comp',['bic','ebic0.5'])]:
    X=pd.concat([pd.read_csv(f,dtype={'support':str}) for f in glob.glob(f'{D}/results/{sub}/sel_{fit}_P*.csv')])
    X=X[X.criterion.isin(crits)&(X.status=='ok')]
    X['ok']=[sup(s)==act[m] for s,m in zip(X.support,X.margin)]
    X['fp']=[len(sup(s)-act[m]) for s,m in zip(X.support,X.margin)]
    g=X.groupby(['sid','criterion','rep']).agg(ok=('ok','all'),fp=('fp','sum')).reset_index()
    r=g.groupby(['sid','criterion']).agg(Pcorr=('ok','mean'),FP=('fp','mean'),reps=('rep','nunique')).reset_index(); r['fit']=fit; out.append(r)
P=pd.concat(out).rename(columns={'criterion':'crit'})
C=pd.read_csv(os.path.join(O,'crit_compare.csv'))
T=C.merge(P,on=['sid','fit','crit'])
T['ise']=T.ise  # mean per cause
print('reps min', T.reps.min())
# ISE summed over causes: recompute from cif files
I=[]
for fit,sub in [('PROP_TRUE','prop'),('PROP_T00','prop'),('CS_LASSO','comp')]:
    X=pd.concat([pd.read_csv(f) for f in glob.glob(f'{D}/results/{sub}/cif_{fit}_P*.csv')])
    X=X.groupby(['sid','criterion','rep']).ise.sum().groupby(['sid','criterion']).mean().reset_index().rename(columns={'criterion':'crit','ise':'ise2'}); X['fit']=fit; I.append(X)
T=T.merge(pd.concat(I),on=['sid','fit','crit'])
T.to_csv(f'{O}/largep_crit_full.csv',index=False)
lab={('PROP_TRUE','EBIC0.25'):'copHCR',('PROP_TRUE','EBIC0.50'):'copHCR',('PROP_T00','EBIC0.25'):'HDCR',('PROP_T00','EBIC0.50'):'HDCR',('CS_LASSO','bic'):'Lasso',('CS_LASSO','ebic0.5'):'Lasso'}
order=[('PROP_TRUE','EBIC0.25'),('PROP_TRUE','EBIC0.50'),('PROP_T00','EBIC0.25'),('PROP_T00','EBIC0.50'),('CS_LASSO','bic'),('CS_LASSO','ebic0.5')]
# Table: gamma comparison at tau0=0.5
t=T[np.isclose(T.tau,0.5)]
L=[r'\begin{tabular}{lrr'+'r'*6+'}',r'\toprule',
 r' & & & \multicolumn{2}{c}{copHCR} & \multicolumn{2}{c}{HDCR} & \multicolumn{2}{c}{Lasso}\\',
 r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
 r'Signal & $n$ & $p$ & EBIC$_{0.25}$ & EBIC$_{0.5}$ & EBIC$_{0.25}$ & EBIC$_{0.5}$ & BIC & EBIC$_{0.5}$\\',r'\midrule']
for metric,name,fmt in [('Pcorr','Pcorr','%.3f'),('FP','FP','%.2f'),('mse','MSE','%.3f'),('ise2',r'CIF error ($\times10^3$)','%.2f')]:
    L.append(r'\multicolumn{9}{l}{\textit{%s}}\\'%name)
    for sg in ('moderate','strong'):
        for i,(n,p) in enumerate([(n,p) for n in (400,600,1000) for p in (100,200)]):
            row=t[(t.signal==sg)&(t.n==n)&(t.p==p)]
            v=[]
            for f,c in order:
                r=row[(row.fit==f)&(row.crit==c)]
                x=r.iloc[0][metric]*(1e3 if metric=='ise2' else 1)
                v.append(fmt%x)
            L.append((sg.capitalize() if i==0 else '')+f' & {n} & {p} & '+' & '.join(v)+r'\\')
    if metric!='ise2': L.append(r'\addlinespace')
L+=[r'\bottomrule',r'\end{tabular}']
open(f'{O}/tab_largep_gamma.tex','w').write('\n'.join(L)+'\n')
# Table: CIF error by tau0 with EBIC0.25 and Lasso BIC
L=[r'\begin{tabular}{lrr'+'r'*9+'}',r'\toprule',
 r' & & & '+' & '.join(r'\multicolumn{3}{c}{$\tau_0=%g$}'%x for x in (0,0.2,0.5))+r'\\',
 r'\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}',
 r'Signal & $n$ & $p$ & '+' & '.join(['copHCR & HDCR & Lasso']*3)+r'\\',r'\midrule']
for sg in ('moderate','strong'):
    if sg=='strong': L.append(r'\addlinespace')
    for i,(n,p) in enumerate([(n,p) for n in (400,600,1000) for p in (100,200)]):
        v=[]
        for tau in (0,0.2,0.5):
            for f,c in [('PROP_TRUE','EBIC0.25'),('PROP_T00','EBIC0.25'),('CS_LASSO','bic')]:
                r=T[(T.signal==sg)&(T.n==n)&(T.p==p)&np.isclose(T.tau,tau)&(T.fit==f)&(T.crit==c)]
                v.append('%.2f'%(r.iloc[0].ise2*1e3))
        L.append((sg.capitalize() if i==0 else '')+f' & {n} & {p} & '+' & '.join(v)+r'\\')
L+=[r'\bottomrule',r'\end{tabular}']
open(f'{O}/tab_largep_cif2.tex','w').write('\n'.join(L)+'\n')
# summary numbers
a=t.pivot_table(index=['signal','n','p'],columns=['fit','crit'],values='mse')
print('MSE ratio lasso ebic0.5 / copHCR EBIC0.50:', (a[('CS_LASSO','ebic0.5')]/a[('PROP_TRUE','EBIC0.50')]).round(2).agg(['min','max']).tolist())
print('MSE ratio HDCR/copHCR 0.25:', (a[('PROP_T00','EBIC0.25')]/a[('PROP_TRUE','EBIC0.25')]).round(2).agg(['min','max']).tolist())
b=t.pivot_table(index=['signal','n','p'],columns=['fit','crit'],values='ise2')
print('ISE ratio HDCR/copHCR 0.25:', (b[('PROP_T00','EBIC0.25')]/b[('PROP_TRUE','EBIC0.25')]).round(2).agg(['min','max']).tolist())
print('ISE ratio LassoBIC/copHCR 0.25:', (b[('CS_LASSO','bic')]/b[('PROP_TRUE','EBIC0.25')]).round(2).agg(['min','max']).tolist())
print((b[('CS_LASSO','ebic0.5')]/b[('PROP_TRUE','EBIC0.50')]).round(2).to_string())
pc=t.pivot_table(index=['signal','n','p'],columns=['fit','crit'],values='Pcorr'); print(pc.round(3).to_string())
