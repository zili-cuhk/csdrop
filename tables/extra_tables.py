"""Extra supplementary tables: Experiment A CIF errors by n, and estimation summary by scenario."""
import pandas as pd, numpy as np, glob, os
import pathlib as _pl
ROOT=_pl.Path(__file__).resolve().parents[1]
R=os.environ.get('CSDROP_RESULTS', str(ROOT/'results')); OUT=str(ROOT/'output'/'tables'); os.makedirs(OUT+'/extra', exist_ok=True)
sc=pd.read_csv(ROOT/'scenarios.csv')
A=sc[sc.sid.str.match(r'A\d\d')][['sid','n','p','signal','tau']].copy()
def load(pat, crit):
    d=pd.concat(pd.read_csv(f) for f in glob.glob(pat))
    d=d[d.criterion==crit]
    return d.groupby(['sid','rep']).ise.mean().reset_index()   # average over the two causes
M={}
M['Lasso']=load(f'{R}/comp/cif_CS_LASSO_A*.csv','bic')
M['Two-margin']=load(f'{R}/prop/cif_LIU2026_A*.csv','EBIC0.25')
M['WI']=load(f'{R}/prop/cif_PROP_T00_A*.csv','EBIC0.25')
P=[]
for t,code in ((0.0,'T00'),(0.2,'T02'),(0.5,'T05')):
    d=load(f'{R}/prop/cif_PROP_{code}_A*.csv','EBIC0.25')
    d=d[d.sid.isin(A[np.isclose(A.tau,t)].sid)]; P.append(d)
M['Proposed']=pd.concat(P)
rows=[]
for m,d in M.items():
    d=d.merge(A,on='sid'); d['method']=m; rows.append(d)
D=pd.concat(rows)
S=D.groupby(['method','sid','n','p','signal','tau']).ise.mean().reset_index()
T=S.groupby(['method','tau','n']).ise.mean().unstack('method')*1e3
cnt=D.groupby(['method','sid']).rep.nunique()
print(cnt.groupby('method').agg(['min','max']))
meths=['Lasso','Two-margin','WI','Proposed']
T=T[meths]; print(T.round(3)); print((T['WI']/T['Proposed']).round(3))
L=[r'\begin{tabular}{rr'+'r'*len(meths)+'r}',r'\toprule',
   r'$\tau_0$ & $n$ & '+' & '.join(meths)+r' & WI/Proposed\\',r'\midrule']
for i,t in enumerate((0.0,0.2,0.5)):
    if i: L.append(r'\addlinespace')
    for j,n in enumerate((400,600,1000)):
        r=T.loc[(t,n)]
        L.append((f'{t:g}' if j==0 else '')+f' & {n} & '+' & '.join(f'{r[m]:.2f}' for m in meths)+f' & {r["WI"]/r["Proposed"]:.2f}'+r'\\')
L+= [r'\bottomrule',r'\end{tabular}']
open(f'{OUT}/extra/tab_A_cif_n.tex','w').write('\n'.join(L)+'\n')
# estimation summary for the proposed method by scenario
E=pd.read_csv(f'{R}/summary/liu/liu_estimation.csv'); E=E[E.method=='Proposed'].copy()
E['ratio']=E.ASE_cond/E.SE_cond
g=E.groupby(['sid','n','p','signal','tau'])
Q=g.apply(lambda x: pd.Series({'nsel':int((x.SelRate>0.95).sum()),'bias':x.Bias_cond.abs().max(),
  'rmin':x.ratio.min(),'rmed':x.ratio.median(),'rmax':x.ratio.max(),'cmin':x.CP_cond.min(),'cmax':x.CP_cond.max(),
  'r95min':x[x.SelRate>0.95].ratio.min(),'r95max':x[x.SelRate>0.95].ratio.max(),
  'c95min':x[x.SelRate>0.95].CP_cond.min(),'c95max':x[x.SelRate>0.95].CP_cond.max()})).reset_index()
print('total sel>0.95', Q.nsel.sum(), 'of', len(E))
print('ratio among sel>.95', Q.r95min.min(), Q.r95max.max(), 'CP', Q.c95min.min(), Q.c95max.max(), 'median ratio all', E.ratio.median())
Q=Q.sort_values(['signal','tau','n','p'])
L=[r'\begin{longtable}{llrrrrrrr}',
 r'\caption{Experiment A. Estimation of the nine active coefficients by the proposed method in each scenario. The table gives the number of coefficients with selection rate above 0.95 (No.), the largest absolute bias, and the range of ASE/SE and of the CP of the 95\% Wald interval over the nine coefficients, all computed over the replications that select the coefficient (500 replications).}\label{tab:supp-A-est-all}\\',
 r'\toprule',r'Signal & $\tau_0$ & $n$ & $p$ & No. & Max $|$Bias$|$ & ASE/SE & CP\\',r'\midrule',r'\endfirsthead',
 r'\multicolumn{8}{c}{Web Table \ref{tab:supp-A-est-all} continued}\\',r'\toprule',r'Signal & $\tau_0$ & $n$ & $p$ & No. & Max $|$Bias$|$ & ASE/SE & CP\\',r'\midrule',r'\endhead',r'\bottomrule',r'\endlastfoot']
prev=None
for _,r in Q.iterrows():
    key=(r.signal,r.tau)
    if prev is not None and key!=prev: L.append(r'\addlinespace')
    lab=(r.signal.capitalize(), f'{r.tau:g}') if key!=prev else ('','')
    prev=key
    L.append(f'{lab[0]} & {lab[1]} & {int(r.n)} & {int(r.p)} & {int(r.nsel)} & {r.bias:.3f} & {r.rmin:.2f}--{r.rmax:.2f} & {r.cmin:.3f}--{r.cmax:.3f}'+r'\\')
L.append(r'\end{longtable}')
open(f'{OUT}/extra/tab_A_est_all.tex','w').write('\n'.join(L).replace('{llrrrrrrr}','{llrrrrrr}')+'\n')
