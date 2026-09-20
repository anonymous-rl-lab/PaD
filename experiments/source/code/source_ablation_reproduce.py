#!/usr/bin/env python3
"""Compact reproduction of the Jonikas source ablation used by the PaD paper.

Retains complete P and selectively masks D coordinates while preserving the
original 23-D distance denominator. The learner is the same pure joint RBF J
control, h=0.5, five lambdas, endpoint-excluded nested CV and F-target paired
one-standard-error selection.
"""
from pathlib import Path
import json, os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np, pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data/jonikas'; OUT=Path(os.environ.get('PAD_OUT',ROOT/'rerun_results/source_ablation')); OUT.mkdir(parents=True,exist_ok=True)
G=pd.read_csv(DATA/'KEGG.csv'); Z=dict(np.load(DATA/'KEGG.npz')); Y=Z['y'].astype(float); N=len(Y)
LAM=np.array([.00003,.0003,.003,.03,.3],float)
S=[0,1,2,11,13,14]; M=[8,9,12]; H=[15]; V=[3,4,5,6,7,10,16,17]; B=[18,19,20,21,22]
GROUPS={'S':S,'M':M,'H':H,'V':V,'B':B}
pair_order=list(dict.fromkeys(G.pair)); pair_rows={p:np.flatnonzero(G.pair.to_numpy()==p) for p in pair_order}
allowed={}
for p,te in pair_rows.items():
    endpoints=set(G.a.iloc[te])|set(G.b.iloc[te]); allowed[p]=np.flatnonzero(~G.a.isin(endpoints)&~G.b.isin(endpoints))
reverse_idx=np.empty(N,int)
for p,ix in pair_rows.items():
    assert len(ix)==2; reverse_idx[ix[0]]=ix[1]; reverse_idx[ix[1]]=ix[0]
assert np.max(np.abs(Z['xdr']-Z['xd'][reverse_idx]))<1e-12

def rbf(a,b,width=.5):
    d=np.maximum((a*a).mean(1)[:,None]+(b*b).mean(1)[None,:]-2*a@b.T/a.shape[1],0.)
    return np.exp(-d/(2*width*width))
P_K=rbf(Z['xp'],Z['xp']); P_R=rbf(Z['xpr'],Z['xp'])

def fit_kernel(K,Kr,tr):
    tr=np.asarray(tr,int); yy=Y[tr]; n1=max(int(yy.sum()),1); n0=max(len(tr)-int(yy.sum()),1)
    w=np.where(yy>0,len(tr)/(2*n1),len(tr)/(2*n0)); sw=np.sqrt(w)
    Mat=K[np.ix_(tr,tr)]*sw[:,None]*sw[None,:]; ev,U=eigh(Mat,check_finite=False); ev=np.maximum(ev,0.)
    rhs=U.T@(sw*(2*yy-1)); alphas=np.column_stack([sw*(U@(rhs/(ev+lam*len(tr)))) for lam in LAM])
    return K[:,tr]@alphas,Kr[:,tr]@alphas

def objective(yy,s):
    v=(2*yy-1-s)**2; return float((v[yy>0].mean()+v[yy==0].mean())/2)

def paired_se(yy,score,best,pairs):
    pos=np.flatnonzero(yy>0); neg=np.flatnonzero(yy==0)
    def concordance(s):
        d=s[pos,None]-s[None,neg]; return (d>0).astype(float)+.5*(d==0)
    d=concordance(best)-concordance(score); mu=d.mean(); inf=np.zeros(len(yy)); inf[pos]=(d.mean(1)-mu)/len(pos); inf[neg]=(d.mean(0)-mu)/len(neg)
    _,idx=np.unique(pairs,return_inverse=True); vv=np.bincount(idx,weights=inf); nn=len(np.unique(pairs)); return float(np.sqrt(np.sum(vv*vv)*nn/max(nn-1,1)))

def nested_from_xd(xd):
    xdr=xd[reverse_idx]; J=P_K*rbf(xd,xd); Jr=P_R*rbf(xdr,xd); out=np.full(N,np.nan); rev=np.full(N,np.nan)
    for p in pair_order:
        tr=allowed[p]; te=pair_rows[p]; inner=np.full((5,N),np.nan); cache={}
        for ip,iv0 in pair_rows.items():
            val=np.intersect1d(tr,iv0)
            if not len(val): continue
            itr=tuple(np.intersect1d(tr,allowed[ip]))
            if itr not in cache: cache[itr]=fit_kernel(J,Jr,itr)
            ps,_=cache[itr]; inner[:,val]=ps[val,:].T
        prim=np.array([roc_auc_score(Y[tr],inner[j,tr]) for j in range(5)]); loss=np.array([objective(Y[tr],inner[j,tr]) for j in range(5)])
        best=min(range(5),key=lambda j:(-prim[j],loss[j],-LAM[j],j)); gap=prim[best]-prim
        se=np.array([paired_se(Y[tr],inner[j,tr],inner[best,tr],G.pair.to_numpy()[tr]) for j in range(5)]); elig=np.flatnonzero(gap<=se+1e-12)
        chosen=min(elig,key=lambda j:(-LAM[j],-prim[j],loss[j],j)); ps,prs=fit_kernel(J,Jr,tr); out[te]=ps[te,chosen]; rev[te]=prs[te,chosen]
    return out,rev

def make(names):
    xd=np.zeros_like(Z['xd']); coords=sorted({j for nm in names for j in GROUPS[nm]}); xd[:,coords]=Z['xd'][:,coords]; return xd

def metrics(s,r):
    d=(s-r)/2; u=(s+r)/2; pos=Y>0
    return dict(F_AUROC=float(roc_auc_score(Y,s)),d_AUROC=float(roc_auc_score(Y,d)),u_AUROC=float(roc_auc_score(Y,u)),known_direction_correct=int(np.sum(d[pos]>0)),known_direction_ties=int(np.sum(d[pos]==0)),n_known_direction=int(pos.sum()))

# Retain every reported source context, including the SMHV direction-count loss.
VARIANTS={
 'S':['S'], 'SM':['S','M'], 'SH':['S','H'], 'SMH':['S','M','H'],
 'SMHB':['S','M','H','B'], 'SMHV':['S','M','H','V'], 'FULL':['S','M','H','V','B'], 'SHVB':['S','H','V','B']
}
requested=os.environ.get('PAD_VARIANTS','').strip()
if requested:
    keep=[x.strip() for x in requested.split(',') if x.strip()]
    VARIANTS={k:v for k,v in VARIANTS.items() if k in keep}
rows=[]; pred=G[['a','b','pair','label']].copy()
for name,names in VARIANTS.items():
    s,r=nested_from_xd(make(names)); mm=metrics(s,r); rows.append(dict(variant=name,**mm)); pred[f'{name}_F']=s; pred[f'{name}_reverse']=r; pred[f'{name}_d']=(s-r)/2
    print(name,mm,flush=True)
pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False); pred.to_csv(OUT/'predictions.csv',index=False)
# Regression checks for branches present in the archived v1 ledger.
ref=pd.read_csv(ROOT/'reference/source_ablation_reference.csv'); checks={}
for new,old in [('S','S'),('FULL','SLB')]:
    if new in VARIANTS:
        checks[new]=float(max(np.max(np.abs(pred[f'{new}_F']-ref[f'{old}_F'])),np.max(np.abs(pred[f'{new}_reverse']-ref[f'{old}_reverse']))))
(OUT/'reproduction_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
if checks and max(checks.values())>1e-10: raise SystemExit('archived-source regression failed')
