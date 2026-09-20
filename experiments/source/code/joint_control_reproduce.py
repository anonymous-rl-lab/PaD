#!/usr/bin/env python3
"""Reproduce the pure-joint RBF (J) control on Costanzo and Jonikas.

This is a compact extraction of the frozen learner used in the PaD paper:
Gaussian product kernel, h=0.5, five ridge values, endpoint-excluded nested
selection, and paired one-standard-error eligibility. No model/search branch
outside the paper claim is included.
"""
from pathlib import Path
import json, os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('PAD_OUT',ROOT/'rerun_results/joint_control'))
OUT.mkdir(parents=True,exist_ok=True)
LAM=np.array([.00003,.0003,.003,.03,.3],float)


def rbf(a,b,width=.5):
    d=np.maximum((a*a).mean(1)[:,None]+(b*b).mean(1)[None,:]-2*a@b.T/a.shape[1],0.)
    return np.exp(-d/(2*width*width))


def objective(y,s,directional):
    v=(2*y-1-s)**2
    if directional or len(set(y))<2:
        return float(v.mean())
    return float((v[y>0].mean()+v[y==0].mean())/2)


def paired_se(y,score,best,pairs,directional):
    if directional:
        delta=((best>=0)==(y>0)).astype(float)-((score>=0)==(y>0)).astype(float)
        inf=(delta-delta.mean())/len(y)
    else:
        pos=np.flatnonzero(y>0); neg=np.flatnonzero(y==0)
        def concordance(s):
            d=s[pos,None]-s[None,neg]
            return (d>0).astype(float)+.5*(d==0)
        d=concordance(best)-concordance(score); mu=d.mean(); inf=np.zeros(len(y))
        inf[pos]=(d.mean(1)-mu)/len(pos); inf[neg]=(d.mean(0)-mu)/len(neg)
    _,idx=np.unique(pairs,return_inverse=True); vv=np.bincount(idx,weights=inf); n=len(np.unique(pairs))
    return float(np.sqrt(np.sum(vv*vv)*n/max(n-1,1)))


def select_inner(y,s,pairs,directional):
    if directional:
        primary=np.array([np.mean((v>=0)==(y>0)) for v in s])
    else:
        primary=np.array([roc_auc_score(y,v) for v in s])
    loss=np.array([objective(y,v,directional) for v in s])
    best=min(range(5),key=lambda j:(-primary[j],loss[j],-LAM[j],j))
    gap=primary[best]-primary
    se=np.array([paired_se(y,v,s[best],pairs,directional) for v in s])
    elig=np.flatnonzero(gap<=se+1e-12)
    chosen=min(elig,key=lambda j:(-LAM[j],-primary[j],loss[j],j))
    return chosen,best,primary,loss,se,elig


def fit(K,Kr,y,tr,directional):
    tr=np.asarray(tr,int); yy=y[tr]
    if directional:
        K0=(K-Kr)/2; Kr0=-K0; w=np.ones(len(tr))
    else:
        K0=K; Kr0=Kr
        n1=max(int(yy.sum()),1); n0=max(len(tr)-int(yy.sum()),1)
        w=np.where(yy>0,len(tr)/(2*n1),len(tr)/(2*n0))
    sw=np.sqrt(w); M=K0[np.ix_(tr,tr)]*sw[:,None]*sw[None,:]
    ev,U=eigh(M,check_finite=False); ev=np.maximum(ev,0.)
    rhs=U.T@(sw*(2*yy-1))
    alphas=np.column_stack([sw*(U@(rhs/(ev+lam*len(tr)))) for lam in LAM])
    return K0[:,tr]@alphas, Kr0[:,tr]@alphas


def run(name):
    stem='Costanzo' if name=='costanzo' else 'KEGG'
    data_dir=ROOT/'data'/name
    g=pd.read_csv(data_dir/f'{stem}.csv'); z=dict(np.load(data_dir/f'{stem}.npz'))
    y=z['y'].astype(float); directional=bool(z['directional'])
    K=rbf(z['xp'],z['xp'])*rbf(z['xd'],z['xd'])
    Kr=rbf(z['xpr'],z['xp'])*rbf(z['xdr'],z['xd'])
    n=len(y)
    if directional:
        groups={str(f):np.flatnonzero(g.fold.to_numpy()==f) for f in sorted(set(g.fold))}
    else:
        groups={p:np.flatnonzero(g.pair.to_numpy()==p) for p in dict.fromkeys(g.pair)}
    allowed={}
    for key,te in groups.items():
        endpoints=set(g.a.iloc[te])|set(g.b.iloc[te])
        allowed[key]=np.flatnonzero(~g.a.isin(endpoints)&~g.b.isin(endpoints))
    score=np.full(n,np.nan); rev=np.full(n,np.nan); choices=[]
    for key,te in groups.items():
        tr=allowed[key]; inner=np.full((5,n),np.nan); cache={}
        for ikey,iv in groups.items():
            val=np.intersect1d(tr,iv)
            if not len(val): continue
            itr=tuple(np.intersect1d(tr,allowed[ikey]))
            if itr not in cache: cache[itr]=fit(K,Kr,y,itr,directional)
            ps,_=cache[itr]; inner[:,val]=ps[val,:].T
        assert np.isfinite(inner[:,tr]).all()
        ch,best,primary,loss,se,elig=select_inner(y[tr],inner[:,tr],g.pair.to_numpy()[tr],directional)
        trkey=tuple(tr)
        if trkey not in cache: cache[trkey]=fit(K,Kr,y,tr,directional)
        ps,prs=cache[trkey]; score[te]=ps[te,ch]; rev[te]=prs[te,ch]
        choices.append(dict(group=str(key),lambda_=float(LAM[ch]),eligible=int(len(elig)),n_train=int(len(tr))))
    assert np.isfinite(score).all() and np.isfinite(rev).all()
    d=(score-rev)/2; u=(score+rev)/2
    if directional:
        result=dict(dataset='Costanzo',n=n,correct=int(np.sum((d>=0)==(y>0))),accuracy=float(np.mean((d>=0)==(y>0))),d_AUROC=float(roc_auc_score(y,d)))
    else:
        result=dict(dataset='Jonikas',n=n,F_AUROC=float(roc_auc_score(y,score)),d_AUROC=float(roc_auc_score(y,d)),u_AUROC=float(roc_auc_score(y,u)),known_direction_correct=int(np.sum(d[y>0]>0)),n_known_direction=int(np.sum(y>0)),average_precision=float(average_precision_score(y,score)))
    tab=g[['a','b','pair']].copy(); tab['label']=y; tab['F']=score; tab['reverse']=rev; tab['d']=d; tab['u']=u
    tab.to_csv(OUT/f'{name}_J_predictions.csv',index=False)
    (OUT/f'{name}_J_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    (OUT/f'{name}_J_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    a=run('costanzo'); b=run('jonikas')
    expected={'costanzo_correct':607,'jonikas_F':0.8756073858114674,'jonikas_d':0.8513119533527697}
    checks={'costanzo_607':a['correct']==expected['costanzo_correct'],
            'jonikas_F':abs(b['F_AUROC']-expected['jonikas_F'])<1e-12,
            'jonikas_d':abs(b['d_AUROC']-expected['jonikas_d'])<1e-12}
    (OUT/'reproduction_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    if not all(checks.values()): raise SystemExit('reproduction check failed: '+repr(checks))
