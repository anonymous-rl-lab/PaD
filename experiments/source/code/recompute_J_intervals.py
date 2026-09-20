"""Paired descriptive AUROC intervals on existing Jonikas predictions; no refitting.
Uses Experiment 2's fixed-prediction endpoint-weight/pair resampling designs.
These intervals do not adjust for development selection or training uncertainty.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pad',type=Path,required=True)
    p.add_argument('--joint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    A=pd.read_csv(args.pad);B=pd.read_csv(args.joint)
    assert not A.duplicated(['a','b']).any() and not B.duplicated(['a','b']).any()
    X=A[['a','b','pair','label','PD_frozen']].merge(B[['a','b','label','F','d']],on=['a','b'],how='inner',validate='one_to_one',suffixes=('_P','_J'))
    assert len(X)==len(A)==len(B)==168
    assert (X.label_P==X.label_J).all()
    scores={(r.a,r.b):r.PD_frozen for r in X.itertuples()}
    X['pad_d']=[(r.PD_frozen-scores[(r.b,r.a)])/2 for r in X.itertuples()]
    y=X.label_P.to_numpy();pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    rows=[];D=[]
    for metric,pa,j in [('F_AUROC','PD_frozen','F'),('d_AUROC','pad_d','d')]:
        mats=[]
        for col in [pa,j]:
            s=X[col].to_numpy();diff=s[pos,None]-s[None,neg]
            mats.append((diff>0).astype(float)+.5*(diff==0))
        delta=mats[0]-mats[1];D.append((metric,delta))
        assert abs(delta.mean()-(roc_auc_score(y,X[pa])-roc_auc_score(y,X[j])))<1e-12
    genes=sorted(set(X.a)|set(X.b));gi={g:i for i,g in enumerate(genes)}
    ai=X.a.map(gi).to_numpy();bi=X.b.map(gi).to_numpy()
    pairs=list(dict.fromkeys(X.pair));pi={g:i for i,g in enumerate(pairs)};ii=X.pair.map(pi).to_numpy()
    for scheme,n_units,seed in [('gene_endpoint_sum',len(genes),20260923),('unordered_pair',len(pairs),20260924)]:
        rng=np.random.default_rng(seed);draws=rng.integers(0,n_units,size=(10000,n_units))
        counts=np.zeros((10000,n_units),dtype=np.int64)
        np.add.at(counts,(np.arange(10000)[:,None],draws),1)
        w=counts[:,ai]+counts[:,bi] if scheme=='gene_endpoint_sum' else counts[:,ii]
        wp=w[:,pos];wn=w[:,neg];den=wp.sum(1)*wn.sum(1);ok=den>0
        for metric,delta in D:
            samples=np.einsum('bi,ij,bj->b',wp,delta,wn,optimize=True)[ok]/den[ok]
            qs=np.quantile(samples,[.025,.5,.975])
            rows.append(dict(comparison='PaD minus J',metric=metric,resampling=scheme,point=float(delta.mean()),q025=float(qs[0]),q50=float(qs[1]),q975=float(qs[2]),n=len(samples),seed=seed))
    out=pd.DataFrame(rows);out.to_csv(args.out/'paired_J_intervals.csv',index=False)
    provenance={'status':'descriptive same-development-panel fixed-prediction reanalysis; no new training or independent validation','input_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [args.pad,args.joint]},'gene_resampling':'weight(pair)=sampled multiplicity(endpoint a)+sampled multiplicity(endpoint b); not a distribution-free dyadic confidence guarantee','pair_resampling':'84 unordered pairs; both directed rows receive same multiplicity','results':rows}
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(out.to_string(index=False))
if __name__=='__main__':main()
