#!/usr/bin/env python3
"""Jonikas candidate-pair context-availability diagnostic.
All three availability levels recompute candidate inner scores under the same masks on
train and held-out records; no sample removal, new scale or P retraining.
"""
import hashlib,json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import engine as e
from paths import run_path

OUT=run_path('background')

def make_data(q,family):
    original=e.v4.Data('KEGG');g=original.g.copy();z={k:v.copy() for k,v in original.z.items()}
    pairs=list(dict.fromkeys(g.pair));salt=e.PROTOCOL['mask_salt']
    ordered=sorted(pairs,key=lambda p:hashlib.sha256((salt+p).encode()).hexdigest())
    retain=set(ordered[:int(len(ordered)*q)]);keep=g.pair.isin(retain).to_numpy()
    for key in ['xd','xdr']:z[key][~keep,18:23]=[0.,0.,0.,0.,.5]
    for key in ['p','odd','xp','xpr','y']:assert np.array_equal(z[key],original.z[key])
    assert np.array_equal(z['xd'][:,:18],original.z['xd'][:,:18])
    assert len(g)==168 and len(pairs)==84
    for p,gg in g.groupby('pair'):assert len(set(keep[gg.index]))==1
    d=e.Data(z,g,family);d.name='KEGG';d.groups=original.groups;d.allowed=original.allowed
    mask=pd.DataFrame(dict(pair=ordered,sha256=[hashlib.sha256((salt+p).encode()).hexdigest() for p in ordered],
                           context_permitted=[p in retain for p in ordered]))
    return d,mask

def save(d,q,family,s,sr,choices,ledger):
    prefix=OUT/f'q{int(q*100)}_{family}'
    pred=d.g[['a','b','pair']].copy();pred['target']=d.y;pred['score']=s;pred['reverse_score']=sr;pred['direction']=(s-sr)/2;pred['support']=(s+sr)/2
    pred.to_csv(str(prefix)+'_predictions.csv',index=False);e.dump(str(prefix)+'_choices.json',choices)
    result=dict(q=q,family=family,**e.v4.metric(d.y,s,sr,False),positive_direction_correct=int(np.sum((s-sr)[d.y>0]>0)),
                positive_direction_n=int(d.y.sum()),ledger=ledger)
    e.dump(str(prefix)+'_result.json',result);return result

def reduce(q,family):
    start=time.monotonic();d,mask=make_data(q,family);groups=list(d.groups);mask.to_csv(OUT/f'q{int(q*100)}_pair_mask.csv',index=False)
    inner=np.full((len(groups),35,d.n),np.nan);outer=np.full((35,d.n),np.nan);rev=outer.copy()
    eig=0;evals=0;cachehits=0;endpoint_checks=0
    for i,theta in enumerate(d.weights):
        cache={}
        for k,key in enumerate(groups):
            tr=d.allowed[key];te=d.groups[key];end=set(d.g.a.iloc[te])|set(d.g.b.iloc[te])
            assert not (set(d.g.a.iloc[tr])|set(d.g.b.iloc[tr]))&end;endpoint_checks+=1
            for ikey,iv in d.groups.items():
                val=np.intersect1d(tr,iv)
                if not len(val):continue
                itr=tuple(np.intersect1d(tr,d.allowed[ikey]));ee=set(d.g.a.iloc[val])|set(d.g.b.iloc[val])
                assert not (set(d.g.a.iloc[list(itr)])|set(d.g.b.iloc[list(itr)]))&ee;endpoint_checks+=1
                if itr not in cache:cache[itr]=e.v4.fit(d,itr,theta);eig+=1
                else:cachehits+=1
                evals+=5
                for j,l in enumerate(e.v4.LAMBDAS):inner[k,i*5+j,val]=cache[itr][l][0][val]
            tk=tuple(tr)
            if tk not in cache:cache[tk]=e.v4.fit(d,tk,theta);eig+=1
            else:cachehits+=1
            for j,l in enumerate(e.v4.LAMBDAS):outer[i*5+j,te]=cache[tk][l][0][te];rev[i*5+j,te]=cache[tk][l][1][te]
        print('BACKGROUND',q,family,'theta',i,'eigendecomp',eig,'seconds',round(time.monotonic()-start,1),flush=True)
    s=np.empty(d.n);sr=np.empty(d.n);choices=[]
    for k,key in enumerate(groups):
        tr=d.allowed[key];te=d.groups[key];assert np.isfinite(inner[k][:,tr]).all()
        chosen,best,primary,loss,gap,se,eligible=e.select(d,tr,inner[k][:,tr]);i,j=divmod(chosen,5);lam=e.v4.LAMBDAS[j]
        s[te]=outer[chosen,te];sr[te]=rev[chosen,te]
        choices.append(dict(group=key,weight_index=i,theta=d.weights[i],regularization=lam,n_train=len(tr),
                            selected_inner_auc=float(primary[chosen]),best_inner_auc=float(primary[best]),paired_SE=float(se[chosen])))
    result=save(d,q,family,s,sr,choices,dict(actual_eigendecompositions=eig,candidate_inner_evaluations=evals,
        cache_hits=cachehits,endpoint_checks=endpoint_checks,seconds=time.monotonic()-start,historical_inner_cache_reused=False))
    np.savez_compressed(OUT/f'q{int(q*100)}_{family}_cache.npz',inner=inner,outer=outer,reverse=rev)
    print('BACKGROUND RESULT',q,family,result['auroc'],flush=True);return result

def summarize(rows):
    pd.DataFrame([{k:v for k,v in r.items() if k!='ledger'} for r in rows]).to_csv(OUT/'summary.csv',index=False)
    changes=[];deletions=[]
    for q in [1.,.5,0.]:
        j=pd.read_csv(OUT/f'q{int(q*100)}_joint_predictions.csv');a=pd.read_csv(OUT/f'q{int(q*100)}_A_match_predictions.csv')
        pos=j.target>0;truth=j.target.to_numpy();fixed=int(((j.direction>0)&(a.direction<=0)&pos).sum());broken=int(((j.direction<=0)&(a.direction>0)&pos).sum())
        pi=np.flatnonzero(pos);ni=np.flatnonzero(~pos)
        def c(df):
            s=df.score.to_numpy();d=s[pi,None]-s[None,ni];return (d>0)+.5*(d==0)
        delta=c(j)-c(a)
        changes.append(dict(q=q,auc_joint=e.roc_auc_score(truth,j.score),auc_additive=e.roc_auc_score(truth,a.score),
                            gain=e.roc_auc_score(truth,j.score)-e.roc_auc_score(truth,a.score),rank_gain_sum=float(delta.sum()),
                            rank_improvements=int((delta>0).sum()),rank_losses=int((delta<0).sum()),direction_fixed=fixed,direction_broken=broken))
        for gene in sorted(set(j.a)|set(j.b)):
            ix=(j.a!=gene)&(j.b!=gene)
            if len(set(truth[ix]))<2:continue
            aj=e.roc_auc_score(truth[ix],j.score[ix]);aa=e.roc_auc_score(truth[ix],a.score[ix])
            deletions.append(dict(q=q,deleted_gene=gene,n=int(ix.sum()),auc_joint=aj,auc_additive=aa,gain=aj-aa))
    pd.DataFrame(changes).to_csv(OUT/'paired_changes.csv',index=False)
    pd.DataFrame(deletions).to_csv(OUT/'leave_gene_out_sensitivity.csv',index=False)
    return changes

