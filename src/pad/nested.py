"""Complete endpoint-excluded nested selection from prepared inputs.

No historical inner-score cache or saved model is required. The selector,
candidate order, loss and solver are the frozen implementation.
"""
import json,time
import numpy as np
import pandas as pd
import kernel as k
from controls import Control
from selection import select_from_inner
from paths import run_path

def train(dataset, family='PaD'):
    start=time.monotonic()
    d=k.Data(dataset) if family=='PaD' else Control(dataset,family)
    weights=k.WEIGHTS if family=='PaD' else d.weights
    groups=list(d.groups);nconfig=len(weights)*len(k.LAMBDAS)
    inner=np.full((len(groups),nconfig,d.n),np.nan)
    outer=np.full((nconfig,d.n),np.nan);reverse=outer.copy()
    eig=0;evals=0
    for i,theta in enumerate(weights):
        cache={}
        for m,key in enumerate(groups):
            tr=d.allowed[key];te=d.groups[key]
            end=set(d.g.a.iloc[te])|set(d.g.b.iloc[te])
            assert not end & (set(d.g.a.iloc[tr])|set(d.g.b.iloc[tr]))
            for ikey,iv in d.groups.items():
                val=np.intersect1d(tr,iv)
                if not len(val):continue
                itr=tuple(np.intersect1d(tr,d.allowed[ikey]))
                ends=set(d.g.a.iloc[val])|set(d.g.b.iloc[val])
                assert not ends & (set(d.g.a.iloc[list(itr)])|set(d.g.b.iloc[list(itr)]))
                if itr not in cache:cache[itr]=k.fit(d,itr,theta);eig+=1
                for j,lam in enumerate(k.LAMBDAS):inner[m,i*5+j,val]=cache[itr][lam][0][val]
                evals+=5
            trkey=tuple(tr)
            if trkey not in cache:cache[trkey]=k.fit(d,trkey,theta);eig+=1
            for j,lam in enumerate(k.LAMBDAS):
                outer[i*5+j,te]=cache[trkey][lam][0][te]
                reverse[i*5+j,te]=cache[trkey][lam][1][te]
        print(dataset,family,'kernel',i+1,'/',len(weights),flush=True)
    score=np.full(d.n,np.nan);rev=score.copy();choices=[]
    for m,key in enumerate(groups):
        tr=d.allowed[key];te=d.groups[key];assert np.isfinite(inner[m][:,tr]).all()
        chosen,best,primary,loss,gap,se,eligible=select_from_inner(d,tr,inner[m][:,tr])
        wi,li=divmod(chosen,5)
        score[te]=outer[chosen,te];rev[te]=reverse[chosen,te]
        choices.append(dict(group=key,weights=weights[wi],regularization=k.LAMBDAS[li],
            config_index=int(chosen),n_train=len(tr),test_indices=te.tolist(),
            selected_inner_primary=float(primary[chosen]),best_inner_primary=float(primary[best]),
            paired_SE=float(se[chosen]),eligible_count=len(eligible)))
    assert np.isfinite(score).all()
    out=run_path('costanzo' if dataset=='Costanzo' else 'jonikas')
    tab=d.g[['a','b','pair']].copy();tab['target']=d.y;tab['score']=score;tab['reverse_score']=rev
    tab['direction']=(score-rev)/2;tab['support']=(score+rev)/2
    tab.to_csv(out/f'{family}_predictions.csv',index=False)
    (out/f'{family}_choices.json').write_text(json.dumps(choices,indent=2)+'\n')
    report=dict(dataset=dataset,family=family,**k.metric(d.y,score,rev,d.directional),
        candidate_inner_evaluations=evals,eigendecompositions=eig,seconds=time.monotonic()-start,
        historical_cache_used=False,prepared_input_boundary=True)
    (out/f'{family}_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report
