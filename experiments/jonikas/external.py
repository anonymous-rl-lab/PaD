"""Frozen jonikas external comparison; outputs are regenerable."""
import argparse, hashlib, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from paths import REPO, DATA, run_path
ROOT=run_path('jonikas')
import kernel as u
def write(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def data():
 d=u.Data('KEGG');z=d.z
 X=np.c_[z['p'],z['xp'][:,1:],z['odd'],z['xd'][:,8:]]
 assert X.shape==(168,31) and np.isfinite(X).all()
 assert len(d.groups)==84 and d.y.sum()==21
 return d,X

def configs(family):
 if family=='svm':return [dict(C=c,gamma=g) for c in [.3,3.,30.] for g in ['scale',.1,1.]]
 if family=='xgboost':return [dict(max_depth=d,reg_lambda=l) for d in [2,4] for l in [1.,10.]]
 if family=='catboost':return [dict(depth=d,l2_leaf_reg=l) for d in [3,5] for l in [1.,10.]]
 raise ValueError(family)

def model(family,c,seed):
 if family=='svm':return make_pipeline(StandardScaler(),SVC(**c,kernel='rbf',probability=False,class_weight='balanced',cache_size=128))
 if family=='xgboost':
  from xgboost import XGBClassifier
  return XGBClassifier(**c,n_estimators=300,learning_rate=.05,objective='binary:logistic',subsample=1.,colsample_bytree=1.,tree_method='hist',random_state=seed,n_jobs=1,eval_metric='logloss',verbosity=0)
 from catboost import CatBoostClassifier
 return CatBoostClassifier(**c,iterations=500,learning_rate=.03,loss_function='Logloss',random_seed=seed,thread_count=1,verbose=False,allow_writing_files=False)

def decision(m,family,X):
 if family=='svm':return m.decision_function(X)
 if family=='xgboost':return m.predict(X,output_margin=True).astype(float)
 return np.asarray(m.predict(X,prediction_type='RawFormulaVal'),float)

def stat(y,s):
 y=np.asarray(y);loss=np.logaddexp(0,-(2*y-1)*s)
 return dict(n=len(y),positive=int(y.sum()),auroc=float(roc_auc_score(y,s)) if len(set(y))==2 else None,average_precision=float(average_precision_score(y,s)) if y.sum() else None,balanced_log_loss=float(np.mean([loss[y==k].mean() for k in [0,1] if np.any(y==k)])))

def fit_one(family,c,seed,tr,X,y,tag,return_model=False):
 t=time.monotonic();tr=np.asarray(tr,int);m=model(family,c,seed);yy=y[tr]
 assert len(set(yy))==2
 weight=np.where(yy>0,len(tr)/(2*yy.sum()),len(tr)/(2*(len(tr)-yy.sum())))
 if family=='svm':m.fit(X[tr],yy)
 else:m.fit(X[tr],yy,sample_weight=weight)
 s=decision(m,family,X);assert np.isfinite(s).all()
 info=dict(family=family,config=c,seed=seed,tag=tag,n_train=len(tr),n_positive=int(yy.sum()),seconds=time.monotonic()-t,train_indices=tr.tolist())
 return s,info,m if return_model else None

def ledger(rows):
 with (ROOT/'fit_ledger.jsonl').open('a') as f:
  for row in rows:f.write(json.dumps(row)+'\n')

def split_plan(d):
 groups=list(d.groups);pool={};outer=[]
 for k,key in enumerate(groups):
  tr=d.allowed[key];te=d.groups[key];ep=set(d.g.a.iloc[te])|set(d.g.b.iloc[te])
  assert not ep&(set(d.g.a.iloc[tr])|set(d.g.b.iloc[tr]))
  entries=[]
  for j,jkey in enumerate(groups):
   iv=np.intersect1d(tr,d.groups[jkey])
   if not len(iv):continue
   itr=tuple(np.intersect1d(tr,d.allowed[jkey]));ie=set(d.g.a.iloc[iv])|set(d.g.b.iloc[iv])
   assert not (ep|ie)&(set(d.g.a.iloc[list(itr)])|set(d.g.b.iloc[list(itr)]))
   if itr not in pool:pool[itr]=len(pool)
   entries.append((pool[itr],iv.tolist()))
  assert sorted(sum([v for _,v in entries],[]))==sorted(tr.tolist())
  outer.append(dict(key=key,tr=tr.tolist(),te=te.tolist(),inner=entries))
 sets=[None]*len(pool)
 for k,v in pool.items():sets[v]=[int(i) for i in k]
 return outer,sets

def nested(family,seed,outer_ids,workers=4):
 d,X=data();plan= json.loads((ROOT/'split_plan.json').read_text());outer=plan['outer'];sets=plan['unique_inner_training_indices'];nsets=len(sets)
 grid=configs(family);needed=sorted(set(v for k in outer_ids for v,_ in outer[k]['inner']))
 signature=hashlib.sha256((REPO/'configs/jonikas.md').read_bytes()+(Path(__file__)).read_bytes()+(DATA/'prepared/KEGG.npz').read_bytes()).hexdigest()
 cubes=[];start=time.monotonic()
 for cidx,c in enumerate(grid):
  path=ROOT/f'cache/{family}_{seed}_c{cidx}.npz';s=np.full((nsets,d.n),np.nan)
  if path.exists():
   cache=np.load(path)
   if str(cache['signature'])==signature:s=cache['score'].copy()
  missing=[i for i in needed if not np.isfinite(s[i]).all()]
  for off in range(0,len(missing),100):
   batch=missing[off:off+100]
   with parallel_config(backend='loky',inner_max_num_threads=1):
    values=Parallel(n_jobs=workers)(delayed(fit_one)(family,c,seed,sets[i],X,d.y,f'inner_{i}') for i in batch)
   for i,v in zip(batch,values):s[i]=v[0]
   ledger([v[1] for v in values]);np.savez_compressed(path,score=s,signature=np.array(signature))
  cubes.append(s)
  print('CANDIDATE',family,seed,cidx,'new_fits',len(missing),'seconds',round(time.monotonic()-start,1),flush=True)
 for k in outer_ids:
  dst=ROOT/f'{family}_{seed}_outer{k:02d}.json'
  if dst.exists() and json.loads(dst.read_text()).get('signature')==signature:continue
  spec=outer[k];tr=np.array(spec['tr']);te=np.array(spec['te']);rows=[]
  for cidx,c in enumerate(grid):
   pred=np.full(d.n,np.nan)
   for i,val in spec['inner']:pred[val]=cubes[cidx][i,val]
   assert np.isfinite(pred[tr]).all()
   rows.append(dict(candidate=cidx,config=c,**stat(d.y[tr],pred[tr])))
  best=min(rows,key=lambda q:(-q['auroc'],-q['average_precision'],q['balanced_log_loss'],q['candidate']))
  s,info,m=fit_one(family,best['config'],seed,tr,X,d.y,f'outer_{k}',True);ledger([info])
  modelpath=ROOT/f'models/{family}_{seed}_outer{k:02d}.pkl'
  with modelpath.open('wb') as f:pickle.dump(m,f)
  result=dict(family=family,seed=seed,outer=k,pair=spec['key'],signature=signature,train_indices=tr.tolist(),test_indices=te.tolist(),score=s[te].tolist(),selected=best,inner=rows,model=str(modelpath.relative_to(ROOT)))
  write(dst.name,result)
 print('DONE',family,seed,'outer',len(outer_ids),'seconds',round(time.monotonic()-start,1),flush=True)
