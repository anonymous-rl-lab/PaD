"""Frozen costanzo external comparison; outputs are regenerable."""
import argparse,hashlib,importlib,json,pickle,sys,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from paths import REPO, DATA, run_path
ROOT=run_path('costanzo')
import kernel as u
def write(name,value):
 p=ROOT/name;p.parent.mkdir(exist_ok=True,parents=True)
 p.write_text(json.dumps(value,indent=2,allow_nan=False))

def data():
 d=u.Data('Costanzo');z=d.z
 X=np.c_[z['p'],z['xp'][:,1:],z['odd'],z['xd'][:,12:]]
 XR=X.copy();XR[:,0]*=-1;XR[:,8:20]*=-1
 assert X.shape==(657,41) and np.isfinite(X).all()
 return d,X,XR

def configs(family):
 if family=='catboost':return [dict(depth=d,l2_leaf_reg=l) for d in [3,5,7] for l in [1.,5.,20.]]
 if family=='xgboost':return [dict(max_depth=d,reg_lambda=l,learning_rate=r) for d in [2,4,6] for l in [1.,10.] for r in [.03,.1]]
 if family=='svm':return [dict(C=c,gamma=g) for c in [.03,.3,3.,30.,300.] for g in ['scale',.01,.1,1.]]
 raise ValueError(family)

def model(family,c,seed):
 if family=='catboost':
  from catboost import CatBoostClassifier
  return CatBoostClassifier(**c,iterations=500,learning_rate=.03,loss_function='Logloss',random_seed=seed,thread_count=1,verbose=False,allow_writing_files=False)
 if family=='xgboost':
  from xgboost import XGBClassifier
  return XGBClassifier(**c,n_estimators=300,objective='binary:logistic',subsample=1.,colsample_bytree=1.,tree_method='hist',random_state=seed,n_jobs=1,eval_metric='logloss',verbosity=0)
 return make_pipeline(StandardScaler(),SVC(**c,kernel='rbf',probability=False,cache_size=256))

def decision(m,family,x):
 if family=='catboost':return np.asarray(m.predict(x,prediction_type='RawFormulaVal'))
 if family=='xgboost':return m.predict(x,output_margin=True)
 return m.decision_function(x)

def train(family,c,seed,tr,X,XR,y,tag):
 t=time.monotonic();m=model(family,c,seed)
 m.fit(np.r_[X[tr],XR[tr]],np.r_[y[tr],1-y[tr]])
 s=(decision(m,family,X)-decision(m,family,XR))/2
 assert np.isfinite(s).all()
 with (ROOT/'fit_ledger.jsonl').open('a') as f:
  f.write(json.dumps(dict(family=family,config=c,seed=seed,n_train_pairs=len(tr),augmented_rows=2*len(tr),tag=tag,seconds=time.monotonic()-t))+'\n')
 return m,s

def stat(y,s):
 return dict(n=len(y),correct=int(np.sum((s>=0)==(y>0))),accuracy=float(np.mean((s>=0)==(y>0))),auroc=float(roc_auc_score(y,s)),log_loss=float(np.logaddexp(0,-(2*y-1)*s).mean()),ties=int(np.sum(s==0)))

def nested(family,seed,folds):
 d,X,XR=data();f=d.g.fold.to_numpy();grid=configs(family)
 signature=hashlib.sha256((REPO/'configs/costanzo.md').read_bytes()+(Path(__file__)).read_bytes()+(DATA/'prepared/Costanzo.npz').read_bytes()).hexdigest()
 cache=ROOT/'cache';cache.mkdir(exist_ok=True)
 for outer in folds:
  dst=ROOT/f'{family}_{seed}_fold{outer}.json'
  if dst.exists():
   old=json.loads(dst.read_text())
   if old.get('signature')==signature:print('REUSE',family,seed,outer,flush=True);continue
  start=time.monotonic();tr=np.flatnonzero(f!=outer);te=np.flatnonzero(f==outer);inner=[];scores=[]
  for j,c in enumerate(grid):
   s=np.full(d.n,np.nan)
   for valfold in sorted(set(f[tr])):
    excluded=tuple(sorted((outer,int(valfold))));itr=np.flatnonzero(~np.isin(f,excluded));iv=np.flatnonzero(f==valfold)
    path=cache/f'{family}_{seed}_c{j}_ex{excluded[0]}{excluded[1]}.npz'
    cached=np.load(path) if path.exists() else None
    if cached is not None and str(cached['signature'])==signature:ps=cached['score']
    else:
     _,ps=train(family,c,seed,itr,X,XR,d.y,f'inner_ex{excluded}')
     np.savez_compressed(path,score=ps,signature=np.array(signature))
    s[iv]=ps[iv]
   assert np.isfinite(s[tr]).all()
   inner.append(dict(candidate=j,config=c,**stat(d.y[tr],s[tr])))
  best=min(inner,key=lambda q:(-q['correct'],-q['auroc'],q['log_loss'],q['candidate']))
  m,s=train(family,best['config'],seed,tr,X,XR,d.y,f'outer{outer}_selected')
  with (ROOT/f'models/{family}_{seed}_fold{outer}.pkl').open('wb') as fh:pickle.dump(m,fh)
  result=dict(family=family,seed=seed,fold=outer,signature=signature,test_indices=te.tolist(),score=s[te].tolist(),selected=best,inner=inner,n_train=len(tr),seconds=time.monotonic()-start,**stat(d.y[te],s[te]))
  dst.write_text(json.dumps(result,indent=2));print('FOLD',family,seed,outer,result['correct'],'/',len(te),'seconds',round(result['seconds'],2),'choice',best['config'],flush=True)

def red(seed):
 from red_py3 import Red
 dst=ROOT/f'red_{seed}.json'
 if dst.exists():print('REUSE RED',seed,flush=True);return
 z=np.load(DATA/'costanzo/red_input.npz');genes=z['genes'].tolist();idx={g:i for i,g in enumerate(genes)}
 t=time.monotonic();np.random.seed(seed);m=Red(z['G'].copy(),z['S'].copy(),z['H'].copy(),genes);iterations=[]
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always')
  m.order(rank=100,lambda_u=1e-4,lambda_v=1e-4,alpha=.1,beta=.1,max_iter=200,callback=lambda _:iterations.append(1))
 d,_,_=data();forward=[];reverse=[]
 for a,b in zip(d.g.a,d.g.b):forward.append(float(m.epistatic_to(b,a)));reverse.append(float(m.epistatic_to(a,b)))
 fw=np.array(forward);rv=np.array(reverse);bad=~(np.isfinite(fw)&np.isfinite(rv));score=fw-rv;score[bad]=0.
 res=dict(seed=seed,seconds=time.monotonic()-t,iterations=len(iterations)-1,genes=len(genes),n_nonfinite=int(bad.sum()),score=score.tolist(),released_forward=np.nan_to_num(fw,nan=.5).tolist(),released_reverse=np.nan_to_num(rv,nan=.5).tolist(),max_exchange_error=float(np.nanmax(abs(fw+rv-1))),primary=stat(d.y,score),released_onesided=stat(d.y,np.nan_to_num(fw,nan=.5)-.5),warnings=[str(w.message) for w in ws[:10]],fro_error=float(m.fro_error))
 np.savez_compressed(ROOT/f'models/red_{seed}.npz',U=m.U,V=m.V,ga=m._ga,gb=m._gb,gc=m._gc,forward=fw,reverse=rv)
 write(dst.name,res);print('RED',seed,json.dumps({k:v for k,v in res.items() if k not in ['score','released_forward','released_reverse']}),flush=True)
