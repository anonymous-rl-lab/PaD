"""Frozen paired one-standard-error selection; candidate order is unchanged."""
import numpy as np
import pandas as pd
import kernel as u
from kernel import WEIGHTS, LAMBDAS, metric
def objective(y,s,directional):
 v=(2*y-1-s)**2
 if directional or len(set(y))<2:return float(v.mean())
 return float((v[y>0].mean()+v[y==0].mean())/2)

def paired_se(y,score,best,pairs,directional):
 if directional:
  delta=((best>=0)==(y>0)).astype(float)-((score>=0)==(y>0)).astype(float)
  influence=(delta-delta.mean())/len(y)
 else:
  pos=np.flatnonzero(y>0);neg=np.flatnonzero(y==0)
  def concordance(s):
   d=s[pos,None]-s[None,neg];return (d>0).astype(float)+.5*(d==0)
  d=concordance(best)-concordance(score);mu=d.mean();influence=np.zeros(len(y));influence[pos]=(d.mean(1)-mu)/len(pos);influence[neg]=(d.mean(0)-mu)/len(neg)
 unique,idx=np.unique(pairs,return_inverse=True);v=np.bincount(idx,weights=influence);n=len(unique)
 return float(np.sqrt(np.sum(v*v)*n/max(n-1,1)))

def select_from_inner(data,tr,s):
 configs=[(i,l) for i in range(len(WEIGHTS)) for l in LAMBDAS];y=data.y[tr]
 ms=[metric(y,v,directional=data.directional) for v in s];primary=np.array([m['accuracy'] if data.directional else m['auroc'] for m in ms]);loss=np.array([objective(y,v,data.directional) for v in s])
 best=min(range(len(configs)),key=lambda j:(-primary[j],loss[j],-configs[j][1],configs[j][0]));gap=primary[best]-primary
 se=np.array([paired_se(y,v,s[best],data.g.pair.to_numpy()[tr],data.directional) for v in s]);eligible=np.flatnonzero(gap<=se+1e-12)
 chosen=min(eligible,key=lambda j:(-configs[j][1],-primary[j],loss[j],configs[j][0]))
 return chosen,best,primary,loss,gap,se,eligible
