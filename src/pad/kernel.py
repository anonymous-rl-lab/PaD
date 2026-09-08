"""Frozen PaD kernels and balanced spectral ridge solver."""
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score, average_precision_score
from paths import DATA
WEIGHTS=[(1.,0.,0.),(0.,1.,0.),(0.,0.,1.),(.5,.5,0.),(.5,0.,.5),(0.,.5,.5),(1/3,1/3,1/3)]
LAMBDAS=[.00003,.0003,.003,.03,.3]
def rbf(a,b,width=.5):
 d=np.maximum((a*a).mean(1)[:,None]+(b*b).mean(1)[None,:]-2*a@b.T/a.shape[1],0.)
 return np.exp(-d/(2*width*width))

def rms(x):
 s=np.sqrt(np.mean(x*x,axis=0));return np.where(s<1e-8,1.,s)

class Data:
 def __init__(self,name):
  self.name=name;self.g=pd.read_csv(DATA/f'prepared/{name}.csv');self.z=dict(np.load(DATA/f'prepared/{name}.npz'));self.y=self.z['y'];self.n=len(self.y);self.directional=bool(self.z['directional'])
  z=self.z;self.J=rbf(z['xp'],z['xp'])*rbf(z['xd'],z['xd']);self.Jr=rbf(z['xpr'],z['xp'])*rbf(z['xdr'],z['xd'])
  if self.directional:self.groups={str(f):np.flatnonzero(self.g.fold==f) for f in sorted(set(self.g.fold))}
  else:self.groups={p:np.flatnonzero(self.g.pair==p) for p in dict.fromkeys(self.g.pair)}
  self.allowed={}
  for key,te in self.groups.items():
   endpoints=set(self.g.a.iloc[te])|set(self.g.b.iloc[te]);self.allowed[key]=np.flatnonzero(~self.g.a.isin(endpoints)&~self.g.b.isin(endpoints))

 def components(self,tr):
  z=self.z;p=z['p']/float(rms(z['p'][tr,None])[0]);o=np.clip(z['odd']/rms(z['odd'][tr]),-8,8)
  # All three kernels have training average diagonal one.
  onorm=float(np.mean(np.sum(o[tr]*o[tr],axis=1)));onorm=max(onorm,1e-8)
  kp=p[:,None]*p[None,:];kd=o@o.T/onorm
  return [kp,kd,self.J],[-kp,-kd,self.Jr]

 def rank(self,metric,lam,order):
  if self.directional:return (-metric['correct'],-metric['auroc'],metric['squared_loss'],-lam,order)
  return (-metric['auroc'],-metric['average_precision'],metric['squared_loss'],-lam,order)

def metric(y,s,sr=None,directional=False):
 d=s if sr is None else (s-sr)/2
 return dict(n=len(y),correct=int(np.sum((d>=0)==(y>0))) if directional else None,accuracy=float(np.mean((d>=0)==(y>0))) if directional else None,auroc=float(roc_auc_score(y,s)) if len(set(y))==2 else .5,average_precision=float(average_precision_score(y,s)) if np.any(y) else 0.,squared_loss=float(np.mean((2*y-1-s)**2)))

def fit(data,tr,theta,lambdas=LAMBDAS):
 tr=np.asarray(tr,int);cc,rr=data.components(tr);K=sum(t*c for t,c in zip(theta,cc));Kr=sum(t*c for t,c in zip(theta,rr))
 if data.directional:K=(K-Kr)/2;Kr=-K
 yy=data.y[tr];w=np.ones(len(tr))
 if not data.directional:
  n1=max(int(yy.sum()),1);n0=max(len(tr)-int(yy.sum()),1);w=np.where(yy>0,len(tr)/(2*n1),len(tr)/(2*n0))
 sw=np.sqrt(w);M=K[np.ix_(tr,tr)]*sw[:,None]*sw[None,:];v,U=eigh(M,check_finite=False);v=np.maximum(v,0.)
 rhs=U.T@(sw*(2*yy-1));alphas=np.column_stack([sw*(U@(rhs/(v+lam*len(tr)))) for lam in lambdas])
 ps=K[:,tr]@alphas;prs=Kr[:,tr]@alphas
 return {lam:(ps[:,k],prs[:,k]) for k,lam in enumerate(lambdas)}
