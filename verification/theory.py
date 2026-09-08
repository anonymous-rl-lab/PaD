#!/usr/bin/env python3
"""Deterministic mechanism and complete-encoder checks. No model refitting."""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from paths import run_path
OUT=run_path("verification") / "theory"
OUT.mkdir(parents=True,exist_ok=True)
Z = np.array([-.8,.8,-.6,.6,-.4,.4,0.])
def load(source=None):
    import features
    from readouts import corr, double_features
    from paths import DATA
    frozen=json.loads((DATA/'prepared/frozen_P.json').read_text())
    return features,corr,double_features,np.array(frozen['coefficients'])/np.array(frozen['scale'])

def encoder(y,c,k,s,qlog,u0,v0,tools):
    core,corr,double_features,coef=tools
    g=np.exp(c*k)
    # Structural equations. Both baselines are free; use actual log2 ratios.
    U=u0*np.exp2(np.exp(s)*Z)
    V=v0*np.power(U/u0,g)
    ru=np.log2(U/u0);rv=np.log2(V/v0)
    if not np.allclose(ru,np.exp(s)*Z,atol=1e-11):raise AssertionError('upstream log observation')
    if not np.allclose(rv,np.exp(s+c*k)*Z,atol=1e-11):raise AssertionError('downstream log observation')
    A,B=(ru,rv) if y==1 else (rv,ru)
    src=[f'E{i}' for i in range(6)]+['W']
    data=dict(sources=src,gene_index={'A':0,'B':1},source_index={e:i for i,e in enumerate(src)},effect=np.array([A,B]),pvalue=np.full((2,7),.001))
    raw=core.FeatureBuilder(data).feature('A','B');p=float(raw@coef)
    ha=abs(A)>=.5;hb=abs(B)>=.5;union=(ha|hb).sum();shared=(ha&hb).sum()
    xp=np.r_[np.tanh(p/2),corr(A,B),A@B/(np.linalg.norm(A)*np.linalg.norm(B)+1e-9),shared/(union+1),np.tanh(np.log1p(shared)/3),np.tanh(np.log1p(union)/4),np.tanh(np.log(np.std(A)*np.std(B)+1e-8)/4),np.tanh(abs(np.log1p(ha.sum())-np.log1p(hb.sum()))/2)]
    # Reporter L=UV/(u0*v0)+exp(qlog)*(1-W)*U/u0. All local knockouts give 0.
    pu=np.zeros(7);pv=np.zeros(7);pv[-1]=np.exp(qlog)
    ga,gb=(pu,pv) if y==1 else (pv,pu)
    ctx=[corr(ga,gb),corr(ga,gb),np.tanh(np.log1p(7)/4),np.tanh(np.mean(abs(ga-gb))),np.mean(np.sign(ga)==np.sign(gb))]
    ss,dd,qq=double_features(np.array([0.]),np.array([0.]),np.array([0.]),np.array([0.]))
    xd=np.r_[np.tanh(ss[0]/.5),np.tanh(dd[0]),qq[0,:3],np.tanh(qq[0,3]),.5,2.,0.,0.,0.,0.,ctx]
    T=int(np.sign(p))
    X=.5*np.log((np.exp(4*np.arctanh(xp[6]))-1e-8)/np.var(Z))
    m=np.expm1(4*np.arctanh(xd[20]));Q=np.log(m*np.arctanh(xd[21]))
    assert xp.shape==(8,) and xd.shape==(23,)
    assert T==y*c and abs(X-(s+c*k/2))<2e-8 and abs(Q-qlog)<2e-8
    return dict(y=y,c=c,k=k,s=s,qlog=qlog,u0=u0,v0=v0,T=T,X=X,Q=Q,p_score=p,raw=raw.tolist(),xp=xp.tolist(),xd=xd.tolist())

def smoke(tools):
    rows=[]
    for k in [.3,.7,1.2]:
      for y in [-1,1]:
       for c in [-1,1]:
        for s in [-1.,0.,1.]:
         for u0,v0 in [(1.,1.),(2.3,7.1)]:
          rows.append(encoder(y,c,k,s,c*.6,u0,v0,tools))
    # Same T,X,k, but opposite background: S is allowed to differ, NOT forced to be constant.
    diffs=[]
    for k in [.3,.7,1.2]:
      for t in [-1,1]:
       for x in [-1.,0.,1.]:
        aa=encoder(t,1,k,x-k/2,.4,1.4,3.6,tools)
        bb=encoder(-t,-1,k,x+k/2,-.4,5.2,.9,tools)
        d=float(np.max(abs(np.array(aa['xp'])-bb['xp'])))
        assert d<1e-9;diffs.append(d)
    # Complete D is orientation invariant.
    for c in [-1,1]:
      aa=encoder(1,c,.7,0.,c*.6,1.,1.,tools);bb=encoder(-1,c,.7,0.,c*.6,1.,1.,tools)
      assert np.max(abs(np.array(aa['xd'])-bb['xd']))<1e-12
    return dict(passed=True,full_encoder_records=len(rows),overlap_pairs=len(diffs),max_complete_P_collision_error=max(diffs),examples=rows[:8])

def analytic():
    import itertools
    ss=[];yy=[]
    for tt,cc,bit in itertools.product([-1,1],repeat=3):
        ss.append((0 if tt==1 else 2*bit)+cc);yy.append(tt*cc)
    ss=np.array(ss);yy=np.array(yy);diff=ss[yy>0,None]-ss[None,yy<0]
    auc=float(((diff>0)+.5*(diff==0)).mean());err=float((np.sign(ss)!=yy).mean())
    assert auc==9/16 and err==.25
    (OUT/'additive_nuisance_counterexample.json').write_text(json.dumps(dict(n_states=8,error=err,auc=auc,P_context_error=.5,note='independent nuisance invalidates a naive transfer of V5 exact AUC equality; does not violate V6 bound'),indent=2))
    rows=[]
    for a in [0.,.1,.25,.5,1.,2.,4.]:
      overlap=quad(lambda x:min(norm.pdf(x-a),norm.pdf(x+a)),-12,12,epsabs=1e-11)[0]
      assert abs(overlap/2-norm.cdf(-a))<1e-9
      for b in [0.,.5,1.,2.,3.]:
        e=norm.cdf(-a);err=norm.cdf(-b);auc=norm.cdf(np.sqrt(2)*b)
        rows.append(dict(a=a,b=b,context_error_in_complete_P=e,additive_error_lower=e/2,antisymmetric_error_lower=e,joint_context_rule_error=err,joint_context_rule_auc=auc,additive_auc_upper=1-e*e/2,classification_gap_lower=e/2-err,antisymmetric_gap_lower=e-err,auc_gap_lower=auc-1+e*e/2,bayes_P_error=e,bayes_P_auc=norm.cdf(np.sqrt(2)*a),bayes_joint_error=norm.cdf(-np.hypot(a,b)),bayes_joint_auc=norm.cdf(np.sqrt(2)*np.hypot(a,b))))
    pd.DataFrame(rows).to_csv(OUT/'analytic_grid.csv',index=False)
    e=norm.cdf(-.5)
    region=dict(a_max=.5,b_min=2.,P_error_lower=e,additive_error_lower=e/2,joint_error_upper=norm.cdf(-2),classification_gap_lower=e/2-norm.cdf(-2),antisymmetric_gap_lower=e-norm.cdf(-2),additive_auc_upper=1-e*e/2,joint_auc_lower=norm.cdf(np.sqrt(2)*2),auc_gap_lower=norm.cdf(np.sqrt(2)*2)-1+e*e/2)
    return dict(quadrature_checks=7,parameter_cells=len(rows),uniform_region=region)

def continuous_gain(tools):
    rng=np.random.default_rng(20260907);rows=[]
    # Bounded full-encoder stress cases, not used to estimate population Bayes risks.
    for i in range(360):
      k=float(rng.uniform(.3,.8));c=int(rng.choice([-1,1]));y=int(rng.choice([-1,1]));s=float(rng.uniform(-2,2));q=float(rng.uniform(-1,1))
      r=encoder(y,c,k,s,q,float(rng.uniform(.2,5)),float(rng.uniform(.2,5)),tools)
      rows.append({key:r[key] for key in ['y','c','k','s','qlog','T','X','Q','p_score']})
    pd.DataFrame(rows).to_csv(OUT/'literal_encoder_continuous_gain.csv',index=False)
    return dict(cases=len(rows),passed=True,note='bounded numerical stress panel; analytic normal tails handled by proofs, not by this bounded panel')
