#!/usr/bin/env python3
"""Execution adapter for frozen v4: raw observations, full encoders, original CV.

No latent variable enters a fitted model. Training-only normalization; rectangular
final prediction avoids constructing a test-by-test Gram matrix. The release verification command checks the encoder and solver interfaces.
"""
from __future__ import annotations
import hashlib, json, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.stats import norm, beta
from sklearn.metrics import roc_auc_score

from paths import REPO, DATA, run_path
ROOT=run_path('mechanism')
import kernel as v4
from selection import select_from_inner, paired_se, objective
from features import FeatureBuilder
from readouts import corr, double_features

PROTOCOL=json.loads((REPO/'configs/mechanism.json').read_text())
Z=np.array(PROTOCOL['physical']['z'])
FP=json.loads((DATA/'prepared/frozen_P.json').read_text())
COEF=np.array(FP['coefficients'])/np.array(FP['scale'])
FAMILIES=['joint','A_match','A_rbf','full_P_RBF']
RW=[1.,0.,.5,1/6,5/6,1/3,2/3]
SCENARIOS={x['id']:x for x in PROTOCOL['scenarios']}

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    def default(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,Path):return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,indent=2,default=default,allow_nan=False)+'\n')

def stream(phase,purpose,scenario,seed,level):
    key=json.dumps([phase,purpose,scenario,int(seed),str(level)],separators=(',',':'))
    digest=hashlib.sha256(key.encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest,dtype='<u4'))),digest.hex()

def raw(phase,purpose,scenario,seed,n,level='ideal',noise=(0.,0.)):
    cfg=SCENARIOS[scenario];a,b,rho=[cfg[x] for x in ['a','b','rho']]
    rng,key=stream(phase,purpose,scenario,seed,level)
    Y=rng.choice([-1,1],n);C=rng.choice([-1,1],n)
    zp=rng.normal(size=n);ze=rng.normal(size=n);zd=rho*zp+np.sqrt(1-rho*rho)*ze
    X=a*C+zp;Q=b*C+zd;T=Y*C;k=PROTOCOL['physical']['kappa']
    # Exact log-observation equations, algebraically evaluated without exp2(U).
    ideal_E=np.stack([np.exp(X-T*k/2)[:,None]*Z,np.exp(X+T*k/2)[:,None]*Z],axis=1)
    ideal_G=np.zeros((n,2,len(Z)));ideal_G[np.arange(n),(Y>0).astype(int),-1]=np.exp(Q)
    re,kp=stream(phase,purpose+'-external-expression',scenario,seed,level)
    rg,kd=stream(phase,purpose+'-external-phenotype',scenario,seed,level)
    E=ideal_E+noise[0]*re.normal(size=ideal_E.shape)
    G=ideal_G+noise[1]*rg.normal(size=ideal_G.shape)
    assert np.isfinite(E).all() and np.isfinite(G).all()
    ids=np.array([key+':'+str(i) for i in range(n)])
    meta=dict(Y=Y,C=C,T=T,X=X,Q=Q,ZP=zp,ZD=zd,instance_id=ids)
    audit=dict(stream=key,expression_noise_stream=kp,phenotype_noise_stream=kd,n=n,
               phase=phase,purpose=purpose,scenario=scenario,seed=seed,level=level,
               noise_std=list(noise),positive_count=int((Y>0).sum()),independent_instances=n)
    return E,G,meta,audit

def encode_one(A,B,ga,gb):
    src=[f'E{i}' for i in range(6)]+['W']
    data=dict(sources=src,gene_index={'A':0,'B':1},source_index={e:i for i,e in enumerate(src)},
              effect=np.array([A,B]),pvalue=np.full((2,7),.001))
    rawp=FeatureBuilder(data).feature('A','B');p=float(rawp@COEF)
    ha=abs(A)>=.5;hb=abs(B)>=.5;union=(ha|hb).sum();shared=(ha&hb).sum()
    xp=np.r_[np.tanh(p/2),corr(A,B),A@B/(np.linalg.norm(A)*np.linalg.norm(B)+1e-9),
        shared/(union+1),np.tanh(np.log1p(shared)/3),np.tanh(np.log1p(union)/4),
        np.tanh(np.log(np.std(A)*np.std(B)+1e-8)/4),
        np.tanh(abs(np.log1p(ha.sum())-np.log1p(hb.sum()))/2)]
    ctx=[corr(ga,gb),corr(ga,gb),np.tanh(np.log1p(7)/4),
         np.tanh(np.mean(abs(ga-gb))),np.mean(np.sign(ga)==np.sign(gb))]
    ss,dd,qq=double_features(np.array([0.]),np.array([0.]),np.array([0.]),np.array([0.]))
    xd=np.r_[np.tanh(ss[0]/.5),np.tanh(dd[0]),qq[0,:3],np.tanh(qq[0,3]),.5,2.,0.,0.,0.,0.,ctx]
    return p,xp,xd,rawp

def encode(E,G,meta):
    # Every record calls the original FeatureBuilder and literal D functions.
    n=len(E);p=np.empty(n);xp=np.empty((n,8));xd=np.empty((n,23));rp=np.empty((n,5))
    for i in range(n):p[i],xp[i],xd[i],rp[i]=encode_one(*E[i],*G[i])
    xpr=xp.copy();xpr[:,0]*=-1;xdr=xd.copy()
    assert np.isfinite(xp).all() and np.isfinite(xd).all() and np.isfinite(p).all()
    # Raw local odd inputs are exactly zero. Their training-only median scales
    # have the inherited 0.01 fallback; no cross-record scale enters fixed xp/xd.
    result=dict(p=p,odd=np.zeros((n,8)),xp=xp,xpr=xpr,xd=xd,xdr=xdr,
                y=(meta['Y']>0).astype(float),directional=np.array(1))
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        m=np.expm1(4*np.arctanh(xd[:,20]));qhat=np.log(m*np.arctanh(xd[:,21]))
    badq=~np.isfinite(qhat);badt=~np.isfinite(p)|(p==0)
    diag=dict(p_raw=rp,decoded_T=np.where(p>=0,1,-1),decoded_Q=qhat,
        bad_T=badt,bad_Q=badq,
        error_T=badt|(np.where(p>=0,1,-1)!=meta['T']),
        error_Q=badq|(np.where(qhat>=0,1,-1)!=np.where(meta['Q']>=0,1,-1)),
        saturation_P=np.any(np.abs(xp[:,[0,6]])==1,axis=1),saturation_D=xd[:,21]==1,
        weak_expression=np.max(np.abs(E),axis=(1,2))<.5,
        near_threshold=np.min(np.abs(np.abs(E)-.5),axis=(1,2))<.01,
        low_readout=np.exp(meta['Q'])<.1)
    return result,diag

def dataset(phase,purpose,scenario,seed,n,level='ideal',noise=(0.,0.)):
    E,G,meta,audit=raw(phase,purpose,scenario,seed,n,level,noise)
    z,diag=encode(E,G,meta)
    audit.update(full_encoder_records=n,feature_dimensions=[8,23],
        finite_feature_arrays=True,decoder_bad_Q=int(diag['bad_Q'].sum()),
        decoder_bad_T=int(diag['bad_T'].sum()),saturation_P=int(diag['saturation_P'].sum()),
        saturation_D=int(diag['saturation_D'].sum()))
    return z,meta,diag,audit

def weights(family):
    if family in ['joint','A_match']:return v4.WEIGHTS
    if family=='A_rbf':return [(w,1-w) for w in RW]
    if family=='full_P_RBF':return [(1.,0.)]
    raise ValueError(family)

class Data(v4.Data):
    def __init__(self,z,g,family):
        self.name='mechanism';self.z=z;self.g=g;self.y=z['y'];self.n=len(self.y)
        self.directional=bool(z['directional']);self.family=family;self.weights=weights(family)
        self.P=v4.rbf(z['xp'],z['xp']);self.Pr=v4.rbf(z['xpr'],z['xp'])
        self.D=v4.rbf(z['xd'],z['xd']);self.Dr=v4.rbf(z['xdr'],z['xd'])
        self.J=self.P*self.D;self.Jr=self.Pr*self.Dr
    def components(self,tr):
        if self.family in ['A_rbf','full_P_RBF']:return [self.P,self.D],[self.Pr,self.Dr]
        cc,rr=super().components(tr)
        if self.family=='A_match':return [cc[0],cc[1],(self.P+self.D)/2],[rr[0],rr[1],(self.Pr+self.Dr)/2]
        return cc,rr

def select(data,tr,s):
    if len(s)==35:return select_from_inner(data,tr,s)
    assert len(s)==5 and data.family=='full_P_RBF'
    # Repetition presents the exact five-lambda boundary to the unchanged
    # 35-row selector; deterministic ties choose the first identical block.
    out=select_from_inner(data,tr,np.tile(s,(7,1)))
    assert out[0]<5 and out[1]<5
    return out[0],out[1],*[x[:5] for x in out[2:6]],out[6][out[6]<5]

def fitted(data,tr,theta,lam):
    tr=np.asarray(tr,int);cc,rr=data.components(tr)
    K=sum(t*c for t,c in zip(theta,cc));R=sum(t*c for t,c in zip(theta,rr))
    if data.directional:K=(K-R)/2
    y=data.y[tr];w=np.ones(len(tr))
    if not data.directional:
        n1=max(int(y.sum()),1);n0=max(len(tr)-int(y.sum()),1)
        w=np.where(y>0,len(tr)/(2*n1),len(tr)/(2*n0))
    sw=np.sqrt(w);v,U=eigh(K[np.ix_(tr,tr)]*sw[:,None]*sw[None,:],check_finite=False);v=np.maximum(v,0)
    alpha=sw*(U@((U.T@(sw*(2*y-1)))/(v+lam*len(tr))))
    z=data.z;ps=float(v4.rms(z['p'][tr,None])[0]);os=v4.rms(z['odd'][tr])
    o=np.clip(z['odd'][tr]/os,-8,8);onorm=max(float(np.mean(np.sum(o*o,axis=1))),1e-8)
    return dict(alpha=alpha,tr=tr,theta=theta,lam=lam,p_scale=ps,odd_scale=os,odd_norm=onorm,
                family=data.family,directional=data.directional,
                train={k:z[k][tr] for k in ['p','odd','xp','xpr','xd','xdr']})

def predict(model,z,batch=512):
    n=len(z['p']);s=np.empty(n);sr=np.empty(n);tz=model['train'];family=model['family']
    pt=tz['p']/model['p_scale'];ot=np.clip(tz['odd']/model['odd_scale'],-8,8)
    for lo in range(0,n,batch):
        ix=slice(lo,min(n,lo+batch));P=v4.rbf(z['xp'][ix],tz['xp']);Pr=v4.rbf(z['xpr'][ix],tz['xp'])
        D=v4.rbf(z['xd'][ix],tz['xd']);Dr=v4.rbf(z['xdr'][ix],tz['xd'])
        if family in ['A_rbf','full_P_RBF']:cc=[P,D];rr=[Pr,Dr]
        else:
            kp=(z['p'][ix]/model['p_scale'])[:,None]*pt
            kd=np.clip(z['odd'][ix]/model['odd_scale'],-8,8)@ot.T/model['odd_norm']
            cc=[kp,kd,P*D if family=='joint' else (P+D)/2]
            rr=[-kp,-kd,Pr*Dr if family=='joint' else (Pr+Dr)/2]
        K=sum(t*c for t,c in zip(model['theta'],cc));R=sum(t*c for t,c in zip(model['theta'],rr))
        if model['directional']:K=(K-R)/2;R=-K
        s[ix]=K@model['alpha'];sr[ix]=R@model['alpha']
    return s,sr

def train_four(z,meta,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);n=len(z['y'])
    ids=meta['instance_id'];g=pd.DataFrame(dict(a=[x+':A' for x in ids],b=[x+':B' for x in ids],pair=ids))
    # Hash partition depends on IDs only, not labels, and gives five balanced groups.
    order=np.argsort([hashlib.sha256(x.encode()).hexdigest() for x in ids]);fold=np.empty(n,int);fold[order]=np.arange(n)%5
    pd.DataFrame(dict(instance_id=ids,fold=fold)).to_csv(out/'inner_groups.csv',index=False)
    models={};choices=[];cache_P=None;ledger=dict(candidate_fold_evaluations=0,eigendecompositions=0,final_fits=0,P_only_CV_reused=True)
    for family in FAMILIES:
        start=time.monotonic();d=Data(z,g,family);ww=d.weights;configs=[(i,l) for i in range(len(ww)) for l in v4.LAMBDAS]
        cv=np.full((len(configs),n),np.nan)
        if family=='full_P_RBF':cv[:]=cache_P
        else:
            for f in range(5):
                tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
                assert not set(g.a[tr])&set(g.a[te]) and not set(g.b[tr])&set(g.b[te])
                for i,theta in enumerate(ww):
                    fit=v4.fit(d,tr,theta);ledger['eigendecompositions']+=1
                    for j,l in enumerate(v4.LAMBDAS):cv[i*5+j,te]=fit[l][0][te]
        ledger['candidate_fold_evaluations']+=len(configs)*5
        if family=='A_rbf':cache_P=cv[:5].copy()
        assert np.isfinite(cv).all()
        tr=np.arange(n);chosen,best,primary,loss,gap,se,eligible=select(d,tr,cv);i,lam=configs[chosen]
        model=fitted(d,tr,ww[i],lam);models[family]=model;ledger['eigendecompositions']+=1;ledger['final_fits']+=1
        choice=dict(family=family,weight_index=i,theta=ww[i],regularization=lam,
                    selected_inner_accuracy=float(primary[chosen]),best_inner_accuracy=float(primary[best]),
                    shortfall=float(gap[chosen]),paired_SE=float(se[chosen]),eligible_count=len(eligible),
                    n_train=n,seconds=time.monotonic()-start,
                    preprocessing=dict(P_linear_RMS=model['p_scale'],odd_RMS=model['odd_scale'],odd_norm=model['odd_norm'],
                    full_summary_constants='fixed measurement maps; zero local raw D has train-only 0.01 median fallback'))
        choices.append(choice);np.savez_compressed(out/(family+'_inner.npz'),score=cv,fold=fold)
        model_to_save={k:v for k,v in model.items() if k!='train'}
        model_to_save.update({'train_'+k:v for k,v in model['train'].items()})
        np.savez_compressed(out/(family+'_fit.npz'),**model_to_save)
        print('TRAIN',family,'n',n,'theta',ww[i],'lambda',lam,'cv',round(float(primary[chosen]),4),flush=True)
    dump(out/'choices.json',choices);dump(out/'training_ledger.json',ledger)
    return models,choices,ledger

def theoretical(scenario):
    a,b,rho=[SCENARIOS[scenario][x] for x in ['a','b','rho']];d=(b-rho*a)/np.sqrt(1-rho*rho);r=np.hypot(a,d)
    e=norm.cdf(-abs(a)/np.sqrt(1-rho*rho))
    return dict(P_error=float(norm.cdf(-abs(a))),P_auc=float(norm.cdf(np.sqrt(2)*abs(a))),
                joint_error=float(norm.cdf(-r)),joint_auc=float(norm.cdf(np.sqrt(2)*r)),
                additive_error_lower=float(e/2),additive_auc_upper=float(1-e*e/2),conditional_d=float(d))

def cp(k,n,alpha=.05):
    return (0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),
            1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k)))

def equivalence(jwrong,pwrong):
    n=len(jwrong);broken=int(np.sum(jwrong&~pwrong));fixed=int(np.sum(~jwrong&pwrong))
    bi=cp(broken,n,.05/(3*2));fi=cp(fixed,n,.05/(3*2));ci=[bi[0]-fi[1],bi[1]-fi[0]]
    return dict(n=n,joint_wrong_P_right=broken,joint_right_P_wrong=fixed,difference=(broken-fixed)/n,
                interval=ci,margin=.02,practical_equivalence=bool(ci[0]>=-.02 and ci[1]<=.02),
                method='two simultaneous exact-binomial intervals; 3-seed Bonferroni family')

def run_block(phase,scenario,seed,ntrain,ntest,level='ideal',noise=(0.,0.)):
    prefix=ROOT/phase/f'{scenario}_seed{seed}_{level}';prefix.mkdir(parents=True,exist_ok=True)
    if (prefix/'result.json').exists():return json.loads((prefix/'result.json').read_text())
    start=time.monotonic();z,meta,diag,ta=dataset(phase,'train',scenario,seed,ntrain,level,noise)
    models,choices,ledger=train_four(z,meta,prefix)
    # Independent formal tests generated only after all four model selections.
    zz,mm,dd,ea=dataset(phase,'test',scenario,seed,ntest,level,noise)
    assert ta['stream']!=ea['stream'] and not set(meta['instance_id'])&set(mm['instance_id'])
    np.savez_compressed(prefix/'train_encoded.npz',**z,**{'meta_'+k:v for k,v in meta.items()})
    np.savez_compressed(prefix/'test_encoded.npz',**zz,**{'meta_'+k:v for k,v in mm.items()})
    theory=theoretical(scenario);rows=[];pred=pd.DataFrame(mm);pred['raw_P_score']=zz['p'];pred['decoded_Q']=dd['decoded_Q']
    corr=np.sqrt(np.log(2*30/.05)/(2*ntest));bad={}
    for family in FAMILIES:
        s,sr=predict(models[family],zz);direction=(s-sr)/2
        wrong=(direction>=0)!=(mm['Y']>0);bad[family]=wrong
        nerr=int(wrong.sum());met=v4.metric(zz['y'],s,sr,True)
        row=dict(family=family,errors=nerr,error_rate=nerr/ntest,**met)
        pred[family+'_score']=s;pred[family+'_reverse']=sr;pred[family+'_wrong']=wrong
        if phase=='formal' and family=='joint':
            row.update(certification_M=30,risk_upper=min(1,nerr/ntest+corr),
                additive_error_lower=theory['additive_error_lower'],
                classification_certified=bool(nerr/ntest+corr<theory['additive_error_lower']),
                B1_integer_rule=(bool(nerr<=456) if scenario=='B1' else None))
            if scenario=='B1':assert row['classification_certified']==row['B1_integer_rule']
            npos=int(zz['y'].sum());nneg=ntest-npos
            auc_penalty=np.sqrt(.5*(1/npos+1/nneg)*np.log(2*30/.05))
            row.update(auc_lower=max(0,met['auroc']-auc_penalty),auc_correction=auc_penalty,
                auc_certified=bool(met['auroc']-auc_penalty>theory['additive_auc_upper']))
        rows.append(row)
    rawerr=int(np.sum((zz['p']>=0)!=(mm['Y']>0)))
    eq=equivalence(bad['joint'],bad['full_P_RBF']) if scenario=='B2' and level=='ideal' else None
    pred.to_csv(prefix/'test_predictions.csv',index=False)
    pd.DataFrame({k:v for k,v in dd.items() if np.asarray(v).ndim==1}).to_csv(prefix/'test_decoder_diagnostics.csv',index=False)
    result=dict(phase=phase,scenario=scenario,seed=seed,level=level,noise_std=noise,n_train=ntrain,n_test=ntest,
        results=rows,raw_P_errors=rawerr,theory_ideal=theory,ideal_B2_equivalence=eq,
        train_audit=ta,test_audit=ea,choices=choices,ledger=ledger,seconds=time.monotonic()-start,
        hidden_parameters_used_for_learning=False)
    dump(prefix/'result.json',result)
    print('BLOCK',phase,scenario,seed,level,[(r['family'],r['errors'],round(r['auroc'],4)) for r in rows],
          'seconds',round(result['seconds'],2),flush=True)
    return result

