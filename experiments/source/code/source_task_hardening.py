from pathlib import Path
import os, json, time, hashlib, multiprocessing as mp
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
if 'fork' not in mp.get_all_start_methods():
    raise RuntimeError('Full mismatch refits require Linux/WSL (POSIX fork); statistical replay is platform-independent.')
import numpy as np, pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score, average_precision_score

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA_DIR=Path(os.environ.get('PAD_DATA_DIR', ROOT/'data/jonikas'))
REF_PRED=Path(os.environ.get('PAD_REF_PRED', ROOT/'reference/source_ablation_reference.csv'))
OUT=Path(os.environ.get('PAD_OUT', ROOT/'rerun_results/source_task_hardening'))
OUT.mkdir(parents=True,exist_ok=True)
G=pd.read_csv(DATA_DIR/'KEGG.csv')
Z=dict(np.load(DATA_DIR/'KEGG.npz'))
Y=Z['y'].astype(float); N=len(Y)
LAM=np.array([.00003,.0003,.003,.03,.3],float)
S=[0,1,2,11,13,14]; M=[8,9,12]; H=[15]; V=[3,4,5,6,7,10,16,17]; B=[18,19,20,21,22]
GROUPS={'S':S,'M':M,'H':H,'V':V,'B':B}
SEED=20260920; N_SHUFFLE=100; N_BOOT=10000
pair_order=list(dict.fromkeys(G.pair)); pair_rows={p:np.flatnonzero(G.pair.to_numpy()==p) for p in pair_order}
allowed={}
for p,te in pair_rows.items():
    endpoints=set(G.a.iloc[te])|set(G.b.iloc[te])
    allowed[p]=np.flatnonzero(~G.a.isin(endpoints)&~G.b.isin(endpoints))
# canonical/reverse row per pair and row reverse index
canon={}; reverse_idx=np.empty(N,int)
for p,ix in pair_rows.items():
    assert len(ix)==2
    first=p.split('|')[0]
    ci=[i for i in ix if G.a.iloc[i]==first]
    assert len(ci)==1
    ci=ci[0]; ri=[i for i in ix if i!=ci][0]
    canon[p]=(ci,ri); reverse_idx[ci]=ri; reverse_idx[ri]=ci
assert np.max(np.abs(Z['xdr']-Z['xd'][reverse_idx]))<1e-12

PROTOCOL={
 'title':'Frozen source-task hardening on Jonikas development panel',
 'status':'development-only; not independent external validation',
 'frozen_before_new_results':True,
 'seed':SEED,'n_pair_orbit_shuffles_per_source':N_SHUFFLE,'n_bootstrap':N_BOOT,
 'learner':'pure joint RBF J, width 0.5, original 23-D denominator, 5 lambdas, endpoint-excluded nested CV, F-target paired 1-SE selection',
 'source_coordinates':GROUPS,
 'primary_tests':{
   'M_orientation':'d_AUROC(S+M+H) - d_AUROC(S+H)',
   'V_ranking':'F_AUROC(S+M+H+V+B) - F_AUROC(S+M+H+B)'
 },
 'cross_task_checks':['M effect on F_AUROC','V effect on d_AUROC','known-direction sign changes'],
 'source_destruction':{
   'M_primary':'remove M from S+M+H','M_full_context':'remove M from full S+M+H+V+B',
   'V_primary':'remove V from full S+M+H+V+B'
 },
 'shuffle_negative_control':'pair-orbit derangement; donor source block replaces recipient source block while preserving the donor block marginal distribution and exact endpoint-exchange orbit; learner is retrained and reselected from scratch',
 'gene_cluster_bootstrap':'resample the 30 endpoint genes with replacement; each unordered pair gets weight equal to the sum of multiplicities of its two endpoint genes; both ordered records share that weight; 95% percentile interval on fixed endpoint-excluded predictions',
 'pair_cluster_bootstrap':'resample 84 unordered pairs with replacement as sensitivity analysis',
 'gene_leave_one_out':'remove every record touching one gene, recompute fixed-prediction contrasts',
 'legacy_plus_one_exceedance_fraction':'(1 + # null_delta >= observed_delta)/(N_SHUFFLE+1)',
 'notes':['No metric/subset/threshold will be changed after observing these results.','Bootstrap and mismatch use the same 168 development records; they do not create external validation.', 'The plus-one exceedance fraction is not a conditional-independence p-value; blocks are mismatched without conditioning on other sources.']
}
(OUT/'PROTOCOL.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n')

# immutable input hashes
for f in [DATA_DIR/'KEGG.npz',DATA_DIR/'KEGG.csv']:
    PROTOCOL.setdefault('input_sha256',{})[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
(OUT/'PROTOCOL.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n')

def rbf(a,b,width=.5):
    d=np.maximum((a*a).mean(1)[:,None]+(b*b).mean(1)[None,:]-2*a@b.T/a.shape[1],0.)
    return np.exp(-d/(2*width*width))
P_K=rbf(Z['xp'],Z['xp']); P_R=rbf(Z['xpr'],Z['xp'])

def fit_kernel(K,Kr,tr):
    tr=np.asarray(tr,int); yy=Y[tr]
    n1=max(int(yy.sum()),1); n0=max(len(tr)-int(yy.sum()),1)
    w=np.where(yy>0,len(tr)/(2*n1),len(tr)/(2*n0)); sw=np.sqrt(w)
    Mat=K[np.ix_(tr,tr)]*sw[:,None]*sw[None,:]
    ev,U=eigh(Mat,check_finite=False); ev=np.maximum(ev,0.)
    rhs=U.T@(sw*(2*yy-1))
    alphas=np.column_stack([sw*(U@(rhs/(ev+lam*len(tr)))) for lam in LAM])
    return K[:,tr]@alphas, Kr[:,tr]@alphas

def objective(yy,s):
    v=(2*yy-1-s)**2
    return float((v[yy>0].mean()+v[yy==0].mean())/2)

def paired_se(yy,score,best,pairs):
    pos=np.flatnonzero(yy>0); neg=np.flatnonzero(yy==0)
    def concordance(s):
        d=s[pos,None]-s[None,neg]
        return (d>0).astype(float)+.5*(d==0)
    d=concordance(best)-concordance(score); mu=d.mean(); inf=np.zeros(len(yy))
    inf[pos]=(d.mean(1)-mu)/len(pos); inf[neg]=(d.mean(0)-mu)/len(neg)
    _,idx=np.unique(pairs,return_inverse=True); vv=np.bincount(idx,weights=inf); nn=len(np.unique(pairs))
    return float(np.sqrt(np.sum(vv*vv)*nn/max(nn-1,1)))

def nested_from_xd(xd):
    xdr=xd[reverse_idx]
    J=P_K*rbf(xd,xd); Jr=P_R*rbf(xdr,xd)
    out=np.full(N,np.nan); rev=np.full(N,np.nan); choice_lam=[]
    for p in pair_order:
        tr=allowed[p]; te=pair_rows[p]
        inner=np.full((5,N),np.nan); cache={}
        for ip,iv0 in pair_rows.items():
            val=np.intersect1d(tr,iv0)
            if not len(val): continue
            itr=tuple(np.intersect1d(tr,allowed[ip]))
            if itr not in cache: cache[itr]=fit_kernel(J,Jr,itr)
            ps,_=cache[itr]; inner[:,val]=ps[val,:].T
        assert np.isfinite(inner[:,tr]).all()
        prim=np.array([roc_auc_score(Y[tr],inner[j,tr]) for j in range(5)])
        loss=np.array([objective(Y[tr],inner[j,tr]) for j in range(5)])
        best=min(range(5),key=lambda j:(-prim[j],loss[j],-LAM[j],j))
        gap=prim[best]-prim
        se=np.array([paired_se(Y[tr],inner[j,tr],inner[best,tr],G.pair.to_numpy()[tr]) for j in range(5)])
        elig=np.flatnonzero(gap<=se+1e-12)
        chosen=min(elig,key=lambda j:(-LAM[j],-prim[j],loss[j],j))
        ps,prs=fit_kernel(J,Jr,tr); out[te]=ps[te,chosen]; rev[te]=prs[te,chosen]
        choice_lam.append(LAM[chosen])
    assert np.isfinite(out).all() and np.isfinite(rev).all()
    return out,rev,np.array(choice_lam)

def make_variant(names):
    xd=np.zeros_like(Z['xd'])
    coords=sorted({j for nm in names for j in GROUPS[nm]})
    xd[:,coords]=Z['xd'][:,coords]
    return xd

def metrics(s,r):
    d=(s-r)/2; u=(s+r)/2; pos=Y>0
    return dict(F_AUROC=float(roc_auc_score(Y,s)),d_AUROC=float(roc_auc_score(Y,d)),u_AUROC=float(roc_auc_score(Y,u)),
                known_correct=int(np.sum(d[pos]>0)),known_ties=int(np.sum(d[pos]==0)),n_known=int(pos.sum()))

def shuffle_block(context_names, source_name, perm):
    xd=make_variant(context_names)
    coords=GROUPS[source_name]
    for ri,rp in enumerate(pair_order):
        dp=pair_order[int(perm[ri])]
        rc,rr=canon[rp]; dc,dr=canon[dp]
        xd[rc,coords]=Z['xd'][dc,coords]
        xd[rr,coords]=Z['xd'][dr,coords]
    return xd

def derangements(n,count,seed):
    rng=np.random.default_rng(seed); out=[]
    base=np.arange(n)
    while len(out)<count:
        p=rng.permutation(n)
        if np.all(p!=base): out.append(p)
    return np.array(out,dtype=int)

# Static source destruction variants.
STATIC={'SH':['S','H'],'SMH':['S','M','H'],'SMHB':['S','M','H','B'],'FULL':['S','M','H','V','B'],'SHVB':['S','H','V','B']}
static_rows=[]; static_pred={}
for name,names in STATIC.items():
    t=time.time(); s,r,l=nested_from_xd(make_variant(names)); met=metrics(s,r); met.update(variant=name,seconds=time.time()-t)
    static_rows.append(met); static_pred[name]=(s,r)
    print('STATIC',name,met,flush=True)
static=pd.DataFrame(static_rows); static.to_csv(OUT/'static_metrics.csv',index=False)
pred=G[['a','b','pair','label']].copy()
for name,(s,r) in static_pred.items():
    pred[f'{name}_F']=s; pred[f'{name}_reverse']=r; pred[f'{name}_d']=(s-r)/2; pred[f'{name}_u']=(s+r)/2
pred.to_csv(OUT/'static_predictions.csv',index=False)

# Verify full exact against archived source-ablation full predictions.
old=pd.read_csv(REF_PRED)
rep_err=max(float(np.max(np.abs(pred.FULL_F-old.SLB_F))),float(np.max(np.abs(pred.FULL_reverse-old.SLB_reverse))))
(OUT/'reproduction_check.json').write_text(json.dumps({'full_prediction_max_abs_error':rep_err},indent=2)+'\n')
print('REPRO_ERR',rep_err,flush=True)

# Observed contrasts.
md={r['variant']:r for r in static_rows}
obs={
 'M_d':md['SMH']['d_AUROC']-md['SH']['d_AUROC'],
 'M_F':md['SMH']['F_AUROC']-md['SH']['F_AUROC'],
 'M_known':md['SMH']['known_correct']-md['SH']['known_correct'],
 'M_full_d':md['FULL']['d_AUROC']-md['SHVB']['d_AUROC'],
 'M_full_F':md['FULL']['F_AUROC']-md['SHVB']['F_AUROC'],
 'V_F':md['FULL']['F_AUROC']-md['SMHB']['F_AUROC'],
 'V_d':md['FULL']['d_AUROC']-md['SMHB']['d_AUROC'],
 'V_known':md['FULL']['known_correct']-md['SMHB']['known_correct'],
}
obs['M_selectivity_d_minus_F']=obs['M_d']-obs['M_F']; obs['V_selectivity_F_minus_d']=obs['V_F']-obs['V_d']
(OUT/'observed_contrasts.json').write_text(json.dumps(obs,indent=2)+'\n')
print('OBS',obs,flush=True)

# Save permutation plans before fitting.
pm=derangements(len(pair_order),N_SHUFFLE,SEED+1); pv=derangements(len(pair_order),N_SHUFFLE,SEED+2)
np.savez_compressed(OUT/'shuffle_plans.npz',M=pm,V=pv,pairs=np.array(pair_order,dtype=object))

def one_shuffle(task_idx):
    task,idx=task_idx
    if task=='M':
        xd=shuffle_block(['S','H'],'M',pm[idx]); base=md['SH']; primary='d_AUROC'
    else:
        xd=shuffle_block(['S','M','H','B'],'V',pv[idx]); base=md['SMHB']; primary='F_AUROC'
    s,r,_=nested_from_xd(xd); mm=metrics(s,r)
    return dict(task=task,shuffle=idx,F_AUROC=mm['F_AUROC'],d_AUROC=mm['d_AUROC'],known_correct=mm['known_correct'],
                delta_F=mm['F_AUROC']-base['F_AUROC'],delta_d=mm['d_AUROC']-base['d_AUROC'],delta_known=mm['known_correct']-base['known_correct'])

tasks=[('M',i) for i in range(N_SHUFFLE)]+[('V',i) for i in range(N_SHUFFLE)]
workers=min(8,os.cpu_count() or 1)
print('SHUFFLES_START',len(tasks),'workers',workers,flush=True)
if 'fork' not in mp.get_all_start_methods():
    raise RuntimeError('Full mismatch refitting requires a POSIX fork environment; use Linux/WSL. Fixed-prediction replay is platform-independent.')
with mp.get_context('fork').Pool(workers) as pool:
    shrows=[]
    for k,row in enumerate(pool.imap_unordered(one_shuffle,tasks,chunksize=1),1):
        shrows.append(row)
        if k%10==0: print('SHUFFLES_DONE',k,'/',len(tasks),flush=True)
sh=pd.DataFrame(shrows).sort_values(['task','shuffle']); sh.to_csv(OUT/'shuffle_results.csv',index=False)
summary=[]
for task,field,real in [('M','delta_d',obs['M_d']),('V','delta_F',obs['V_F'])]:
    vals=sh.loc[sh.task==task,field].to_numpy()
    p=(1+np.sum(vals>=real-1e-15))/(len(vals)+1)
    summary.append(dict(task=task,primary=field,observed=real,null_mean=float(vals.mean()),null_sd=float(vals.std(ddof=1)),null_q025=float(np.quantile(vals,.025)),null_q975=float(np.quantile(vals,.975)),one_sided_p=float(p),n=len(vals)))
pd.DataFrame(summary).to_csv(OUT/'shuffle_summary.csv',index=False)
print('SHUFFLE_SUMMARY',summary,flush=True)

# Reuse the exact same generator as the verification-only fixed-score replay.
from recompute_source_statistics import recompute
recompute(OUT/'static_predictions.csv', OUT, OUT/'shuffle_results.csv', N_BOOT, True)
print('DONE: all primary, full-context and selectivity statistics regenerated.', flush=True)
