#!/usr/bin/env python3
"""Recompute every source-statistics summary from frozen out-of-fold scores.

No fitting. Endpoint-sum and unordered-pair weights use the original seeds.
Percentile ranges are descriptive fixed-prediction sensitivities, not general
network-valid confidence intervals or corrections for development selection.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = {
    'M_d': ('SMH_d', 'SH_d'), 'M_F': ('SMH_F', 'SH_F'),
    'V_F': ('FULL_F', 'SMHB_F'), 'V_d': ('FULL_d', 'SMHB_d'),
    'M_full_d': ('FULL_d', 'SHVB_d'), 'M_full_F': ('FULL_F', 'SHVB_F'),
}
PRIMARY = ['M_d','M_F','V_F','V_d']
SELECTIVITY = ['M_full_d_minus_F','V_F_minus_d','combined_selectivity']

def concordance(s, pos, neg):
    d = s[pos, None] - s[None, neg]
    return (d > 0).astype(float) + .5 * (d == 0)

def matrices(frame):
    y=frame.label.to_numpy(); pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    if len(pos)==0 or len(neg)==0: raise ValueError('Both classes are required.')
    cache={c:concordance(frame[c].to_numpy(),pos,neg) for ab in CONTRASTS.values() for c in ab}
    ds={k:cache[a]-cache[b] for k,(a,b) in CONTRASTS.items()}
    ds['M_full_d_minus_F']=ds['M_full_d']-ds['M_full_F']
    ds['V_F_minus_d']=ds['V_F']-ds['V_d']
    ds['combined_selectivity']=ds['M_full_d_minus_F']+ds['V_F_minus_d']
    return pos,neg,ds

def summarize(arr, label, names, include_n=True):
    out=[]
    for name in names:
        x=arr[name].to_numpy(); q=np.quantile(x,[.025,.5,.975])
        row={'resampling':label,'contrast':name}
        if include_n: row['n']=len(x)
        row.update(mean=float(x.mean()),q025=float(q[0]),q50=float(q[1]),q975=float(q[2]),positive_fraction=float(np.mean(x>0)))
        out.append(row)
    return out

def recompute(pred_path, out, shuffle_path=None, n=10000, save_draws=False):
    out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(pred_path)
    if p.duplicated(['a','b']).any(): raise ValueError('Duplicate ordered endpoint keys.')
    if len(p)!=168: raise ValueError('Expected the frozen 168-record development panel.')
    pos,neg,ds=matrices(p)
    genes=sorted(set(p.a)|set(p.b)); gidx={g:i for i,g in enumerate(genes)}
    pairs=list(dict.fromkeys(p.pair)); pidx={g:i for i,g in enumerate(pairs)}
    ai=p.a.map(gidx).to_numpy(); bi=p.b.map(gidx).to_numpy(); pi=p.pair.map(pidx).to_numpy()
    prim=[]; full=[]; sel=[]; provenance={'n_draws':n,'inputs':{pred_path.name:hashlib.sha256(pred_path.read_bytes()).hexdigest()},'schemes':{},'scope':'same development panel; fixed predictions; no refitting or independent validation; percentile ranges are descriptive','legacy_selectivity_name':'combined_double_dissociation is now called combined_selectivity; numerical definition unchanged'}
    for scheme,seed,units in [('gene',20260923,len(genes)),('pair',20260924,len(pairs))]:
        rng=np.random.default_rng(seed); draws=rng.integers(0,units,size=(n,units))
        cnt=np.zeros((n,units),dtype=np.int64);np.add.at(cnt,(np.arange(n)[:,None],draws),1)
        w=cnt[:,ai]+cnt[:,bi] if scheme=='gene' else cnt[:,pi]
        wp=w[:,pos];wn=w[:,neg];den=wp.sum(1)*wn.sum(1);ok=den>0
        vals={name:np.einsum('bi,ij,bj->b',wp,d,wn,optimize=True)[ok]/den[ok] for name,d in ds.items()}
        arr=pd.DataFrame(vals)
        prim.extend(summarize(arr,'gene_endpoint_cluster' if scheme=='gene' else 'unordered_pair_cluster',PRIMARY))
        full.extend(summarize(arr,scheme,['M_full_d','M_full_F']))
        sel.extend(summarize(arr,'gene_endpoint' if scheme=='gene' else 'pair',SELECTIVITY,False))
        provenance['schemes'][scheme]={'seed':seed,'units':units,'valid_draws':int(ok.sum()),'weights':'endpoint multiplicity(a) + multiplicity(b)' if scheme=='gene' else 'shared multiplicity of unordered pair'}
        if save_draws:arr.to_csv(out/f'{scheme}_all_bootstrap_draws.csv',index=False)
    pd.DataFrame(prim).to_csv(out/'hardening_bootstrap_summary.csv',index=False)
    pd.DataFrame(full).to_csv(out/'hardening_M_full_bootstrap_summary.csv',index=False)
    pd.DataFrame(sel).to_csv(out/'hardening_selectivity_summary.csv',index=False)
    rows=[]
    for gene in genes:
        keep=~((p.a==gene)|(p.b==gene)).to_numpy();wp=keep[pos].astype(float);wn=keep[neg].astype(float);den=wp.sum()*wn.sum()
        if den==0:continue
        row={'gene':gene,'n':int(keep.sum()),'n_pos':int(wp.sum())}
        row.update({name:float(wp@d@wn/den) for name,d in ds.items()});rows.append(row)
    loo=pd.DataFrame(rows)
    loo[['gene','n','n_pos']+PRIMARY].to_csv(out/'hardening_gene_leave_one_out.csv',index=False)
    loo[['gene','n','n_pos','M_full_d','M_full_F']].to_csv(out/'hardening_gene_leave_one_out_M_full.csv',index=False)
    los=[]
    for name in CONTRASTS:
        x=loo[name].to_numpy();los.append(dict(contrast=name,min=float(x.min()),median=float(np.median(x)),max=float(x.max()),positive_all=bool(np.all(x>0)),n=len(x)))
    pd.DataFrame(los).to_csv(out/'hardening_gene_leave_one_out_summary.csv',index=False)
    obs={name:float(d.mean()) for name,d in ds.items()}
    obs['M_known']=int(((p.SMH_d>0)&(p.label==1)).sum()-((p.SH_d>0)&(p.label==1)).sum())
    obs['V_known']=int(((p.FULL_d>0)&(p.label==1)).sum()-((p.SMHB_d>0)&(p.label==1)).sum())
    obs['M_full_known']=int(((p.FULL_d>0)&(p.label==1)).sum()-((p.SHVB_d>0)&(p.label==1)).sum())
    (out/'hardening_observed_contrasts.json').write_text(json.dumps(obs,indent=2)+'\n')
    if shuffle_path is not None:
        sh=pd.read_csv(shuffle_path);ss=[]
        for task,field,actual in [('M','delta_d',obs['M_d']),('V','delta_F',obs['V_F'])]:
            vals=sh.loc[sh.task==task,field].to_numpy();exceed=int(np.sum(vals>=actual-1e-15));q=np.quantile(vals,[.025,.975])
            ss.append(dict(task=task,primary=field,observed=actual,mismatch_mean=float(vals.mean()),mismatch_sd=float(vals.std(ddof=1)),mismatch_q025=float(q[0]),mismatch_q975=float(q[1]),n_exceed=exceed,n=len(vals),plus_one_exceedance_fraction=(1+exceed)/(len(vals)+1),interpretation='pair-orbit block mismatch; not a conditional-independence p-value'))
        pd.DataFrame(ss).to_csv(out/'hardening_shuffle_summary.csv',index=False)
        provenance['inputs'][shuffle_path.name]=hashlib.sha256(shuffle_path.read_bytes()).hexdigest()
    (out/'statistics_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(full).to_string(index=False));print(pd.DataFrame(sel).to_string(index=False))
    return provenance

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--predictions',type=Path,default=ROOT/'results/hardening_static_predictions.csv')
    a.add_argument('--shuffles',type=Path,default=ROOT/'results/hardening_shuffle_results.csv')
    a.add_argument('--out',type=Path,default=ROOT/'rerun_results/source_statistics')
    a.add_argument('--draws',type=int,default=10000);a.add_argument('--save-draws',action='store_true')
    x=a.parse_args()
    if x.draws<1:raise ValueError('--draws must be positive.')
    recompute(x.predictions,x.out,x.shuffles,x.draws,x.save_draws)
if __name__=='__main__':main()
