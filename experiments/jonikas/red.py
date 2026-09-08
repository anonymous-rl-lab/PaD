#!/usr/bin/env python3
"""Author Red reproduction; Python-2-to-3 syntax changes only in vendor code."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from paths import DATA, run_path
ROOT=run_path('jonikas')
from red_py3 import Red

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=42);args=ap.parse_args()
    t=time.monotonic();d=loadmat(DATA/'jonikas/080930a_DM_data.mat')
    genes=[str(x[0]) for x in d['qnames_out'][0]];G=d['DM_array_merged'];H=np.nan_to_num(d['DM_Hill_merged']);S=np.nan_to_num(d['qindex_out'][:,1,None]);idx={g:i for i,g in enumerate(genes)}
    np.random.seed(args.seed)
    with threadpool_limits(limits=1):
        m=Red(G.copy(),S.copy(),H.copy(),genes)
        m.order(rank=100,lambda_u=1e-4,lambda_v=1e-4,alpha=.1,beta=.1,max_iter=200)
    Ghat=m._g(m.U.T@m.V)
    # Preserve the official release's asymmetric downstream score implementation.
    np.savez_compressed(ROOT/f'red_seed{args.seed}.npz',genes=genes,Ghat=Ghat,U=m.U,V=m.V,ga=m._ga,gb=m._gb,gc=m._gc,down_u=m._linear_u_downstream,down_v=m._linear_v_downstream)
    rows=[];metrics=[]
    for panel,files in [('KEGG',['KEGG_ordered.txt','KEGG_nonordered.txt']),('glycans',['N-linked-glycans_positive.txt','N-linked-glycans_negative.txt'])]:
        out=[]
        for y,f in zip([1,0],files):
            for line in (DATA/'jonikas'/f).read_text().splitlines():
                a,b=line.split();i,j=idx[a],idx[b];s1,s2=float(S[i,0]),float(S[j,0]);z=float(G[i,j]);h=float(H[i,j])
                du=abs(Ghat[i,j]-s1);dv=abs(Ghat[i,j]-s2)
                dp=(2*expit(-dv))/(2*expit(-dv)+2*expit(-du))
                p=float(m.epistatic_to(b,a));rev=float(m.epistatic_to(a,b))
                out.append(dict(panel=panel,a=a,b=b,label=y,unordered_pair='|'.join(sorted([a,b])),single_a=s1,single_b=s2,double=z,expected_no_interaction=h,n_measurements=int(d['num_measurements'][i,j]),double_observed=bool(np.isfinite(z)),red_released=p,red_reverse=rev,red_exchange_error=abs(p+rev-1),red_point_distance=dp,raw_mask=abs(z-s1)-abs(z-s2) if np.isfinite(z) else 0.,single_only=s1-s2))
        tab=pd.DataFrame(out);rows.extend(out)
        for name in ['red_released','red_point_distance','raw_mask','single_only']:
            metrics.append(dict(panel=panel,model=name,n=len(tab),n_positive=int(tab.label.sum()),n_true_AB=int(tab.double_observed.sum()),auroc=float(roc_auc_score(tab.label,tab[name])),average_precision=float(average_precision_score(tab.label,tab[name]))))
    pd.DataFrame(rows).to_csv(ROOT/f'red_seed{args.seed}_pairs.csv',index=False)
    res=dict(seed=args.seed,seconds=time.monotonic()-t,n_genes=len(genes),n_observed_unordered_AB=int(np.isfinite(G[np.triu_indices(len(G),1)]).sum()),fro_error=float(m.fro_error),nrmse=float(m.nrmse),matrix_asymmetry=float(np.nanmax(abs(G-G.T))),max_release_probability_exchange_error=max(r['red_exchange_error'] for r in rows),metrics=metrics)
    (ROOT/f'red_seed{args.seed}_summary.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2),flush=True)
if __name__=='__main__':main()
