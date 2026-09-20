#!/usr/bin/env python3
"""Verify retained scores and regenerate all reported source/J statistics.
No training. Does not treat a stored sign check as statistical reproduction.
"""
from pathlib import Path
import importlib.util, tempfile, subprocess, sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from recompute_source_statistics import recompute
R=Path(__file__).resolve().parents[1]
CHECKS={}
def check(name,yes):
    CHECKS[name]=bool(yes)
    if not yes:raise AssertionError(name)

def equal_tables(new,stored,keys):
    a=pd.read_csv(new).set_index(keys).sort_index();b=pd.read_csv(stored).set_index(keys).sort_index()
    check(stored.name+'_index',a.index.equals(b.index))
    for col in a.columns:
        if col not in b.columns:continue
        if pd.api.types.is_numeric_dtype(a[col]):check(stored.name+'_'+col,np.allclose(a[col],b[col],atol=1e-11,rtol=0,equal_nan=True))
        else:check(stored.name+'_'+col,a[col].equals(b[col]))

cj=pd.read_csv(R/'results/joint_control/costanzo_J_predictions.csv')
j=pd.read_csv(R/'results/joint_control/jonikas_J_predictions.csv')
check('J_costanzo_from_scores_607',np.sum((cj.d>0)==(cj.label>0))==607)
check('J_jonikas_F_from_scores',abs(roc_auc_score(j.label,j.F)-.8756073858114674)<1e-12)
check('J_jonikas_d_from_scores',abs(roc_auc_score(j.label,j.d)-.8513119533527697)<1e-12)
p=pd.read_csv(R/'results/paper_source_predictions.csv');h=pd.read_csv(R/'results/hardening_static_predictions.csv')
check('ordered_keys_unique',not p.duplicated(['a','b']).any())
check('frozen_record_counts',len(p)==168 and p.pair.nunique()==84 and p.label.sum()==21)
for var,n in [('SH',18),('SMH',21),('SMHB',21),('FULL',21),('SHVB',20),('SMHV',20)]:
    check(var+'_direction_count',np.sum((p[var+'_d']>0)&(p.label==1))==n)
    check(var+'_odd_identity',np.allclose(p[var+'_d'],(p[var+'_F']-p[var+'_reverse'])/2,atol=1e-13,rtol=0))
    if var in ['SH','SMH','SMHB','FULL','SHVB']:
        for metric in ['F','d','reverse']:check(var+'_'+metric+'_ledgers_aligned',np.allclose(p[var+'_'+metric],h[var+'_'+metric],atol=2e-13,rtol=0))
check('FULL_J_prediction_agreement',np.allclose(h.FULL_F,j.F,atol=1e-10,rtol=0))
# Source invariance and stored mismatch plans.
z=np.load(R/'data/jonikas/KEGG.npz');M=[8,9,12]
check('M_exchange_invariant',np.array_equal(z['xd'][:,M],z['xdr'][:,M]))
plans=np.load(R/'results/hardening_shuffle_plans.npz',allow_pickle=True)
for source in ['M','V']:
    ar=plans[source];check(source+'_100_derangements',ar.shape==(100,84) and np.all(ar!=np.arange(84)[None,:]) and all(np.array_equal(np.sort(row),np.arange(84)) for row in ar))
with tempfile.TemporaryDirectory(prefix='pad_source_verify_') as t:
    tmp=Path(t)
    recompute(R/'results/hardening_static_predictions.csv',tmp,R/'results/hardening_shuffle_results.csv')
    for filename in ['hardening_bootstrap_summary.csv','hardening_M_full_bootstrap_summary.csv','hardening_selectivity_summary.csv']:
        equal_tables(tmp/filename,R/'results'/filename,['resampling','contrast'])
    for filename in ['hardening_gene_leave_one_out.csv','hardening_gene_leave_one_out_M_full.csv']:
        equal_tables(tmp/filename,R/'results'/filename,['gene'])
    equal_tables(tmp/'hardening_gene_leave_one_out_summary.csv',R/'results/hardening_gene_leave_one_out_summary.csv',['contrast'])
    equal_tables(tmp/'hardening_shuffle_summary.csv',R/'results/hardening_shuffle_summary.csv',['task'])
    a=json.loads((tmp/'hardening_observed_contrasts.json').read_text());b=json.loads((R/'results/hardening_observed_contrasts.json').read_text())
    check('all_observed_contrasts_regenerated',a.keys()==b.keys() and all(abs(a[k]-b[k])<1e-12 for k in a))
    subprocess.run([sys.executable,str(R/'code/recompute_J_intervals.py'),'--pad',str(R/'reference/PaD_jonikas_predictions.csv'),'--joint',str(R/'results/joint_control/jonikas_J_predictions.csv'),'--out',str(tmp/'J')],check=True)
    equal_tables(tmp/'J/paired_J_intervals.csv',R/'results/paired_J_intervals.csv',['metric','resampling'])
summary=pd.read_csv(R/'results/hardening_selectivity_summary.csv')
for name in ['M_full_d_minus_F','combined_selectivity']:
    row=summary[(summary.resampling=='pair')&(summary.contrast==name)].iloc[0]
    check(name+'_pair_range_spans_zero',row.q025<0<row.q975)
loo=pd.read_csv(R/'results/hardening_gene_leave_one_out.csv')
check('V_DIE2_limitation',loo.loc[loo.gene=='DIE2','V_F'].iloc[0]<0 and int((loo.V_F>0).sum())==29)
print(json.dumps({'passed':all(CHECKS.values()),'checks':len(CHECKS),'scope':'fixed-score numeric regeneration; not new training or statistical validation'},indent=2))
if '--report' in sys.argv:
    out=Path(sys.argv[sys.argv.index('--report')+1]);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'passed':all(CHECKS.values()),'checks':CHECKS},indent=2)+'\n')
