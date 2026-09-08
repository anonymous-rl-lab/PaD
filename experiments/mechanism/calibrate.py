#!/usr/bin/env python3
"""Two raw-noise levels: independent scale, selection and confirmation samples.

The rule below is fixed before executing any calibration or formal tests.
Select the two largest prespecified relative scales with simultaneous upper
symbol-error sum <= .05. Confirm exactly these levels on independent streams;
do not replace a level following failed confirmation. B1 sets absolute scales,
which are reused unchanged in B2. No learner is fitted during calibration.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import beta,norm
from threadpoolctl import threadpool_limits
import engine as e

OUT=e.ROOT/'calibration'

def upper(k,n,alpha):
    return 1. if k==n else float(beta.ppf(1-alpha,k+1,n-k))

def assess(stage,relative,std,n,alpha):
    level='relative_'+format(relative,'.6g')
    E,G,m,au=e.raw('calibration-'+stage,'decode','B1',17,n,level,std)
    z,d=e.encode(E,G,m);wrong_rank=d['error_T']|d['bad_Q']|(abs(d['decoded_Q']-m['Q'])>.1)
    row=dict(stage=stage,relative_scale=relative,omega_P=std[0],omega_D=std[1],n=n,
             count_T=int(d['error_T'].sum()),count_Q=int(d['error_Q'].sum()),
             epsilon_T_upper=upper(int(d['error_T'].sum()),n,alpha),
             epsilon_Q_upper=upper(int(d['error_Q'].sum()),n,alpha),
             numerical_bad_Q=int(d['bad_Q'].sum()),numerical_bad_T=int(d['bad_T'].sum()))
    for y,label in [(1,'positive'),(-1,'negative')]:
        ii=m['Y']==y;k=int(wrong_rank[ii].sum());nn=int(ii.sum())
        row['rank_bad_'+label]=k;row['n_'+label]=nn;row['zeta_'+label+'_upper']=upper(k,nn,alpha)
    eps=row['epsilon_T_upper']+row['epsilon_Q_upper'];zz=row['zeta_positive_upper']+row['zeta_negative_upper']
    er=e.theoretical('B1')['additive_error_lower']-norm.cdf(-2)-eps
    ac=norm.cdf(1.9*np.sqrt(2))-e.theoretical('B1')['additive_auc_upper']-zz
    row.update(epsilon_sum_upper=eps,zeta_sum_upper=zz,classification_gap_lower=float(er),
               auc_gap_lower=float(ac),classification_budget_met=bool(eps<=.05),auc_budget_met=bool(zz<=.01),
               per_event_one_sided_alpha=alpha,stream=au['stream'])
    rec=pd.DataFrame(dict(instance_id=m['instance_id'],Y=m['Y'],X=m['X'],Q=m['Q'],
                         error_T=d['error_T'],error_Q=d['error_Q'],rank_bad=wrong_rank,
                         bad_Q=d['bad_Q'],saturation_P=d['saturation_P'],saturation_D=d['saturation_D'],
                         weak_expression=d['weak_expression'],near_threshold=d['near_threshold'],low_readout=d['low_readout']))
    rec.to_csv(OUT/f'{stage}_{level}_records.csv',index=False)
    tails=[]
    for col in ['weak_expression','near_threshold','low_readout','bad_Q','saturation_P','saturation_D']:
        ix=rec[col].to_numpy();tails.append(dict(group=col,n=int(ix.sum()),error_T=int(d['error_T'][ix].sum()),error_Q=int(d['error_Q'][ix].sum())))
    e.dump(OUT/f'{stage}_{level}_tails.json',tails)
    print('CALIBRATION',stage,relative,'eps_upper',round(eps,6),'class_gap',round(er,6),'auc_gap',round(ac,6),flush=True)
    return row

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    p=e.PROTOCOL;grid=p['calibration_grid_relative_scales']
    rule=dict(selection='two largest grid entries whose simultaneous upper symbol-error sum is <= 0.05',
              grid=grid,selection_alpha=.05,selection_event_count=5*4,
              confirmation_alpha=.05,confirmation_event_count=2*4,
              primary_budget='classification',secondary_budget='AUROC at t=0.1; not used to select noise',
              failed_confirmation='retain the selected level and report failure; no reselection',
              scale_source='independent B1 raw arrays, global RMS over n x 2 x 7 entries',
              transfer_to_B2='same absolute standard deviations, no B2 rescaling',
              tail_handling='no exclusions; nonfinite decodes conservatively count as errors',
              registered_before_results=True)
    e.dump(OUT/'calibration_rule.json',rule)
    E,G,m,a=e.raw('calibration-scale','raw-RMS','B1',17,p['calibration_scale_n'])
    ps=float(np.sqrt(np.mean(E*E)));ds=float(np.sqrt(np.mean(G*G)))
    scale=dict(P_RMS=ps,D_RMS=ds,source_audit=a,
               per_instance_P_RMS_quantiles=np.quantile(np.sqrt(np.mean(E*E,axis=(1,2))),[0,.01,.05,.5,.95,.99,1]),
               per_instance_D_RMS_quantiles=np.quantile(np.sqrt(np.mean(G*G,axis=(1,2))),[0,.01,.05,.5,.95,.99,1]))
    e.dump(OUT/'frozen_scales.json',scale)
    rows=[]
    for q in grid:rows.append(assess('select',q,(q*ps,q*ds),p['calibration_selection_n'],.05/20))
    selected=[r for r in rows if r['classification_budget_met']][-2:]
    levels=[dict(id='noise'+str(i+1),relative_scale=r['relative_scale'],omega_P=r['omega_P'],omega_D=r['omega_D']) for i,r in enumerate(selected)]
    e.dump(OUT/'selected_levels_before_confirmation.json',dict(levels=levels,selection_rows=rows))
    for lev in levels:
        r=assess('confirm',lev['relative_scale'],(lev['omega_P'],lev['omega_D']),p['calibration_confirmation_n'],.05/8)
        rows.append(r);lev['classification_budget_confirmed']=r['classification_budget_met'];lev['auc_budget_confirmed']=r['auc_budget_met']
    pd.DataFrame(rows).to_csv(OUT/'scale_error_guarantee.csv',index=False)
    result=dict(levels=levels,rule=rule,scale=scale,seconds=time.monotonic()-start,
        encoder_records=len(grid)*p['calibration_selection_n']+len(levels)*p['calibration_confirmation_n'],
        selection_confirmation_independent=True,formal_tests_used=False)
    e.dump(OUT/'result.json',result);print('CALIBRATION COMPLETE',result['seconds'],'LEVELS',levels,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
