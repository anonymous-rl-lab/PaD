"""Build analysis tables exclusively from current generated fits and frozen inputs."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from paths import run_path
from kernel import Data

def build():
    status={}
    for name,ds in [('costanzo','Costanzo'),('jonikas','KEGG')]:
        root=run_path(name);d=Data(ds);y=d.y;methods={'P_frozen':d.z['p']};missing=[]
        for family in ['PaD','A_match','A_rbf']:
            path=root/f'{family}_predictions.csv'
            if not path.exists():missing.append(family);continue
            p=pd.read_csv(path);assert list(zip(p.a,p.b))==list(zip(d.g.a,d.g.b))
            methods[family]=p.direction.to_numpy() if d.directional else p.score.to_numpy()
        for family in ['svm','xgboost','catboost']:
            seeds=([17] if family=='svm' else [17,29,43]) if name=='costanzo' else ([17,29,43] if family=='catboost' else [17])
            for seed in seeds:
                paths=sorted(root.glob(f'{family}_{seed}_'+('fold*.json' if name=='costanzo' else 'outer*.json')))
                method=f'{family}_{seed}'
                if len(paths)!=len(d.groups):missing.append(method);continue
                s=np.full(d.n,np.nan);covered=np.zeros(d.n,int)
                for path in paths:
                    r=json.loads(path.read_text());ix=r['test_indices'];s[ix]=r['score'];covered[ix]+=1
                assert np.all(covered==1) and np.isfinite(s).all();methods[method]=s
        for seed in ([17,29,43] if name=='costanzo' else [42,43,44]):
            if name=='costanzo':
                path=root/f'red_{seed}.json'
                if path.exists():
                    r=json.loads(path.read_text());methods[f'Red_{seed}']=np.array(r['score']);methods[f'Red_released_onesided_{seed}']=np.array(r['released_forward'])-.5
                else:missing.append(f'Red_{seed}')
            else:
                path=root/f'red_seed{seed}_pairs.csv'
                if path.exists():
                    p=pd.read_csv(path);p=p[p.panel=='KEGG']
                    assert list(zip(p.a,p.b))==list(zip(d.g.a,d.g.b));methods[f'Red_{seed}']=p.red_released.to_numpy()
                else:missing.append(f'Red_{seed}')
        pairs=d.g[['a','b','pair']].copy();pairs['target']=y;rows=[]
        for method,s in methods.items():
            pairs[method]=s
            row=dict(method=method,n=d.n,positive=int(y.sum()),auroc=float(roc_auc_score(y,s)),average_precision=float(average_precision_score(y,s)))
            if d.directional:row.update(correct=int(((s>=0)==(y>0)).sum()),accuracy=float(((s>=0)==(y>0)).mean()))
            rows.append(row)
        pairs.to_csv(root/'pair_predictions.csv',index=False);pd.DataFrame(rows).to_csv(root/'method_summary.csv',index=False)
        status[name]=dict(completed_methods=list(methods),missing_methods=missing,complete=not missing)
    root=run_path('mechanism');rows=[]
    for path in sorted((root/'formal').glob('*/result.json')):
        r=json.loads(path.read_text())
        for row in r['results']:rows.append(dict(scenario=r['scenario'],seed=r['seed'],level=r['level'],**row))
    if rows:
        tab=pd.DataFrame(rows);tab.to_csv(root/'formal_summary.csv',index=False)
        tab.groupby(['scenario','level','family'])[['error_rate','auroc']].agg(['mean','min','max']).to_csv(root/'formal_aggregate.csv')
    status['mechanism']=dict(completed_flows=len(rows),scheduled_flows=120,complete=len(rows)==120,reserved_certification_family=30)
    root=run_path('background');rows=[]
    for path in sorted(root.glob('q*_result.json')):
        r=json.loads(path.read_text());rows.append({k:v for k,v in r.items() if k!='ledger'})
    if rows:pd.DataFrame(rows).to_csv(root/'summary.csv',index=False)
    status['background']=dict(completed_fits=len(rows),scheduled_fits=6,complete=len(rows)==6)
    out=run_path('analysis');(out/'completion.json').write_text(json.dumps(status,indent=2)+'\n')
    return status
