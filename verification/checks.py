"""Independent rescore of all reference predictions plus deterministic interface checks."""
import ast,gzip,hashlib,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import norm
from sklearn.metrics import roc_auc_score,average_precision_score
from paths import REPO,DATA,REFERENCE,run_path
import kernel as k
import engine as e

def close(a,b,tol=1e-10):
    assert abs(float(a)-float(b))<=tol,(a,b)

def frame(path):return pd.read_csv(path,float_precision='round_trip')

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def integrity():
    path=REPO/'MANIFEST_SHA256.json';count=0
    if path.exists():
        manifest=json.loads(path.read_text())
        for rel,h in manifest['files'].items():
            assert hashlib.sha256((REPO/rel).read_bytes()).hexdigest()==h,rel
            count+=1
    mapping=json.loads((REPO/'docs/SOURCE_MAP.json').read_text());copied=0
    for entry in mapping['copies']:
        p=REPO/entry['destination'];raw=gzip.decompress(p.read_bytes()) if 'uncompressed_sha256' in entry else p.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==entry.get('uncompressed_sha256',entry.get('sha256')),p
        copied+=1
    adapted=0
    for entry in mapping.get('adapted_files',[]):
        # Preserve the archival source hash separately from the corrected file.
        assert entry['source_sha256'] and entry['change']
        p=REPO/entry['destination']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['release_sha256'],p
        adapted+=1
    # Path-independent scientific functions/classes were extracted without changing their ASTs.
    expected={ (v['source'],v['symbol']):v['ast_sha256'] for v in mapping['extracted_symbols'] }
    checks=[('src/pad/kernel.py','01_Mechanism_120/Proposer_Learning_Experiments_v1/vendor/unified.py',['rbf','rms','metric','fit']),
            ('src/pad/selection.py','01_Mechanism_120/Proposer_Learning_Experiments_v1/vendor/stable_selection.py',['paired_se','select_from_inner']),
            ('src/pad/features.py','01_Mechanism_120/Proposer_Learning_Experiments_v1/vendor/yeast_core.py',['FeatureBuilder'])]
    verified=0
    for file,source,names in checks:
        tree=ast.parse((REPO/file).read_text())
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
                assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==expected[(source,node.name)]
                verified+=1
    assert verified==7
    return dict(manifest_files=count,source_files_losslessly_preserved=copied,documented_adaptations=adapted,unchanged_core_ASTs=verified)

def mechanism():
    correction=math.sqrt(math.log(2*30/.05)/(2*4096));lower=.5*norm.cdf(-.5/math.sqrt(.75))
    blocks=sorted((REFERENCE/'mechanism/formal').iterdir());assert len(blocks)==30
    cert=[];allmetrics=[];eq=[];encoded_cases=0
    seen=set()
    for path in blocks:
        r=json.loads((path/'result.json').read_text());p=frame(path/'test_predictions.csv.gz')
        assert len(p)==r['n_test']==4096 and p.instance_id.nunique()==4096
        assert r['n_train']==512 and not seen.intersection(p.instance_id)
        seen.update(p.instance_id)
        for row in r['results']:
            family=row['family'];s=p[family+'_score'].to_numpy();rev=p[family+'_reverse'].to_numpy()
            y=(p.Y.to_numpy()>0).astype(float);wrong=((s-rev)/2>=0)!=(y>0)
            assert np.array_equal(wrong,p[family+'_wrong'].to_numpy())
            assert int(wrong.sum())==row['errors']
            close(roc_auc_score(y,s),row['auroc']);close(average_precision_score(y,s),row['average_precision'])
            close(wrong.mean(),row['error_rate'])
            if family=='joint':
                upper=min(1,wrong.mean()+correction);close(upper,row['risk_upper'])
                passed=bool(upper<row['additive_error_lower']);assert passed==row['classification_certified']
                if r['scenario']=='B1':
                    assert (int(wrong.sum())<=456)==passed
                    cert.append(dict(seed=r['seed'],level=r['level'],errors=int(wrong.sum()),risk_upper=upper,passed=passed))
            allmetrics.append(dict(scenario=r['scenario'],seed=r['seed'],level=r['level'],family=family,error_rate=float(wrong.mean())))
        if r['scenario']=='B2' and r['level']=='ideal':
            actual=e.equivalence(p.joint_wrong.to_numpy(),p.full_P_RBF_wrong.to_numpy())
            assert actual==r['ideal_B2_equivalence'];assert actual['practical_equivalence'];eq.append(actual)
        E,G,meta,audit=e.raw('formal','test',r['scenario'],r['seed'],4096,r['level'],tuple(r['noise_std']))
        assert np.array_equal(meta['instance_id'],p.instance_id.to_numpy())
        for index in [0,1,31,255,511,1023,2047,4095]:
            score,xp,xd,rawp=e.encode_one(*E[index],*G[index])
            close(score,p.raw_P_score.iloc[index],1e-9);encoded_cases+=1
        assert r['ledger']['candidate_fold_evaluations']==550
    assert len(allmetrics)==120 and len(cert)==9 and all(v['passed'] for v in cert)
    result=dict(blocks=len(blocks),training_selection_flows=len(allmetrics),independent_test_instances=len(seen),
        B1_certificates=cert,correction=correction,additive_lower=float(lower),
        minimum_certified_margin=float(lower-max(v['risk_upper'] for v in cert)),B2_equivalence=eq,
        regenerated_raw_encoded_cases=encoded_cases,candidate_inner_evaluations=16500)
    return result

def calibration():
    from scipy.stats import beta
    r=json.loads((REFERENCE/'mechanism/calibration/result.json').read_text());checks=[]
    for level in r['levels']:
        p=frame(REFERENCE/f'mechanism/calibration/confirm_relative_{level["relative_scale"]}_records.csv.gz')
        assert len(p)==16384 and p.instance_id.nunique()==16384
        counts={c:int(p[c].sum()) for c in ['error_T','error_Q','rank_bad']}
        def upper(count):return 1. if count==len(p) else float(beta.ppf(1-.05/8,count+1,len(p)-count))
        ub=upper(counts['error_T'])+upper(counts['error_Q'])
        assert ub<=.05 and level['classification_budget_confirmed']
        assert not level['auc_budget_confirmed']
        checks.append(dict(level=level['id'],n=len(p),counts=counts,classification_error_budget_upper=ub))
    return checks

def yeast():
    result={}
    for name,ds,count in [('costanzo','Costanzo',657),('jonikas','KEGG',168)]:
        p=frame(REFERENCE/name/'pair_predictions.csv');d=k.Data(ds);y=d.y
        assert len(p)==count and list(zip(p.a,p.b))==list(zip(d.g.a,d.g.b))
        for key,tr in d.allowed.items():
            te=d.groups[key];assert not (set(d.g.a.iloc[te])|set(d.g.b.iloc[te])) & (set(d.g.a.iloc[tr])|set(d.g.b.iloc[tr]))
        summary=frame(REFERENCE/name/'method_summary.csv')
        for row in summary.to_dict('records'):
            method=row['method'];s=p[method]
            close(roc_auc_score(y,s),row['auroc'])
            if name=='costanzo':assert int(((s>=0)==(y>0)).sum())==row['correct']
            else:close(average_precision_score(y,s),row['average_precision'])
        nselected=0
        for path in sorted((REFERENCE/name/'selected').glob('*.json')):
            r=json.loads(path.read_text());ix=r['test_indices'];method=r['family']+'_'+str(r['seed'])
            assert np.max(abs(p[method].to_numpy()[ix]-np.array(r['score'])))<1e-12
            if 'train_indices' in r:
                tr=r['train_indices'];assert not (set(d.g.a.iloc[ix])|set(d.g.b.iloc[ix])) & (set(d.g.a.iloc[tr])|set(d.g.b.iloc[tr]))
            nselected+=1
        assert nselected==(35 if name=='costanzo' else 420)
        if name=='costanzo':
            a=frame(REFERENCE/name/'internal_controls.csv')
            counts={m:int(((a[m+'_direction']>=0)==(a.target>0)).sum()) for m in ['V4','A_match','A_rbf']}
            assert counts==dict(V4=611,A_match=613,A_rbf=606)
            assert int(((p.P_frozen>=0)==(y>0)).sum())==597
            result[name]=dict(n=count,PaD_correct=611,nonlinear_controls=counts,selected_external_models=nselected)
        else:
            close(roc_auc_score(y,p.PD_frozen),.8811143505021056)
            result[name]=dict(n=count,positive=int(y.sum()),PaD_AUROC=roc_auc_score(y,p.PD_frozen),selected_external_models=nselected)
    return result

def background():
    rows=[]
    expected={100:(.8811143505021056,.8276643990929705),50:(.7120181405895691,.7324263038548753),0:(.8717201166180758,.8075801749271137)}
    for q in [100,50,0]:
        pair=[]
        for family in ['joint','A_match']:
            p=frame(REFERENCE/f'background/q{q}_{family}_predictions.csv')
            assert len(p)==168;pair.append(roc_auc_score(p.target,p.score))
        for a,b in zip(pair,expected[q]):close(a,b,1e-8)
        rows.append(dict(availability=q,PaD_AUROC=pair[0],A_match_AUROC=pair[1]))
    assert rows[1]['PaD_AUROC']<rows[1]['A_match_AUROC']
    return rows

def verify():
    import theory
    from regressions import check_readouts
    t=theory.smoke(theory.load());continuous=theory.continuous_gain(theory.load())
    report=dict(integrity=integrity(),readout_regressions=check_readouts(),mechanism=mechanism(),calibration=calibration(),yeast=yeast(),background=background(),
        encoder_checks=t,continuous_encoder_checks=continuous,
        scope='All delivered predictions rescored; raw encoder samples regenerated; no full model retraining by this command.')
    (run_path('verification')/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def smoke():
    # Actual four-learner fit on a separate smoke stream; no formal test reuse.
    r=e.run_block('release-smoke','B1',17,64,128)
    z,m,diag,a=e.dataset('release-smoke','direct-solver','B1',17,48)
    d=e.Data(z,pd.DataFrame(dict(a=m['instance_id'],b=m['instance_id'],pair=m['instance_id'])),'joint')
    tr=np.arange(32);theta=k.WEIGHTS[-1];lam=k.LAMBDAS[2]
    cc,rr=d.components(tr);K=sum(w*c for w,c in zip(theta,cc));R=sum(w*c for w,c in zip(theta,rr));K=(K-R)/2
    alpha=solve(K[np.ix_(tr,tr)]+lam*len(tr)*np.eye(len(tr)),2*d.y[tr]-1,assume_a='pos')
    actual=k.fit(d,tr,theta,[lam])[lam][0];err=float(np.max(abs(actual-K[:,tr]@alpha)));assert err<1e-8
    external=[]
    for name in ['costanzo','jonikas']:
        mod=import_file('external_'+name,REPO/f'experiments/{name}/external.py');data=mod.data();dd,X=data[:2]
        te=next(iter(dd.groups.values()));tr=next(iter(dd.allowed.values()))[:64]
        if len(set(dd.y[tr]))<2:tr=next(iter(dd.allowed.values()))
        for family in ['svm','xgboost','catboost']:
            if name=='costanzo':_,score=mod.train(family,mod.configs(family)[0],17,tr,X,data[2],dd.y,'release-smoke')
            else:score,info,_=mod.fit_one(family,mod.configs(family)[0],17,tr,X,dd.y,'release-smoke')
            assert np.isfinite(score).all();external.append(dict(dataset=name,family=family,finite=True))
    prep=import_file('prepare_costanzo',REPO/'experiments/costanzo/prepare.py').prepare()
    report=dict(mechanism_learners=len(r['results']),direct_solver_max_error=err,external_fits=external,raw_panel_preparation=prep)
    (run_path('verification')/'smoke.json').write_text(json.dumps(report,indent=2)+'\n');return report
