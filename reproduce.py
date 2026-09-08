#!/usr/bin/env python3
"""Single entry point for PaD release verification and reproduction."""
import argparse,importlib.util,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src/pad'),str(ROOT/'experiments/mechanism'),str(ROOT/'third_party/red'),str(ROOT/'verification')]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'runs',help='Generated outputs only; defaults to ./runs')
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Recompute all delivered metrics, certificates, splits and deterministic encoder checks')
    sub.add_parser('report',help='Summarize newly generated outputs and list incomplete result blocks')
    sub.add_parser('smoke',help='Run interface checks and small actual fits for all three studies')
    mech=sub.add_parser('mechanism',help='Train the frozen four learners on mechanism data')
    mech.add_argument('--scenario',choices=['all','B1','B2','B3','B4','B5','B6'],default='all')
    mech.add_argument('--seed',type=int,choices=[17,29,43]);mech.add_argument('--level',choices=['all','ideal','noise1','noise2'],default='all')
    sub.add_parser('calibrate',help='Repeat independent noise selection and confirmation; no model fitting')
    real=sub.add_parser('yeast',help='Refit PaD and nonlinear additive controls from prepared inputs')
    real.add_argument('--dataset',choices=['all','costanzo','jonikas'],default='all')
    real.add_argument('--method',choices=['all','PaD','A_match','A_rbf'],default='all')
    ext=sub.add_parser('external',help='Repeat complete external nested comparisons (CPU intensive)')
    ext.add_argument('--dataset',choices=['costanzo','jonikas'],required=True)
    ext.add_argument('--family',choices=['all','svm','xgboost','catboost'],default='all')
    ext.add_argument('--seed',type=int);ext.add_argument('--workers',type=int,default=4)
    red=sub.add_parser('red',help='Refit the released Red implementation')
    red.add_argument('--dataset',choices=['costanzo','jonikas'],required=True);red.add_argument('--seed',type=int)
    bg=sub.add_parser('background',help='Refit all three candidate-pair background availability levels')
    bg.add_argument('--q',type=float,choices=[0.,.5,1.]);bg.add_argument('--family',choices=['joint','A_match'])
    prep=sub.add_parser('prepare-costanzo',help='Rebuild Red matrices from bundled raw panel records, or an official archive')
    prep.add_argument('--archive',type=Path)
    args=ap.parse_args();os.environ['PAD_OUTPUT_ROOT']=str(args.output.resolve())
    # Keep all numerical routines single-threaded; external Jonikas may use worker processes.
    import numpy
    import scipy.linalg
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if args.command in ['verify','smoke']:
            import checks
            result=checks.verify() if args.command=='verify' else checks.smoke()
            print(json.dumps(result,indent=2));return
        if args.command=='report':
            from reports import build
            print(json.dumps(build(),indent=2));return
        if args.command=='calibrate':
            import calibrate
            calibrate.main();return
        if args.command=='mechanism':
            import engine as e
            from paths import REFERENCE
            calibration=json.loads((REFERENCE/'mechanism/calibration/result.json').read_text())
            noise={'ideal':(0.,0.)}|{v['id']:(v['omega_P'],v['omega_D']) for v in calibration['levels']}
            scenarios=list(e.SCENARIOS) if args.scenario=='all' else [args.scenario]
            seeds=e.PROTOCOL['seeds'] if args.seed is None else [args.seed]
            levels=list(noise) if args.level=='all' else [args.level]
            completed=[]
            for level in levels:
                for scenario in scenarios:
                    if level!='ideal' and scenario not in e.PROTOCOL['noise_scenarios']:continue
                    for seed in seeds:
                        completed.append(e.run_block('formal',scenario,seed,512,4096,level,noise[level]))
            if not completed:raise ValueError('Noise experiments are prescribed for B1 and B2 only.')
            print('Completed',4*len(completed),'training-selection flows; reserved certification family M=30.');return
        if args.command=='yeast':
            from nested import train
            names={'costanzo':'Costanzo','jonikas':'KEGG'}
            for name in (names if args.dataset=='all' else [args.dataset]):
                for method in (['PaD','A_match','A_rbf'] if args.method=='all' else [args.method]):train(names[name],method)
            return
        if args.command=='external':
            m=module('external_'+args.dataset,ROOT/f'experiments/{args.dataset}/external.py')
            if args.dataset=='jonikas':
                d,_=m.data();outer,sets=m.split_plan(d)
                m.write('split_plan.json',dict(outer=outer,unique_inner_training_indices=sets))
            for family in (['svm','xgboost','catboost'] if args.family=='all' else [args.family]):
                prescribed=([17] if family=='svm' else [17,29,43]) if args.dataset=='costanzo' else ([17,29,43] if family=='catboost' else [17])
                seeds=prescribed if args.seed is None else [args.seed]
                if any(s not in prescribed for s in seeds):raise ValueError('Seed outside the frozen experiment schedule.')
                for seed in seeds:
                    if args.dataset=='costanzo':m.nested(family,seed,list(range(5)))
                    else:m.nested(family,seed,list(range(84)),args.workers)
            return
        if args.command=='red':
            prescribed=[17,29,43] if args.dataset=='costanzo' else [42,43,44]
            seeds=prescribed if args.seed is None else [args.seed]
            if any(s not in prescribed for s in seeds):raise ValueError('Seed outside the frozen Red schedule.')
            for seed in seeds:
                if args.dataset=='costanzo':module('external_costanzo',ROOT/'experiments/costanzo/external.py').red(seed)
                else:
                    m=module('red_jonikas',ROOT/'experiments/jonikas/red.py')
                    sys.argv=[str(ROOT/'experiments/jonikas/red.py'),'--seed',str(seed)];m.main()
            return
        if args.command=='background':
            import background as b
            rows=[]
            for q in ([1.,.5,0.] if args.q is None else [args.q]):
                for family in (['joint','A_match'] if args.family is None else [args.family]):rows.append(b.reduce(q,family))
            if len(rows)==6:b.summarize(rows)
            return
        if args.command=='prepare-costanzo':
            module('prepare_costanzo',ROOT/'experiments/costanzo/prepare.py').prepare(args.archive)

if __name__=='__main__':main()
