"""Rebuild the Red panel inputs without changing prepared PaD inputs."""
import csv,io,json,hashlib,time,zipfile
import numpy as np
import pandas as pd
from paths import DATA,run_path

def prepare(archive=None):
    start=time.monotonic();panel=pd.read_csv(DATA/'prepared/Costanzo.csv')
    genes=sorted(set(panel.a)|set(panel.b));keep=set(genes)
    source_hash='05adf2aa309e5336cd1c3045eb033a2bc7b24e6f249d57a0cc25549ec6772522'
    if archive is None:
        raw=pd.read_csv(DATA/'costanzo/red_raw_within_panel.csv.gz',float_precision='round_trip')
        n=None
    else:
        h=hashlib.sha256()
        with archive.open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        if h.hexdigest()!=source_hash:raise ValueError('Official archive SHA256 mismatch.')
        rows=[];n=0
        with zipfile.ZipFile(archive) as z:
            member=next(x for x in z.namelist() if x.endswith('/SGA_NxN.txt'))
            with io.TextIOWrapper(z.open(member),encoding='utf-8-sig') as f:
                reader=csv.reader(f,delimiter='\t');next(reader)
                for row in reader:
                    n+=1;a,b=row[1].upper(),row[3].upper()
                    if a in keep and b in keep and row[4]=='DMA30':
                        rows.append((a,b,float(row[7]),float(row[8]),float(row[9]),float(row[5]),float(row[10])))
        raw=pd.DataFrame(rows,columns=['a','b','single_a','single_b','double','epsilon','sd'])
    out=run_path('costanzo')/'prepared';out.mkdir(exist_ok=True)
    single=pd.concat([raw[['a','single_a']].rename(columns={'a':'gene','single_a':'s'}),raw[['b','single_b']].rename(columns={'b':'gene','single_b':'s'})]).groupby('gene').s.median()
    S=single.reindex(genes).to_numpy();missing=~np.isfinite(S);S[missing]=1.
    raw['u']=raw[['a','b']].min(axis=1);raw['v']=raw[['a','b']].max(axis=1)
    vals=raw.groupby(['u','v']).double.median();index={g:i for i,g in enumerate(genes)}
    G=np.full((len(genes),len(genes)),np.nan)
    for (u,v),v0 in vals.items():G[index[u],index[v]]=G[index[v],index[u]]=v0
    np.fill_diagonal(G,np.nan);H=np.outer(S,S)
    np.savez_compressed(out/'red_input.npz',genes=np.array(genes),G=G,S=S[:,None],H=H)
    original=np.load(DATA/'costanzo/red_input.npz')
    errors={}
    for key,value in [('G',G),('S',S[:,None]),('H',H)]:
        assert np.array_equal(np.isnan(value),np.isnan(original[key]))
        errors[key]=float(np.nanmax(abs(value-original[key])))
        assert errors[key]<1e-12,(key,errors[key])
    assert genes==original['genes'].tolist()
    audit=dict(genes=len(genes),source_rows_scanned=n,selected_raw_rows=len(raw),
        missing_single_imputed_wildtype=list(np.array(genes)[missing]),max_matrix_errors=errors,
        original_archive_used=archive is not None,seconds=time.monotonic()-start)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2));return audit
