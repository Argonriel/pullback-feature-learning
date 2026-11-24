"""Outcome-neutral serving benchmark: load frozen tables once into memory.

The original frozen experiment timed archive-backed NpzFile lookups, including
repeated array decompression. Both measurements are retained and distinguished.
"""
from pathlib import Path
import csv,json,time,hashlib,datetime
import numpy as np
from threadpoolctl import threadpool_limits
import study

ROOT=Path(__file__).resolve().parent
METHODS={'base_independent':('base','independent'),'base_joint':('base','joint'),'uniform_independent':('uniform','independent'),'uniform_joint':('uniform','joint'),'adaptive_joint':('adaptive','joint')}
CFG=dict(cohort='fresh',epsilon=.1,n_models=16,batches_per_measurement=8,batch_size=256,warmup_blocks=2,measurement_blocks=8,randomization_seed=8660000,blas_threads=1)

def savecsv(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=ROOT/'serving';out.mkdir(exist_ok=False);dump(out/'benchmark_protocol.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),config=CFG,source_sha256=sha(__file__),policy_source_sha256=sha(ROOT/'study.py'),outcome_neutral=True,scientific_policy_changed=False))
    rng=np.random.default_rng(CFG['randomization_seed']);rows=[];audits=[];loads=[]
    commit=json.loads((ROOT/'fresh/calibration/selector_commit.json').read_text())
    for pair in range(1,17):
        theta=study.model('fresh',pair);x=study.sample(study.CONFIG['fresh_test_seed']+pair,32768)[:2048];ties=np.random.default_rng(study.CONFIG['fresh_tie_seed']+pair).random(2048)
        tables={};archives={}
        for grid in ['base','uniform','adaptive']:
            p=ROOT/f'fresh/calibration/{pair}_{grid}.npz';assert sha(p)==commit['npz_sha256'][p.name]
            t=time.perf_counter();archives[grid]=np.load(p);tables[grid]={k:archives[grid][k] for k in archives[grid].files}
            loads.append(dict(pair=pair,grid=grid,load_seconds=time.perf_counter()-t,array_bytes=sum(a.nbytes for a in tables[grid].values()),archive_bytes=p.stat().st_size))
        for name,(grid,kind) in METHODS.items():
            outputs=[];masks=[];allcalls=[]
            for table in [archives[grid],tables[grid]]:
                calls=[]
                def predict(z):calls.append(len(z));return study.core.representation(theta,z)[:,0]
                result,mask=study.deploy(theta,x[:256],table,.1,kind,ties[:256],predict);outputs.append(result);masks.append(mask);allcalls.append(calls)
            assert np.array_equal(outputs[0],outputs[1]) and np.array_equal(masks[0],masks[1]) and allcalls[0]==allcalls[1]
            audits.append(dict(pair=pair,method=name,array_equal=True,mask_equal=True,archive_calls=allcalls[0],memory_calls=allcalls[1]))
        def run(name):
            for k in range(8):
                z=x[k*256:(k+1)*256];tt=ties[k*256:(k+1)*256]
                if name in METHODS:
                    grid,kind=METHODS[name];result,mask=study.deploy(theta,z,tables[grid],.1,kind,tt)
                else:
                    a=study.core.representation(theta,z)[:,0]
                    if name=='pointwise_oracle':
                        b=study.core.representation(theta,-z)[:,0];mask=study.allocate(study.objective(a,b,.1),tt);result=np.where(mask,.5*(a+b),a)
        names=['raw']+list(METHODS)+['pointwise_oracle']
        for block in range(CFG['warmup_blocks']+CFG['measurement_blocks']):
            order=rng.permutation(names)
            for position,name in enumerate(order):
                t=time.perf_counter_ns();run(name);elapsed=(time.perf_counter_ns()-t)/8/1e6
                if block>=CFG['warmup_blocks']:rows.append(dict(pair=pair,method=name,block=block-CFG['warmup_blocks'],position=position,ms_per_256=elapsed))
        for archive in archives.values():archive.close()
    summary=[]
    for method in names:
        models=np.array([np.median([r['ms_per_256'] for r in rows if r['pair']==i and r['method']==method]) for i in range(1,17)])
        summary.append(dict(method=method,n_models=16,measurement_blocks_per_model=8,median_of_model_medians_ms_per_256=float(np.median(models)),mean_of_model_medians_ms_per_256=float(models.mean()),min_model_median=float(models.min()),max_model_median=float(models.max())))
    savecsv(out/'timings.csv',rows);savecsv(out/'summary.csv',summary);savecsv(out/'table_loading.csv',loads);dump(out/'equivalence_checks.json',audits)
    dump(out/'benchmark_complete.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),n_equivalence_checks=len(audits),all_checks_pass=True,config=CFG,summary=summary,csv_sha256={p.name:sha(p) for p in out.glob('*.csv')}))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
