"""Frozen inference over six training-seed pairs, plus crossed product resampling."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
from pathlib import Path
from datetime import datetime,timezone
import csv,json,math
import numpy as np

P=Path(__file__).resolve().parent
PAIRS=list(range(1,7));DATASETS=['holdout','myntra_seen','zappos_new']
SPECS=[('S-B','S','B'),('S-C','S','C'),('S-T','S','T'),('T-B','T','B')]
def write(name,rows):
    with (P/name).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def loss(z,y):
    m=z.max(1);return m+np.log(np.exp(z-m[:,None]).sum(1))-z[np.arange(len(y)),y]
def t5_cdf(t):
    theta=math.atan(t/math.sqrt(5))
    return .5+theta/math.pi+2*math.sin(2*theta)/(3*math.pi)+math.sin(4*theta)/(12*math.pi)
def t5_quantile(p):
    lo,hi=-1000.,1000.
    for _ in range(100):
        mid=(lo+hi)/2
        if t5_cdf(mid)<p:lo=mid
        else:hi=mid
    return (lo+hi)/2
def vectors(dataset,temperatures):
    methods={};common_y=None;common_ids=None;common_groups=None
    for row in temperatures:
        pair,arm=row['pair'],row['arm'];arr=np.load(P/'predictions'/f'{dataset}_{pair}_{arm}.npz')
        y=arr['labels'];h=arr['raw'].astype(np.float64)
        if common_y is None:common_y=y;common_ids=arr['indices'];common_groups=arr['groups']
        assert np.array_equal(y,common_y) and np.array_equal(arr['indices'],common_ids) and np.array_equal(arr['groups'],common_groups)
        zz={'B':h*row['raw']['beta'],'T':(h+arr['reflected'])*.5*row['full']['beta']} if arm=='teacher' else {('S' if arm=='student' else 'C'):h*row['raw']['beta']}
        for name,z in zz.items():methods[pair,name]=np.stack([loss(z,y),(z.argmax(1)!=y).astype(float)])
    matrix=np.stack([np.concatenate([methods[p,a]-methods[p,b] for _,a,b in SPECS],axis=0) for p in PAIRS])
    return methods,matrix,common_y,common_groups

def main():
    temps=json.loads((P/'temperatures.json').read_text());summary=[];contrasts=[];seedrows=[];primary={}
    critical_primary=t5_quantile(1-.05/2/2);critical_secondary=t5_quantile(.975)
    assert abs(t5_cdf(critical_primary)-.9875)<1e-12
    for dataset in DATASETS:
        methods,matrix,y,groups=vectors(dataset,temps)
        for method in 'BTSC':
            a=np.stack([methods[p,method].mean(1) for p in PAIRS]);summary.append(dict(dataset=dataset,method=method,
                ce=float(a[:,0].mean()),error=float(a[:,1].mean()),seed_sd_ce=float(a[:,0].std(ddof=1)),seed_sd_error=float(a[:,1].std(ddof=1))))
        seed_means=matrix.mean(2)
        for j,(name,aa,bb) in enumerate(SPECS):
            for k,metric in enumerate(['ce','error']):
                col=2*j+k;d=seed_means[:,col];mean=float(d.mean());sd=float(d.std(ddof=1));se=sd/math.sqrt(6)
                is_primary=dataset=='zappos_new' and name in ['S-B','S-C'] and metric=='error'
                critical=critical_primary if is_primary else critical_secondary
                row=dict(dataset=dataset,contrast=name,metric=metric,mean=mean,seed_sd=sd,seed_standard_error=se,
                    lower=mean-critical*se,upper=mean+critical*se,nominal_interval=.975 if is_primary else .95,
                    primary=is_primary,pairs=6,df=5,critical_t=critical,negative_seed_pairs=int(np.sum(d<0)),
                    inference_unit='paired training seed; fixed training dataset and evaluated population')
                contrasts.append(row)
                for pair,value in zip(PAIRS,d):seedrows.append(dict(dataset=dataset,pair=pair,contrast=name,metric=metric,difference=float(value)))
                if is_primary:primary[name]=row
        if dataset=='zappos_new':
            assert len(np.unique(groups))==len(groups)==1500
            rng=np.random.default_rng(2428000);boots=[];seedcounts=[];strata=[np.flatnonzero(y==c) for c in np.unique(y)]
            for _ in range(20000//50):
                sc=rng.multinomial(6,np.full(6,1/6),size=50);seedcounts.append(sc)
                mix=np.einsum('bs,smn->bmn',sc/6,matrix,optimize=True)
                counts=np.zeros((50,len(y)),dtype=np.int16)
                for ids in strata:counts[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=50)
                # One common product-index multiset per replicate, across every selected seed and method.
                boots.append(np.einsum('bn,bmn->bm',counts,mix,optimize=True)/len(y))
            boot=np.concatenate(boots);sc=np.concatenate(seedcounts)
            np.savez_compressed(P/'new_source_crossed_bootstrap.npz',values=boot,seed_counts=sc,
                names=np.array([n+'_'+metric for n,_,_ in SPECS for metric in ['ce','error']]))
            bootstrap_rows=[]
            for j,(name,_,_) in enumerate(SPECS):
                for k,metric in enumerate(['ce','error']):
                    adjusted=name in ['S-B','S-C'] and metric=='error';alpha=.025 if adjusted else .05
                    lo,hi=np.quantile(boot[:,j*2+k],[alpha/2,1-alpha/2]);bootstrap_rows.append(dict(dataset=dataset,contrast=name,metric=metric,
                        mean=float(seed_means[:,j*2+k].mean()),lower=float(lo),upper=float(hi),nominal_interval=1-alpha,
                        role='secondary crossed seed/product bootstrap; not a finite-sample guarantee',
                        common_product_indices_across_chosen_seeds_and_methods=True))
            write('crossed_bootstrap_contrasts.csv',bootstrap_rows)
    write('summary.csv',summary);write('seed_contrasts.csv',contrasts);write('per_seed_contrasts.csv',seedrows)
    checks={name:dict(mean_error_gain=-r['mean'],at_least_one_percentage_point_gain=r['mean']<=-.01,
        adjusted_seed_interval_upper_below_zero=r['upper']<0,improves_at_least_five_of_six=r['negative_seed_pairs']>=5,
        negative_seed_pairs=r['negative_seed_pairs'],upper=r['upper']) for name,r in primary.items()}
    success=all(v['at_least_one_percentage_point_gain'] and v['adjusted_seed_interval_upper_below_zero'] and v['improves_at_least_five_of_six'] for v in checks.values())
    record=dict(utc=datetime.now(timezone.utc).isoformat(),new_source_error_ordering_replication_success=success,criteria=checks,
        primary_comparisons=2,primary_interval_family='Bonferroni 2; 97.5% two-sided paired-seed Student t intervals',
        assumptions='Six independent training randomizations of one fixed training split; paired differences approximately normal. No guarantee at n=6.',
        crossed_product_bootstrap_role='Secondary sensitivity; resample six seed pairs and 500 unique product groups per class, common product draws across chosen seeds/methods.',
        no_training_dataset_resampling=True,brand_dependence_unresolved=True,serving_quality_criteria_separate=True)
    with (P/'SUCCESS_CRITERIA.json').open('x') as f:json.dump(record,f,indent=2)
    print(json.dumps(dict(summary=summary,success=record),indent=2))

if __name__=='__main__':main()
