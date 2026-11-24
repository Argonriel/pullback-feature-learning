"""Reconstruct every affine bound and stress outcome independently."""
from pathlib import Path
import csv,datetime,hashlib,json,math,sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'robustness'
sys.path.insert(0,str(ROOT/'training'));import model
from adapters import load_variant,policy_paths
sys.path.insert(0,str(ROOT/'training/reference'));import baseline
def read(name):return list(csv.DictReader((HERE/name).open()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
bound_rows=read('certificate_bounds.csv');stress_rows=read('stress_results.csv');costs=read('certificate_costs.csv')
max_bound=max_stress=0.;checks=0;extra_calls=0;replays=[]
with threadpool_limits(limits=1):
 for commit,folder in [('CERTIFICATE_COMMIT.json','certificate'),('STRESS_COMMIT.json','stress')]:
    for path,h in json.loads((HERE/commit).read_text())['manifest'].items():assert sha(HERE/folder/path)==h;checks+=1
 for cost in costs:
    pair=int(cost['pair']);variant=cost['variant'];a,theta,meta=load_variant(pair,variant);paths=policy_paths(pair,variant)
    grid=dict(np.load(paths['grid']));fold=dict(np.load(paths['folded']));c=float(np.clip(np.quantile(grid['folded_jensen'],.95),1e-6,.5))
    L=1.;C=0.
    for A,B,b in model.task.unpack(theta):L*=1+.5*np.linalg.norm(A,2)*np.linalg.norm(B,2);C+=.5*abs(B[0]).sum()
    D=meta['beta']*min(1.4+C,1.4*L);data=dict(np.load(HERE/'certificate'/f'{pair}_{variant}_batches.npz'))
    for row in [r for r in bound_rows if int(r['pair'])==pair and r['variant']==variant]:
        n=int(row['n_batches']);name=row['policy'];A=data[name+'_A'][:n];B=data[name+'_B'][:n];delta=.05/512
        za=math.sqrt(2*A.var(ddof=1)*math.log(2/delta)/n)+7*c/4*math.log(2/delta)/(3*(n-1))
        zb=math.sqrt(2*B.var(ddof=1)*math.log(2/delta)/n)+7*D/4*math.log(2/delta)/(3*(n-1))
        scale=math.sqrt(math.log(1/delta)/(2*n*256))
        low=max(0,A.mean()-za,A.mean()-c*scale);up=min(D/4,B.mean()+zb,B.mean()+D*scale)
        values=dict(c=c,D=D,A_eb_lower=A.mean()-za,A_mcd_lower=A.mean()-c*scale,
            B_eb_upper=B.mean()+zb,B_mcd_upper=B.mean()+D*scale,A_lower=low,B_upper=up,epsilon_critical=min(1,2*low/up))
        for key,val in values.items():
            err=abs(float(val)-float(row[key]));max_bound=max(max_bound,err);assert err<2e-13,(pair,variant,key,err)
    if pair in (1,8,16):
        x=model.task.sample_inputs(2121000+pair,2048*256);ties=np.random.default_rng(2122000+pair).random(len(x))
        assert ash(x)==cost['input_sha256'] and ash(ties)==cost['tie_sha256']
        x=x[:128*256];ties=ties[:len(x)];ms=[baseline.select(a,x,grid,ties),baseline.select_folded(a,x,fold,ties)];union=ms[0]|ms[1]
        h=np.zeros(len(x));t=h.copy();h[union]=a.predict(theta,x[union]);t[union]=a.predict(theta,a.partner(x[union]));calls=2*int(union.sum());extra_calls+=calls
        J=np.maximum((np.logaddexp(0,h)+np.logaddexp(0,t))/2-np.logaddexp(0,(h+t)/2),0);d=abs(h-t)/2
        for name,mask in zip(('symmetric_jensen','folded'),ms):
            assert np.max(abs((mask*np.minimum(J,c)).reshape(-1,256).mean(1)-data[name+'_A'][:128]))<1e-14
            assert np.max(abs((mask*d).reshape(-1,256).mean(1)-data[name+'_B'][:128]))<1e-14
        assert calls==data['forward_examples'][:128].sum();replays.append(dict(pair=pair,variant=variant,batches=128,forwards=calls))
    saved=dict(np.load(HERE/'stress'/f'{pair}_{variant}.npz'));x=model.task.sample_inputs(2124000+pair,65536);ties=np.random.default_rng(2125000+pair).random(65536)
    p0=model.task.posterior(x);assert np.array_equal(saved['p0'],p0)
    h=saved['first'];t=saved['partner'];m=(h+t)/2;d=(h-t)/2
    qcos=x[:,0]/np.linalg.norm(x[:,:2],axis=1);assert np.array_equal(qcos,saved['qcos'])
    masks=np.unpackbits(saved['masks'],axis=1)[:,:65536];other=np.unpackbits(saved['confidence_partner_mask'])[:65536]
    newm=[np.zeros(65536,bool),np.ones(65536,bool),baseline.select(a,x,grid,ties),baseline.select_folded(a,x,fold,ties),baseline.allocate(-abs(h),ties)]
    assert np.array_equal(np.stack(newm),masks) and np.array_equal(other,baseline.allocate(-abs(t),ties))
    cache={}
    for j,name in enumerate(saved['policy_names']):
        out=np.where(masks[j],m,h);out_t=np.where(other if name=='confidence' else masks[j],m,t)
        cache[str(name)]=(out,out_t,np.logaddexp(0,out),np.logaddexp(0,out_t))
    sp_h=np.logaddexp(0,h);sp_t=np.logaddexp(0,t)
    for row in [r for r in stress_rows if int(r['pair'])==pair and r['variant']==variant]:
        eps=float(row['epsilon']);q=qcos if row['world']=='cosine' else np.sign(d)
        amp=np.minimum(eps/2,np.minimum(p0,1-p0));p=p0+amp*q;pt=p0-amp*q
        out,out_t,sp,sp_ot=cache[row['policy']];loss=sp-p*out;loss_t=sp_ot-pt*out_t
        gain=sp_h-p*h-loss;gain_t=sp_t-pt*t-loss_t
        err=np.where(out>=0,1-p,p);err_t=np.where(out_t>=0,1-pt,pt)
        values=dict(expected_nll=loss.mean(),expected_error=err.mean(),gain=gain.mean(),
             antithetic_nll=(loss.mean()+loss_t.mean())/2,antithetic_error=(err.mean()+err_t.mean())/2,
             antithetic_gain=(gain.mean()+gain_t.mean())/2,observed_posterior_gap=max(abs(p-pt)))
        for key,v in values.items():
            error=abs(float(v)-float(row[key]));max_stress=max(max_stress,error);assert error<2e-12,(pair,variant,key,error)
assert len(bound_rows)==128 and len(stress_rows)==2560
primary=[r for r in bound_rows if int(r['n_batches'])==2048]
positive=sum(float(r['A_lower'])-.05*float(r['B_upper'])>0 for r in primary);assert positive==64
ledger=read('certificate_ledger.csv');assert len(ledger)==32*2048
for r in ledger:assert 64<=int(r['union_selected'])<=128 and int(r['forwards'])==2*int(r['union_selected'])
assert sum(int(r['forwards']) for r in ledger)==sum(int(r['forwards']) for r in costs)
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',certificate_rows=128,
 elementary_bounds=512,max_bound_error=max_bound,stress_rows=2560,max_stress_metric_error=max_stress,
 committed_files=checks,primary_lines_positive_at_epsilon_point1=positive,
 minimum_primary_critical=min(float(r['epsilon_critical']) for r in primary),certificate_batch_rows=len(ledger),
 actual_certificate_forwards=sum(int(r['forwards']) for r in ledger),certificate_replays=replays,
 additional_audit_forward_examples=extra_calls)
(ROOT/'root_audit/ROBUSTNESS_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
