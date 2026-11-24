"""Independent reconstruction of final statistical bounds and six input streams."""
from pathlib import Path
import csv,datetime,hashlib,json,math,sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'validation'));sys.path.insert(0,str(ROOT/'method'))
from adapters import get_adapter
import baseline
def read(name):return list(csv.DictReader((ROOT/'certification'/name).open()))
def ash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
rows=read('final_per_model_bounds.csv');costs=read('costs.csv');audits=[];maximum=0.;replays=[]
delta=.05/864;allcalls=0
with threadpool_limits(limits=1):
 for cost in costs:
    c=cost['cohort'];i=int(cost['pair']);a=get_adapter(c)
    theta=np.load(ROOT/'validation/states'/f'{c}_{i}.npz')['pb']
    C=0.;L=1.
    for A,B,b in a.network.unpack(theta):
        C+=.5*np.abs(B[0]).sum();L*=1+.5*np.linalg.norm(B,2)*np.linalg.norm(A,2)
    R={'rings':3.,'waveform':1.,'two_invariants':1.4}[c];D=min(R+C,R*L)
    Jmax=float(np.logaddexp(D/2,-D/2)-np.log(2))
    tail=0.
    if c=='rings':
        tail=float(np.mean([.5*(3+C+.15**2/(3-r))*math.exp(-(3-r)**2/(2*.15**2)) for r in [.5,1.2]]))
    assert abs(Jmax-float(cost['gap_bound']))<1e-12 and abs(tail-float(cost['tail_upper']))<1e-40
    data=dict(np.load(ROOT/'certification/fresh'/f'{c}_{i}_batches.npz'))
    for row in [r for r in rows if r['cohort']==c and int(r['pair'])==i]:
        n=int(row['n_batches']);relative=row['claim']=='candidate_minus_folded'
        clip=.25 if c=='rings' else .5
        z=data['relative_truncated' if relative else row['claim']+'_clipped'][:n]
        width=Jmax/2 if relative else clip/4;sensitivity=2*Jmax if relative else clip
        correction=tail if relative else 0.
        eb=z.mean()-math.sqrt(2*np.var(z,ddof=1)*math.log(2/delta)/n)-7*width*math.log(2/delta)/(3*(n-1))-correction
        bd=z.mean()-sensitivity*math.sqrt(math.log(1/delta)/(2*n*256))-correction
        for key,v in [('empirical_bernstein_lower',eb),('bounded_difference_lower',bd),('lower',max(eb,bd))]:
            e=abs(v-float(row[key]));maximum=max(maximum,e);assert e<2e-14
        assert int(row['actual_shared_forward_examples'])==int(data['forward_examples'][:n].sum())
    if i in (1,16):
        x=a.sample_inputs(int(cost['input_seed']),2048*256)
        ties=np.random.default_rng(int(cost['tie_seed'])).random(len(x))
        assert ash(x)==cost['input_sha256'] and ash(ties)==cost['tie_sha256']
        x=x[:128*256];ties=ties[:len(x)]
        grid=dict(np.load(ROOT/'validation/policies'/f'{c}_{i}_grid.npz'))
        folded=dict(np.load(ROOT/'validation/policies'/f'{c}_{i}_folded.npz'))
        m=baseline.select(a,x,grid,ties);f=baseline.select_folded(a,x,folded,ties);union=m|f
        h=np.zeros(len(x));t=h.copy();h[union]=a.predict(theta,x[union]);t[union]=a.predict(theta,a.partner(x[union]));calls=2*int(union.sum());allcalls+=calls
        J=np.maximum((np.logaddexp(0,h)+np.logaddexp(0,t))/2-np.logaddexp(0,(h+t)/2),0)
        values={'symmetric_jensen_clipped':(m*np.minimum(J,clip)).reshape(-1,256).mean(1),
                'folded_clipped':(f*np.minimum(J,clip)).reshape(-1,256).mean(1),
                'relative_truncated':((m.astype(float)-f)*J*(np.linalg.norm(x,axis=1)<=3 if c=='rings' else 1)).reshape(-1,256).mean(1)}
        errors={k:float(np.max(abs(v-data[k][:128]))) for k,v in values.items()}
        assert max(errors.values())<1e-14
        assert calls==data['forward_examples'][:128].sum()
        replays.append(dict(cohort=c,pair=i,batches=128,extra_audit_forwards=calls,max_errors=errors))
    audits.append(dict(cohort=c,pair=i,Jmax=Jmax,tail=tail,bounds=9))
assert len(rows)==432 and len(costs)==48
assert sum(int(c['actual_forward_examples']) for c in costs)==14497086
result=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS',models=48,
    combined_rows=432,individual_bounds=864,maximum_bound_error=maximum,range_and_tail_recomputations=48,
    batch_replays=replays,additional_audit_forward_examples=allcalls,full_certificate_forwards=14497086,
    qualification='Independent numerical reconstruction, not directed-rounding machine verification.')
(ROOT/'root_audit/certificate_check.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
