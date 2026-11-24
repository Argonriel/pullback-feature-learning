"""Independent moment/bound/truth reconstruction and fresh-stream replay."""
from pathlib import Path
from types import SimpleNamespace
import csv,datetime,hashlib,importlib.util,json,math
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent.parent;B=ROOT/'structured'
read=lambda p:list(csv.DictReader(p.open(newline='')))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ash=lambda a:hashlib.sha256(np.asarray(a).tobytes()).hexdigest()

def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
task=load('audit_structured_task',ROOT/'root_audit/reference/task.py')
policy=load('audit_structured_policy',B/'reference/baseline.py')
adapter=SimpleNamespace(quotient=task.quotient)

def forward(theta,x):
    z=x
    for offset in (0,72):
        a=theta[offset:offset+32].reshape(8,4);b=theta[offset+32:offset+64].reshape(4,8);bias=theta[offset+64:offset+72]
        z=z+np.tanh(z@a.T+bias)@b.T/2
    return z[:,0]

def sample(seed,n,a):
    rng=np.random.default_rng(seed);r=rng.uniform(.4,1.4,(n,2));angles=[];got=0
    while got<n:
        k=max(1024,int((n-got)*(1+abs(a))*1.03)+8);phi=rng.uniform(0,2*np.pi,k);u=rng.random(k)
        keep=phi[u<=(1+a*np.cos(phi))/(1+abs(a))];angles.append(keep);got+=len(keep)
    phi=np.concatenate(angles)[:n]
    return np.stack([r[:,0]*np.cos(phi),r[:,0]*np.sin(phi),r[:,1]*np.cos(2*phi),r[:,1]*np.sin(2*phi)],axis=1)

def nll(h,p):return np.logaddexp(0,h)-p*h

def main():
    assert (B/'STRESS_COMMIT.json').exists()
    delta=.05/(8*4*2*5*2*2+4);assert delta==.05/1284
    constants={int(r['pair']):r for r in json.loads((B/'CONSTANTS.json').read_text())['records']}
    hist=json.loads((B/'HISTORICAL_INPUTS.json').read_text())['models']
    mm=read(B/'moment_bounds.csv');curves=read(B/'certificate_curves.csv');stress=read(B/'stress_results.csv')
    momentmap={(int(r['pair']),float(r['a']),r['policy'],int(r['n_batches']),r['moment']):r for r in mm}
    stresmap={(int(r['pair']),float(r['a']),r['world'],float(r['epsilon']),r['policy']):r for r in stress}
    param=read(B/'parameter_intervals.csv');pr=np.load(B/'parameter_raw.npz');max_m=max_b=max_s=max_forward=0.;extra=0
    for j,a in enumerate((0.,.01,.1,.6)):
        x=sample(2338000+j,65536,a);u=x[:,0]/np.linalg.norm(x[:,:2],axis=1)
        assert np.array_equal(u,pr[f'cosine_{j}']);r=param[j];assert ash(x)==r['input_sha256']
        est=2*u.mean();rad=np.sqrt(8*np.log(2/delta)/65536)
        assert max(abs(est-float(r['estimate'])),abs(rad-float(r['radius'])),abs(np.clip(est-rad,0,1)-float(r['lower'])),abs(np.clip(est+rad,0,1)-float(r['upper'])))<1e-14
    costs=read(B/'certificate_costs.csv');costmap={(int(r['pair']),float(r['a'])):r for r in costs}
    for r in hist:
        pair=r['pair'];theta=np.load(B/r['theta_path']);grid=dict(np.load(B/r['grid_path']));fold=dict(np.load(B/r['folded_path']))
        # Deterministic parameter bounds independently assembled.
        lip=1.;amplitude=0.
        for offset in (0,72):
            a=theta[offset:offset+32].reshape(8,4);b=theta[offset+32:offset+64].reshape(4,8)
            lip*=1+.5*np.linalg.svd(b,compute_uv=False)[0]*np.linalg.svd(a,compute_uv=False)[0]
            amplitude+=.5*np.abs(b[0]).sum()
        D=min(1.4+amplitude,1.4*lip);c=min(.5,max(1e-6,float(np.quantile(grid['folded_jensen'],.95))))
        assert abs(D-constants[pair]['D'])<1e-12 and c==constants[pair]['c']
        for j,a in enumerate((0.,.01,.1,.6)):
            dat=dict(np.load(B/'certificate'/f'{pair}_{j}.npz'));n=2048*256
            mask=np.unpackbits(dat['packed_masks'],axis=1)[:,:n].astype(bool);ids=dat['union_ids'];u=dat['cosphi'];h=dat['first'];ht=dat['partner']
            assert np.array_equal(ids,np.flatnonzero(mask.any(0))) and np.all(mask.reshape(2,-1,256).sum(2)==64)
            J=(np.logaddexp(0,h)+np.logaddexp(0,ht))/2-np.logaddexp(0,(h+ht)/2)
            d=(h-ht)/2;S=abs(d);t=(np.logaddexp(0,h)-np.logaddexp(0,ht))/2-d/2
            W=abs(u[ids])*S/2-u[ids]*t;alo=float(param[j]['lower']);ahi=float(param[j]['upper'])
            values=dict(A=np.maximum(0,np.minimum(J,c)),W=W,B=S/2,R_known=(1-a*abs(u[ids]))*S/2,R_unknown=(1-alo*abs(u[ids]))*S/2)
            for pi,name in enumerate(['symmetric_jensen','folded']):
                for key,val in values.items():
                    dense=np.zeros(n);dense[ids]=val;raw=(dense*mask[pi]).reshape(-1,256).mean(1)
                    e=float(np.max(abs(raw-dat[name+'_'+key])));max_m=max(max_m,e);assert e<2e-12
                    R=c if key=='A' else D if key=='W' else D/2
                    for look in (512,2048):
                        v=raw[:look];mean=v.mean();var=v.var(ddof=1)
                        eb=math.sqrt(2*var*math.log(2/delta)/look)+7*(R/4)*math.log(2/delta)/(3*(look-1))
                        bd=R*math.sqrt(math.log(1/delta)/(2*look*256))
                        result=max(0,mean-eb,mean-bd) if key=='A' else min(R/4,mean+eb,mean+bd)
                        saved=momentmap[pair,a,name,look,key]
                        for field,val in [('mean',mean),('variance',var),('eb_radius',eb),('mcd_radius',bd),('bound',result)]:
                            e=abs(val-float(saved[field]));max_b=max(max_b,e);assert e<2e-12
            counts=mask.any(0).reshape(-1,256).sum(1)
            assert np.array_equal(dat['forwards'],2*counts) and 2*counts.sum()==int(costmap[pair,a]['forward_examples'])
            # Reconstruct all input streams; independent forwards/masks for 16-batch prefix.
            x=sample(2331000+100*j+pair,n,a);ties=np.random.default_rng(2332000+100*j+pair).random(n)
            assert ash(x)==costmap[pair,a]['input_sha256'] and ash(ties)==costmap[pair,a]['tie_sha256']
            assert np.array_equal(u,x[:,0]/np.linalg.norm(x[:,:2],axis=1))
            limit=4096;xx=x[:limit];tt=ties[:limit]
            for pi,fn,table in [(0,policy.select,grid),(1,policy.select_folded,fold)]:
                m=fn(adapter,xx,table,tt);assert np.array_equal(m,mask[pi,:limit])
                for coordinate in (0,17,63,255):
                    altered=xx.copy();altered[coordinate,:2]*=-1;assert np.array_equal(m,fn(adapter,altered,table,tt))
            keep=ids<limit;xx=x[ids[keep]];xt=xx.copy();xt[:,:2]*=-1
            for z,saved in [(forward(theta,xx),h[keep]),(forward(theta,xt),ht[keep])]:
                e=float(np.max(abs(z-saved)));max_forward=max(max_forward,e);assert e<2e-12;extra+=len(z)
            truth=dict(np.load(B/'stress'/f'{pair}_{j}.npz'));h=truth['first'];ht=truth['partner'];p0=truth['p0'];u=truth['cosphi'];mask=np.unpackbits(truth['packed_masks'],axis=1)[:,:65536].astype(bool)
            d=(h-ht)/2;m=(h+ht)/2;J=(np.logaddexp(0,h)+np.logaddexp(0,ht))/2-np.logaddexp(0,m)
            for world in ('cosine','adversarial','orbit_worst'):
                for eps in (0.,.025,.05,.1,.2,.4,.8,1.):
                    if world=='orbit_worst':
                        avg=.5+.5*(1-eps)*np.sign(u*d);p=avg+eps*np.sign(d)/2;pt=avg-eps*np.sign(d)/2
                    else:
                        q=u if world=='cosine' else np.sign(d);amp=np.minimum(eps/2,np.minimum(p0,1-p0));p=p0+amp*q;pt=p0-amp*q
                    for pi,name in enumerate(['raw','full','symmetric_jensen','folded']):
                        out=np.where(mask[pi],m,h);los=nll(out,p);g=nll(h,p)-los;gt=nll(ht,pt)-nll(np.where(mask[pi],m,ht),pt)
                        corrected=(g+gt)/2+a*u*(g-gt)/2;gb=g.reshape(-1,256).mean(1);cb=corrected.reshape(-1,256).mean(1)
                        vals=dict(expected_nll=los.mean(),expected_error=np.where(out>=0,1-p,p).mean(),direct_gain=g.mean(),known_density_gain=corrected.mean(),unweighted_paired_gain=((g+gt)/2).mean(),density_correction=(a*u*(g-gt)/2).mean(),direct_batch_se=gb.std(ddof=1)/16,known_density_batch_se=cb.std(ddof=1)/16)
                        report=stresmap[pair,a,world,eps,name]
                        for key,val in vals.items():
                            e=abs(float(report[key])-val);max_s=max(max_s,e);assert e<2e-12
    for r in curves:
        pair=int(r['pair']);a=float(r['a']);name=r['policy'];n=int(r['n_batches']);eps=float(r['epsilon'])
        vals={k:float(momentmap[pair,a,name,n,k]['bound']) for k in ['A','W','B','R_known','R_unknown']}
        j=[0.,.01,.1,.6].index(a);ahi=float(param[j]['upper'])
        results=dict(known_lower=vals['A']-a*vals['W']-eps*vals['R_known'],estimated_lower=vals['A']-ahi*vals['W']-eps*vals['R_unknown'],old_lower=vals['A']-eps*vals['B']-constants[pair]['D']*2*a/np.pi)
        for key,val in results.items():assert abs(val-float(r[key]))<2e-12
    ledger=read(B/'certificate_ledger.csv');assert len(ledger)==65536
    total=sum(int(r['forward_examples']) for r in ledger);assert total==sum(int(r['forward_examples']) for r in costs)
    for r in ledger:assert int(r['forward_examples'])==2*int(r['union_selected']) and 64<=int(r['union_selected'])<=128
    counts=[]
    for a in [0.,.01,.1,.6]:
        subset=[r for r in curves if float(r['a'])==a and int(r['n_batches'])==2048 and float(r['epsilon'])==0]
        counts.append(dict(a=a,lines=len(subset),**{k+'_positive':sum(float(r[k])>0 for r in subset) for k in ['known_lower','estimated_lower','old_lower']}))
    out=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),moment_rows=len(mm),elementary_moment_events=2*len(mm),parameter_intervals=4,
        curve_rows=len(curves),stress_rows=len(stress),max_moment_error=max_m,max_bound_error=max_b,max_truth_metric_error=max_s,
        exact_input_stream_reconstructions=32,model_policy_prefix_batches_per_law=16,max_prefix_forward_error=max_forward,
        mask_discrepancies=0,actual_certificate_forwards=total,additional_audit_forward_examples=extra,
        primary_epsilon0_counts=counts,proof_review='Sharp posterior simplex optimization, conditional orbit identity, independent parameter CI and simultaneous union bound checked by root; finite enumeration in PREFLIGHT.json.',
        limitation='Model replay uses certificate prefixes; stress metrics reconstructed from all retained logits. No natural-image certificate or distribution-family membership test.')
    (ROOT/'root_audit/STRUCTURED_RESULTS.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
