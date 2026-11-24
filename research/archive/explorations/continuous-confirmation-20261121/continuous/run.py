"""Frozen continuous conditional-balance experiment and numerical diagnostics."""
from pathlib import Path
import sys,json,csv,hashlib,datetime,argparse,gzip,time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
import policy as p
sys.path.insert(0,str(ROOT.parent/'legacy'/'rings-mechanism-20261120'))
import polar
CONFIG=dict(old_pairs=list(range(1,17)),new_pairs=list(range(1,17)),nr=129,na=512,k=8,radial_max=3.,test_n=65536,cal_seed_base=835000,test_seed_base=836000,tie_seed_base=837000,group_seed_base=838000,label_seed_base=839000,bootstrap_seed=840000,bootstrap_repeats=10000,diagnostic_nr=2049,diagnostic_na=1024,diagnostic_phase=.013,refinement_na=2048,refinement_phase=.029)
POLICIES=['group_balanced','group_unbalanced','group_even','group_uniform','group_confidence','confidence','radius','folded','raw','full']

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(x):return hashlib.sha256(x).hexdigest()
def jo(f,x):
    if f.exists():raise FileExistsError(f)
    f.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def co(f,rows):
    with f.open('x',newline='',encoding='utf8') as stream:w=csv.DictWriter(stream,list(rows[0]));w.writeheader();w.writerows(rows)
def ci(f):
    with f.open(newline='',encoding='utf8') as stream:return list(csv.DictReader(stream))
def sample(seed,n):
    rng=np.random.default_rng(seed);r=rng.choice([.5,1.2],n);a=rng.uniform(0,2*np.pi,n);return r[:,None]*np.c_[np.cos(a),np.sin(a)]+rng.normal(0,.15,(n,2))
def density(r):return .5*(polar.class_density(r,.5)+polar.class_density(r,1.2))
def post(r):return np.exp(-np.logaddexp(0,-polar.bayes_logit(r)))
def nll(h,prob):return np.logaddexp(0,h)-prob*h
def trap(v,r):return float(np.sum((v[1:]+v[:-1])*.5*np.diff(r)))

def moments(table,r,na=1024,phase=.013):
    h,angles=p.orbit(table['theta'],r,na,phase);other=np.roll(h,na//2,axis=1);d=(h-other)/2;A=np.logaddexp(0,h)-np.logaddexp(0,(h+other)/2);psi=p.basis(angles)
    result={}
    for field in p.FIELDS:
        u=p.coefficients_at(table,r,field)@psi.T
        result[field+'_b']=(u*d).mean(axis=1)/8
        result[field+'_C']=((.25+u/8)*A).mean(axis=1)
        result[field+'_asymmetry']=abs(u-np.roll(u,na//2,axis=1)).mean(axis=1)
        result[field+'_mean_u']=u.mean(axis=1)
    return result

def oldpilot():
    out=ROOT/'old';out.mkdir(exist_ok=False);bank=np.load(ROOT.parent/'legacy'/'budgeted-rotation-final'/'results'/'models.npz');rows=[]
    for seed in CONFIG['old_pairs']:
        table=p.fit(bank[f'{seed}_pb']);np.savez_compressed(out/f'{seed}_policy.npz',**table)
        radii=(table['radii'][:-1]+table['radii'][1:])/2;m=moments(table,radii,1024,.019);a,b=p.bound_at(table,radii)
        rows.append(dict(seed=seed,state_sha256=sha(table['theta'].tobytes()),L=float(table['L']),H=float(table['H']),C_h=float(table['C_h']),max_anchor_residual=float(table['anchor_residual'].max()),max_intermediate_balance=float(abs(m['balanced_b']).max()),max_bound=float((a+b).max()),max_even_balance=float(abs(m['even_b']).max()),max_abs_angular_mean=float(max(abs(m[k]).max() for k in m if k.endswith('_mean_u'))),balanced_orientation_asymmetry=float(m['balanced_asymmetry'].mean()),calibration_forwards=p.NR*p.NA))
    co(out/'pilot.csv',rows);print('Old16 pilot completed',flush=True)

def freeze():
    paths=[ROOT/'policy.py',ROOT/'run.py',ROOT/'THEORY.md',ROOT/'PROTOCOL.md',ROOT/'old'/'pilot.csv']
    jo(ROOT/'freeze.json',dict(timestamp_utc=now(),config=CONFIG,policies=POLICIES,hashes={str(f.relative_to(ROOT)):sha(f.read_bytes()) for f in paths},new_states_opened=False,new_outcomes_accessed=False))

def calibrate():
    fr=json.loads((ROOT/'freeze.json').read_text());assert fr['hashes']['policy.py']==sha((ROOT/'policy.py').read_bytes()) and fr['hashes']['run.py']==sha(Path(__file__).read_bytes())
    out=ROOT/'fresh';out.mkdir(exist_ok=False);(out/'policies').mkdir();manifest={};rows=[]
    for seed in CONFIG['new_pairs']:
        source=ROOT.parent/'confirmation'/'states'/f'original_{seed}.npz';state=np.load(source);theta=state['pb'];start=time.perf_counter();table=p.fit(theta);seconds=time.perf_counter()-start
        path=out/'policies'/f'{seed}_policy.npz';np.savez_compressed(path,**table);manifest[str(path.relative_to(out))]=sha(path.read_bytes())
        x=sample(CONFIG['cal_seed_base']+seed,p.NR*p.NA//2);calls=[]
        def counted(z):calls.append(len(z));return p.pred(theta,z)
        start=time.perf_counter();gate=p.Gate.fit(x,counted);gate_seconds=time.perf_counter()-start;assert sum(calls)==p.NR*p.NA
        dest=out/'policies'/f'{seed}_gate.npz';np.savez_compressed(dest,edges=gate.edges,radius_scores=gate.radius_scores,folded_scores=gate.folded_scores);manifest[str(dest.relative_to(out))]=sha(dest.read_bytes())
        rows.append(dict(seed=seed,training_seed=821000+seed,initialization_seed=822000+seed,state_sha256=sha(theta.tobytes()),source_sha256=sha(source.read_bytes()),field_calibration_forwards=p.NR*p.NA,legacy_calibration_forwards=sum(calls),field_seconds=seconds,legacy_seconds=gate_seconds,cal_seed=CONFIG['cal_seed_base']+seed,L=float(table['L']),H=float(table['H']),C_h=float(table['C_h'])))
        print(f'calibrated continuous seed {seed}/16',flush=True)
    co(out/'calibration.csv',rows);jo(out/'policy_commit.json',dict(timestamp_utc=now(),manifest=manifest,calibration_sha256=sha((out/'calibration.csv').read_bytes()),test_started=False))

def selected_mass(r,rho,b,threshold,high=True):
    # Integrate the piecewise-linear radial density over b-linear threshold regions.
    if not high:return trap(rho,r)-selected_mass(r,rho,b,threshold,True)
    b0=b[:-1];b1=b[1:];delta=np.diff(r);full=delta*(rho[:-1]+rho[1:])/2;mass=np.where((b0>threshold)&(b1>threshold),full,0.)
    crossing=(b0>threshold)!=(b1>threshold);fraction=np.divide(threshold-b0,b1-b0,out=np.zeros_like(b0),where=b1!=b0);fraction=np.clip(fraction,0,1)
    left=delta*(rho[:-1]*fraction+.5*(rho[1:]-rho[:-1])*fraction*fraction)
    mass[crossing]=np.where(b0[crossing]>threshold,left[crossing],full[crossing]-left[crossing])
    return float(mass.sum())

def worlds_for(r,rho,m):
    worlds=[dict(name='rings',kind='rings'),dict(name='constant_half',kind='constant',value=.5),dict(name='increasing',kind='sigmoid',sign=1),dict(name='decreasing',kind='sigmoid',sign=-1)]
    for field in p.FIELDS:
        b=np.zeros(len(r)) if field=='even' else m[field+'_b']
        for high,label in ((True,'min'),(False,'max')):
            worlds.append(dict(name=field+'_unrestricted_'+label,kind='threshold',field=field,threshold=0.,high=high,constant_value=(0. if high else 1.) if field=='even' else -1.))
            if field=='even':threshold=0.;mass=.5
            else:
                lo=float(b.min());hi=float(b.max())
                for _ in range(60):
                    threshold=(lo+hi)/2;mass=selected_mass(r,rho,b,threshold,high)
                    if (mass>.5)==high:lo=threshold
                    else:hi=threshold
                threshold=(lo+hi)/2;mass=selected_mass(r,rho,b,threshold,high)
            worlds.append(dict(name=field+'_prevalence_half_'+label,kind='threshold',field=field,threshold=threshold,high=high,constant_value=.5 if field=='even' else -1.,numerical_prevalence=mass))
    return worlds

def world_prob(world,r,radial,m):
    if world['kind']=='rings':return post(r)
    if world['kind']=='constant':return np.full(len(r),world['value'])
    if world['kind']=='sigmoid':return np.exp(-np.logaddexp(0,-world['sign']*5*(r-.85)))
    if world['constant_value']>=0:return np.full(len(r),world['constant_value'])
    b=np.interp(r,radial,m[world['field']+'_b'],left=0,right=0);return (b>world['threshold']).astype(float) if world['high'] else (b<world['threshold']).astype(float)

def posterior_integral(r,rho,target,world,m):
    if world['kind']!='threshold':return trap(rho*target*world_prob(world,r,r,m),r)
    if world['constant_value']>=0:return world['constant_value']*trap(rho*target,r)
    source=m[world['field']+'_b'];threshold=world['threshold'];high=world['high'];b0=source[:-1];b1=source[1:];delta=np.diff(r)
    da=np.diff(target);dr=np.diff(rho);f0=rho[:-1]*target[:-1];f1=rho[:-1]*da+dr*target[:-1];f2=dr*da
    full=delta*(f0+f1/2+f2/3)
    fraction=np.divide(threshold-b0,b1-b0,out=np.zeros_like(b0),where=b1!=b0);fraction=np.clip(fraction,0,1)
    left=delta*(f0*fraction+f1*fraction**2/2+f2*fraction**3/3)
    first=b0>threshold;last=b1>threshold;selected=np.where(first&last,full,0.);cross=first!=last;selected[cross]=np.where(first[cross],left[cross],full[cross]-left[cross])
    return float(selected.sum()) if high else float((full-selected).sum())

def diagnostics(table,seed,out):
    r=np.linspace(0,3,CONFIG['diagnostic_nr']);rho=density(r);m=moments(table,r,CONFIG['diagnostic_na'],CONFIG['diagnostic_phase']);finer=moments(table,r[::8],CONFIG['refinement_na'],CONFIG['refinement_phase']);a,b=p.bound_at(table,r);bound=a+b
    worlds=worlds_for(r,rho,m);rows=[];bounds=[]
    E=table['anchor_residual']+table['angular_bound'];g=table['gamma'];dlt=np.diff(table['radii']);kr=float(table['L'])*dlt*(g[:-1]+g[1:]);tstar=np.clip(np.divide(E[1:]-E[:-1]+kr,2*kr,out=np.zeros_like(kr),where=kr>0),0,1);uniform_bound=float(np.max((E[:-1]+(E[1:]-E[:-1])*tstar+kr*tstar*(1-tstar))/8))
    for field in p.FIELDS:
        observed=m[field+'_b'];theoretical_b=np.zeros(len(r)) if field=='even' else observed
        C=trap(rho*m[field+'_C'],r);positive=trap(rho*np.maximum(theoretical_b,0),r);negative=trap(rho*np.maximum(-theoretical_b,0),r)
        refinement=float(max(np.max(abs(m[field+'_b'][::8]-finer[field+'_b'])),np.max(abs(m[field+'_C'][::8]-finer[field+'_C']))))
        coarse=trap(rho[::2]*abs(observed[::2]),r[::2]);fine=trap(rho*abs(observed),r)
        bounds.append(dict(seed=seed,field=field,C=C,unrestricted_lower=C-positive,unrestricted_upper=C+negative,unrestricted_width=positive+negative,observed_max_balance=float(abs(observed).max()),observed_integrated_abs_balance=fine,uniform_bound=0. if field=='even' else (uniform_bound if field=='balanced' else ''),integrated_bound=0. if field=='even' else (trap(rho*bound,r) if field=='balanced' else ''),angular_bound_integral=trap(rho*a,r) if field=='balanced' else '',radial_bound_integral=trap(rho*b,r) if field=='balanced' else '',angular_refinement_error=refinement,radial_refinement_abs_moment_error=abs(coarse-fine),orientation_asymmetry=trap(rho*m[field+'_asymmetry'],r),max_abs_angular_mean=float(abs(m[field+'_mean_u']).max()),radial_mass=trap(rho,r)))
        for world in worlds:
            pw=world_prob(world,r,r,m);rows.append(dict(seed=seed,field=field,world=world['name'],continuous_gain=C-posterior_integral(r,rho,theoretical_b,world,m),posterior_prevalence=posterior_integral(r,rho,np.ones_like(r),world,m),C=C))
    assert np.all(abs(m['balanced_b'])<=bound+1e-10)
    np.savez_compressed(out/f'{seed}_moments.npz',r=r,density=rho,angular_bound=a,radial_bound=b,**m);jo(out/f'{seed}_worlds.json',worlds)
    return r,m,worlds,rows,bounds

def score():
    out=ROOT/'fresh';commit=json.loads((out/'policy_commit.json').read_text());assert all(sha((out/k).read_bytes())==v for k,v in commit['manifest'].items());jo(out/'score_start.json',dict(timestamp_utc=now(),policy_commit_sha256=sha((out/'policy_commit.json').read_bytes())));diag=out/'diagnostics';diag.mkdir()
    rows=[];audits=[];batchrows=[];continuous=[];bounds=[];sample_records=[];certificates=[]
    for seed in CONFIG['new_pairs']:
        table=dict(np.load(out/'policies'/f'{seed}_policy.npz'));gt=np.load(out/'policies'/f'{seed}_gate.npz');gate=p.Gate(gt['edges'],gt['radius_scores'],gt['folded_scores']);radial,m,worlds,cr,br=diagnostics(table,seed,diag);continuous+=cr;bounds+=br
        x=sample(CONFIG['test_seed_base']+seed,CONFIG['test_n']);r=np.linalg.norm(x,axis=1);ties=np.random.default_rng(CONFIG['tie_seed_base']+seed).random(len(x));uniform=np.random.default_rng(CONFIG['group_seed_base']+seed).random(len(x)//4);truep=post(r);labels=(np.random.default_rng(CONFIG['label_seed_base']+seed).random(len(x))<truep).astype(float);wprobs={w['name']:world_prob(w,r,radial,m) for w in worlds};first=None;full=None
        sample_records.append(dict(seed=seed,x_sha256=sha(x.tobytes()),tie_sha256=sha(ties.tobytes()),group_uniform_sha256=sha(uniform.tobytes()),label_sha256=sha(labels.tobytes()),test_seed=CONFIG['test_seed_base']+seed,n=len(x)))
        for name in POLICIES:
            predictor=lambda xx:p.pred(table['theta'],xx);output,mask=p.deploy(x,predictor,table,gate,name,ties,uniform)
            if name=='raw':first=output.copy()
            if name=='full':full=output.copy()
            sampled=nll(output,labels);blocks=[dict(seed=seed,policy=name,batch=j,sampled_label_risk=float(sampled[j*256:(j+1)*256].mean())) for j in range(len(x)//256)]
            for world in worlds:
                risk=nll(output,wprobs[world['name']]);bybatch=risk.reshape(-1,256).mean(axis=1)
                rows.append(dict(seed=seed,policy=name,world=world['name'],expected_risk=float(risk.mean()),sampled_label_risk=float(sampled.mean()) if world['name']=='rings' else '',selected=int(mask.sum()),per_input_calls=1+mask.mean(),mask_sha256=sha(np.packbits(mask).tobytes())))
                for j,row in enumerate(blocks):row[world['name']]=float(bybatch[j])
            batchrows+=blocks;calls=[]
            def counted(xx):calls.append(len(xx));return predictor(xx)
            replay,selected=p.deploy(x[:256],counted,table,gate,name,ties[:256],uniform[:64]);expected=[256] if name=='raw' else ([256,256] if name=='full' else [256,64]);assert calls==expected and np.array_equal(selected,mask[:256]);err=float(abs(replay-output[:256]).max());assert err<1e-12
            audits.append(dict(seed=seed,policy=name,calls=json.dumps(calls),selected=int(selected.sum()),max_output_error=err))
        # A separate valid but possibly vacuous finite-sample lower certificate.
        u=p.field_value(table,x,'balanced');W=(.25+u/8)*(np.logaddexp(0,first)-np.logaddexp(0,full))*(r<=3)
        epsilon=.75*(3+float(table['C_h']))*np.sqrt(np.log(20)/(2*len(x)));B=next(z['uniform_bound'] for z in br if z['field']=='balanced')
        certificates.append(dict(seed=seed,n=len(x),confidence=.95,C_truncated_sample_mean=float(W.mean()),hoeffding_epsilon=float(epsilon),conditional_uniform_bound=B,gain_lower=float(W.mean()-epsilon-B),tail_note='Outside radius3 u=0, marginal uniform selection has nonnegative Jensen gain; omitted tail does not weaken lower bound.'))
        print(f'scored continuous seed {seed}/16',flush=True)
    co(out/'per_seed.csv',rows);co(out/'deployment_audit.csv',audits);co(out/'continuous_gains.csv',continuous);co(out/'balance_bounds.csv',bounds);co(out/'sample_records.csv',sample_records);co(out/'finite_sample_certificates.csv',certificates)
    with gzip.open(out/'per_batch.csv.gz','wt',newline='',encoding='utf8') as f:w=csv.DictWriter(f,list(batchrows[0]));w.writeheader();w.writerows(batchrows)
    jo(out/'score_complete.json',dict(timestamp_utc=now(),per_seed_rows=len(rows),batch_rows=len(batchrows),deploy_audits=len(audits),policy_immutable=all(sha((out/k).read_bytes())==v for k,v in commit['manifest'].items()),outcomes_sha256=sha((out/'per_seed.csv').read_bytes())))

def summarize():
    rows=ci(ROOT/'fresh'/'per_seed.csv');groups={};ix={(int(r['seed']),r['policy'],r['world']):r for r in rows};rng=np.random.default_rng(CONFIG['bootstrap_seed']);draw=rng.integers(0,16,(CONFIG['bootstrap_repeats'],16));summary=[];contrasts=[]
    worlds=list(dict.fromkeys(r['world'] for r in rows))
    for world in worlds:
        for policy in POLICIES:
            a=np.array([float(ix[k,policy,world]['expected_risk']) for k in range(1,17)]);q=np.quantile(a[draw].mean(axis=1),[.025,.975]);summary.append(dict(policy=policy,world=world,n_seed=16,mean_risk=a.mean(),ci_lo=q[0],ci_hi=q[1]))
        for ref in ('group_even','group_unbalanced','group_uniform','group_confidence','confidence','radius','folded'):
            d=np.array([float(ix[k,'group_balanced',world]['expected_risk'])-float(ix[k,ref,world]['expected_risk']) for k in range(1,17)]);q=np.quantile(d[draw].mean(axis=1),[.025,.975]);contrasts.append(dict(policy='group_balanced',reference=ref,world=world,n_seed=16,mean_difference=d.mean(),ci_lo=q[0],ci_hi=q[1],lower_risk_seeds=int((d<0).sum())))
    co(ROOT/'summary.csv',summary);co(ROOT/'contrasts.csv',contrasts)
    print(json.dumps([r for r in contrasts if r['world']=='rings'],indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['oldpilot','freeze','calibrate','score','summarize']);args=parser.parse_args()
    with threadpool_limits(1):globals()[args.action]()
