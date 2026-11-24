"""Frozen bounded checks of approximate symmetry and finite-support identification."""
from __future__ import annotations
import argparse,csv,datetime,hashlib,itertools,json,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'source'))
from pullback_feature_learning import core
MODELS=ROOT.parent/'legacy/budgeted-rotation-final/results/models.npz'
CONFIG=dict(old_model_keys=[f'{i}_pb' for i in range(1,17)],epsilons=[0.,.1,.25,.5,.75,1.],
    radius_max=3.,radius_cells=128,angle_cells=128,n_test=32768,batch=256,quota=64,
    test_seed_base=7731000,tie_seed_base=7732000,bootstrap_seed=7733000,
    finite_radii=16,finite_angles=16,finite_radius_range=[.25,1.55],per_radius_quota=4,
    independent_grid_batches=2048,grid_random_seed_base=7734000,
    nature='Fixed previously inspected old pullback models; no fresh training replication')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')
def csvout(path,rows):
    with Path(path).open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def sig(a):return np.exp(-np.logaddexp(0.,-a))
def soft(a):return np.logaddexp(0.,a)
def gap(a,b):return .5*(soft(a)+soft(b))-soft(.5*(a+b))

def freeze():
    if (ROOT/'freeze.json').exists():raise FileExistsError('Protocol already frozen')
    text='''# Frozen ideas 3 and 4 protocol

All sixteen archived old PB500 models are fixed and previously inspected. This is a bounded sensitivity and mathematical-validation study, not new model replication. No setting is selected from the following held-out scores. Retain every epsilon, model, policy and failure.

Idea3: epsilon 0,.1,.25,.5,.75,1; p_epsilon=(1-epsilon)p0(r)+epsilon(1+cos(phi))/2. Independent 32768-input draws per model; known posterior is evaluator-only. Offline interval propagation through the fixed model encloses both logits on 128 radial by128 folded-angular cells covering radius0..3. Jensen minimum at closest disjoint interval endpoints, minus epsilon times the maximum endpoint separation/4, is a deterministic conservative lower bound. Canonical half-plane coordinates give exactly antipodally invariant cell indices. Outside the grid, a global residual-output bound gives a conservative lower score. Never use held-out partner logits to allocate executable policies.

Compare raw, universal half-turn, random64, certified optional at-most64, and forced64 largest cell bounds per256. A pointwise-bound oracle is separately labeled as a2-call diagnostic. All tie keys are independent. Both orientations are evaluated offline to calculate the exact orbit-average risk and bound identity; this does not subsidize deployment. Count actual executable model calls on representative batches and report offline interval work separately. A negative bound is not a harm claim; report positive-gain/negative-bound cases, actual harmful fraction, first tested epsilon with net harm, and unused budgets.

Idea4 finite population:16 equally weighted radii linearly spaced.25..1.55,16 equally weighted equally spaced orientations. Each independent stratified256-input batch contains every support point once in a random permutation; this is not256iid points. Uniform selection within a radius uses exact4-element subsets. Enumerate1820 subsets, compute their C and b, and use the upper concave envelope to maximizeC with expectedb0. Mixture of atmost2 subsets perradius preserves64calls every realization. Compare a radius-only4-radius rule, best2antipodal-pairs perradius, confidence64 globally, and the orientation-sensitive balanced mixture. Freeze tables before independently randomized batches. Verify actual marginal antipodal asymmetry and conditional signed balance, not just individual action asymmetry.

Evaluate finite-population expectations and2048 independent randomized stratified batches. Test constant.5, increasing/decreasing and smooth radial posteriors plus each policy's unrestricted and fixed-prevalence.5 sharp extremes. Expected risks use those hidden posteriors only in the evaluator. Report interval widths and attainability. Offline table uses256model forwards permodel; support-only cached predictions could make future calls unnecessary. The executed256+64 implementation deliberately stores only selection tables for deployment. This is a feasibility demonstration, not continuous generalization or a fundamental compute-saving claim.

Validate the four-point fixed-prevalence counterexample, sharp epsilon bound, known-prevalence fractional allocation bounds, and a continuous-radius bin-cancellation example with a smoothness bound. All scripts/theory, model hashes, selector commits, raw scores and numerical checks are retained. Resample complete old model records for descriptive aggregate intervals; policy repetitions are not independent trained models.
'''
    (ROOT/'PROTOCOL.md').write_text(text,encoding='utf8')
    dump(ROOT/'freeze.json',dict(timestamp_utc=now(),config=CONFIG,script_sha256=sha(__file__),protocol_sha256=sha(ROOT/'PROTOCOL.md'),
        theory_sha256=sha(ROOT/'THEORY.md'),model_sha256=sha(MODELS),core_sha256=sha(core.__file__)))

def interval_forward(theta,lo,hi):
    for a,b,bias in core.unpack(theta):
        ap=np.maximum(a,0);an=np.minimum(a,0);bp=np.maximum(b,0);bn=np.minimum(b,0)
        tl=np.tanh(lo@ap.T+hi@an.T+bias);tu=np.tanh(hi@ap.T+lo@an.T+bias)
        lo,hi=lo+.5*(tl@bp.T+tu@bn.T),hi+.5*(tu@bp.T+tl@bn.T)
    return lo[:,0],hi[:,0]

def polar_boxes():
    re=np.linspace(0,CONFIG['radius_max'],CONFIG['radius_cells']+1);ae=np.linspace(0,np.pi,CONFIG['angle_cells']+1)
    r0=np.repeat(re[:-1],len(ae)-1);r1=np.repeat(re[1:],len(ae)-1);a0=np.tile(ae[:-1],len(re)-1);a1=np.tile(ae[1:],len(re)-1)
    cl=np.cos(a1);cu=np.cos(a0);sl=np.minimum(np.sin(a0),np.sin(a1));su=np.maximum(np.sin(a0),np.sin(a1));su[(a0<=np.pi/2)&(a1>=np.pi/2)]=1.
    cx=np.stack([r0*cl,r0*cu,r1*cl,r1*cu]);sy=np.stack([r0*sl,r0*su,r1*sl,r1*su])
    lo=np.c_[cx.min(0),sy.min(0)]-1e-12;hi=np.c_[cx.max(0),sy.max(0)]+1e-12
    return re,ae,lo,hi

def cell_lookup(x,table,epsilon):
    canonical=np.where(((x[:,1]<0)|((x[:,1]==0)&(x[:,0]<0)))[:,None],-x,x)
    radius=np.linalg.norm(canonical,axis=1);angle=np.arctan2(canonical[:,1],canonical[:,0])
    ri=np.searchsorted(table['r_edges'][1:-1],radius,side='right');ai=np.searchsorted(table['a_edges'][1:-1],angle,side='right')
    ids=ri*CONFIG['angle_cells']+ai;inside=radius<=CONFIG['radius_max']
    lower=table['jmin'][ids]-epsilon*table['dmax'][ids]/4
    lower=np.where(inside,lower,-epsilon*(2*np.abs(x[:,0])+2*float(table['residual_bound']))/4)
    return lower,ids,inside

def allocate(score,tie,optional=False):
    n=len(score);out=np.zeros(n,dtype=bool)
    for lo in range(0,n,256):
        ind=np.lexsort((tie[lo:lo+256],-score[lo:lo+256]))[:64]+lo
        if optional:ind=ind[score[ind]>1e-14]
        out[ind]=True
    return out

def deploy_continuous(theta,x,table,epsilon,tie,policy,predict=None):
    predict=(lambda z:core.representation(theta,z)[:,0]) if predict is None else predict
    h=predict(x)
    if policy=='always':mask=np.ones(len(x),dtype=bool)
    elif policy=='random64':mask=allocate(np.zeros(len(x)),tie)
    else:mask=allocate(cell_lookup(x,table,epsilon)[0],tie,optional=policy=='safe64')
    ans=h.copy()
    if mask.any():ans[mask]=.5*(h[mask]+predict(-x[mask]))
    return ans,mask

def upper_hull_mixture(b,c):
    order=np.lexsort((-c,b));unique=[]
    for idx in order:
        if unique and b[idx]==b[unique[-1]]:continue
        unique.append(int(idx))
    hull=[]
    for i in unique:
        while len(hull)>=2:
            a,z=hull[-2:];cross=(b[z]-b[a])*(c[i]-c[z])-(c[z]-c[a])*(b[i]-b[z])
            if cross>=-1e-15:hull.pop()
            else:break
        hull.append(i)
    best=(-np.inf,None)
    for i in hull:
        if abs(b[i])<1e-14 and c[i]>best[0]:best=(c[i],(i,i,1.))
    for i,j in zip(hull[:-1],hull[1:]):
        if b[i]<=0<=b[j] and b[j]>b[i]:
            p=b[j]/(b[j]-b[i]);value=p*c[i]+(1-p)*c[j]
            if value>best[0]:best=(value,(i,j,p))
    if best[1] is None:raise ArithmeticError('No conditional-balance mixture found')
    return best[1]

def fit():
    fr=json.loads((ROOT/'freeze.json').read_text());assert fr['script_sha256']==sha(__file__) and fr['model_sha256']==sha(MODELS)
    out=ROOT/'calibration';out.mkdir(exist_ok=False);models=np.load(MODELS);records=[];re,ae,lo,hi=polar_boxes()
    r=np.linspace(*CONFIG['finite_radius_range'],16);a=2*np.pi*np.arange(16)/16
    grid=(r[:,None,None]*np.stack([np.cos(a),np.sin(a)],axis=-1)).reshape(256,2)
    subsets=np.zeros((1820,16),dtype=np.int8)
    for i,ss in enumerate(itertools.combinations(range(16),4)):subsets[i,list(ss)]=1
    for key in CONFIG['old_model_keys']:
        t=models[key];start=time.perf_counter();al,au=interval_forward(t,lo,hi);bl,bu=interval_forward(t,-hi,-lo)
        jmin=np.where(al>bu,gap(al,bu),np.where(bl>au,gap(au,bl),0.));jmin=np.maximum(jmin,0)
        dmax=np.maximum(abs(al-bu),abs(au-bl));residual=sum(.5*np.abs(b[0]).sum() for _,b,_ in core.unpack(t))
        np.savez_compressed(out/(key+'_interval.npz'),r_edges=re,a_edges=ae,box_lo=lo,box_hi=hi,a_lo=al,a_hi=au,b_lo=bl,b_hi=bu,jmin=jmin,dmax=dmax,residual_bound=residual)
        interval_seconds=time.perf_counter()-start;start=time.perf_counter()
        h=core.representation(t,grid)[:,0].reshape(16,16);m=.5*(h+np.roll(h,8,axis=1));cc=soft(h)-soft(m);dd=h-m
        probs=np.zeros((4,16,16));names=np.array(['radius_only','antipodal_pairs','confidence','balanced_orientation'])
        best_r=np.argsort(-gap(h,np.roll(h,8,axis=1)).mean(1),kind='stable')[:4];probs[0,best_r]=1
        for j in range(16):
            pairs=np.argsort(-gap(h[j,:8],h[j,8:]),kind='stable')[:2];probs[1,j,pairs]=1;probs[1,j,pairs+8]=1
        conf=np.argsort(np.abs(h).ravel(),kind='stable')[:64];probs[2].ravel()[conf]=1
        left=[];right=[];mix=[]
        for j in range(16):
            bs=subsets@dd[j]/16;cs=subsets@cc[j]/16;i,k,p=upper_hull_mixture(bs,cs)
            left.append(subsets[i]);right.append(subsets[k]);mix.append(p);probs[3,j]=p*subsets[i]+(1-p)*subsets[k]
        b=(probs*dd).mean(2);c=(probs*cc).mean((1,2));assert np.max(abs(b[3]))<1e-12 and np.max(abs(probs.sum((1,2))-64))<1e-12
        np.savez_compressed(out/(key+'_finite.npz'),radii=r,angles=a,x=grid,h=h,m=m,names=names,probabilities=probs,b=b,c=c,left=np.array(left),right=np.array(right),mix=np.array(mix),subsets=subsets)
        records.append(dict(model=key,interval_cells=len(lo),interval_propagations=2*len(lo),interval_seconds=interval_seconds,
            finite_support_forward_examples=256,finite_policy_seconds=time.perf_counter()-start,
            balanced_max_abs_b=float(np.max(abs(b[3]))),balanced_antipodal_probability_asymmetry=float(np.max(abs(probs[3]-np.roll(probs[3],8,axis=1))))))
    csvout(out/'costs_and_balance.csv',records)
    dump(out/'selector_commit.json',dict(timestamp_utc=now(),script_sha256=sha(__file__),freeze_sha256=sha(ROOT/'freeze.json'),npz_sha256={p.name:sha(p) for p in out.glob('*.npz')},risk_scoring_started=False))

def sample_inputs(seed,n):
    rng=np.random.default_rng(seed);r=rng.choice([.5,1.2],n);a=rng.uniform(0,2*np.pi,n)
    return r[:,None]*np.c_[np.cos(a),np.sin(a)]+rng.normal(0,.15,(n,2))

def rings_posterior(r):
    return sig((.5**2-1.2**2)/(2*.15**2)+np.log(np.i0(1.2*r/.15**2))-np.log(np.i0(.5*r/.15**2)))

def fractional_extreme(b,weights,prevalence,largest):
    p=np.zeros(len(b));remaining=prevalence
    for i in np.argsort(-b if largest else b,kind='stable'):
        take=min(remaining,weights[i]);p[i]=take/weights[i];remaining-=take
        if remaining<=1e-14:break
    assert abs(p@weights-prevalence)<1e-12
    return p

def risk_interval(c,b,weights,prevalence=None):
    if prevalence is None:worst=(b>0).astype(float);best=(b<0).astype(float)
    else:worst=fractional_extreme(b,weights,prevalence,True);best=fractional_extreme(b,weights,prevalence,False)
    return c-weights@(worst*b),c-weights@(best*b),worst,best

def evaluate3(models,out):
    rows=[];audit=[];batchrows=[]
    for i,key in enumerate(CONFIG['old_model_keys'],1):
        t=models[key];table=np.load(ROOT/'calibration'/f'{key}_interval.npz');x=sample_inputs(CONFIG['test_seed_base']+i,CONFIG['n_test']);ties=np.random.default_rng(CONFIG['tie_seed_base']+i).random(len(x))
        a=core.representation(t,x)[:,0];b=core.representation(t,-x)[:,0];m=.5*(a+b);J=gap(a,b);radius=np.linalg.norm(x,axis=1);p0=rings_posterior(radius);cos=x[:,0]/radius
        lower,ids,inside=cell_lookup(x,table,0);assert np.array_equal(ids,cell_lookup(-x,table,0)[1])
        canonical=np.where(((x[:,1]<0)|((x[:,1]==0)&(x[:,0]<0))),-1.,1.)
        aa=np.where(canonical==1,a,b);bb=np.where(canonical==1,b,a)
        violation=max(float(np.max(table['a_lo'][ids[inside]]-aa[inside])),float(np.max(aa[inside]-table['a_hi'][ids[inside]])),float(np.max(table['b_lo'][ids[inside]]-bb[inside])),float(np.max(bb[inside]-table['b_hi'][ids[inside]])))
        assert violation<1e-10
        for epsilon in CONFIG['epsilons']:
            p=(1-epsilon)*p0+epsilon*(1+cos)/2;q=(1-epsilon)*p0+epsilon*(1-cos)/2
            raw=.5*(soft(a)-p*a+soft(b)-q*b);full=soft(m)-.5*(p+q)*m;actual=raw-full
            point=J-epsilon*abs(a-b)/4;cell=cell_lookup(x,table,epsilon)[0]
            assert np.max(abs(actual-(J-(p-q)*(a-b)/4)))<1e-11 and np.min(actual-cell)>-1e-10
            masks=dict(raw=np.zeros(len(x),bool),always=np.ones(len(x),bool),random64=allocate(np.zeros(len(x)),ties),safe64=allocate(cell,ties,True),forced64=allocate(cell,ties),oracle_safe64=allocate(point,ties,True))
            for policy,mask in masks.items():
                gain=mask*actual;risk=raw-gain;calls=2. if policy=='oracle_safe64' else 1+mask.mean()
                if policy=='safe64':assert np.min(actual[mask],initial=0)>=-1e-10 and (mask.reshape(-1,256).sum(1)<=64).all()
                if policy in ('forced64','random64'):assert (mask.reshape(-1,256).sum(1)==64).all()
                rows.append(dict(model=key,epsilon=epsilon,policy=policy,risk=float(risk.mean()),gain=float(gain.mean()),selected_fraction=float(mask.mean()),model_evaluations_per_input=float(calls),
                    mean_certified_lower_bound=float((mask*cell).mean()) if policy!='oracle_safe64' else float((mask*point).mean()),
                    harmful_selected_fraction=float(np.mean(mask&(actual< -1e-12))),positive_gain_despite_negative_point_bound=float(np.mean((actual>1e-12)&(point< -1e-12))),
                    positive_gain_despite_negative_cell_bound=float(np.mean((actual>1e-12)&(cell< -1e-12))),outside_grid=int((~inside).sum()),input_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
                for j,value in enumerate(gain.reshape(-1,256).mean(1)):
                    batchrows.append(dict(model=key,epsilon=epsilon,policy=policy,batch=j,gain=float(value),selected=int(mask.reshape(-1,256)[j].sum())))
            if i in (1,16):
                for policy in ('always','random64','safe64','forced64'):
                    calls=[]
                    def counted(z):calls.append(len(z));return core.representation(t,z)[:,0]
                    result,mask=deploy_continuous(t,x[:256],table,epsilon,ties[:256],policy,counted)
                    ref=np.where(mask,m[:256],a[:256]);assert np.max(abs(result-ref))<1e-12
                    assert calls==([256,int(mask.sum())] if mask.any() else [256])
                    audit.append(dict(model=key,epsilon=epsilon,policy=policy,calls=calls,selected=int(mask.sum()),max_deploy_error=float(np.max(abs(result-ref))),max_interval_violation=violation))
        print('idea3',key,flush=True)
    csvout(out/'idea3_by_model.csv',rows);csvout(out/'idea3_batches.csv',batchrows);dump(out/'idea3_deployment_audit.json',audit)

def finite_masks(data,rng,n):
    probs=data['probabilities'];out=np.broadcast_to(probs[:3,None],(3,n,16,16)).copy()
    coin=rng.random((n,16))<data['mix'][None,:]
    balanced=np.where(coin[:,:,None],data['left'][None,:,:],data['right'][None,:,:])
    return np.concatenate([out,balanced[None]],axis=0)

def evaluate4(models,out):
    intervals=[];worldrows=[];randomrows=[];audits=[]
    weights=np.ones(16)/16
    for i,key in enumerate(CONFIG['old_model_keys'],1):
        data=np.load(ROOT/'calibration'/f'{key}_finite.npz');h=data['h'];m=data['m'];b=data['b'];c=data['c'];names=list(data['names']);probs=data['probabilities']
        rng=np.random.default_rng(CONFIG['grid_random_seed_base']+i);masks=finite_masks(data,rng,CONFIG['independent_grid_batches']);assert np.all(masks.sum((2,3))==64)
        empirical_b=(masks*(h-m)[None,None]).mean((1,3));empirical_c=(masks*(soft(h)-soft(m))[None,None]).mean((1,2,3))
        worlds={'constant_half':np.full(16,.5),'increasing':sig(8*(data['radii']-.9)),'decreasing':sig(-8*(data['radii']-.9)),'smooth_wave':.5+.4*np.cos(2*np.pi*np.arange(16)/16)}
        for j,name in enumerate(names):
            for mode,pi in (('unrestricted',None),('prevalence_half',.5)):
                low,high,worst,best=risk_interval(c[j],b[j],weights,pi)
                worlds[name+'_'+mode+'_worst']=worst;worlds[name+'_'+mode+'_best']=best
                intervals.append(dict(model=key,policy=name,constraint=mode,c=float(c[j]),mean_abs_conditional_b=float(np.mean(abs(b[j]))),max_abs_conditional_b=float(np.max(abs(b[j]))),
                    lower=float(low),upper=float(high),width=float(high-low),worst_prevalence=float(worst.mean()),best_prevalence=float(best.mean()),
                    antipodal_selection_probability_asymmetry=float(np.max(abs(probs[j]-np.roll(probs[j],8,axis=1))))))
        for world,p in worlds.items():
            raw=soft(h)-p[:,None]*h;changed=soft(m)-p[:,None]*m;pointgain=raw-changed
            for j,name in enumerate(names):
                expected=float((probs[j]*pointgain).mean());identity=float(c[j]-weights@(p*b[j]));assert abs(expected-identity)<1e-12
                perbatch=(masks[j]*pointgain[None]).mean((1,2))
                worldrows.append(dict(model=key,policy=name,world=world,prevalence=float(p.mean()),exact_gain=expected,monte_carlo_gain=float(perbatch.mean()),
                    monte_carlo_se=float(perbatch.std(ddof=1)/np.sqrt(len(perbatch))),max_identity_error=abs(expected-identity),raw_risk=float(raw.mean()),policy_risk=float(raw.mean()-expected)))
        for j,name in enumerate(names):
            randomrows.append(dict(model=key,policy=name,batches=len(masks[j]),min_extra_calls=int(masks[j].sum((1,2)).min()),max_extra_calls=int(masks[j].sum((1,2)).max()),
                exact_c=float(c[j]),empirical_c=float(empirical_c[j]),max_empirical_abs_b=float(np.max(abs(empirical_b[j]))),max_exact_abs_b=float(np.max(abs(b[j]))),
                mean_realized_absolute_balance=float(np.abs((masks[j]*(h-m)[None]).mean(2)).mean())))
        # Independent batch permutation and executed calls; selection uses saved support-only table.
        permutation=rng.permutation(256);x=data['x'][permutation];mask=masks[3,0].ravel()[permutation].astype(bool);calls=[]
        def counted(z):calls.append(len(z));return core.representation(models[key],z)[:,0]
        first=counted(x);pred=first.copy();pred[mask]=.5*(first[mask]+counted(-x[mask]));assert calls==[256,64]
        ref=np.where(mask,m.ravel()[permutation],h.ravel()[permutation]);assert np.max(abs(pred-ref))<1e-11
        audits.append(dict(model=key,calls=calls,max_deploy_error=float(np.max(abs(pred-ref))),max_expected_balance=float(np.max(abs(b[3]))),
            balanced_minus_antipodal_gain=float(c[3]-c[1]),marginal_asymmetry=float(np.max(abs(probs[3]-np.roll(probs[3],8,axis=1))))))
        print('idea4',key,flush=True)
    csvout(out/'idea4_intervals.csv',intervals);csvout(out/'idea4_hidden_worlds.csv',worldrows);csvout(out/'idea4_randomized_batches.csv',randomrows);dump(out/'idea4_deployment_audit.json',audits)

def math_checks():
    rng=np.random.default_rng(7735000);a=rng.normal(0,4,10000);b=rng.normal(0,4,10000);epsilon=rng.random(10000);p=(1+epsilon*np.sign(a-b))/2;q=1-p;m=(a+b)/2
    actual=.5*(soft(a)-p*a+soft(b)-q*b)-(soft(m)-.5*(p+q)*m);bound=gap(a,b)-epsilon*abs(a-b)/4
    counter=[float((soft(0.)-soft(1.)+p)/4) for p in (1.,0.)]
    u=(np.arange(200000)+.5)/200000;bb=np.where(u<.5,.5,-.5);C=.25*(soft(2.)+soft(0.)-2*soft(1.));pp=.5+.5*np.tanh(10*(.5-u))
    smooth=[float(C-np.mean(pp*bb)),float(C-np.mean((1-pp)*bb))]
    # Exhaustively verify fixed-prevalence optimization for a small equal-mass grid.
    v=np.array([-2.,-.3,.4,1.7]);ww=np.ones(4)/4;lo,hi,worst,best=risk_interval(.2,v,ww,.5)
    allvalues=[.2-np.mean(np.array(p)*v) for p in itertools.product((0.,1.),repeat=4) if sum(p)==2]
    assert abs(lo-min(allvalues))<1e-12 and abs(hi-max(allvalues))<1e-12
    return dict(max_sharp_epsilon_identity_error=float(np.max(abs(actual-bound))),four_point_gains=counter,
        continuous_bin_mean_b=float(bb.mean()),continuous_bin_C=float(C),opposite_half_interval_gains=[float(C-.25),float(C+.25)],smooth_opposite_gains=smooth,
        smooth_posterior_lipschitz_bound=5.,smoothness_ambiguity_bound=float(5*.5*np.mean(abs(bb))),
        known_prevalence_exhaustive_bounds=[float(lo),float(hi)],equal_logits_gain=0.)

def evaluate():
    commit=json.loads((ROOT/'calibration/selector_commit.json').read_text());assert commit['script_sha256']==sha(__file__)
    for name,value in commit['npz_sha256'].items():assert sha(ROOT/'calibration'/name)==value
    out=ROOT/'results';out.mkdir(exist_ok=False);dump(out/'score_start.json',dict(timestamp_utc=now(),selector_commit_sha256=sha(ROOT/'calibration/selector_commit.json')))
    models=np.load(MODELS);evaluate3(models,out);evaluate4(models,out);dump(out/'mathematical_checks.json',math_checks())
    dump(out/'score_complete.json',dict(timestamp_utc=now(),csv_sha256={p.name:sha(p) for p in out.glob('*.csv')}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','fit','evaluate']);args=ap.parse_args()
    with threadpool_limits(1):globals()[args.action]()
