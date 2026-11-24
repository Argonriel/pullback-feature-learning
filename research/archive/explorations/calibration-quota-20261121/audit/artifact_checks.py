"""Independent arithmetic and deployment audit; does not call investigator selectors."""
from pathlib import Path
import csv,hashlib,json,sys,datetime
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'source'))
from pullback_feature_learning import core
M=('gd','pb','hybrid');B=(0,128,512,2048,8192,65536);N=(0,1,4,16,64,512)
checks={};errors={}
def rows(p):return list(csv.DictReader((ROOT/p).open(encoding='utf8')))
def js(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def digest(x):return hashlib.sha256(x).hexdigest()
def same(a,b,name,tol=2e-11):
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0));assert err<tol,(name,err);errors[name]=max(errors.get(name,0),err)
def pred(t,x):return core.representation(t,x)[:,0]
def soft(h):return np.logaddexp(0,h)
def sample(seed,n):
    r=np.random.default_rng(seed);rr=r.choice([.5,1.2],n);a=r.uniform(0,2*np.pi,n)
    return rr[:,None]*np.c_[np.cos(a),np.sin(a)]+r.normal(0,.15,(n,2))
def posterior(x):
    r=np.linalg.norm(x,axis=1);v=(.25-1.44)/.045+np.log(np.i0(1.2*r/.0225))-np.log(np.i0(.5*r/.0225))
    return 1/(1+np.exp(-v))
def rotate(x,a):return np.c_[x[:,0]*np.cos(a)-x[:,1]*np.sin(a),x[:,0]*np.sin(a)+x[:,1]*np.cos(a)]
def top(score,tie):
    mask=np.zeros(len(score),bool)
    for k in range(0,len(score),256):mask[k+np.lexsort((tie[k:k+256],-score[k:k+256]))[:64]]=True
    return mask
def canonical_angle(x):
    v=x.copy();v[(v[:,1]<0)|((v[:,1]==0)&(v[:,0]<0))]*=-1
    return np.minimum((8*(np.arctan2(v[:,1],v[:,0])%np.pi)/np.pi).astype(int),7)

def provenance():
    a=js('idea2a/freeze.json')
    for p,h in a['code_hashes'].items():assert digest((ROOT/p).read_bytes())==h,p
    for d,manifest,base in [('idea2a/fresh/policy_commit.json','artifacts','idea2a/fresh'),('idea2a/fresh/evaluation_commit.json','artifacts','idea2a/fresh'),('idea2b/fresh/policy_commit.json','manifest','idea2b/fresh'),('ideas34/calibration/selector_commit.json','npz_sha256','ideas34/calibration')]:
        for p,h in js(d)[manifest].items():assert digest((ROOT/base/p).read_bytes())==h,p
    assert digest((ROOT/'idea2b/experiment.py').read_bytes())==js('idea2b/freeze_phase_addendum.json')['code_sha256']
    assert digest((ROOT/'ideas34/study.py').read_bytes())==js('ideas34/freeze.json')['script_sha256']
    def timestamp(p):
        d=js(p);return datetime.datetime.fromisoformat(d.get('timestamp',d.get('timestamp_utc')))
    for chain in [
      ['idea2a/history/historical_commit.json','idea2a/freeze.json','idea2a/fresh/training_commit.json','idea2a/fresh/policy_commit.json','idea2a/fresh/evaluation_commit.json'],
      ['idea2b/freeze.json','idea2b/freeze_addendum.json','idea2b/freeze_phase_addendum.json','idea2b/fresh/policy_commit.json','idea2b/fresh/score_start.json','idea2b/fresh/score_complete.json'],
      ['ideas34/freeze.json','ideas34/calibration/selector_commit.json','ideas34/results/score_start.json','ideas34/results/score_complete.json']]:
        times=list(map(timestamp,chain));assert times==sorted(times)
    assert not (ROOT/'idea2b/fresh_v2_shared_phase/per_seed.csv').exists()
    checks['frozen_hashes_and_chronology']='pass; superseded phase calibration has no test outcomes'

def check2a():
    scores=np.load(ROOT/'idea2a/fresh/prefix_score_curves.npz')['scores'];hist=np.load(ROOT/'idea2a/history/historical_curves.npz');prior=hist['prior']
    same(prior,hist['curves'].mean(1),'historical_prior');assert np.array_equal(prior.argmax(1),hist['indices'])
    bank=np.load(ROOT/'idea2a/fresh/models.npz');decisions=rows('idea2a/fresh/decisions.csv');di={}
    for row in decisions:
        i=int(row['pair'])-1;m=M.index(row['model']);rep=int(row['repeat'])-1;budget=int(row['budget']);bi=B.index(budget);n=N[bi];pseudo=16 if row['rule']=='shrink16' else 0
        curve=prior[m] if not n else (n*scores[i,m,rep,bi-1]+pseudo*prior[m])/(n+pseudo)
        ix=int(curve.argmax())+1;assert ix==int(row['candidate_index']);same(curve[ix-1],float(row['calibration_score']),'all_choice_scores')
        assert budget==int(row['new_model_forward_budget'])==128*n
        di[i+1,row['model'],row['rule'],budget,rep+1]=ix
    for pair in (1,8,16):
        z=np.load(ROOT/f'idea2a/fresh/orbits/{pair}_calibration_orbits.npz')
        for rep in (0,19):
            r=z['radii'][rep];phase=z['phases'][rep]
            for m,method in enumerate(M):
                h=z['h'][m,rep];a=phase[:8,None]+np.arange(128)*2*np.pi/128;x=(r[:8,None,None]*np.stack([np.cos(a),np.sin(a)],axis=-1)).reshape(-1,2)
                same(pred(bank[f'{pair}_{method}'],x).reshape(8,128),h[:8],'raw_calibration_logits')
                curves=np.array([(.5*(soft(h)+soft(np.roll(h,-j,axis=1)))-soft((h+np.roll(h,-j,axis=1))/2)).mean(1) for j in range(1,65)]).T
                for bi,n in enumerate(N[1:]):same(curves[:n].mean(0),scores[pair-1,m,rep,bi],'nested_prefix_scores')
    pop={ (int(r['pair']),r['model']):r for r in rows('idea2a/fresh/population_models.csv')};gains=np.load(ROOT/'idea2a/fresh/population_gain_curves.npz')['gains'];risks={}
    for row in rows('idea2a/fresh/per_draw_risks.csv'):
        pair=int(row['pair']);method=row['model'];key=(pair,method,row['rule'],int(row['budget']),int(row['repeat']));ix=di[key]
        risk=float(pop[pair,method]['population_loss'])-gains[pair-1,M.index(method),ix-1];same(risk,float(row['population_risk']),'all_population_risks')
        risks.setdefault((method,row['rule'],int(row['budget'])),[]).append(risk)
    for row in rows('idea2a/fresh/summary.csv'):same(np.mean(risks[row['model'],row['rule'],int(row['budget'])]),float(row['mean_population_risk']),'2a_summary_means')
    iid={(int(r['pair']),r['model'],int(r['candidate_index'])):r for r in rows('idea2a/fresh/iid_candidate_risks.csv')}
    for pair in (1,16):
        rng=np.random.default_rng(724000+pair);y=rng.integers(0,2,32768);r=np.where(y==0,.5,1.2);a=rng.uniform(0,2*np.pi,len(y));x=r[:,None]*np.c_[np.cos(a),np.sin(a)]+rng.normal(0,.15,(len(y),2));p=posterior(x)
        for method in M:
            t=bank[f'{pair}_{method}'];first=pred(t,x)
            for ix in (0,1,51,64):
                h=first if ix==0 else (first+pred(t,rotate(x,ix*2*np.pi/128)))/2
                row=iid[pair,method,ix];assert digest(x.tobytes())==row['input_hash'];same(np.mean(soft(h)-y*h),float(row['sampled_risk']),'2a_iid_sampled');same(np.mean(soft(h)-p*h),float(row['expected_risk']),'2a_iid_expected')
    checks['2a']='11520 choices and risks; all summaries; 18 raw orbit/prefix reconstructions; 24 independent IID risks passed'

def check2b():
    data=rows('idea2b/fresh/per_seed.csv');idx={(int(r['seed']),r['method'],int(r['budget']),r['policy']):r for r in data}
    for seed in range(1,17):
      for method in M:
       for budget in (8192,65536):
        for mask in ('random','confidence','radius_pi'):
            assert len({idx[seed,method,budget,mask+'_'+a]['mask_sha256'] for a in ('pi','default','global','radial')})==1
    for r in data:assert float(r['calls_per_input'])==(1 if r['policy']=='raw' else 2 if r['policy'].startswith('full_') else 1.25)
    for row in rows('idea2b/fresh/deployment_audit.csv'):
        expected=[256] if row['policy']=='raw' else [256,256] if row['policy'].startswith('full_') else [256,64]
        assert json.loads(row['calls'])==expected
    for row in rows('idea2b/contrasts.csv'):
        method=row['method'];budget=int(row['budget']);metric=row['metric'];v=[float(idx[s,method,budget,row['policy']][metric])-float(idx[s,method,budget,row['reference']][metric]) for s in range(1,17)]
        same(np.mean(v),float(row['mean_difference']),'2b_all_contrasts')
    for seed in (1,16):
      x=sample(736000+seed,65536);p=posterior(x);ties=np.random.default_rng(737000+seed).random(len(x));y=np.random.default_rng(738000+seed).random(len(x))<p
      for method in M:
       raw=np.load(ROOT/f'idea2b/fresh/calibration/{seed}_{method}.npz');t=raw['theta'];h=pred(t,x)
       same(raw['phases'],np.random.default_rng(745000+seed).uniform(0,2*np.pi,512),'2b_independent_phase')
       for budget in (8192,65536):
        tab=dict(np.load(ROOT/f'idea2b/fresh/policies/{seed}_{method}_{budget}.npz'));tab.update(np.load(ROOT/f'idea2b/fresh/policies/{seed}_{method}_{budget}_iid.npz'))
        rb=np.searchsorted(tab['edges'],np.linalg.norm(x,axis=1),side='right')
        names=[r['policy'] for r in data if int(r['seed'])==seed and r['method']==method and int(r['budget'])==budget]
        for name in names:
            if name=='raw':mask=np.zeros(len(x),bool);angle=np.zeros(len(x))
            else:
                if name.startswith('legacy_iid_'):
                    bins=np.searchsorted(tab['iid_edges'],np.linalg.norm(x,axis=1));score=tab['iid_radius_scores'][bins] if name=='legacy_iid_radius_pi' else tab['iid_folded_scores'][8*bins+canonical_angle(x)];angle=np.full(len(x),np.pi)
                else:
                    kind='radial' if name=='joint' else name.rsplit('_',1)[1]
                    ix=np.full(len(x),64) if kind=='pi' else np.full(len(x),int(tab['default_index'])) if kind=='default' else np.full(len(x),int(tab['global_index'])) if kind=='global' else tab['radial_index'][rb]
                    angle=ix*2*np.pi/128
                    if name.startswith('random_'):score=np.zeros(len(x))
                    elif name.startswith('confidence_'):score=-abs(h)
                    elif name.startswith('radius_pi_'):score=tab['curves'][rb,63]
                    elif name=='folded_pi':score=tab['folded'][rb,np.minimum(((np.arctan2(x[:,1],x[:,0])%np.pi)/np.pi*8).astype(int),7)]
                    else:score=tab['curves'][rb,ix-1]
                mask=np.ones(len(x),bool) if name.startswith('full_') else top(score,ties)
            result=h.copy();result[mask]=(h[mask]+pred(t,rotate(x[mask],angle[mask])))/2
            row=idx[seed,method,budget,name];assert digest(np.packbits(mask).tobytes())==row['mask_sha256'];same(np.mean(soft(result)-p*result),float(row['expected_risk']),'2b_independent_expected');same(np.mean(soft(result)-y*result),float(row['sampled_label_risk']),'2b_independent_sampled')
    checks['2b']='All 2016 call records/fixed-mask groups/contrasts; 252 independently reconstructed deployment risks/masks passed'

def check34():
    bank=np.load(ROOT/'legacy/budgeted-rotation-final/results/models.npz');rng=np.random.default_rng(987711)
    for seed in range(1,17):
        t=bank[f'{seed}_pb'];z=np.load(ROOT/f'ideas34/calibration/{seed}_pb_finite.npz');h=pred(t,z['x']).reshape(16,16);m=(h+np.roll(h,8,axis=1))/2;same(h,z['h'],'4_logits');same(m,z['m'],'4_means');q=z['probabilities'];b=(q*(h-m)).mean(2);c=(q*(soft(h)-soft(m))).mean((1,2));same(b,z['b'],'4_conditional_balance');same(c,z['c'],'4_C');same(b[3],0,'4_exact_balance')
        assert np.all(z['left'].sum(1)==4) and np.all(z['right'].sum(1)==4)
        for radius in range(16):
            bs=z['subsets']@(h[radius]-m[radius])/16;cs=z['subsets']@(soft(h[radius])-soft(m[radius]))/16
            p=z['mix'][radius];ls=z['left'][radius];rs=z['right'][radius];bl=ls@(h[radius]-m[radius])/16;br=rs@(h[radius]-m[radius])/16;cl=ls@(soft(h[radius])-soft(m[radius]))/16;cr=rs@(soft(h[radius])-soft(m[radius]))/16
            if abs(bl-br)>1e-12:
                lam=(cl-cr)/(bl-br);optimal=p*cl+(1-p)*cr;assert np.max(cs-lam*bs)-optimal<1e-10
        interval=np.load(ROOT/f'ideas34/calibration/{seed}_pb_interval.npz');ids=rng.integers(0,len(interval['box_lo']),5000);x=interval['box_lo'][ids]+rng.random((len(ids),2))*(interval['box_hi'][ids]-interval['box_lo'][ids]);a=pred(t,x);partner=pred(t,-x)
        assert np.min(a-interval['a_lo'][ids])>-1e-10 and np.min(interval['a_hi'][ids]-a)>-1e-10
        assert np.min(partner-interval['b_lo'][ids])>-1e-10 and np.min(interval['b_hi'][ids]-partner)>-1e-10
        J=(soft(a)+soft(partner))/2-soft((a+partner)/2)
        for eps in (0,.1,.25,.5,.75,1):assert np.min(J-eps*abs(a-partner)/4-interval['jmin'][ids]+eps*interval['dmax'][ids]/4)>-1e-10
    for row in rows('ideas34/results/idea4_intervals.csv'):
        z=np.load(ROOT/f"ideas34/calibration/{row['model']}_finite.npz");j=list(z['names']).index(row['policy']);b=z['b'][j];c=z['c'][j]
        low=c-np.maximum(b,0).mean() if row['constraint']=='unrestricted' else c-np.sort(b)[8:].sum()/16
        high=c+np.maximum(-b,0).mean() if row['constraint']=='unrestricted' else c-np.sort(b)[:8].sum()/16
        same([low,high],[float(row['lower']),float(row['upper'])],'4_all_identification_intervals')
    for row in rows('ideas34/results/idea3_by_model.csv'):
        if row['policy']=='safe64':assert float(row['gain'])>=-1e-12 and float(row['harmful_selected_fraction'])==0 and float(row['selected_fraction'])<=.25
        if row['policy'] in ('forced64','random64'):assert float(row['model_evaluations_per_input'])==1.25
    checks['ideas34']='All finite tables/128 sharp intervals and mixture quota/LP duals; 80000 random interval-box points across 16 models/6 epsilon bounds passed'

if __name__=='__main__':
    with threadpool_limits(1):
        for f in (provenance,check2a,check2b,check34):f();print(f.__name__,'PASS',flush=True)
    result=dict(status='PASS',independent=True,checks=checks,max_errors=errors)
    (ROOT/'audit/artifacts.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
