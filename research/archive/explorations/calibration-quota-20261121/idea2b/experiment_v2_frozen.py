"""Prospective quota/angle factorial with exact deployment forward counts."""
from pathlib import Path
import sys,json,csv,hashlib,datetime,argparse,time,gzip
from collections import defaultdict
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;STUDY=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(STUDY/'source'))
from pullback_feature_learning import core
sys.path.insert(0,str(STUDY/'legacy'/'budgeted-rotation-final'))
from gate import Gate
CONFIG=dict(n_seed=16,models=['gd','pb','hybrid'],budgets=[8192,65536],angles=128,radius_bins=16,folded_bins=8,min_radii=4,batch=256,quota=64,test_n=65536,cal_seed_base=735000,test_seed_base=736000,tie_seed_base=737000,label_seed_base=738000,bootstrap_seed=739000,bootstrap_repeats=10000,default_index=dict(gd=64,pb=51,hybrid=58),iid_cal_seed_base=741000)
SPECS=[(f'{m}_{a}',m,a) for m in ('random','confidence','radius_pi') for a in ('pi','default','global','radial')]+[('folded_pi','folded','pi'),('radius_angle_default','radius_angle','default'),('radius_angle_global','radius_angle','global'),('joint','joint','radial')]
POLICIES=[p[0] for p in SPECS]+['legacy_iid_radius_pi','legacy_iid_folded_pi','raw','full_pi','full_radial']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(v):return hashlib.sha256(v).hexdigest()
def jo(p,x):
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf8')
def co(p,rows):
    with p.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def ci(p):
    with p.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def sample(seed,n):
    rng=np.random.default_rng(seed);r=rng.choice([.5,1.2],n);a=rng.uniform(0,2*np.pi,n)
    return r[:,None]*np.column_stack([np.cos(a),np.sin(a)])+rng.normal(0,.15,(n,2))
def posterior(x):
    r=np.linalg.norm(x,axis=1);b=(.5**2-1.2**2)/(2*.15**2)+np.log(np.i0(1.2*r/.15**2))-np.log(np.i0(.5*r/.15**2))
    return np.exp(-np.logaddexp(0,-b))
def nll(h,p):return (1-p)*np.logaddexp(0,h)+p*np.logaddexp(0,-h)
def predict(t,x):return core.representation(t,x)[:,0]
def rotate(x,a):
    c=np.cos(a);s=np.sin(a);return np.column_stack([c*x[:,0]-s*x[:,1],s*x[:,0]+c*x[:,1]])
def allocate(score,tie):
    z=score.reshape(-1,256);t=tie.reshape(-1,256);order=np.lexsort((t,-z),axis=1);m=np.zeros_like(z,dtype=bool);np.put_along_axis(m,order[:,:64],True,axis=1);assert np.all(m.sum(axis=1)==64);return m.ravel()
def fit(h,r,phase,method):
    n,na=h.shape;edges=np.quantile(r,np.arange(1,16)/16);bins=np.searchsorted(edges,r,side='right');counts=np.bincount(bins,minlength=16)
    soft=np.logaddexp(0,h);gap=np.stack([(.5*(soft+np.roll(soft,-j,axis=1))-np.logaddexp(0,.5*(h+np.roll(h,-j,axis=1)))).mean(axis=1) for j in range(1,65)],axis=1)
    assert np.min(gap)>-1e-12
    glob=gap.mean(axis=0);curves=np.array([gap[bins==b].mean(axis=0) if counts[b]>=4 else glob for b in range(16)])
    folded=np.empty((16,8));cell_counts=np.zeros((16,8),dtype=int);independent_radii=np.zeros_like(cell_counts)
    angles=phase[:,None]+2*np.pi*np.arange(na)[None,:]/na;fb=np.minimum(((angles%np.pi)/np.pi*8).astype(int),7)
    pi=.5*(soft+np.roll(soft,-64,axis=1))-np.logaddexp(0,.5*(h+np.roll(h,-64,axis=1)))
    for b in range(16):
        for a in range(8):
            cells=(bins[:,None]==b)&(fb==a);cell_counts[b,a]=int(cells.sum());independent_radii[b,a]=int(np.any(cells,axis=1).sum())
            folded[b,a]=pi[cells].mean() if independent_radii[b,a]>=4 else curves[b,63]
    return dict(edges=edges,curves=curves,global_curve=glob,global_index=np.array(glob.argmax()+1),radial_index=curves.argmax(axis=1)+1,default_index=np.array(CONFIG['default_index'][method]),folded=folded,radius_counts=counts,cell_counts=cell_counts,cell_independent_radii=independent_radii)
def choose(x,first,tab,name,tie):
    b=np.searchsorted(tab['edges'],np.linalg.norm(x,axis=1),side='right')
    if name.startswith('legacy_iid_'):
        gate=Gate(tab['iid_edges'],tab['iid_radius_scores'],tab['iid_folded_scores'])
        priority=gate.priority(x,'folded' if name=='legacy_iid_folded_pi' else 'radius')
        return allocate(priority,tie),np.full(len(x),np.pi)
    if name in ('full_pi','full_radial'):
        idx=np.full(len(x),64) if name=='full_pi' else tab['radial_index'][b];return np.ones(len(x),bool),idx*2*np.pi/128
    if name=='raw':return np.zeros(len(x),bool),np.zeros(len(x))
    _,mask,angle=next(z for z in SPECS if z[0]==name)
    if angle=='pi':idx=np.full(len(x),64)
    elif angle=='default':idx=np.full(len(x),int(tab['default_index']))
    elif angle=='global':idx=np.full(len(x),int(tab['global_index']))
    else:idx=tab['radial_index'][b]
    if mask=='random':priority=np.zeros(len(x))
    elif mask=='confidence':priority=-abs(first)
    elif mask=='radius_pi':priority=tab['curves'][b,63]
    elif mask=='folded':
        ab=np.minimum(((np.arctan2(x[:,1],x[:,0])%np.pi)/np.pi*8).astype(int),7);priority=tab['folded'][b,ab]
    else:priority=tab['curves'][b,idx-1]
    return allocate(priority,tie),idx*2*np.pi/128

def deploy(x,pred,tab,name,tie):
    first=pred(x);mask,alpha=choose(x,first,tab,name,tie);out=first.copy()
    if mask.any():out[mask]=.5*(first[mask]+pred(rotate(x[mask],alpha[mask])))
    return out,mask

def oldpilot():
    out=ROOT/'old_pilot';out.mkdir(exist_ok=False)
    old=STUDY/'legacy'/'budgeted-rotation-final'/'results'/'models.npz';bank=np.load(old)
    prior=STUDY.parent/'predictive-symmetry-final'/'direction2'/'exploration'
    histories={};sources=[]
    for method in CONFIG['models']:
        curves=[]
        for seed in range(1,17):
            path=prior/f'{seed}_{method}.npz';z=np.load(path);names=z['names'].tolist();curves.append(z['gain_curves'][names.index('exact_global')]);sources.append(dict(file=path.name,sha256=sha(path.read_bytes())))
        curve=np.mean(curves,axis=0);ix=int(curve.argmax()+1);assert ix==CONFIG['default_index'][method]
        histories[method]=curve.tolist()
    jo(out/'historical_defaults.json',dict(default_index=CONFIG['default_index'],mean_gain_curves=histories,source_hashes=sources,old_model_bank_sha256=sha(old.read_bytes())))
    rows=[]
    for seed in (1,2):
        rng=np.random.default_rng(734000+seed);r=np.linalg.norm(sample(734000+seed,512),axis=1);phase=rng.uniform(0,2*np.pi,512)
        x=(r[:,None,None]*np.stack([np.cos(phase[:,None]+np.arange(128)*2*np.pi/128),np.sin(phase[:,None]+np.arange(128)*2*np.pi/128)],axis=-1)).reshape(-1,2)
        for method in CONFIG['models']:
            h=predict(bank[f'{seed}_{method}'],x).reshape(512,128)
            for budget in CONFIG['budgets']:
                nr=budget//128;t=fit(h[:nr],r[:nr],phase[:nr],method);rows.append(dict(seed=seed,method=method,budget=budget,min_radius_count=int(t['radius_counts'].min()),min_folded_independent_radii=int(t['cell_independent_radii'].min()),global_angle=float(t['global_index'])*360/128,radial_unique_angles=len(np.unique(t['radial_index'])),minimum_gain=float(t['curves'].min())))
    co(out/'checks.csv',rows);print('Old-only pilot complete',flush=True)
def freeze():
    jo(ROOT/'freeze.json',dict(timestamp_utc=now(),config=CONFIG,policies=SPECS,code_sha256=sha(Path(__file__).read_bytes()),design_sha256=sha((ROOT/'DESIGN.md').read_bytes()),historical_sha256=sha((ROOT/'old_pilot'/'historical_defaults.json').read_bytes()),old_checks_sha256=sha((ROOT/'old_pilot'/'checks.csv').read_bytes()),core_sha256=sha(Path(core.__file__).read_bytes()),fresh_outcomes_accessed=False))
def amend():
    jo(ROOT/'freeze_addendum.json',dict(timestamp_utc=now(),reason='Pre-outcome calibration sampling fairness: preserve original iid-calibrated strong baseline at equal forward budget.',config=CONFIG,code_sha256=sha(Path(__file__).read_bytes()),prior_freeze_sha256=sha((ROOT/'freeze.json').read_bytes()),addendum_sha256=sha((ROOT/'ADDENDUM.md').read_bytes()),legacy_gate_sha256=sha((STUDY/'legacy'/'budgeted-rotation-final'/'gate.py').read_bytes()),fresh_outcomes_accessed=False))
def calibrate():
    fr=json.loads((ROOT/'freeze_addendum.json').read_text());assert fr['code_sha256']==sha(Path(__file__).read_bytes());out=ROOT/'fresh';out.mkdir(exist_ok=False);pol=out/'policies';pol.mkdir();cal=out/'calibration';cal.mkdir();provenance=[];manifest={}
    for seed in range(1,17):
        path=STUDY/'idea2a'/'fresh'/'training'/f'{seed}_training_states.npz';states=np.load(path)
        rng=np.random.default_rng(CONFIG['cal_seed_base']+seed);r=np.linalg.norm(sample(CONFIG['cal_seed_base']+seed,512),axis=1);phase=rng.uniform(0,2*np.pi,512)
        angles=phase[:,None]+2*np.pi*np.arange(128)[None,:]/128;x=(r[:,None,None]*np.stack([np.cos(angles),np.sin(angles)],axis=-1)).reshape(-1,2)
        for method in CONFIG['models']:
            theta=states[method];t0=time.perf_counter();first=predict(theta,x[:8192]);t1=time.perf_counter();rest=predict(theta,x[8192:]);t2=time.perf_counter();h=np.r_[first,rest].reshape(512,128)
            raw=cal/f'{seed}_{method}.npz';np.savez_compressed(raw,radii=r,phases=phase,h=h,theta=theta);manifest[str(raw.relative_to(out))]=sha(raw.read_bytes())
            for budget in CONFIG['budgets']:
                n=budget//128;start=time.perf_counter();table=fit(h[:n],r[:n],phase[:n],method);fitseconds=time.perf_counter()-start
                dest=pol/f'{seed}_{method}_{budget}.npz';np.savez_compressed(dest,**table);manifest[str(dest.relative_to(out))]=sha(dest.read_bytes())
                provenance.append(dict(seed=seed,method=method,budget=budget,calibration_design='orbit',calibration_radii=n,orbit_nodes=128,model_forwards=budget,first64_shared=True,cal_seed=CONFIG['cal_seed_base']+seed,state_sha256=sha(theta.tobytes()),policy_sha256=manifest[str(dest.relative_to(out))],calibration_seconds=(t1-t0 if budget==8192 else t2-t0)+fitseconds,forward_seconds=t1-t0 if budget==8192 else t2-t0,fit_seconds=fitseconds,min_radius_count=int(table['radius_counts'].min()),min_folded_independent_radii=int(table['cell_independent_radii'].min()),radius_fallbacks=int((table['radius_counts']<4).sum()),folded_fallbacks=int((table['cell_independent_radii']<4).sum()),global_angle_degrees=float(table['global_index'])*360/128))
        iid_x=sample(CONFIG['iid_cal_seed_base']+seed,32768)
        for method in CONFIG['models']:
            theta=states[method]
            for budget in CONFIG['budgets']:
                n=budget//2;calx=iid_x[:n];calls=[]
                def counted(z):calls.append(len(z));return predict(theta,z)
                started=time.perf_counter();gate=Gate.fit(calx,counted);elapsed=time.perf_counter()-started
                assert calls==[n,n]
                dest=pol/f'{seed}_{method}_{budget}_iid.npz'
                np.savez_compressed(dest,iid_edges=gate.edges,iid_radius_scores=gate.radius_scores,iid_folded_scores=gate.folded_scores)
                manifest[str(dest.relative_to(out))]=sha(dest.read_bytes())
                b=np.searchsorted(gate.edges,np.linalg.norm(calx,axis=1));counts=np.bincount(b,minlength=16)
                from gate import angle_bin
                cells=np.bincount(8*b+angle_bin(calx),minlength=128)
                provenance.append(dict(seed=seed,method=method,budget=budget,calibration_design='legacy_iid_antipodes',calibration_radii=n,orbit_nodes=2,model_forwards=budget,first64_shared=True,cal_seed=CONFIG['iid_cal_seed_base']+seed,state_sha256=sha(theta.tobytes()),policy_sha256=manifest[str(dest.relative_to(out))],calibration_seconds=elapsed,forward_seconds='',fit_seconds='',min_radius_count=int(counts.min()),min_folded_independent_radii=int(cells.min()),radius_fallbacks=0,folded_fallbacks=int((cells==0).sum()),global_angle_degrees=180.))
        print(f'calibrated seed{seed}/16',flush=True)
    co(out/'calibration.csv',provenance);jo(out/'policy_commit.json',dict(timestamp_utc=now(),manifest=manifest,calibration_sha256=sha((out/'calibration.csv').read_bytes()),n_policies=len(provenance),test_evaluation_started=False))
def score():
    out=ROOT/'fresh';commit=json.loads((out/'policy_commit.json').read_text());assert not (out/'per_seed.csv').exists()
    for p,h in commit['manifest'].items():assert sha((out/p).read_bytes())==h
    jo(out/'score_start.json',dict(timestamp_utc=now(),policy_commit_sha256=sha((out/'policy_commit.json').read_bytes())))
    rows=[];audits=[];batchrows=[];bench=[];samples=[]
    for seed in range(1,17):
        x=sample(CONFIG['test_seed_base']+seed,CONFIG['test_n']);p=posterior(x);tie=np.random.default_rng(CONFIG['tie_seed_base']+seed).random(len(x));y=(np.random.default_rng(CONFIG['label_seed_base']+seed).random(len(x))<p).astype(float)
        samples.append(dict(seed=seed,test_seed=CONFIG['test_seed_base']+seed,tie_seed=CONFIG['tie_seed_base']+seed,label_seed=CONFIG['label_seed_base']+seed,x_sha256=sha(x.tobytes()),tie_sha256=sha(tie.tobytes()),label_sha256=sha(y.tobytes()),positive_labels=int(y.sum()),n=len(x)))
        for method in CONFIG['models']:
            raw=np.load(out/'calibration'/f'{seed}_{method}.npz');theta=raw['theta'];pred=lambda z:predict(theta,z)
            for budget in CONFIG['budgets']:
                tab=dict(np.load(out/'policies'/f'{seed}_{method}_{budget}.npz'));tab.update(dict(np.load(out/'policies'/f'{seed}_{method}_{budget}_iid.npz')));batches=[dict(seed=seed,method=method,budget=budget,batch=b) for b in range(len(x)//256)];maskhash={}
                for name in POLICIES:
                    start=time.perf_counter();result,mask=deploy(x,pred,tab,name,tie);elapsed=time.perf_counter()-start
                    er=nll(result,p);lr=nll(result,y);be=er.reshape(-1,256).mean(1);bl=lr.reshape(-1,256).mean(1)
                    for b in range(len(batches)):batches[b][name+'_expected']=be[b];batches[b][name+'_sampled']=bl[b]
                    mh=sha(np.packbits(mask).tobytes());maskhash[name]=mh
                    rows.append(dict(seed=seed,method=method,budget=budget,policy=name,expected_risk=float(er.mean()),sampled_label_risk=float(lr.mean()),extra_forwards=int(mask.sum()),base_forwards=len(x),calls_per_input=(len(x)+mask.sum())/len(x),mask_sha256=mh,output_sha256=sha(result.tobytes()),evaluation_seconds=elapsed))
                    calls=[]
                    def counted(z):calls.append(len(z));return pred(z)
                    check,selected=deploy(x[:256],counted,tab,name,tie[:256]);expected=[256] if name=='raw' else ([256,256] if name.startswith('full_') else [256,64])
                    assert calls==expected and np.array_equal(selected,mask[:256]);error=float(np.max(abs(check-result[:256])));assert error<1e-12
                    audits.append(dict(seed=seed,method=method,budget=budget,policy=name,calls=json.dumps(calls),max_output_error=error,selected=int(selected.sum()),expected_selected=0 if name=='raw' else (256 if name.startswith('full_') else 64)))
                for maskname in ('random','confidence','radius_pi'):
                    assert len(set(maskhash[f'{maskname}_{a}'] for a in ('pi','default','global','radial')))==1
                assert maskhash['joint']!= ''
                batchrows.extend(batches)
                # Randomized, warm-started actual batch deployment timing,16batches/repetition.
                timing_rng=np.random.default_rng(740000+seed+budget)
                for rep in range(4):
                    for name in timing_rng.permutation(POLICIES):
                        start=time.perf_counter_ns()
                        for j in range(16):deploy(x[j*256:(j+1)*256],pred,tab,str(name),tie[j*256:(j+1)*256])
                        if rep:bench.append(dict(seed=seed,method=method,budget=budget,policy=str(name),repeat=rep,ms_per_batch=(time.perf_counter_ns()-start)/16/1e6))
        print(f'scored quota seed{seed}/16',flush=True)
    co(out/'per_seed.csv',rows);co(out/'deployment_audit.csv',audits);co(out/'samples.csv',samples);co(out/'latency.csv',bench)
    with gzip.open(out/'per_batch.csv.gz','wt',newline='',encoding='utf8') as f:w=csv.DictWriter(f,list(batchrows[0]));w.writeheader();w.writerows(batchrows)
    jo(out/'score_complete.json',dict(timestamp_utc=now(),n_seed_rows=len(rows),n_batch_rows=len(batchrows),n_audits=len(audits),max_deployment_error=max(r['max_output_error'] for r in audits),policies_immutable=all(sha((out/p).read_bytes())==h for p,h in commit['manifest'].items()),outcome_sha256=sha((out/'per_seed.csv').read_bytes()),batch_sha256=sha((out/'per_batch.csv.gz').read_bytes())))
def summarize():
    rows=ci(ROOT/'fresh'/'per_seed.csv');index={(r['method'],int(r['budget']),r['policy'],int(r['seed'])):r for r in rows};rng=np.random.default_rng(CONFIG['bootstrap_seed']);draw=rng.integers(0,16,(CONFIG['bootstrap_repeats'],16));summ=[];contrasts=[]
    comparisons=[('joint','confidence_pi','primary_joint_confidence'),('joint','legacy_iid_folded_pi','primary_joint_legacy_folded'),('joint','folded_pi','orbit_folded_comparison'),('legacy_iid_folded_pi','folded_pi','iid_vs_orbit_folded'),('legacy_iid_radius_pi','radius_pi_pi','iid_vs_orbit_radius'),('joint','radius_pi_radial','angle_aware_vs_fixed_mask'),('folded_pi','confidence_pi','strong_gate_comparison')]
    for m in ('random','confidence','radius_pi'):
        comparisons += [(f'{m}_{a}',f'{m}_pi','fixed_mask_angle') for a in ('default','global','radial')]
    for a in ('pi','default','global','radial'):
        comparisons += [(f'{m}_{a}',f'random_{a}','fixed_angle_mask') for m in ('confidence','radius_pi')]
    comparisons += [('radius_angle_default','radius_pi_default','angle_aware_mask'),('radius_angle_global','radius_pi_global','angle_aware_mask'),('joint','raw','budgeted_vs_raw')]
    for method in CONFIG['models']:
        for budget in CONFIG['budgets']:
            for policy in POLICIES:
                for metric in ('expected_risk','sampled_label_risk'):
                    z=np.array([float(index[method,budget,policy,k][metric]) for k in range(1,17)]);interval=np.quantile(z[draw].mean(1),[.025,.975]);summ.append(dict(method=method,budget=budget,policy=policy,metric=metric,n_seed=16,mean=z.mean(),ci_lo=interval[0],ci_hi=interval[1],sd=z.std(ddof=1)))
            for low,ref,kind in comparisons:
                for metric in ('expected_risk','sampled_label_risk'):
                    z=np.array([float(index[method,budget,low,k][metric])-float(index[method,budget,ref,k][metric]) for k in range(1,17)]);interval=np.quantile(z[draw].mean(1),[.025,.975]);contrasts.append(dict(method=method,budget=budget,policy=low,reference=ref,comparison=kind,metric=metric,n_seed=16,mean_difference=z.mean(),ci_lo=interval[0],ci_hi=interval[1],lower_risk_seeds=int((z<0).sum()),higher_risk_seeds=int((z>0).sum()),ties=int((z==0).sum())))
    co(ROOT/'summary.csv',summ);co(ROOT/'contrasts.csv',contrasts)
    b=ci(ROOT/'fresh'/'latency.csv');lat=[]
    for method in CONFIG['models']:
        for budget in CONFIG['budgets']:
            for policy in POLICIES:
                z=np.array([float(r['ms_per_batch']) for r in b if r['method']==method and int(r['budget'])==budget and r['policy']==policy]);lat.append(dict(method=method,budget=budget,policy=policy,median_ms=np.median(z),p10_ms=np.quantile(z,.1),p90_ms=np.quantile(z,.9),measurements=len(z)))
    co(ROOT/'latency_summary.csv',lat)
    print(json.dumps([r for r in contrasts if r['method']=='pb' and r['budget']==8192 and r['comparison'].startswith('primary')],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['oldpilot','freeze','amend','calibrate','score','summarize']);a=p.parse_args()
    with threadpool_limits(1):globals()[a.action]()
