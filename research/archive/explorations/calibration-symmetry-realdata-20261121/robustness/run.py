"""Frozen robust certificate and independent synthetic posterior stress evaluation."""
from pathlib import Path
import argparse,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from common import HERE,ROOT,components,sha,ash,now,jo,co,predict_chunks
from bounds import BATCH,QUOTA,LOOKS,EPSILONS,DELTA,gap,clip_from_grid,moment_bounds,robust_line,perturb,nll

CONFIG=dict(pairs=16,variants=['legacy_raw','pb_temp'],safe=['symmetric_jensen','folded'],
    policies=['raw','full','symmetric_jensen','folded','confidence'],looks=list(LOOKS),primary_batches=2048,
    stress_n=65536,epsilons=list(EPSILONS),worlds=['cosine','adversarial'],family_events=512,delta=DELTA,
    certificate_input_base=2121000,certificate_tie_base=2122000,
    stress_input_base=2124000,stress_tie_base=2125000,bootstrap_seed=2129000,bootstrap_repeats=20000,
    clip_rule='min(.5,max(1e-6,quantile95(grid.folded_jensen)))')

def freeze():
    files=[Path(__file__),HERE/'bounds.py',HERE/'common.py',HERE/'PROTOCOL.md',HERE/'THEORY.md',
        HERE/'reference'/'baseline.py',ROOT/'training'/'reference'/'task.py',ROOT/'training'/'reference'/'core4d.py']
    jo(HERE/'FREEZE.json',dict(timestamp_utc=now(),config=CONFIG,fresh_outcomes_accessed=False,
        fresh_manifest_absent=not (ROOT/'training'/'robustness_inputs.json').exists(),
        hashes={p.relative_to(ROOT).as_posix():sha(p) for p in files}))

def verify(approval=False):
    frozen=json.loads((HERE/'FREEZE.json').read_text());assert frozen['config']==CONFIG
    for p,digest in frozen['hashes'].items():assert sha(ROOT/p)==digest,p
    if approval:
        record=json.loads((HERE/'ROOT_APPROVAL.json').read_text())
        assert record['robustness_freeze_sha256']==sha(HERE/'FREEZE.json')
        assert record['fresh_execution_authorized'] is True

def manifest():
    path=ROOT/'training'/'robustness_inputs.json';data=json.loads(path.read_text())
    rows=data['models'];assert len(rows)==32
    assert {(int(r['pair']),r['variant']) for r in rows}=={(i,v) for i in range(1,17) for v in CONFIG['variants']}
    for row in rows:
        for key,digest in [('theta_path','theta_file_sha256'),('grid_path','grid_sha256'),('folded_path','folded_sha256')]:
            target=(ROOT/row[key]).resolve();assert target.is_relative_to(ROOT.resolve())
            assert sha(target)==row[digest],target
        assert np.isfinite(row['beta']) and row['beta']>0
        if row['variant']=='legacy_raw':assert row['beta']==1.
    return rows,sha(path)

def load_model(row):
    return (np.load(ROOT/row['theta_path'])[row['theta_key']],
        dict(np.load(ROOT/row['grid_path'])),dict(np.load(ROOT/row['folded_path'])))

def prepare():
    verify(True);rows,digest=manifest();task,policy,adapter=components();prepared=[]
    for row in rows:
        theta,grid,folded=load_model(row);L,C=task.network_constants(theta);beta=float(row['beta'])
        prepared.append(dict(pair=row['pair'],variant=row['variant'],beta=beta,c=clip_from_grid(grid),
            L_raw=L,C_raw=C,D=beta*min(1.4+C,1.4*L),grid_sha256=row['grid_sha256']))
    jo(HERE/'CLIP_COMMIT.json',dict(timestamp_utc=now(),manifest_sha256=digest,
        freeze_sha256=sha(HERE/'FREEZE.json'),certificate_draws_accessed=False,records=prepared))

def ready():
    verify(True);rows,digest=manifest();commit=json.loads((HERE/'CLIP_COMMIT.json').read_text())
    assert digest==commit['manifest_sha256']
    records={(r['pair'],r['variant']):r for r in commit['records']}
    return rows,records

def certificate():
    entries,prepared=ready();task,policy,adapter=components();folder=HERE/'certificate';folder.mkdir(exist_ok=False)
    jo(HERE/'certificate_start.json',dict(timestamp_utc=now(),clip_commit_sha256=sha(HERE/'CLIP_COMMIT.json')))
    rows=[];curves=[];costs=[];ledger=[];hashes={};start_all=time.perf_counter()
    for entry in entries:
        start=time.perf_counter();pair=int(entry['pair']);variant=entry['variant'];beta=float(entry['beta'])
        constants=prepared[(pair,variant)];c=constants['c'];D=constants['D'];theta,grid,folded=load_model(entry)
        assert clip_from_grid(grid)==c
        x=task.sample_inputs(CONFIG['certificate_input_base']+pair,max(LOOKS)*BATCH)
        ties=np.random.default_rng(CONFIG['certificate_tie_base']+pair).random(len(x));tx=task.partner(x)
        masks=[policy.select(adapter,x,grid,ties),policy.select_folded(adapter,x,folded,ties)]
        transformed=[policy.select(adapter,tx,grid,ties),policy.select_folded(adapter,tx,folded,ties)]
        assert all(np.array_equal(a,b) for a,b in zip(masks,transformed))
        union=masks[0]|masks[1];ids=np.flatnonzero(union)
        h,calls=predict_chunks(task,theta,x[ids],beta);ht,other=predict_chunks(task,theta,tx[ids],beta);calls+=other
        J=np.zeros(len(x));d=np.zeros(len(x));J[ids]=gap(h,ht);d[ids]=.5*abs(h-ht)
        assert d.max()<=D+1e-11
        data={}
        for name,mask in zip(CONFIG['safe'],masks):
            assert np.all(mask.reshape(-1,BATCH).sum(1)==QUOTA)
            A=(mask*np.minimum(J,c)).reshape(-1,BATCH).mean(1)
            B=(mask*d).reshape(-1,BATCH).mean(1)
            data[name+'_A']=A;data[name+'_B']=B
            data[name+'_unclipped']=(mask*J).reshape(-1,BATCH).mean(1)
            data[name+'_fixed_half']=(mask*np.minimum(J,.5)).reshape(-1,BATCH).mean(1)
            for n in LOOKS:
                result=moment_bounds(A[:n],B[:n],c,D)
                result.update(pair=pair,variant=variant,policy=name,
                    empirical_unclipped_gain=float(data[name+'_unclipped'][:n].mean()),
                    fixed_half_mean=float(data[name+'_fixed_half'][:n].mean()))
                rows.append(result)
                for eps,value in zip(EPSILONS,robust_line(result,EPSILONS)):
                    curves.append(dict(pair=pair,variant=variant,policy=name,n_batches=n,epsilon=eps,
                        lower=float(value),A_lower=result['A_lower'],B_upper=result['B_upper']))
        data['union_count']=union.reshape(-1,BATCH).sum(1);data['forward_examples']=2*data['union_count']
        assert int(data['forward_examples'].sum())==sum(calls)
        for b,used in enumerate(data['union_count']):
            ledger.append(dict(pair=pair,variant=variant,batch=b,inputs=BATCH,
                candidate_selected=QUOTA,folded_selected=QUOTA,union_selected=int(used),forwards=2*int(used)))
        path=folder/f'{pair}_{variant}_batches.npz';np.savez_compressed(path,**data);hashes[path.name]=sha(path)
        costs.append(dict(pair=pair,variant=variant,input_seed=CONFIG['certificate_input_base']+pair,
            tie_seed=CONFIG['certificate_tie_base']+pair,coordinate_inputs=len(x),forwards=sum(calls),
            invocations=len(calls),seconds=time.perf_counter()-start,input_sha256=ash(x),tie_sha256=ash(ties),
            candidate_mask_sha256=ash(masks[0]),folded_mask_sha256=ash(masks[1]),mask_T_exact=True))
        print(f'Robust certificate {pair} {variant}',flush=True)
    co(HERE/'certificate_bounds.csv',rows);co(HERE/'certificate_curves.csv',curves)
    co(HERE/'certificate_costs.csv',costs);co(HERE/'certificate_ledger.csv',ledger)
    jo(HERE/'CERTIFICATE_COMMIT.json',dict(timestamp_utc=now(),seconds=time.perf_counter()-start_all,
        n_predictors=len(entries),n_lines=len(rows),family_events=512,uniform_all_epsilon=True,
        manifest=hashes,clip_commit_sha256=sha(HERE/'CLIP_COMMIT.json')))

def stress():
    entries,prepared=ready();task,policy,adapter=components();folder=HERE/'stress';folder.mkdir(exist_ok=False)
    jo(HERE/'stress_start.json',dict(timestamp_utc=now(),clip_commit_sha256=sha(HERE/'CLIP_COMMIT.json')))
    rows=[];costs=[];audits=[];hashes={};start_all=time.perf_counter()
    for entry in entries:
        start=time.perf_counter();pair=int(entry['pair']);variant=entry['variant'];beta=float(entry['beta'])
        theta,grid,folded=load_model(entry);n=CONFIG['stress_n']
        x=task.sample_inputs(CONFIG['stress_input_base']+pair,n);ties=np.random.default_rng(CONFIG['stress_tie_base']+pair).random(n)
        tx=task.partner(x);a,calls=predict_chunks(task,theta,x,beta);b,other=predict_chunks(task,theta,tx,beta);calls+=other
        J=gap(a,b);d=.5*(a-b);m=.5*(a+b);p0=task.posterior(x)
        qcos=x[:,0]/np.linalg.norm(x[:,:2],axis=1)
        masks={'raw':np.zeros(n,bool),'full':np.ones(n,bool),
            'symmetric_jensen':policy.select(adapter,x,grid,ties),
            'folded':policy.select_folded(adapter,x,folded,ties),
            'confidence':policy.allocate(-abs(a),ties)}
        masks_t=dict(masks);masks_t['confidence']=policy.allocate(-abs(b),ties)
        assert np.array_equal(masks['symmetric_jensen'],policy.select(adapter,tx,grid,ties))
        assert np.array_equal(masks['folded'],policy.select_folded(adapter,tx,folded,ties))
        for name in CONFIG['safe']+['confidence']:
            assert np.all(masks[name].reshape(-1,BATCH).sum(1)==QUOTA)
        # Actual independent one-batch deployment replay, separately charged.
        audit_forwards=0
        for name in CONFIG['safe']+['confidence']:
            first=beta*task.predict(theta,x[:BATCH]);audit_calls=[BATCH]
            if name=='symmetric_jensen':mask=policy.select(adapter,x[:BATCH],grid,ties[:BATCH])
            elif name=='folded':mask=policy.select_folded(adapter,x[:BATCH],folded,ties[:BATCH])
            else:mask=policy.allocate(-abs(first),ties[:BATCH])
            second=beta*task.predict(theta,task.partner(x[:BATCH][mask]));audit_calls.append(int(mask.sum()))
            output=first.copy();output[mask]=.5*(first[mask]+second)
            expected=np.where(masks[name][:BATCH],m[:BATCH],a[:BATCH])
            error=float(np.max(abs(output-expected)));assert error<1e-11 and audit_calls==[256,64]
            audit_forwards+=sum(audit_calls)
            audits.append(dict(pair=pair,variant=variant,policy=name,first_calls=256,second_calls=64,
                max_output_error=error,mask_equal=bool(np.array_equal(mask,masks[name][:BATCH]))))
        max_identity=0.
        for world,q in [('cosine',qcos),('adversarial',np.sign(d))]:
            for eps in EPSILONS:
                p,pt,amplitude=perturb(p0,eps,q)
                assert p.min()>=-1e-14 and p.max()<=1+1e-14 and np.max(abs(p-pt))<=eps+1e-14
                raw=nll(a,p);raw_t=nll(b,pt)
                for name in CONFIG['policies']:
                    out=np.where(masks[name],m,a);out_t=np.where(masks_t[name],m,b)
                    loss=nll(out,p);loss_t=nll(out_t,pt)
                    gain=raw-loss;antithetic=.5*(gain+raw_t-loss_t)
                    if name in CONFIG['safe']:
                        identity=masks[name]*(J-.5*(p-pt)*d)
                        error=float(np.max(abs(identity-antithetic)));max_identity=max(max_identity,error)
                        assert error<2e-12
                    batch=antithetic.reshape(-1,BATCH).mean(1)
                    error=(out>=0)*(1-p)+(out<0)*p
                    error_t=(out_t>=0)*(1-pt)+(out_t<0)*pt
                    rows.append(dict(pair=pair,variant=variant,world=world,epsilon=eps,policy=name,
                        expected_nll=float(loss.mean()),expected_error=float(error.mean()),gain=float(gain.mean()),
                        antithetic_nll=float(.5*(loss.mean()+loss_t.mean())),
                        antithetic_error=float(.5*(error.mean()+error_t.mean())),
                        antithetic_gain=float(batch.mean()),batch_mc_se=float(batch.std(ddof=1)/np.sqrt(len(batch))),
                        observed_posterior_gap=float(np.max(abs(p-pt))),known_uniform_gap_bound=eps))
        path=folder/f'{pair}_{variant}.npz'
        np.savez_compressed(path,first=a,partner=b,p0=p0,qcos=qcos,
            policy_names=np.array(CONFIG['policies']),masks=np.packbits(np.stack([masks[name] for name in CONFIG['policies']]),axis=1),
            confidence_partner_mask=np.packbits(masks_t['confidence']))
        hashes[path.name]=sha(path)
        costs.append(dict(pair=pair,variant=variant,input_seed=CONFIG['stress_input_base']+pair,
            tie_seed=CONFIG['stress_tie_base']+pair,inputs=n,shared_research_forwards=sum(calls),
            extra_deployment_audit_forwards=audit_forwards,seconds=time.perf_counter()-start,
            input_sha256=ash(x),tie_sha256=ash(ties),max_safe_identity_error=max_identity))
        print(f'Posterior stress {pair} {variant}',flush=True)
    co(HERE/'stress_results.csv',rows);co(HERE/'stress_costs.csv',costs);co(HERE/'deployment_audits.csv',audits)
    jo(HERE/'STRESS_COMMIT.json',dict(timestamp_utc=now(),seconds=time.perf_counter()-start_all,
        n_rows=len(rows),manifest=hashes,no_realdata_guarantee=True,confidence_is_empirical_only=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','prepare','certificate','stress'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):globals()[args.stage]()
