"""Explore only archived models, then freeze, fit fresh states, and evaluate.

Stages are explicit. The evaluate stage requires the selectors manifest that
fit writes before importing any population-risk computation.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from pullback_feature_learning import core
from rotation_selectors import calibrated_angle_selection, radii_sample, phi, deploy, rotate

HERE=Path(__file__).resolve().parent
TMP=HERE.parents[1]
LEGACY=HERE.parent/'legacy'
OLD=(LEGACY if (LEGACY/'budgeted-rotation-final').exists() else TMP)/'budgeted-rotation-final'
MECHANISM=(LEGACY if (LEGACY/'rings-mechanism-20261120').exists() else TMP)/'rings-mechanism-20261120'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def sha(value):
    return hashlib.sha256(value).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')


def csvout(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0])); w.writeheader(); w.writerows(rows)


def evaluate_one(theta,cal,model,pair,selected=None,nr=32,na=512):
    polar=module('polar',MECHANISM/'polar.py')
    # Split quadrature at EVERY selector discontinuity and Bayes boundary.
    bounds=np.unique(np.r_[0.,cal.edges[1:-1],polar.decision_radius(),3.])
    bounds=bounds[(bounds>=0)&(bounds<=3)]
    nodes,weights=np.polynomial.legendre.leggauss(nr)
    r=np.concatenate([(a+b)/2+(b-a)/2*nodes for a,b in zip(bounds[:-1],bounds[1:])])
    dr=np.concatenate([(b-a)/2*weights for a,b in zip(bounds[:-1],bounds[1:])])
    w0=.5*dr*polar.class_density(r,.5); w1=.5*dr*polar.class_density(r,1.2)
    unit=np.c_[np.cos(2*np.pi*(np.arange(na)+.5)/na),np.sin(2*np.pi*(np.arange(na)+.5)/na)]
    h=polar.logits_on_circles(theta,r,unit)
    base=polar.integrate(h,r,w0,w1)
    convex=phi(h); mass=w0+w1
    candidate=np.arange(1,65)*2*np.pi/128
    gain=np.array([mass@(convex-phi(.5*(h+np.roll(h,-int(j*na/128),1)))).mean(1)
                   for j in range(1,65)])
    point_gain=np.array([(convex-phi(.5*(h+np.roll(h,-int(j*na/128),1)))).mean(1)
                        for j in range(1,65)]).T
    # Uniform random alpha on the complete 128-angle group. A mixture of
    # two-call policies; identity alpha has mass 1/128 and is included.
    random_gain=(2*gain.sum()-gain[-1])/128
    rows=[]
    fixed={'raw':(0.,None),'pi':(gain[-1],np.full(len(r),np.pi)),
           'random_alpha':(random_gain,None)}
    for name,(g,a) in fixed.items():
        rows.append(dict(pair=pair,model=model,policy=name,risk=base['population_loss']-g,
            gain=g,angular=base['angular_excess'],fraction=g/base['angular_excess'],
            predicted_gain=np.nan,alpha_mean=np.nan,max_identity_error=0.,nr=nr,na=na))
    selected=cal.names if selected is None else selected
    for name in selected:
        alpha=cal.alpha(r,name)
        j=np.rint(alpha/(2*np.pi/128)).astype(int)-1
        actual=mass@point_gain[np.arange(len(r)),j]
        shift=np.rint(alpha/(2*np.pi/na)).astype(int)
        combined=.5*(h+np.take_along_axis(h,(np.arange(na)[None,:]+shift[:,None])%na,axis=1))
        result=polar.integrate(combined,r,w0,w1)
        error=abs(base['population_loss']-result['population_loss']-actual)
        assert error<1e-10 and abs(result['radial_excess']-base['radial_excess'])<1e-10
        rows.append(dict(pair=pair,model=model,policy=name,risk=result['population_loss'],
            gain=actual,angular=base['angular_excess'],fraction=actual/base['angular_excess'],
            predicted_gain=cal.predicted_gains[cal.names.index(name)],
            alpha_mean=float(np.mean(cal.angles[cal.names.index(name)])),
            max_identity_error=error,nr=nr,na=na))
    oracle=int(gain.argmax())
    rows.append(dict(pair=pair,model=model,policy='test_oracle_global_DIAGNOSTIC_ONLY',
        risk=base['population_loss']-gain[oracle],gain=gain[oracle],
        angular=base['angular_excess'],fraction=gain[oracle]/base['angular_excess'],
        predicted_gain=np.nan,alpha_mean=candidate[oracle],max_identity_error=0.,nr=nr,na=na))
    curve=[dict(pair=pair,model=model,angle=float(a),gain=float(g)) for a,g in zip(candidate,gain)]
    return rows,curve


def load_selection(path):
    from rotation_selectors import RotationSelection
    with np.load(path) as d:
        return RotationSelection(list(d['names']),d['edges'],d['angles'],d['predicted_gains'],
            d['gain_curves'],d['radial_gain_curves'],d['radii'],d['phases'],d['h'],d['power'],
            float(d['calibration_seconds']),int(d['forward_examples']),int(d['velocity_examples']))


def explore(out):
    out.mkdir(exist_ok=False)
    bank=np.load(OLD/'results/models.npz')
    rows=[]; curves=[]; costs=[]
    for key in bank.files:
        pair,model=key.split('_'); pair=int(pair)
        radii,phases=radii_sample(603000+pair,512)
        cal=calibrated_angle_selection(bank[key],radii,phases)
        cal.save(out/(key+'.npz'))
        r,c=evaluate_one(bank[key],cal,model,pair,nr=24,na=512)
        rows.extend(r); curves.extend(c)
        costs.append(dict(pair=pair,model=model,seconds=cal.calibration_seconds,
                          forward_examples=cal.forward_examples))
        print('explore',key,flush=True)
    csvout(out/'risks.csv',rows); csvout(out/'curves.csv',curves); csvout(out/'costs.csv',costs)


def freeze():
    path=HERE/'protocol.json'
    if path.exists(): raise FileExistsError(path)
    cfg=dict(name='Independent label-free two-view rotation choice',pairs=16,
        train_seed_base=611000,init_seed_base=612000,cal_seed_base=613000,test_seed_base=614000,
        calibration_radii=512,calibration_angles=128,radial_bins=8,
        truncations=[4,8,16,64],main_policies=['pi','random_alpha','fourier_K8_global',
            'exact_global','exact_radial'],secondary_policies=['fourier_K4_global',
            'fourier_K16_global','fourier_K64_global','energy_K8_global','fourier_K8_radial'],
        training={'gd':{'steps':500,'rate':.5},'pb':{'steps':500,'rate':8,'damping':.1},
            'hybrid':{'pullback_steps':100,'pullback_rate':8,'gd_steps':400,'gd_rate':2}},
        evaluation={'radial_gauss_nodes_per_interval':[32,64],'angles':[512,1024],
            'atol':2e-7,'rtol':2e-6,'r_max':3,'random_alpha':'uniform on 128 rotations',
            'independent_iid_inputs':32768,'iid_draws':'fresh inputs and Bernoulli labels; posterior expected risk also reported'},
        selections='All calibration policies/states are saved and hashed before fresh risk evaluation.',
        hypothesis='Exact unlabeled Jensen-selected global alpha improves fixed pi; radial choice may improve further.',
        failure_preservation='All K=4,8,16,64 weighted/unweighted global/radial proxies retained.',
        scripts={p.name:sha(p.read_bytes()) for p in (HERE/'rotation_selectors.py',HERE/'experiment.py',
            OLD/'generator.py',MECHANISM/'polar.py',Path(core.__file__))})
    dump(path,cfg)
    dump(HERE/'freeze.json',dict(protocol_sha256=sha(path.read_bytes()),
        exploration_sha256=sha((HERE/'exploration/risks.csv').read_bytes()),
        frozen_before_fresh_training=True))
    print('FROZEN',sha(path.read_bytes()),flush=True)


def advance(theta,x,y,method,rate,n):
    for _ in range(n):
        d,_=core.direction(theta,x,y,method=method,damping=.1)
        theta=theta+rate*d
    assert np.isfinite(theta).all()
    return theta


def fit(out):
    out.mkdir(exist_ok=False)
    cfg=json.loads((HERE/'protocol.json').read_text())
    assert sha((HERE/'protocol.json').read_bytes())==json.loads((HERE/'freeze.json').read_text())['protocol_sha256']
    rings=module('generator',OLD/'generator.py').rings
    states={}; rows=[]
    for pair in range(1,cfg['pairs']+1):
        start=perf_counter(); x,y=rings(cfg['train_seed_base']+pair)
        initial=core.initialize_parameters(cfg['init_seed_base']+pair)
        prefix=advance(initial,x,y,'pullback',8,100)
        models=dict(gd=advance(initial,x,y,'gd',.5,500),
                    pb=advance(prefix,x,y,'pullback',8,400),
                    hybrid=advance(prefix,x,y,'gd',2,400))
        np.savez_compressed(out/f'{pair}_training_states.npz',x=x,y=y,initial=initial,prefix=prefix,**models)
        train_seconds=perf_counter()-start
        radii,phases=radii_sample(cfg['cal_seed_base']+pair,cfg['calibration_radii'])
        states[f'{pair}_initial']=initial; states[f'{pair}_prefix']=prefix
        for model,theta in models.items():
            key=f'{pair}_{model}'; states[key]=theta
            cal=calibrated_angle_selection(theta,radii,phases,
                n_angles=cfg['calibration_angles'],bins=cfg['radial_bins'],truncations=cfg['truncations'])
            cal.save(out/(key+'_selector.npz'))
            rows.append(dict(pair=pair,model=model,train_seed=cfg['train_seed_base']+pair,
                init_seed=cfg['init_seed_base']+pair,cal_seed=cfg['cal_seed_base']+pair,
                train_hash=sha(x.tobytes()+y.tobytes()),state_hash=sha(theta.tobytes()),
                cal_hash=sha(radii.tobytes()+phases.tobytes()),
                selector_hash=sha((out/(key+'_selector.npz')).read_bytes()),
                calibration_seconds=cal.calibration_seconds,
                calibration_forward_examples=cal.forward_examples,
                all_three_training_seconds=train_seconds))
            print('fit',key,'train',round(train_seconds,2),'cal',round(cal.calibration_seconds,2),flush=True)
    np.savez_compressed(out/'models.npz',**states)
    csvout(out/'training_and_calibration.csv',rows)
    dump(out/'selectors_frozen.json',dict(protocol_hash=sha((HERE/'protocol.json').read_bytes()),
        state_bank_hash=sha((out/'models.npz').read_bytes()),
        selector_hashes={p.name:sha(p.read_bytes()) for p in sorted(out.glob('*_selector.npz'))},
        labels_or_population_risks_used_in_calibration=False,
        python=platform.python_version(),numpy=np.__version__))


def evaluate(out):
    frozen=json.loads((out/'selectors_frozen.json').read_text())
    assert sha((out/'models.npz').read_bytes())==frozen['state_bank_hash']
    bank=np.load(out/'models.npz'); rows=[]; curves=[]; checks=[]; iid=[]
    for filename,digest in frozen['selector_hashes'].items():
        path=out/filename
        assert sha(path.read_bytes())==digest
        key=filename.replace('_selector.npz',''); pair,model=key.split('_'); pair=int(pair)
        cal=load_selection(path)
        r,c=evaluate_one(bank[key],cal,model,pair,nr=32,na=512)
        rr,cc=evaluate_one(bank[key],cal,model,pair,nr=64,na=1024)
        delta=max(abs(a['risk']-b['risk']) for a,b in zip(r,rr))
        if delta>2e-7+2e-6*max(x['risk'] for x in rr):
            rrr,ccc=evaluate_one(bank[key],cal,model,pair,nr=96,na=2048)
            delta=max(abs(a['risk']-b['risk']) for a,b in zip(rr,rrr))
            rr,cc=rrr,ccc
        assert delta<2e-7+2e-6*max(x['risk'] for x in rr), (key,delta)
        rows.extend(rr); curves.extend(cc)
        checks.append(dict(pair=pair,model=model,max_risk_grid_difference=delta,
            max_identity_error=max(x['max_identity_error'] for x in rr),
            calibration_label_reads=0,deployment_model_evaluations=2))
        iid.extend(iid_evaluate(bank[key],cal,pair,model))
        print('evaluate',key,'grid_delta',delta,flush=True)
    csvout(out/'risks.csv',rows); csvout(out/'population_gain_curves.csv',curves)
    csvout(out/'numerical_audits.csv',checks)
    csvout(out/'independent_iid_test.csv',iid)


def iid_evaluate(theta,cal,pair,model):
    """Independent fresh inputs; no selection or tuning occurs in this function."""
    polar=module('polar_iid',MECHANISM/'polar.py')
    rng=np.random.default_rng(614000+pair)
    n=32768
    y=rng.integers(0,2,n)
    radius=np.where(y==0,.5,1.2)
    angle=rng.uniform(0,2*np.pi,n)
    x=radius[:,None]*np.c_[np.cos(angle),np.sin(angle)]+rng.normal(0,.15,(n,2))
    posterior=np.exp(-np.logaddexp(0,-polar.bayes_logit(np.linalg.norm(x,axis=1))))
    h=core.representation(theta,x)[:,0]
    predictions={'raw':h,'pi':.5*(h+core.representation(theta,-x)[:,0]),
        'random_alpha':.5*(h+core.representation(theta,rotate(x,rng.integers(0,128,n)*2*np.pi/128))[:,0])}
    for name in cal.names:
        predictions[name]=deploy(theta,x,cal,name)
    records=[]
    raw_expected=np.logaddexp(0,h)-posterior*h
    raw_sampled=np.logaddexp(0,h)-y*h
    for name,pred in predictions.items():
        expected=np.logaddexp(0,pred)-posterior*pred
        sampled=np.logaddexp(0,pred)-y*pred
        records.append(dict(pair=pair,model=model,policy=name,n=n,test_seed=614000+pair,
            input_hash=sha(x.tobytes()),label_hash=sha(y.tobytes()),
            expected_risk=float(expected.mean()),sampled_label_risk=float(sampled.mean()),
            expected_gain=float((raw_expected-expected).mean()),
            expected_gain_se=float((raw_expected-expected).std(ddof=1)/np.sqrt(n)),
            sampled_gain=float((raw_sampled-sampled).mean()),
            sampled_gain_se=float((raw_sampled-sampled).std(ddof=1)/np.sqrt(n))))
    return records


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['explore','freeze','fit','evaluate'])
    ap.add_argument('--out',type=Path); args=ap.parse_args()
    with threadpool_limits(limits=1):
        if args.stage=='freeze': freeze()
        else: globals()[args.stage](args.out or HERE/('exploration' if args.stage=='explore' else 'fresh_results'))

