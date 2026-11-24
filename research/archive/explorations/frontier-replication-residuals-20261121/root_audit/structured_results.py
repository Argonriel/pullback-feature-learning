"""Independent density/label evidence, all moments and truth calculations."""
import importlib.util,itertools,math,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
from common import ROOT,read,get,sha,record,synthetic_forward
B=ROOT/'structured';DELTA=.05/8622
ash=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sample(seed,n,law):
    rng=np.random.default_rng(seed);radii=rng.uniform(.4,1.4,(n,2));pieces=[];accepted=0
    maximum=abs(law['theta'][0])+abs(law['theta'][1])+abs(law['residual'])
    while accepted<n:
        count=max(1024,int((n-accepted)*(1+maximum)*1.03)+8)
        phi=rng.uniform(0,2*np.pi,count);u=rng.random(count)
        tilt=law['theta'][0]*np.cos(phi)+law['theta'][1]*np.cos(3*phi)+law['residual']*np.cos(5*phi)
        good=phi[u<=(1+tilt)/(1+maximum)];pieces.append(good);accepted+=len(good)
    phi=np.concatenate(pieces)[:n]
    return np.stack([radii[:,0]*np.cos(phi),radii[:,0]*np.sin(phi),radii[:,1]*np.cos(2*phi),radii[:,1]*np.sin(2*phi)],axis=1)
def features(x):
    u=x[:,0]/np.linalg.norm(x[:,:2],axis=1)
    return np.column_stack([u,4*u**3-3*u,16*u**5-20*u**3+5*u])
def cells(x):
    a=(np.linalg.norm(x[:,:2],axis=1)>=.9).astype(int);b=(np.linalg.norm(x[:,2:],axis=1)>=.9).astype(int)
    right=(x[:,0]>0)|((x[:,0]==0)&(x[:,1]>=0))
    return 2*(2*a+b)+right
def geometry(h,ht):
    d=(h-ht)/2;J=np.maximum(0.,(np.logaddexp(0,h)+np.logaddexp(0,ht))/2-np.logaddexp(0,(h+ht)/2))
    return J,abs(d),(np.logaddexp(0,h)-np.logaddexp(0,ht))/2-d/2
def posterior_cell(cell,defect):
    radial=cell//2;return np.where((radial==0)|(radial==3),.8,.2)+(2*(cell%2).astype(int)-1)*defect/2
def interval(v,L,side):
    n=len(v);m=v.mean();s=v.var(ddof=1)
    eb=np.sqrt(2*s*np.log(2/DELTA)/n)+7*(L/4)*np.log(2/DELTA)/(3*(n-1))
    mc=L*np.sqrt(np.log(1/DELTA)/(2*n*256))
    bound=max(0,m-eb,m-mc) if side=='lower' else min(L/4,m+eb,m+mc)
    return dict(mean=m,variance=s,eb_radius=eb,mcd_radius=mc,bound=bound,item_range=L,batch_range=L/4)
def masks_prefix(x,ties,grid,folded):
    # Independent three-dimensional interpolation and histogram implementation.
    radii=np.column_stack([np.linalg.norm(x[:,:2],axis=1),np.linalg.norm(x[:,2:],axis=1)])
    xy=x[:,:2].copy();negative=(xy[:,1]<0)|((xy[:,1]==0)&(xy[:,0]<0));xy[negative]*=-1;xy[xy==0]=0
    angle=np.arctan2(xy[:,1],xy[:,0])%np.pi;lower=[];fraction=[]
    for j in range(2):
        axis=grid[f'axis_{j}'];v=np.clip(radii[:,j],axis[0],axis[-1]);lo=np.clip(np.searchsorted(axis,v,side='right')-1,0,len(axis)-2)
        lower.append(lo);fraction.append((v-axis[lo])/(axis[lo+1]-axis[lo]))
    field=grid['folded_jensen'];na=field.shape[-1];position=angle*na/np.pi-.5;ai=np.floor(position).astype(int);af=position-ai
    score=np.zeros(len(x))
    for i,j in itertools.product((0,1),repeat=2):
        weight=(fraction[0] if i else 1-fraction[0])*(fraction[1] if j else 1-fraction[1])
        score+=weight*((1-af)*field[lower[0]+i,lower[1]+j,ai%na]+af*field[lower[0]+i,lower[1]+j,(ai+1)%na])
    conf=folded['config'];bins=np.zeros(len(x),int)
    for j in range(2):bins=int(conf[j])*bins+np.searchsorted(folded[f'edges_{j}'],radii[:,j])
    anglebin=np.minimum((int(conf[-1])*angle/np.pi).astype(int),int(conf[-1])-1)
    fs=folded['scores'][int(conf[-1])*bins+anglebin];answer=[]
    for scores in (score,fs):
        order=np.lexsort((ties.reshape(-1,256),-scores.reshape(-1,256)),axis=1);mask=np.zeros((len(x)//256,256),bool)
        np.put_along_axis(mask,order[:,:64],True,axis=1);answer.append(mask.ravel())
    return answer
def main():
    cfg=get(B/'FREEZE.json')['config'];laws=cfg['laws'];constants={r['pair']:r for r in get(B/'CONSTANTS.json')['records']}
    hashes=0
    for rel,h in get(B/'FREEZE.json')['hashes'].items():assert sha(B/rel)==h;hashes+=1
    evidence=get(B/'EVIDENCE_COMMIT.json');ecost=read(B/'evidence_costs.csv');parameters=evidence['parameters'];labels=evidence['labels']
    max_evidence=max_bound=max_array=max_truth=max_pred=0.;extra=0
    assert cfg['family_events']==8622 and cfg['moment_events']==8576
    for j,law in enumerate(laws):
        x=sample(2431000+j,262144,law);f=features(x);saved=dict(np.load(B/'evidence'/f'parameter_{j}.npz'))
        assert np.array_equal(f,saved['features'])
        cost=next(r for r in ecost if r['kind']=='parameter' and r['law']==law['name']);assert ash(x)==cost['input_sha256']
        estimate=2*f[:,:2].mean(0);radius=np.sqrt(8*np.log(2/DELTA)/len(x))
        assert np.array_equal(estimate,parameters[j]['estimate']) and np.max(abs(np.array(parameters[j]['radius'])-radius))<1e-15
    for lab in labels:
        j=lab['law_index'];k=lab['audit_index'];law=laws[j];x=sample(2436000+100*j+k,131072,law);cell=cells(x)
        prob=posterior_cell(cell,lab['defect']);y=np.random.default_rng(2437000+100*j+k).random(len(x))<prob
        saved=dict(np.load(B/'evidence'/f'labels_{j}_{k}.npz'))
        assert np.array_equal(cell,saved['cell']) and np.array_equal(y,saved['y']) and np.array_equal(prob,saved['probability'])
        cost=next(r for r in ecost if r['kind']=='labels' and r['law']==law['name'] and float(r['defect'])==lab['defect']);assert ash(x)==cost['input_sha256']
        bounds=[]
        for c in range(8):
            n=np.sum(cell==c);phat=y[cell==c].mean();rad=np.sqrt(np.log(2/DELTA)/(2*n));bounds.append((max(0,phat-rad),min(1,phat+rad)))
        eps=min(1,max(max(bounds[2*t+1][1]-bounds[2*t][0],bounds[2*t][1]-bounds[2*t+1][0],0) for t in range(4)))
        assert abs(eps-lab['epsilon_upper'])<1e-15
    moments=read(B/'moment_bounds.csv');curves=read(B/'certificate_curves.csv');stress=read(B/'stress_results.csv')
    ccost=read(B/'certificate_costs.csv');tcost=read(B/'stress_costs.csv');boundmap={}
    recovery=get(B/'STRESS_RECOVERY_COMMIT.json')
    assert recovery['probability_roundoff']['maximum_absolute_excursion']<=1e-12
    for folder,commit in [('evidence',evidence),('certificate',get(B/'CERTIFICATE_COMMIT.json')),('stress_recovery',recovery)]:
        for rel,h in commit['raw_files'].items():assert sha(B/folder/rel)==h;hashes+=1
    for rec in get(B/'HISTORICAL_INPUTS.json')['models']:
        pair=rec['pair'];theta=np.load(B/rec['theta_path']);grid=dict(np.load(B/rec['grid_path']));folded=dict(np.load(B/rec['folded_path']));D=constants[pair]['D'];c=constants[pair]['c']
        assert c==min(.5,max(1e-6,np.quantile(grid['folded_jensen'],.95)))
        for j,law in enumerate(laws):
            x=sample(2432000+100*j+pair,2048*256,law);ties=np.random.default_rng(2433000+100*j+pair).random(len(x));saved=dict(np.load(B/'certificate'/f'{pair}_{j}.npz'))
            cr=next(r for r in ccost if int(r['pair'])==pair and r['law']==law['name']);assert ash(x)==cr['input_sha256'] and ash(ties)==cr['ties_sha256']
            masks=np.unpackbits(saved['packed_masks'],axis=1,count=len(x)).astype(bool);ids=saved['union_ids'];assert np.array_equal(ids,np.flatnonzero(masks.any(0)))
            assert np.all(masks.reshape(2,-1,256).sum(2)==64)
            manual=masks_prefix(x[:4096],ties[:4096],grid,folded)
            for index in (0,1):assert np.array_equal(manual[index],masks[index,:4096])
            h=synthetic_forward(theta,x[ids]);tx=x[ids].copy();tx[:,:2]*=-1;ht=synthetic_forward(theta,tx);extra+=2*len(ids)
            err=max(float(np.max(abs(h-saved['first']))),float(np.max(abs(ht-saved['partner']))));max_pred=max(max_pred,err);assert err<2e-12
            f=features(x[ids]);assert np.array_equal(f,saved['features']) and np.array_equal(cells(x),saved['cell']);J,S,t=geometry(h,ht);assert S.max()<=D+1e-10
            environments={}
            for mode in ('known','estimated'):
                center=np.array(law['theta'] if mode=='known' else parameters[j]['estimate']);rad=np.zeros(2) if mode=='known' else np.array(parameters[j]['radius'])
                central=f[:,:2]@center;uncertainty=abs(f[:,:2])@rad+law['rho'];low=np.clip(central-uncertainty,-1,1);up=np.clip(central+uncertainty,-1,1)
                environments[mode]=(low,up,min(1,np.sum(abs(center)+rad)+law['rho']))
            actual={}
            relevant=[r for r in moments if int(r['pair'])==pair and r['law']==law['name']]
            for policy,mask in zip(('symmetric_jensen','folded'),masks):
                selected=mask[ids];reduce=lambda weight:np.bincount(ids//256,weights=selected*weight,minlength=2048)/256
                actual[policy+'_A']=reduce(np.minimum(J,c));actual[policy+'_B']=reduce(S/2)
            for row in relevant:
                policy=row['policy'];mode=row['mode'];moment=row['moment'];n=int(row['n_batches'])
                key=policy+'_'+moment if moment!='H' else policy+'_'+mode+'_'+row['scenario']
                if moment=='H':
                    eps=float(row['epsilon']);lo,hi,rbar=environments[mode]
                    if key not in actual:
                        H=np.maximum((abs(lo)+(1-abs(lo))*eps)*S/2-lo*t,(abs(hi)+(1-abs(hi))*eps)*S/2-hi*t)
                        selected=masks[0 if policy=='symmetric_jensen' else 1,ids]
                        actual[key]=np.bincount(ids//256,weights=selected*H,minlength=2048)/256
                    L=D*(eps/2+(1-eps/2)*rbar)
                else:L=c if moment=='A' else D/2
                err=float(np.max(abs(actual[key]-saved[key])));max_array=max(max_array,err);assert err<2e-12
                out=interval(actual[key][:n],L,'lower' if moment=='A' else 'upper')
                for field,value in out.items():
                    err=abs(value-float(row[field]));max_bound=max(max_bound,err);assert err<2e-12,(pair,j,key,field)
                boundmap[pair,law['name'],policy,n,moment,mode,row['scenario']]=out['bound']
            # Full independent truth-output replay plus each posterior world's risks.
            x=sample(2434000+100*j+pair,65536,law);ties=np.random.default_rng(2435000+100*j+pair).random(len(x));sv=dict(np.load(B/'stress_recovery'/f'{pair}_{j}.npz'))
            cost=next(r for r in tcost if int(r['pair'])==pair and r['law']==law['name']);assert ash(x)==cost['input_sha256'] and ash(ties)==cost['ties_sha256']
            h=synthetic_forward(theta,x);tx=x.copy();tx[:,:2]*=-1;ht=synthetic_forward(theta,tx);extra+=2*len(x)
            assert max(np.max(abs(h-sv['first'])),np.max(abs(ht-sv['partner'])))<2e-12
            f=features(x);r=f[:,:2]@np.array(law['theta'])+law['residual']*f[:,2];assert np.array_equal(r,sv['true_r'])
            masks=np.unpackbits(sv['packed_masks'],axis=1,count=len(x)).astype(bool);manual=masks_prefix(x[:4096],ties[:4096],grid,folded)
            assert np.array_equal(manual[0],masks[2,:4096]) and np.array_equal(manual[1],masks[3,:4096])
            J,S,t=geometry(h,ht);d=(h-ht)/2;m=(h+ht)/2
            radii=np.column_stack([np.linalg.norm(x[:,:2],axis=1),np.linalg.norm(x[:,2:],axis=1)])
            p0=.5+.4*np.tanh(8*(radii[:,0]-.9))*np.tanh(8*(radii[:,1]-.9))
            assert np.max(abs(p0-sv['p0']))<1e-15
            cell=cells(x)
            nll=lambda v,p:(1-p)*np.logaddexp(0,v)+p*np.logaddexp(0,-v)
            rows=[z for z in stress if int(z['pair'])==pair and z['law']==law['name']]
            for row in rows:
                eps=float(row['epsilon']);world=row['world'];mask=masks[['raw','full','symmetric_jensen','folded'].index(row['policy'])]
                if world=='cell':p=posterior_cell(cell,eps);pt=posterior_cell(cell^1,eps)
                elif world=='orbit_worst':
                    orient=np.where(r==0,f[:,0],r);mean=.5+.5*(1-eps)*np.sign(orient*d);p=mean+eps*np.sign(d)/2;pt=mean-eps*np.sign(d)/2
                else:
                    direction=f[:,0] if world=='cosine' else np.sign(d);amp=np.minimum(eps/2,np.minimum(p0,1-p0));p=p0+amp*direction;pt=p0-amp*direction
                assert min(p.min(),pt.min())>=-1e-12 and max(p.max(),pt.max())<=1+1e-12
                p=np.clip(p,0,1);pt=np.clip(pt,0,1)
                output=np.where(mask,m,h);loss=nll(output,p);gain=nll(h,p)-loss;other=nll(ht,pt)-nll(np.where(mask,m,ht),pt)
                paired=(gain+other)/2;corrected=paired+r*(gain-other)/2;sharp=mask*(J-(abs(r)+(1-abs(r))*eps)*S/2+r*t)
                batch=gain.reshape(-1,256).mean(1);cb=corrected.reshape(-1,256).mean(1)
                values=dict(expected_nll=loss.mean(),expected_error=np.where(output>=0,1-p,p).mean(),direct_gain=batch.mean(),direct_batch_se=batch.std(ddof=1)/np.sqrt(len(batch)),
                    known_density_gain=cb.mean(),known_density_batch_se=cb.std(ddof=1)/np.sqrt(len(cb)),unweighted_paired_gain=paired.mean(),density_correction=(corrected-paired).mean(),sharp_unclipped_mean=sharp.mean(),observed_posterior_tv=np.max(abs(p-pt)))
                for field,value in values.items():
                    err=abs(value-float(row[field]));max_truth=max(max_truth,err);assert err<2e-12,(pair,j,world,field)
            print('Audited structured pair/law',pair,j,flush=True)
    for row in curves:
        pair=int(row['pair']);law=row['law'];policy=row['policy'];n=int(row['n_batches']);scenario=row['scenario'];eps=float(row['epsilon'])
        definition=next(v for v in laws if v['name']==law)
        eta=min(1,2*sum(abs(v) for v in definition['theta'])/np.pi+definition['rho'])
        assert abs(eta-float(row['assumed_tv_upper']))<1e-15
        assert (row['assumptions_covered']=='True')==definition['covered']
        A=boundmap[pair,law,policy,n,'A','shared','shared'];bb=boundmap[pair,law,policy,n,'B','shared','shared']
        for mode in ('known','estimated'):
            value=A-boundmap[pair,law,policy,n,'H',mode,scenario];assert abs(value-float(row[mode+'_lower']))<2e-12
        old=A-eps*bb*1-float(row['D'])*eta;assert abs(old-float(row['old_global_lower']))<2e-12
    assert len(moments)*2==8576
    record('STRUCTURED_RESULTS.json',dict(parameter_intervals=14,cell_intervals=32,independent_labeled_items=524288,
        moment_rows=len(moments),moment_events=8576,total_family_events=8622,curve_rows=len(curves),truth_rows=len(stress),
        max_moment_array_error=max_array,max_bound_error=max_bound,max_truth_metric_error=max_truth,max_forward_error=max_pred,
        verification_streams_regenerated=56,truth_streams_regenerated=56,full_selected_and_truth_forward_replay=True,
        independently_replayed_selector_prefix_batches=16*2*56,additional_audit_forward_examples=extra,committed_files_checked=hashes,
        failed_attempt_forward_overhead=131712,recovery_endpoint_excursion=recovery['probability_roundoff']['maximum_absolute_excursion'],
        limits='Selector masks independently checked on 16-batch prefixes per stream; all retained moments/risks and selected logits replayed. Floating-point audit, not directed-rounding proof.'))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
