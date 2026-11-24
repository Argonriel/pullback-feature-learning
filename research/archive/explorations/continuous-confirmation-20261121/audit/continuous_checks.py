"""Independent Fourier balance, LP dual, exact-quota and risk audit."""
from pathlib import Path
import json,csv,hashlib,sys,datetime
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'source'))
from pullback_feature_learning import core
def sha(x):return hashlib.sha256(x).hexdigest()
def rows(p):return list(csv.DictReader((ROOT/p).open(encoding='utf8')))
def js(p):return json.loads((ROOT/p).read_text())
def sample(seed,n):
    rng=np.random.default_rng(seed);r=rng.choice([.5,1.2],n);a=rng.uniform(0,2*np.pi,n)
    return r[:,None]*np.c_[np.cos(a),np.sin(a)]+rng.normal(0,.15,(n,2))
def basis(a):return np.concatenate([np.cos(a[:,None]*np.arange(1,9)),np.sin(a[:,None]*np.arange(1,9))],1)
def field(tab,x,name):
    r=np.linalg.norm(x,axis=1);j=np.clip(np.floor(r/3*128).astype(int),0,127);t=np.clip((r-tab['radii'][j])/(3/128),0,1);co=(1-t[:,None])*tab[name][j]+t[:,None]*tab[name][j+1];co[r>3]=0
    return (co*basis(np.arctan2(x[:,1],x[:,0]))).sum(1)
def top(score,tie):
    out=np.zeros(len(score),bool)
    for k in range(0,len(score),256):out[k+np.lexsort((tie[k:k+256],-score[k:k+256]))[:64]]=True
    return out
def main():
    frozen=js('continuous/freeze.json');commit=js('continuous/fresh/policy_commit.json')
    for name,value in frozen['hashes'].items():assert sha((ROOT/'continuous'/name).read_bytes())==value,name
    for name,value in commit['manifest'].items():assert sha((ROOT/'continuous/fresh'/name).read_bytes())==value
    sequence=[js(p)['timestamp_utc'] for p in ['continuous/freeze.json','continuous/fresh/policy_commit.json','continuous/fresh/score_start.json','continuous/fresh/score_complete.json']];assert sequence==sorted(sequence)
    records=rows('continuous/fresh/per_seed.csv');ix={(int(r['seed']),r['policy'],r['world']):r for r in records};maxrisk=0.;maxlp=0.;maxquad=0.;replays=0
    for row in rows('continuous/fresh/deployment_audit.csv'):
        assert json.loads(row['calls'])==([256] if row['policy']=='raw' else [256,256] if row['policy']=='full' else [256,64])
    angles=2*np.pi*np.arange(512)/512;psi=basis(angles);odd=np.tile(np.arange(1,9)%2==1,2)
    for seed in range(1,17):
        tab=dict(np.load(ROOT/f'continuous/fresh/policies/{seed}_policy.npz'));h=tab['h'];partner=np.roll(h,256,1);d=(h-partner)/2;change=np.logaddexp(0,h)-np.logaddexp(0,(h+partner)/2);ac=change@psi/512;bc=d@psi/512;bc[:,~odd]=0
        assert np.max(abs(ac-tab['ac']))<1e-12 and np.max(abs(bc-tab['bc']))<1e-12
        for name in ('balanced','unbalanced','even'):
            c=tab[name];assert abs(c).sum(1).max()<=1+1e-12;assert np.all(c[[0,-1]]==0)
        assert np.all(tab['even'][:,odd]==0)
        for j in range(1,128):
            c=tab['balanced'][j];value=c@ac[j];maxlp=max(maxlp,abs(c@bc[j]));sa=np.r_[ac[j],-ac[j]];sb=np.r_[bc[j],-bc[j]]
            lo=max([-np.inf,*((sa[sb>1e-13]-value)/sb[sb>1e-13])]);hi=min([np.inf,*((sa[sb< -1e-13]-value)/sb[sb< -1e-13])]);assert lo<=hi+1e-8 and np.max(sa[abs(sb)<=1e-13],initial=0)<=value+1e-11
        # Independent constants and first-order quadrature bound.
        L=1.;H=0.;C=0.
        for a,b,_ in core.unpack(tab['theta']):
            na=np.linalg.norm(a,2);nb=np.linalg.norm(b,2);ell=1+.5*na*nb;H=ell*H+.5*nb*4/(3*np.sqrt(3))*na*na*L*L;L*=ell;C+=.5*abs(b[0]).sum()
        assert abs(L-float(tab['L']))<1e-9 and abs(H-float(tab['H']))<1e-6
        gamma=abs(tab['balanced']).sum(1);k=np.tile(np.arange(1,9),2);kap=abs(tab['balanced'])@k;quad=np.pi/(2*512)*(kap*(tab['radii']+C)+gamma*L*tab['radii']);maxquad=max(maxquad,float(abs(quad-tab['angular_bound1']).max()))
        diag=dict(np.load(ROOT/f'continuous/fresh/diagnostics/{seed}_moments.npz'));assert np.max(abs(diag['balanced_b'])-diag['angular_bound']-diag['radial_bound'])<1e-10;assert np.max(abs(diag['even_b']))<1e-10
        # Fresh angles and off-anchor radii independently validate moment arrays.
        ridx=np.arange(5,2048,37);r=diag['r'][ridx];phi=2*np.pi*(np.arange(2048)+.371)/2048;points=(r[:,None,None]*np.c_[np.cos(phi),np.sin(phi)][None,:,:]).reshape(-1,2)
        hh=core.representation(tab['theta'],points)[:,0].reshape(len(r),2048);dd=(hh-np.roll(hh,1024,1))/2;uu=field(tab,points,'balanced').reshape(len(r),2048);computed=(uu*dd).mean(1)/8
        assert np.max(abs(computed-diag['balanced_b'][ridx]))<2e-9
        if seed not in (1,8,16):continue
        x=sample(836000+seed,65536);r=np.linalg.norm(x,axis=1);logodds=(.25-1.44)/.045+np.log(np.i0(1.2*r/.0225))-np.log(np.i0(.5*r/.0225));p=np.exp(-np.logaddexp(0,-logodds));ties=np.random.default_rng(837000+seed).random(len(x));coins=np.random.default_rng(838000+seed).random(len(x)//4);y=np.random.default_rng(839000+seed).random(len(x))<p
        first=core.representation(tab['theta'],x)[:,0];other=core.representation(tab['theta'],-x)[:,0];m=(first+other)/2;order=np.argsort(ties.reshape(-1,256),axis=1)+np.arange(len(x)//256)[:,None]*256;groups=order.reshape(-1,4);gate=dict(np.load(ROOT/f'continuous/fresh/policies/{seed}_gate.npz'))
        for name in ('group_balanced','group_unbalanced','group_even','group_uniform','group_confidence','confidence','radius','folded','raw','full'):
            mask=np.zeros(len(x),bool)
            if name.startswith('group_'):
                if name=='group_confidence':chosen=np.argmin(abs(first[groups]),axis=1)
                else:
                    u=np.zeros(len(x)) if name=='group_uniform' else field(tab,x,name[6:]);g=u[groups];prob=.25+(g-g.mean(1,keepdims=True))/6;assert prob.min()>=-1e-12 and np.max(abs(prob.sum(1)-1))<1e-12;cum=np.cumsum(prob,1);cum[:,-1]=1;chosen=(coins[:,None]>=cum).sum(1)
                mask[groups[np.arange(len(groups)),chosen]]=True
            elif name=='full':mask[:]=True
            elif name!='raw':
                if name=='confidence':priority=-abs(first)
                else:
                    bins=np.searchsorted(gate['edges'],r)
                    if name=='radius':priority=gate['radius_scores'][bins]
                    else:
                        canon=np.where(((x[:,1]<0)|((x[:,1]==0)&(x[:,0]<0)))[:,None],-x,x);ab=np.minimum((8*(np.arctan2(canon[:,1],canon[:,0])%np.pi)/np.pi).astype(int),7);priority=gate['folded_scores'][8*bins+ab]
                mask=top(priority,ties)
            if name not in ('raw','full'):assert np.all(mask.reshape(-1,256).sum(1)==64)
            out=np.where(mask,m,first);row=ix[seed,name,'rings'];assert sha(np.packbits(mask).tobytes())==row['mask_sha256'];er=(np.logaddexp(0,out)-p*out).mean();lr=(np.logaddexp(0,out)-y*out).mean();error=max(abs(er-float(row['expected_risk'])),abs(lr-float(row['sampled_label_risk'])));maxrisk=max(maxrisk,error);assert error<3e-12;replays+=1
    # Finite sample bound accounting and all prescribed intervals retain failures.
    for row in rows('continuous/fresh/finite_sample_certificates.csv'):
        same=float(row['C_truncated_sample_mean'])-float(row['hoeffding_epsilon'])-float(row['conditional_uniform_bound']);assert abs(same-float(row['gain_lower']))<1e-11
    result=dict(status='PASS',checked_lp_anchors=16*127,max_lp_balance_residual=maxlp,max_first_order_quadrature_formula_error=maxquad,independent_full_test_masks_and_risks=replays,max_risk_error=maxrisk,checked_fresh_continuous_moment_points=16*len(ridx))
    (ROOT/'audit/continuous.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':
    with threadpool_limits(1):main()
