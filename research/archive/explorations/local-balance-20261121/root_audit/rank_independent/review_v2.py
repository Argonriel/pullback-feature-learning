"""Independent numerical review of the developing global-rank construction."""
from pathlib import Path
import importlib.util,json,math,sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;SEL=ROOT.parents[1]/'selection'
sys.path.insert(0,str(ROOT.parents[1]/'source'))
spec=importlib.util.spec_from_file_location('peer_selection',ROOT/'policy_postphase_review.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def main():
    table=dict(np.load(SEL/'development_v2/bases/121_1.npz'));table.update(dict(np.load(SEL/'development_v2/actions/121_1_tau1.npz')))
    rng=np.random.default_rng(949992);u=np.r_[0.,1.,rng.uniform(0,1,1000)]
    q=sum(float(math.comb(255,j))*u**(255-j)*(1-u)**j for j in range(64));qerror=float(abs(q-p.rank_probability(u)).max());assert qerror<1e-11
    a=np.sort(rng.normal(size=100));weights=rng.random(100);weights/=weights.sum();scores=rng.normal(size=120);width=.7
    direct=np.clip((scores[:,None]-a)/width,0,1)@weights;cdferror=float(abs(direct-p.mixture_cdf(a,weights,scores,width)).max());assert cdferror<1e-12
    phase=np.arange(512);phi=2*np.pi*(phase+.5)/512;checks=[];hull=[]
    for row in [10,32,53,64,90,127]:
        h=table['h'][row];partner=np.roll(h,256);d=(h-partner)/2;A=np.logaddexp(0,h)-np.logaddexp(0,(h+partner)/2)
        for k in [1,2,3,4]:
            for shift in [0,1,17,117,255,511]:
                ideal=((k*(4*phase+2)+4*shift+1)%(2048))/2048
                floating=(k*phi/(2*np.pi)+(shift+.25)/512)%1
                values=[]
                for fraction in [ideal,floating]:
                    q=p.rank_probability(p.score_cdf(table,table['gain_profile'][row]+float(table['width'])*fraction));values.append((float(q@A/512),float(q@d/512)))
                ix=(k-1)*512+shift;fft=np.array([table['action_objectives'][row,ix],table['action_moments'][row,ix]])
                checks.append(dict(row=row,k=k,shift=shift,modular_C_error=abs(values[0][0]-fft[0]),modular_b_error=abs(values[0][1]-fft[1]),floating_C_error=abs(values[1][0]-fft[0]),floating_b_error=abs(values[1][1]-fft[1])))
        b=table['action_moments'][row];C=table['action_objectives'][row];i,j,w,value=p.mixture_at_zero(b,C);pos=np.flatnonzero(b>0);neg=np.flatnonzero(b<0);brute=float(C[b==0].max())
        if len(pos) and len(neg):
            wp=-b[neg][None,:]/(b[pos][:,None]-b[neg][None,:]);brute=max(brute,float((wp*C[pos,None]+(1-wp)*C[neg][None,:]).max()))
        assert abs(value-brute)<1e-12 and abs(w*b[i]+(1-w)*b[j])<1e-12;hull.append(dict(row=row,value_error=abs(value-brute),moment_residual=abs(w*b[i]+(1-w)*b[j])))
    assert max(x['modular_C_error'] for x in checks)<1e-12 and max(x['modular_b_error'] for x in checks)<1e-12
    r=rng.choice([.5,1.2],4096);phi=rng.uniform(-np.pi,np.pi,4096);x=r[:,None]*np.c_[np.cos(phi),np.sin(phi)]+rng.normal(0,.15,(4096,2));keys=rng.random(4096);ties=rng.random(4096);first=np.zeros(4096)
    quota=[]
    for name in p.POLICIES:
        mask=p.select(x,first,table,name,ties,keys);quota.append(dict(policy=name,min_count=int(mask.reshape(-1,256).sum(1).min()),max_count=int(mask.reshape(-1,256).sum(1).max())))
    mask=p.select(x,first,table,'rank_even',ties,keys);flip=x.copy();flip[::3]*=-1;other=p.select(flip,first,table,'rank_even',ties,keys);assert np.array_equal(mask,other)
    result=dict(rank_probability_error=qerror,cdf_formula_error=cdferror,fft_modular_max_C_error=max(x['modular_C_error'] for x in checks),fft_modular_max_b_error=max(x['modular_b_error'] for x in checks),fft_float_wrap_max_C_error=max(x['floating_C_error'] for x in checks),fft_float_wrap_max_b_error=max(x['floating_b_error'] for x in checks),hull_checks=hull,quota_checks=quota,even_partial_flip_mask_identical=True,notes='Quarter-grid phases remove the old wrap-node ambiguity. Both modular and floating-coordinate direct sums agree with FFT. Finite anchor residuals still do not establish exact continuous balance.')
    (ROOT/'review_v2.json').write_text(json.dumps(result,indent=2));(ROOT/'fft_v2.json').write_text(json.dumps(checks,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    with threadpool_limits(1):main()
