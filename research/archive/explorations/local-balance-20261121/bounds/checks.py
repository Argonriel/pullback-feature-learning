"""Numerical spot checks, separate from the real-arithmetic theorem."""
import numpy as np
import local

def exact_jets(theta,r,phi):
    h=r[:,None]*np.c_[np.cos(phi),np.sin(phi)];hr=np.c_[np.cos(phi),np.sin(phi)];hrr=np.zeros_like(h);hp=r[:,None]*np.c_[-np.sin(phi),np.cos(phi)];hpp=-h.copy()
    for A,B,bias in local.core.unpack(theta):
        z=h@A.T+bias;t=np.tanh(z);d=1-t*t;dd=-2*t*d
        zr=hr@A.T;zrr=hrr@A.T;zp=hp@A.T;zpp=hpp@A.T
        h=h+.5*t@B.T;hr=hr+.5*(d*zr)@B.T;hrr=hrr+.5*(dd*zr*zr+d*zrr)@B.T;hp=hp+.5*(d*zp)@B.T;hpp=hpp+.5*(dd*zp*zp+d*zpp)@B.T
    return [h[:,0],hr[:,0],hrr[:,0],hp[:,0],hpp[:,0]]

def verify(table,bound,seed):
    rng=np.random.default_rng(seed);r=bound['r'];nr=len(r)-1;na=int(bound['na']);ids=rng.integers(0,nr,128);j=rng.integers(0,na,128);lo=r[ids];hi=r[ids+1];pa=j*np.pi/na;pz=(j+1)*np.pi/na
    enclosure=local.propagate(table['theta'],local.init_jets(lo,hi,pa,pz));maximum=-np.inf;finitediff_first=0.;finitediff_second=0.
    for _ in range(4):
        rr=rng.uniform(lo,hi);pp=rng.uniform(pa,pz);truth=exact_jets(table['theta'],rr,pp)
        for interval,value in zip(enclosure,truth):maximum=max(maximum,float(np.max(interval[0]-value)),float(np.max(value-interval[1])))
    assert maximum<1e-9
    def value(rr,pp):return local.core.representation(table['theta'],rr[:,None]*np.c_[np.cos(pp),np.sin(pp)])[:,0]
    center=value(rr,pp);convergence=[]
    for step in [2e-4,1e-4,5e-5,2.5e-5,1.25e-5,6.25e-6]:
        finitediff_first=0.;finitediff_second=0.;normalized_first=0.;normalized_second=0.
        for which in [0,1]:
            plus=value(rr+step,pp) if which==0 else value(rr,pp+step);minus=value(rr-step,pp) if which==0 else value(rr,pp-step)
            a=truth[1 if which==0 else 3];b=truth[2 if which==0 else 4]
            finitediff_first=max(finitediff_first,float(np.max(abs((plus-minus)/(2*step)-a))));finitediff_second=max(finitediff_second,float(np.max(abs((plus-2*center+minus)/(step*step)-b))))
            normalized_first=max(normalized_first,float(np.max(abs((plus-minus)/(2*step)-a)/(1+abs(a)))));normalized_second=max(normalized_second,float(np.max(abs((plus-2*center+minus)/(step*step)-b)/(1+abs(b)))))
        convergence.append(dict(step=step,first_error=finitediff_first,second_error=finitediff_second,normalized_first=normalized_first,normalized_second=normalized_second))
    fd_pass=normalized_first<2e-6 and min(v['normalized_second'] for v in convergence)<1e-5
    radii=np.r_[rng.uniform(0,3,128),table['radii']];c=np.column_stack([np.interp(radii,table['radii'],table['balanced'][:,k]) for k in range(16)])
    observed=local.exact_moments(table['theta'],radii,c,1024);lower,upper=local.pointwise_bound(bound,radii);violation=float(max(np.max(lower-observed),np.max(observed-upper)))
    # Exact-even nodes can have numerical cancellation at approximately machine precision.
    assert violation<1e-11
    return dict(interval_cells=128,points_per_cell=4,maximum_jet_enclosure_violation=maximum,first_derivative_fd_error=finitediff_first,second_derivative_fd_error=finitediff_second,finite_difference_diagnostic_pass=fd_pass,finite_difference_convergence=convergence,moment_radii=len(radii),moment_angular_nodes_full_circle=2048,maximum_moment_bound_violation=violation,diagnostic_point_forwards=2*len(radii)*1024)
