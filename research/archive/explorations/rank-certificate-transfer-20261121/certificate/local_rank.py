"""Jump-split angular integration and local radial derivative rank bounds."""
from pathlib import Path
import sys, time
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'reference')); sys.path.insert(0,str(ROOT/'source'))
sys.path.insert(0, str(ROOT/'transfer'))
import local_jets as j
from pullback_feature_learning import core
import task as waveform
import known_cdf as kc


def task_core(task):
    return core if task == 'rings' else waveform


def initial(rlo, rhi, plo, phi, task):
    if task == 'rings':
        return j.init_jets(rlo,rhi,plo,phi)
    c=j.trig(plo,phi,'cos');s=j.trig(plo,phi,'sin')
    c2=j.trig(2*plo,2*phi,'cos');s2=j.trig(2*plo,2*phi,'sin')
    r=(rlo,rhi);rc=j.mul(r,c2);rs=j.mul(r,s2)
    def stack(*values):return np.column_stack([v[0] for v in values]),np.column_stack([v[1] for v in values])
    zero=np.zeros_like(rlo);z=(zero,zero)
    return [stack(c,s,rc,rs),stack(z,z,c2,s2),stack(z,z,z,z),
            stack(j.neg(s),c,j.scale(rs,-2),j.scale(rc,2)),
            stack(j.neg(c),j.neg(s),j.scale(rc,-4),j.scale(rs,-4))]


def propagate(theta, inputs, task):
    h=[j.pad(a) for a in inputs]
    for A,B,bias in task_core(task).unpack(theta):
        z=[j.linear(a,A) for a in h];z[0]=(z[0][0]+bias,z[0][1]+bias)
        t,td,tdd=j.tanh_ranges(z[0])
        v=[t,j.mul(td,z[1]),j.add(j.mul(tdd,j.square(z[1])),j.mul(td,z[2])),
           j.mul(td,z[3]),j.add(j.mul(tdd,j.square(z[3])),j.mul(td,z[4]))]
        h=[j.pad(j.add(a,j.scale(j.linear(vv,B),.5))) for a,vv in zip(h,v)]
    return [(a[:,0],b[:,0]) for a,b in h]


def difference_jets(theta,rlo,rhi,plo,phi,task):
    inputs=initial(rlo,rhi,plo,phi,task)
    sign=np.array([-1.,-1.]) if task=='rings' else np.array([-1.,-1.,1.,1.])
    transformed=[(np.minimum(a*sign,b*sign),np.maximum(a*sign,b*sign)) for a,b in inputs]
    a=propagate(theta,inputs,task);b=propagate(theta,transformed,task)
    return [j.scale(j.add(x,j.neg(y)),.5) for x,y in zip(a,b)]


def angular_cells(k,phase,na):
    jumps=2*np.pi*(np.arange(-1,k+2)-phase)/k
    split=np.unique(np.r_[0.,jumps[(jumps>0)&(jumps<2*np.pi)],2*np.pi])
    left=[];right=[];u0=[];u1=[]
    for a,b in zip(split[:-1],split[1:]):
        n=max(1,int(np.ceil((b-a)*na/(2*np.pi))))
        x=np.linspace(a,b,n+1);mid=(a+b)/2;base=np.floor(k*mid/(2*np.pi)+phase)
        left.extend(x[:-1]);right.extend(x[1:])
        u0.extend(np.clip(k*x[:-1]/(2*np.pi)+phase-base,0,1))
        u1.extend(np.clip(k*x[1:]/(2*np.pi)+phase-base,0,1))
    return tuple(np.array(x) for x in (left,right,u0,u1))


def action_integral(table,r,k,phase,dist,qt,task,na,inner):
    if k%2==0:return 0.,0.,0,0
    lo,hi,u0,u1=angular_cells(k,phase,na);h=hi-lo;mid=(lo+hi)/2
    dbox=difference_jets(table['theta'],np.full(len(lo),r),np.full(len(lo),r),lo,hi,task)
    dpoint=difference_jets(table['theta'],np.full(len(lo),r),np.full(len(lo),r),mid,mid,task)
    d=(dpoint[0][0]+dpoint[0][1])/2;d_error=(dpoint[0][1]-dpoint[0][0])/2
    dp=np.maximum(abs(dpoint[3][0]),abs(dpoint[3][1]));dpp=np.maximum(abs(dbox[4][0]),abs(dbox[4][1]))
    location=np.interp(r,table['radii'],table['gain_profile']);width=float(table['width'])
    u=u0[:,None]+(u1-u0)[:,None]*np.arange(inner+1)/inner
    qlo,qhi=kc.q_bracket(dist,location+width*u,qt)
    # Monotone left/right sums enclose each cell's mean inclusion probability.
    qbar=(qlo[:,:-1].mean(1),qhi[:,1:].mean(1))
    center=(qbar[0]+qbar[1])/2;err=(qbar[1]-qbar[0])/2
    variation=np.maximum(0,qhi[:,-1]-qlo[:,0])
    remainder=abs(d)*err+d_error*qhi[:,-1]+dp*variation*h/8+dpp*qhi[:,-1]*h*h/24
    integral=float(np.sum(h*d*center)/(2*np.pi));error=float(np.sum(h*remainder)/(2*np.pi))
    return integral,error,4*len(lo),len(u.ravel())


def action_derivative_interval(table,rlo,rhi,k,phase,dist,qt,task,na,slope):
    if k%2==0:return (0.,0.),(0.,0.),0,0
    lo,hi,u0,u1=angular_cells(k,phase,na);weight=(hi-lo)/(2*np.pi)
    d=difference_jets(table['theta'],np.full(len(lo),rlo),np.full(len(lo),rhi),lo,hi,task)
    a=np.interp([rlo,rhi],table['radii'],table['gain_profile']);w=float(table['width'])
    slow=min(a)+w*u0;shigh=max(a)+w*u1
    q=(kc.q_bracket(dist,slow,qt)[0],kc.q_bracket(dist,shigh,qt)[1])
    qs=kc.q_score_derivative(dist,slow,shigh)
    f=j.mul(q,d[0]);derivative=j.add(j.mul(q,d[1]),j.scale(j.mul(qs,d[0]),slope))
    integ=(float(weight@f[0]),float(weight@f[1]));der=(float(weight@derivative[0]),float(weight@derivative[1]))
    return integ,der,2*len(lo),4*len(lo)


def weighted_pair(table,anchor):
    mix=float(table['rank_balanced_mix'][anchor])
    result=[]
    for component,weight in [(0,mix),(1,1-mix)]:
        if weight>0:
            result.append((int(table['rank_balanced_frequency'][anchor,component]),
                           float(table['rank_balanced_phase'][anchor,component]),weight))
    return result


def construct(table,task='rings',sub=4,na=128,inner=64,cdf_bins=131072,distribution=None):
    start=time.perf_counter();dist=kc.distribution(table,task,cdf_bins) if distribution is None else distribution;qt=kc.q_table()
    edges=table['radii'];r=np.linspace(edges[0],edges[-1],(len(edges)-1)*sub+1)
    node=np.zeros(len(r));error=np.zeros(len(r));jet_calls=0;cdf_calls=0
    cache={}
    for index,rr in enumerate(r):
        parent=min(index//sub,len(edges)-2);t=(rr-edges[parent])/(edges[parent+1]-edges[parent])
        for anchor,aw in [(parent,1-t),(parent+1,t)]:
            if aw<=0:continue
            for k,phase,mix in weighted_pair(table,anchor):
                key=(index,k,phase)
                if key not in cache:
                    cache[key]=action_integral(table,rr,k,phase,dist,qt,task,na,inner)
                    jet_calls+=cache[key][2];cdf_calls+=cache[key][3]
                value,err,_,_=cache[key];node[index]+=aw*mix*value;error[index]+=aw*mix*err
    node_seconds=time.perf_counter()-start;dl=np.zeros(len(r)-1);dh=np.zeros_like(dl);even=np.zeros(len(dl),bool)
    for band in range(len(dl)):
        parent=band//sub;delta=edges[parent+1]-edges[parent]
        tl=(r[band]-edges[parent])/delta;th=(r[band+1]-edges[parent])/delta
        a_slope=(table['gain_profile'][parent+1]-table['gain_profile'][parent])/delta
        total=(0.,0.);local_cache={};odd=False
        for anchor,weight,weight_prime in [(parent,(1-th,1-tl),-1/delta),(parent+1,(tl,th),1/delta)]:
            for k,phase,mix in weighted_pair(table,anchor):
                odd |= k%2==1
                key=(k,phase)
                if key not in local_cache:
                    local_cache[key]=action_derivative_interval(table,r[band],r[band+1],k,phase,dist,qt,task,na,a_slope)
                    jet_calls+=local_cache[key][2];cdf_calls+=local_cache[key][3]
                value,der,_,_=local_cache[key]
                endpoints=[]
                for index in (band,band+1):
                    point_key=(index,k,phase)
                    if point_key not in cache:
                        cache[point_key]=action_integral(table,r[index],k,phase,dist,qt,task,na,inner)
                        jet_calls+=cache[point_key][2];cdf_calls+=cache[point_key][3]
                    v,e,_,_=cache[point_key];endpoints.append((v-e,v+e))
                step=r[band+1]-r[band]
                # Uniform range from both endpoint integrals and the enclosed
                # derivative; intersect with the independent direct box range.
                value=(max(value[0],endpoints[0][0]+min(0,der[0])*step,
                           endpoints[1][0]-max(0,der[1])*step),
                       min(value[1],endpoints[0][1]+max(0,der[1])*step,
                           endpoints[1][1]-min(0,der[0])*step))
                assert value[0]<=value[1]+1e-10
                total=j.add(total,j.scale(j.add(j.scale(value,weight_prime),j.mul(weight,der)),mix))
        dl[band],dh[band]=total;even[band]=not odd
    angular=np.maximum(error[:-1],error[1:]);radial=(dh-dl)*np.diff(r)/4
    bound=np.maximum(abs(node[:-1]),abs(node[1:]))+angular+radial
    bound[even]=0;radial[even]=0;angular[even]=0
    cdf,tail=kc.radial_cdf(r,task);mass=np.maximum(0,np.diff(cdf))
    result=dict(r=r,moment=node,node_error=error,derivative_lo=dl,derivative_hi=dh,
        angular=angular,radial=radial,bound=bound,exact_even=even,mass=mass,
        integrated_bound=np.array(mass@bound+tail*bound.max()),uniform_bound=np.array(bound.max()),
        integrated_endpoint=np.array(mass@np.maximum(abs(node[:-1]),abs(node[1:]))),
        integrated_angular=np.array(mass@angular),integrated_radial=np.array(mass@radial),
        interval_jet_propagations=np.array(jet_calls),coordinate_cdf_queries=np.array(cdf_calls),
        node_seconds=np.array(node_seconds),total_seconds=np.array(time.perf_counter()-start),
        task=np.array(task),sub=np.array(sub),na=np.array(na),inner=np.array(inner),cdf_bins=np.array(cdf_bins),
        distribution_mode=np.array(str(dist['task'])),
        integrated_bound_uses_known_radial_law=np.array(True),
        pointwise_bound_uses_known_radial_law=np.array(str(dist['task'])!='empirical_DKW'))
    return result,dist


def pointwise_bound(result,r):
    r=np.asarray(r);grid=result['r'];i=np.clip(np.searchsorted(grid,r,side='right')-1,0,len(grid)-2)
    width=grid[i+1]-grid[i];t=np.clip((r-grid[i])/width,0,1)
    value=(1-t)*result['moment'][i]+t*result['moment'][i+1]
    err=(1-t)*result['node_error'][i]+t*result['node_error'][i+1]
    err+=(result['derivative_hi'][i]-result['derivative_lo'][i])*width*t*(1-t)
    zero=result['exact_even'][i]|(r<grid[0])|(r>grid[-1])
    return np.where(zero,0,value-err),np.where(zero,0,value+err)
