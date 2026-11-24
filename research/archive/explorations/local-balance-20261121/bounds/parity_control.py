"""Frozen attribution control: global derivatives after removing even harmonics."""
from pathlib import Path
import argparse,csv,json,hashlib,datetime
import numpy as np
ROOT=Path(__file__).resolve().parent

def compute(table,r):
    k=np.tile(np.arange(1,9),2);co=table['balanced']*(k%2==1);gamma=abs(co).sum(1);kap=abs(co)@k;kap2=abs(co)@(k*k);R=table['radii'];D=R+float(table['C_h']);L=float(table['L']);H=float(table['H']);M=512
    residual=abs(np.sum(co*table['bc'],axis=1))
    first=np.pi/(2*M)*(kap*D+gamma*L*R);second=(2*np.pi/M)**2/24*(kap2*D+(2*kap+gamma)*R*L+gamma*R*R*H);E=residual+np.minimum(first,second)
    ix=np.clip(np.searchsorted(R,r,side='right')-1,0,len(R)-2);w=R[ix+1]-R[ix];t=np.clip((r-R[ix])/w,0,1)
    angular=((1-t)*E[ix]+t*E[ix+1])/8;radial=L*t*(1-t)*w*(gamma[ix]+gamma[ix+1])/8
    v=L*np.diff(R)*(gamma[:-1]+gamma[1:]);vertex=np.clip(np.divide(E[1:]-E[:-1]+v,2*v,out=np.zeros_like(v),where=v>0),0,1)
    uniform=float(np.max((E[:-1]+(E[1:]-E[:-1])*vertex+v*vertex*(1-vertex))/8))
    return angular,radial,uniform

def main(action):
    fr=ROOT/'parity_control_freeze.json'
    if action=='freeze':
        if fr.exists():raise FileExistsError(fr)
        fr.write_text(json.dumps(dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),purpose='Separate parity simplification from local derivative bounds; no policy or primary target change.',fresh_states_accessed=False,additional_model_forwards=0),indent=2));return
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==json.loads(fr.read_text())['source_sha256']
    target=ROOT/action/'parity_control.csv';rows=[]
    for seed in range(1,17):
        source=ROOT/'development_inputs'/f'{seed}_policy.npz' if action=='old' else ROOT.parent/'validation/policies'/f'{seed}_continuous.npz'
        table=dict(np.load(source));r=np.linspace(0,3,2049);rho=sum(.5*r/.15**2*np.exp(-(r*r+a*a)/(2*.15**2))*np.i0(r*a/.15**2) for a in [.5,1.2]);a,b,u=compute(table,r)
        integral=lambda x:float(np.sum((x[:-1]+x[1:])*np.diff(r)/2))
        rows.append(dict(seed=seed,global_odd_uniform_bound=u,global_odd_integrated_bound=integral(rho*(a+b)),global_odd_angular_integral=integral(rho*a),global_odd_radial_integral=integral(rho*b),additional_model_forwards=0))
    with target.open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','old','fresh']);main(ap.parse_args().action)
