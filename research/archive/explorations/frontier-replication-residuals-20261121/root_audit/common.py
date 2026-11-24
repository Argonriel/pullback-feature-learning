"""Independent audit primitives; no experiment-author imports."""
from pathlib import Path
import csv,datetime,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
def read(path):
    with Path(path).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def get(path):return json.loads(Path(path).read_text(encoding='utf8'))
def record(name,obj):
    obj=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**obj)
    if '--record' in sys.argv:(ROOT/'root_audit'/name).write_text(json.dumps(obj,indent=2),encoding='utf8')
    print(json.dumps(obj,indent=2));return obj
def synthetic_forward(theta,x):
    h=np.asarray(x)
    for offset in (0,72):
        a=theta[offset:offset+32].reshape(8,4);b=theta[offset+32:offset+64].reshape(4,8)
        h=h+np.tanh(h@a.T+theta[offset+64:offset+72])@b.T/2
    return h[:,0]
def binary_beta(h,y):
    lo,hi=np.exp(-3),np.exp(3)
    def g(b):return np.mean(-y*h/(1+np.exp(np.clip(y*b*h,-700,700))))
    if g(lo)>=0:return lo
    if g(hi)<=0:return hi
    for _ in range(90):
        mid=(lo+hi)/2
        if g(mid)>0:hi=mid
        else:lo=mid
    return (lo+hi)/2
def multiclass_loss(z,y):
    z=np.asarray(z,np.float64);return np.logaddexp.reduce(z,axis=1)-z[np.arange(len(y)),y]
def multiclass_beta(z,y):
    z=np.asarray(z,np.float64);chosen=z[np.arange(len(y)),y]
    def g(b):
        w=b*z;w-=w.max(1,keepdims=True);q=np.exp(w);q/=q.sum(1,keepdims=True)
        return np.mean(np.sum(q*z,axis=1)-chosen)
    lo,hi=.25,4.
    if g(lo)>=0:return lo
    if g(hi)<=0:return hi
    for _ in range(90):
        mid=(lo+hi)/2
        if g(mid)>0:hi=mid
        else:lo=mid
    return (lo+hi)/2
