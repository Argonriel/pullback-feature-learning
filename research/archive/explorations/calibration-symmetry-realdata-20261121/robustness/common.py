from pathlib import Path
from types import SimpleNamespace
import csv,datetime,hashlib,importlib.util,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def jo(path,obj):
    with Path(path).open('x',encoding='utf8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def co(path,rows):
    with Path(path).open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
def ci(path):
    with Path(path).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))
def components():
    task=load('robustness_compact_task',ROOT/'training'/'reference'/'task.py')
    policy=load('robustness_even_policy',HERE/'reference'/'baseline.py')
    adapter=SimpleNamespace(name='two_invariants',quotient=task.quotient,dimension=4,
        coordinates=task.coordinates,partner=task.partner)
    return task,policy,adapter
def predict_chunks(task,theta,x,beta=1.,chunk=8192):
    h=np.empty(len(x));calls=[]
    for start in range(0,len(x),chunk):
        stop=min(start+chunk,len(x));h[start:stop]=beta*task.predict(theta,x[start:stop]);calls.append(stop-start)
    return h,calls
