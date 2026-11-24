"""Finite differences independently check nested-bracket derivatives."""
from pathlib import Path
from fractions import Fraction as Q
import argparse, hashlib, json
import numpy as np
import exact_rank_checks as e

p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
x=e.matrix([[2,'1/3','1/3'],['1/3',2,'1/3'],['1/3','1/3',2]])
g=e.Geometry(x,Q(3)); h=g.h[0]; c=g.bracket(0,1); exact=g.nested(0,0,1)
rows=[]
for delta in (Q(1,100),Q(1,200),Q(1,400)):
    dc=(e.Geometry(x+delta*h,Q(3)).bracket(0,1)-e.Geometry(x-delta*h,Q(3)).bracket(0,1))/(2*delta)
    dh=(e.Geometry(x+delta*c,Q(3)).h[0]-e.Geometry(x-delta*c,Q(3)).h[0])/(2*delta)
    err=max(abs(float(z)) for z in (dc-dh-exact).flat)
    rows.append({'delta':str(delta),'maximum_entry_error':err})
assert rows[0]['maximum_entry_error']>3.9*rows[1]['maximum_entry_error']>3.9**2*rows[2]['maximum_entry_error']
result={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dependency_sha256':hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest(),'exact_nested_bracket':e.rational(exact),'rows':rows}
(args.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
