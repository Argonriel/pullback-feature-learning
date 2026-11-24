"""Targeted follow-up after the v1 tiny-increment finite difference hit roundoff.

The initial source and result stay unchanged. Larger increments isolate truncation
error, then a time-step refinement checks the smallest singular value's margin.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import validate_static_range as check


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    data=json.loads(args.input.read_text(encoding='utf-8'))
    x=np.array(data['initial_state']); b=np.array(data['weight'])
    v=np.array(data['target_direction']); db=np.array(data['weight_directions'])[0]
    expected=np.array(data['endpoint_rows'][1]['endpoint_jacobian'])[:,0]
    rows=[]
    for delta in (.04,.02,.01):
        plus=check.run(x,np.diag(x)+.04*v,b+delta*db,[],data['damping'],.02)
        minus=check.run(x,np.diag(x)+.04*v,b-delta*db,[],data['damping'],.02)
        fd=check.skewvec(np.array(plus['endpoint'])-np.array(minus['endpoint']))/(2*delta)
        rows.append({'delta':delta,'derivative':fd.tolist(),
                     'error':float(np.linalg.norm(fd-expected)),
                     'minimum_weight_eigenvalue':float(min(np.linalg.eigvalsh(b+delta*db)[0],
                                                          np.linalg.eigvalsh(b-delta*db)[0]))})
    last=data['endpoint_rows'][-1]
    refined=data['refined_endpoint']
    scaled_step_error=float(np.linalg.norm(np.array(last['endpoint_jacobian'])-
                                          np.array(refined['endpoint_jacobian']))/.02**2)
    min_sv=float(min(last['scaled_singular_values']))
    out={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'dependency_sha256':hashlib.sha256(Path(check.__file__).read_bytes()).hexdigest(),
         'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
         'finite_difference_rows':rows,'scaled_step_error':scaled_step_error,
         'smallest_scaled_singular_value':min_sv,'rank_margin_ratio':min_sv/scaled_step_error,
         'checks':{'fourfold_difference_decay':all(rows[i+1]['error']<rows[i]['error']*.3 for i in range(2)),
                   'rank_margin_over_1000':min_sv>1000*scaled_step_error,
                   'weights_stay_positive':all(r['minimum_weight_eigenvalue']>0 for r in rows)}}
    (args.output/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))


if __name__=='__main__': main()
