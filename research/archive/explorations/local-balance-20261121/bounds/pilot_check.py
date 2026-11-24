import json
import numpy as np
from threadpoolctl import threadpool_limits
import local,checks

with threadpool_limits(1):
    output=[]
    for seed in [1,2,8,16]:
        table=dict(np.load(local.PRIOR/f'continuous/fresh/policies/{seed}_policy.npz'))
        bound=dict(np.load(local.ROOT/f'pilot/{seed}_4_256.npz'))
        output.append(dict(seed=seed,**checks.verify(table,bound,947000+seed)))
    (local.ROOT/'pilot/numerical_checks.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))
