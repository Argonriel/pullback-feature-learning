"""Exact arithmetic check of a classical Gaussian moment-error constant."""
from fractions import Fraction as F
from math import comb
from pathlib import Path
import hashlib, json

ROOT = Path(__file__).resolve().parent
OLD = Path(r'C:\Users\liwei\Documents\Research\pullback-feature-learning\explorations\three-candidate-exploration-20261121\universality\validation\results.json')

def add(a,b):
    c=[F(0)]*max(len(a),len(b))
    for i,v in enumerate(a): c[i]+=v
    for i,v in enumerate(b): c[i]+=v
    return c

def mul(a,b):
    c=[F(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b): c[i+j]+=x*y
    return c

def legendre(n):
    p,q=[F(1)],[F(0),F(1)]
    if n==0:return p
    for k in range(1,n):
        r=add([F(0)]+[F(2*k+1,k+1)*x for x in q],[-F(k,k+1)*x for x in p])
        p,q=q,r
    return q

old=json.loads(OLD.read_text(encoding='utf-8'))['positive_quadrature_moment_checks']
rows=[]
for m in range(1,5):
    poly=legendre(2*m)
    assert all(poly[j]==0 for j in range(1,len(poly),2))
    p=[poly[j]/poly[-1] for j in range(0,len(poly),2)]
    orth=[sum(v*F(1,2*(i+j)+1) for i,v in enumerate(p)) for j in range(m)]
    sq=mul(p,p)
    norm=sum(v*F(1,2*i+1) for i,v in enumerate(sq))
    formula=F(2**(4*m),(4*m+1)*comb(4*m,2*m)**2)
    assert not any(orth) and norm==formula
    discrepancy=abs(float(norm)-old[m-1]['first_unmatched_moment'])
    assert discrepancy < 1e-14
    rows.append({'atoms':m,'monic_coefficients':[str(v) for v in p],
                 'orthogonality_exact':not any(orth),'squared_norm_exact':str(norm),
                 'formula_exact':str(formula),'old_float_absolute_error':discrepancy,
                 'nonsharp_moment_max_error_lower_bound':str(norm/(4**m-1))})
out={'scope':'Exact polynomial arithmetic; no new flow experiment; support ceiling is proved in MOMENT_CEILING.md.',
     'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'old_results_sha256':hashlib.sha256(OLD.read_bytes()).hexdigest(),'records':rows,'passed':True}
(ROOT/'moment_constant_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'passed':True,'exact_cases':len(rows),'max_old_float_error':max(r['old_float_absolute_error'] for r in rows)}))
