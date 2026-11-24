"""Independent true-density check of the Poisson-Gamma radial masses."""
import json
import numpy as np
import local

r=np.linspace(0,3,513);cdf,tail=local.rice_cdf(r);nodes,weights=np.polynomial.legendre.leggauss(24)
x=(r[:-1,None]+r[1:,None])/2+np.diff(r)[:,None]*nodes/2
rho=sum(.5*x/.15**2*np.exp(-(x*x+a*a)/(2*.15**2))*np.i0(x*a/.15**2) for a in [.5,1.2])
mass=(rho*weights).sum(axis=1)*np.diff(r)/2
error=float(np.max(abs(np.diff(cdf)-mass)));assert error<1e-13
record=dict(maximum_mass_difference=error,total_quadrature_mass=float(mass.sum()),total_cdf_mass=float(cdf[-1]-cdf[0]),poisson_remainder_bound=tail,scope='Independent numerical check; not directed-rounding verification')
(local.ROOT/'cdf_check.json').write_text(json.dumps(record,indent=2));print(record)
