# Commuting geometry is not a scalar-clock condition

## Claim screened and decision

Claim: if learned damped representation metrics commute with their initial metric and retain fixed eigenvectors, adaptive and frozen-metric prediction trajectories differ only by a scalar clock.

**Rejected by an exact residual-network slice.** Commutation fixes eigendirections but permits their relative weights to change. The resulting preconditioned gradient need not remain collinear. This is elementary linear algebra, not a new mechanism or innovation claim. Conversely, observing eigenvector rotation is not sufficient to infer a beneficial predictive effect.

## Exact relation to the original architecture

Take the original two-block width-8 residual tanh model. Freeze all parameters except first-block A[0,0]=u and A[1,1]=v, whose original flattened indices are 0 and 3. Fix first-block B[0,0]=B[0,1]=2 and all remaining parameters to zero. The second residual block is the identity. On two training inputs x1=(1,0), x2=(0,1), both labeled +1, the predictive logits are

    f1=1+tanh(u), f2=tanh(v).

The second representation coordinate is fixed. The restricted two-column representation Jacobian is exactly diagonal on the predictive rows. Its damped pullback metric is

    M(u,v)=diag(sech(u)^4+lambda, sech(v)^4+lambda).

All M(u,v) commute with M(initial), and their coordinate axes never rotate. This is a restriction with 78 parameters held fixed; the unrestricted 80-parameter gradient flow would leave this slice. No result about full-model performance is claimed.

## Remove the scalar training clock exactly

Let L=(ell(f1)+ell(f2))/2 with ell(f)=log(1+exp(-f)), q=grad L, and s=L(initial)-L(theta). For adaptive or frozen M, integrate

    dtheta/ds = -M^(-1)q / (q^T M^(-1)q).

The continuous equation gives dL/ds=-1. Thus comparison at the same s removes the scalar loss clock by construction. The denominator is positive away from stationarity.

The initial point is (0.15,0.35), lambda=0.1. Four fixed loss drops are checked: 0.02, 0.06, 0.12, 0.18. RK4 steps in s are 0.0002 and 0.0001. The implementation independently checks the closed-form logits and restricted Jacobian against the existing full-model core functions.

At s=0.18, the refined endpoints are

    adaptive: (1.6156741788, 2.3555252408)
    frozen:   (1.7150194865, 2.2057108752).

Both training losses equal 0.22719281215 to numerical precision, but predictive RMS difference is 0.0103410. At a common adaptive state, adaptive and frozen vector fields are not parallel. Therefore their parameter and predictive paths are not merely a rescaled clock despite exact commutation of every metric.

The decisive identity is elementary: a pair of positive definite commuting matrices can rescale different components of the gradient differently. Only collinearity of their action on the actual gradient yields a common local orbit. A metric-wide proportionality condition is sufficient, but is much stronger than commutation.

## Why a geometry-only sign claim also fails

The per-input loss difference, adaptive minus frozen, is approximately (+0.0016806044,-0.0016806044). Consider two population laws on the same two inputs, with deterministic +1 labels and input weights (0.8,0.2) or (0.2,0.8). Their risk contrasts are respectively

    +0.00100836266  (adaptive harms)
    -0.00100836266  (adaptive helps).

The observed training sample containing one of each input has probability 0.32 under either iid population. Thus this witness does not require a deliberate train/test distribution shift. It shows conditional non-identifiability from that observed sample and training geometry alone.

It does **not** show impossibility when the population mixture weights or the known synthetic task law are supplied to the predictor. It also does not demonstrate a surprising generalization mechanism: with equal average training loss, exchanging population weights is an elementary way to reverse the risk ranking. The useful outcome is the elimination of a universal geometry-only criterion.

## Numerical verification and limitations

- Exact core-versus-slice forward and restricted-Jacobian discrepancies: zero.
- Metric commutator norm: zero.
- Maximum loss-clock error over both resolutions and all endpoints: 4.74e-12.
- Maximum risk-contrast change on halving the RK4 step: 2.50e-12.
- Nonparallel directions, different predictions, and opposite population-risk signs occur at every declared endpoint.

The margins greatly exceed observed numerical errors, but this is not an interval-arithmetic proof of finite-precision bounds. The algebraic diagonal metric and embedding are exact; the finite endpoints are numerically verified witnesses. There is no sweep over hidden seeds or selection of the four endpoints from their outcomes.

Source: `experiments/geometry_condition_screen_20261121/commuting.py`. Full parameters, rows and provenance are in `commuting/results.json`.

Decision: reject the commutation/rotation condition as stated; do not promote this ordinary counterexample as an innovation. A usable sign condition still needs task-specific out-of-sample information and control of the gradient's relative components, not merely a spectral summary.
