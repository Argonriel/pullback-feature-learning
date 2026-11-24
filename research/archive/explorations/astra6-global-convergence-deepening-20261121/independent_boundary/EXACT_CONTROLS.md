# Exact controls and rejected extensions

These deterministic examples have explicit solutions or rational KKT certificates. They are counterexamples and scope controls, not claimed discoveries or new training experiments. All statements hold for every fixed `lambda>0` unless stated otherwise. No reserved validation data are involved.

## A. Strict feasibility cannot be dropped from an SPD endpoint or an exponential-loss claim

Take `n=1`, `S=J=R`, `f(x)=x^2/2`, and `x(0)=x0>0`. The only PSD interpolant is zero. The exact equation is

    xdot=-2x^2/(2x+lambda).

Separation of variables gives the strictly decreasing implicit solution

    t=log(x0/x(t))+(lambda/2)(1/x(t)-1/x0).

Independent differential check: the derivative of the right-hand side with respect to `x` is `-(2x+lambda)/(2x^2)`; multiplying by the vector field gives exactly 1. Therefore

    lim t x(t)=lambda/2,
    lim t^2 f(x(t))=lambda^2/8.

The endpoint is singular, no uniform positive spectral lower bound exists along this single trajectory, and no inequality `f(t)<=C exp(-c t)` with finite `C` and `c>0` can hold for all sufficiently large time. The ordinary entropy Hessian-flow interpretation already applies to this scalar example, so the polynomial asymptotic is a control rather than an innovation.

For strictly positive target `b=epsilon`, the unique endpoint is `epsilon`. Linearization of

    xdot=-2x(x-epsilon)/(2x+lambda)

at that endpoint has decay exponent `2 epsilon/(2 epsilon+lambda)`. Thus even among strictly feasible scalar problems, neither an endpoint lower bound nor a positive asymptotic exponential exponent can be uniform as `epsilon` tends to zero. This rejects a target-independent guarantee without a quantitative feasibility margin.

## B. Singular initialization can stick at a nonglobal solution

Take `n=1`, target `b=1`, and `f(x)=(x-1)^2/2`. The factor equation is

    Fdot=(1-F^2)F/(2F^2+lambda).

With the singular initialization `F(0)=0`, its unique solution is `F(t)=0`; thus `x(t)=0` and `f(t)=1/2` for every time. The PSD optimum is `x=1` with zero loss. An initially invertible factor is essential to the expanding-cone argument and cannot be omitted from the theorem.

## C. Zero damping can hit the boundary in finite time despite strict feasibility

Take `n=2`, `J=Sym(2)`, `S=Sym_0(2)`, `X0=I`, and target

    D=diag(a,-a),  a>1,
    f(X)=1/2 ||P_S X-D||_F^2.

This is a classified noncommuting measurement space: `N_J=R I` is central. Strictly positive definite interpolants exist, for example `(a+1)I+D`. For `lambda=0` and while `X>0`, the equation reduces to ordinary Euclidean flow and its exact solution is

    X(t)=I+(1-exp(-t))D.

Its smallest eigenvalue is `1-a(1-exp(-t))`, reaching zero at

    t_hit=-log(1-1/a)<infinity.

At `a=2`, `t_hit=log 2`. Continuing the Euclidean matrix equation beyond this time becomes indefinite. Thus a global SPD-preserving theorem for every compatible start and every strictly feasible target is false at zero damping.

## D. Inconsistent data destroy independence from the observation weight

Let `J=S` be the diagonal two-by-two matrices, let `A X=(X11,X22)`, let the target be `y=(-1,1)`, and start at any positive diagonal `X0`. Fix `0<=r<1` and use

    W_r=[[1,r],[r,1]].

The unique PSD optimum is

    X_r=diag(0,1-r).

Its residual is `(1,-r)` and its measurement gradient is

    W_r(1,-r)^T=(1-r^2,0)^T.

This is nonnegative and complementary to the diagonal optimum, proving PSD KKT optimality. Strict convexity in the two diagonal observations gives the unique diagonal optimum; the zero first diagonal entry forces the off-diagonal entry of any ambient PSD optimum to vanish. Thus the ambient optimum is also unique.

For `r=0`, the endpoint is `diag(0,1)`. For `r=1/2`, it is `diag(0,1/2)`. Both weights are SPD, with eigenvalues `1+-r`. Their endpoints differ. The extended theorem therefore must say “Bregman projection onto the PSD minimizer set for that weight,” not “weight-independent endpoint for arbitrary inconsistent data.”

## E. Failed shortcuts retained

1. **Ambient log-gradient at a singular endpoint.** Invalid: `log X*` is not a finite ambient matrix. Repair: work on the common minimal face and use `log_P X*`.
2. **Pass every normal invariant directly to the boundary.** Invalid: divergent logarithmic terms can occur. Repair: first prove that each difference between projected minimizers annihilates the limiting kernel; only then pass its invariant to the limit.
3. **Declare whole-metric Hessian structure from normal central identities.** Invalid on noncommuting blocks. Repair: analytic factor convergence and the common-face argument do not require such a representation.
4. **Claim weight independence after removing interpolation.** Refuted by example D.
5. **Claim a uniform positive rate after removing strict feasibility.** Refuted by example A.

The examples have no sampling or integration error. A separate small rational arithmetic check is recorded in `exact_checks.json`; the boundary trajectory validation run by the parent is separate and must retain its own numerical tolerances and errors.
