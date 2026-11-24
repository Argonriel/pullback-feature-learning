# Independent quantitative review

Date: 2026-11-21 (America/Los_Angeles). Scope: the classified real symmetric matrix flow, with the central-normal property taken as a premise. This note is an independent derivation. It does not certify priority.

## Disposition

The old global convergence theorem is not contradicted. A direct, computable uniform spectral lower bound and exponential convergence from time zero follow for a natural restricted family of quadratic weights. The argument is a standard Bregman comparison argument, not a new global convergence principle. It does **not** extend to arbitrary positive-definite weights by the same Lyapunov function: an explicit 2-by-2 counterexample below disproves that shortcut.

## Precisely quantified setting

Fix a unital matrix Jordan algebra `J subset Sym(n)`, a linear space `S subset J`, and `N=J intersect S-perp subset center(J)`. Assume `J=Jordan(S,I)` only to invoke the already classified decomposition; the calculations themselves need only the stated central-normal property. Fix `lambda>0`, `X0 in J intersect SPD`, and a feasible `Y in J intersect SPD`. Let

`r=P_S(X-Y)`, `f(X)=0.5 <r,B r>`, `G=B r`,

where `B:S->S` is fixed self-adjoint positive definite. Denote its extreme eigenvalues by `beta>0` and `L`. Consider

`Xdot=-M_X G`, `M_X=K_X(K_X+lambda I)^(-1)`, `K_X U=XU+UX`.

The potential is `Phi(X)=0.5 ||X||_F^2 + (lambda/2) tr(X log X-X)`. All Bregman divergences below have the order `D_Phi(Y,X)`, with the moving point in the second argument.

The additional weight hypothesis for the quantitative theorem is

`[B,C_X|_S]=0 for every X in J intersect SPD`,

where `C_X=Hess Phi(X) M_X`. It suffices to impose the following concrete condition. Write the classified decomposition

`S=S_c orthogonal-direct-sum (direct-sum_i (J_i)_0)`,

where `S_c` is the measured center and `(J_i)_0` is the complete traceless component of each non-scalar simple factor (with potentially noncommuting matrix realization). Let `B` preserve this decomposition, be arbitrary SPD on `S_c`, and equal `b_i I`, `b_i>0`, on each `(J_i)_0`. In particular, any single scalar weight `B=bI` is allowed. There is no coupling between these summands under this sufficient condition. This condition is not asserted necessary.

## 1. Exact metric comparison

Let `a=lambda/2`. In an eigenbasis of `X`, the Hessian and the product mobility are simultaneous entrywise multipliers:

`Hess Phi(X)_(ij)=1+a ell(x_i,x_j)`,

`M_X_(ij)=(x_i+x_j)/(x_i+x_j+lambda)`,

where `ell(x,y)=(log x-log y)/(x-y)` with `ell(x,x)=1/x`. The scalar logarithmic-mean inequality

`ell(x,y)>=2/(x+y)`

implies `C_X>=I`. This inequality follows directly from

`ell(x,y)=integral_0^1 1/[(1-s)x+sy] ds >= 1/[(x+y)/2]`

by scalar Jensen. Equality holds when `x=y`. These operators are self-adjoint, commute, and preserve `J`. Preservation of the logarithm derivative follows by differentiating analytic functional calculus in `J`; preservation of `M_X` follows from the Jordan closure of `K_X` and its resolvent.

For every matrix commuting with `X`, `C_X U=U`. Hence `C_X N=N`; self-adjointness implies `C_X S subset S`. It also fixes each central identity and preserves every traceless simple factor. Consequently the concrete weight condition above ensures `[B,C_X|_S]=0` and `C_X B>=B` on `S`.

For a fixed feasible `Y`, differentiation gives the exact formula

`d/dt D_Phi(Y,X) = -<X-Y,C_X G> = -<r,C_X B r>`.

The second equality uses the centrality of the normal component `X-Y-r`, not an assertion that the product metric is a Hessian metric. Under the additional weight hypothesis,

`Ddot <= -<r,B r> = -2 f`.

This is the only added weight restriction in the proof. For arbitrary SPD `B`, `C_X B` need not have a positive quadratic form, even though both factors are SPD.

## 2. Computable spectral enclosure

Set `D0=D_Phi(Y,X0)` and `eta=lambda_min(Y)>0`. Define on `0<u<=eta`

`h_eta(u)=0.5(eta-u)^2 + a [eta log(eta/u)-eta+u]`.

This function is continuous, strictly decreasing on `(0,eta)`, equals zero at `eta`, and diverges at zero. If `D0>0`, define `m` as its unique root `h_eta(m)=D0`; if `D0=0`, define `m=eta`. This is a one-dimensional monotone root problem involving only input data and one supplied feasible witness. Then, throughout the trajectory,

`m I <= X(t) <= R I`, `R=||Y||op+sqrt(2D0)`.

An elementary closed-form, weaker lower bound is

`m_exp=eta exp[-1-2D0/(lambda eta)]`.

### Proof of the lower bound

Diagonalize `X` and write `y_i` for the diagonal entries of `Y` in that basis. Scalar Jensen, applied to the spectral decomposition of `Y`, gives

`tr(Y log Y) >= sum_i y_i log y_i`.

Therefore

`D_Phi(Y,X) >= sum_i {0.5(y_i-x_i)^2 + a[y_i log(y_i/x_i)-y_i+x_i]}`.

Every term is nonnegative. If `mu=lambda_min(X)<=eta`, choose the corresponding index. Its `y_i>=eta`. For fixed `mu`, the scalar summand is increasing in its first argument on `[eta,infinity)` because its derivative is `y_i-mu+a log(y_i/mu)>=0`. Thus `D_Phi(Y,X)>=h_eta(mu)`. Since `D<=D0`, monotonicity yields `mu>=m`. If `mu>eta`, the bound is immediate. Dropping the quadratic part and the nonnegative term `a mu` gives the displayed elementary bound. The upper bound follows from nonnegativity of matrix relative entropy and `0.5||X-Y||_F^2<=D0`.

This proof treats entropy's **derivative** singularity correctly. The ordinary entropy value remains finite at the PSD boundary, whereas the reverse Bregman divergence with an SPD first argument diverges.

## 3. Global existence and rate, without a KL step

The preceding compact SPD enclosure holds before any maximal existence time, and the vector field is smooth on a neighborhood of that compact set. Standard ODE continuation gives global existence. Put

`kappa=2m/(2m+lambda)`.

Then `M_X>=kappa I` and `||G||_F^2>=2 beta f`, so

`f(X(t)) <= f(X0) exp[-2 beta kappa t]`, for every `t>=0`.

Also `||Xdot||_F<=||G||_F<=sqrt(2L f)`. Its integrable tail proves convergence and yields

`||X(t)-X_infinity||_F <= sqrt(2L f(X0))/(beta kappa) exp[-beta kappa t]`.

The limit remains SPD and interpolates. Central conservation gives the original exact Bregman endpoint characterization. Thus this restricted-weight theorem gives explicit constants from time zero and removes the need for bounded analytic factor convergence, the limiting-generator cone argument, and a posteriori eventual positivity. It does not improve the general-weight theorem's quantitative guarantee.

The same conclusion holds for inconsistent data if an SPD cone minimizer `Y` is supplied, with `f` replaced throughout by the excess loss `f-f(Y)`. Interior optimality gives `G(Y)=0`, so the gradient and excess loss have exactly the displayed quadratic forms in `r`. This is a direct restatement, not an additional argument or independent contribution.

For commuting `J`, `C_X=I` and the argument applies to every SPD `B`; this is precisely the familiar Hessian-metric case. In the noncommuting case, the sufficient geometry-adapted weight restriction is essential to this proof, and an unrestricted claim is false.

## 4. Exact counterexample to the arbitrary-weight Lyapunov shortcut

Take `n=2`, `J=Sym(2)`, `S` its complete traceless part, `lambda=2`, and `epsilon=exp(-10)`. In orthonormal measurement coordinates ordered as

`E=(E11-E22)/sqrt(2)`, `F=(E12+E21)/sqrt(2)`,

let

`X=diag(epsilon,1)`,

`Y=X+2E+F+I=[[1+sqrt(2)+epsilon,1/sqrt(2)],[1/sqrt(2),2-sqrt(2)]]`,

`B=[[1,-9/10],[-9/10,1]]`.

Both `X` and `Y` are SPD; `det Y=sqrt(2)-1/2+epsilon(2-sqrt(2))>0`. The loss weight has eigenvalues `1/10,19/10`. The residual coordinates are `r=(-2,-1)`, and the gradient coordinates are `Br=(-11/10,4/5)`. The unmeasured normal space is the central line `R I`, so this is an actual proper classified measurement space. On the measured basis `C_X=diag(1,c)`, where

`c=[(1+epsilon)/(3+epsilon)] [1+10/(1-epsilon)] > 11/3`.

Consequently

`d/dt D_Phi(Y,X) = (4/5)c-11/5 > 11/15 > 0`.

This violates monotonicity at an admissible SPD initial point with a strictly feasible SPD target and a fixed SPD observation weight. It does not violate convergence: the ordinary loss still strictly decreases. No claim about arbitrary SPD weights can be deduced from a falsely assumed reverse-Bregman Lyapunov function.

More generally, if an SPD operator `C` has distinct eigenvalues `c1,c2`, choose a positive correlation `rho` satisfying `2sqrt(c1 c2)/(c1+c2)<rho<1` and let `B=[[1,-rho],[-rho,1]]` on their two-dimensional span. Then the symmetric part of `CB` has negative determinant. Positive definiteness of each factor is therefore intrinsically insufficient; this is not a numerical artifact.

## 5. Minimal independent verification and limitations

`check_quantitative.py` evaluates the exact counterexample, compares the analytic derivative with centered differences of the actual matrix Bregman divergence, and checks the scalar root bound on one explicitly specified noncommuting 2-by-2 traceless-observation trajectory at two adaptive integration tolerances. It does not repeat the prior large-target convergence experiment or introduce a dataset. Runtime, source hash, matrices, all finite-difference errors, and the failed initial Python-alias attempt are retained in `results.json` and `FAILED_ATTEMPTS.md`. Sampled spectral extrema are numerical checks, not continuous-time certificates; the latter follow from the proof.

The exact counterexample has derivative `+0.7335431897150655` while its ordinary loss derivative is `-0.515867255825085`. Centered-difference errors decrease from `5.28e-4` at step `1e-3` to `2.70e-9` at step `1e-6`. The scalar-weight test has `||[X0,Y]||_F=0.735391`, proving it is noncommuting, and gives the computable bound `m=0.0284825515365` and loss exponent `0.0671676557566`. The sampled minimum eigenvalue is `0.598080`; the rigorous bound is conservative and no sharpness claim is made. At integration tolerances `2e-9` and `2e-11`, invariant drift improves from `8.35e-10` to `8.76e-12`. The endpoint refinement difference is `3.62e-10`. Both endpoints are approximately `1.73e-9` from the independently solved invariant endpoint at time 40, with residual `1.72e-9`; finite-time residual dominates that comparison. All checks pass.

The evidence supports a **narrow quantitative supplement**. The spectral lower bound is a known-style consequence of an SPD-witness reverse-Bregman sublevel, and the commuting case is standard Hessian-barrier theory. The specific comparison `Hess Phi M>=I` on this damped noncommuting family may be a convenient lemma but should not be promoted as an independent innovation. Arbitrary-weight explicit uniform rates remain unresolved by this argument.
