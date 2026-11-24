# Candidate: global convergence on the classified measurement spaces

## Statement

Let S subset Sym(n) satisfy the two polynomial closure conditions of the previous classification, and let J=Jordan(S,I). Thus N_J=J intersect S-perp is central. Let X0 be any SPD element of J, not necessarily isotropic. Let f(X) be a fixed SPD-weighted quadratic loss on measurements spanning S. Assume its zero set contains an SPD matrix. For every fixed lambda>0, the balanced full-representation damped flow

    Xdot=-K_X(K_X+lambda I)^-1 G(X),  K_X(U)=XU+UX,

exists for all t>=0, stays SPD, and converges to an SPD interpolant. Its endpoint is the unique ambient Bregman projection

    argmin_{X SPD, A X=y} D_(Phi_lambda)(X,X0),
    Phi_lambda(X)=||X||_F^2/2+(lambda/2)tr(X log X-X).

This removes the former small-target or assumed-convergence qualification within this class. It applies to every compatible SPD initialization and every fixed SPD loss weight, without a restricted-isometry assumption. It does not cover arbitrary S, arbitrary X0 outside J, discrete updates, rank-deficient starts, or lambda=0. It supplies no uniform rate over vanishing damping or targets approaching loss of strict feasibility.

## 1. Algebra invariance and conserved central coordinates

The exact product reduction and classification are inherited from the previous archive. For X in J intersect SPD, K_X and its resolvents preserve J. Thus X(t) remains in J. For every N in N_J,

    F_N(X)=<N,X+(lambda/2)log X>

is conserved as long as the trajectory is SPD. This uses centrality, not a full ambient Hessian representation.

Strict feasibility can be witnessed within J. The Frobenius projection P_J is positive and unital: if A is PSD and P_J A had a negative spectral subspace with projector E, then E belongs to J by spectral calculus and <E,P_J A>=<E,A> would be nonnegative, a contradiction. Applying the same argument to A-alpha I proves P_J A>=alpha I whenever A>=alpha I. Therefore projecting any ambient SPD feasible matrix yields an SPD feasible Y in J with the same measurements.

## 2. Bounded loss and central invariants force bounded products

Because the quadratic weight is SPD on all measured directions, f decreasing implies P_S X(t) is bounded. Suppose a sequence X_k on the trajectory has norm tending to infinity. Pass to a subsequence with

    X_k/||X_k||_F -> V,  ||V||_F=1.

Then V belongs to N_J, is PSD, and is central. Decompose J into its simple factors and scalar central factors. On every factor where V has coefficient v_i>0, the central part of X_k divided by ||X_k|| tends to v_i times the identity, while the measured traceless part remains bounded. All eigenvalues on that factor therefore tend to infinity. Factors where V has zero coefficient contribute nothing to F_V.

It follows that <V,X_k> tends to infinity and <V,log X_k> tends to positive infinity or is eventually nonnegative. This contradicts the fixed value of F_V. Thus X is bounded on its entire interval of existence. This classification-based coercivity step is the substantive addition: the quadratic measurement loss itself need not be coercive in the ambient matrix space.

## 3. Balanced factors exist globally and converge

Write X=F F^T with a square invertible F(0), and define

    H=(K_X+lambda I)^-1 G(X),  Fdot=-H F.

This reproduces Xdot=-(HX+XH) and the balanced two-factor dynamics. Since X is bounded, F is bounded; lambda>0 keeps H smooth and bounded even at PSD boundary points. The factor ODE therefore exists globally. Its fundamental matrix remains invertible for every finite time, so X(t) stays SPD.

For completeness, convergence does not require treating the inverse product metric as smooth at the PSD boundary. The factor objective f(F F^T) is analytic. If ||X||op<=R, direct spectral estimates for K_X give

    -d f/dt >= ||grad_F f||_F^2 / [2(2R+lambda)],
    ||Fdot||_F <= ||grad_F f||_F / (2lambda).

Indeed ||G F||_F^2=<G,K_X G>/2 and

    -d f/dt=<G,K_X(K_X+lambda I)^-1 G>.

These are the usual uniform angle and descent estimates used with the analytic Lojasiewicz inequality. The standard finite-length argument on a bounded analytic descent trajectory gives F(t)->F_infinity. This is established analytic convergence machinery, not an originality claim. Consequently X and H also have limits, and H_infinity F_infinity=0.

## 4. An invertible fundamental solution cannot converge to a negative limiting generator

We use an elementary linear-ODE fact. Suppose H(t) is symmetric, H(t)->H_infinity, and a fundamental matrix F(t) solving Fdot=-H(t)F is bounded with invertible F(T) at every finite T. Then H_infinity has no negative eigenvalue.

Proof: if A_infinity=-H_infinity has a positive eigenspace E_u, let a>0 be its smallest positive eigenvalue and let E_s be the orthogonal complement, where A_infinity has nonpositive spectrum. For sufficiently large T, the perturbation of A_infinity has norm epsilon<a/4. The cone ||y_s||<=||y_u|| is forward invariant for ydot=-H(t)y: on its boundary the derivative of ||y_s||^2-||y_u||^2 is strictly negative. Inside the cone, ||y_u|| grows at least as exp((a-2epsilon)(t-T)). Choose nonzero y(T) in E_u. Since F(T) is invertible, y(t)=F(t)F(T)^-1 y(T), contradicting boundedness of F. This proves the assertion without a probabilistic saddle-avoidance argument.

Apply this to obtain H_infinity PSD. Stationarity gives H_infinity X_infinity=X_infinity H_infinity=0, hence

    G_infinity=(K_(X_infinity)+lambda I)H_infinity
              =lambda H_infinity PSD,
    <G_infinity,X_infinity>=0.

These are the PSD convex optimality conditions. Pairing the convex first-order inequality with a strictly feasible Y yields

    0=f(Y)>=f(X_infinity)+<G_infinity,Y-X_infinity>
                     >=f(X_infinity)>=0.

Therefore the limit interpolates. This argument holds for every invertible initial factor in the compatible class; it does not discard a measure-zero exceptional set of balanced starts.

## 5. Central logarithmic conservation excludes a singular interpolant

Suppose the interpolating limit X_infinity were singular. Choose the SPD feasible Y in J from step 1 and set N=Y-X_infinity. Then N belongs to N_J and is central. On every simple/central factor where X_infinity has a zero eigenvalue, the scalar coefficient of N is strictly positive, because Y=X_infinity+N is positive definite. Factors where N has a negative coefficient must have positive-definite limits.

Consequently <N,log X(t)> tends to negative infinity: all singular-limit contributions have positive coefficients, and negative-coefficient contributions stay finite. The term <N,X(t)> remains bounded. This contradicts conservation of F_N, since lambda>0. Thus X_infinity is SPD.

This uses divergence of the derivative of entropy, not divergence of entropy's value at the boundary. The entropy potential itself has a finite continuous extension to PSD matrices.

## 6. Exact endpoint and eventual rate

Now logarithmic functional calculus and the conserved coordinates imply

    grad Phi_lambda(X_infinity)-grad Phi_lambda(X0) in S.

Together with interpolation, this is the ambient affine-constraint optimality condition. Strict convexity gives the unique Bregman minimizer stated above. Existence of an interior minimizer has been proved by the flow argument rather than assumed to prove the flow converges.

Once X is close to its SPD limit, choose a>0 with X(t)>=aI. If beta is the smallest eigenvalue of the induced positive loss Hessian on S in orthonormal Frobenius coordinates, the loss has the eventual bound

    f(X(t)) <= f(X(T)) exp[-2 beta (2a/(2a+lambda))(t-T)].

The positive constant a is trajectory/target dependent. This is not an explicit uniform complexity guarantee. Small damping can drive the selected endpoint extremely close to the boundary.

## Minimal verification outside the former local domain

One new coupled-center six-dimensional matrix case uses target displacement norm 3.29132. A known SPD feasible target has minimum eigenvalue .3, while the ordinary Euclidean affine interpolant has minimum eigenvalue -.508738. Thus the old small-neighborhood continuation argument cannot certify this example; the unconstrained zero-damping endpoint is outside the SPD domain.

With lambda=4, two fixed SPD weights, and two integration steps, all four runs converged numerically to the predicted interior solution, whose minimum eigenvalue is .111715. The fine-step endpoint difference between weights is 1.41e-11; the largest fine-step prediction error is 1.92e-11. The largest measurement residual over all runs is 1.70e-12, and the largest step-refinement change is 2.87e-10. Records retain all matrices, settings, source/dependency hashes, and sampled invariant errors in `validation/`.

This is consistency evidence, not a proof of the global statement. In particular a finite numerical horizon cannot validate a theorem for every strictly feasible target. The proof above supplies that extension.

## Innovation boundary

The Jordan class, central conservation law, analytic finite-length machinery, convex PSD optimality, and Bregman uniqueness are established ingredients. The candidate contribution is their strengthened consequence with a complete boundedness and boundary-exclusion proof: globally convergent, exact endpoint selection for this underdetermined noncommuting class, all compatible full-rank starts, all SPD weights, and every positive damping.

The primary-source audit in the companion record compares full-input deep-linear results, preconditioned low-rank sensing guarantees, Hessian barrier flows, and signed-covariance results. No generic global convergence principle or new Jordan structure is claimed. The result is a scoped candidate with potentially modest novelty; its priority remains unverified.
