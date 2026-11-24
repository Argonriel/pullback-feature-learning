# Continuous conditional balance with an exact quota

The input law is rotationally invariant on the continuous plane. Conditional on radius R=r, Phi is uniform. The unknown posterior p(r) is otherwise unrestricted. Write a=h(x), m=(h(x)+h(-x))/2, d=a-m, and A=softplus(a)-softplus(m). A selected replacement has risk gain A-p(r)d.

## Exact quota without repair

Randomly partition each iid batch of256 into64 groups of4, independently of inputs. For a fixed measurable field u(x) in[-1,1], choose exactly one member of a group with probabilities

    q_i = 1/4 + (u_i - group_mean(u))/6.

These sum to1. Since u_i-group_mean(u) lies in[-3/2,3/2], every q_i lies in[0,1/2]. Choices across groups give exactly64 extras, without probability clipping, global quota repair or extra partner queries. For a given point, independence of its three group companions gives marginal inclusion probability

    qbar(x) = 1/4 + (u(x)-E u)/8.

Our field has zero angular mean at EVERY radius, hence E u=0 for any radial input law and qbar=1/4+u/8. With C=E[qbar A] and b(r)=E[qbar d|r]=E[u d|r]/8, exact gain is

    Delta(p)=C-E[p(R)b(R)].

There is no quota-repair error term. Dependence among the256 selections does not change this population expectation. Conditional balance is exact only if b(r)=0 almost everywhere, not if it averages to zero in broad radius bins. Global confidence/radius/folded top64 allocators are stronger allocation classes than the grouped rule; comparisons distinguish the grouping cost from the balance constraint.

## Continuous fitted field

At129 deterministic radii r_j from0 to3 and512 equally spaced angular nodes, evaluate the frozen network once per point. Half-turn partners are cached shifts of those same offline values, so calibration costs66048 example forwards, not twice that. Let psi contain cos(k phi),sin(k phi), k=1,...,8. Fit coefficients c_j to maximize c_j dot E_grid[psi A], subject to c_j dot E_grid[psi d]=0 and ||c_j||_1<=1. A linear program over signed basis atoms has an optimum using at most two atoms: enumerate zero-balance atoms and all opposite-signed-balance pairs. No held-out outcome chooses coefficients. A zero field is feasible.

Interpolate coefficient vectors linearly between radii. Set both endpoint coefficients to zero and set u=0 outside[0,3]. The resulting u(r,phi)=c(r) dot psi(phi) is a continuous function of x, has zero conditional angular mean exactly, and is bounded by1 because the l1 ball is convex. It can be orientation-sensitive; that is checked rather than assumed. An even-harmonic-only field is an exact antipodally invariant control. An unconstrained field uses the same l1 class and objective without the conditional-moment constraint.

## Explicit deterministic approximation bound

For the two residual blocks, define

    L = product_l (1 + 0.5 ||B_l||_2 ||A_l||_2),
    C_h = 0.5 sum_l ||first_row(B_l)||_1.

The first logit is L-Lipschitz in input and |h(x)|<=||x||+C_h. For d=(h(x)-h(-x))/2, |d(r,phi)|<=D(r)=r+C_h, |partial_r d|<=L, and |partial_phi d|<=L r.

At anchor j let gamma_j=||c_j||_1<=1, kappa_j=sum_k k(|c_cos,k|+|c_sin,k|)<=8, and e_j=|mean_grid u_j d_j|. Equally spaced angular quadrature gives

    |E_phi[u_j d_j]| <= E_j
      := e_j + pi/(2M) [kappa_j D(r_j) + gamma_j L r_j], M=512.

Proof: u_j d_j is angular-Lipschitz with the bracketed constant, and the mean distance to a cell's central quadrature node is pi/(2M). The bound applies to any phase of an equally spaced circle.

For r=(1-t)r_j+t r_(j+1), linearity of u and the radial Lipschitz bound imply

    |b(r)| <= 1/8 [(1-t)E_j+t E_(j+1)
                 + L t(1-t)(r_(j+1)-r_j)(gamma_j+gamma_(j+1))].

Outside[0,3], b=0 exactly because u=0. This is a pointwise-in-radius bound, not bin-mean cancellation. It controls every measurable hidden radial posterior, with |Delta(p)-C|<=E|b| bounded by its radial integral. Separate angular quadrature, radial interpolation and observed numerical residual terms are retained. The theorem is in exact arithmetic; floating-point residuals and refinement discrepancies are reported separately, without claiming interval-certified library arithmetic.

There is no random calibration-statistical error term for this deterministic anchor method. Its costs/errors are numerical quadrature and interpolation, not thousands of allegedly independent angular samples. Randomness in trained models and test inputs is separately evaluated. A global spectral-norm bound can be very loose; a positive numerical gain is not a rigorous non-harm certificate if the bound is uninformative.

## Hidden posteriors and exact continuous existence

The sharp unrestricted range is[C-E b_+, C+E b_-]. Known prevalence1/2 restricts p to put half the radial mass on the largest or smallest b values, respectively. We report all these extremes for the balanced, unconstrained and even fields, using independent refined continuous-radius integration and retaining numerical approximation error. Held-out continuous iid points replay every world for every deployed policy. No tiny aggregate bin residual is promoted to conditional identification.

Root's independently checked continuous example h(r,phi)=1+2r cos(phi), r uniform[.5,1.5], u=-cos(3phi), has E[u d|r]=0 for every radius by harmonic orthogonality. It is orientation-sensitive, uses the same exact64 grouping rule, and gives the same population gain under all radial posteriors. The existence proof and numerical example are separate from approximate balance for trained neural networks; the latter does not inherit exactness.

## Additional valid second-derivative quadrature bound

The implementation also computes a global input Hessian bound. For each residual block, ell=1+||B||2||A||2/2 and h2=||B||2||A||2^2*(4/(3sqrt(3)))/2, because max|tanh second derivative|=4/(3sqrt(3)). Composition updates H_new=h2*L_old^2+ell*H_old and L_new=ell*L_old.

At anchor j let kappa2_j=sum k^2(|c_cos,k|+|c_sin,k|). Then the angular second derivative of u_j*d_j is bounded by

    kappa2_j D(r_j) + (2kappa_j+gamma_j) r_j L + gamma_j r_j^2 H.

The centered-cell quadrature error is at most this expression times(2pi/M)^2/24. The stored angular error uses the minimum of this valid bound and the first-derivative bound. Both components remain reported. The radial interpolation bound remains the previously stated first-derivative bound. A smaller deterministic error bound is not a claim that it is numerically informative.

For a fixed fitted balanced field, an independent iid sample also supplies a finite-sample lower bound for C. Let W=qbar*A*1_{R<=3}. Its range is contained in[-3(3+C_h)/8,3(3+C_h)/8]. The one-sided Hoeffding deduction is C>=mean(W)-3(3+C_h)/4*sqrt(log(20)/(2N)) with probability at least.95. The omitted tail has u=0 and nonnegative rotational Jensen gain. Subtracting the analytic uniform bound on |b| gives a valid, possibly very loose, simultaneous-in-p gain lower bound. This sample is evaluation-only and does not fit the gate.
