# Exceptional initializations: exact statements and limits

Status: complete independent subtask. Date: November 21, 2026 (America/Los_Angeles).
All historical materials were read-only. This note deepens the initialization qualification; it does not claim a complete all-dimensional orbit classification.

## Setup and bracket convention

For real symmetric positive definite X, fix a finite lambda > 0. Let K_X(U)=XU+UX, M_X=K_X(K_X+lambda I)^(-1), F_i=M_X E_ii, A_ji=(F_i)_jj, and H_i=sum_k F_k (A^(-1))_ki. The matrix A is positive definite, since v^T A v=<diag(v), M_X diag(v)> > 0 for v != 0. Thus H_i is analytic and diag(H_i)=e_i. We use [U,V]=D V[U]-D U[V]. Multiplying the frame by an invertible state-dependent matrix changes neither its Lie orbits nor the rank of the span of fields and first brackets modulo the original distribution.

The conclusions below concern arbitrary controls in these fields. They do not assert controllability by a single fixed positive quadratic loss. Where an autonomous-loss endpoint consequence is used, its convergence and finite-action hypothesis is stated explicitly.

## Proposition 1: complete two-dimensional Lie-rank classification

For every finite lambda > 0 and every X=[[a,c],[c,b]] in SPD(2):

- If c=0, the Lie rank is exactly 2, at every bracket length.
- If c != 0 and a != b, the fields and brackets of length at most 2 have rank 3.
- If c != 0 and a=b, brackets of length at most 2 have rank 2, and brackets of length at most 3 have rank 3.

Proof. Put d=ab-c^2, Q=(a+b)d+lambda ab. The normalized frame is

    H_a = partial_a + (lambda c b/(2Q)) partial_c,
    H_b = partial_b + (lambda c a/(2Q)) partial_c.

Its first bracket is C=[H_a,H_b]=F partial_c with

    F = lambda c(b-a)d/(2Q^2).

At a=b=s and c != 0, F=0 and partial_c F=0. Therefore

    [H_a,C] = -lambda c(s^2-c^2)/(2Q^2) partial_c,
    [H_b,C] = +lambda c(s^2-c^2)/(2Q^2) partial_c.

These coefficients are nonzero. When c=0, both original and normalized fields are tangent to c=0. Brackets of tangent fields remain tangent, and H_a,H_b already span that two-dimensional tangent space. This proves every rank assertion.

In particular, equality of the two observed diagonal entries is a first-bracket degeneracy, not a conservation leaf. This distinction is exact, rather than inferred from a numerical rank tolerance.

Corollary (control orbits). The three connected control orbits in SPD(2) are c>0, c<0, and c=0. The first two sets are convex and carry full Lie rank. Their orbits are therefore open, and connectedness forces each to be one orbit. The diagonal set is connected and its two normalized fields span its tangent bundle. The sign cannot change at finite time because c=0 is invariant and local uniqueness applies backwards as well as forwards. This is a controlled-orbit statement, not a fixed-loss endpoint range statement.

## Proposition 2: the invariant object is the component partition, not every zero edge

For any n and any partition P={I_1,...,I_k} of coordinate indices, let M_P be the SPD matrices whose entries between different partition blocks vanish. Then:

1. M_P is invariant under all the F_i and H_i, and under any locally integrable time-dependent control for which a solution exists in SPD.
2. At finite times the connected-component partition of the graph with edges X_ij != 0 is exactly preserved.
3. If X(t) stays in a compact subset of SPD and its controls have finite L1 norm on [0,infinity), the same component partition is preserved by any limiting matrix X_infinity.

Proof of 1. If X is block diagonal, K_X preserves each block-diagonal symmetric subspace, and so does its shifted inverse. E_ii lies in that subspace, hence F_i does too. For i in I_r, the field is supported entirely on that block. The observed matrix A is block diagonal under the same partition, so normalization preserves the property.

Proof of 2. Forward uniqueness prevents two initial components from merging. If the graph split at a finite time t, the newly block-diagonal state would stay in that block manifold under the time-reversed control. Backward uniqueness would force the earlier state to have the same split, a contradiction. The result allows individual edges to appear or disappear while connectivity is unchanged.

Proof of 3. Fix any prospective new cut. Write Y for the cross-cut block and X_bd for X with that block set to zero. The principal diagonal blocks of an SPD matrix are SPD, so X_bd remains in a compact SPD set whenever X does. Smoothness and block invariance give

    ||P_cross F_i(X)|| <= L ||Y||

on the relevant compact set, with one finite L. For X_dot=sum_i u_i F_i, the norm of Y therefore obeys

    ||Y(t)|| >= ||Y(0)|| exp(-L integral_0^t sum_i |u_i(s)| ds).

A nonzero cut cannot vanish at an infinite-time limit if that integral is finite. There are only finitely many cuts. A local small-target positive quadratic flow has exponentially decaying residual and hence finite action, so it satisfies this condition.

Exact counterexample to edgewise sparsity. At lambda=1 and

    X = [[2, 1/4, 0], [1/4, 3, 1/5], [0, 1/5, 4]],

all leading principal minors are positive (strict diagonal dominance also certifies SPD), but

    (F_2(X))_13 = -125695/418393579 != 0.

Thus the absent 1--3 edge is immediately created by an allowed field. The initial graph is connected. Calling each zero coherence an invariant would be false.

## Proposition 3: exact product structure and its remaining gap

On M_P the Lie algebra is the direct product of the Lie algebras for the individual blocks. Consequently its rank at X is the sum of those block ranks and is bounded above by sum_r |I_r|(|I_r|+1)/2.

The direct-product assertion follows because a field indexed in one block depends only on that block; cross-block brackets vanish. For every fixed partition, the upper rank bound is attained on an open dense full-measure subset of M_P by applying the archived analytic determinant argument separately to each block. This is a relative measure statement in the block manifold.

If every connected block has size at most 2, Proposition 1 gives an exact all-state answer: the rank is n plus the number of two-element connected blocks. Equal observed diagonals inside a coherent pair require length 3, while unequal ones require only length 2. The connected orbits in that stratum additionally fix the sign of each pair coherence.

For connected blocks of size >=3, this note does not prove that connectedness is sufficient for full Lie rank at every state. The general theorem remains generic. No failed search or finite set of checks is promoted to an all-state theorem.

## A connected three-dimensional counterexample to universal two-step rank

Take lambda=3 and X with all diagonal entries 2 and every off-diagonal entry 1/3. This is SPD, with eigenvalues 5/3, 5/3, 8/3. In vertical coordinate order (12,13,23), the first-bracket columns are

    (2565/3795148) * [[0, 1, 1], [1, 0, -1], [-1, -1, 0]].

The vertical matrix has rank 2: all its column sums vanish. Together with H_1,H_2,H_3, this gives rank 5, despite complete nonzero coherence support.

The vertical entries of [H_1,[H_1,H_2]] are

    (-8760247605/3600787085476,
     -123114855285/482505469453784,
      187248807675/482505469453784).

Their sum is -146205/63568729, which is nonzero. This adds the missing vertical direction, giving rank 6 at length 3. Thus complete coherence does not remove first-bracket exceptions, and a first-bracket exception does not imply an invariant leaf. These are exact rational calculations with saved source, not floating-rank heuristics.

## Validation and reproducibility

`validation_v1/results.json` records six deliberately selected cases, evaluated with exact Fraction arithmetic. The implementation solves K_X+lambda I directly in a symmetric coordinate basis, rather than using the earlier spectral implementation. It differentiates the inverse using

    D M[V]U = lambda R K_V R U,
    D^2 M[V,W]U = -lambda R(K_W R K_V + K_V R K_W)R U,
    R=(K_X+lambda I)^(-1),

and differentiates A^(-1) to obtain normalized brackets. It saves exact matrices, ranks, the state, parameter, runtime versions, and source SHA-256. Source: `experiments/astra6_blended_geometry_deepening_20261121/exceptional/exact_rank_checks.py`.

The separate `derivative_validation_v1/results.json` checks the nested-bracket derivative by centered finite differences of first brackets and of H_1 at the symmetric 3x3 state. At steps 1/100, 1/200, 1/400, the maximum entry errors are 9.0467170e-8, 2.2616349e-8, 5.6540596e-9. Their factor-four decrease supports the independently implemented second-derivative formula. This check records both its own and its dependency's source hashes. Source: `exceptional/finite_difference_check.py` under the same new experiment directory.

The n=2 third-bracket formula is proved for all positive damping and all coherent equal-diagonal states. Its one rational instance equals -315/123008 exactly. The n=3 calculation is a counterexample and a diagnostic example, not an all-state classification inferred from samples. No trajectories were trained, no reserved samples were used, and no dependencies were installed.

## Unsuccessful attempts and novelty boundary

- The unqualified claim 'connected or complete support gives two-step rank everywhere' is false by the equicorrelated 3x3 counterexample above.
- The claim 'every initially zero off-diagonal entry stays zero' is false by the connected path example above.
- A proof of full Lie rank at every connected state in arbitrary dimension was not obtained. Invariant block algebras alone do not exclude additional nonlinear exceptional orbits, so that argument is insufficient.
- Initial environment discovery through the `python` Windows application alias failed because its executable could not be accessed. The explicit bundled Python runtime was then used. A read-only probe for SymPy returned `ModuleNotFoundError`; all completed calculations use standard-library Fraction plus the already available NumPy. Both failures are recorded in `FAILED_ATTEMPTS.md`.

The Lie-orbit theorem, smooth tangent-field facts, backward uniqueness, and block preservation are established tools. The useful scoped additions are the exact two-dimensional stratification, the explicit higher-bracket repair at coherent symmetric states, and precise counterexamples delimiting possible all-state upgrades. No standalone broad originality claim is made for these short propositions. They strengthen the parent candidate's hypothesis accounting and protect its endpoint theorem from an unsupported replacement of 'generic' by 'every coherent state'.
