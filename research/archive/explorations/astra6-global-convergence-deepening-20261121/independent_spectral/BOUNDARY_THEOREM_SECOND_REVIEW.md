# Second independent review of the boundary theorem

Date: 2026-11-21 (America/Los_Angeles). Reviewed `../independent_boundary/THEOREM_BOUNDARY.md`, including its arbitrary-data quantifiers, common-face conclusion, and ambient closed-PSD Bregman characterization. This is a mathematical review, not an independent priority certification and not a rerun of the boundary experiment.

## Verdict

No fatal gap was found. Under the stated central-normal premise, fixed positive damping, and a compatible strictly positive definite initial matrix, the argument extends convergence and endpoint selection to **every** observation vector. Strict feasibility is needed exactly for a positive definite endpoint; it is not needed for convergence, PSD optimality, or the closed-cone Bregman minimization statement. This is a substantive extension of the prior statement, while its mechanism remains a synthesis of established tools.

## Checks with particular attention to the new claims

1. **Boundedness before feasibility.** The new proof is shorter and valid. A normalized escape limit `V` is a fixed PSD element of `N_J`, and therefore commutes with every `X_k`. Uniform convergence `X_k/r_k -> V` implies `X_k >= (r_k nu/2)I` on `ran(V)`, where `nu>0` is the smallest positive eigenvalue of `V`. The zero eigenspace of `V` has exactly zero pairing weight. Consequently neither small eigenvalues elsewhere nor a negative logarithmic contribution can cancel the divergent conserved quantity. No feasible point or minimizer is assumed.

2. **Existence of a cone minimizer is a conclusion.** Bounded factors, the analytic descent argument, and the invertible-fundamental-matrix cone argument give a limiting pair with `G*>=0` and `<G*,X*>=0`. The displayed convex first-order inequality then proves optimality over the entire PSD cone. This remains valid when the desired measurements are unattainable, and it avoids assuming that a linear image of an arbitrary PSD cone is closed. The classification-based boundedness is the property that enables attainment here.

3. **All minimizers have identical measurements.** Strict convexity of the quadratic in measurement coordinates supplies this, even if `A` has a kernel or redundant rows. If two minimizers have distinct measured vectors, their PSD midpoint has strictly smaller loss because `W>0`. No claim that the matrix objective is strictly convex is needed.

4. **The central boundary lemma has the correct sign logic.** For any central `N` with `X*+N>=0`, its negative eigenspaces are bounded away from the PSD boundary at the limit. On a positive eigenspace, any zero limiting eigenvalue makes the logarithmic pairing tend to negative infinity. Zero-coefficient eigenspaces cannot cancel it. Conservation therefore forces `N` to vanish on the kernel, and all its nonzero eigenspaces have regular logarithmic limits. This establishes the finite restricted-logarithm pairing without differentiating entropy at a singular matrix.

5. **The common support statement is ambient.** For ambient `Z>=0`, positivity of `P_J` and equality of measured values allow the central lemma to be applied to `P_J Z-X*`. The kernel projector `E` belongs to `J`, so `<E,Z>=<E,P_J Z>=0`. PSD positivity then implies `ZE=0`; for example, `tr(EZ)=||Z^(1/2)E||_F^2`. Thus ambient off-algebra components cannot restore a missing support direction. This is the step needed to upgrade a within-`J` face statement to the advertised ambient statement.

6. **The face gradient is legitimate.** On the common face supported by `P`, `X*` is strictly positive definite relative to `ran(P)`, so the entropy restriction is differentiable there. The stated gradient `B=X*+(lambda/2)log_P X*-P A0 P` lies in `J`: both the spectral functions and `P A0 P` lie in `J`, the latter by the quadratic representation. Pairing an arbitrary ambient minimizer with `B` therefore reduces exactly to its `P_J` projection. Since the resulting central difference is supported on `P`, the initial anchor pairing reduces to the conserved quantity. The first-order convex inequality on the face proves ambient optimality, and the quadratic term gives uniqueness. No unjustified ambient logarithm at a singular matrix or conditional-expectation Jensen property is used.

7. **Exact boundary criterion.** The common-kernel property gives an SPD endpoint whenever one cone minimizer is SPD. Conversely an SPD endpoint is itself such a minimizer. Thus the claimed equivalence has both directions. For consistent data this is exactly the strict-feasibility boundary; for inconsistent data it is a condition on the optimal set rather than on the infeasible interpolation equations.

## Minor formalization point

The compact analytic descent paragraph is standard and correct, but a publication proof should explicitly note that the omega limit set consists of critical points. One route is: the descent inequality makes `||grad L||^2` integrable; boundedness of `F`, smoothness of the vector field, and compactness make `grad L(F(t))` uniformly continuous; hence it tends to zero. A finite covering of the compact omega limit set then provides one uniform Lojasiewicz exponent and constant. This fills a routine justification rather than changing the theorem.

## Quantitative boundary

The proof gives no uniform spectral lower bound if the endpoint is singular, and does not imply global exponential convergence in that case. The separate exact scalar boundary controls are appropriate discriminators. For strictly feasible targets and geometry-adapted observation weights, `QUANTITATIVE_REVIEW.md` supplies a different reverse-Bregman proof with an explicit positive lower bound and exponential rate. Its arbitrary-weight shortcut is disproved by the preserved 2-by-2 example, so it must not be used to strengthen the general-data theorem beyond its stated rate-free conclusion.
