# Discrete factor-step audit protocol

Date: 2026-11-21. Primary-source screening preceded this protocol and all new numerical checks. This lane does not claim generic Euler drift, perturbed mirror descent, finite-rate implicit regularization, or projection-based invariant preservation as innovations.

The specific hypothesis is a damping-dependent sign reversal of the leading **limiting endpoint** error of the actual simultaneous balanced two-factor pullback step, with universal quadratic cancellation at `lambda=2s`. Loss weights will have equal trace to avoid a pure loss/learning-rate rescaling comparison.

Fixed model: `Sym(2)`, `X0=I`, orthonormal measurement basis `diag(1,-1)/sqrt(2)` and `[[0,1],[1,0]]/sqrt(2)`, target coordinates `(epsilon,epsilon)`. Compare fixed positive weights `(1,1)` and `(0.5,1.5)`. Step size `eta=0.5`; damping `lambda=0.5,2,4`; amplitudes `epsilon=0.1,0.05,0.025`. No adaptive scan. A degree-four Taylor-jet recurrence, derived from the factor map, independently predicts the small-amplitude coefficients. Ordinary matrix iterations validate the coefficients and convergence. One explicit eight-factor-coordinate Jacobian solve validates the induced product update.

Expected quadratic scalar-normal difference between anisotropic and isotropic weights:

`eta*(s-lambda/2)/(2*(2s+lambda)^2) * sum_i[b_i/(2-eta*m*b_i)-1/(2-eta*m)]`,

where `m=2s/(2s+lambda)`. This expression is positive at `lambda=0.5`, zero at `lambda=2`, and negative at `lambda=4`. At critical damping, spin-factor parity predicts an even scalar response, so the next possible term is fourth order in epsilon. This is a prediction, not yet an established novel result.

Failure criteria: product-step identity fails; converged endpoint differences fail the predicted order/sign; critical quadratic coefficient is nonzero beyond numerical tolerance; or the precise damping-cancellation statement is already a direct prior theorem. Record all outputs regardless of result. This bounded check cannot establish a useful optimizer or a generalization benefit.
