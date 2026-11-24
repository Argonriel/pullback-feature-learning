# Local deterministic conditional-moment bounds

The deployed field and allocation rule are unchanged from the prior study. Write d(r,phi)=[h(r,phi)-h(r,phi+pi)]/2 and u(r,phi)=c(r) dot psi(phi), where psi contains the sine and cosine harmonics 1 through 8. The coefficients are linearly interpolated between the original 129 radial knots, vanish at both endpoints, and vanish outside radius 3. The marginal inclusion probability is qbar=1/4+u/8. Its conditional imbalance is

    b(r) = E_phi[u(r,phi) d(r,phi)] / 8.

Only odd harmonics of u contribute, because d changes sign after a half turn. Let u_o be this odd part and F=u_o d. Then F is pi-periodic and b(r)=(1/(8pi)) integral_0^pi F(r,phi) dphi. This reduction is exact, not a numerical small-value threshold.

## Local interval differentiation

Subdivide each original radial interval into four equal bands. Subdivide the folded angular domain [0,pi] into 256 equal cells. For each resulting polar rectangle, propagate intervals for five quantities through the network: h, h_r, h_rr, h_phi, and h_phiphi. Initial intervals enclose the trigonometric extrema on the entire cell, not just corners. Radial and angular input derivatives are enclosed directly.

For a residual block h_new=h+B tanh(Ah+b)/2, let z=Ah+b and t=tanh(z). The componentwise derivative formulas for either coordinate v=r or phi are

    (t)_v  = (1-t^2) z_v,
    (t)_vv = -2t(1-t^2) z_v^2 + (1-t^2) z_vv.

Signed interval matrix products, interval multiplication, the exact squared-interval range, monotonicity of tanh, and the extrema of -2t(1-t^2) at t=plus/minus 1/sqrt(3) enclose every term. Residual addition preserves the enclosure. Propagate the antipodal input and its corresponding derivatives separately; subtract the resulting intervals and divide by two to enclose d and its derivatives. Dependencies may widen these intervals but cannot invalidate them in real arithmetic.

Inside each original radial interval, c is linear, so u_rr=0. Enclose the trigonometric field, its angular derivatives, and its radial derivative on each cell. The required product derivatives are

    F_phiphi = u_phiphi d + 2 u_phi d_phi + u d_phiphi,
    F_rr     = 2 u_r d_r + u d_rr.

Both formulas use the odd field. Signed lower and upper F_rr intervals are averaged across angular cells, retaining cancellation in the derivative bound instead of applying a global spectral norm.

## Angular quadrature and continuous-radius interpolation

Let angular width be h_phi=pi/M, M=256. Evaluate the signed midpoint quadrature Q_j for b at every fine radial node. These are exact network point evaluations in the real-arithmetic argument. If K_ik bounds |F_phiphi| on radial band i and angular cell k, the same angular error applies at both ends of that band:

    E_i = h_phi^2 / (24*8) * mean_k K_ik.

This follows by integrating the second-order midpoint remainder separately over every angular cell. It does not interpret angular points as independent random samples.

If the cell enclosures of F_rr are [l_ik,v_ik], then throughout radial band i,

    L_i = mean_k l_ik / 8 <= b''(r) <= mean_k v_ik / 8 = U_i.

Let the band have width w, r=r_i+t*w, and Qlin=(1-t)Q_i+t Q_(i+1). The standard linear-interpolation remainder, including its sign, gives the pointwise bounds

    Qlin - E_i - U_i*w^2*t*(1-t)/2 <= b(r),
    b(r) <= Qlin + E_i - L_i*w^2*t*(1-t)/2.

No interval crosses an original coefficient knot, so the required one-sided endpoint derivatives and interior second derivative exist. A simpler bound valid throughout the whole band is

    B_i = max(|Q_i|,|Q_(i+1)|) + E_i
          + max(|L_i|,|U_i|)*w^2/8.

Thus |b(r)|<=B_i for every radius in that band. Outside radius 3, b=0 exactly. If the odd coefficients are identically zero at both endpoints of a band, linearity proves b=0 throughout it; the implementation records this structural case and sets the bound to zero. No numerical tolerance is used to declare balance.

These bounds are pointwise in radius. They are not broad-bin cancellation statements and do not require the radial input density to be known. They do require uniform conditional orientation, as in the original problem.

## Integrated diagnostics and unknown-posterior gain

For the known noisy-ring input law, an integrated bound is sum_i P(R in band i) B_i. The Rice radial CDF is evaluated through the noncentral chi-square Poisson-Gamma mixture. For class radius a and sigma=0.15,

    P(R<=r) = sum_(k>=0) Pois(k;lambda)
                    GammaCDF(k+1, r^2/(2sigma^2)),
    lambda = a^2/(2sigma^2).

All terms are nonnegative. Truncating after k=160 leaves at most the Poisson tail, bounded by exp[-lambda+161*(1+log(lambda/161))]. The implementation adds the maximum mixture-tail bound times max_i B_i to the integrated upper bound. Numerical CDF cancellation is not claimed to be formally directed-rounded. Angular, radial, and signed-node-envelope contributions are reported separately. The numerical integrated actual |b| is a diagnostic, not the proof.

For any hidden measurable radial posterior p(r), Delta(p)=C-E[p(R)b(R)]. Centering at one-half gives

    Delta(p) >= E[W0 - B(R)/2],
    E W0 = C - E b(R)/2.

The root study independently evaluates this lower bound with independent unlabeled inputs and a finite-sample confidence correction. That sample is not used to fit the field, select local resolution, or shrink B. A positive finite-sample lower bound is therefore distinct from a positive numerical estimate of C. The local numerical construction does not itself assert that the finite-sample target will succeed on fresh models.

## Numerical scope and cost

The theorem above is a real-arithmetic statement. The implementation uses float64 arithmetic with small interval padding, ordinary library trigonometric functions, and explicit numerical spot checks. It is not a formally verified directed-rounding certificate.

There are 512 radial bands and 256 folded angular cells: 131,072 polar cells. Two antipodal five-component interval jets per cell cost 262,144 interval-jet propagations. The 513 radial nodes each require 512 full-circle point evaluations, totaling 262,656 ordinary example forwards. Both costs are reported; an interval jet is not called a single cheap forward pass. Numerical validation and all development alternatives are counted separately. The deployed predictor and exact 64/256 extra-call quota are unchanged by this offline analysis.
