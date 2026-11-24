# Independent certificate source review

Review timestamp: 2026-11-21T10:15:37.132021+00:00. This is a read-only mathematical and implementation review by the transfer investigator; no owner files were modified.

The reviewed rank marginal is the actual global top64-of256 inclusion probability Q(F(S)), averaged over the four possible radial-anchor/action choices. The conditional-uniform score law makes the fitted action family irrelevant to the companion score CDF. Radius-dependent mixtures and the even fallback preserve that statement. The score-CDF brackets use the correct stochastic order: cell maximum locations give a lower CDF, and cell minimum locations give an upper CDF.

The centered antithetic variable is correctly defined. For convex,1/2-Lipschitz g(z)=log(2cosh(z/2)), e=g(h)-g(m), f=g(h_partner)-g(m) satisfy |e|,|f|<=|d|/2 and e+f>=0. At most one is negative. Choosing lower q bounds for nonnegative coefficients and upper q bounds for negative coefficients yields a valid lower variable in[-D/4,D/2]. Subtracting B(r)/2 gives the stated range[-D/4-Bmax/2,D/2]. The rings restriction D=3+C_h and waveform D=1+C_h are correct. Outside the calibration interval, the exactly-even fallback gives b=0 and nonnegative centered gain, so replacing its contribution by zero is conservative.

The empirical-Bernstein constants and unbiased sample variance agree with [Maurer and Pontil, COLT2009, Theorem4](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf), checked directly in the primary paper. Applying the upper-deviation theorem to one minus the affine rescaling gives the lower-mean statement used here. An antithetic pair is one observation; pairs are iid across the independently drawn inputs. The known-law family allocates.05/32 to32 events. The empirical-law family allocates.05/64 to32 DKW and32 conditional empirical-Bernstein events. These are separate95% families, not a joint95% statement across both modes. Independent verification radii must be separate from both fitted policy calibration and the mean-verification inputs.

The local bound also passes review. Splitting all angular sawtooth jumps permits monotone left/right sums for q. The midpoint derivative remainder uses the correct h/8 bound for the covariance of a monotone weight with centered position, and h^2/24 for the second-derivative term. Radial differentiation includes both the network derivative and the score-location derivative. Derivatives of the radial anchor mixture weights are included. The derivative-width interpolation error (upper_derivative-lower_derivative)*bandwidth*t*(1-t) is valid. Structural even bands receive exactly zero moments.

The polynomial waveform radial CDF and first-moment helpers agree with the declared Beta mixture. Known-law radius-cell enclosures are deterministic in real arithmetic. Empirical-law derivative enclosures correctly use a uniform DKW band on the pushforward location CDF before convolution; action fitting does not require an additional pointwise union bound. The positive Poisson-mixture tail accounting for rings is consistent with assigning omitted mass to an endpoint and charging the declared tail.

No mathematical blocker was found. One API omission was reported: adapters initially lacked network_constants, which root_audit/certificate.py requires. The validation owner confirmed adding it before their source freeze. The current review is conditional on independent numerical enclosure checks; padded float64 interval operations are not a directed-rounding or machine-verified proof.

Scope is essential: the certified posterior family is all full-orbit-invariant p(r). It does not include arbitrary half-turn-invariant p(r,phi mod pi). Bounds concern expected logistic gain under the stated input marginal, not every realized batch, classification accuracy, arbitrary covariate shifts, or arbitrary non-invariant labels.

## Reviewed source hashes

- `root_audit/certificate.py`: `03b951115dbf95d997cbea4c75d095dd0380d200100321da53dc0c1d0362860b`
- `root_audit/CERTIFICATE.md`: `9e0e0f3ab9b85b32c37b2fc7e0da804c0cd414b9e030ae12672c4c222a028834`
- `certificate/known_cdf.py`: `9eb325a3f4fe4b2488b0bc19ff708ac8b78595e76f46d01b4431892d45ce5ef7`
- `certificate/local_rank.py`: `8e7b36d7fe1cfa4a5ae8b208e60241a732a57aa7d54a6d2195412d0a6f9161ca`
- `reference/local_jets.py`: `f92586ac2c97b31871c92c0c5e7eea5f6afd6a114c9d125fec6b6d9f16c9e691`
- `validation/adapters.py`: `a614d61c180a7b3adcc8fb6d3cfa0421fca1a337e1989b2e7aed8a780c261097`

## Frozen runner independence check

Checked 2026-11-21T10:22:21.892351+00:00: `certificate/experiment.py` uses verification-only radius seeds1013000+index and1053000+index, separate from calibration, performance tests, root mean-verification seeds1011000/1051000, and independent crosschecks1012000/1052000. It fixes1,048,576 radii and alpha=.05/64 before fresh inputs. Its known-law mass/integrated summary fields in empirical mode are descriptive only; the actual empirical-law guarantee uses the pointwise envelope and another independent mean sample. This separation is correct.

- `certificate/experiment.py`: `e3c8c2d3b345f17cabe8e3274a58354d31432f7ae8527686c0438029c310ca19`
- `certificate/PROTOCOL.md`: `e7bb0a99c52306e59970860088864a587e5de2fe2aa7f85234fb79f46493ada1`
