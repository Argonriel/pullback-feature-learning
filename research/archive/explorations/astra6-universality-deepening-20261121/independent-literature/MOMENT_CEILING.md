# Classical moment-support ceiling and its endpoint consequence

Status: exact proof checked independently. This is a supporting corollary, not a new quadrature theorem or a separate innovation claim.

Let mu be the probability measure du/(2 sqrt(u)) on [0,1]. Its moments are mu_j=1/(2j+1). For any probability measure nu supported on at most m points of [0,1], define b_j=mu_j-int u^j dnu. The corresponding symmetric monotone resolvent mixture is

    m_nu(x,y)=a^(-1) int (1-u v^2)^(-1) dnu(u),
    a=(x+y)/2, v=(x-y)/(x+y).

Thus ell-m_nu=a^(-1) sum_(j>=1) b_j v^(2j) for |v|<1.

## 1. Sharp order ceiling

For every such nu there exists j in {1,...,2m} with b_j != 0. Indeed, let its distinct support points be u_1,...,u_r with r<=m. The monic degree-m polynomial

    p(u)=u^(m-r) product_(i=1)^r (u-u_i)

vanishes on the support, so int p^2 dnu=0, whereas int p^2 dmu>0. Matching all moments through degree 2m would give equality, a contradiction.

If b_0,...,b_(2m-1) all vanish, necessarily r=m. For q of degree <=m-1,

    int p q dmu=int p q dnu=0.

Hence p is the unique monic degree-m orthogonal polynomial for mu. The nodes and their interpolatory weights are uniquely the Gaussian rule. Conversely the Gaussian rule has exactly this moment matching. Positivity and nodes in (0,1) follow from the usual Gaussian quadrature construction.

These are classical Gaussian quadrature facts. Fawzi, Saunderson, and Parrilo's [Appendix B.1, Proposition 5](https://arxiv.org/pdf/1705.00812) explicitly identifies the moment matching with Taylor/Padé matching in their matrix-logarithm approximation setting. The endpoint application still requires the independently proved fixed-weight response theorem.

## 2. Exact first error

Let P_(2m) be the ordinary Legendre polynomial normalized by P_(2m)(1)=1. Its leading coefficient is

    L_(2m)=2^(-2m) binom(4m,2m).

Then p_m(u)=P_(2m)(sqrt(u))/L_(2m) is monic of degree m. Substituting u=r^2 proves its orthogonality for mu. Its squared norm is

    h_m=int p_m(u)^2 dmu(u)
       =1/[(4m+1)L_(2m)^2]
       =2^(4m)/[(4m+1)binom(4m,2m)^2].

For the Gaussian rule, subtract p_m(u)^2 from u^(2m); the difference has degree at most 2m-1 and is integrated exactly. Since the rule annihilates p_m^2,

    b_(2m)=h_m>0.

The constants and orthogonality were independently checked with rational polynomial arithmetic for m=1,2,3,4. These checks verify transcription and the relation with the preserved previous numerical records; the all-m statement follows from the proof. See `moment_constant_check.json` and `verify_moment_constant.py`.

## 3. A nonsharp quantitative certificate

For any at-most-m-atom nu, let eta=max_(1<=j<=2m)|b_j|. Monic orthogonal polynomial minimization gives int p^2 dmu>=h_m. For roots in [0,1], the coefficients of p^2 alternate in sign (zeros are allowed), so their absolute sum except the constant coefficient equals

    product_(i=1)^m(1+u_i)^2-product_(i=1)^m u_i^2 <= 4^m-1,

where zero roots pad the support to m roots. The last expression is increasing in each root. Because b_0=0,

    h_m <= int p^2 d(mu-nu) <= (4^m-1) eta.

Therefore eta>=h_m/(4^m-1). No sharpness is claimed. This simple quantitative moment bound uses only the support restriction and polynomial positivity; it is not a calibrated lower bound on every endpoint measurement, which can also be small because the observation space is nearly neutral.

## 4. Precisely limited optimizer consequence

Assume the independent universality and static-weight theorem, and fix a nonneutral measurement space S. Let k be the first nonzero b_j. There is a fixed SPD quadratic-weight contrast whose first damping response starts at target order 2k+1 if Lie-triple closure fails, or 2k+2 if Lie-triple closure holds but associator-ideal closure fails. Since k<=2m, some such contrast appears by order 4m+1 or 4m+2 respectively; Gaussian mixtures attain those upper order limits. The directions and weights may depend on S and the kernel. This does not imply one universal experiment achieves those orders for all S, a noise-independent measurement guarantee, or an impossibility result for all finite diagnostics.

In particular, knowing a bound on the number of mixture atoms restores a finite order ceiling. The previous absence of a uniform finite Taylor order concerns the unrestricted kernel family; it must not be restated as absence of a finite symbolic algebraic test.
