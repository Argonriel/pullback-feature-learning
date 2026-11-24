# Mathematical specification

This note derives the update implemented in this project. The object of study is a Euclidean metric on the final representations of a fixed training sample. The derivations are local statements about a finite network; they do not assume an infinite-width limit or establish a generalization guarantee. Related methods and primary sources are discussed in [the literature notes](literature.md).

## Network and objective

For an input $x\in\mathbb R^2$, define

$$
h_0=x,\qquad h_{\ell+1}=h_\ell+\frac12 B_\ell\tanh(A_\ell h_\ell+b_\ell),\qquad \ell=0,1.
$$

Each block contains $A_\ell\in\mathbb R^{8\times2}$, $B_\ell\in\mathbb R^{2\times8}$, and $b_\ell\in\mathbb R^8$, so there are $2(16+16+8)=80$ trainable parameters. There is no trainable readout. Let $z_i=h_2(x_i)$ and stack coordinates in sample-major order:

$$
Z=R_X(\theta)=(z_{1,1},z_{1,2},\ldots,z_{n,1},z_{n,2})^\top\in\mathbb R^{2n}.
$$

The representation Jacobian is $J=D_\theta R_X(\theta)\in\mathbb R^{2n\times80}$. For $n=256$, it has shape $512\times80$.

With labels $y_i\in\{-1,1\}$ and fixed $w=(1,0)^\top$, the objective is

$$
L(\theta)=\Phi(R_X(\theta)),\qquad
\Phi(Z)=\frac1n\sum_{i=1}^n\log(1+\exp(-y_iw^\top z_i)).
$$

In code, evaluate this expression with a stable softplus implementation. If $g=\nabla_Z\Phi(Z)$, then

$$
g_i=-\frac{y_i}{n}\operatorname{sigmoid}(-y_iw^\top z_i)w,
\qquad b:=\nabla_\theta L=J^\top g.
$$

Although the second coordinate of each $g_i$ is zero, both coordinates of $R_X$ belong in $J$. The chosen metric penalizes motion in the second coordinate as well. Replacing $J$ by the Jacobian of the scalar logits defines a different method.

## Damped representation pullback

The representation metric is the **sum** of squared Euclidean changes, $\|\delta Z\|^2=\sum_i\|\delta z_i\|_2^2$. Choose a fixed $\lambda>0$ for a run and solve

$$
u_\lambda=\arg\min_u\left[\frac12\|Ju+g\|_2^2+\frac\lambda2\|u\|_2^2\right].
$$

Expanding and differentiating gives

$$
(J^\top J+\lambda I)u_\lambda=-J^\top g=-b.
$$

Because $M=J^\top J+\lambda I$ is positive definite, the minimizer is unique. Compute it using a linear solve, rather than explicitly forming $M^{-1}$. The matrix $J^\top J$ is the pullback of the representation metric and can be singular. The damping term is a penalty on the **update direction**, not weight decay and not an extra term added to the training loss.

Equivalently, the direction minimizes

$$
b^\top u+\frac12u^\top Mu.
$$

The actual discrete update is $\theta^+=\theta+\eta u_\lambda$. The baseline uses $u_{\mathrm{GD}}=-b$ and its own learning rate. Neither method uses momentum, an adaptive optimizer, or a line search in the pilot.

## Spectral explanation

Let the compact singular value decomposition at the current parameters be $J=U_r\Sigma_rV_r^\top$, where the $r$ singular values $\sigma_i$ are positive. The first-order representation velocities are

$$
Ju_{\mathrm{GD}}=-U_r\operatorname{diag}(\sigma_i^2)U_r^\top g,
$$

$$
Ju_\lambda=-U_r\operatorname{diag}\left(\frac{\sigma_i^2}{\sigma_i^2+\lambda}\right)U_r^\top g.
$$

Thus ordinary GD weights a reachable representation direction by $\sigma_i^2$, whereas the damped pullback weights it by $\sigma_i^2/(\sigma_i^2+\lambda)$. The realized finite-step displacement is

$$
R_X(\theta+\eta u)-R_X(\theta)=\eta Ju+O(\eta^2)
$$

for a fixed direction $u$ at a fixed point as $\eta\to0$. Each method has its own $\eta$, so the displayed spectral factors alone do not establish faster training.

The factors act on gradient coefficients $a_i=u_i^\top g$, not on the spectrum alone. Useful diagnostics include $\sigma_i$, $a_i^2/\|g\|^2$, and the unreachable gradient fraction $\|g-U_rU_r^\top g\|^2/\|g\|^2$, with an explicit convention for a zero gradient. At repeated singular values, individual singular vectors are not unique; summed alignment within the corresponding subspace is the meaningful quantity.

## Descent and finite steps

At a point with $b\ne0$,

$$
DL(\theta)[u_\lambda]
=b^\top u_\lambda
=-b^\top M^{-1}b<0.
$$

If $b=0$, then $u_\lambda=0$. Therefore the update is a strict descent direction at every nonstationary point. Since the network and objective are smooth, sufficiently small positive steps reduce the loss. This does not say that any fixed learning rate reduces loss at every iteration.

More explicitly, if the parameter gradient is $\beta$-Lipschitz along the relevant segment, the descent lemma gives

$$
L(\theta+\eta u_\lambda)\le L(\theta)+\eta b^\top u_\lambda+\frac{\beta\eta^2}{2}\|u_\lambda\|^2.
$$

For $u_\lambda\ne0$, a sufficient condition for strict decrease is

$$
0<\eta<\frac{-2b^\top u_\lambda}{\beta\|u_\lambda\|^2}.
$$

The local value of $\beta$ is generally unknown, so this is an explanation of the small-step requirement, not the pilot's learning-rate selection rule.

## Nullspaces, damping, and scale

- Any component of $g$ in $\ker(J^\top)$ is unreachable to first order. Neither method can move the representations along that component at the current parameters. The Jacobian and its reachable subspace can change during training.
- The damped solution has no parameter component in $\ker(J)$. As $\lambda\downarrow0$ at a fixed $J$, it tends to the minimum-norm direction $-J^+g$, and its feature velocity tends to $-U_rU_r^\top g$. This is a projection of $-g$, not necessarily $-g$ itself.
- Small singular values can make the undamped parameter direction large. Damping limits this amplification: $\sigma/(\sigma^2+\lambda)\le1/(2\sqrt\lambda)$, hence $\|u_\lambda\|\le\|g\|/(2\sqrt\lambda)$. Finite-step nonlinear effects can still be substantial.
- As $\lambda\to\infty$, $u_\lambda\sim-\lambda^{-1}b$. At large damping the method approaches a rescaled GD direction; comparing equal learning-rate numbers would be misleading.
- The loss is averaged over samples but the representation metric is summed. This convention is intentional. Averaging $J^\top J$, changing the loss reduction, rescaling the features, or changing the sample size changes the numerical scale of the direction and damping. Such changes require an explicitly revised convention and hyperparameters.
- Adding $\lambda I$ uses the selected Euclidean parameter coordinates. Arbitrary reparameterization invariance should not be claimed for this damped implementation.

## What this method is, and is not

The matrix $J^\top J$ measures movement of endpoint representations. It is not automatically the Fisher information of the logistic prediction model. A generalized Gauss-Newton matrix for this objective would use the loss curvature in representation space, $J^\top\nabla_Z^2\Phi\,J$, which is not the matrix used here. A Wasserstein metric on a distribution of features is another construction and is also not implemented.

The project therefore uses the name **damped representation pullback**. Connections to natural gradients, tangent kernels, and feature learning motivate the study; they do not identify the implementation with every method in those literatures.

## Numerical checks before a training comparison

1. Compare the independently computed parameter gradient with $J^\top g$.
2. Check a centered finite-difference directional derivative against $Jv$.
3. Check the relative residual of $(J^\top J+\lambda I)u=-J^\top g$.
4. Check both spectral formulas against direct Jacobian-vector products.
5. Check $b^\top u<0$ at a nonstationary initialization.
6. For a sequence of decreasing step sizes, compare the actual feature displacement with $\eta Ju$. The absolute linearization error should decrease quadratically, and the relative error approximately linearly, until floating-point effects dominate.
7. Verify small-step loss decrease and that two paired methods begin from identical data and parameters.

Tolerances, step sizes, and measured errors belong in the generated validation report. Passing these checks supports implementation correctness; it is not evidence that the proposed method outperforms GD.
