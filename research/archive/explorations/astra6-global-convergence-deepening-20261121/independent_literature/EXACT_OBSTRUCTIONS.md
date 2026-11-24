## Exact tied-covariance zero-gap calculation

This independent calculation uses the primary definition above. Write
\[
\Sigma=\begin{bmatrix}X&X\\X&X\end{bmatrix},\qquad
J=\operatorname{diag}(I,-I),\qquad X\succ0.
\]
For any nonzero \(v\), \(z=(v,-v)\) is in the kernel of \(\Sigma\) and \(z^TJz=0\).
If \(\Sigma-aJ\succeq0\), then \(z^T(\Sigma-aJ)z=0\), so positivity implies
\((\Sigma-aJ)z=0\). But this vector equals \(-a(v,v)\), forcing \(a=0\).
The same argument applies to \(\Sigma+aJ\). Thus \(a_+=a_-=g=0\).
Equivalently \(J\Sigma\) has square zero. No numerical rank threshold is involved.

## Exact failure of the ordinary Bregman Lyapunov route

Let \(J=\operatorname{Sym}(2)\), \(S=\{Z:\operatorname{tr}Z=0\}\), and \(N=\mathbb RI\).
Use Frobenius-orthonormal measurement matrices
\[
D=\operatorname{diag}(1,-1)/\sqrt2,\qquad
O=\begin{bmatrix}0&1\\1&0\end{bmatrix}/\sqrt2.
\]
For \(X=\operatorname{diag}(a,b)\), with \(a,b>0\) and \(a\ne b\), define
\[
B_X=\nabla^2\Phi(X)=I+(\lambda/2)D\log_X,\quad
M_X=I+\lambda K_X^{-1},\quad Q_X=B_XM_X^{-1}.
\]
On the \(D,O\) coordinates, \(Q_X=\operatorname{diag}(1,q)\), where
\[
q=\frac{1+(\lambda/2)\log(a/b)/(a-b)}
        {1+\lambda/(a+b)}>1.
\]
The strict inequality follows from the strict logarithmic-mean/arithmetic-mean inequality.
Choose any \(c\in(2\sqrt q/(1+q),1)\), so
\[
W=\begin{bmatrix}1&c\\c&1\end{bmatrix}\succ0.
\]
Set \(d=\epsilon(-c(1+q)/2,1)\) and \(Y=X-d_1D-d_2O\).
For every sufficiently small \(\epsilon>0\), \(Y\succ0\).
Take the measurement map \(\mathcal AZ=(\langle D,Z\rangle,\langle O,Z\rangle)\) and
\[
f(Z)=\tfrac12(\mathcal AZ-\mathcal AY)^TW(\mathcal AZ-\mathcal AY).
\]
At initialization \(X\), its gradient has coordinates \(Wd\). Along the actual damped flow,
\[
\frac{d}{dt}D_\Phi(Y,X(t))\Big|_{t=0}
=-d^TQ_XWd
=\epsilon^2\left(\frac{c^2(1+q)^2}{4}-q\right)>0.
\]
This derivative is unchanged by replacing \(Y\) with another feasible comparator in \(J\):
their difference \(N\) is central, \(B_XN=M_XN\), and the derivative change is
\(-\langle N,G\rangle=0\).
It therefore also applies to the selected Bregman endpoint once its existence is independently established.

The metric is not a Hessian in these affine matrix coordinates. For \(E=E_{11}\),
the necessary symmetry of third derivatives would require
\[
\partial_E\langle O,M_XO\rangle
=-\frac{\lambda}{(a+b)^2}
=\partial_O\langle E,M_XO\rangle
=-\frac{\lambda}{2a(a+b)},
\]
which fails if \(a\ne b\).
This rules out an affine-coordinate Hessian potential for \(M\), not arbitrary transformations,
auxiliary Lyapunov functions, or loss-dependent constructions.


