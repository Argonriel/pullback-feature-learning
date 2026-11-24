# Comparing the two directions at equal predicted feature motion

Consider a fixed training state. Write \(g=\nabla_Z\Phi\), \(J=D_\theta Z\), and \(A=JJ^\top\). Parameter GD induces the feature velocity \(v_G=-Ag\). The damped pullback update induces \(v_P=-A(A+\lambda I)^{-1}g\) for \(\lambda>0\). The identity follows immediately by taking an SVD of \(J\).

**Proposition.** If \(Ag\ne0\), rescale each candidate velocity to the same Euclidean feature norm \(r>0\). The pullback direction yields at least as large a first-order decrease in training loss as GD. The comparison is strict whenever \(g\) has nonzero projections onto two eigenspaces of \(A\) with distinct positive eigenvalues. A component of \(g\) in the nullspace can be arbitrary. Equal RMS motion gives the same comparison because both candidates are evaluated on the same training batch.

**Proof.** At fixed norm \(r\), the predicted decrease equals \(r\|g\|\) times the cosine of the angle between the chosen velocity and \(-g\). Use an orthonormal eigenbasis \(e_i\) of \(A\), with eigenvalues \(s_i\ge0\), and weights \(w_i=(e_i^\top g)^2/\|g\|^2\). For \(t\ge0\), define

\[
a_i(t)=\frac{s_i}{1+t s_i},\qquad m_k(t)=\sum_i w_i a_i(t)^k.
\]

For a direction with spectral factors \(a_i(t)\), its cosine with the desired direction is \(C(t)=m_1/\sqrt{m_2}\). At \(t=0\) this describes GD. At \(t=1/\lambda\) it describes pullback up to a positive scalar, which disappears in the cosine. Since \(a'_i=-a_i^2\),

\[
C'(t)=\frac{m_1m_3-m_2^2}{m_2^{3/2}}\ge 0 .
\]

The inequality follows by Cauchy–Schwarz applied to the vectors with entries \(\sqrt{w_i a_i}\) and \(\sqrt{w_i a_i^3}\). Equality holds precisely when all positive \(s_i\) carrying gradient weight are equal. If \(Ag=0\), both velocities vanish and no rescaling to positive motion exists.

**Empirical check.** The paired one-step experiment starts both methods at identical parameters and matches the norm of the *predicted* feature velocity before applying a finite parameter update. There were five states per pair and two prescribed lengths per state, across eight pairs. Pullback produced lower observed training loss in all 80 comparisons. At the 16 comparisons starting from late pullback states, it produced lower validation loss in only five. The finite-step outcome is empirical. The proof predicts the local training derivative. It makes no claim about finite-step validation accuracy or computation time.

The formal identity connects the previously derived spectral filters to the direction-versus-length experiment. A precedent for experimentally exchanging step direction and magnitude in parameter space is [Agarwal et al.](https://arxiv.org/abs/2002.11803). Here the matched length is measured in training feature space.
