# Strict novelty audit of the endpoint-neutrality classification

Date: 2026-11-21. This is a bounded primary-literature and corollary audit, not a priority certificate. It supplements the earlier endpoint-selection audit. No training or new numerical experiment was performed. Mathematical claims attributed to the team are being checked independently in the companion audit files.

## Decision

The broad interpretations should be rejected: this is not a new matrix metric, a new link between noncommutativity and nonintegrability, a new Jordan decomposition, or a new conservation-law method. The available primary sources already supply those ingredients.

The narrow remaining result is a **finite polynomial test for a particular normal projected Lyapunov one-form, its reduction to central normal directions in the generated Jordan algebra, and its equivalence to endpoint neutrality under fixed positive quadratic weights**. The team has independently validated the algebraic bridge and the resulting finite-damping endpoint statement under its convergence assumptions. I have not obtained the finite generator test and operational converse by direct substitution into an inspected prior theorem. However, much of their proof reduces to standard matrix functional calculus, a radial primitive, and elementary perturbation of a linear ODE. Consequently the defensible status is a potentially original, specialized theorem or corollary package of modest scope. It is not yet evidence of a substantial new optimization direction.

The Jordan structural interpretation is especially vulnerable to overclaiming. Its center-plus-simple-traceless decomposition is a direct consequence of existing representation theory once the relevant invariance is established. The potentially additional content is the elementary bridge from the finite generator conditions to that hypothesis and its operational optimization converse. The subsequent finite-damping conservation/Bregman argument is a standard application, not a separate new theory.

## Exact claims being separated

Let S be a real linear subspace of Sym_n, N=S-perp under the Frobenius inner product, X0=sI, and T_X=(L_X+R_X)^(-1). On the convex open set Omega=(sI+S) intersect SPD, consider

    omega_X(U) = P_N T_X U,   U in S.

The proposed finite criterion is

    omega is closed
    iff [Z,[Z,U]] is in S and [Z^2,[Z,U]] is in S
        for every Z,U in S.                                      (C)

The second expression equals {Z,[Z,[Z,U]]}. The first condition alone is Lie-triple closure after polarization and is insufficient.

The optimization statement concerns Xdot=-(I+lambda T_X)^(-1) grad L_W(X), where L_W is a fixed weighted squared measurement loss and W is positive definite. For small target displacement and after justified convergence, the coefficient C_W of lambda in the endpoint is

    C_W = - integral along the lambda=0 trajectory of omega.

The additional assertion is that (C) is equivalent to independence of C_W from every fixed positive-definite weight operator for all sufficiently small targets. This is stronger than detecting nonclosedness using arbitrary controlled paths. The completed structural bridge also yields finite-lambda endpoint neutrality whenever an SPD interpolating limit is reached; local convergence is established for sufficiently small targets, a fixed bounded finite-lambda range, and a compact positive-definite weight family. Those qualifications must accompany the finite-damping statement.

For target sI+epsilon E, whitened B0=I_S, B_delta=I+delta D, and V=DE, the team reports

    partial_delta C_delta |0
      = -5 epsilon^3/(288 s^3) P_N [E,[E,V]]
        +13 epsilon^4/(576 s^4) P_N {E,[E,[E,V]]}
        +O(epsilon^5).                                          (W)

The fractions in (W) depend on this target, time, damping, and weight convention; they are not new universal geometric constants.

## Corollary attack 1: known Bures/BKM comparison plus functional calculus

The inverse Lyapunov operator is the Bures/SLD metric operator up to normalization. The logarithmic derivative is the BKM operator. These identifications and the operator-monotone metric framework are established in [Lesniewski–Ruskai](https://arxiv.org/pdf/math-ph/9808016). Entropy as the corresponding Hessian potential is also established; [Grasselli–Streater](https://math.mcmaster.ca/~grasselli/GrasselliStreater01.pdf) prove BKM uniqueness under their monotonicity and mixture/exponential-duality assumptions. These sources were read in the previous audit.

Here is an explicit reduction of the finite criterion, rather than a title-level comparison. Define R_X=(1/2)D log_X-T_X. If omega is closed, its value on radial segments from sI forces its primitive to be (1/2)P_N log X. Therefore closedness is equivalent to

    P_N R_X P_S = 0 on Omega.                                   (R)

With X=sI+Z, expansion of the two known operators gives

    R_X U = ad_Z^2 U/(24s^3)
            - {Z,ad_Z^2 U}/(16s^4) + higher terms.

This immediately supplies necessity of the two conditions in (C). For sufficiency, the team's finite-dimensional operator lemma uses the commuting self-adjoint operators A=ad_Z^2 and C=L_Z+R_Z. Invariance under A and CA yields invariance under C on S intersect range(A), using A's pseudoinverse. Since R_X is a joint spectral function of A,C and vanishes on ker(A), all higher orders follow.

This is a short specialization of familiar analytic-operator tools. The inspected metric papers do not state the **normal cross-block** condition (R), or the finite two-coefficient reduction for arbitrary S. Thus they are not an exact theorem-level duplicate. Nevertheless, the finite criterion should be described as a specialized consequence of established metric formulas plus an invariant-subspace lemma, not as new Bures geometry.

An essential distinction prevents a false direct identification: (R) is a zero off-diagonal block of the metric difference, not equality of the metrics on S. If S=Sym_n then P_N=0 and (R) is automatic, although BKM and Bures differ on noncommuting tangent vectors. Similarly, exactness of P_N T_X dX is not Hessian integrability of the induced metric P_S T_X P_S.

## Corollary attack 2: classical Jordan associator ideals

[Brešar–Cabrera–Fošner–Villena, 2005](https://www.impan.pl/shop/en/publication/transaction/download/product/90340), pp.209–215, define a Lie-triple/associator ideal M of a Jordan algebra J by [M,J,J] subset M, where this bracket is the Jordan associator. Their Theorem 1.1 treats nondegenerate J: if U is the Jordan ideal generated by [M,M,M], then M is central or contains the nonzero [U,J,J]. Lemma 3.3 explicitly classifies the ideals of a nondegenerate spin factor as 0, the scalar center, the vector part, and the entire algebra. The paper identifies its definition with Jacobson's associator ideals.

This is a genuine prior-result threat, but its hypothesis cannot be silently substituted for (C). Let J be the unital Jordan algebra generated by S. Condition (C) initially controls expressions using elements of S, including one repeated-square expression. The 2005 theorem assumes closure involving arbitrary elements of all J. Proving that generator condition implies the ideal hypothesis is a separate algebraic step.

[Jacobson's 1949 original paper](https://www.math.uci.edu/~brusso/jacobsonAM1949.pdf), Theorem 2, says every derivation of a finite-dimensional semisimple Jordan algebra in characteristic zero is inner. The theorem and proof were available as an indexed primary-text excerpt; full PDF retrieval failed. It supplies Der(J)=span[L_a,L_b] with a,b in J. It does not, as stated, reduce invariance under all these operators to the finite generator conditions in (C).

There is also a precise source for the final decomposition: [Barnum–Hilgert, Spectral Properties of Convex Bodies](https://winephysicssong.com/wp-content/uploads/2021/09/BarnumHilgertSpectralPropertiesOfConvexBodiesPreprintVersion2.pdf), Appendix A Lemma A.1 and its proof, establishes irreducibility of the connected automorphism action on the traceless part of a simple Euclidean Jordan algebra. Hence, **if** Der(J)S subset S has been proved, standard direct-sum representation theory yields S as a central subspace plus whole traceless simple components. The cited proof traces this structure to Satake and Faraut–Korányi. Those books were not independently inspected here.

Two qualifications matter. First, invariance under inner derivations and the all-slot associator-ideal condition should not be identified without a proof or the Euclidean semisimple decomposition. Second, J is generated by S and I, so a proposed decomposition must also respect that generation condition; omitting a noncentral component entirely cannot generate that component. The resulting classification is a classical corollary once the bridge is supplied. It is not a fresh classification of Jordan algebras or modules.

A further original source narrows that bridge's possible novelty: [Benito–Draper–Elduque, 1999](https://investigacion.unirioja.es/documentos/5bbc680bb750603269e8017c/f/5ddc084152922232560fc774.pdf), Theorems in Sections 3.7 and 4.3, classifies invariant bilinear products on simple finite-dimensional Lie triple systems in characteristic zero. In particular, nonzero symmetric invariant products occur on Jordan-type traceless modules; Section 3.2 identifies the projected Jordan product. Section 4.2 reconstructs a Jordan algebra by adjoining a scalar and an invariant bilinear form. This already occupies much of an argument that recognizes irreducible Lie-triple components by their invariant projected product. It does not state (C), the normal square-defect condition, or a theorem about a specified symmetric-matrix embedding's generated Jordan algebra. Those remaining identifications must be exhibited rather than hidden under the abstract classification.

**Final mathematical status received from the team:** the root and independent referee validated the bridge

    (C) iff N intersect J is contained in Center(J)
        iff S = S_c direct-sum (all nontrivial simple factors J_i,0),

where S_c is a subspace of the center and J is the unital Jordan algebra generated by S. A key intermediate consequence is that P_N(Z^2) is central for every Z in S. Central square defects alone were not treated as proving the whole statement: a separate compact-module argument supplies the extension to the generated algebra. The bridge is therefore a proved team result rather than an assumed prior theorem. Its exact finite-degree generator formulation was not located in the inspected primary literature. The module recognition and resulting Jordan decomposition, by contrast, substantially overlap the classical sources above.

For the novelty accounting, the relevant distinction is **a new sufficient-and-necessary test for entering a known structural class** versus a new description of that class. Only the former remains a plausible claim here. The direct elementary proof may be preferable to invoking the full 1999 classification, but an elementary alternate proof of a known structural outcome does not make that outcome new.

## Corollary attack 3: statistical sufficiency and positive projections

[Petz, 1986](https://math.bme.hu/~petz/pdf/29suff.pdf), Theorem 4, characterizes weak sufficiency of a von Neumann subalgebra for two faithful normal states with finite relative entropy through their Radon–Nikodym cocycles and recovery/conditional expectations. Theorem 6 adds modular-orbit conditions for inner perturbations. These assume an algebra and a positive reduction map; (C) starts with an arbitrary real measurement subspace and has neither assumption.

[Jenčová, 2012](https://arxiv.org/pdf/1107.0453), Theorem 8, relates reversibility of the adjoint of a unital Schwarz map to preservation of all monotone Fisher metrics, or one metric whose representing measure has sufficient support. Remark 4 explicitly constructs Bures preservation without reversibility. Therefore an informal identification of Bures-related neutrality with Petz sufficiency is invalid even within quantum statistics.

Two current primary papers were also checked. [van Luijk–Wilming, 2026](https://arxiv.org/html/2604.08380), Theorem 6.3 and Corollaries 6.4–6.5, extend recovery/sufficiency statements to positive maps and sufficient Jordan structures. [Yamagata, 2026](https://arxiv.org/html/2604.23292), Theorems 3–5, treats real unital positive maps, Hermitian fixed-point Jordan algebras, likelihood/SLD data, and modular invariance, including self-adjoint tangent models. Their algebraic hypotheses remain substantive. Neither theorem was found to turn the cross-block identity (R) into a positivity-preserving conditional expectation.

A concrete failed substitution is S=span(diag(1,2,4)). It is commuting and satisfies (C), while S+RI is not closed under squares: diag(1,4,16) is not an affine combination of I and diag(1,2,4). Thus neutral S+RI need not be the Jordan range required by the sufficient-algebra theorems. Passing to its generated J restores algebra closure but loses the particular S that is being classified. Together with the full-S cross-block example above, this blocks the most tempting direct reductions.

## Corollary attack 4: SLD torsion, Frobenius, and conservation laws

[Nagaoka–Fujiwara, 2026](https://arxiv.org/html/2307.03431), Eq.3.26 and Theorem 8.6, remains the closest inspected geometric source. It gives SLD double-commutator torsion and an integrability/autoparallel-leaf criterion for distributions defined by a fixed observable score space through every faithful density matrix. Its variables are point-dependent SLD score distributions, whereas (C) concerns a fixed mixture-coordinate affine leaf and normal projection. Its all-state quantifier is also different. Proposition 8.12 shows that real-matrix score spaces in dimension at least three need not produce such a foliation through every complex state; our full real Sym_n example has P_N=0 and passes trivially. A restricted version of their general Frobenius machinery may explain the mechanism, but it does not supply the finite generator test or fixed-weight attainability by unchanged substitution.

[Ciaglia–Di Cosmo–Ibort–Marmo](https://arxiv.org/html/2207.08694), Propositions 1–2, relates dual-connection torsion, gradient frames, and closed one-forms. This already occupies the generic integrability mechanism. The candidate should not be advertised as discovering that mechanism or as a new holonomy construction.

[Marcotte–Gribonval–Peyré, 2023](https://papers.nips.cc/paper_files/paper/2023/file/c7bee9b76be21146fd592fc2b46614d5-Paper-Conference.pdf) and [2024](https://raw.githubusercontent.com/mlresearch/v235/main/assets/marcotte24a/marcotte24a.pdf) supply general conservation-law frameworks, including beyond Euclidean flows. Full theorem-level comparison is assigned to the companion finite-damping audit; this literature audit inspected their primary abstracts/search excerpts and does not claim a completed independent proof review.

A current nearby result is [Zhang–Peyré, 2026](https://arxiv.org/html/2610.09356), Proposition 2.1 and Appendix D: split covariance Bures flows admit signed-spectrum conservation laws, with a generated-Lie-algebra completeness analysis. Its metric and variables differ: it is ordinary factor gradient flow, and the paper's damping concerns heavy-ball dynamics. The setup, proposition, and appendix structure were inspected; it is not evidence for an identical PB-damping endpoint theorem.

## Corollary attack 5: finite damping and partial conserved coordinates

The completed structural bridge makes the finite-damping result especially transparent. The flow stays in J. For N in J intersect S-perp, centrality implies that N commutes with every trajectory point. The inverse Lyapunov and logarithmic derivative then agree in that normal pairing. Consequently the normal coordinates of

    X + (lambda/2) log X

are conserved. These are the normal coordinates of the gradient of the familiar strictly convex potential

    Phi_lambda(X) = (1/2)||X||_F^2
                    +(lambda/2) tr(X log X-X).

At an SPD interpolating limit, these conservation equations are precisely the first-order optimality equations of the corresponding Bregman projection. Strict convexity makes the endpoint unique and independent of positive loss weights. This is a direct application of standard partial-conservation and Bregman-minimization reasoning, of the kind already covered by the Marcotte framework and entropy-mirror literature. It strengthens the useful consequence of the algebraic criterion but should not be counted as an independent conceptual innovation. Sufficiency from a nonisotropic X0 in J is the same routine argument; the isotropic start remains crucial to the particular necessary finite-test derivation and static witness expansion audited here.

This audit relies on the companion proof review for ambient projection equivalence and convergence. It does not promote the conditional endpoint argument to a theorem of global convergence from arbitrary starts or for arbitrary targets.

## Corollary attack 6: fixed-loss endpoint realization

[Li–Wang–Lee–Arora, 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/dfa1106ea7065899b13f2be9da04efb4-Supplemental-Conference.pdf), Theorems 4.8–4.9, characterize universal mirror representations using commuting/Lie-algebra conditions, and Corollary 4.11 gives a matrix-factorization obstruction. These concern equivalence for arbitrary time-dependent losses. They do not imply that two constant positive quadratic weights with the same zero set select different interpolants. [Azulay et al., 2021](https://proceedings.mlr.press/v139/azulay21a/azulay21a.pdf), Section 5, supplies the additional warning that a non-Hessian metric may admit a scalar integrating factor preserving trajectories. These sources were inspected in the earlier audit.

There is nonetheless a standard derivation route to (W): at lambda=0 the observable trajectory is explicitly X(t)=sI+epsilon(I-exp(-Bt))E. Differentiate exp(-Bt) at B=I in direction D, insert the known Taylor expansion of R_X, and integrate scalar products of t and exp(-t). Any V can be realized as DE by a self-adjoint D for E nonzero; sufficiently small delta preserves positivity. The nonzero cubic/quartic coefficients then turn failure of either polynomial closure into a static witness.

This derivation is technically elementary once the correct normal one-form is identified. Its value lies in the restricted quantifier: static positive weights rather than time-dependent control, and all small targets rather than a selected example. No inspected source supplied that exact bridge. The rational coefficients and minimal 2x2/3x3 examples are useful verification details, but are not independent innovation claims.

## Claim-by-claim disposition

| Claim | Strict disposition |
|---|---|
| New Euclidean-plus-Bures metric or entropy replacement | Retire; established metric and mirror ingredients |
| Noncommutativity/non-Hessian geometry can affect trajectories | Retire as novelty; established obstruction/torsion framework |
| Lie-triple closure alone characterizes neutrality | False; the team's star-space quartic witness excludes it |
| Center plus whole traceless simple Jordan components | Classical corollary after derivation invariance is proved |
| Cubic+quartic generator test implies central normal directions in the generated J | Team proof independently validated; potentially additional finite-degree recognition lemma, not a new structural class |
| Exact finite criterion (C) for normal Lyapunov-form closedness | Narrow potentially additional lemma; largely standard functional calculus |
| Equivalence to first-lambda neutrality for all fixed SPD weights and small targets | Strongest potentially additional optimization statement; convergence and quantifiers are essential |
| Fractions 5/288 and 13/576 | Protocol-specific calculation, not standalone novelty |
| Finite-lambda neutrality and Bregman endpoint for passing spaces | Valid under the companion convergence/SPD-limit assumptions; standard conserved-coordinate consequence of the structural bridge |

## What this audit establishes and leaves open

The audit does more than report no exact keyword match: it exhibits the transformation into known metric operators, identifies which Jordan classification becomes an actual corollary, and provides concrete reasons that sufficiency, SLD foliation, and universal mirror theorems cannot be substituted unchanged. It also makes clear how short and standard the residual derivations may be.

A defensible research claim should name the precise normal-projection criterion and the fixed-weight endpoint equivalence, cite the existing geometry and Jordan structure, and present a proof of every bridge. The current evidence supports retaining a narrow theorem for further scrutiny, with a modest novelty assessment. It does not justify an unconditional originality claim, a broad new-method narrative, or substantial training expenditure. The source access limits and search families are recorded in SEARCH_LEDGER.md.
