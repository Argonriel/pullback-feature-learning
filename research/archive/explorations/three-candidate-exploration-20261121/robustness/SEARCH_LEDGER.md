# Search ledger: robustness and global convergence

Date: 2026-11-21. Primary sources were preferred. Exact source content inspected is distinguished from indexed search snippets.

## Robustness search

Queries included approximate Jordan subalgebra stability; near-commuting real symmetric matrices; dimension-independent matrix rounding; singular Riemannian barrier methods; vanishing Hessian barrier parameter; and Bures gradient-flow boundary limits.

- Cerjan–Loring (2024), *Even spheres as joint spectra of matrix models*, original author-hosted PDF, Section 7 inspected: https://alexcerjan.com/documents/papers/loring_even_spheres_jmaa_2024.pdf . Relevant to distinctions between real and complex almost-commuting tuples and topology. Not an exact theorem about our defect functional.
- Attouch–Bolte–Redont–Teboulle (2004), original author-hosted PDF, introduction, Section 2, Theorem 2.1, Figure 1 description, and links to Sections 3/5 inspected: https://bolte.perso.math.cnrs.fr/SingularRiemannianBarrier.pdf . Exact reduction of the diagonal baseline. This is the decisive reason not to count the observed boundary layer as an independent innovation.
- Alvarez–Bolte–Brahic (2004), original author-hosted paper setup and Hessian/Bregman characterization inspected, also available as arXiv:1811.10331 (later posting of an older paper): https://www.math.tecnico.ulisboa.pt/~brahic/Legendre10.pdf . No assertion that the 2018 arXiv posting makes the result new.
- Hyers–Ulam stability and Jordan-homomorphism search results were not promoted: their objects differ from a fixed matrix subspace satisfying the finite closure criterion. No claimed theorem was derived from those snippets.
- No theorem-level quantitative modulus beyond standard continuity/algebraic error-bound methods was proved. No negative result about all possible robustness research is claimed.

## Global convergence search

Representative queries:

- global convergence damped Gauss Newton matrix factorization positive semidefinite matrix sensing no RIP full rank initialization
- Bah Rauhut Terstiege Westdickenberg 2022 deep linear networks global convergence Riemannian gradient flows
- Bures gradient flow convex function positive definite global convergence full rank matrix factorization
- gradient flow full rank convex positive semidefinite convergence factorization
- matrix sensing gradient flow positive definite convergence arbitrary measurements
- factorization gradient flow bounded full-rank convex optimization convergence

Sources actually read:

| Primary source | Material inspected | Scope outcome |
|---|---|---|
| Bah et al., arXiv:1910.05505 v5 / published 2022, https://www.instmath.rwth-aachen.de/~mwest/files/linear.pdf | Setup, Theorem 5, Theorems 39–40, full-rank covariance and balanced-initialization qualifications | Coercive regression, different metric; no direct corollary |
| Zhang–Peyre 2026, https://arxiv.org/html/2610.09356 | Assumption1.1, Theorem3.1, support-gap definition and AppendixC.3 | Allows underdetermined PL loss, but tied PSD embedding has zero signed support gap; different damping |
| Zhang–Fattahi–Zhang JMLR2023, https://jmlr.org/papers/volume24/22-0882/22-0882.pdf | Definitions1–2, Theorem4, Theorem8, Corollary9 and surrounding assumptions | Different preconditioner; local restricted convexity and perturbed global method |
| Wu–Rebeschini NeurIPS2021, https://proceedings.neurips.cc/paper/2021/file/abf0931987f2f8eb7a8d26f2c21fe172-Paper.pdf | Proposition5 / Corollary6 in original indexed text; previously read full primary in earlier audit | Commuting subclass already occupied |
| Alvarez–Bolte–Brahic 2004, link above | General framework and metric assumptions | Hessian requirement blocks direct noncommuting reduction |

Additional indexed-only primary results:

- *Globally Q-linear Gauss-Newton Method for Overparameterized Non-convex Matrix Sensing*, NeurIPS2024, https://papers.nips.cc/paper_files/paper/2024/file/244da015b91e64f2d9362703fa2a902b-Paper-Conference.pdf . Indexed Theorem2/nearby assumptions involve RIP. Full proof was not audited in this round; no stronger exclusion claim is made.
- GNMR original publication indexed under https://pmc.ncbi.nlm.nih.gov/articles/PMC11784930/ . Indexed convergence statement requires RIP and sufficiently accurate balanced initialization. Not used as the main novelty boundary.
- Generic full-observation symmetric matrix-factorization stability papers were found, but not used to certify priority.

## Not searched exhaustively

Classical Jordan-cone barrier optimization, all factor-side analytic metric flows, and historical all-full-rank-start convergence theorems have not been exhausted. A search failure is not an originality proof. The audit supports a narrowed potential contribution, not a priority guarantee.
