# Preserved failed attempts and limitations

1. The default `python` command resolved to the Microsoft WindowsApps alias and could not start. The bundled runtime Python supplied by the parent was then used. No computation ran in the failed alias attempt.
2. Native PowerShell and initial bundled-Python source downloads failed because the default execution network could not resolve scholarly hosts. The Python failure records are preserved in `FAILED_SOURCE_RETRIEVAL.json`; the PowerShell script and its error pattern remain recorded here. The normal read-network escalation then succeeded for all requested originals. No source data were inferred from the failed requests.
3. An attempted direct HTML URL for arXiv 2206.10394v2 returned 404 through the browsing tool. The original arXiv PDF was available and was inspected instead.
4. Several exact-term searches returned unrelated Jordan normal-form or monotone-submanifold results. Those were excluded as terminological false positives.
5. The finite-support noise bound in `MOMENT_CEILING.md` is deliberately nonsharp. It is not an endpoint signal lower bound without a quantitative condition on the observation geometry.

No negative mathematical finding was discarded: the established nature of the algebraic class, metric framework, and quadrature construction is part of the final narrowed verdict.
