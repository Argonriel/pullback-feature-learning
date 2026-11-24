# Theory report review

Result: CLEAR / PASS. No outstanding semantic or visual issue on the inspected pages.

Reviewer: structured-branch author reviewing the root manuscript. This is a report-consistency and visual review, not a new independent audit of this reviewer's theorem or experiment. Root's separate independent reconstruction is recorded in STRUCTURED_RESULTS.json.

## Scope and final inspection

Visually inspected all of pages 6, 7 and 8 of the nine-page rendered PDF, with the manuscript and committed result/counterexample/recovery records beside them. After two precision comments, re-inspected the regenerated final pages 6 and 8. Page 7 was unchanged. Text, formulas, the label table, margins and page footers are readable and unclipped; no overlap or broken glyphs were observed. No manuscript file was edited by this reviewer.

## Checks

- Page 6: IID same-P batch records and independent tie keys, the single-coordinate-invariant selector, 64/256 quota and 320-forward deployment cost are explicit. The retained harmonics, odd orthogonal supplied residual, |r|<=1 and 0<=epsilon<=1 are stated. The joint endpoint penalty, evenness, deterministic range D[epsilon/2+(1-epsilon/2)rbar], and lower confidence expression match the frozen derivation. The 8,576+14+32=8,622 event allocation and finite-grid limitation are correct.
- Page 7: assumed-symmetry primary results are 16/16, mean .014078 and minimum .002031. The independently labeled epsilon upper bound .044102 leaves 14/16, mean .005430. The other label rows are .141525 -> 0/16, .046253 -> 15/16, and .143335 -> 0/16. Within-cell posterior constancy is explicitly assumed, not established by labels. The sine counterexample and distinction between eight models/two policies and sixteen independent models are preserved.
- Pages 7-8: the covered strong-third law retains negative bounds; the omitted fifth with false rho=0 is outside coverage and its sixteen positive formula values are excluded. The analytic counterexample numbers -.00687578 and +.00062422 agree with the saved quadrature; equality is correctly qualified as the un-clipped expression. Model-adversarial posterior worlds are identified as model-dependent stress cases.
- Page 8: successful verification inputs/forwards 29,360,128 / 17,967,530; truth 3,670,016 / 7,340,032; replay 35,840; parameter inputs 1,835,008; labels 524,288. Failed-attempt overhead 65,536 regenerated inputs and 131,712 forwards is separate. These match the branch records.
- Page 8: the final wording correctly attributes the maximum 2.78e-17 excursion to the recovery check, with the <=1e-12 guard and identical-stream replay. The original failure/source preservation and absence of a scientific change are clear.

## Resolved review comments

1. Added the explicit epsilon domain needed by endpoint convexity, and clarified IID same-P inputs/tie keys.
2. Changed the original-failure claim from a specific -2.78e-17 stopping value to tiny negative roundoff, with the maximum 2.78e-17 correctly attributed to recovery.

Root's independent reconstruction separately reports PASS for 4,288 moments, 1,920 certificate rows, 5,504 truth rows, all 56 verification and 56 truth streams, and 25,307,562 additional model forwards. This review does not replace that audit or claim formal directed-rounding verification.

Reviewed UTC: 2026-11-22T00:38:49.903689+00:00

## Inspected artifact hashes

- `REPORT.md`: `d7819b3e4200bea6645292b1543651cef5115a6113765d28dc37185f74b8760e`
- `research_report.pdf`: `8ebdbd6f04bfae69c35e91b24afbec36878b6ad01ef834a23ac9ca6440fb0af1`
- `pdf_preview/page-6.png`: `3bada5bf960988d56a3c75da84fb59c1f25be6dff047e78c7aa30cf17c3c4ea0`
- `pdf_preview/page-7.png`: `5fe3e63c6eb3a819ec36970462c176ea833a66aab9d9194f0318e0b5c58a4bde`
- `pdf_preview/page-8.png`: `10a28545138f3d6b0e2fc9566845d94bd755848e2bc99b529943712719d38cc7`
- `structured/RESULTS.json`: `51577b90f625a27314ea4354eb5eeba8b090ded3e52ad839bb3c1d74545a1c21`
- `structured/COUNTEREXAMPLES.json`: `851cf74032a12672e373c739484c57b24090c89cf09cdec36a5038988f936313`
- `structured/STRESS_RECOVERY_FREEZE.json`: `233b26eae367c63bf96e1213b06ee761fb0b41aa65a916eba7c4d25a7e4c0b8b`
- `root_audit/STRUCTURED_RESULTS.json`: `c38d7f6f6df6e519c490c243c24f9a6f00825531a748521e5905434510556e28`

Final artifact refresh UTC: 2026-11-22T00:40:33.314023+00:00. Pages 6-8 PNG hashes were checked and remain byte-identical to the reviewed pages. Only the manuscript/PDF binding hashes above were refreshed after the page-5 timing-exclusion wording update; no new numerical or proof review was required.
