# Allocation review of the final root report

Status: **CLEAR**. Reviewed 22 November 2026 UTC by the allocation branch reviewer. Scope: root `REPORT.md` and the final rendered PDF pages 2, 5 and 9. No report or PDF edits were made by this reviewer.

- Page 2 correctly reports raw-objective development selection of the 2,304-training baseline, followed by sixteen fresh baseline-only fits. Absolute raw/full expected NLL and error agree with committed results. Identity differences are not given confidence intervals or improvement tests, and no global-optimality claim is made.
- The slightly better development full-output loss at 3,072 is explicitly exploratory, outside the frozen raw selection rule, and without independent higher-training confirmation. It was not promoted after selection.
- The 4,352 accepted-label budget is correctly procedure-specific. Development fitting/selection costs of 47,104 / 524,288 labels and confirmation fitting/evaluation costs of 69,632 / 1,048,576 labels are accurate and separately disclosed. The 48 model states, 96 temperature fits and 240 risk rows agree with the branch artifacts.
- Page 5 accurately describes actual complete baseline fitting: one warmup and three measured repeats, one BLAS thread, fixed development pair 1, median 11.275 seconds and range 11.138-11.319 seconds. The final wording includes all specified exclusions: input generation/loading, output I/O, test scoring and verification comparisons. It makes no relative-speed claim. The initial omission of verification comparisons was corrected by root and the regenerated page was rechecked.
- Page 9 appropriately recommends separating update-count and calibration-label effects, and requires new prospective streams for a dedicated full-output objective. It does not treat the exploratory 3,072 observation as confirmed evidence.
- Final rendered pages 2, 5 and 9 are readable and consistent with the Markdown. Tables, paragraphs, headers and footers show no clipping, overlap or broken glyphs.

## Reviewed artifact hashes

SHA-256:

| Artifact | Hash |
|---|---|
| `REPORT.md` | `d7819b3e4200bea6645292b1543651cef5115a6113765d28dc37185f74b8760e` |
| `pdf_preview/page-2.png` | `9c2e579bb8ca08825bd338bb97054148a6df8b5beaf70f79f0489e249110cc83` |
| `pdf_preview/page-5.png` | `c28f839abdb47cf246c3ab213090790bc95848671c1e2008d66c6c10a7e6c62f` |
| `pdf_preview/page-9.png` | `62bed5f670777e4b8911cb097991fccf58fa85dc93bbe3fc7720883893bc3fa7` |

No outstanding allocation-specific scientific or layout issue remains in this reviewed version. This review does not replace the separate numerical reconstruction or the other branches' scientific reviews.
