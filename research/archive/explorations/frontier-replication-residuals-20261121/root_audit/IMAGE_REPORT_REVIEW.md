# Independent image-study report review

**Result: PASS; no material numerical, interpretive, or visual defect found.**

Reviewer: distillation branch agent, 21 November 2026. Scope: final rendered PDF pages 3, 4, and 5, viewed individually at full preview resolution and compared with frozen branch outputs and independent root audits. No experiment, fitting, or benchmark was rerun, and no frozen or report file was edited.

- **Page 3 — source and population.** The mapping, eligible counts (2,607 / 6,008 / 3,365), 500 products per class, seven ambiguous-product exclusions, 1,500 distinct products, 435 brands, preprocessing, and zero exact historical pixel overlap match the source record and `EXTERNAL_SELECTION.json`. Academic/noncommercial terms and historical exposure are disclosed. The text explicitly states that retailer origin and category composition change together, only sandals overlap with the preceding six-category population, and product grouping does not remove brand/design dependence. It does not claim an isolated causal retailer effect or a ten-class IID replication.
- **Page 4 — quality and inference.** All twelve displayed error means match `summary.csv` to the printed precision. Primary S-B is +0.233 pp, adjusted interval [-5.523, +5.990]; S-C is +2.211 pp, [-10.583, +15.005]. Both have only three strict seed wins and fail all frozen success conditions. Crossed-bootstrap limits match the saved results. Six-pair approximate Student-t assumptions, multiplicity adjustment, common product resampling, and unresolved dependence are stated. The chart displays all six model points per method and all seed contrasts; neither the wide intervals nor the +20.07 pp contrast is clipped. The failure interpretation is appropriately limited to this replication.
- **Page 5 — cost and serving.** Network-work totals and 997,270 comparator examples match `full_cost_accounting.csv`; exclusions from exact whole-program/time matching are explicit. Displayed median paired full/S ratios are 2.023, 1.729, and 1.361; separately stated pooled ratios are 2.02531, 1.85469, and 1.37319. These distinct aggregates agree with `serving_aggregate.csv` and `IMAGE_TIMING_RESULTS.json`. The frozen pooled S/B <=1.10 rule, FP32 hardware/thread setup, warmup and measured blocks, exclusive window, transfer boundary, individual reversals, and CPU/GPU agreement are correctly reported. Forward counts and all 630 rows agree with the audit. The allocation-fit timing figures agree with `ALLOCATION_TIMING_RESULTS.json`; no comparative allocation speed claim is made. The separate serving pass is not presented as rescuing the failed quality replication.

Visual checks: tables, captions, source contact sheet, plot labels, intervals, prose, page numbers, and margins are readable; no overlap, truncation, missing glyph, or misleading axis clipping was observed.

Final-render recheck: page 5 was inspected again after adding verification comparisons to the allocation benchmark exclusions. Its layout remains clear, and the image-study text and numbers are unchanged. Page 3 and page 4 preview hashes are unchanged. The hashes below identify this final render.

## Reviewed render identity

- PDF SHA-256: `8ebdbd6f04bfae69c35e91b24afbec36878b6ad01ef834a23ac9ca6440fb0af1`
- Page 3 PNG SHA-256: `1b71dff376c9fd4c6a3426af30febece38743d0a3625e370f6f3370dc2c49a33`
- Page 4 PNG SHA-256: `0241ac3712601b464bd90253587bb3e7a41b3ff121d3f3643064c4eb067e754f`
- Page 5 PNG SHA-256: `c28f839abdb47cf246c3ab213090790bc95848671c1e2008d66c6c10a7e6c62f`

This review binds the inspected render. A later regeneration requires checking whether these pages changed.
