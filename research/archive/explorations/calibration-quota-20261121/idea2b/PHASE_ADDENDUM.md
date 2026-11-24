# Pre-test calibration phase correction

Root review identified that v2 calibration restarted the same seed for phases as for the radius sampler, reusing pseudorandom bits. Complete128-angle orbits make this chiefly a finite-aliasing issue, but the intended independent phase stream should be explicit. The v2 policy calibration had already been committed; no test data or outcomes had yet been generated or evaluated.

Preserve v2 code as experiment_v2_frozen.py, its freeze_addendum.json, and its full calibration/policy commit in fresh_v2_shared_phase. Generate the final fresh calibration with identical radii and an independent phase stream745000+i. The IID legacy baseline is unchanged. All budgets, methods, policies, endpoints and statistical comparisons remain fixed. This is a documented pre-test correction; there is no outcome comparison or choice between phase variants.
