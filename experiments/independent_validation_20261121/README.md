# Independent validation sources

These sources mirror the independent audits delivered with the 21 November 2026 budget/distillation/structured-bound study. The data and completed outputs remain in the separate Research archive, under `budget-distillation-structured-bounds-20261121/root_audit`.

Execute the archived audit copies in their original study layout: the scripts resolve data relative to their own location. They independently reconstruct metrics, calibration, uncertainty, costs, source selection, and theorem computations. They do not retrain all models. `external_selection.py --zip PATH` additionally needs the retained versioned full source ZIP to replay excluded candidates; the compact archive contains every evaluated original image.

NumPy, threadpoolctl and Pillow are required. The image replay also requires CUDA-enabled PyTorch. No completed research output should be overwritten when starting a new experiment; use fresh destinations and preserve source commitments. See the archive's README and audit execution notes for exact run records and limits.
