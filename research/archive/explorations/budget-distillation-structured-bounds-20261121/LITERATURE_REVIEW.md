# Primary-source context and claim boundaries

Consulted 21 November 2026. Branch notes add sources specific to their experiments and data licenses.

- Hinton, Vinyals and Dean, *Distilling the Knowledge in a Neural Network* (2015), https://arxiv.org/abs/1503.02531 . Soft targets transfer an ensemble's information to a single model; this predates the present project. Here the teacher averages two transformed views, with explicit target-generation and teacher costs.
- Guo, Pleiss, Sun and Weinberger, *On Calibration of Modern Neural Networks*, ICML 2017, https://proceedings.mlr.press/v70/guo17a.html . Positive temperature scaling is established. Our question concerns allocation of scarce labels and fitting the final deployed pipeline, rather than proposing a new calibration method.
- Chen, Dobriban and Lee, *A Group-Theoretic Framework for Data Augmentation*, JMLR 21(245), 2020, https://jmlr.org/papers/v21/20-163.html . Orbit averaging and approximate-invariance analysis are prior art. The new theory specializes an involution, coordinatewise invariant quota mask, and known or bounded orbit density imbalance.
- Shanmugam, Blalock, Balakrishnan and Guttag, *Better Aggregation in Test-Time Augmentation*, ICCV 2021, https://arxiv.org/abs/2011.11156 . Averaging can turn some correct predictions into errors even when its average effect is favorable. Neither transformation semantics nor teacher agreement guarantees improved external accuracy.

The external data are used to challenge transfer, not to redefine the original task after seeing scores. A shifted corpus with restricted class coverage is an external stress test and must not be presented as a full ten-class IID replication. Mathematical guarantees require the stated input-family and posterior assumptions; unlabeled finite samples cannot supply unrestricted posterior symmetry or a distribution-free nonvacuous TV upper bound.
