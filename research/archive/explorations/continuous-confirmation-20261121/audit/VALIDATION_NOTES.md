# Validation notes

The first standalone group-quota algebra check generated 20,000 groups, which is not divisible by 64, so its batch reshape failed before producing results. The audit harness was corrected to use 20,480 groups (320 full batches). No experiment, policy or scientific result was changed in response.

The independent rectangle checker initially used 513 edge-grid nodes before local refinement. Its reference minimum at epsilon 0.01 was 2.8e-6 above the cubic result, failing a tightness comparison while satisfying the safety inequality. The independent reference resolution was increased to 2,049 nodes; the investigated bound was not changed.

The first fresh-certificate audit was interrupted because repeatedly indexing compressed NPZ arrays inside a node loop needlessly decompressed entire tables. It was changed to load each table once and verify all child rectangles in arrays. This performance correction did not alter any scientific artifacts or audit conditions.
