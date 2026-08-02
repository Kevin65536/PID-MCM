# Adapter implementation status

The synchronized public adapter now preserves frozen real EEG/HbO/HbR
inventories and rejects missing or padded support. The feature layer implements
the public `avg_raw` NVC branch, dynamic Pearson-contribution sequences,
training-only NVC-pair selection, and deterministic regularized CSP on CPU or
CUDA.

Base-estimator selection, leakage-safe out-of-fold stacking, and complete
pipeline serialization remain the next serial implementation gate.
