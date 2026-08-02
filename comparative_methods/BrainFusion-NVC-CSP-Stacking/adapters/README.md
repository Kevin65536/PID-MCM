# Adapter implementation status

The synchronized public adapter now preserves frozen real EEG/HbO/HbR
inventories and rejects missing or padded support. The feature layer implements
the public `avg_raw` NVC branch, dynamic Pearson-contribution sequences,
training-only NVC-pair selection, and deterministic regularized CSP on CPU or
CUDA.

The complete fold-local pipeline now selects SVM/RF base learners independently
for EEG, HbO, HbR, and NVC views using training-only grouped OOF predictions.
The linear-SVM meta-estimator sees only those OOF scores. Feature state and
classical estimator state are serialized separately, hash-bound in a public
manifest, and reload to identical predictions.
