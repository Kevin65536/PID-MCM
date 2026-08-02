# B2-B3 test queue

`test_gpu_nvc.py` pins the public NVC source file and verifies that the
vectorized CUDA `avg_raw` implementation matches the published CPU formula for
every EEG-to-fNIRS channel pair. It also checks CUDA residency, finite values,
gradients, and constant-input rejection.

`test_alignment_data_v2.py` verifies the support-matched observation budget,
full public identity boundary, real-channel reordering, and rejection of bad,
missing, or padded support. `test_fold_local_features.py` verifies that dynamic
NVC contributions sum to the public Pearson coefficient and that pair
selection plus CSP are deterministic and bound to one training identity set.

Base estimators, stacking, and the complete reload smoke remain pending, so
BrainFusion has not yet passed B2.
