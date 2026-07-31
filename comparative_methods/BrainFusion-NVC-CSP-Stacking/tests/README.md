# B2-B3 test queue

`test_gpu_nvc.py` pins the public NVC source file and verifies that the
vectorized CUDA `avg_raw` implementation matches the published CPU formula for
every EEG-to-fNIRS channel pair. It also checks CUDA residency, finite values,
gradients, and constant-input rejection.

GPU CSP, base estimators, stacking, fold-local fitting, and the complete reload
smoke remain pending, so BrainFusion has not yet passed B2.
