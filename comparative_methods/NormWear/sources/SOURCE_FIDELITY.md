# NormWear source-fidelity boundary

The code and release assets are pinned to official project locations. The
upstream pretraining table covers PPG, ECG, EEG, GSR, PCG, and IMU, but not
fNIRS.

The official backbone checkpoint safely loads as a direct 261-entry
`OrderedDict` and strictly matches the pinned full pretraining model. The
primary track uses only its encoder; the fixed-four-variable reconstruction
decoder and optional MSiTF text module are excluded.

The planned route preserves the upstream optimized Ricker CWT, patch
projection, encoder, cross-channel CLS fusion, and documented mean-patch then
concatenate aggregation. It adds explicit EEG/HbO/HbR identity and deterministic
200/10-to-65 Hz polyphase rate conversion. These are material adaptations.
Every result must therefore use `normwear_eeg_fnirs_adapted`; it cannot be
presented as an original-paper fNIRS reproduction.

The executable audit verifies that an unchunked adapter is bitwise identical
to the pinned upstream encoder. For the full real channel inventories,
per-channel attention is chunked only to bound memory; every even-layer
cross-channel CLS fusion still sees the complete channel set. Retained smoke
evidence bounds the small float32 batching-order difference before any public
matrix is launched. B3 passes for the explicitly adapted reporting name.
