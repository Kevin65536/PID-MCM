# NormWear source-fidelity boundary

The code and release assets are pinned to official project locations. The
upstream pretraining table covers PPG, ECG, EEG, GSR, PCG, and IMU, but not
fNIRS.

The planned route preserves the upstream CWT/tokenization and backbone
principles while adding explicit EEG/HbO/HbR identity, rate conversion, masks,
and aggregation. Those are material adaptations. Every result must therefore
use `normwear_eeg_fnirs_adapted`; it cannot be presented as an original-paper
fNIRS reproduction. B3 remains pending until the official embedding example
and a complete deviation table pass.

