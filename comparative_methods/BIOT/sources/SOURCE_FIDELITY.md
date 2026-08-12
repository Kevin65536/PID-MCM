# BIOT source-fidelity boundary

The code and all three EEG checkpoints come from the official BIOT repository
at the revision in `method_manifest.yaml`. The checkpoint corpus descriptions
are taken from that revision's README.

The project route reuses the exact upstream `BIOTEncoder`, verifies the PREST-16
checkpoint before `weights_only=True` deserialization, and loads its state
strictly. The audited GPU path preserves the upstream 200 Hz rate, 200-sample
STFT window, 100-sample hop, 256-dimensional mean encoder output, and frozen
backbone.

| Surface | Upstream example | Project public-performance v1 |
| --- | --- | --- |
| Checkpoint | PREST-16 / SHHS+PREST-18 / six-dataset-18 | PREST-16 globally |
| Channel semantics | fixed 16-channel bipolar PREST/TUH montage | task-frozen 16 measured native electrodes, reordered by identity |
| Window | typically 5 or 10 seconds | 8 seconds; DSR 2 seconds |
| Head | task-specific end-to-end classifier | train-only standardized frozen feature + linear head |
| Split | upstream corpus-specific | shared public strict-cross-subject registry |

The native-electrode panel is a deliberate positional-transfer deviation: no
bipolar montage is synthesized when the required source electrodes are absent.
It must not be described as an original PREST/TUAB/TUEV input reproduction.
The named-method boundary is therefore
`official_pretrained_biot_encoder_native_electrode_positional_transfer_v1`.
B3 passes for that explicitly adapted transfer boundary, not for reproduction
of the paper's downstream numbers.
