# REVE source-fidelity boundary

The training and downstream code is pinned to the official GitHub revision.
The position bank is pinned to its official Hugging Face revision. Encoder
weights remain gated by the authors' Responsible Use Agreement and may only be
downloaded by a user who has accepted it.

The project uses the same local Hugging Face `AutoModel` route shown by the
official model card, after hashing both executable snapshot code and
safetensors. It strict-loads REVE-base and the official position bank, keeps
the 200 Hz rate, 200-sample patch, 20-sample overlap, encoder attention pooling,
and 512-dimensional output.

| Surface | Official model-card example | Project public-performance v1 |
| --- | --- | --- |
| Encoder | REVE-base | REVE-base |
| Positions | official `reve-positions` bank | same bank; unknown names fail closed |
| Channels | dataset-native named electrodes | shared frozen 16-channel task panels |
| Head | downstream code/config dependent | train-only standardized linear head |
| Split | upstream task-specific | shared public strict-cross-subject registry |

Project geometry provenance is audited before extraction, while numerical
positions are resolved by the official REVE bank. Single-Trial remains in the
known-overlap reporting track because `Shin2017A` is declared in pretraining.
B3 passes for `official_reve_base_position_bank_linear_probe_v1`, not for an
original-task numerical reproduction.
