# CBraMod source-fidelity boundary

The model code is pinned to the official GitHub revision; the checkpoint is
pinned separately to the official Hugging Face revision. The upstream code
expects 200 Hz EEG divided into 200-sample patches.

The project loader verifies the official checkpoint before
`weights_only=True` deserialization, instantiates the pinned upstream
`CBraMod`, and strict-loads all 211 state entries. Its GPU smoke preserves the
upstream 200 Hz rate, 200-sample patch, spectral projection, criss-cross
encoder, and 200-dimensional token width.

| Surface | Upstream quick example | Project public-performance v1 |
| --- | --- | --- |
| Channels | example uses 22 channels | the same frozen 16 measured channels used by BIOT/REVE |
| Window | four 200-sample patches | eight patches for 8-second tasks; two for DSR |
| Token head | flatten + multi-layer classifier | mean over channel/time tokens + linear head |
| Optimization | downstream example trains a nonlinear classifier | frozen encoder; train-only standardized linear probe |
| Split | example-only | shared public strict-cross-subject registry |

Mean token pooling is a project transfer adapter, not an upstream downstream
head. The named boundary is
`official_pretrained_cbramod_encoder_mean_token_linear_probe_v1`; B3 passes for
that documented transfer boundary and does not claim reproduction of the
paper's task values.
