# CBraMod representation-layer audit

This audit freezes the representation boundary used by the project adapter.
It is intentionally separate from the method manifest so that pre-existing
manifest work is not rewritten.

## Decision

The adapter must expose the 200-dimensional encoder latent tokens produced
immediately before CBraMod's pretraining reconstruction projection. It then
averages those latent tokens over the real-channel and real-patch axes. The
result is one 200-dimensional embedding per sample.

The implementation therefore:

1. constructs the upstream CBraMod architecture;
2. verifies and strictly loads the complete official checkpoint;
3. replaces `proj_out` with `torch.nn.Identity` only after the strict load;
4. requires the adapter pooling mode `official_avgpooling_patch_reps`; and
5. rejects encoders that still expose the reconstruction projection.

## Upstream evidence

The frozen upstream revision is
`0ff6be918985689e7df679bc731ffb70e6c6224f`.

| Evidence | SHA-256 | Finding |
| --- | --- | --- |
| `upstream/models/cbramod.py` | `caa7c5a0e4acd488a6f625b56f7b54dc4e7190d6b68ef5b76dff2cef86874a14` | `forward` applies `proj_out` after the transformer latent representation. |
| `upstream/quick_example.py` | `ba9b8483f25bf60a4c6d706c1e5f233f34ccae639c6e22b9e91a5b9cd6d782a1` | The official transfer example replaces `proj_out` with `Identity` after checkpoint loading. |
| `upstream/models/model_for_physio.py` | `ad4e87d12c968e24795117776a22da1ba42afa0cfd9a28e060ccaf174432f094` | A released downstream wrapper removes `proj_out`; `avgpooling_patch_reps` adaptively averages the latent channel/patch axes. |
| `upstream/models/model_for_bciciv2a.py` | `a7d720d2c1603cffdca4485d33d8f3d1db66876c9b9e48dd18f063af5e65b297` | A second released downstream wrapper independently confirms the same representation boundary and pooling route. |

All released `model_for_*.py` downstream wrappers in the frozen source tree
replace `backbone.proj_out` with `Identity`. The former project adapter instead
averaged the output while `proj_out` was still active. Although that tensor had
the same last dimension, it represented the pretraining reconstruction head,
not the official downstream latent representation. This audit closes that
ambiguity.

## Project amplitude-coordinate decision

CBraMod's pretraining trainer and many released downstream dataset loaders
divide their source arrays by 100. That operation is not stacked on the project
input. The comparison registry supplies EEG in the already-frozen canonical
coordinate `canonical_recordwise_robust_standard_deviation`; applying another
fixed `/100` would no longer reproduce the upstream raw-array coordinate
conversion and would attenuate both the convolutional and spectral branches a
second time.

This is an explicit dataset-coordinate adaptation, not an undocumented model
change:

- shared comparison input remains the same canonical 200 Hz EEG signal used by
  the other support-matched methods;
- no method-specific refiltering, recentering, or rescaling is added;
- the adapter receives only real channels and complete real-time support; and
- the deviation is to omit the upstream raw-array `/100` because its role is
  absorbed by the project-level robust-SD coordinate.

## Frozen output contract

- input: `(batch, real_channels, samples)` at exactly 200 Hz;
- temporal partition: contiguous non-overlapping 200-sample patches;
- support: all channels and all samples must be real; no copying or padding;
- latent tensor: `(batch, real_channels, patch_count, 200)`;
- pooling: mean over `real_channels` and `patch_count`;
- output: finite `(batch, 200)` embedding;
- training state: encoder frozen and always evaluated in inference mode.

The adapter smoke test also computes the same latent-token average manually and
requires exact numerical agreement with the public adapter route.
