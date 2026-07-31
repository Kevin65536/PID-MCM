# BIOT comparison workspace

This isolated workspace prepares BIOT for the EEG-only official-pretrained
linear-probe track. B0 source fixing is complete; B1-B4 remain pending.

- Official code: `upstream/`, revision
  `d138e32634e52ae9fa6ec98ac9c4087b14ca869a`.
- Official weights: the three hash-verified files in
  `upstream/pretrained-models/`.
- Machine-readable provenance: `sources/method_manifest.yaml`.
- Re-fetch helper: `scripts/fetch_weights.sh` (downloads copies to
  `checkpoints/`).

The adapter may consume EEG only, resampled under the shared data contract. It
must not consume fNIRS, derived teacher features, or synthetic channel copies.
No formal run is authorized until the adapter, smoke, source-fidelity, and
protocol gates are frozen.
