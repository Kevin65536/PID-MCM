# BIOT comparison workspace

This isolated workspace prepares BIOT for the EEG-only official-pretrained
linear-probe track. A0–A8 alignment and the 90-job public-development matrix
are complete; protected evaluation remains separately locked.

- Official code: `upstream/`, revision
  `d138e32634e52ae9fa6ec98ac9c4087b14ca869a`.
- Official weights: the three hash-verified files in
  `upstream/pretrained-models/`.
- Machine-readable provenance: `sources/method_manifest.yaml`.
- Re-fetch helper: `scripts/fetch_weights.sh` (downloads copies to
  `checkpoints/`).

The adapter may consume EEG only, resampled under the shared data contract. It
must not consume fNIRS, derived teacher features, or synthetic channel copies.
The current frozen public status is recorded in
`evidence/alignment_v2/summary_final.json` and
`evidence/public_development_v2/matrix_completion_summary.json`. Public
validation results are not table-admissible and do not authorize protected
evaluation.
