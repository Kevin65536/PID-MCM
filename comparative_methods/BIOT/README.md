# BIOT comparison workspace

> **STATUS: STOPPED（legacy method record）。** BIOT alignment, public delivery and
> its later joint-campaign cell evaluation are complete. This workspace is retained
> for evidence and replay only; no queue, launch, or protected follow-up here is current.

_Project-level execution and scientific verdicts are generated in
[`docs/PROJECT_STATUS.md`](../../docs/PROJECT_STATUS.md). Public-development and
data-boundary statements here describe the frozen implementation/evidence layer._

This isolated workspace prepared BIOT for the EEG-only official-pretrained
linear-probe track. A0–A8 alignment and the 90-job public-development matrix
completed before the later joint campaign.

- Official code: `upstream/`, revision
  `d138e32634e52ae9fa6ec98ac9c4087b14ca869a`.
- Official weights: the three hash-verified files in
  `upstream/pretrained-models/`.
- Machine-readable provenance: `sources/method_manifest.yaml`.
- Re-fetch helper: `scripts/fetch_weights.sh` (downloads copies to
  `checkpoints/`).

The adapter may consume EEG only, resampled under the shared data contract. It
must not consume fNIRS, derived teacher features, or synthetic channel copies.
The frozen public-development snapshot is recorded in
`evidence/alignment_v2/summary_final.json` and
`evidence/public_development_v2/matrix_completion_summary.json`. Public
validation results alone are not table-admissible; final cell verdicts come from
the tracked joint-campaign result report.
