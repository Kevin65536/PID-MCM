# REVE comparison workspace

> **STATUS: STOPPED（legacy method record）。** REVE alignment, public delivery and
> its later joint-campaign cell evaluation are complete. This workspace is retained
> for evidence and replay only; no queue, launch, or protected follow-up here is current.

_Project-level execution and scientific verdicts are generated in
[`docs/PROJECT_STATUS.md`](../../docs/PROJECT_STATUS.md). Public-development and
data-boundary statements here describe the frozen implementation/evidence layer._

This isolated workspace prepared REVE for coordinate-aware EEG-only transfer.
A0–A8 alignment and the 90-job public-development matrix completed before the later
joint campaign.

- Official code: `upstream/`, revision
  `06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7`.
- Official position bank: downloaded under `checkpoints/reve-positions/`.
- Official encoders: `brain-bzh/reve-base` and `brain-bzh/reve-large`; both
  are downloaded and hash-verified locally after gated access, and remain
  subject to the authors' Responsible Use Agreement.
- Gated fetch helper: `scripts/fetch_weights.py`.

The official pretraining corpus contains `Shin2017A`, which is the project's
Single-Trial dataset. That cell must remain in the
`open_world_pretrained_with_target_corpus_overlap` track. Encoder weights and
derivatives must not be redistributed. Only registry-backed electrode
coordinates may enter the model.

The frozen public-development snapshot is recorded in
`evidence/alignment_v2/summary_final.json` and
`evidence/public_development_v2/matrix_completion_summary.json`. Public
validation results are not table-admissible. Single-Trial MI/MA remain in the
target-corpus-overlap track and do not enter the support-matched direct table.
