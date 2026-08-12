# REVE comparison workspace

This isolated workspace prepares REVE for coordinate-aware EEG-only transfer.
A0–A8 alignment and the 90-job public-development matrix are complete;
protected evaluation remains separately locked.

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

The current frozen public status is recorded in
`evidence/alignment_v2/summary_final.json` and
`evidence/public_development_v2/matrix_completion_summary.json`. Public
validation results are not table-admissible. Single-Trial MI/MA remain in the
target-corpus-overlap track and do not enter the support-matched direct table.
