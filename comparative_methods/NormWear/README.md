# NormWear EEG-fNIRS adaptation workspace

This workspace is explicitly an EEG-fNIRS adaptation of NormWear, not an
original-paper fNIRS reproduction. Source fixing, release-asset acquisition,
safe checkpoint inspection, A0 cell registration, full-public A1-A4 data
validation, executable A5/A6 adapter smoke, full-public A7 production replay,
and the 90-job serial A8 public-development matrix are complete. Protected
evaluation remains separately locked.

- Official code: `upstream/`, revision
  `07517fcb13def8c89cb586128359cec02f86ec8d`.
- Official backbone: `checkpoints/normwear_pretrain_ckpt.pth`.
- Optional official MSiTF module:
  `checkpoints/normwear_msitf_zeroshot_last_checkpoint-5.pth`.
- Provenance and hashes: `sources/method_manifest.yaml`.
- Frozen adapter decisions: `configs/alignment_v2.yaml` and
  `IDENTITY_AND_ADAPTATION_AUDIT.md`.
- Full-public data evidence: `evidence/alignment_v2/data_boundary_summary.json`.
- Executable GPU smoke: `evidence/adapter_smoke_v2/summary.json`.
- Full-public production replay: `evidence/alignment_v2/summary.json`.
- Final A0-A8 freeze: `evidence/alignment_v2/summary_final.json`.
- Public matrix summary:
  `evidence/public_development_v2/matrix_completion_summary.json`.
- Re-fetch helper: `scripts/fetch_weights.sh`.

All results must use the name `normwear_eeg_fnirs_adapted`. The adapter must
record EEG/HbO/HbR identities, native and target rates, CWT parameters, masks,
and channel aggregation. NormWear's upstream pretraining did not include
fNIRS.
