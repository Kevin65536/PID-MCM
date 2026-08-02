# NormWear EEG-fNIRS adaptation workspace

This workspace is explicitly an EEG-fNIRS adaptation of NormWear, not an
original-paper fNIRS reproduction. Source fixing, release-asset acquisition,
safe checkpoint inspection, and A0 cell registration are complete. Public data
and executable adapter validation remain gated work.

- Official code: `upstream/`, revision
  `07517fcb13def8c89cb586128359cec02f86ec8d`.
- Official backbone: `checkpoints/normwear_pretrain_ckpt.pth`.
- Optional official MSiTF module:
  `checkpoints/normwear_msitf_zeroshot_last_checkpoint-5.pth`.
- Provenance and hashes: `sources/method_manifest.yaml`.
- Frozen adapter decisions: `configs/alignment_v2.yaml` and
  `IDENTITY_AND_ADAPTATION_AUDIT.md`.
- Re-fetch helper: `scripts/fetch_weights.sh`.

All results must use the name `normwear_eeg_fnirs_adapted`. The adapter must
record EEG/HbO/HbR identities, native and target rates, CWT parameters, masks,
and channel aggregation. NormWear's upstream pretraining did not include
fNIRS.
