# CBraMod comparison workspace

_Project-level execution and scientific verdicts are generated in
[`docs/PROJECT_STATUS.md`](../../docs/PROJECT_STATUS.md). Public-development and
data-boundary statements here describe the frozen implementation/evidence layer._

This isolated workspace prepared CBraMod for the EEG-only
official-pretrained linear-probe track. A0–A8 alignment and the 90-job
public-development matrix completed before the later joint campaign.

- Official code: `upstream/`, revision
  `0ff6be918985689e7df679bc731ffb70e6c6224f`.
- Official checkpoint: `checkpoints/pretrained_weights.pth`, pinned to the
  Hugging Face revision and SHA-256 in `sources/method_manifest.yaml`.
- Re-fetch helper: `scripts/fetch_weights.sh`.

The current upstream model assumes 200 Hz EEG and 200-sample patches. Input
adaptation must preserve measured channels and masks; fNIRS is outside this
method's comparison surface.

The frozen public-development snapshot is recorded in
`evidence/alignment_v2/summary_final.json` and
`evidence/public_development_v2/matrix_completion_summary.json`. Public
validation results alone are not table-admissible; final cell verdicts come from
the tracked joint-campaign result report.
