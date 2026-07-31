# CBraMod comparison workspace

This isolated workspace prepares CBraMod for the EEG-only
official-pretrained linear-probe track. B0 source fixing and checkpoint
acquisition are complete; B1-B4 remain pending.

- Official code: `upstream/`, revision
  `0ff6be918985689e7df679bc731ffb70e6c6224f`.
- Official checkpoint: `checkpoints/pretrained_weights.pth`, pinned to the
  Hugging Face revision and SHA-256 in `sources/method_manifest.yaml`.
- Re-fetch helper: `scripts/fetch_weights.sh`.

The current upstream model assumes 200 Hz EEG and 200-sample patches. Input
adaptation must preserve measured channels and masks; fNIRS is outside this
method's comparison surface.
