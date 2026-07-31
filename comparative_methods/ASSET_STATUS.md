# Comparison method source and weight status

_B0 asset snapshot: 2026-07-31 (Asia/Shanghai). This is an inventory, not an
authorization to open protected evaluation or start a formal run._

The fixed comparison set contains three EEG-only models and four paired
EEG-fNIRS methods. Upstream code is kept in local, ignored checkouts; tracked
method manifests preserve the exact revision, license finding, checkpoint
provenance, pretraining-corpus identity, and target-corpus overlap decision.

| Method | Track | Code reference | Weight state | B0 boundary |
| --- | --- | --- | --- | --- |
| BIOT | EEG-only official-pretrained linear probe | Official checkout pinned at `d138e326` | Three official EEG checkpoints present in the checkout and hash-verified | Declared corpora do not match the four target datasets |
| CBraMod | EEG-only official-pretrained linear probe | Official checkout pinned at `0ff6be91` | Official Hugging Face checkpoint downloaded and hash-verified | TUEG does not match the four target datasets |
| REVE | EEG-only open-world pretrained linear probe | Official checkout pinned at `06a7059a` | Position bank plus base/large encoders downloaded and hash-verified; encoder use remains subject to the accepted agreement | `Shin2017A` is the Single-Trial corpus, so that cell is an overlap track |
| NormWear | `normwear_eeg_fnirs_adapted` linear probe | Official checkout pinned at `07517fcb` | Backbone and optional MSiTF release assets downloaded and hash-verified | No named target-corpus overlap; fNIRS was not an upstream pretraining modality |
| EFRM | Target-dataset-excluded multimodal representation | Existing independent PyTorch workspace; official source pinned at `a62bf3d4` | No official checkpoint; project checkpoints are trained from random initialization | Upstream has no license file; do not redistribute its source |
| BrainFusion NVC-CSP Stacking | Fold-local supervised traditional fusion | Official BrainFusion checkout pinned at `1d9dcf40` | Not applicable; estimators train inside each outer fold | Official code exposes NVC, but not the complete paper-case CSP/stacking execution |
| STA-Net | Strict cross-subject supervised fusion | Existing independent PyTorch workspace; official source pinned at `b6db8bb5` | Not applicable; fold checkpoints are project-trained artifacts | Upstream has no detected license; existing formal result remains frozen |

## Directory contract

Each new method uses:

```text
comparative_methods/<method>/
├── adapters/       # B1 input-contract work
├── checkpoints/    # local binary assets; ignored except README
├── configs/        # B4 frozen configs when admitted
├── runs/           # local method-owned artifacts; ignored
├── scripts/        # reproducible asset acquisition
├── sources/        # B0 manifest and source-fidelity record
├── tests/          # B2/B3 smoke and fidelity checks
└── upstream/       # local pinned checkout; ignored
```

EFRM and STA-Net retain their existing `*-PyTorch` integration roots and
separate local upstream checkouts. They are not renamed or rerun by this asset
preparation step.

## Reproduction boundary

- No new method has passed B1-B4 merely because its source and weights are
  present.
- PyTorch pickle checkpoints are only hashed here; they are not deserialized
  during acquisition or audit.
- REVE encoder weights must not be mirrored or redistributed. Local use remains
  subject to the model repository's Responsible Use Agreement.
- BrainFusion must be reported as an independent reimplementation unless the
  authors publish the complete CSP and ensemble execution used for the paper
  case study.
- NormWear remains an EEG-fNIRS adaptation. Its upstream paper and checkpoint
  do not establish native fNIRS support or an original-paper fNIRS result.

Run the non-deserializing local audit with:

```bash
.venv/bin/python comparative_methods/audit_assets.py
```
