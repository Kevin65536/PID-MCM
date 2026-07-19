# STA-Net feasibility source record

Lookup date: 2026-07-18 (Asia/Shanghai)

## Queries

- `"STA-Net: Spatial-temporal alignment network" 103023 PDF`
- `site:github.com/MutianLiu-SHU/STA-Net STA-Net official`
- `doi 10.1016/j.inffus.2025.103023`
- targeted searches for subject-specific evaluation, task names, 3-second windows, and 16-by-16 inputs

## Primary sources

1. Liu, M. et al. (2025), *STA-Net: Spatial-temporal alignment network for hybrid EEG-fNIRS decoding*, Information Fusion 119, 103023. DOI: https://doi.org/10.1016/j.inffus.2025.103023
   - Publisher page: https://www.sciencedirect.com/science/article/pii/S156625352500096X
   - The publisher abstract and section preview identify STA-Net as an end-to-end paired EEG-fNIRS classifier with fNIRS-guided spatial alignment and EEG-guided temporal alignment.
   - Reported tasks are binary MI, MA, and WG, evaluated subject-specifically on two public datasets.
   - Reported mean accuracies are 69.65% (MI), 85.14% (MA), and 79.03% (WG).
   - The preview states that accuracy and Kappa are used because the source tasks are balanced binary classifications.

2. Official implementation: https://github.com/MutianLiu-SHU/STA-Net
   - Upstream `main` resolved to `b6db8bb5eb2f6491a13f0938880ee70e32162ee7` on 2026-07-18, matching the local nested checkout.
   - GitHub repository metadata reported no detected license, `pushed_at=2025-03-13T03:27:12Z`, and a public, non-archived repository.
   - The README requires Python 3.9.7 and TensorFlow 2.10.
   - The released runner is a per-subject, three-session binary classifier over pre-generated NPZ tensors; it does not provide a unified-loader adapter, multiclass head, continuous regression head, mask-aware loss, shared subject split support, or common result-artifact writer.

## Audit boundary

The publisher full text is access-controlled in this environment. Claims about exact tensor construction and training behavior were therefore checked against the official source tree at the matching upstream revision, while paper-level claims were limited to the publisher abstract and visible section previews. No secondary source was used to infer unexposed experimental details.

## Checkout-grounded adapter probe

- A current `simultaneous_eeg_nirs:wg` sample from `UnifiedPhysiologyWindowDataset` has the exact source channel inventory used by the released preprocessing path: 28 EEG channels and 36 fNIRS locations with HbO/HbR components.
- An in-memory projection using the official 16-by-16 coordinate templates produced finite tensors with the exact model signatures: EEG `[16,16,600,1]` and fNIRS `[11,16,16,30,2]`.
- This proves WG tensor adaptation is structurally feasible. It does not prove training fidelity, split validity, mask consumption, or scientific performance.
