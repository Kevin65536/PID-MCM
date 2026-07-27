# STA-Net five-fold benchmark: strict cross-subject vs sample-level random split

Values are mean ± sample SD across the five outer folds. Classification values are percentages; REFED CCC is unitless. Hyperparameters were frozen before opening any outer test fold.

All 70 fold trainings ended by the frozen validation-convergence rule rather than the maximum-epoch cap. Artifact masks were not consumed; only real record support from `valid_mask` was used.

## Primary endpoints

| Task | Metric | Strict cross-subject | Sample-level random split | Δ random − strict |
|---|---|---:|---:|---:|
| Motor imagery (MI) | Macro-F1 (%) | 56.40 ± 1.58 | 53.20 ± 3.53 | -3.20 |
| Mental arithmetic (MA) | Macro-F1 (%) | 62.84 ± 4.25 | 68.87 ± 2.17 | +6.03 |
| Word generation (WG) | Macro-F1 (%) | 62.11 ± 3.13 | 63.18 ± 1.13 | +1.07 |
| n-back | Macro-F1 (%) | 37.52 ± 2.32 | 37.07 ± 4.93 | -0.45 |
| DSR | Macro-F1 (%) | 60.69 ± 2.38 | 65.29 ± 1.42 | +4.60 |
| Visual | Macro-F1 (%) | 25.01 ± 0.77 | 27.74 ± 1.38 | +2.73 |
| REFED regression | CCC | 0.081 ± 0.048 | 0.126 ± 0.038 | +0.046 |

## Classification accuracy

| Task | Strict cross-subject Accuracy (%) | Sample-level random Accuracy (%) | Δ (pp) |
|---|---:|---:|---:|
| Motor imagery (MI) | 56.50 ± 1.57 | 53.33 ± 3.48 | -3.17 |
| Mental arithmetic (MA) | 63.32 ± 4.01 | 68.97 ± 2.17 | +5.64 |
| Word generation (WG) | 62.27 ± 2.99 | 63.27 ± 1.05 | +1.00 |
| n-back | 38.72 ± 2.20 | 38.18 ± 4.13 | -0.54 |
| DSR | 66.60 ± 2.23 | 71.61 ± 1.51 | +5.01 |
| Visual | 33.12 ± 1.52 | 36.22 ± 1.82 | +3.10 |

## Accuracy comparison with the original STA-Net paper

| Source task | Strict cross-subject (%) | Sample-level random (%) | Original paper (%) | Δ strict − paper (pp) | Δ random − paper (pp) |
|---|---:|---:|---:|---:|---:|
| Motor imagery (MI) | 56.50 ± 1.57 | 53.33 ± 3.48 | 69.65 ± 9.52 | -13.15 | -16.32 |
| Mental arithmetic (MA) | 63.32 ± 4.01 | 68.97 ± 2.17 | 85.14 ± 7.17 | -21.82 | -16.17 |
| Word generation (WG) | 62.27 ± 2.99 | 63.27 ± 1.05 | 79.03 ± 8.41 | -16.76 | -15.76 |

The original-paper column is contextual rather than a same-protocol statistical comparison: the paper reports subject-specific MI/MA/WG evaluation, whereas the two new columns use this project's unified preprocessing and outer five-fold definitions.

The sample-level split deliberately does not isolate subjects, recordings, trials, or overlapping-window dependency groups. It measures an information-visible upper-bound setting requested for this benchmark and must be labeled exactly as such.

n-back, DSR, Visual, and REFED are project adaptations not evaluated by the original STA-Net paper; no original-paper number is imputed for them.
