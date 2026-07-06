# Cross-dataset shared neural state diagnostic

_Two subjects per dataset; delayed innovation state; diagnostic only_

---

## Decision

At a preregistered five-second EEG-leading lag, the tested three-dimensional shared state did **not** expose cross-subject-stable information that could be inferred independently from EEG or fNIRS. In all four datasets, adding the other-modality CCA state made held-out prediction slightly worse than the self-history, trial-phase, and condition baseline. The conservative cross-inferable shared fraction is therefore `0%` for both modality innovation and total raw-data-derived feature variance.

A joint teacher that can see both modalities obtained a small optimistic compression ceiling. The balanced fraction was limited by the fNIRS side: `0.62%–3.97%` of innovation and `0.56%–3.21%` of total standardized feature variance. This joint-input ceiling is not evidence that either independent tokenizer can recover the state.

![Shared-information summary](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_173530_cross_dataset_shared_neural_state_v1/figures/shared_information_summary.svg)

## Estimand

One-second EEG log-bandpower features and fNIRS canonical-coordinate mean/slope features were first adapted per subject using a versioned median/MAD transform. For each modality, a ridge baseline predicted the current feature vector from three seconds of its own history, trial phase, and condition. The residual was treated as modality innovation.

At lag $\ell$, CCA was fitted to paired innovations:

\[
z_t^E=W_E e_t^E,\qquad z_{t+\ell}^F=W_F e_{t+\ell}^F.
\]

The primary directional fraction was:

\[
S_{E\rightarrow F}=1-\frac{\operatorname{SSE}(F_{base}+D_F(z^E))}{\operatorname{SSE}(F_{base})},
\]

with the reciprocal back-projection defined analogously. The conservative balanced estimate is the smaller non-negative directional fraction. A second estimate used $z^{joint}=(z^E+z^F)/2$ and is explicitly labeled an optimistic joint-teacher ceiling.

These are reconstruction fractions in standardized raw-data-derived feature space. They are not waveform information fractions or nonparametric mutual information.

## Data and split

| Dataset | Subjects | Paired units per subject | fNIRS representation |
| --- | --- | ---: | --- |
| EEG+NIRS Single-Trial | 1, 2 | 120 trials | 760/850 nm pair |
| REFED | 1, 2 | 15 videos | HbO/HbR |
| Simultaneous EEG&NIRS | 1, 2 | 60 word-generation trials | Oxy/Deoxy |
| Visual Cognitive Motivation | S01, S02 | 250 trials | two probes, Oxy/Deoxy |

Each dataset used two reciprocal folds: one subject fitted every model and the other subject was evaluated, then the roles were reversed. The physiology-semantic protected Single-Trial subjects 24–29 were not opened. REFED and Visual used experiment-local read adapters because their registered paired loaders remain planned.

## Primary five-second result

| Dataset | Cross-inferable EEG innovation | Cross-inferable fNIRS innovation | Balanced joint innovation ceiling | Balanced joint total-variance ceiling |
| --- | ---: | ---: | ---: | ---: |
| Single-Trial | `0.00%` | `0.00%` | `3.97%` | `3.21%` |
| REFED | `0.00%` | `0.00%` | `0.62%` | `0.56%` |
| Simultaneous | `0.00%` | `0.00%` | `1.63%` | `1.21%` |
| Visual motivation | `0.00%` | `0.00%` | `2.56%` | `2.49%` |

The unclipped directional cross-inference medians ranged from `−0.15%` to `−0.51%`. Fifteen of sixteen within-held-out-subject trial/video bootstrap intervals were wholly below zero; the remaining upper endpoint was `+0.059%`. Validation canonical correlations remained small and no dataset showed a reciprocal positive lag profile.

![Lag profiles](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_173530_cross_dataset_shared_neural_state_v1/figures/lag_profiles.svg)

## Interpretation

The joint ceiling shows that a low-dimensional joint bottleneck can compress a small amount of both-modality residual structure. It does not identify that structure as a shared neural cause because each joint state contains information from its target modality. The independent-tokenizer requirement is stricter: an EEG-only state must predict future fNIRS innovation and an fNIRS-only state must recover the corresponding earlier EEG innovation in a held-out subject. That requirement received no positive evidence here.

The result therefore:

1. supports retaining the dynamic shared-state hypothesis as a research hypothesis;
2. does not support using this CCA state as a physical teacher;
3. does not support calling residuals modality-private solely because a joint bottleneck was fitted;
4. leaves E0 blocked and the protected test closed.

With only two subjects per dataset, these are diagnostic cross-subject failures, not population estimates or proof that no neurovascular shared state exists. The next admissible attempt would need stronger subject-specific spatial/HRF adapters or repeated-subject calibration, while preserving independent single-modality evaluation.

## Artifacts

- Formal run: [`20260706_173530_cross_dataset_shared_neural_state_v1`](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_173530_cross_dataset_shared_neural_state_v1/)
- Configuration: [`cross_dataset_shared_neural_state.yaml`](../../experiments/configs/physiology_semantic_tokenizer/cross_dataset_shared_neural_state.yaml)
- Implementation: [`evaluate_cross_dataset_shared_neural_state.py`](../../experiments/evaluate_cross_dataset_shared_neural_state.py)
- Full lag metrics: [`lag_metrics.csv`](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_173530_cross_dataset_shared_neural_state_v1/lag_metrics.csv)
- Alignment nulls: [`alignment_null_metrics.csv`](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_173530_cross_dataset_shared_neural_state_v1/alignment_null_metrics.csv)
- Dataset inventory: [`dataset_inventory.csv`](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_173530_cross_dataset_shared_neural_state_v1/dataset_inventory.csv)

**Status:** diagnostic complete; cross-inferable shared information not detected; E0 remains blocked.
