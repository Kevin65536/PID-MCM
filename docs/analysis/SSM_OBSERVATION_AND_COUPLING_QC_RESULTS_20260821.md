# SSM observation and LC-SPVQ coupling QC results (2026-08-21)

## Scope and evidence boundary

This document preserves the compact, aggregate results from the architecture/objective QC run. The work is exploratory fit-parameter/fit-selection analysis, not an independent protected-test confirmation. Protected cohorts remained closed. The historical LC-SPVQ checkpoint audit is a zero-retraining diagnostic of the v2 same-token-time checkpoint contract; future endpoint-aligned training uses the v3 contract and has not yet been rerun. The continuous SSM screen used no vector quantizer. Because the continuous gate failed, K16 and q0/q1 analyses were not run.

Local source artifacts (kept outside Git by the experiment-output policy):

- `experiments/runs/physiology_semantic_tokenizer/coupling_calibration_audit/20260821_existing_checkpoint_controls_v2/`
- `experiments/runs/physiology_semantic_tokenizer/ssm_observation_target_screen/20260821_ssm_observation_screen_pilot_seed1_v4/`
- `experiments/runs/physiology_semantic_tokenizer/ssm_observation_target_screen/20260821_ssm_observation_screen_pilot_seeds23_v1/`
- `experiments/runs/physiology_semantic_tokenizer/ssm_observation_target_screen/20260821_ssm_observation_screen_multiseed_32step_v2/`
- `experiments/runs/physiology_semantic_tokenizer/ssm_observation_target_screen/20260821_mi_ssm_self_budget_probe_v1/`

The source manifests in those directories contain exact input and code hashes. This compact report contains no sample-level predictions, subject identifiers, checkpoints, or protected data.

## Historical LC-SPVQ checkpoint controls

Candidate: `lag05_h23`. Calibration was fitted on fit-parameter only. Shuffling used 200 fixed-seed within-subject/condition row permutations. Development rows are frozen descriptive reuse, not an independent holdout.

### Fit-selection macro-F1 deltas versus private-only

| Task | Train-only class bias | Temperature/intercept | Bias-only head | Interaction-only | Shared marginal | Shared marginal + interaction | Historical combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Motor imagery | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.1116 | -0.1116 | 0.0000 |
| Word generation | -0.0179 | -0.0179 | -0.0194 | 0.0000 | -0.1041 | -0.1041 | -0.0194 |

Fit-selection private-only macro-F1 was 0.7273 for motor imagery and 0.6420 for word generation. The 200-shuffle coupling distributions were:

- Motor imagery: mean 0.7244, SD 0.0070, range 0.7074–0.7273.
- Word generation: 0.6225 for every shuffle, identical to the historical combined result.

The fit-selection sample-level pooled, class-centered interaction logit variance was approximately `1.54e-6` for motor imagery and `1.69e-6` for word generation. These values are scale-dependent diagnostics, not effect sizes.

### Position-only control

For nonzero lags 1–5, the corrected endpoint-aligned mask showed no position-only loss reduction. Under the historical same-position mask, optimization reduced losses of approximately 2.29–3.51 to approximately 0.00012–0.00026. This demonstrates that the historical null admits a position shortcut; it does not prove that the full historical model used only that shortcut.

### Coupling interpretation

The historical selected checkpoint does not provide stable evidence that sample-dependent lag interaction improves fit-selection classification. Motor-imagery performance was already explained by private logits, while the word-generation historical combined head was worse than private-only. The old checkpoint remains a v2 same-token-time result and must not be relabelled as v3 endpoint-aligned evidence.

## Deterministic three-seed continuous SSM screen

Seeds: 20260821, 20260822, 20260823. Each cell used 32 representation and 32 task-head optimizer steps. Values below are fit-selection mean ± sample SD across seeds. Delta-R² is relative to a fit-parameter condition-time mean baseline.

### Representation endpoints

| Task | Mode | EEG delta-R² | fNIRS delta-R² | EEG passed every seed | fNIRS passed every seed |
| --- | --- | ---: | ---: | --- | --- |
| Motor imagery | NATIVE | -0.0554 ± 0.0147 | 0.4737 ± 0.0245 | No | Yes |
| Motor imagery | SSM-SELF | -0.0701 ± 0.0139 | 0.4737 ± 0.0245 | No | Yes |
| Motor imagery | SSM-JOINT | -0.0702 ± 0.0137 | 0.4737 ± 0.0245 | No | Yes |
| Motor imagery | SSM-SELF-XPRED-0.02 | -0.0699 ± 0.0135 | 0.4737 ± 0.0245 | No | Yes |
| Motor imagery | SSM-SELF-XPRED-0.05 | -0.0698 ± 0.0129 | 0.4737 ± 0.0245 | No | Yes |
| Word generation | NATIVE | 0.0347 ± 0.0054 | 0.4252 ± 0.0163 | Yes | Yes |
| Word generation | SSM-SELF | 0.0286 ± 0.0104 | 0.4251 ± 0.0163 | Yes | Yes |
| Word generation | SSM-JOINT | 0.0288 ± 0.0104 | 0.4252 ± 0.0163 | Yes | Yes |
| Word generation | SSM-SELF-XPRED-0.02 | 0.0286 ± 0.0099 | 0.4251 ± 0.0163 | Yes | Yes |
| Word generation | SSM-SELF-XPRED-0.05 | 0.0285 ± 0.0093 | 0.4251 ± 0.0163 | Yes | Yes |

A one-seed 128-step motor-imagery SSM-SELF budget probe did not repair the EEG failure: EEG delta-R² was -0.2610 while fNIRS delta-R² was 0.8505. This is descriptive evidence of objective/endpoint imbalance, not a multi-seed full-budget estimate.

### Downstream decomposition

| Task | Mode | Private macro-F1 | Private + shared marginal | Private + shared marginal + interaction | Interaction increment |
| --- | --- | ---: | ---: | ---: | ---: |
| Motor imagery | SSM-SELF | 0.5267 | 0.4979 | 0.5096 | 0.0117 ± 0.0109 |
| Word generation | SSM-SELF | 0.4883 | 0.5117 | 0.5195 | 0.0078 ± 0.0135 |
| Motor imagery | SSM-SELF-XPRED-0.02 | 0.5430 | 0.4789 | 0.4668 | -0.0121 ± 0.0107 |
| Motor imagery | SSM-SELF-XPRED-0.05 | 0.5430 | 0.4824 | 0.4702 | -0.0122 ± 0.0403 |

The interaction increments are small and unstable relative to their across-seed variation. They are not a proper-score coupling endpoint.

### Leakage control and stage decision

A classifier using teacher uncertainty, missingness, and constant provenance only had balanced accuracy 0.5 in every task/mode control row.

The fail-closed decision was:

```text
advance_to_independent_k16_vq = false
q0_q1_status = deferred
```

No non-privileged SSM mode achieved positive EEG and fNIRS delta-R² in every task and seed. Therefore no K16 stage or q0/q1 coupling claim is supported.

## What worked

- Endpoint-aligned lag masks, reverse-mask transposition, and v2/v3 contract separation.
- Explicit separation of private, shared marginal, bias, and class-centered interaction estimands.
- Fit-only calibration/shuffle controls and position-only sensitivity analysis.
- Label-free, fit-parameter-only modality SSM fitting with predictive uncertainty and missing-value-aware smoothing.
- Structurally private-only residual decoders and a genuinely no-VQ continuous screen.
- Strong positive fNIRS trajectory reconstruction relative to the condition-time baseline.
- Provenance/missingness leakage control and fail-closed K16 gating.

## What remains unresolved or failed

- Motor-imagery EEG reconstruction failed the representation gate in every seed and worsened in the longer one-seed probe.
- SSM-SELF did not outperform NATIVE; SSM-JOINT did not provide a useful empirical upper-bound gain.
- The low-weight causal FIR XPRED auxiliary objective did not repair the EEG failure or provide stable downstream gains.
- Shared marginal and interaction terms were not consistently beneficial across tasks.
- Corrected v3 LC-SPVQ masks have not been used to retrain M1/N1 checkpoints.
- The private residual is defined in observation-feature space, not raw high-rate EEG voltage space.
- Adaptive fNIRS-only reliability code now has v2 modality provenance and model-specific masks, but the v2 reliability experiment has not been rerun; no new fNIRS-only reliability numbers are claimed here.

## Scientific conclusion

The round validates the measurement, provenance, ablation, and stage-gating architecture more strongly than it validates a new physiological coupling mechanism. The fNIRS observation branch is learnable, but the EEG target—especially for motor imagery—remains the blocking failure. Current results do not support SSM superiority, a unique shared physiological source, causal EEG→fNIRS prediction, trial-specific lag interaction, M1-vs-N1 superiority, or a K16/q0-q1 coupling claim.
