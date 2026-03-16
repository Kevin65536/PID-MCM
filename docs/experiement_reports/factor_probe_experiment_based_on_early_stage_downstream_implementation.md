# Phase A Report: Foundation Model Diagnostics

Date: 2026-03-16

Related plan:
- docs/EXPERIMENT_PLAN_V2_FOUNDATION.md
- docs/THEORY_V2_FOUNDATION_MODEL.md

## 1. Scope

This report summarizes Phase A diagnostic results for the current foundation-model mainline using the three latest downstream token runs:

1. EEG token classifier
2. fNIRS token classifier
3. Multimodal EEG+fNIRS token classifier

The goal was not to validate the final V2 architecture, but to diagnose whether the current representation stack already shows evidence of:

1. task-relevant separability,
2. reduced subject leakage,
3. useful multimodal complementarity.

## 2. Protocol

### 2.1 Downstream split

The current downstream protocol is strict cross-subject evaluation:

- train subjects: 1-20
- val subjects: 21-25
- test subjects: 26-29

Source:
- experiments/configs/downstream/base_downstream.yaml

### 2.2 Corrected factor probe design

The original subject probe was invalid under this protocol because it trained a subject classifier on train subjects and evaluated it on unseen subject IDs from val/test. That probe naturally collapsed to zero and could not be interpreted as subject invariance.

The corrected probe script now uses:

1. task probe on held-out val/test splits,
2. task probe within-train cross-validation,
3. subject probe within-train cross-validation,
4. explicit "incompatible" status for subject probes across unseen subject-ID spaces.

Source:
- experiments/scripts/run_factor_probes.py

Interpretation rule:

- higher task probe means stronger task information in representation,
- higher subject probe means stronger subject leakage in representation,
- desirable direction for V2 is: strong task probe, weak subject probe.

## 3. Run Summary

### 3.1 Main downstream results

| Run | Test Acc | Balanced Acc | Macro-F1 | Class Recall Gap | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| mi_eeg_token_20260316_114909 | 0.5125 | 0.5125 | 0.2148 | 0.7583 | severe one-class bias |
| mi_fnirs_token_20260316_120320 | 0.4875 | 0.4875 | 0.4675 | 0.0750 | near-chance but balanced |
| mi_multimodal_token_20260316_121015 | 0.5417 | 0.5417 | 0.5565 | 0.0667 | small gain over single modality |

Sources:
- experiments/runs/mi_eeg_token_20260316_114909/results.json
- experiments/runs/mi_fnirs_token_20260316_120320/results.json
- experiments/runs/mi_multimodal_token_20260316_121015/results.json

### 3.2 Corrected factor probe results

| Run | Task Probe Test | Task Probe Train-CV | Subject Probe Train-CV | Diagnostic |
| --- | ---: | ---: | ---: | --- |
| mi_eeg_token_20260316_114909 | 0.4792 | 0.4526 ± 0.0364 | 0.3801 ± 0.0201 | task weak, subject moderate |
| mi_fnirs_token_20260316_120320 | 0.5250 | 0.5042 ± 0.0190 | 0.4502 ± 0.0287 | task weak, subject moderate-high |
| mi_multimodal_token_20260316_121015 | 0.5083 | 0.5025 ± 0.0215 | 0.5591 ± 0.0288 | subject stronger than task |

Sources:
- experiments/runs/mi_eeg_token_20260316_114909/probes/factor_probe_report.json
- experiments/runs/mi_fnirs_token_20260316_120320/probes/factor_probe_report.json
- experiments/runs/mi_multimodal_token_20260316_121015/probes/factor_probe_report.json

## 4. Key Findings

### Finding 1: Current representations do not contain strong task factors

Across all three runs, task probes remain close to chance:

- EEG test probe: 0.4792
- fNIRS test probe: 0.5250
- multimodal test probe: 0.5083

This means the exported latent representations are only weakly linearly separable for the motor imagery task. The issue is not only the classifier head. The underlying token-based representation itself is not strongly task-aligned.

Implication:

- The current stack has not yet realized the V2 requirement that Zt should be predictively strong for downstream task decoding.

### Finding 2: Multimodal fusion currently amplifies subject information more than task information

The most important Phase A signal is:

- multimodal task probe train-CV: 0.5025
- multimodal subject probe train-CV: 0.5591

This means multimodal fused features are more linearly predictive of subject identity than of task label.

Implication:

- The current multimodal pathway is likely learning stable subject/style signatures across EEG and fNIRS rather than extracting a task-dominant shared physiological factor.
- This is directly contrary to the V2 goal of separating subject factors from task factors.

### Finding 3: EEG branch is especially unstable and collapses toward class bias

The EEG run shows:

- test macro-F1: 0.2148
- confusion matrix: [[107, 13], [104, 16]]
- positive recall: 0.1333
- class recall gap: 0.7583

This is not merely low accuracy. It indicates heavy decision bias toward one class and poor robustness under held-out subjects.

Implication:

- EEG token representations, combined with the current downstream aggregation head, are not reliably capturing cross-subject MI structure.
- The current frozen-tokenizer setup is insufficient as a foundation-model mainline.

### Finding 4: fNIRS is more balanced but still not strongly discriminative

The fNIRS run is more class-balanced than EEG, but the absolute task signal remains weak:

- test accuracy: 0.4875
- task probe test: 0.5250
- subject probe train-CV: 0.4502

Implication:

- fNIRS contributes smoother but still weak task information.
- It is not enough by itself to carry the mainline foundation objective.

### Finding 5: Multimodal gives only a small downstream gain despite much richer input

The multimodal run improves test accuracy to 0.5417, but this gain is small relative to the increase in representation complexity and the growth in subject leakage.

Implication:

- Current fusion is not yet delivering a high-value shared task representation.
- The gain looks more like shallow complementarity than successful factorized multimodal learning.

## 5. What These Results Say About the Current Design

### 5.1 The current mainline is still a baseline family, not yet the V2 foundation architecture

At the moment, the mainline is effectively:

1. frozen unimodal tokenizers,
2. per-modality lead aggregation,
3. shallow downstream fusion/classification.

What is still missing from V2:

1. explicit subject branch,
2. explicit task branch,
3. shared-factor branch,
4. adversarial subject invariance,
5. multi-scale temporal cross-modal fusion,
6. factor-level regularization.

So these results should be read as a diagnostic baseline against which Phase B must improve, not as a test of the full V2 theory.

### 5.2 The main bottleneck is representational objective mismatch

The current tokenizers were not trained with the V2 objective of preserving task-discriminative but subject-invariant information. The observed behavior is therefore expected:

- task signal weak,
- subject signal persistent,
- multimodal fusion drifts toward stable subject/style cues.

### 5.3 The current multimodal fusion is too shallow for the problem structure

Current multimodal processing compresses each modality into one feature vector before final fusion. That discards most temporal interaction structure.

For EEG-fNIRS, where temporal scales are mismatched and hemodynamic delay matters, this is too aggressive.

Implication:

- The current fusion design cannot test the V2 multi-scale alignment hypothesis.

## 6. Gate Assessment

### Gate G1 status

Gate G1 from the V2 plan asks whether Phase B can show robust improvement over the current baseline.

Current judgment:

- Phase A diagnostic completed.
- Baseline is now sufficiently characterized.
- No evidence yet that current representations satisfy the V2 task/invariance objectives.

Decision:

- Do not scale architecture width/depth yet.
- Move next to minimal Phase B implementation with explicit invariance and factor separation.

## 7. Required Next Changes

### Priority 1: Implement the minimal subject-invariance branch

Add:

1. task pathway,
2. subject discriminator,
3. gradient reversal,
4. joint logging of task and subject probe behavior.

Why:

- Current multimodal features show subject leakage stronger than task separability.

### Priority 2: Keep probe diagnostics mandatory in every mainline run

The corrected probe should remain part of the standard reporting contract for all future foundation runs.

Why:

- Accuracy alone would have hidden the key failure mode that multimodal features are still more predictive of subject than task.

### Priority 3: Replace one-vector fusion with at least a lightweight temporal fusion module

Before attempting the full V2 multi-scale block, add a minimal temporal fusion variant that preserves token-level or short-segment-level interactions across modalities.

Why:

- Current one-vector compression likely removes the exact cross-modal timing structure we need to test.

## 8. Final Conclusion

Phase A shows that the current mainline has two real design problems:

1. task information in the learned representations is weak,
2. multimodal representations still retain substantial subject information.

The strongest diagnostic signal is that the latest multimodal representation is more predictive of subject identity within the training population than of the downstream task. That is the clearest evidence so far that the current architecture is not yet implementing the intended V2 factorization.

Therefore, the correct next move is not broader experimentation on the current shallow stack, but immediate transition to Phase B minimal invariance architecture.