# Croce 2017 Real-Data Validation Plan

> Created: 2026-05-18
> Status: Active validation plan for Phase 2B physical-model adoption on EEG+NIRS Single-Trial
> Scope: Decide whether the current Croce-style physical model should be kept as the preferred real-data analysis model for EEG+NIRS Single-Trial motor imagery, downgraded to an auxiliary analysis path, or replaced by a different approximation.
> Workspace root: [croce_validation/README.md](../croce_validation/README.md)

---

## 1. Decision Framing

This validation plan answers a narrower and more defensible question than the original Croce paper.

The EEG+NIRS Single-Trial dataset does **not** provide:

1. subject-specific EEG lead fields,
2. optical Jacobians or full photon transport models,
3. invasive or simulator-grade ground-truth neural source trajectories.

Therefore the project should **not** try to validate the claim “we recover the true raw neural activity”.

The strongest defensible claim on this dataset is:

1. we recover a **reproducible latent neural driver** that is constrained by simultaneous EEG and fNIRS,
2. the driver generates **physiologically plausible denoised observation components** through explicit forward approximations,
3. the resulting latent-state and denoised-observation estimates support a defensible analysis of motor-imagery physiology on real data.

If the model fails these tests, the correct fallback is not rhetorical softening. The correct fallback is to demote the model to an auxiliary analysis tool, a descriptive heuristic, or to replace it with a different approximation family.

---

## 2. Current Status From Docs, Git, and Existing Artifacts

Current repository status is more mature than “idea only”, but still short of a formal real-data decision.

### 2.1 What is already implemented

1. Phase 2B architecture is declared stabilized in [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
2. Croce-style target construction is implemented in [src/tokenizers/factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py): shared neural state, fNIRS HRF target, EEG signed-RMS carrier target.
3. A general SMC inference module exists in [src/inference/neurovascular_smc.py](../src/inference/neurovascular_smc.py).
4. Spatial geometry utilities exist in [src/data/channel_adjacency.py](../src/data/channel_adjacency.py).
5. Three real-data diagnostic entry points already exist:
   - [experiments/scripts/run_croce2017_smc_analysis.py](../experiments/scripts/run_croce2017_smc_analysis.py)
   - [experiments/scripts/validate_croce2017_smc.py](../experiments/scripts/validate_croce2017_smc.py)
   - [experiments/scripts/signal_visualization/analyze_croce_pf_reconstruction.py](../experiments/scripts/signal_visualization/analyze_croce_pf_reconstruction.py)

### 2.2 What git history says about project progress

The recent commit chain shows the project moved from architecture work into physical-model integration, then stopped before a full real-data decision loop was closed:

1. `0635641`: neuro forward model investigated and tested
2. `5b4b77f`: source target and coupling loss changed
3. `754a86c`: full-data implementation of the source-target shift
4. `5c7e111`: documents re-aligned to current progress

This is consistent with the project-level log in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md): architecture is marked complete, while the next focus is Gate 3 validation, diagnostic refinement, and downstream evaluation.

### 2.3 What is already evidenced, and what is still missing

| Question | Current evidence | Gap that still blocks adoption |
|----------|------------------|--------------------------------|
| Is preprocessing aligned with the mainline pipeline? | Yes. Existing validation scripts explicitly use the Phase 1 default preprocessing. | No gap here. Keep fixed. |
| Can a Croce-style model reconstruct slow fNIRS structure on real data? | Partially yes. The reduced PF summary shows strong low/high wavelength fit on event averages. | Need held-out, subject-level, and null-controlled comparison across approximation families. |
| Can the current model reconstruct EEG faithfully? | Weakly. Existing summaries show low EEG correlation and low explained variance for the current exploratory setup. | Need to redefine the target as slow denoised neural driver, not full waveform recovery, and test competing approximations. |
| Does the recovered state carry task-relevant signal? | Not demonstrated. Current SMC validation uses 20 trials from one subject and reports non-significant label separation. | Need cross-subject, cross-session, event-locked, and null-controlled task tests. |
| Is the inverse solution stable on real data? | Not established. Current scripts do not provide a full reproducibility sweep across seeds, particles, or parameter perturbations. | Need identifiability and sensitivity experiments before adoption. |
| Can the model support a defensible project-level real-data analysis path? | Not established. Existing evidence is still exploratory and split coverage is incomplete. | Need a held-out comparison across approximation families, nulls, and stability tests. |

---

## 3. Validation Questions and Decision Outcomes

The validation program must answer three primary questions.

### Q1. Forward-model compromise

Given missing lead fields and optical Jacobians, which approximation best preserves spatial information while remaining physiologically constrained?

### Q2. Real-data inverse adequacy

Under realistic noise and limited observability, is the recovered latent state reproducible, identifiable enough, and better than non-physiological controls?

### Q3. Practical utility on motor-imagery real data

Does the selected approximation support stable, interpretable motor-imagery analysis on held-out subjects?

### Allowed end states

| Outcome | Interpretation | Mainline action |
|---------|----------------|-----------------|
| **Adopt** | Forward approximation beats controls, inverse solution is stable enough, and held-out motor-imagery analysis remains interpretable. | Promote the selected approximation as the default Croce real-data analysis path. |
| **Auxiliary only** | Physiological plausibility is acceptable, but inverse stability or task utility is weak. | Keep the model only as a supporting analysis tool. |
| **Reject / replace** | Model cannot beat simple controls or remains unstable under small perturbations. | Freeze current exploratory path and move to another approximation family. |

---

## 4. Data Protocol

### 4.1 Primary dataset and task

Primary validation should use the current mainline dataset and task only:

1. dataset: EEG+NIRS Single-Trial,
2. task: motor imagery,
3. split: the current project train/val/test subject split for the motor-imagery pipeline.

No cross-task extension is in scope for this plan.

### 4.2 Split policy

Use the existing subject split from [experiments/configs/source_observation/phase1/default.yaml](../experiments/configs/source_observation/phase1/default.yaml):

1. train subjects: 1-20,
2. validation subjects: 21-25,
3. test subjects: 26-29.

Rules:

1. choose approximation family and hyperparameters on train,
2. freeze the choice on validation,
3. report the final decision only on test.

### 4.3 Unit of analysis

To avoid pseudo-replication, all primary metrics must be aggregated in this order:

1. trial or continuous segment,
2. session,
3. subject,
4. split summary.

The subject is the primary statistical unit for go/no-go decisions.

### 4.4 Time views

Every shortlisted approximation must be tested on two complementary views.

1. **Continuous session view**: stability, reconstruction, lag structure, ESS, uncertainty.
2. **Event-locked view**: cue-aligned dynamics, condition separation, laterality, event-average physiology.

All formal conclusions must be drawn from the motor-imagery task only.

### 4.5 Preprocessing policy

Primary validation must keep the exact preprocessing used by the current project motor-imagery data path:

1. EEG resample rate 200 Hz,
2. EEG bandpass 0.5-45 Hz,
3. fNIRS lowpass 0.2 Hz,
4. exclude EOG,
5. HbO-only path for the primary report.

Secondary diagnostic sweeps may use raw lowWL/highWL or optical-density style signals, but these are supporting analyses only. They do not replace the primary report because HbO-only is the current primary analysis path for this validation program.

---

## 5. Approximation Families To Compare

All comparisons below must keep the same train/validation protocol and the same reporting surface. Only the forward approximation changes.

| ID | Approximation family | Spatial information | Why it must be tested |
|----|----------------------|--------------------|-----------------------|
| **F0** | Global non-spatial control: channel-mean EEG power or signed-RMS proxy driving broadcast HRF | None | Minimum baseline. If a spatial model cannot beat this, it has no reason to stay. |
| **F1** | PC1 scalar-state SMC baseline | Very weak | Matches the current exploratory SMC setup and quantifies the value of stronger spatial priors. |
| **F2** | ROI-anchored Gaussian sensor maps | Moderate | Matches the current reduced PF compromise already present in the repo. |
| **F3** | Adjacency-weighted local driver using EEG-fNIRS geometry | Strong, fixed | Uses [src/data/channel_adjacency.py](../src/data/channel_adjacency.py) directly and is the cleanest non-learned spatial compromise. |
| **F4** | Sparse learned forward maps with geometry regularization | Strong, adaptive | Tests whether fixed geometry is too rigid while still banning unconstrained black-box fitting. |

Shared rules across all families:

1. keep the Croce dynamic core and HRF structure fixed unless the experiment explicitly studies that factor,
2. compare equal state dimensionalities when possible,
3. log exact observation matrices and geometry settings for every run.

---

## 6. Nulls and Non-Physiological Controls

Every serious real-data inverse claim must beat explicit nulls.

| ID | Null / control | Purpose |
|----|----------------|---------|
| **N0** | Pure temporal smoothing without cross-modal constraint | Tests whether Croce adds value beyond low-pass denoising. |
| **N1** | Time-shifted EEG-to-fNIRS pairing | Tests whether fit survives when physiological timing is broken. |
| **N2** | Phase-randomized or block-shuffled EEG power | Tests whether fit depends on real temporal structure. |
| **N3** | Spatially permuted EEG channels or fNIRS channels | Tests whether spatial mapping matters. |
| **N4** | Cross-subject mismatched EEG-fNIRS pairing | Tests whether the model is exploiting generic slow trends rather than paired physiology. |
| **N5** | Unconstrained linear regression from EEG proxy to fNIRS | Tests whether the Croce prior beats a flexible but non-physiological predictor. |

The selected approximation must beat at least N0, N1, N3, and N5 on the primary evaluation metrics. Otherwise the physiological narrative is not earned.

---

## 7. Experiment Packages

### EXP-CROCE-V0: Reproduce Existing Evidence

**Objective**: consolidate all currently scattered results into a single reproducible baseline table.

**Inputs**:

1. [experiments/scripts/run_croce2017_smc_analysis.py](../experiments/scripts/run_croce2017_smc_analysis.py)
2. [experiments/scripts/validate_croce2017_smc.py](../experiments/scripts/validate_croce2017_smc.py)
3. [experiments/scripts/signal_visualization/analyze_croce_pf_reconstruction.py](../experiments/scripts/signal_visualization/analyze_croce_pf_reconstruction.py)

**Outputs**:

1. unified summary table of existing metrics,
2. exact subject/session/task coverage,
3. gap annotation showing which metrics are exploratory only.

**Pass condition**:

Reproduced metrics must match the currently archived artifacts closely enough to serve as the baseline reference for all later sweeps.

### EXP-CROCE-V1: Forward Approximation Sweep

**Objective**: choose the best forward approximation under the current data limitations.

**Factors**:

1. approximation family: F0-F4,
2. state dimension: 1, 2, 4,
3. shared-state smoothing alpha: 0.85, 0.90, 0.95,
4. geometry scale or neighbor count for spatial approximations,
5. default HRF versus one mild tuned variant.

**Primary metrics**:

1. held-out fNIRS channel-wise correlation,
2. held-out fNIRS nRMSE,
3. joint log-likelihood on held-out segments,
4. leave-one-channel-out prediction accuracy,
5. lag of peak state to fNIRS correlation,
6. condition-specific laterality consistency in motor ROI summaries.

**Decision rule**:

Select the simplest approximation that clearly beats F0 and N5 on validation subjects. If two families are statistically tied, keep the more constrained one.

### EXP-CROCE-V2: Inverse Stability and Identifiability

**Objective**: determine whether the inferred state is stable enough to justify real-data use.

**Factors**:

1. random seeds: at least 5,
2. particles: 200, 500, 1000,
3. initialization perturbations,
4. plus or minus 10 percent perturbation of alpha, HRF parameters, and noise covariances.

**Primary metrics**:

1. within-subject state correlation across seeds,
2. within-subject ICC across seeds,
3. ESS divided by particle count,
4. posterior uncertainty width,
5. sign consistency of task effects,
6. metric separation from N1-N4 nulls.

**Minimum bar for adoption**:

1. median within-subject state correlation across seeds at least 0.70,
2. mean ESS ratio at least 0.30,
3. qualitative conclusions unchanged under small parameter perturbations for most subjects,
4. selected approximation beats temporal and spatial nulls on the primary metrics.

If these criteria fail, the model may still be used as a soft regularizer, but not as a trusted source estimate.

### EXP-CROCE-V3: Event-Locked Physiological Plausibility

**Objective**: test whether the inferred state and forward predictions obey expected task timing and spatial structure.

**Protocol**:

1. cue-lock windows from -4 s to +16 s,
2. analyze left-hand and right-hand conditions separately,
3. compute both single-trial summaries and event averages,
4. retain session-wise results before subject averaging.

**Primary metrics**:

1. state onset relative to cue,
2. state to fNIRS peak delay,
3. peak-delay distribution across subjects,
4. condition-wise laterality index around contralateral motor anchors,
5. event-average fit quality versus reduced PF baseline,
6. optional wavelength-pair polarity checks when using the raw optical path.

**Minimum bar for adoption**:

1. the state to fNIRS peak should fall inside a physiologically plausible window for a clear majority of subjects,
2. laterality summaries should beat shuffled-channel controls,
3. event-average physiology should remain visible after denoising rather than being flattened away.

### EXP-CROCE-V4: Denoising Utility Versus Simple Proxies

**Objective**: test whether the Croce-style approximation gives more than a visually plausible slow signal.

**Comparators**:

1. signed-RMS carrier without cross-modal state inference,
2. low-pass envelope proxy,
3. unconstrained regression baseline,
4. selected Croce approximation.

**Primary metrics**:

1. task-label discriminability of denoised state summaries,
2. subject leakage of state summaries,
3. fNIRS reconstruction quality,
4. physiological delay plausibility,
5. spatial laterality preservation.

**Decision rule**:

If the selected Croce approximation does not beat simple smoothing on at least two independent metric families, it should not be promoted as the default denoising mechanism.

---

## 8. Metrics Dashboard

To make experiment reports comparable, every run should emit the same metric groups.

### 8.1 Fit metrics

1. fNIRS correlation, nRMSE, and log-likelihood,
2. EEG slow-feature correlation,
3. leave-one-channel-out prediction score,
4. event-average reconstruction score.

### 8.2 Stability metrics

1. state correlation across seeds,
2. ICC across seeds,
3. ESS ratio,
4. posterior entropy or interval width,
5. sensitivity to small parameter changes.

### 8.3 Physiological metrics

1. state to fNIRS peak delay,
2. laterality index,
3. ROI-specific fit,
4. preservation of event-related morphology,
5. null-gap against broken timing and broken geometry.

### 8.4 Task-utility metrics

1. task-label discriminability,
2. subject leakage,
3. split-wise reproducibility of condition effect sizes,
4. robustness of laterality summaries across sessions and subjects.

---

## 9. Reporting Standard

All new standardized code and outputs for this plan must live under [croce_validation/README.md](../croce_validation/README.md), not under `experiments/`.

Each experiment package must write the following artifacts under `croce_validation/results/<run_name>/`:

1. `run_manifest.json`: exact config, seeds, subject split, preprocessing,
2. `metrics.json`: split-level metrics,
3. `subject_level_metrics.csv`: one row per subject,
4. `plots/`: standard figure set,
5. `design_summary.md`: interpretation and decision.

Every summary must state all three items explicitly:

1. what approximation was tested,
2. what nulls were beaten or not beaten,
3. whether the result supports Adopt, Auxiliary only, or Reject / replace.

Legacy scripts and results already under `experiments/` remain valid historical references, but they should not be the destination for new standardized Croce validation work.

---

## 10. Go / No-Go Rule For Mainline Adoption

The Croce-style model may become the default source-target generator only if all of the following hold on held-out test subjects:

1. the selected approximation beats the non-spatial and non-physiological controls,
2. inverse-state estimates are stable enough across seeds and small parameter perturbations,
3. physiological timing and spatial structure remain visible,
4. motor-imagery condition effects remain reproducible on held-out subjects,
5. no primary metric family shows catastrophic failure relative to simple denoising controls.

If only the physiology checks pass but stability or task utility fails, the model should remain auxiliary.

If the model cannot beat simple controls or nulls, the current narrative must be narrowed to “physiology-inspired target construction experiment” and the default training path should revert to the best simpler approximation.

---

## 11. Immediate Next Actions

1. move new standardized Croce validation work into `croce_validation/`,
2. turn the three current exploratory Croce analysis scripts into one standardized runner with approximation-family flags,
3. run EXP-CROCE-V0 to freeze the current baseline,
4. run EXP-CROCE-V1 on train and validation subjects only,
5. run EXP-CROCE-V2 to decide whether the inverse solution is stable enough to keep.

This ordering matters. Without V1 and V2, any stronger conclusion about the physical model would still be premature because the forward approximation itself remains an uncontrolled variable.