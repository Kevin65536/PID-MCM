# Physiology-semantic tokenizer experiment log

_Active run registry; historical runs retain their original contracts, new runs use the 2026-07-14 measurement-first revision_

---

## 📋 Current status

The complete tokenizer training loop is runnable, but the Croce E0-v2 target remains blocked at validation. The 2026-07-14 contract no longer treats that target as the architecture input or universal token semantics. All new experiments must use `UnifiedPhysiologyWindowDataset`; teacher-free objectives may proceed through their own gates, and each optional target family requires an independent scoped admission. Historical runs below retain their original loader/teacher contract and are not retroactively relabeled.

| Date | ID | Suite | Status | Result root |
| --- | --- | --- | --- | --- |
| 2026-07-01 | `PST-DESIGN-FREEZE` | Documentation | Complete | Not applicable |
| 2026-07-02 | `PST-P1-DRYRUN` | E0 contract dry-run | Passed; G0 not evaluated | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260702_191234_p1_contract_dry_run/` |
| 2026-07-02 | `PST-P1-SMOKE` | E0 contract smoke | Passed; G0 not evaluated | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260702_191234_p1_contract_smoke/` |
| 2026-07-02 | `PST-P2-P5-DRYRUN` | Migration integration | Passed; correctness only | `experiments/runs/physiology_semantic_tokenizer/software_smoke/20260702_235450_p2_p5_software_smoke/` |
| 2026-07-02 | `PST-P2-P5-SMOKE` | Migration software smoke and P5 export | Passed; optimizer blocked by E0; no gate evaluated | `experiments/runs/physiology_semantic_tokenizer/software_smoke/20260702_235459_p2_p5_software_smoke/` |
| 2026-07-03 | `PST-E0-PILOT-V1` | Teacher validity | Blocked on validation; protected test unopened | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260703_165153_e0_teacher_validity_pilot_v1/` |
| 2026-07-03 | `PST-E0-V2-VALIDATION` | Layered teacher information contract and visual audit | Blocked on physical observation and uncertainty calibration; protected test unopened | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260703_232754_e0_teacher_validity_v2/` |
| 2026-07-06 | `PST-E0-D1-SHARED-BOUND` | Croce-independent shared-state reconstruction bound | Diagnostic complete; supports shared + private redesign; E0 remains blocked | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_105937_shared_state_reconstruction_bound_v1/` |
| 2026-07-06 | `PST-E0-D2-CROSS-DATASET-SHARED` | Four-dataset delayed-innovation shared-state diagnostic | Diagnostic complete; cross-inferable fraction 0% in all datasets; E0 remains blocked | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_173530_cross_dataset_shared_neural_state_v1/` |
| 2026-07-08 | `PST-E0-D3-LIN2024-SUBJECT-HRF` | Lin 2024 inspired subject-specific NVC diagnostic | Diagnostic complete; subject-specific HRF path remains weak; E0 remains blocked | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_113000_lin2024_subject_specific_nvc_v1/` |
| 2026-07-08 | `PST-E0-D4-LIN2024-RAW-TRTD` | Lin 2024 raw continuous session TRTD diagnostic | Diagnostic complete; in-sample upper bound remains weak; E0 remains blocked | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_160736_lin2024_raw_session_trtd_s19_sess2/` |
| 2026-07-08 | `PST-E0-D5-LIN2024-SIM-RAW-TRTD` | Lin 2024 Simultaneous EEG&NIRS raw TRTD confirmation | Diagnostic complete; Simultaneous concentration data show same weak EEG-to-fNIRS recovery; E0 remains blocked | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_170809_lin2024_simultaneous_raw_trtd_vp001_wg/` |
| 2026-07-10 | `PST-P1-FOUR-DATASET-QUALITY` | Four-dataset unified loader and quality audit | Correctness passed; 8-second report is historical and does not imply artifact-clean data | `experiments/runs/physiology_semantic_tokenizer/data_quality_audit/final_four_dataset_check_20260710/` |
| 2026-07-14 | `PST-INPUT-CONTRACT-REVISION` | Architecture decision | Measurement-first entrance approved; all new E0-E9 runs require unified loader | Not applicable |
| 2026-07-15 | `PST-E0-D6-UNIFIED-SHARED-DRIVER` | Unified-loader Croce/Lin raw-vs-clean retest | Diagnostic complete; artifact correction does not resolve fNIRS amplitude/variance failure; target families remain blocked | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260715_shared_neural_driver_unified_formal_v3/` |
| 2026-07-16 | `PST-E0-D7-ADAPTIVE-SHARED-SSM` | Local adaptive five-state fixed-interval shared-driver test | Diagnostic complete; joint compromise restores HbO variance/cycles while retaining EEG fit; candidate soft teacher retained, formal admission pending | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_shared_neural_ssm_formal_v2/` |
| 2026-07-16 | `PST-E0-D8-TASK-PARAMETER-AUDIT` | Within-subject adaptive-SSM task-parameter audit | Diagnostic complete; no parameter survives FDR; nominal differences concentrate in driver/noise terms, not identified hemodynamic constants | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_ssm_task_parameter_audit_v1/` |
| 2026-07-03 | `PST-TRAIN-DRYRUN-V1` | Full trainer dry-run | Passed; no optimizer step | `experiments/runs/physiology_semantic_tokenizer/tokenizer_training/20260703_164728_physiology_semantic_tokenizer_pilot_v1/` |
| 2026-07-03 | `PST-E1-TF-SMOKE-V1` | Teacher-free reconstruction/VQ | Passed; CUDA, 2 optimizer steps | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260703_165220_tokenizer_reconstruction_baseline_pilot_v1/` |
| 2026-07-03 | `PST-E1-TF-RESUME-V1` | Teacher-free checkpoint resume | Passed; resumed to 4 optimizer steps | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260703_165236_tokenizer_reconstruction_baseline_pilot_v1/` |

### E0 validation decision

On the five validation subjects, EEG normalized predictive gain was `+0.756030` with a subject-bootstrap 95% interval of `[0.691380, 0.805454]`. fNIRS normalized predictive gain was `-1.503607` with interval `[-2.383524, -0.662615]`. Although all nine fNIRS state-summary coordinates exceeded their feature-permutation observability null, the teacher's posterior-predictive clean observation failed the declared history-baseline endpoint. This distinction prevents coordinate observability from being mistaken for physical-teacher validity.

### E0-v2 layered validation and visual review

E0-v2 evaluated 17,280 train and 4,800 validation local patches plus 8,640 train and 2,400 validation fixed-history rows. Four raw fNIRS datasets were audited without reading their protected partitions. Train-only full-record baseline/robust-scale adapters achieved exact crop-position invariance and a validation pooled-pair scale ratio of `1.463`; original voltage, concentration, and exported relative-signal semantics remain explicit.

Patch-local observability admitted EEG `r_mean`, `r_slope`, and `s_slope`, while excluding `s_mean`. All six fNIRS level/slope coordinates exceeded their label-permutation references, with validation R² from `0.046` to `0.258`. The admitted target geometry remained transmissible through 128 prototypes: global standardized reconstruction R² was `0.918` for EEG and `0.949` for fNIRS, with perplexities `96.8` and `107.6`. The continuous upper bound was positive beyond shuffled EEG history for both fNIRS levels (`0.177` nats) and innovations (`0.172` nats); gains concentrated in `delta_f` and `delta_hbo`, while `delta_hb` added approximately zero.

Two independent layers failed. Corrected wavelength-space fNIRS clean prediction had MSE `2.193` versus the selected history baseline `0.834`, a subject-mean gain of `-1.359` with `0/5` positive subjects. Synthetic-truth variance scaling improved posterior coverage but `delta_f`, `delta_hbo`, and `delta_hb` remained outside the sample-size-derived 95% band; real-data student errors were also much larger than teacher posterior SD. The visual review confirmed the fNIRS baseline/amplitude mismatch and uncertainty under-coverage, so its overall status is `fail`. Protected-test eligibility is `false`.

Every numerical layer has a corresponding replayable visual artifact with source JSON, SVG, 300 dpi PNG, hashes, and review notes in the run root. The definitive decision files are `summary.json`, `visual_audit_manifest.json`, `visual_review.json`, and `target_contract.json`.

### Shared-state reconstruction-bound diagnostic

The E0-D1 diagnostic used raw paired observations from subjects 1–23 without reading protected subjects 24–29. At five dimensions, an optimistic validation-oracle linear model reconstructed waveform EEG/fNIRS with $R^2=0.162/0.973$ and descriptors with $R^2=0.893/0.931$, but its cross-modal loading balance was only `0.016/0.041`. When latent axes were required to be cross-modally correlated, validation canonical correlation fell to `0.090` for waveforms and `0.004` for descriptors, while five-dimensional shared descriptor reconstruction reached only $R^2=0.098/-0.222$. Separate modality models reached `0.880/0.965` descriptor $R^2$. The result supports narrowing the shared teacher contract and making modality-private observation state explicit; it does not change G0 or protected-test eligibility.

### Cross-dataset delayed-innovation diagnostic

E0-D2 used two subjects per dataset and reciprocal cross-subject folds. After removing self-history, trial phase, and condition, no independent modality state improved prediction of the paired innovation at the fixed five-second EEG-leading lag. The conservative cross-inferable innovation and total-feature fractions were `0%` for all four datasets. A joint-input three-dimensional CCA state compressed a balanced `3.97%/0.62%/1.63%/2.56%` of innovation in Single-Trial/REFED/Simultaneous/Visual, corresponding to `3.21%/0.56%/1.21%/2.49%` of total standardized feature variance. Because this ceiling uses both modalities, it is supportive capacity evidence only and cannot validate an independently inferable shared neural teacher.

### Lin 2024 subject-specific NVC diagnostic

E0-D3 tested a Lin-style task-related EEG component plus double-gamma HRF model on the Croce-local E0 pilot cache. The Lin-aligned split is `subject_specific_leave_one_event`, where each validation subject and anchor is fitted on three events and evaluated on the held-out event. A `subject_held_out_group` result is retained only as a stress-control because strong cross-subject degradation is expected for a subject-specific HRF method.

The subject-specific leave-one-event EEG-HRF paths remained weak: optimized stimulus, task-PLS EEG, and fNIRS-supervised EEG components reached R2 of `-0.024`, `-0.053`, and `-0.152`, respectively. Even the in-sample `subject_specific_fit_all` diagnostic reached only about `0.04-0.08` R2 for EEG-derived HRF paths, while fNIRS self-persistence remained near `0.997-0.998` R2. The revised run includes raw EEG, EEG-driver, true/predicted fNIRS, residual, and HRF-parameter visualizations. The result supports subject/measurement adapters and private hemodynamic state, not promotion of a Lin-style shared teacher.

### Lin 2024 raw continuous session TRTD diagnostic

E0-D4 repeated the Lin-style test from the original continuous MATLAB records for subject 19, session 2, rather than from Croce cache windows. The implementation used 20 s epochs from -5 to 15 s, EEG 1-40 Hz filtering, 0.5 Hz time-frequency tensors, shared spatial/frequency CP factors with trial-specific temporal factors, TRCA temporal filtering, approximate MBLL conversion of 760/850 nm NIRS intensity to HbO, GLM active-channel selection, and subject-specific double-gamma HRF fitting.

The top active HbO channels were `C1FC1`, `C1C3`, and `C2C4`. The favorable in-sample upper-bound check reached only `R2=0.022` for optimized TRTD+HRF. Leave-one-trial optimized TRTD+HRF reached `R2=-1.889`, `PCC=-0.269`, and amplitude ratio `0.150`, while fNIRS self-persistence reached `R2=0.997`. Heatmaps show that the prediction is compressed near zero and leaves the true trial-specific fNIRS state structure in the residual. The result points to a state-definition mismatch: the observed fNIRS trajectory is dominated by private slow trajectory/baseline/vascular components not captured by a one-dimensional EEG-derived HRF driver.

### Lin 2024 Simultaneous EEG&NIRS raw TRTD confirmation

E0-D5 repeated the raw-record Lin-style diagnostic on Simultaneous EEG&NIRS subject `VP001`, task `wg`, target class `WG`. This dataset already stores fNIRS as oxy/deoxy concentration in `mmol/L`, so the run tests whether the E0-D4 failure was caused by the approximate optical-to-HbO conversion in the Single-Trial dataset. EEG and fNIRS class labels matched for all 60 events; timestamps formed three stable 20-event offset blocks, so epochs used each modality's own marker time for the same event index.

The top active oxy channels were `C5h`, `CCP3`, and `C4h`. The optimized TRTD+HRF in-sample upper bound reached only `R2=0.004`, `PCC=0.060`, and amplitude ratio `0.060`. Leave-one-trial optimized TRTD+HRF reached `R2=-0.616`, `PCC=0.150`, and amplitude ratio `0.063`, while fNIRS self-persistence reached `R2=0.992`. The waveform overlay and all-trial heatmap again show compressed EEG-derived predictions and residual-dominant fNIRS structure. This confirms that the core problem is not just optical conversion; the candidate one-dimensional EEG-HRF shared trajectory is not a sufficient fNIRS semantic state for the current data.

### Unified-loader Croce/Lin raw-versus-clean retest

E0-D6 reran the Croce-2017-inspired SMC and Lin-2024-inspired TRTD/HRF paths through `UnifiedPhysiologyWindowDataset`. Single-Trial validation subjects 19-23 used identical `session_01` MA events in paired raw and admitted v3-clean EEG conditions; Simultaneous VP019-VP021 provided a separate `WG` replication. Active HbO channels were selected inside each training fold, and the main evaluation was subject-specific leave-one-trial.

Artifact correction improved Croce joint leave-one-trial R2 by `+0.1039` on average with a subject-bootstrap interval of `[0.0393, 0.1919]`, positive in `5/5` subjects. This relative change did not solve the endpoint. In the v3-clean condition, Croce joint/EEG-only and optimized Lin reached R2 of `-0.073/-0.184/-0.295`, amplitude ratios of `0.261/0.149/0.267`, and variance ratios of `0.069/0.022/0.080`. Baseline bias remained `0.32-0.38` canonical robust-SD units, trajectory-direction agreement remained near chance, and affine-oracle R2 stayed below `0.09`. Lin did not improve under cleaning. Simultaneous leave-one-trial R2 remained `-0.138/-0.239/-0.264` for the same three paths.

The corrected in-sample model-family upper bounds also stayed weak: clean Single-Trial Croce joint/Lin R2 was `0.102/0.105`, and Simultaneous was `0.025/0.057`. In contrast, fNIRS self-persistence remained around `0.997` R2 with near-unit amplitude and variance. This supports a teacher/model-family limitation with strong fNIRS-private history, not a claim that the target waveform is generally unpredictable. The diagnostic is exploratory and does not open the historical protected E0 test, but it provides no support for admitting Croce/Lin shared-driver supervision. Full analysis: [`analysis/E0_D6_UNIFIED_SHARED_DRIVER_RETEST.md`](analysis/E0_D6_UNIFIED_SHARED_DRIVER_RETEST.md).

### Adaptive physiology-constrained shared state

E0-D7 first corrected two implementation defects in the old comparison: sequential particle weights and stationary initial covariance in Croce SMC, and the C-order Khatri-Rao axis ordering in Lin CP-ALS. The corrected legacy rerun remained weak: clean Single-Trial Croce joint and optimized Lin variance ratios were `0.014/0.101`, with R2 `-0.077/-0.186`; Simultaneous reached `0.016/0.078`, with R2 `-0.093/-0.329`.

The revised candidate restores one paired HbO/HbR spatial anchor plus its six nearest scalp EEG channels. It replaces the scalar causal fixed-HRF filter with an exactly discretized Croce-linearized five-state model and whole-window RTS smoothing. Hemodynamic shape, AR dynamics, observation gains/noise, and modality balance are fitted only inside each nine-trial training fold. Event baseline remains the zero rest coordinate, and an explicit scale gauge keeps the linearized relative-flow state positive.

Local joint leave-one-trial inference reached HbO R2/PCC/variance ratio of `0.187/0.804/0.744` on the five clean Single-Trial subjects and `0.106/0.767/0.614` on the three Simultaneous subjects, while retaining EEG-proxy R2 of `0.754/0.726`. Joint drivers remain correlated `0.927/0.936` with their EEG-only counterparts but shift by `0.357/0.329` EEG-only standard deviations when fNIRS is admitted, quantitatively realizing the requested multimodal compromise. Driver monotonic fractions are about `0.52`; reconstructed HbO contains multiple turns and no longer retains only the slow climb. Local inference improves HbO R2 over all-scalp inference by `+0.241/+0.216` with negligible EEG-R2 cost.

The strict EEG-only HbO path remains poor, and frequent bounded-parameter solutions show that the physiological parameter values are not independently identifiable. E0-D7 therefore retains the joint state as a privileged soft-teacher candidate rather than a recovered physical source or admitted E0 target. Full analysis: [`analysis/E0_D7_ADAPTIVE_SHARED_NEURAL_SSM.md`](analysis/E0_D7_ADAPTIVE_SHARED_NEURAL_SSM.md).

### Task dependence of adaptive SSM parameters

E0-D8 fitted one full adaptive SSM per subject and dataset-native task condition,
using nine events per condition. The primary path held the pooled fNIRS anchor,
six local EEG channels, normalization, and EEG PCA projection fixed across
tasks. It covered all 29 Single-Trial subjects for `MA/BL_MA/LMI/RMI` and 25
Simultaneous subjects for `WG/BL_WG/0BACK/2BACK/3BACK/DSR`; VP005 was excluded
because its DSR record is not admitted by the unified loader.

No fitted parameter passed the prespecified within-family BH-FDR threshold. The
closest fixed-coordinate signals were Simultaneous driver persistence `phi`
(`W=0.126`, permutation `p=0.0063`, `q=0.1014`) and process-noise multiplier
`q_scale` (`W=0.122`, `p=0.0070`, `q=0.0974`). No hemodynamic-shape coefficient
showed an adjusted task effect. Task-specific spatial reselection also produced
no passing parameter and changed the fNIRS anchor for every subject, exposing a
strong spatial-selection confound.

All 532 optimizations converged numerically, but fixed-coordinate boundary rates
were `33-47%` for `kas/kaf/tau0`; the smallest fNIRS-noise multiplier was chosen
in `75.0%/94.7%` of Single-Trial/Simultaneous fits. The result therefore means
that robust task dependence has not been demonstrated, not that physiological
parameters are proven task invariant. Current evidence favors shared or
hierarchically shrunk hemodynamic mapping parameters with task variation routed
through the neural trajectory and observation/noise adapters. Full analysis:
[`analysis/E0_D8_ADAPTIVE_SSM_TASK_PARAMETER_AUDIT.md`](analysis/E0_D8_ADAPTIVE_SSM_TASK_PARAMETER_AUDIT.md).

## 🚦 Scientific-result admission rule

A correctness-only dry-run or smoke may be logged with an explicit non-gate status. A scientific result or gate decision is promoted only when it has:

1. a run or suite manifest under the active result root;
2. an immutable resolved configuration and split hash;
3. a declared primary endpoint from [`05_EXPERIMENT_DESIGN.md`](05_EXPERIMENT_DESIGN.md);
4. a versioned `decision_protocol.yaml`, `metric_registry.json`, and `evidence_calibration.json`;
5. a completion status that distinguishes smoke, short-formal, and full-formal evidence;
6. a link to the run-level summary rather than only a pooled suite report.

## 🗂️ Historical results

All source/observation, coupling-strengthening, exchange, alignment, and old downstream results were moved to:

```text
experiments/archive/pre_physiology_semantic_20260701/runs/
```

Their narrative log is preserved at [`source_observation/EXPERIMENT_LOG.md`](../archive/pre_physiology_semantic_20260701/source_observation/EXPERIMENT_LOG.md). Historical results are baseline evidence and never appear in this table.

## 🔗 Related documents

- [Experiment design](05_EXPERIMENT_DESIGN.md)
- [Implementation and validation plan](04_IMPLEMENTATION_VALIDATION_PLAN.md)
- [Code migration plan](07_CODE_MIGRATION_PLAN.md)
- [Storage layout](../STORAGE_LAYOUT.md)
- [Archived-run inventory](../../experiments/archive/pre_physiology_semantic_20260701/README.md)

_Last updated: 2026-07-16_
