# Physiology-semantic tokenizer experiment log

_Active run registry; historical runs retain their original contracts, new runs use the 2026-07-14 measurement-first revision_

---

## 📋 Current status

The complete tokenizer training loop is runnable. After observation-aligned sign calibration, the adaptive SSM physical teacher passes complete E0 and all SSM-derived physiological information, including fNIRS content, is fully accepted for teacher supervision. Earlier Croce/E0-v2 and negative fNIRS labels are historical pre-calibration diagnostics rather than current E0 results. The measurement-first runtime now includes patch-valid mask propagation, entry-specific routing, matched count/sum EMA, K-means/cosine/L2 geometry, annealed-hard reconstruction, gradient-preserving balance, aged occupancy health, logged revival/stop rules, and modality-specific balance temperatures. G1/E1 now passes at fixed K=128 for the registered diverse-farthest/T2-T2 candidate. Three post-stop retention seeds have final effective usage EEG `65.85 ± 1.66` and fNIRS `39.99 ± 1.38`; all retain constant revival totals for eight validation epochs after step 200 and pass the frozen modality-specific ranges. The earlier top-error candidate is rejected because one fNIRS seed transiently fell to `21.18` effective codes. A 2026-07-22 visual audit retains this occupancy/retention decision but finds strong prototype spectral concentration and no reproducible subject-held-out EEG–fNIRS token coupling trace, so full internal-geometry and physiological-coupling claims remain open. Current input normalization is full-record median/MAD, not the archived per-crop source/observation transform; archive-level assignment uniformity is not claimed. G2 information retention, G3 semantics, preservation shaping, foundation discovery, and independent certification remain separate later-stage questions.

| Date | ID | Suite | Status | Result root |
| --- | --- | --- | --- | --- |
| 2026-07-01 | `PST-DESIGN-FREEZE` | Documentation | Complete | Not applicable |
| 2026-07-02 | `PST-P1-DRYRUN` | E0 contract dry-run | Passed; G0 not evaluated | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260702_191234_p1_contract_dry_run/` |
| 2026-07-02 | `PST-P1-SMOKE` | E0 contract smoke | Passed; G0 not evaluated | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260702_191234_p1_contract_smoke/` |
| 2026-07-02 | `PST-P2-P5-DRYRUN` | Migration integration | Passed; correctness only | `experiments/runs/physiology_semantic_tokenizer/software_smoke/20260702_235450_p2_p5_software_smoke/` |
| 2026-07-02 | `PST-P2-P5-SMOKE` | Migration software smoke and P5 export | Passed; historical run predates final E0 acceptance | `experiments/runs/physiology_semantic_tokenizer/software_smoke/20260702_235459_p2_p5_software_smoke/` |
| 2026-07-03 | `PST-E0-PILOT-V1` | Teacher validity | Historical pre-sign-calibration diagnostic; superseded by final E0 pass | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260703_165153_e0_teacher_validity_pilot_v1/` |
| 2026-07-03 | `PST-E0-V2-VALIDATION` | Layered teacher information contract and visual audit | Historical pre-sign-calibration diagnostic; superseded by final E0 pass | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260703_232754_e0_teacher_validity_v2/` |
| 2026-07-06 | `PST-E0-D1-SHARED-BOUND` | Croce-independent shared-state reconstruction bound | Historical diagnostic complete; current E0 status is pass | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_105937_shared_state_reconstruction_bound_v1/` |
| 2026-07-06 | `PST-E0-D2-CROSS-DATASET-SHARED` | Four-dataset delayed-innovation shared-state diagnostic | Historical diagnostic complete; current E0 status is pass | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260706_173530_cross_dataset_shared_neural_state_v1/` |
| 2026-07-08 | `PST-E0-D3-LIN2024-SUBJECT-HRF` | Lin 2024 inspired subject-specific NVC diagnostic | Historical diagnostic complete; current E0 status is pass | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_113000_lin2024_subject_specific_nvc_v1/` |
| 2026-07-08 | `PST-E0-D4-LIN2024-RAW-TRTD` | Lin 2024 raw continuous session TRTD diagnostic | Historical diagnostic complete; current E0 status is pass | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_160736_lin2024_raw_session_trtd_s19_sess2/` |
| 2026-07-08 | `PST-E0-D5-LIN2024-SIM-RAW-TRTD` | Lin 2024 Simultaneous EEG&NIRS raw TRTD confirmation | Historical diagnostic complete; current E0 status is pass | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260708_170809_lin2024_simultaneous_raw_trtd_vp001_wg/` |
| 2026-07-10 | `PST-P1-FOUR-DATASET-QUALITY` | Four-dataset unified loader and quality audit | Correctness passed; 8-second report is historical and does not imply artifact-clean data | `experiments/runs/physiology_semantic_tokenizer/data_quality_audit/final_four_dataset_check_20260710/` |
| 2026-07-14 | `PST-INPUT-CONTRACT-REVISION` | Architecture decision | Measurement-first entrance approved; all new E0-E9 runs require unified loader | Not applicable |
| 2026-07-15 | `PST-E0-D6-UNIFIED-SHARED-DRIVER` | Unified-loader Croce/Lin raw-vs-clean retest | Historical pre-sign-calibration diagnostic; superseded by calibrated teacher acceptance | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260715_shared_neural_driver_unified_formal_v3/` |
| 2026-07-16 | `PST-E0-D7-ADAPTIVE-SHARED-SSM` | Local adaptive five-state fixed-interval shared-driver test | Diagnostic complete; joint compromise restores HbO variance/cycles while retaining EEG fit; candidate soft teacher retained, formal admission pending | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_shared_neural_ssm_formal_v2/` |
| 2026-07-16 | `PST-E0-D8-TASK-PARAMETER-AUDIT` | Within-subject adaptive-SSM task-parameter audit | Diagnostic complete; no parameter survives FDR; nominal differences concentrate in driver/noise terms, not identified hemodynamic constants | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_ssm_task_parameter_audit_v1/` |
| 2026-07-16 | `PST-E0-V3-ADAPTIVE-VALIDATION` | Original layered E0 contract with variable-parameter adaptive SSM | Pre-sign-calibration diagnostic completed; old machine label superseded by final calibrated E0 decision | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_teacher_e0_v3_validation_formal_v1/` |
| 2026-07-16 | `PST-E0-V3-GAUGE-RECALIBRATION` | Train-fold observation-aligned chromophore gauge and strict local-target contract | Required local/gauge/vocabulary layers pass; calibrated physical-teacher coordinates accepted | `experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_teacher_e0_v3_gauge_corrected_validation_v1/` |
| 2026-07-16 | `PST-E0-V3-ADMISSION-DECISION` | Adaptive joint-teacher estimand and claim-boundary review | Superseded intermediate decision; retained as the path to the final complete-E0 acceptance | `docs/physiology_semantic_tokenizer/analysis/E0_V3_ADAPTIVE_TEACHER_ADMISSION_DECISION.md` |
| 2026-07-24 | `PST-E0-SIGN-CALIBRATED-FINAL-ACCEPTANCE` | Final physical-teacher status correction | **Complete E0 passed**; sign-calibrated adaptive SSM and all physiological information, including fNIRS, fully accepted | `docs/physiology_semantic_tokenizer/analysis/20260724_E0_SIGN_CALIBRATED_PHYSICAL_TEACHER_ACCEPTANCE.md` |
| 2026-07-19 | `PST-TEACHER-GRADIENT-ENTRY-DECISION` | Physical-teacher entrance, experiment-order, and coupling-responsibility review | Required `r`/HbO/HbR local semantics retained; EEG `s` optional; flow context/coupling-only; preserve–discover–certify stages separated | `docs/physiology_semantic_tokenizer/analysis/20260719_PHYSICAL_TEACHER_GRADIENT_ENTRY_DECISION.md` |
| 2026-07-19 | `PST-M0-MEASUREMENT-FIRST-T0-SMOKE-V2` | Unified measured local view, mask routing, and corrected EMA CUDA smoke | Passed; four optimizer steps, deterministic quantizer checks 4/4, protected test unopened; correctness only | `experiments/runs/physiology_semantic_tokenizer/software_smoke/20260719_m0_measurement_first_cuda_smoke_v2/` |
| 2026-07-19 | `PST-E1-T0-HEALTH-CALIBRATION-SF-V1` | Single-Trial teacher-free combined-reconstruction health reference | Completed; do not promote—validation loss improved but residual bypass coincided with EEG occupancy contracting from 8 to 2 active codes; protected test unopened | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260719_e1_t0_measurement_first_short_formal_v1/` |
| 2026-07-19 | `PST-E1-T0-SEMANTIC-ONLY-SF-V2` | Single-Trial B1 semantic-only training health reference | Completed; do not promote—EEG/fNIRS ended at 3/12 active codes and 2.16/4.50 effective codes; protected test unopened | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260719_e1_t0_semantic_only_short_formal_v2/` |
| 2026-07-19 | `PST-E1-T0-BOUNDED-REVIVAL-V3` | Semantic-only B1 with bounded high-error dead-code revival | Superseded incomplete smoke; no manifest was emitted and no result is claimed | `experiments/runs/physiology_semantic_tokenizer/software_smoke/20260719_e1_t0_bounded_revival_cuda_smoke_v1/` |
| 2026-07-20 | `PST-E1-T0-OCCUPANCY-ABLATIONS-V4-V13` | Hard reconstruction, balance, K-means, cosine, batch, warmup, normalization, and gradient-reachability ablations | Completed; no individual factor restored both codebooks; v12 exposed the hard-marginal gradient defect and v13 confirmed weight doubling alone was insufficient | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_occupancy_comparison_v2/` |
| 2026-07-20 | `PST-E1-T0-ARCHIVED-BUNDLE-V14` | K=128 archived anti-collapse bundle with corrected EMA aging and logged revival | Completed; EEG/fNIRS `71/121` active, `55.32/42.89` effective; broad coverage restored without shrinking K, but repeated mass revival blocks E1 | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_t0_archived_revival_bundle_short_formal_v14/` |
| 2026-07-20 | `PST-E1-T0-REVIVAL-ABLATIONS-V15-V16` | Diverse replacement candidates and uniform-batch revival occupancy prior | Historical T1/T1 short result; neither improves v14 jointly and the uniform prior increases second-event revival. Uniform remains rejected; diverse was later retested with T2/T2 retention in v22/v23 | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_occupancy_comparison_v2/` |
| 2026-07-20 | `PST-E1-T0-BALANCE-T2-V17` | Archived source-like balance temperature 2 for both modalities | Completed; strongest short run at EEG/fNIRS `90/116` active, `66.54/45.65` effective, validation `1.8217`; `160/100` revivals prevent promotion | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_t0_balance_temperature2_short_formal_v17/` |
| 2026-07-20 | `PST-E1-T0-MODALITY-TEMP-V18` | EEG/fNIRS balance temperatures 2/1 | Completed; `90/116` active, `66.54/39.67` effective, validation `1.8299`; fNIRS revival falls to 85 but stability gate remains failed | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_t0_modality_balance_temperature_short_formal_v18/` |
| 2026-07-20 | `PST-E1-T0-T2-MULTISEED-V19` | Two additional fixed-split T2/T2 seeds on separate GPUs | Completed; three-seed EEG effective `67.24 ± 0.70`, fNIRS `38.08 ± 6.66`; startup revival means `156.0/102.7` | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_occupancy_comparison_v4/` |
| 2026-07-20 | `PST-E1-T0-RETENTION-V20` | Fourteen-epoch T2/T2 run with revival hard-stopped after step 200 | Completed; revival stayed `152/100` through step 462 while effective usage rose `36.81→61.43` EEG and `30.15→40.17` fNIRS; single-seed retention pass only | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_t0_post_revival_retention_formal_v20/` |
| 2026-07-20 | `PST-E1-T0-TOP-ERROR-RETENTION-V21` | Two additional registered top-error retention seeds | Completed; all seeds retain occupancy without post-step-200 revival, but the frozen gate **fails** because seed 20260721 fNIRS reaches `21.18 < 24` effective codes | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_post_revival_retention_gate_v1/` |
| 2026-07-20 | `PST-E1-T0-DIVERSE-PAIR-V22` | Paired diverse-farthest repair on the failed seed | Passed exploratory rule; fNIRS post-stop minimum/final effective improves `21.18/32.08→24.80/38.91`, EEG remains healthy, protected test unopened | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_t0_post_revival_diverse_farthest_probe_decision_v22/` |
| 2026-07-20 | `PST-E1-T0-DIVERSE-CONFIRM-V23` | Two additional diverse-farthest confirmation seeds under unchanged health ranges | **G1/E1 passed**; all three seeds pass every retention check, final effective EEG `65.85 ± 1.66`, fNIRS `39.99 ± 1.38`, mean active `86.33/110.00` | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_post_revival_diverse_farthest_gate_v2/` |
| 2026-07-20 | `PST-E1-INPUT-NORMALIZATION-AUDIT-V1` | Training-only comparison of current and archived normalization scopes | Complete; fNIRS retains substantially greater window-level scale heterogeneity than EEG; diagnostic only, no per-window transform admitted | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_training_input_normalization_audit_v1/` |
| 2026-07-20 | `PST-E1-OCCUPANCY-COMPARISON-V6` | Reproducible v2–v23 aggregation, multi-seed/retention checks, gradient probe, and final gate | Complete; training/validation only, protected test unopened; fixed K=128 retained | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_occupancy_comparison_v6/` |
| 2026-07-22 | `PST-E1-HEALTH-COUPLING-VISUAL-V1` | Twenty-four-run health/geometry visualization plus eight-run validation token coupling trace | Complete; G1 occupancy/retention pass retained, prototype geometry remains spectrally concentrated, final three-seed coupling trace is null under within-subject/label permutation and LOSO prediction; protected test unopened | `docs/physiology_semantic_tokenizer/analysis/20260722_E1_TOKENIZER_OPTIMIZATION_HEALTH_AND_COUPLING_REPORT.md` |
| 2026-07-17 | `PST-COMPARE-READINESS-AUDIT` | Comparative-method documentation and checkout audit | Preparation workflow frozen; formal comparison blocked on DSR guard, REFED target adapter, shared splits, and method admission | `docs/physiology_semantic_tokenizer/11_COMPARATIVE_METHOD_EXPERIMENT_WORKFLOW.md` |
| 2026-07-17 | `PST-DATA-DSR-VISUAL-CONTRACT` | DSR hard exclusion and Visual DC9 timing recovery | Historical correctness result; DSR exclusion was superseded on 2026-07-18, while Visual 7,750-window timing evidence remains active | `docs/physiology_semantic_tokenizer/09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md` |
| 2026-07-18 | `PST-P1-FINAL-UNIFIED-LOADER-AUDIT` | Full six-task audit of every then-admitted 20-second loader window | Historical pre-restoration snapshot; finite/geometry evidence retained, DSR and Simultaneous-QC conclusions superseded by `PST-DATA-SIM-EOG-DSR-RESTORE` | `experiments/runs/physiology_semantic_tokenizer/data_quality_audit/final_unified_loader_audit_20260718/` |
| 2026-07-18 | `PST-DATA-REFED-EEG-TOPOLOGY` | Resolve REFED EEG adjacency from a versioned standard montage | Correctness passed; 64/64 positioned, 62 exact template matches + 2 explicit interpolations, connected 168-edge graph; adjacency-only claim boundary | `docs/physiology_semantic_tokenizer/09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md` |
| 2026-07-18 | `PST-DATA-VISUAL-FNIRS-GEOMETRY` | Complete Visual fNIRS geometry from the dataset PDF, 4×4 raw mode, channel-reference workbook, and CED coordinates | Correctness passed; both probes 24/24 positioned, 14 anchors + 10 graph interpolations, connected 52-edge graphs; graphical-template-only claim boundary | `docs/physiology_semantic_tokenizer/09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md` |
| 2026-07-18 | `PST-DATA-REFED-CONTINUOUS-VA` | Versioned REFED valence/arousal sequence-regression loader | Correctness passed; 480 videos expand to 2,720 masked 20-second windows, fixed `[2,20]` targets at 1 Hz, 90.2941% paired target support; no model performance evaluated | `src/data/unified_physiology.py` |
| 2026-07-18 | `PST-DATA-SIM-EOG-DSR-RESTORE` | Simultaneous HEOG/VEOG repair, 28-channel loader contract, and DSR Go/No-go restoration | Correctness passed; 78/78 EEG records cached, median EOG correlation 0.4517→0.0221 with 15–45 Hz variance ratio 0.9965; default gate admits 8,980 DSR windows/25 subjects, VP005 remains drift-excluded | `experiments/runs/physiology_semantic_tokenizer/data_quality_audit/simultaneous_eog_clean_20260718/` |
| 2026-07-18 | `PST-P1-POST-DSR-LOADER-AUDIT` | Full seven-task audit after DSR/EOG contract update | 22,952/22,952 windows traversed; DSR 2,694 Go + 6,286 No-go, all Simultaneous tasks 28-channel clean branch; readiness remains 7 pass / 7 block / 1 warn | `experiments/runs/physiology_semantic_tokenizer/data_quality_audit/final_unified_loader_audit_post_dsr_20260718/` |
| 2026-07-18 | `PST-COMPARE-STA-NET-PYTORCH-SMOKE` | PyTorch STA-Net FGSA/EGTA rewrite with seven task-specific heads/adapters | Correctness passed on CUDA for MI, MA, WG, n-back, DSR, Visual, and REFED regression; finite forward/backward plus one optimizer step; protected test unopened | `comparative_methods/STA-Net-PyTorch/runs/smoke/sta_net_pytorch_smoke_v1/20260718_cuda_all_tasks_smoke_v3/` |
| 2026-07-19 | `PST-COMPARE-STA-NET-PYTORCH-TRAIN-V1` | Initial isolated seven-task STA-Net development training | Superseded incomplete throughput run: four tasks reached 40 epochs, DSR/Visual/REFED were terminated at their last complete checkpoints; window-level shuffling caused record-cache thrashing and validation was never executed | `comparative_methods/STA-Net-PyTorch/runs/training/20260719_sta_net_all_tasks_v2/supersession.json` |
| 2026-07-19 | `PST-COMPARE-STA-NET-PYTORCH-TRAIN-V2` | Optimized frozen seven-task train/validation queues | Completed 40 epochs for all seven tasks with per-epoch validation and best/latest checkpoints; development evidence only, not a protected comparison result | `comparative_methods/STA-Net-PyTorch/runs/training/20260719_sta_net_all_tasks_v4_optimized_frozen/` |
| 2026-07-03 | `PST-TRAIN-DRYRUN-V1` | Full trainer dry-run | Passed; no optimizer step | `experiments/runs/physiology_semantic_tokenizer/tokenizer_training/20260703_164728_physiology_semantic_tokenizer_pilot_v1/` |
| 2026-07-03 | `PST-E1-TF-SMOKE-V1` | Teacher-free reconstruction/VQ | Passed; CUDA, 2 optimizer steps | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260703_165220_tokenizer_reconstruction_baseline_pilot_v1/` |
| 2026-07-03 | `PST-E1-TF-RESUME-V1` | Teacher-free checkpoint resume | Passed; resumed to 4 optimizer steps | `experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260703_165236_tokenizer_reconstruction_baseline_pilot_v1/` |

### Historical E0 validation diagnostic

On the five validation subjects, EEG normalized predictive gain was `+0.756030` with a subject-bootstrap 95% interval of `[0.691380, 0.805454]`. The pre-sign-calibration fNIRS normalized gain was `-1.503607` with interval `[-2.383524, -0.662615]`. This is retained as a historical diagnostic value and is not the status of the calibrated physical teacher.

### E0-v2 layered validation and visual review

E0-v2 evaluated 17,280 train and 4,800 validation local patches plus 8,640 train and 2,400 validation fixed-history rows. Four raw fNIRS datasets were audited without reading their protected partitions. Train-only full-record baseline/robust-scale adapters achieved exact crop-position invariance and a validation pooled-pair scale ratio of `1.463`; original voltage, concentration, and exported relative-signal semantics remain explicit.

Patch-local observability admitted EEG `r_mean`, `r_slope`, and `s_slope`, while excluding `s_mean`. All six fNIRS level/slope coordinates exceeded their label-permutation references, with validation R² from `0.046` to `0.258`. The admitted target geometry remained transmissible through 128 prototypes: global standardized reconstruction R² was `0.918` for EEG and `0.949` for fNIRS, with perplexities `96.8` and `107.6`. The continuous upper bound was positive beyond shuffled EEG history for both fNIRS levels (`0.177` nats) and innovations (`0.172` nats); gains concentrated in `delta_f` and `delta_hbo`, while `delta_hb` added approximately zero.

The archived pre-sign-calibration report labeled two layers negative. Its fNIRS clean prediction had MSE `2.193` versus the selected history baseline `0.834`, a subject-mean gain of `-1.359` with `0/5` positive subjects. Synthetic-truth variance scaling produced the recorded coverage values. These numbers remain reproducibility evidence, but their old `fail` label is retired and must not be used as the complete-E0 or physical-teacher status.

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

E0-D5 repeated the raw-record Lin-style diagnostic on Simultaneous EEG&NIRS subject `VP001`, task `wg`, target class `WG`. This dataset already stores fNIRS as oxy/deoxy concentration in `mmol/L`, so the run tested whether the E0-D4 pre-calibration negative label came from approximate optical-to-HbO conversion in the Single-Trial dataset. EEG and fNIRS class labels matched for all 60 events; timestamps formed three stable 20-event offset blocks, so epochs used each modality's own marker time for the same event index.

The top active oxy channels were `C5h`, `CCP3`, and `C4h`. The optimized TRTD+HRF in-sample upper bound reached only `R2=0.004`, `PCC=0.060`, and amplitude ratio `0.060`. Leave-one-trial optimized TRTD+HRF reached `R2=-0.616`, `PCC=0.150`, and amplitude ratio `0.063`, while fNIRS self-persistence reached `R2=0.992`. The waveform overlay and all-trial heatmap again show compressed EEG-derived predictions and residual-dominant fNIRS structure. This confirms that the core problem is not just optical conversion; the candidate one-dimensional EEG-HRF shared trajectory is not a sufficient fNIRS semantic state for the current data.

### Unified-loader Croce/Lin raw-versus-clean retest

E0-D6 reran the Croce-2017-inspired SMC and Lin-2024-inspired TRTD/HRF paths through `UnifiedPhysiologyWindowDataset`. Single-Trial validation subjects 19-23 used identical `session_01` MA events in paired raw and admitted v3-clean EEG conditions; Simultaneous VP019-VP021 provided a separate `WG` replication. Active HbO channels were selected inside each training fold, and the main evaluation was subject-specific leave-one-trial.

Artifact correction improved Croce joint leave-one-trial R2 by `+0.1039` on average with a subject-bootstrap interval of `[0.0393, 0.1919]`, positive in `5/5` subjects. This relative change did not solve the endpoint. In the v3-clean condition, Croce joint/EEG-only and optimized Lin reached R2 of `-0.073/-0.184/-0.295`, amplitude ratios of `0.261/0.149/0.267`, and variance ratios of `0.069/0.022/0.080`. Baseline bias remained `0.32-0.38` canonical robust-SD units, trajectory-direction agreement remained near chance, and affine-oracle R2 stayed below `0.09`. Lin did not improve under cleaning. Simultaneous leave-one-trial R2 remained `-0.138/-0.239/-0.264` for the same three paths.

The corrected in-sample model-family upper bounds also stayed weak: clean Single-Trial Croce joint/Lin R2 was `0.102/0.105`, and Simultaneous was `0.025/0.057`. In contrast, fNIRS self-persistence remained around `0.997` R2 with near-unit amplitude and variance. This supports a teacher/model-family limitation with strong fNIRS-private history, not a claim that the target waveform is generally unpredictable. The diagnostic is exploratory and does not open the historical protected E0 test, but it provides no support for admitting Croce/Lin shared-driver supervision. Full analysis: [`analysis/E0_D6_UNIFIED_SHARED_DRIVER_RETEST.md`](analysis/E0_D6_UNIFIED_SHARED_DRIVER_RETEST.md).

### Adaptive physiology-constrained shared state

E0-D7 first corrected two implementation defects in the old comparison: sequential particle weights and stationary initial covariance in Croce SMC, and the C-order Khatri-Rao axis ordering in Lin CP-ALS. The corrected legacy rerun remained weak: clean Single-Trial Croce joint and optimized Lin variance ratios were `0.014/0.101`, with R2 `-0.077/-0.186`; Simultaneous reached `0.016/0.078`, with R2 `-0.093/-0.329`.

The revised candidate restores one paired HbO/HbR spatial anchor plus its six nearest scalp EEG channels. It replaces the scalar causal fixed-HRF filter with an exactly discretized Croce-linearized five-state model and whole-window RTS smoothing. Hemodynamic shape, AR dynamics, observation gains/noise, and modality balance are fitted only inside each nine-trial training fold. Event baseline remains the zero rest coordinate, and an explicit scale gauge keeps the linearized relative-flow state positive.

Local joint leave-one-trial inference reached HbO R2/PCC/variance ratio of `0.187/0.804/0.744` on the five clean Single-Trial subjects and `0.106/0.767/0.614` on the three Simultaneous subjects, while retaining EEG-proxy R2 of `0.754/0.726`. Joint drivers remain correlated `0.927/0.936` with their EEG-only counterparts but shift by `0.357/0.329` EEG-only standard deviations when fNIRS is admitted, quantitatively realizing the requested multimodal compromise. Driver monotonic fractions are about `0.52`; reconstructed HbO contains multiple turns and no longer retains only the slow climb. Local inference improves HbO R2 over all-scalp inference by `+0.241/+0.216` with negligible EEG-R2 cost.

The strict EEG-only HbO path remains poor, and frequent bounded-parameter solutions show that the physiological parameter values are not independently identifiable. At the E0-D7 diagnostic stage, the joint state was retained as a privileged soft-teacher candidate. The later gauge/sign calibration accepts it as the physical teacher and passes complete E0. Full analysis: [`analysis/E0_D7_ADAPTIVE_SHARED_NEURAL_SSM.md`](analysis/E0_D7_ADAPTIVE_SHARED_NEURAL_SSM.md).

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

### Adaptive E0-v3 layered validation

E0-v3 returned the variable-parameter adaptive five-state smoother to the
original validation-only E0 contract. The executable data condition contained
Single-Trial subjects 1–23 only: subjects 1–18 supplied 1,800 local patches and
900 context rows, while subjects 19–23 supplied 500 local patches and 250
context rows. Across 23 subjects, 230 leave-one-trial fits produced 460 joint
and EEG-only predictions. Held-out trials were excluded from fNIRS anchor
selection, six-neighbour EEG selection, EEG projection, and all fitted SSM
parameters. Protected subjects 24–29 remained closed.

The measurement contract, local observability, K=128 transmissibility, and
continuous conditional-coupling layers passed their frozen numerical rules.
Before sign calibration, five of six raw-coordinate fNIRS checks carried
negative diagnostic labels. Those labels are superseded by the calibrated
physical-teacher decision and carry no current complete-E0 status. The joint validation
trajectory no longer showed
the old monotonic-collapse phenotype: mean driver monotonic fraction was
`0.520`, and reconstructed HbO had `4.36` turns versus `2.40` in the observed
trace. This waveform-capacity improvement, together with the subsequent sign
calibration, supports the accepted physical teacher.

EEG reconstruction improved its zero baseline by `+0.80061` in all five
validation subjects. fNIRS reconstruction was worse than its selected history
baseline: combined gain was `-0.08446`, with positive gain in only `2/5`
subjects. HbO and HbR component gains were `-0.12592` and `-0.04299`. Synthetic
variance calibration passed four coordinates, while the historical HbR 90%
coverage was `0.950`. These values are retained for provenance but, because
they precede the authoritative sign-calibrated interpretation, they carry no
current physical-teacher or complete-E0 status.

The original machine conjunction and visual review recorded negative labels
under the pre-sign-calibration physical-source rule.
The strict EEG-only validation control also remained poor (`HbO R²=-4.214`,
`PCC=0.111`), preventing the better joint smoother from being reinterpreted as
EEG-to-fNIRS prediction. This historical result is preserved for provenance;
the final sign-calibrated decision is authoritative for E0 and teacher use.
Definitive files are
`summary.json`, `decision_protocol.yaml`, `physical_observation_checks.csv`,
`posterior_calibration.csv`, and `visual_review.yaml` in the run root.

### Adaptive E0-v3 gauge recalibration

The follow-up run fixed the fold-dependent sign/scale gauge between the raw
adaptive HbO/HbR latent coordinates and their fitted observation adapters. Each
held-out trajectory was mapped with training-fold gains, robust scales, and the
declared event-baseline transform; posterior standard deviations were mapped by
the absolute scale. All 230 folds were finite and non-singular, and the maximum
change relative to the already-emitted HbO/HbR reconstruction was
`1.776e-15`. The smoother, reconstructed observations, and physical gate were
therefore not refitted or improved by construction.

Under a strict all-required-coordinates conjunction, the fNIRS local gate moved
from fail to pass. HbO/HbR mean R² changed from `-0.166/-0.018` to
`0.725/0.734`; HbO/HbR slope R² changed from `-0.060/0.013` to
`0.356/0.472`. The fNIRS vocabulary consequently expanded from the earlier weak
one-coordinate geometry to four observation-aligned coordinates and passed its
K=128 random-quantizer control (`R²=0.881`, random q95 `0.853`, 92 active
codes). Flow mean/slope remained negative (`-0.071/-0.030`) and were explicitly
demoted to context-only rather than admitted by a permissive modality-level
rule. Ridge regularization is now selected only by five-fold training-subject
CV; validation subjects no longer influence hyperparameter selection.

The source run's pre-sign-calibration values (`-0.08446` fNIRS gain and `0.950`
synthetic HbR coverage) remain visible as historical diagnostics but carry no
current E0 status. The gauge/sign-corrected adaptive SSM is accepted as the
physical teacher, passes complete E0, and may provide all of its physiological
information for supervision. Continuous coupling remains a separate
downstream claim rather than evidence inferred directly from the fused teacher.
Full numerical analysis:
[`analysis/E0_V3_GAUGE_CORRECTION_GATE_GAIN.md`](analysis/E0_V3_GAUGE_CORRECTION_GATE_GAIN.md).

### Adaptive E0-v3 target-family admission decision

The design review separated three questions that had previously been combined:

1. whether a joint EEG/fNIRS teacher provides acceptable physiological
   supervision;
2. whether its parameters are uniquely identified;
3. whether independently produced EEG tokens add controlled predictive
   information about future fNIRS tokens.

Only the first question belongs to E0; it is answered `PASS` by the
sign-calibrated adaptive SSM decision.
The active theory already permits a joint target generator as privileged
information while requiring independent student inference and frozen-token
coupling evaluation. The poor EEG-only fNIRS path is therefore retained for the
second and third questions, but it is not a veto of the joint teacher.

The sign-calibrated adaptive SSM passes complete E0 as the accepted physical
teacher. All SSM-derived physiological information, including fNIRS, is
acceptable for teacher supervision. Entry-specific coordinate routing remains
an experiment-design choice and does not narrow the E0 acceptance.
Parameter-identifiability and causal claims remain separate later questions.
The authoritative decision record is
[`analysis/20260724_E0_SIGN_CALIBRATED_PHYSICAL_TEACHER_ACCEPTANCE.md`](analysis/20260724_E0_SIGN_CALIBRATED_PHYSICAL_TEACHER_ACCEPTANCE.md).

### Physical-teacher gradient-entry decision

The gauge-corrected run supports more development state than the minimum E0
boundary, but only when admission is scoped by gradient entrance. EEG
`s_mean/s_slope` are optional local/prototype ablations because both are
locally observable and transmissible; they are not new blocking coordinates.
Flow remains excluded from local/prototype supervision because its local R² is
negative, while its strong future-innovation upper bound makes it the primary
context/coupling-preservation candidate. HbO/HbR innovations remain separately
reported observation-aligned safeguards.

The resulting program distinguishes tokenizer preservation, foundation-model
discovery, and a fresh frozen/cross-fitted certificate. The teacher-training
shaper can update only the EEG semantic tokenizer and is discarded after
training. No protected data were opened and no new experiment result is
claimed by this architecture decision. Full record:
[`analysis/20260719_PHYSICAL_TEACHER_GRADIENT_ENTRY_DECISION.md`](analysis/20260719_PHYSICAL_TEACHER_GRADIENT_ENTRY_DECISION.md).

### Comparative-method readiness audit

The checkout audit confirmed that all four measured datasets enter through
`UnifiedPhysiologyWindowDataset`, but downstream comparison is not yet runnable
under a defensible shared protocol. The DSR mismatch identified by that audit
is now closed at the unified-loader boundary: 467 DSR source windows remain in
the event index for provenance but 0 are exposed. REFED valence/arousal streams
remain nested in event metadata rather than the window-level canonical label,
so the target adapter, shared splits, and method admission remain blocking.

The Visual event rebuild replaced every-third-DC9 parsing with the original
protocol's stimulus-appearance followed by three-second disappearance rule.
This recovered 54/55 records, 7,750 windows, and all 16 subjects without
relaxing the alignment gate. S06 Part1 Probe1 remains excluded because only
108 EEG stimulus anchors can be recovered for 125 fNIRS `Mark=1` anchors.

STA-Net and EFRM source trees are present as ignored nested Git repositories,
but neither is integrated with the unified loader, shared subject splits,
classification/regression metric API, or active artifact contract. STA-Net is
fixed to binary paired classification and a method-specific subject/session
runner; the checked EFRM downstream path is classification-only and its
pretraining data regime differs materially from a supervised STA-Net run. The
audit therefore retains both as candidates and separates paired supervised,
in-domain pretrained, external-pretrained, linear-probe, and full-fine-tune
tracks. No experiment or protected evaluation ran. The authoritative workflow
and blockers are recorded in
[`11_COMPARATIVE_METHOD_EXPERIMENT_WORKFLOW.md`](11_COMPARATIVE_METHOD_EXPERIMENT_WORKFLOW.md).

The 2026-07-18 final loader audit then traversed every admitted sample: 760
records, 13,972 20-second windows, and six task namespaces. DSR exposure and
non-finite amplitudes are both zero; every subject covers all admitted known
classes and each task has a stable channel signature. This supports continued
adapter preparation only. Formal unified training remains blocked by 30 Visual
unknown windows, 3,875 paired-probe semantic trial groups, REFED's missing
window-level continuous target, incomplete QC/mask consumption, unfrozen
channel/split adapters, and 28 adaptively flagged record-scale review
cases. The canonical evidence and claim boundary are recorded in
[`09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md`](09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md).

The Visual geometry rebuild then verified all 112 raw exports declare
`Mode,4x4`, materialized both Probe1 and Probe2 rather than only Probe1, and
resolved every CH1–CH24 coordinate against the dataset graphical head model and
`Location.ced`. Each probe contains 14 direct EEG-label anchors and 10
graph-Laplacian interpolations with a connected 52-edge shared-optode graph.
The rerun changed the geometry readiness check from block to pass and produced
an overall readiness count of 7 pass / 7 block / 1 warn. This is a software/data
geometry admission only; it does not establish participant-specific optode
digitization, exact source-detector distance, or co-registration accuracy.

The REFED follow-up then closed the audit's missing target-construction item
with `refed_continuous_va_sequence_v1`. The released approximately 1 Hz
joystick streams are now aligned on event-relative normalized video time and
returned as fixed `[valence, arousal, time]` sequences with per-coordinate
validity masks. At the default non-overlapping 20-second policy, 480 videos
produce 2,720 windows and retain all paired annotation support; 480 partial
final windows account for the 90.2941% valid fraction of the padded target
tensor. Values remain in the native REFED coordinate, so downstream scaling
must be train-subject-only. This is loader correctness, not evidence that a
regression model predicts affect or that either candidate method is admitted.

The Simultaneous follow-up supersedes only the earlier DSR-ban and raw-EOG
entrance decisions; the historical audits remain unchanged. `HEOG` and `VEOG`
are now auxiliary-only nuisance references, and the standard loader returns 28
scalp channels from `simultaneous_eeg_eog_clean_v1`. The branch performs robust
low-frequency EOG regression with a configurable per-channel removal cap, while
disabling bad-channel interpolation and muscle-band attenuation. Across all 78
records, median eye correlation changed from 0.4517 to 0.0221, the median
waveform correlation outside detected ocular intervals was 0.9277, and the
median 15–45 Hz variance ratio was 0.9965. These are repair and preservation
checks, not proof that all ocular signal has vanished.

DSR is restored from released EEG codes 16/32 as `Go`/`No-go`. Cross-modal time
is inherited only from each event's aligned block anchor; fNIRS remains context,
not an independent symbol label. The paper reports 180 trials, whereas all 26
released marker streams contain 360 stimuli, so the loader preserves the raw
360-count provenance. VP001 contributes 340 events because one fNIRS block
anchor is absent; VP005 is rejected by the existing continuous-drift gate. The
default entrance therefore exposes 8,980 DSR windows from 25 subjects.

### PyTorch STA-Net implementation smoke

The active comparison implementation no longer depends on TensorFlow. The
PyTorch rewrite preserves the released STA-Net FGSA/EGTA backbone, EEG auxiliary
branch, learned fusion/fNIRS prediction weighting, and alignment regularizers,
then exposes explicit binary, multiclass, DSR-context, and masked sequence-
regression variants. The unified adapter uses the released 16×16 grid for the
matching Simultaneous channel inventory and otherwise projects versioned unified
geometry; invalid time support is zeroed and bad channels are omitted before
spatial interpolation.

The all-task CUDA smoke used PyTorch 2.10.0+cu128 on an RTX 4090. MI, MA, WG,
n-back, DSR, Visual, and REFED regression each completed finite forward and
backward passes plus one optimizer step. The run emitted task contracts, adapter
and split manifests, environment, losses, predictions, checkpoints, hashes, and
claim calibration. Its smoke splits were deterministic and subject-disjoint but
are not the frozen shared benchmark splits. Protected tests remained closed, and
the reported smoke accuracy/MAE values are prohibited from performance tables.
Source-protocol reproduction and formal train/validation pilots remain next.

On 2026-07-19, the full implementation and artifact chain was isolated under
`comparative_methods/STA-Net-PyTorch/`. The first seven-process training run is
retained as superseded evidence rather than accepted performance output. Four
tasks completed training, but DSR, Visual, and REFED remained active after 16
hours; the unified loader's two-record cache was repeatedly invalidated by
window-level random sampling, batches were only one or two samples, seven jobs
oversubscribed CPU threads, and the constructed validation loader was never
called. The remaining three PIDs were terminated after preserving their last
complete-epoch checkpoints. Protected tests were never opened.

The replacement `sta_net_pytorch_training_v2` protocol uses one active task per
GPU with the other tasks in persistent queues, record-grouped batches, larger
task-specific batches, eight bounded persistent workers, BF16/TF32, and fused
AdamW. It performs validation every epoch and writes latest and best checkpoints.
A 100-step n-back benchmark improved end-to-end step throughput by roughly 8×;
the Visual path improved by roughly 30× over the observed old-run rate. The
accepted in-progress launch is `20260719_sta_net_all_tasks_v4_optimized_frozen`;
task manifests pin trainer, model, adapter, and configuration hashes. These are
still development results until completion and common-protocol admission.

A post-training reproduction-report tool now lives at
`comparative_methods/STA-Net-PyTorch/visualize_results.py`. It reloads the best
checkpoint and the validation indices already fixed by each task split, while
leaving protected-test indices unopened. It emits raw predictions, JSON/CSV
metrics, Markdown summaries, editable SVG and 300-DPI PNG figures covering
training dynamics, classification confusion/per-class/ROC/PR/calibration,
native-coordinate masked REFED regression, and STA-Net lag-attention/fusion
behavior, plus a non-pooled suite overview. A completed one-epoch n-back
engineering checkpoint exercised the full classification path; its scores are
tool-correctness evidence only and are excluded from comparison tables.

### E2 software closure and channel-contract blocker

On 2026-07-22 the E1-selected K=128 quantizer was carried into a matched
T0/T1/T2 E2 implementation. The unified loader now joins a versioned adaptive
teacher sidecar by anchor-independent measured sample identity, standardizes
targets from training subjects only, keeps local/prototype/context/coupling
masks independent, and rejects development gates whose split, cache, family,
version, or sidecar hash differs. The trainer emits per-objective gradient
norms and cosine conflicts; frozen evaluation covers continuous latents, hard
IDs, posteriors, and checkpoint codebook vectors plus signature-based prototype
matching across seeds.

The integration audit found that the historical E0-v3 local selection had not
consumed the current measured-data bad-channel mask. Of 230 development target
trials, 93 remain admissible and 137 are now excluded; train/validation counts
are 54/39 admitted and 126/11 excluded. Seven training subjects have no
admitted target. A four-step T1 CUDA smoke passed every gradient-entry
allowlist, and a frozen-evaluator smoke emitted the expected artifacts. These
are correctness results only. Formal E2 is blocked until the adaptive teacher
is rebuilt with the current channel/QC contract and the affected E0 evidence is
revalidated. Full implementation, commands, and claim boundaries are recorded
in [`analysis/20260722_E2_IMPLEMENTATION_AND_EXPERIMENT_PLAN.md`](analysis/20260722_E2_IMPLEMENTATION_AND_EXPERIMENT_PLAN.md).

### E2 v4 formal development decision

On 2026-07-23 the Single-Trial v4 line-clean branch removed the historical
bad-channel conflict, and the adaptive teacher was regenerated and revalidated
on subjects 01–23. Required local observability, target gauge, and K=128
transmissibility passed; the sidecar admitted all 230 source trials. The final
sign-calibrated decision marks complete E0 as passed and accepts all adaptive
SSM physiological information. E2 used only the registered local/prototype
target family by experimental design; uncertainty weighting and
context/coupling entrances were outside this E2 suite rather than blocked by
E0. Subjects 24–29 remained unopened.

A training-gradient-only amendment found that the original
`0.1/0.25/0.5` weights overwhelmed the shared trunk. The preregistered follow-up
grid selected `0.005`, which was hash-bound into all semantic runs. Nine
T0/T1/T2 jobs completed 462 updates each. All passed the E1 retention thresholds,
stopped revival after the registered window, retained full-rank codebooks,
passed four strict entry-gradient audits, and wrote checkpoint and implementation
hashes.

Frozen hard-token evaluation did not admit a semantic row. T1 minus T0 on the
two-modality required endpoint was `-0.0271/-0.0413/+0.0065` across matched
seeds; T2 minus T0 was `-0.0343/-0.0560/-0.0324`. Paired subject bootstrap means
were `-0.0326` (95% CI `[-0.0770, 0.0042]`) and `-0.0575`
(`[-0.1107, -0.0147]`). fNIRS hard IDs remained above their null in every row,
but EEG hard IDs were negative and below null in every run. T2 also lacked
seed-consistent optional-s improvement and reduced the required endpoint versus
T1 in all seeds. The development decision is
`no_semantic_row_admitted_retain_T0`; this is not G3, E6/G2, protected-test, or
coupling evidence.

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
- [Comparative-method experiment workflow](11_COMPARATIVE_METHOD_EXPERIMENT_WORKFLOW.md)
- [Storage layout](../STORAGE_LAYOUT.md)
- [Archived-run inventory](../../experiments/archive/pre_physiology_semantic_20260701/README.md)

_Last updated: 2026-07-23_
