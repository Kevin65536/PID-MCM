# Comparative-method experiment workflow

_Four-dataset downstream benchmark contract and readiness audit, updated
2026-07-22; final-number addendum added 2026-07-30_

---

## 📋 Decision and current readiness

The comparison program is approved to enter **implementation smoke**, but it is not ready for formal performance training. The measured-data entrance, restored DSR Go/No-go contract, and Simultaneous EOG-clean branch are available through `UnifiedPhysiologyWindowDataset`; REFED sequence regression uses its contract-preserving subclass `REFEDContinuousSequenceDataset`. A task-configurable PyTorch STA-Net reimplementation and unified-loader adapter pass correctness smoke on all seven task tracks. EFRM now has an isolated synchronized-data implementation, public-split pretraining boundary, seven-task transfer contract, CLIP-pair evidence exporter, and real-data CPU smoke; full ViT-base smoke, source-protocol reproduction, frozen performance protocols, and protected evaluation remain incomplete.

This document fixes the workflow while leaving the final comparison-method set open. STA-Net and EFRM are admitted as **implemented comparison candidates** for source-fidelity and shared-protocol development; neither has entered protected performance evaluation. The project explicitly permits task-specific classification and regression heads when the variant name and deviation manifest are preserved. The labels “traditional-model SOTA” and “foundation-model SOTA” remain literature-positioning hypotheses until the method review records the exact paper scope, evaluation regime, code revision, license, and relevance to each task. They are not project conclusions.

Final table values additionally follow the result-only
[final-number acceptance rules](13_COMPARATIVE_METHOD_FINAL_METRIC_ACCEPTANCE.md)
and [machine-readable targets](../../comparative_methods/comparison_metric_targets_v1.yaml).

### Audit verdict

| Area | Checkout evidence | Verdict | Required action |
| --- | --- | --- | --- |
| Four measured datasets | Unified loader registers Single-Trial, REFED, Visual, and Simultaneous | Ready for adapter work | Freeze cache and contract hashes |
| Three discrete / one continuous families | `refed_continuous_va_sequence_v1` emits fixed-shape valence/arousal sequences and masks | Target adapter ready | Preserve subject/video grouping and prove every regression loss consumes the target mask |
| DSR restoration | EEG codes 16/32 yield Go/No-go events; fNIRS times use admitted block anchors; default gate admits 8,980 windows/25 subjects | Ready with claim boundary | Use 2 s EEG epochs for ERP comparison and treat fNIRS as context, not symbol-native ground truth |
| Simultaneous ocular repair | `simultaneous_eeg_eog_clean_v1` caches all 78 records as 28 scalp EEG channels; HEOG/VEOG are auxiliary-only | Ready | Preserve branch/hash provenance; cached detections are audit-only and never runtime validity masks |
| Visual timing | Documented DC9 appearance→3-second disappearance semantics replace every-third-row parsing | Ready; 54/55 records | Keep S06 Part1 excluded unless stronger raw evidence appears |
| Visual fNIRS geometry | PDF optode layout + 112 raw `Mode,4x4` exports + partial EEG anchors projected onto `Location.ced`; both probes have connected 24-node/52-edge graphs | Ready for adjacency inputs | Keep graphical-template provenance and prohibit exact distance/co-registration claims |
| Subject-independent comparison | Shared cross-subject and single-subject registries exist for all seven tasks; EFRM fingerprints match the STA-Net task ordering | Implemented; protected folds locked | Reuse the same public hashes and keep the explicit unlock boundary |
| STA-Net | Official revision is pinned; the independent PyTorch FGSA/EGTA reimplementation, unified spatial/temporal adapter, binary/multiclass heads, and masked sequence-regression head pass seven-task CUDA smoke | Implemented comparison candidate; correctness only | Freeze formal splits/protocol, reproduce a source task, then run train/validation pilots |
| EFRM | Official code is pinned at `a62bf3d4c092ac3022b6c0bad90ec3993d5a5720`; isolated 200/10 Hz variable-channel model, paired pretraining adapter, public-split boundary, seven-task heads, and CLIP evidence tools pass unit/CPU smoke | Implemented comparison candidate; correctness only | Run full ViT-base architecture smoke after STA-Net HPO, then public development pretraining/transfer |
| Method provenance | Both isolated methods pin source URL/revision and deviations; neither upstream checkout exposes a license file | Partially ready; blocking for release/formal admission | Resolve upstream license status before redistribution |
| Fair result table | Cross-subject and within-subject result families, common metrics, and seed policy are not yet jointly frozen | Not ready | Complete C0–C5 for both evaluation families before any formal result |

The pre-contract read-only audit yielded 9,921 window references, including DSR and only eight Visual subjects. It is retained below as historical evidence of the gaps that prompted this change:

| Dataset/task | Windows | Subjects | Audit implication |
| --- | ---: | ---: | --- |
| Single-Trial / motor imagery | 1,740 | 29 | Discrete track available |
| Single-Trial / mental arithmetic | 1,740 | 29 | Discrete track available |
| REFED / emotion video | 480 | 32 | Continuous stream adapter missing |
| Visual / cognitive motivation | 3,250 | 8 | Subject-independent uncertainty requires special care |
| Simultaneous / n-back | 702 | 26 | Discrete track available |
| Simultaneous / WG | 1,560 | 26 | Discrete track available |
| Simultaneous / DSR | 449 | 25 | Historical session-block representation; superseded by stimulus-level contract |

After the 2026-07-18 DSR rebuild, the default loader exposes 22,952 windows: Single-Trial 3,480, REFED 480, Visual 7,750, and Simultaneous 11,242. The last count includes 8,980 DSR Go/No-go windows from 25 admitted subjects; VP005 DSR remains excluded for continuous clock drift. These counts remain cache snapshots rather than permanent benchmark denominators; every formal protocol records fresh counts and hashes from its pinned cache and admission policy.

The post-restoration full audit traversed all 22,952 windows and confirmed finite amplitudes, stable 28-channel Simultaneous signatures, the EOG-clean branch on all three Simultaneous tasks, and the exact DSR distribution Go 2,694 / No-go 6,286. Readiness remains 7 pass / 7 block / 1 warn because DSR restoration does not solve Visual unknown labels/probe dependence, REFED/Visual QC, mask consumption, shared splits, or channel adapters. Evidence: [`final_unified_loader_audit_post_dsr_20260718`](../../experiments/runs/physiology_semantic_tokenizer/data_quality_audit/final_unified_loader_audit_post_dsr_20260718/quality_report.md).

The pre-restoration 2026-07-18 full-loader audit checked all then-admitted 13,972 windows rather than a small signal sample. It remains historical evidence for finite loading, geometry, Visual label/probe dependence, and scale-review cases, but its DSR and Simultaneous-QC conclusions are superseded here. REFED's EEG-topology gap is closed for this benchmark: all 64 standard 10–10 labels have versioned template coordinates, with 62 exact `standard_1005` matches, two reference-figure-backed `CB1/CB2` interpolations, and a connected 168-edge within-EEG adjacency graph. Visual fNIRS position availability is also 100% under the documented graphical-template boundary. The historical evidence bundle is [`final_unified_loader_audit_20260718`](../../experiments/runs/physiology_semantic_tokenizer/data_quality_audit/final_unified_loader_audit_20260718/quality_report.md); the Simultaneous repair evidence is [`simultaneous_eog_clean_20260718`](../../experiments/runs/physiology_semantic_tokenizer/data_quality_audit/simultaneous_eog_clean_20260718/report.md).

The subsequent REFED adapter closes that audit's missing-target item. With the
current event index, 480 video events expand to 2,720 non-overlapping 20-second
windows; 480 final windows are partial, and the per-coordinate target mask
retains 90.2941% of the padded target tensor (exactly the paired annotation
support). The remaining formal blocker is no longer target construction itself,
but split freezing and proof that each candidate regression head/loss consumes
both signal and target masks.

> ⚠️ **Claim boundary:** Passing loader, adapter, reproduction, or smoke checks establishes software fidelity only. It does not establish that a method is a field-wide SOTA, that it is scientifically superior, or that a representation has discovered physiological coupling.

### Post-training visual audit

Completed STA-Net task runs are summarized with
`comparative_methods/STA-Net-PyTorch/visualize_results.py`. The tool re-evaluates
only the split-manifest validation indices from the best checkpoint and stores
the predictions needed to audit every plotted aggregate. It reports training
and validation curves, runtime/throughput, confusion matrices, per-class and
calibration diagnostics, Accuracy and Cohen's Kappa, subject-level summaries,
native-coordinate masked regression diagnostics, and EGTA
lag-attention/fusion-gate distributions. Both SVG and 300-DPI PNG are
required so figures remain editable and reviewable. The suite overview keeps
classification and regression endpoints in separate panels; it never pools
them into a synthetic ranking. Protected-test figures may be generated only by
the later frozen C6 evaluation path, not by this validation-report command.

## 🎯 Fixed benchmark scope

### Dataset and task matrix

The four raw datasets are fixed. `croce_local_cache` remains a derived optional supervision cache and never becomes a fifth downstream dataset. Dataset-native tasks are evaluated separately; labels from different namespaces are never pooled into one classifier or regressor.

| Dataset | Task track | Target type | Target contract | Scope decision |
| --- | --- | --- | --- | --- |
| `eeg_fnirs_single_trial` | `motor_imagery` | Discrete | `LMI` / `RMI` | Primary classification track |
| `eeg_fnirs_single_trial` | `mental_arithmetic` | Discrete | `MA` / `BL` | Primary classification track |
| `refed` | `emotion_video` | Continuous | 1 Hz time-aligned valence and arousal sequences | Primary regression track; adapter implemented |
| `visual_cognitive_motivation` | `visual_cognitive_motivation` | Discrete | `RR` / `RF` / `FF` / `FR` | Primary classification track; reject `unknown` |
| `simultaneous_eeg_nirs` | `nback` | Discrete | semantic levels `0-back` / `2-back` / `3-back` | Primary classification track |
| `simultaneous_eeg_nirs` | `wg` | Discrete | `WG` / `BL` | Primary classification track |
| `simultaneous_eeg_nirs` | `dsr` | Discrete, EEG-primary | `Go` / `No-go` from EEG codes 16/32 | Restored; 2 s EEG epoch recommended, fNIRS is synchronized context |

The apparent class index order in a cache is not a semantic definition. Every task adapter must carry an explicit ordered `class_names` list and reject unknown or out-of-vocabulary labels before split generation.

### Continuous-target decision

REFED is the only continuous-label dataset in the benchmark. The event index
retains the released joystick streams inside
`event.metadata.continuous_label_stream`; `REFEDContinuousSequenceDataset`
promotes them under `refed_continuous_va_sequence_v1` with this fixed contract:

1. each video is expanded from its event-relative origin with a configurable
   stride; the default is non-overlapping 20-second windows;
2. `target` has shape `[2, 20]` at the default 1 Hz target rate, ordered
   `[valence, arousal]`; batched shape is `[sample, 2, 20]`;
3. the released approximately 1 Hz annotation grid is aligned by normalized
   video time, absorbing only the nominal 47.62 Hz duration discrepancy;
4. `target_valid_mask` is per coordinate and time step, excludes non-finite
   labels and any time without both EEG and fNIRS support, and invalid values
   are zero-filled;
5. the final partial window is retained by default so every annotation is
   addressable; the mask, not an unexplained deletion threshold, determines
   regression support;
6. joystick values remain in the native REFED coordinate. Any centering or
   scaling is fit on training subjects only and recorded by the downstream
   adapter;
7. `video_context_label` retains the video category only as provenance. It is
   not the regression target;
8. the contract summary records the event-index SHA-256, source-rate range,
   target coverage, window policy, required held-out split key `subject`, and
   within-video dependency keys `subject, record_id`.

PyTorch jobs must use `collate_refed_continuous_sequences`. It stacks EEG,
fNIRS, signal masks, `[B,2,T]` targets, and target masks while retaining nullable
geometry/event/preprocessing provenance as a per-sample list; using the default
collator is prohibited because nullable provenance fields are not tensor data.

```python
from torch.utils.data import DataLoader
from src.data import (
    REFEDContinuousSequenceDataset,
    collate_refed_continuous_sequences,
)

dataset = REFEDContinuousSequenceDataset(
    window_duration_s=20.0,
    window_stride_s=20.0,
    target_sample_rate_hz=1.0,
    include_partial_windows=True,
)
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_refed_continuous_sequences,
)
```

Formal jobs wrap the dataset in indices from the frozen subject-grouped split
manifest; the example above demonstrates only the loading contract.

The primary endpoint is sequence-to-sequence prediction. A scalar window mean
may be retained as a preregistered sensitivity analysis, not substituted after
validation. Overlapping strides are allowed only when all windows from the same
subject/video remain in one split and effective sample dependence is reported.
Because the native joystick traces contain long plateaus and some individual
coordinates are constant for an entire video, CCC is computed over concatenated
valid support within each held-out subject/video aggregation, not averaged from
potentially undefined per-window CCC values. MAE, RMSE, coordinate-specific
coverage, and a train-mean predictor remain required companion reports.

### DSR restoration invariant

`simultaneous_eeg_nirs:dsr` is admitted only when labels are EEG-native `Go`/`No-go`, both modality timestamps are present, and the record passes the same alignment gate as other tasks. A session-only label, an inferred fNIRS symbol marker, or a block without an aligned anchor must fail before split generation.

The loader/event index emit the required preflight evidence:

```text
forbidden_tasks: []
selected_sample_count_by_task:
  simultaneous_eeg_nirs:dsr: 8980
class_names: [Go, No-go]
fnirs_label_role: synchronized_context_not_symbol_native_marker
alignment_exclusion: simultaneous_eeg_nirs|VP005|cnt_dsr
```

The paper states 180 trials per participant, while every released EEG marker stream contains 360 code-16/32 markers. The benchmark retains the released marker count and reports the discrepancy; no outcome-dependent deduplication or halving is allowed. Diagnostic alignment mode may inspect VP005 but does not silently promote it to the formal split.

## 🔍 Candidate-method audit

### STA-Net

STA-Net is an end-to-end paired EEG–fNIRS decoder designed around fNIRS-guided spatial alignment and EEG-guided temporal alignment. Its paper evaluates binary MI, MA, and WG decoding with subject-specific evaluation on two public datasets.[^1] The checked official repository requires Python 3.9.7 and TensorFlow 2.10.[^2]

The untouched upstream implementation remains non-portable as a shared four-dataset baseline:

- `sta.py` fixes EEG and fNIRS input tensors to method-specific 3D layouts;
- the output layer is fixed to two classes;
- `run_sta_net.py` reads pre-generated NPZ files from an absolute Windows path;
- the runner performs per-subject session holdout, not the planned shared subject-held-out protocol;
- there is no regression head or regression loss for REFED;
- preprocessing and spatial projection currently live outside the unified-loader contract.

The isolated comparison implementation is [`comparative_methods/STA-Net-PyTorch`](../../comparative_methods/STA-Net-PyTorch/README.md), with its model, unified-loader adapter, launchers, tests, provenance, configurations, and artifacts kept outside both `src/` and the project's `experiments/runs/` tree. It does not import TensorFlow. The reimplementation preserves the two FGSA blocks, EGTA cross-attention, EEG auxiliary prediction, fNIRS/fusion decision weighting, and three correlation regularizers, while exposing these explicitly named variants:

| Variant family | Tasks | Head/temporal contract | Reporting name |
| --- | --- | --- | --- |
| Source-task PyTorch reimplementation | MI, MA, WG | Binary classification; source grid used when the released channel inventory matches | `sta_net_pytorch_source_task` |
| Multiclass comparison adapter | n-back, Visual | Three/four-class heads; unified geometry projected to the STA-Net grid when needed | `sta_net_pytorch_multiclass_adapter` |
| EEG-primary context sensitivity | DSR | Two-second EEG input; fNIRS retained only as synchronized context | `sta_net_pytorch_dsr_context_adapter` |
| Masked sequence regression | REFED | `[valence, arousal, time]` output with per-coordinate target mask | `sta_net_pytorch_regression_adapter` |

The seven-task CUDA smoke at [`20260718_cuda_all_tasks_smoke_v3`](../../comparative_methods/STA-Net-PyTorch/runs/smoke/sta_net_pytorch_smoke_v1/20260718_cuda_all_tasks_smoke_v3/) used real unified-loader samples, disjoint smoke train/validation subjects, finite forward/backward passes, one optimizer step, prediction files, checkpoints, manifests, implementation hashes, and metric artifacts. All seven task statuses are `smoke_passed`; protected tests remained closed. This proves software connectivity only. The smoke losses and accuracies/MAE are not performance estimates because each task used only a handful of deterministic samples and one update.

The first full training launch was superseded after a throughput audit found
window-shuffled record-cache thrashing, one/two-sample batches, seven-way CPU
contention, and an unused validation loader. The replacement v2 protocol uses
record-grouped batches and two sequential per-GPU task queues, performs
validation every epoch, saves best/latest checkpoints, and pins implementation
and configuration hashes. Its active run is
[`20260719_sta_net_all_tasks_v4_optimized_frozen`](../../comparative_methods/STA-Net-PyTorch/runs/training/20260719_sta_net_all_tasks_v4_optimized_frozen/).
It is completed development evidence, not an admitted performance result.

STA-Net may therefore enter the **paired supervised architecture track** for public train/validation development. Source-protocol reproduction remains required before calling the PyTorch implementation a faithful named-method reproduction. For MI, MA, and WG, the source-aligned reproduction must generate the paper's two reported performance indicators—per-subject Accuracy and Cohen's Kappa—and aggregate each as mean, sample standard deviation, and subject-level confidence interval. Original subject-specific results, source-task PyTorch reproduction, shared within-subject results, and subject-independent adapted results must occupy separate, explicitly labeled tables.

### EFRM

EFRM is a two-stage representation-learning method that combines modality-specific masked autoencoding with paired EEG–fNIRS contrastive alignment, followed by downstream transfer. The paper reports pretraining on approximately 1,250 hours from 918 participants and evaluates label-efficient classification.[^3] The checked official code exposes pretraining, fine-tuning, and linear-probe paths for classification.[^4]

That total is predominantly unpaired: 868 hours / 766 participants are EEG-only, 364 hours / 123 participants are fNIRS-only, and 15.5 hours / 29 participants are paired EEG-fNIRS. In the released loop, the first two pools supervise their respective MAE reconstruction losses; only the paired loader defines the CLIP identity positives. Cycling the shorter loaders makes all three losses available at each optimization step, but it does not align unpaired people or experiments.

The released implementation establishes the following source constraints:

- the pretraining loader expects separate EEG-only, fNIRS-only, and paired directory trees;
- method inputs use fixed 8-second targets at 128 Hz for EEG and 16 Hz for fNIRS, unlike the canonical 200/10 Hz unified coordinates;
- channel shortfalls are handled through repetition or mirroring in the vendor loader;
- downstream datasets and class counts are enumerated in code;
- downstream solvers use classification heads and `CrossEntropyLoss`;
- no released path defines continuous valence/arousal regression.

The isolated reproduction is [`comparative_methods/EFRM-PyTorch`](../../comparative_methods/EFRM-PyTorch/README.md), reported as `efrm_sync_200_10_variable_channel_v1`. It freezes the following decisions:

- admit the current `homer2_aligned_fnirs` HbO/HbR branch after component construction and full-record robust amplitude alignment, while retaining the provenance warning that intensity→optical-density uses `-log` and MBLL and is not globally a linear raw-measurement transform;
- use EEG at 200 Hz and fNIRS at 10 Hz, with 50-sample EEG and 20-sample fNIRS temporal patches so the physical patch durations remain 0.25 s and 2 s;
- train EEG MAE, fNIRS MAE, and symmetric CLIP retrieval from the same synchronized pair batch; no single-modality external duration is introduced;
- retain every measured good EEG channel and every name-paired HbO/HbR location, consume validity masks in reconstruction and pooling, and never repeat/mirror channels;
- form stackable batches within the same measured channel inventory but sample records round-robin, so CLIP negatives are not systematically adjacent/overlapping windows from one recording;
- cover MI, MA, WG, n-back, DSR, Visual, and the explicitly named `efrm_sync_regression_adapter` for REFED;
- retain fold-specific pretraining as the requirement for any future exact full-dataset fold-matched benchmark, while the active compute-bounded track uses one source-only checkpoint and a completely disjoint target cohort for both strict and sample-random five-fold downstream evaluation; all-subject pretraining remains diagnostic/transductive only;
- export the exact identity positive-pair mask, raw cosine matrix, scaled logits, bidirectional ranks/top-k/MRR, within-record hard negatives, and paired embedding projection. The side-by-side physiological figure labels EFRM's diagonal as synchronized co-occurrence, not direction, hemodynamic delay, or mechanism.

Public-boundary preflight reuses the seven frozen STA-Net development split manifests. With the strict “common allowed subjects; validation role wins” rule, it admits 14,194 synchronized training windows and 3,540 validation windows across the four datasets, with no protected manifest opened. Ten implementation tests and a real Single-Trial CPU forward/backward smoke pass; these are connectivity evidence, not performance estimates.

EFRM may enter the **pretrained transfer track** only with the pretraining data regime explicit. The unavailable paper checkpoint cannot populate the `external_pretraining` track. The primary run is instead `in_domain_pretraining`, trained only on the four admitted synchronized datasets. It must not be described as a numerical reproduction of the paper's 1,247.5-hour pretraining result.

The active EFRM performance protocol is frozen in
[`20260725_RESOURCE_BOUNDED_DUAL_PROTOCOL_FREEZE.md`](../../comparative_methods/EFRM-PyTorch/sources/20260725_RESOURCE_BOUNDED_DUAL_PROTOCOL_FREEZE.md).
Dataset-level subjects are divided once into a source cohort and a disjoint
target cohort. One source-only EFRM checkpoint is frozen before the target
cohort supplies strict cross-subject and direct sample-random five-fold
downstream folds. The required primary matrix is paired-modality linear
probing for seven tasks, two protocols, and five outer folds. Macro-F1 and
Accuracy are reported for classification, native CCC is primary for REFED,
and the formal standard deviation is the sample SD across the five target
outer folds. Seed SD and pooled out-of-fold metrics cannot replace fold SD.

This design estimates resource-bounded source-to-target transfer rather than
the current STA-Net full-dataset estimand. Direct EFRM-versus-STA-Net ranking
is permitted only after STA-Net is run on the exact same EFRM target cohort
and fold manifests. A fixed checkpoint that has seen target samples during
self-supervised pretraining may be reported only as a separately named
transductive diagnostic and is excluded from both primary result families.

### Method-selection rule

STA-Net and EFRM answer different questions. Selecting both is scientifically coherent only if the result tables remain stratified:

| Track | Question | Candidate | Data regime |
| --- | --- | --- | --- |
| Paired supervised | How does a task-trained fusion architecture compare under the same labeled folds? | STA-Net | No external pretraining |
| In-domain pretrained transfer | How does representation pretraining on the admitted corpus affect transfer? | EFRM and project model | Same four-dataset pretraining pool |
| External pretrained transfer | What is achievable with the released large-corpus prior? | EFRM released checkpoint, if obtainable and licensed | External data declared |
| Linear probe | What information is present in frozen representations? | EFRM and project model | Same frozen folds and head budget |
| Full fine-tune | What is the end-to-end adapted performance? | EFRM and project model | Same label budget and stopping rule |

No method is required to participate in a task it cannot represent faithfully. A regression head added to STA-Net or EFRM is an explicit project adaptation and must be named `<method>_regression_adapter`, never reported as the untouched official method.

## 🔄 Executable workflow

```mermaid
flowchart LR
    accTitle: Comparative Method Admission Workflow
    accDescr: Comparative experiments progress from a frozen task contract through split, provenance, adapter, and reproduction gates before protected evaluation; any failed gate returns the method to preparation.

    freeze_scope[📋 Freeze task contract] --> build_splits[🔒 Build subject splits]
    build_splits --> audit_method[🔍 Audit method provenance]
    audit_method --> adapt_inputs[🔧 Build tensor adapters]
    adapt_inputs --> reproduce_source{🧪 Reproduce source?}
    reproduce_source -->|No| revise_adapter[✏️ Revise adapter]
    revise_adapter --> adapt_inputs
    reproduce_source -->|Yes| run_smoke[⚡ Run shared smoke]
    run_smoke --> freeze_protocol[🔒 Freeze evaluation]
    freeze_protocol --> formal_eval[📊 Run protected test]
    formal_eval --> publish_results([✅ Publish stratified results])

    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef decision fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class freeze_scope,build_splits,audit_method,adapt_inputs,revise_adapter,run_smoke,freeze_protocol,formal_eval process
    class reproduce_source decision
    class publish_results success
```

### C0 — Freeze the benchmark contract

Create a versioned `benchmark_protocol.yaml` that records:

- four dataset IDs and dataset/task namespaces;
- the DSR Go/No-go class order, EEG-primary label source, epoch policy, and fNIRS-context boundary;
- discrete class names and REFED continuous-target schema;
- loader class, loader contract, cache/index hashes, signal branches, masks, and window policy;
- primary modality regime, label budget, split family, seed policy, and result aggregation rule;
- mandatory subject-independent and within-subject evaluation families, including their grouping units and eligibility rules;
- protected-test boundary and protocol hash.

**Gate C0:** a machine-readable inventory shows exactly the declared tasks, zero forbidden/unknown labels, target coverage, subject counts, and per-task class or target distributions.

### C1 — Generate shared cross-subject and within-subject splits

Generate splits once, before method-specific tensors. Every admitted task must produce both result families in the same formal protocol version:

1. **Subject-independent / cross-subject:** group-exclusive outer-fold evaluation with nested train/validation selection. Each dataset/task receives versioned fold manifests containing train, validation, and protected outer-test subjects plus hashes of ordered sample IDs.
2. **Within-subject / non-cross-subject:** one evaluation is constructed per eligible subject, with sessions, records, blocks, videos, or semantic trials—not individual windows—as the indivisible split unit. Each subject receives versioned public train/validation and protected test manifests with ordered sample-ID hashes. Random window-level splitting is prohibited.

The fold counts and within-subject grouping units are frozen before outcome inspection from admitted subject/session/record coverage rather than imposed as one universal number. The current 16-subject Visual entrance still requires cross-subject uncertainty to remain explicit; in the within-subject family, Probe1/Probe2 views of the same semantic trial must remain grouped and cannot inflate the effective trial denominator.

Rules:

- in the cross-subject family, all sessions, records, windows, and probes from one subject remain in one partition;
- cross-subject task tracks within one dataset reuse the same subject partition whenever coverage permits;
- REFED windows from one subject never cross cross-subject partitions, and windows from one video never cross within-subject partitions;
- within-subject folds keep all windows from the same session, record, block, video, or semantic trial together, using the strongest dataset-native dependency key available;
- a subject without enough independent groups or class/target support is marked protocol-ineligible with a recorded reason and denominator; it is never silently dropped after outcomes are observed;
- hyperparameters, early stopping, calibration, channel selection, and target scaling use train/validation only;
- cross-subject and within-subject evaluation use different protocol IDs, manifests, result roots, and tables even when they share a task and model configuration.

**Gate C1:** both result families exist for every eligible task; there is zero subject, dependency-group, sample-ID, normalization-statistic, or target-scaling leakage; and all methods in the same task/result family reuse identical fold and split hashes.

### C2 — Pin method provenance and environments

Each method receives `method_manifest.yaml` containing source URL, paper DOI, commit, local patch hash, dependency lock, hardware/runtime notes, checkpoint provenance, upstream preprocessing assumptions, license status, and deviations from the paper.

Vendor repositories remain read-only evidence. Project adapters live outside the nested repositories so upstream revision and local modifications are distinguishable. A missing or incompatible license status blocks redistribution and publication packaging even if local research execution is technically possible.

**Gate C2:** a clean environment can construct the unmodified source model and a reviewer can distinguish upstream code, project adapter, and project evaluation code.

### C3 — Build split-aware data and target adapters

The sole measured-data entrance is `UnifiedPhysiologyWindowDataset`. Method adapters may reshape, resample, crop, spatially project, or batch tensors only after the shared split is resolved.

Every transformation records:

- input and output shapes, sampling rates, channel order, geometry source, and time support;
- padding, interpolation, missing-channel, bad-channel, artifact-mask, and validity-mask behavior;
- train-only learned statistics and their hashes;
- class/target mapping and target-validity mask;
- whether the transformation is required by the source paper or introduced by this project.

STA-Net's spatial grids and EFRM's fixed sampling/channel targets require explicit sensitivity controls because these transformations may materially change the task. Repeating channels, fabricating geometry, or discarding masks silently is prohibited.

**Gate C3:** deterministic adapter tests, DSR contract validation, no cross-split fitted state, sample-order round trip, and modality/time alignment checks pass on every task.

### C4 — Establish method fidelity

Before the shared benchmark, reproduce one source-supported task close to the paper protocol. For STA-Net, the source-fidelity suite covers MI, MA, and WG when their original data are available and reports per-subject Accuracy and Cohen's Kappa, followed by the across-subject mean and sample standard deviation used by the paper. Record the source preprocessing, session partition, evaluation regime, metric definitions, uncertainty, and any gap from the published result. Exact numerical identity is not required, but unexplained failure blocks claims that the implementation represents the named method.

Then run a shared-protocol dry run and smoke on public train/validation subjects only. A method must emit finite losses, predictions, checkpoints, runtime/resource measurements, and evaluation artifacts using the common split and metric API.

**Gate C4:** source-fidelity status is `reproduced`, `approximately_reproduced`, or `not_reproduced` with evidence. Only the first two may enter formal comparison under the named method.

### C5 — Freeze task metrics and selection

Classification and regression use different endpoints; raw values are never pooled across target types. Metric roles are also explicit by result family:

| Result family | Scope | Primary endpoint | Required additional evidence |
| --- | --- | --- | --- |
| Source-aligned within-subject reproduction | STA-Net MI / MA / WG | mean per-subject Accuracy | Cohen's Kappa, subject-level Accuracy/Kappa rows, sample SD, confidence interval, and the exact paper-compatible session split |
| Shared subject-independent discrete benchmark | All classification tasks | subject-level macro F1 across protected outer subjects | **Accuracy and Cohen's Kappa**, balanced accuracy, per-class recall, confusion matrix, calibration, loss, and subject-level bootstrap interval |
| Shared within-subject discrete benchmark | All eligible classification subjects | mean per-subject macro F1 across protected within-subject groups | **Accuracy and Cohen's Kappa**, balanced accuracy, per-class recall, confusion matrix, calibration, loss, sample SD, and subject-level bootstrap interval |
| Shared subject-independent continuous benchmark | REFED | subject-level concordance correlation coefficient | MAE, RMSE, R², Pearson/Spearman correlation, target coverage, and subject-level bootstrap interval |
| Shared within-subject continuous benchmark | Eligible REFED subjects | mean per-subject concordance correlation coefficient across protected videos/records | MAE, RMSE, R², Pearson/Spearman correlation, target coverage, sample SD, and subject-level bootstrap interval |

Accuracy and Cohen's Kappa are mandatory outputs for every discrete STA-Net evaluation because they are the original paper's reported indicators; they do not replace macro F1 as the shared imbalanced/multiclass benchmark primary endpoint. Every classification prediction artifact therefore emits Accuracy, Kappa, balanced accuracy, and macro F1 at the fold/subject level before aggregation. Pooled-window versions may be retained as diagnostics but cannot substitute for subject-level estimates.

Each suite has exactly one declared primary endpoint as specified above. Primary endpoints are aggregated over held-out subjects or protected within-subject dependency groups, with subject-level bootstrap uncertainty where applicable. Class imbalance handling, checkpoint selection, and metric calibration are fixed from training/validation data. No universal numerical pass threshold is imposed.

Before a result is filled into the paper table, apply the lightweight
[final-number acceptance rules](13_COMPARATIVE_METHOD_FINAL_METRIC_ACCEPTANCE.md).
They check only the resulting number: validity and comparability, improvement
over the correct simple baseline, a per-cell reasonable target band, and any
applicable source-paper value or task relation. They do not impose a new
training, disclosure, hashing, or protected-test process.

For the overall summary, report every dataset/task separately. A cross-task summary may use paired ranks or normalized effect sizes as secondary evidence; it cannot average macro F1 and concordance correlation into one score.

**Gate C5:** `decision_protocol.yaml`, `metric_registry.json`, and `evidence_calibration.json` enumerate and freeze both result families, the source-paper Accuracy/Kappa outputs, aggregation units, and uncertainty procedures before either protected evaluation is opened.

### C6 — Run formal comparison and release artifacts

Formal runs execute both the subject-independent and within-subject task–split–seed matrices for every admitted method and project model. A formal method suite is incomplete until both eligible matrices finish or every ineligible subject/task has a pre-outcome reason recorded. Compare matched tracks only: scratch with scratch, in-domain pretraining with in-domain pretraining, released external checkpoints in a separate table, linear probe with linear probe, and full fine-tune with full fine-tune.

The two evaluation families are never merged into one headline number. Reports contain separate tables for source-aligned STA-Net reproduction, shared within-subject comparison, and shared cross-subject comparison. The within-subject table reports the per-subject rows behind every aggregate so that a large number of windows cannot masquerade as a large subject denominator.

The complete protected evaluation for each family is opened once per frozen protocol version: outer-subject folds for the cross-subject family and protected dependency-group folds for the within-subject family. Failed or incomplete methods remain visible; they are not dropped after inspecting project-model results. Revisions require a new protocol version and fresh protected evidence.

The final table accepts `TABLE_READY` and `TABLE_READY_WITH_NOTE` values.
Other results are marked `REJECTED_VALUE`, `FAILURE_RESULT`, or
`INVALID_VALUE`; this result-only rule does not prescribe any additional run.
No result is clipped to chance or replaced with the paper value.

**Gate C6:** every table cell resolves to an immutable run manifest, prediction file, subject-level metric table, environment, checkpoint hash, and completion status.

## ⚙️ Fairness and reporting contract

### Matched factors

The following factors must match within a comparison track unless the factor itself is the declared intervention:

| Factor | Required handling |
| --- | --- |
| Measured samples | Same ordered split/sample IDs within each evaluation family |
| Label access | Same train labels and label budget |
| Pretraining corpus | Same in-domain corpus, or separate external track |
| Input modality | Paired, EEG-only, and fNIRS-only reported separately |
| Model selection | Same validation subjects or within-subject dependency groups and the same family-specific primary selection metric |
| Evaluation family | Both cross-subject and within-subject results are mandatory and separately labeled; neither may stand in for the other |
| Randomness | Same declared seed set; method-native nondeterminism recorded |
| Training budget | Matched optimizer-step or compute-budget policy plus actual resource report |
| Augmentation | Method-native versus shared augmentations declared and ablated when material |
| Missing data | Same admitted samples; method-specific rejection counts reported |

Parameter count alone is not a sufficient fairness rule. Both matched-budget and method-faithful configurations may be useful, but they belong in separately named tables.

### Required ablations

At minimum, paired methods report EEG-only, fNIRS-only, and paired inputs where the architecture supports them. The project model and EFRM additionally separate linear probing from full fine-tuning. STA-Net reports whether project-added configurable heads or regression adapters change the source architecture.

The primary comparison concerns downstream prediction. Coupling heatmaps, attention maps, reconstruction loss, and embedding geometry remain diagnostic and cannot replace the declared task endpoint.

## 📦 Artifact and directory contract

```text
comparative_methods/<method_id>/runs/<protocol_id>/<evaluation_family>/<dataset_id>/<task_id>/<run_id>/
├── benchmark_protocol.yaml
├── method_manifest.yaml
├── adapter_manifest.json
├── split_manifest.json
├── resolved_config.yaml
├── decision_protocol.yaml
├── metric_registry.json
├── evidence_calibration.json
├── environment.json
├── manifest.json
├── checkpoints/
├── metrics/
│   ├── train.jsonl
│   ├── validation.jsonl
│   ├── subject_metrics.csv
│   └── test_summary.json
├── predictions/
├── figures/
├── figure_data/
└── summary.md
```

`manifest.json` records repository commit and dirty state, nested method commit, protocol/split/adapter hashes, cache/index hashes, command, seeds, hardware, start/end time, completion status, and all checkpoint/prediction hashes. Aggregate reports are regenerated only from immutable run-level artifacts.

## ✅ Implementation order and definition of done

### Immediate implementation backlog

| Priority | Deliverable | Blocking evidence resolved |
| ---: | --- | --- |
| 1 | `comparative_task_contract_v1` | Task namespaces, class names, REFED target definition, DSR restoration boundary |
| 2 | Unified downstream label adapter | REFED continuous targets and masks |
| 3 | DSR event/preflight tests | **Complete:** 8,980 admitted Go/No-go windows; VP005 remains alignment-excluded |
| 4 | Shared subject split generator | **Complete:** cross-subject and single-subject registries; EFRM ordering hashes match |
| 5 | Method provenance manifests | **STA-Net and EFRM manifests present;** both upstream license files remain unavailable |
| 6 | Common prediction/metric API | **STA-Net classification/regression smoke API complete;** cross-method formal API pending |
| 7 | STA-Net tensor/head adapter | **Complete:** PyTorch FGSA/EGTA, binary/multiclass/regression heads, unified geometry/mask adapter |
| 8 | EFRM data/head adapter | **Implemented:** synchronized 200/10 Hz variable-channel MAEs, seven-task heads, REFED regression adapter, split boundary, and CLIP evidence export |
| 9 | Source-fidelity reproductions | Named-method validity |
| 10 | Shared train/validation smokes | **STA-Net complete on all seven tasks; EFRM real-data CPU correctness complete;** EFRM full-model/public development training pending |

The full-loader audit adds three prerequisites ahead of method tensor export: reject Visual unknown labels before split generation; freeze paired-probe grouping/fusion or weighting; and test that method adapters consume time-validity, analysis-valid, artifact, bad-channel, channel, and target masks while preserving geometry provenance and template-coordinate sensitivity controls.

### Preparation is complete when

1. the four datasets and seven admitted task tracks have machine-readable target contracts;
2. every comparative entrance proves valid DSR Go/No-go provenance and zero unknown labels;
3. REFED continuous targets are aligned, masked, versioned, and tested;
4. shared cross-subject and within-subject split manifests are both present and reused by all methods in each matched track;
5. STA-Net and EFRM have pinned provenance, environment, license status, and source-fidelity results;
6. every method consumes unified-loader samples through deterministic, split-aware adapters;
7. classification and regression dry runs emit the required artifact schema;
8. both subject-specific/within-subject and subject-independent/cross-subject result matrices are generated, while data-regime, modality, probe/fine-tune, and evaluation-family results cannot enter the same unlabeled table;
9. protected-test protocols remain unopened until C0–C5 are frozen.

## 🔗 Related documents

- [Experiment design](05_EXPERIMENT_DESIGN.md)
- [Experiment log](06_EXPERIMENT_LOG.md)
- [Implementation and validation plan](04_IMPLEMENTATION_VALIDATION_PLAN.md)
- [Data normalization and unified cache audit](09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md)
- [Final performance-number targets](13_COMPARATIVE_METHOD_FINAL_METRIC_ACCEPTANCE.md)
- [Machine-readable final-number targets](../../comparative_methods/comparison_metric_targets_v1.yaml)
- [Dataset descriptions](../DATASETS_DESCRIPTION.md)

## 📚 References

[^1]: Liu, M., et al. (2025). “STA-Net: Spatial–temporal alignment network for hybrid EEG-fNIRS decoding.” _Information Fusion_, 119, 103023. https://doi.org/10.1016/j.inffus.2025.103023

[^2]: Liu, M., et al. “STA-Net official implementation.” GitHub. https://github.com/MutianLiu-SHU/STA-Net

[^3]: Jung, E., & An, J. (2025). “EFRM: A Multimodal EEG–fNIRS Representation-learning Model for few-shot brain-signal classification.” _Computers in Biology and Medicine_, 199, 111292. https://doi.org/10.1016/j.compbiomed.2025.111292

[^4]: Jung, E., & An, J. “EFRM official implementation.” GitHub. https://github.com/EuijinMisp/EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model

_Readiness audit last updated: 2026-07-22; final-number addendum: 2026-07-30_
