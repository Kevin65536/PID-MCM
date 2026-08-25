# Adapter alignment and result-readiness audit

> **STATUS: STOPPED / ABANDONED（历史快照）。** 已完成的 public delivery 和后续
> campaign 记录均为 STOPPED evidence；本文“Exact remaining path”中的 protected
> review/unlock 步骤已被后续结果 supersede，属于 ABANDONED legacy queue。本文不
> 是当前执行、launch 或 authorization 入口。

_Evidence snapshot: 2026-08-11_

_Historical pre-campaign snapshot; it is not a current status source. The later
campaign completed on 2026-08-14. Use the generated
[`PROJECT_STATUS.md`](../PROJECT_STATUS.md#对比实验) for current execution and
scientific verdicts._

## What is complete, and what is not

Three states must not be conflated:

1. **Adapter-aligned** means the method can consume the declared public data
   surface and has passed the A0-A8 information-budget, support, identity,
   replay, and protocol-freeze gates (or has a preregistered unsupported cell).
2. **Public matrix complete** means public training/validation selection and
   refit artifacts exist. These scores are development evidence and are
   explicitly `table_admissible=false`.
3. **Paper-ready** means the frozen method has passed a reviewed joint unlock,
   has been evaluated exactly once on the protected outer-test identities, and
   has been aggregated into the frozen mean/standard-deviation estimand.

At this snapshot, no newly added method was at state 3 yet. STA-Net had a completed formal
aggregate, but its observation budget is method-native and is therefore kept as
a context reference rather than silently ranked in a support-matched table.

## Four-dataset coverage

| Method | Single-Trial: MI / MA | Simultaneous: WG / n-back / DSR | Visual | REFED regression | Snapshot result state |
| --- | --- | --- | --- | --- | --- |
| STA-Net | Context reference | Context reference | Context reference | Context reference | Formal aggregate retained; not support-matched direct evidence |
| EFRM LODO v2 | Pass / pass | Pass / pass / pass | Pass | Pass with input/target masks | 105/105 public jobs and A0-A8 complete; protected locked |
| BIOT | Pass / pass | Pass / pass / pass | Pass | Unsupported | 90/90 public jobs complete; protected locked |
| CBraMod | Pass / pass | Pass / pass / pass | Pass | Unsupported | 90/90 public jobs complete; protected locked |
| REVE-base | Pass / pass, overlap track | Pass / pass / pass | Pass | Unsupported | 90/90 public jobs complete; protected locked |
| BrainFusion NVC-CSP Stacking | Pass / pass | Pass / pass / unsupported | Pass | Unsupported | 75/75 public jobs complete; protected locked |
| NormWear EEG-fNIRS adapted | Pass / pass | Pass / pass / pass | Pass | Unsupported | 90/90 public jobs complete; protected locked |

The six completed public deliveries contribute 42 registered cells: 36 passes
and 6 preregistered unsupported dispositions. Direct-profile equality is
checked only among delivered pass cells; an unsupported sentinel is not an
input surface and must not be compared with a materialized peer cell.

## Recovered interruption point

EFRM LODO v2 pretraining was not still running. Its 4/4 target-dataset-excluded
selection jobs and 4/4 final refits completed on 2026-08-03. The interrupted
work was the downstream public phase after those checkpoints. That phase has
now also completed:

- loading the checkpoint that excludes the target dataset;
- extracting and hashing the full public EEG/fNIRS feature inventory;
- selecting a frozen linear probe on outer-training/public-validation data;
- reinitializing and refitting on the allowed public membership;
- retaining masks and train-only target scaling for REFED;
- the serial `7 tasks × 5 folds × 3 seeds = 105` public matrix completed with
  no failures or retries; and
- every retained run and all seven EFRM cells were independently audited,
  closing A0-A8 at `public_complete` scope.

The implementation for this phase is now in
[`run_downstream_public_v2.py`](../../comparative_methods/EFRM-PyTorch/run_downstream_public_v2.py),
with its candidate matrix builder, independent run auditor, configuration, and
tests beside it. It refuses protected paths and records
`protected_test_opened=false`, `target_dataset_exposure=false`, and
`table_admissible=false`.

The retained completion summary is
[`matrix_completion_summary.json`](../../comparative_methods/EFRM-PyTorch/evidence/public_development_v2/matrix_completion_summary.json),
and the seven-cell A0-A8 evidence is
[`summary_final.json`](../../comparative_methods/EFRM-PyTorch/evidence/alignment_v2/summary_final.json).

## Exact remaining path to the paper table（ABANDONED legacy queue）

The following path was not a current authorization and is retained only to show
what the pre-campaign snapshot had left unresolved; the later dated campaign
report is the evidence authority.

1. Human-review the frozen
   [42-cell candidate](../../comparative_methods/evidence/joint_protected_unlock_candidate_v2.json).
   Its 36 pass and 6 unsupported dispositions keep REVE's MI/MA overlap track
   and STA-Net's context-reference track separately labeled.
2. Issue a distinct joint protected authorization manifest if that review is
   approved. Public adapter or
   matrix completion does not authorize this step.
3. Evaluate each eligible frozen cell exactly once, then aggregate the frozen
   outer-fold/seed estimand to paper-facing performance and standard deviation.

Until step 2 is explicitly approved, the existing public-validation means and
standard deviations are useful diagnostics but are not publishable comparison
numbers.
