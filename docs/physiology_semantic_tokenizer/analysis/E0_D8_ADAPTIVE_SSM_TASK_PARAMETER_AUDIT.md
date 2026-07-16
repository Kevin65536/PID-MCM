# E0-D8 adaptive SSM task-parameter audit

_Exploratory within-subject task-label analysis · 2026-07-16_

---

## 📋 Abstract

This audit asks whether the adaptive five-state model introduced in E0-D7
already yields statistically distinguishable parameters across dataset-native
task labels. The primary analysis holds the fNIRS anchor, six neighbouring EEG
channels, normalization, and EEG principal-component projection fixed across
tasks within each subject. It includes 29 complete Single-Trial subjects and 25
complete Simultaneous EEG&NIRS subjects, with nine events per condition and one
full-data fit per subject-condition.

No parameter passed the prespecified Benjamini-Hochberg false-discovery-rate
threshold of `q < 0.05`. The strongest fixed-representation signals were the
Simultaneous EEG-driver persistence `phi` (`W=0.126`, permutation `p=0.0063`,
`q=0.1014`) and process-noise multiplier `q_scale` (`W=0.122`, `p=0.0070`,
`q=0.0974`). No hemodynamic-shape parameter showed an adjusted task effect.
The result therefore does not support task-specific physiological parameter
claims. It also does not establish task invariance because boundary fits,
scale-gauge freedom, and mixed event structures limit identifiability.

**Keywords:** adaptive state-space model, task effect, repeated measures,
parameter identifiability, EEG-fNIRS

## 🎯 Research question and scope

The E0-D7 formal run cannot answer this question directly: it fits `MA` only in
Single-Trial and `WG` only in Simultaneous, so task, dataset, measurement family,
subject cohort, and protocol are perfectly confounded. This audit instead asks:

> Within one dataset and the same subjects, does the distribution of fitted SSM
> parameters differ across the task labels exposed by the unified loader?

The audit does not compare parameter magnitudes across datasets and does not
treat a fitted optimum as a recovered biological constant.

## 🔬 Methodology

### Cohorts and labels

| Dataset | Complete subjects | Conditions | Events per condition |
| --- | ---: | --- | ---: |
| EEG+NIRS Single-Trial | 29 | `MA`, `BL_MA`, `LMI`, `RMI` | 9 |
| Simultaneous EEG&NIRS | 25 | `WG`, `BL_WG`, `0BACK`, `2BACK`, `3BACK`, `DSR` | 9 |

VP005 was excluded because its DSR events exist in the raw event index but its
DSR record is not admitted by `UnifiedPhysiologyWindowDataset`. The analysis
therefore follows the loader's alignment-quality boundary rather than bypassing
it with raw manifest rows.

### Analysis flow

```mermaid
flowchart LR
    accTitle: Task Parameter Audit Flow
    accDescr: Dataset-native labels enter fixed-representation subject-task fits, followed by repeated-measures permutation tests, effect sizes, and multiple-comparison control

    labels[🏷️ Read admitted labels] --> balance[⚙️ Balance nine events]
    balance --> fixed[🔒 Fix spatial representation]
    fixed --> fit[🧠 Fit subject-task SSM]
    fit --> omnibus[🔍 Permute task labels]
    omnibus --> adjust[🛡️ Control family FDR]
    balance --> sensitivity[🔄 Reselect task anchors]
    sensitivity --> fit
    adjust --> decision([📊 Report evidence boundary])

    classDef process fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef caution fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef outcome fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class labels,balance,fixed,fit,omnibus process
    class sensitivity,adjust caution
    class decision outcome
```

### Statistical unit and inference

The statistical unit is one subject-condition fit, not a leave-one-trial fold.
The primary `fixed_pooled` path selects one pooled HbO/HbR anchor and one fixed
six-channel EEG projection per subject-dataset, then applies those coordinates
unchanged to every task. The `task_specific` selector is retained only as a
sensitivity analysis because changing tasks otherwise changes both the signal
condition and spatial coordinate.

For each parameter and dataset, a Friedman rank statistic is evaluated using
50,000 within-subject task-label permutations. Kendall's `W` reports effect
size. Benjamini-Hochberg correction is applied separately to:

- Eight dynamics/driver parameters: `epsilon`, `kas`, `kaf`, `tau0`, `alpha`,
  `e0`, `phi`, and `q_driver`
- Seven observation/nuisance parameters: `q_scale`, `fnirs_noise_scale`,
  `hbo_gain`, `hbr_gain`, `eeg_noise`, `hbo_noise_base`, and `hbr_noise_base`

Pairwise Wilcoxon tests with within-parameter Holm correction are secondary
localization checks and are not promoted when the FDR-controlled omnibus family
does not pass.

## 📊 Findings

### No FDR-confirmed task-dependent parameter

No primary fixed-representation parameter reached `q < 0.05`.

| Dataset | Family | Parameter | Kendall's W | Permutation p | BH q | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Simultaneous | Dynamics/driver | `phi` | 0.126 | 0.00634 | 0.10144 | Nominal only |
| Simultaneous | Observation/nuisance | `q_scale` | 0.122 | 0.00696 | 0.09744 | Nominal only |
| Simultaneous | Observation/nuisance | `eeg_noise` | 0.091 | 0.04090 | 0.28629 | Nominal only |
| Single-Trial | Observation/nuisance | `eeg_noise` | 0.072 | 0.09462 | 0.44155 | Not supported |
| Simultaneous | Dynamics/driver | `kas` | 0.073 | 0.10158 | 0.68967 | Not supported |

The complete effect-size plot is
[task_parameter_effects.svg](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_ssm_task_parameter_audit_v1/figures/task_parameter_effects.svg).

### Nominal differences concern driver statistics, not hemodynamic constants

For Simultaneous EEG&NIRS, median fixed-representation `phi` was `0.572` for
WG, `0.589` for WG baseline, `0.622/0.601/0.620` for 0/2/3-back, and `0.570`
for DSR. The 0-back versus WG paired contrast had rank-biserial magnitude
`0.706`, but its Holm-adjusted p value was `0.069`; the family-level BH q value
for `phi` was `0.101`.

The median `q_scale` was `1.0` for WG, WG baseline, and 0-back, versus `0.5`
for 2-back, 3-back, and DSR. This parameter is selected from the discrete grid
`{0.5, 1.0, 2.0}`, so the signal describes modality-balance selection rather
than a continuous physiological coefficient.

One secondary pairwise contrast survived its within-parameter Holm adjustment:
WG had higher `eeg_noise` than 0-back by a median `0.0408` standardized units
(`r_rb=0.735`, Holm `p=0.0107`). Because the FDR-controlled nuisance omnibus did
not pass, this remains a localization clue rather than a confirmed task effect.

### Planned task contrasts do not survive localization correction

| Contrast | Strongest parameter | Raw p | Holm p | Interpretation |
| --- | --- | ---: | ---: | --- |
| `MA` vs `BL_MA` | `kaf` | 0.0264 | 0.1584 | Not supported |
| `LMI` vs `RMI` | `phi` | 0.0148 | 0.0886 | Nominal laterality signal |
| `WG` vs `BL_WG` | `phi` | 0.0157 | 0.2042 | Not supported |
| `0BACK` vs `2BACK` | `eeg_noise` | 0.0051 | 0.0710 | Nominal load signal |
| `0BACK` vs `3BACK` | `q_driver` | 0.0051 | 0.0761 | Nominal load signal |

These contrasts reinforce the same pattern: the smallest p values appear in
the EEG-driver prior or observation noise, while the hemodynamic response shape
does not show a corrected task difference.

### Spatial reselection changes the apparent signal

The task-specific selector chose at least two distinct fNIRS anchor pairs for
every subject in both datasets; no subject retained exactly the same anchor
across all tasks. Under this sensitivity path, no parameter passed FDR either.
The closest result was Simultaneous `hbr_gain` (`W=0.135`, `p=0.00364`,
`q=0.05096`). Its disappearance from the fixed-coordinate primary analysis
indicates that spatial activation/measurement selection can masquerade as a
task-specific observation gain.

![Task-parameter effect sizes for fixed and task-specific spatial coordinates](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_ssm_task_parameter_audit_v1/figures/task_parameter_effects.svg)
_Figure 1: Kendall's W for every parameter; orange points would denote BH-FDR q below 0.05. No point passes._

## 💡 Identifiability analysis

All 532 optimizations reported numerical success, but convergence is not the
same as physical identification. In the fixed-coordinate fits, boundary rates
for core parameters were:

| Parameter | Single-Trial boundary | Simultaneous boundary |
| --- | ---: | ---: |
| `kas` | 43.1% | 47.3% |
| `kaf` | 40.5% | 33.3% |
| `tau0` | 45.7% | 40.0% |
| `alpha` | 19.8% | 14.7% |
| `e0` | 5.2% | 7.3% |
| `phi` | 49.1% | 24.0% |
| `q_driver` | 1.7% | 0.0% |

The smallest fNIRS-noise multiplier `0.25` was selected in 75.0% of
Single-Trial and 94.7% of Simultaneous fits. Conversely, Single-Trial selected
the largest process-noise multiplier `q_scale=2.0` in 62.1% of fits. These
patterns show that several parameters are being used as constraint/noise knobs
at the edges of the current search space.

Consequently, the absence of an adjusted task effect has two compatible
explanations: the shared neurovascular mapping is approximately task-stable, or
the current independent bounded fits are too weakly identified to resolve a
task effect. This audit cannot distinguish those explanations.

## 🎯 Decision and next model use

The supported conclusion is:

> Current fitted SSM parameters have not demonstrated statistically robust task
> dependence. The small nominal differences are concentrated in EEG-driver
> persistence and noise balance, not the hemodynamic constants. Do not encode
> task-specific fitted physiological parameters as tokenizer supervision.

For the next model revision:

1. Fit `kas/kaf/tau0/alpha/e0` as subject-level or hierarchically shrunk mapping
   parameters shared across tasks.
2. Allow task variation primarily through the neural-drive trajectory and,
   if needed, explicit observation/noise adapters.
3. Keep the spatial coordinate fixed for confirmatory task comparisons; report
   task-specific spatial selection as a separate activation-localization result.
4. Preregister a small contrast set such as MA-versus-baseline, WG-versus-
   baseline, and n-back load trend before using labels in model selection.

This result does not alter the E0-D7 status: the joint shared state remains a
candidate privileged soft teacher, not an admitted or uniquely identified
physical source.

## 🔗 Artifacts

- [Run summary](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_ssm_task_parameter_audit_v1/summary.md)
- [Subject-task parameters](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_ssm_task_parameter_audit_v1/subject_task_parameters.csv)
- [Omnibus tests](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_ssm_task_parameter_audit_v1/omnibus_tests.csv)
- [Pairwise tests](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_ssm_task_parameter_audit_v1/pairwise_tests.csv)
- [Descriptive statistics](../../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_ssm_task_parameter_audit_v1/descriptive_statistics.csv)
- [Replay script](../../../experiments/analyze_adaptive_ssm_task_parameters.py)
- [Resolved configuration](../../../experiments/configs/physiology_semantic_tokenizer/adaptive_ssm_task_parameter_audit.yaml)

**Status:** exploratory task-parameter audit complete; no FDR-confirmed task
effect; task-specific physiological parameter supervision is not supported.

