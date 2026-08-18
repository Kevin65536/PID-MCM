# Method rationale and claim boundary

_Consolidated from the legacy postmortem, theoretical foundations, and
architecture-return review; dated snapshot 2026-07-30_

## Research question

The project asks whether separately observed EEG and fNIRS can support useful
representations of a shared physiological process without writing one modality
directly into the other. This is stricter than showing that a multimodal model
can reconstruct signals, classify tasks, or produce visually structured token
co-occurrence.

Three propositions must remain separate:

1. a coordinate can be constructed from joint EEG–fNIRS data;
2. each modality can independently predict that coordinate;
3. the coordinate is physically adequate and supports a reproducible
   incremental cross-modal claim.

The first two do not prove the third.

## What the earlier tokenizer taught us

The pre-physiology-semantic generation used source/residual targets, four
quantizers, coupling or exchange modules, and hard token IDs. Its strongest
audited control was the X3 causal-exchange run. The evidence remains useful as
a failure surface, not as an active architecture.

| Observation | Direct evidence | Method lesson |
| --- | --- | --- |
| Hard IDs lost usable structure | LOSO CCA: continuous `0.1483`, soft `0.1589`, hard one-hot `0.0584`, quantized embedding `0.0602` | Export posteriors, embeddings, and continuous latents; do not treat hard ID as the whole representation |
| Global dependence was not task-local | Global held-out NLL gain `0.3044`; mental arithmetic `-0.0189`, motor imagery `-0.0149`; WG interval crossed zero | Compare against task/phase/history marginals and report local uncertainty |
| Strong exchange contaminated the test | EEG context entered the fNIRS source encoder before quantization | Inference paths must remain modality-specific when testing cross-modal information |
| High utilization did not imply semantic geometry | Effective ranks included EEG source `12.28`, fNIRS source `7.20`, fNIRS observation `3.51` | Audit rank, stability, support, and phenotype rather than occupancy alone |
| Downstream performance was confounded | Source identity balanced accuracy `0.6476`; fine task label `0.2851` | Dataset/style information cannot be presented as physiological semantics |

Reconstruction, codebook utilization, a coupling heatmap, and a downstream
score are therefore engineering or descriptive observations unless an
explicit scientific gate binds them to the intended claim.

## E0–E2 generation

The physiology-semantic generation introduced a physical-teacher coordinate
and a corrected fixed `K=128` quantizer.

- E0 admitted a sign-calibrated adaptive teacher for development supervision
  only. It did not establish teacher identifiability or ground truth.
- E1 established a healthy three-seed K128 software/occupancy surface.
- E2 completed nine runs. Weak state/prototype/context/coupling objectives did
  not improve a preregistered semantic row, so the final decision retained T0.

This result rules against the tested weak multi-entry objective route. It does
not prove that no EEG–fNIRS relationship exists.

## Why the SD-SVQ return was tested

The 2026-07-25 Shared-Driver Semantic VQ proposal deliberately removed same-ID
semantics, shared codebooks, pre-VQ exchange, and a mandatory foundation-model
consumer. Its intended core was:

- raw-only modality-specific full-window encoders;
- independent `K=128,D=64` codebooks;
- a training-only complete joint-driver proxy trajectory;
- frozen exports followed by an independent evaluator.

The full-window scope was always offline/bidirectional. A future raw-fNIRS
prediction claim required a separate completed-window cutoff experiment.

The original frozen architecture, implementation, migration, and experiment
documents remain in
[`physiology_semantic_tokenizer/`](physiology_semantic_tokenizer/README.md) as
the preregistered 2026-07-25 generation. They are no longer active work
instructions.

## What R0–R2 established

The R series tested the prerequisite for quantization rather than assuming it.

| Gate | Result | Interpretation |
| --- | --- | --- |
| R0-P raw alpha–HbO lag | validation AUC `-0.02202`, CI `[-0.06685, 0.04020]`, `p=0.8224`; no 30-family FWER discovery | the registered low-dimensional raw-lag effect did not replicate |
| R2-D continuous observability | EEG ΔR² `0.031296`, CI `[-0.002166, 0.069625]`; fNIRS `-0.023018`, CI `[-0.035806, -0.007696]` | bilateral full-trajectory criterion failed |
| R1-P teacher physical consistency | HbO gain `0.234535`, but only `3/5` subjects exceeded the frozen threshold; HbR passed `4/5` | a non-degenerate, decodable coordinate still failed physical qualification |
| R1-P jointness/observability/nulls | G3–G6 passed | positive necessary properties did not compensate for G2 |
| D1B validation | serializer stopped before endpoint calculation and atomic publication | scientifically undetermined, not pass or fail |

The snapshot decision is:

```text
promotion_eligible = false
next_action = do_not_enter_r2_p
protected_subjects_24_29 = closed
```

Detailed methods and numbers remain in the
[`R-series report`](physiology_semantic_tokenizer/analysis/20260728_R_SERIES_EXPERIMENT_REPORT.md).

## Interpretation hierarchy

Use the narrowest applicable description:

1. **engineering token** — a stable discrete interface with software and
   occupancy checks;
2. **descriptive physiological token** — a supported token–measurement
   phenotype that is stable under the declared development analysis;
3. **coupling-relevant token** — a frozen representation that improves a
   preregistered cross-modal endpoint over appropriate marginal, timing, and
   subject-level nulls;
4. **mechanistic or causal token** — requires intervention or identification
   evidence not present in this project.

The completed development-only E2 T0 Atlas can support level 1 and carefully
bounded level-2 descriptions under its observed support/stability limits. The
R-series stop decision blocks promotion to levels 3–4.

## If the main method is restarted

No next SD-SVQ run is supported by this snapshot. A restart would require:

- a genuinely new independent holdout;
- a newly frozen estimator, null family, threshold, and stopping contract;
- synthetic end-to-end tests for dtype, serialization, cache-to-summary, and
  atomic publication before measured access;
- explicit competition among phase/history/systemic and physiological
  explanations;
- a target such as held-out innovation beyond task phase and fNIRS history,
  rather than visual token co-occurrence.

Only a newly qualified continuous target may reopen the decision about VQ.

## Evidence-supported statements

This evidence supports:

- the tested K128 quantizer can be software-healthy;
- the E2 weak auxiliary teacher objectives did not improve the registered
  semantic endpoint;
- the population-frozen R1-P coordinate was non-degenerate and bilaterally
  decodable but failed its full physical qualification;
- the R2-D bilateral continuous prerequisite failed;
- SD-SVQ/R2-P/R3–R7 are blocked under the frozen generation.

## Claims not supported by this evidence

This evidence does not support:

- calling the shared driver physiological ground truth;
- treating teacher reconstruction as coupling discovery;
- assigning cross-modal meaning to equal token IDs;
- selecting an attractive patch or lag after viewing the full family;
- opening protected data to rescue a failed public/development gate.
