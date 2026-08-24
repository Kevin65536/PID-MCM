# Method rationale and claim boundary

_Consolidated from the legacy postmortem, theoretical foundations, and
architecture-return review; v1 historical boundary and replaceable future
candidates, 2026-08-22_

## Version boundary and status

This document has two deliberately different jobs. **v1** names the
implemented E2/SSM_OBSERVATION/LC-SPVQ development surfaces and their audited
negative results. **v2 exploration** groups replaceable, unimplemented
candidates; it does not name the project, define the next architecture, or
freeze a method identity. A diagram, schema, or design paragraph does not
authorize a measured run, open a protected split, or relabel a v1 result.

The v1 code, checkpoints, manifests, and reports remain reproducible history.
They must not be silently upgraded by changing their names in prose. The
forward-looking claim boundary is therefore:

```text
v1 implementation/negative evidence  ->  historical fact
v2 exploration note                   ->  candidate menu, not an architecture
future held-out coupling claim         ->  unavailable until one candidate is
                                          selected, versioned, synthetic-tested,
                                          and independently evaluated
```

No measured or protected data was read for this documentation update.

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

## Comparative positioning and novelty boundary

The candidate exploration changes possible estimands and evidence interfaces; it does not, by
itself, establish priority or superiority over earlier multimodal methods.
[DMSL](https://pubmed.ncbi.nlm.nih.gov/41632672/) combines multimodal attention,
reconstructed-modality graph processing, consistency/disparity constraints,
and adaptive fusion for EEG--fNIRS classification. [FOCAL](https://proceedings.neurips.cc/paper_files/paper/2023/file/93e98ddf39a9beb0a97fbbe56a986c80-Paper-Conference.pdf)
uses a factorized orthogonal latent space with modal matching, private-view
invariance, and temporal structure; [FactorCL](https://proceedings.neurips.cc/paper_files/paper/2023/file/6818dcc65fdf3cbd4b05770fb957803e-Paper-Conference.pdf)
uses mutual-information bounds and multimodal augmentation assumptions to
target task-relevant shared and unique information. [EEGPT](https://papers.nips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html)
motivates high-signal, semantically informed EEG representation targets rather
than treating a noisy raw waveform as the only self-supervision target.

Available candidates include continuous trajectory, uncertainty, and residual
targets; modality-specific observation/source inference and vocabularies; a
fine-to-coarse endpoint grammar; and a preregistered held-out
proper-score/null evidence surface. These are hypotheses about input ownership,
estimands, and claim contracts, not a frozen bundle. Direct performance or novelty claims require a
same-track comparison with matched inputs, splits, pretraining exposure, and
training budget; none is supplied by this documentation update.

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

## What the 2026-08 v1 screen did and did not establish

The latest exploratory screen is a v1 implementation audit, not a v2
qualification. The corrected endpoint-aligned lag mask removed the historical
same-position shortcut; the old LC-SPVQ checkpoint still showed no stable
sample-dependent interaction increment (motor-imagery private-only macro-F1
was already `0.7273`, while the historical word-generation combined head was
worse). The interaction-logit variation was on the order of `10^-6`, and the
shuffle control was essentially unchanged.

The v1 observation target also did not have the geometry claimed for a
continuous teacher. A 20 s window was split into ten 2 s patches; fNIRS patch
samples were flattened into features and the full-rank observation AR smoother
then operated on the ten patch positions. It therefore saw approximately
`[B, 10, 40]` for the two-channel fNIRS branch, not a `[B, 200, 2]` 10 Hz
trajectory. EEG used absolute patch log-band power with a generic
full-patch-convolution/LayerNorm stem. These are historical v1 contracts.

The three-seed v1 SSM screen found motor-imagery EEG delta-R² from `-0.0554`
to `-0.0702` across NATIVE/SSM modes while fNIRS remained about `0.4737`;
word-generation EEG was only `0.0285`--`0.0347` while fNIRS was
`0.4251`--`0.4252`. SSM-SELF did not beat NATIVE, SSM-JOINT provided no useful
upper-bound gain, and low-weight XPRED did not provide a stable repair. The
longer one-seed motor-imagery probe made the EEG endpoint worse (`-0.2610`,
with fNIRS `0.8505`) and is not a multi-seed full-budget estimate. The
condition-by-time mean was a useful secondary trial-residual baseline, but it
was too strong to serve as the only admission gate. The v1 private decoder
reconstructed an observation residual rather than complete raw/high-rate
modality information; in NATIVE that residual is identically zero.

These facts rule out the tested v1 implementation as evidence for a learned
coupling mechanism or SSM superiority. They do **not** rule out separately
testing a modality-specific dynamic target, an observation--source split,
an independent vocabulary, or a lagged grammar. The v1 gate consequently
deferred K16/q0/q1 work; it did not select or preserve a successor architecture.

## Observation–source candidate family (unimplemented)

The following path is a design-space sketch. Each branch is optional and
replaceable; the sketch is not the project's name or a method contract:

```text
X_m  -- modality-specific dynamic teacher T_m -->
      (O~_m(t), Sigma_m(t), epsilon_m(t), masks)
       |                         |
       v                         v
  source encoder S_m        observation branch O_m^res
       |                         |
       v                         v
   fine Q_m -> coarse A_m     raw/masked or feature preservation
       |
       +--> endpoint-aligned lag grammar G_tau
                    |
                    v
       optional conditional-contribution diagnostics
```

The candidate menu and its safety boundaries are:

1. **Continuous, aligned supervision candidate.** EEG may be converted from the 200 Hz
   waveform to channel-preserving band-envelope trajectories (the primary
   candidate is baseline-relative log envelope/ERD--ERS, with absolute
   energy only as an auxiliary coordinate), then represented at a common
   10 Hz grid. fNIRS remains a continuous 10 Hz HbO/HbR trajectory. The
   teacher runs on this continuous axis before any 2 s patch/token operation.
   NATIVE/direct targets remain comparators, and the v1 patch-flattened target
   is not silently relabelled.
2. **Optional teacher residual outputs.** A candidate teacher may emit,
   for each modality $m$, a posterior trajectory mean $\widetilde O_m(t)$, predictive
   uncertainty $\Sigma_m(t)$ (at minimum a non-negative diagonal
   standard deviation with a recorded covariance convention), and the
   supported innovation $\epsilon_m(t)=O_m(t)-\widetilde O_m(t)$.
   The uncertainty may weight the source target; the innovation may supervise
   the observation branch and is not silently treated as missing data.
3. **Self versus privileged joint Croce candidate.** One candidate is a
   label-blind, modality-specific (self) teacher. A privileged joint
   candidate may fit the adaptive five-state Croce/Balloon equations on
   aligned EEG+fNIRS in the fit partition, but it is an offline training or
   ablation teacher and is never an inference input. The E0 sign-calibrated
   Croce implementation is accepted only for offline privileged development
   supervision; the later population-frozen R1-P qualification was rejected.
   The existing Croce algorithm/parameter bounds are therefore a versioned
   candidate to revalidate, not a protected-qualified teacher, ground truth,
   causal estimator, unique parameterization, or main self teacher. The
   accepted adaptive implementation is
   [adaptive_neurovascular_ssm.py](../src/inference/adaptive_neurovascular_ssm.py);
   its E0 status is recorded in the
   [E0 admission decision](physiology_semantic_tokenizer/analysis/E0_V3_ADAPTIVE_TEACHER_ADMISSION_DECISION.md)
   (with the sign-calibration record in
   [the E0 acceptance report](physiology_semantic_tokenizer/analysis/20260724_E0_SIGN_CALIBRATED_PHYSICAL_TEACHER_ACCEPTANCE.md)).
   The legacy particle-filter lane under croce_validation/ is an independent
   simulation/real-data audit lane with an inconclusive decision; it is not a
   qualified future teacher.
   The accepted adaptive runner is a useful continuous-window joint baseline:
   it runs a five-state RTS model on a 20 s/200-point 10 Hz window using an
   EEG-PCA neural driver and one HbO/HbR pair. It is not a multi-channel self
   target; such a candidate could retain EEG band-envelope coordinates (for
   example `[B,200,18]`) and compare low-rank dimensions such as `{4,8,16}`.
   The existing runner remains a baseline/ablation.
4. **Independent-path candidate.** When testing modality-specific inference,
   EEG and fNIRS source paths and observation
   paths receive only their own modality before quantization. $Q_E$ and
   $Q_F$ are independent vocabularies; equal numeric IDs have no shared
   semantics. Any cross-modal relation is learned only after the two state
   paths have produced their own tokens.
5. **Optional fine-to-coarse hierarchy.** A searched fine vocabulary
   $Z_m^f=Q_m(S_m)$ is mapped by a declared, fit-only aggregation
   $Z_m^c=A_m(Z_m^f)$ to a smaller physiological meta-vocabulary. Fine
   capacity serves downstream performance; the coarse vocabulary serves
   support, stability, and readable coupling maps. `K=16` is not a frozen
   v2 requirement. Candidate fine (K_f) values are open (for example
   `{16,32,64,128}`) and coarse (K_c) values are open (for example
   `{8,12,16,24}`). The aggregation may use posterior/expected embeddings,
   decoded-trajectory similarity, support, prototype stability, transition
   similarity, and grammar utility, but only under a fit/selection-only
   Pareto rule.
6. **Optional endpoint-aligned grammar.** One candidate coupling object is
   $G_\tau=P(Z_F^c(t+\tau)\mid Z_E^c(t),H_F,C)$, with an explicit
   endpoint-aligned mask and declared lag set. Same-position negative masks
   are forbidden. A grammar may be predictive, sparse, or otherwise learned;
   it is not automatically scientific evidence.

### Optional conditional-contribution probe

A development-only diagnostic may report three ablatable task contributions:

```text
observation  per-modality observation-branch endpoints;
source       incremental proper score from source marginals after conditioning
             on the observation baseline;
interaction  incremental proper score from an endpoint-aligned grammar beyond
             observation + source marginals.
```

The corresponding downstream head may be written

$$
\widehat Y=f_{observation}(O_E^{res},O_F^{res})+f_{source}(Z_E^f,Z_F^f)+f_{interaction}(G_\tau).
$$

These are conditional predictive increments, not identified components of an
information decomposition. The probe may be replaced or removed before method
selection. If it reaches evaluation, use task-appropriate held-out proper
scores, subject-level resampling, and explicit matched, deranged,
circular-shift, and history/phase nulls.

### Learned grammar versus held-out evidence

If tested, a coarse grammar may use a coupling-aware loss and a fit-selection map
quality objective (support, seed/fold stability, null separation,
concentration, lag localization, and a posterior-entropy penalty). This is
**learned grammar**. It can choose a Pareto point among downstream score,
coupling score, and map quality without looking at held-out rows.

One auditable fit/selection objective is

$$
J_{map}=\lambda_1S_{support}+\lambda_2S_{stability}+\lambda_3S_{null}
 +\lambda_4S_{concentration}+\lambda_5S_{lag}-\lambda_6S_{diffuse},
$$

where each term is computed without held-out rows. The formula itself is only a
candidate; a selected estimator's exact weights and capacity must be
preregistered before held-out evaluation.

**Empirical coupling evidence** is computed only after selecting the method and
preregistering the estimator, split, nulls, and stopping rule, then
recomputing held-out proper-score increments and matched-minus-null maps. A
pretty or sparse learned map, code usage, reconstruction, or training loss is
not evidence by itself. This distinction permits the grammar to be optimized
for readability without turning a training artifact into a discovery claim.

### Candidate gates and openness

The former rule requiring every non-privileged task/seed to have positive EEG
and fNIRS condition-time delta-R² is retired as the sole VQ admission gate.
Evaluation is staged:

1. basic learnability against a time-only/global mean baseline;
2. secondary trial-specific residual performance beyond the
   condition-by-time template;
3. observation, source-marginal, observation+source, and full-grammar task utility;
4. teacher increment (reliability, downstream score, prototype stability, and
   cross-seed stability) relative to NATIVE.

No listed model component is fixed method identity: observation--source
branches, teacher family, independent paths, codebooks, token hierarchy,
grammar, and conditional-contribution probes all remain replaceable. The fixed
boundaries are data provenance, modality/input ownership declared per
comparison, fit/selection versus held-out separation, and no protected access
without its owning protocol. Only a candidate selected after development may
receive a versioned implementation and evaluation contract.

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
