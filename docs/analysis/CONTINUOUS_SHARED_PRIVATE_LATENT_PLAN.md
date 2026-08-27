# Continuous shared/private latent experiment plan

_Protocol status: active exploratory plan; depends on the SSM reconstruction
reliability contract; current execution status is owned by
[`research_state/registry.json`](../../research_state/registry.json); 2026-08-19_

## Purpose and claim boundary

This experiment removes vector quantization for one complete development round
and asks whether paired EEG and fNIRS contain a usable continuous shared latent
under the current SSM shared-driver target. It also gives each modality an
independent continuous private branch so that raw reconstruction need not pass
through an information bottleneck.

The experiment tests sharedness relative to a development SSM proxy. A positive
result would show bilateral proxy observability and cross-modal latent
interchangeability under the registered controls; it would not make the SSM
trajectory ground truth, prove physiological identifiability, or authorize a
new VQ generation.

This is exploratory work. Single-Trial subjects 24–29 remain closed, and the
old R1-P/R2-D decisions remain frozen evidence rather than mutable gates.

## Data and execution matrix

Four task-specific models are trained independently:

1. Single-Trial mental arithmetic;
2. Single-Trial motor imagery;
3. Simultaneous word generation;
4. Simultaneous n-back.

The task cells, 20 s windows, subject splits, dependency groups, raw-view
selection, and SSM cross-fit identities are inherited exactly from
`SSM_RECONSTRUCTION_RELIABILITY_PLAN.md`. Each training sample joins measured
EEG/fNIRS and its out-of-fold SSM trajectory by canonical sample identity, not
array position. A target may select a previously frozen raw view but may not
supply either measured modality.

Each task uses seeds `20260819`, `20260820`, and `20260821`. Seed is algorithmic
variation, not a biological replicate. Full validation statistics average the
three seed-specific subject estimates before resampling subjects.

## Model contract

The model has four modality-specific encoders and three decoder roles:

- an EEG shared encoder and an fNIRS shared encoder, each producing ten
  64-dimensional continuous tokens;
- one modality-agnostic trajectory decoder that maps either shared latent to
  the complete SSM shared-driver patch;
- independent EEG/private and fNIRS/private encoders with dimensions 64 and 32;
- modality-specific raw decoders receiving
  `[stop_gradient(shared), private]`.

The private encoders do not reuse trainable shared-encoder features. Raw
reconstruction gradients may update only the private encoder and corresponding
raw decoder. A parameter-level gradient allowlist is part of the software
contract.

The model contains no vector quantizer, codebook, hard ID, soft assignment,
commitment loss, occupancy target, revival rule, or modality identifier in the
shared decoder. Direct EEG–fNIRS latent alignment, adversarial separation, and
orthogonality penalties are omitted in v1 so that the validation does not force
the desired shared/private geometry in advance.

## Objectives and model selection

For each modality, the shared objective is masked trajectory error against the
same SSM joint-driver target. The two modality losses receive equal weight. The
private objective is masked raw-patch MSE in the train-fold normalized
measurement coordinate. Because its gradient is isolated, its scalar weight
cannot change the shared encoder.

The initial optimizer contract matches the completed R2-D continuous model:
AdamW, learning rate `3e-4`, weight decay `0.01`, batch size 32, at most 80
epochs, gradient norm cap 1.0, and early-stopping patience 15. No
hyperparameter search is conducted. Checkpoint selection uses only validation
equal-modality masked SSM trajectory loss.

## Primary sharedness endpoints

### Bilateral SSM-target observability

For EEG shared and fNIRS shared separately, compute subject-level
\(\Delta R^2\) against a train-only condition-by-relative-time mean baseline.
The target, baseline, predictions, and masks are accumulated within subject
before taking the ratio.

### Cross-modal shared swap

With model parameters frozen:

- reconstruct EEG from matched fNIRS shared + EEG private;
- reconstruct fNIRS from matched EEG shared + fNIRS private.

The null replaces the shared latent with a trial-deranged latent from the same
subject, condition, and token time wherever a non-identity derangement exists.
The endpoint is

\[
\Delta R^2_{\mathrm{swap}}
=1-\frac{\mathrm{SSE}_{\mathrm{matched}}}
        {\mathrm{SSE}_{\mathrm{deranged}}}.
\]

All masks and denominators are fixed before viewing the matched result.

### Secondary separation diagnostics

Frozen, capacity-matched ridge probes compare shared and private latents for
SSM-trajectory decoding under the same subject split. Private leakage is
reported as observed; failure to find a difference is not called equivalence.
Latent rank, variance, and shared/private cross-covariance are descriptive
diagnostics, not replacement success criteria.

## Statistical decision rule

The primary family contains 16 cells:

```text
4 tasks × (EEG target ΔR² + fNIRS target ΔR²
           + fNIRS→EEG swap ΔR² + EEG→fNIRS swap ΔR²)
```

Use 10,000 subject-block bootstrap draws and a max-stat simultaneous 95%
interval while preserving task pairing within each dataset. Seed-specific
estimates are averaged within subject first.

Continuous sharedness is supported under the strict exploratory rule only if
all 16 simultaneous lower bounds are greater than zero. If any cell fails, the
overall result is “not supported under the strict rule”; positive task or
modality cells remain reportable as partial evidence, and a pooled average
cannot rescue the failed family.

The outcome does not change the protected-data or VQ state. Any later VQ
proposal requires a separate decision after this continuous result and the SSM
reliability profile are reviewed together.

## Artifact and reporting contract

Each task/seed run writes checkpoints, resolved configuration, loss curves,
validation predictions, masks, latent exports, and source/split/target hashes
below the existing physiology-semantic run root. Analysis writes:

- per-seed and seed-averaged subject endpoint tables;
- the 16-cell simultaneous interval table and bootstrap draws;
- matched/deranged swap tables with the exact derangement registry;
- private-leakage and latent-rank diagnostics;
- manifest-bound SVG/PNG figures and source tables;
- a summary containing negative, null, failed, and inconclusive cells as well
  as positive cells.

Figures show subject observations and named subject-bootstrap intervals.
Teacher decoding and raw swap occupy separate panels; axes are not tuned to
create apparent agreement and missing values remain visible.

## Tests and readiness gates

Implementation is ready for measured training only after tests establish:

- exact model shapes for ten tokens and both raw patch geometries;
- absence of every quantizer/codebook surface in modules, outputs, losses, and
  manifests;
- raw-loss gradients are zero for all shared-encoder parameters and non-zero
  for the admitted private/raw parameters;
- the common trajectory decoder accepts either modality without a modality ID;
- matched cross-modal swap and trial derangement preserve masks and sample
  identity constraints;
- seed averaging occurs before subject bootstrap;
- a synthetic common-factor fixture can pass target and swap endpoints, while
  an independent-modalities fixture does not pass the strict family;
- dry-run serialization, checkpoint reload, atomic publication, and analysis
  recomputation all succeed.

The high-cost four-task, three-seed matrix is launched only after these tests
and per-task smoke runs pass.
