# Token Physiology Atlas

_Analysis contract and current development result, consolidated 2026-07-30_

Token Physiology Atlas describes measurement phenotypes associated with a
frozen tokenizer. It is an analysis surface, not a token-naming oracle and not
an experiment-admission mechanism. Equal token IDs across modalities or
checkpoints have no implied shared meaning.

## Current result

The latest Atlas is a completed **development-only Core run** on the retained
E2 T0 checkpoint:

[`token_physiology_atlas_standard_loader_core_20260730`](../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/runs/t0_seed20260719/analysis/token_physiology_atlas_standard_loader_core_20260730/)

Its manifest reports `status=complete`, `splits=[train,val]`, and
`protected_test_opened=false`.

| Split / modality | Active tokens | Support-qualified tokens |
| --- | ---: | ---: |
| train EEG | 102/128 | 82 |
| validation EEG | 93/128 | 43 |
| train fNIRS | 126/128 | 62 |
| validation fNIRS | 112/128 | 25 |

Train-to-validation matched phenotype cosine was `0.7247` for EEG and `0.9828`
for fNIRS. These are descriptive stability values among support-qualified
signatures; Core ran zero bootstrap iterations, so they are not uncertainty
statements.

Count-weighted posterior normalized entropy was approximately `0.989/0.988`
for train/validation EEG and `0.964/0.964` for train/validation fNIRS.
Assignments are therefore diffuse despite broad hard-token activity. Hard
assignments should not be described as sharp physiological states.

The next authorized Atlas step is the **Statistical tier** on this already
frozen development artifact. The Full coupling-null tier is conditional on a
specific preregistered question after the statistical audit. A new SD-SVQ/VQ
Atlas remains blocked with the main method; analysis of T0 cannot override
`do_not_enter_r2_p`.

## Claim boundary

The Atlas estimates token-conditioned properties of the tokenizer's
dimensionless `canonical_robust_sd` input patches. It may support statements
such as:

- a token has enough patch and subject support under a declared split;
- a supported token is enriched for a named standardized patch feature;
- a supported phenotype direction is or is not stable across train/validation;
- a result changes under hard versus soft assignment or nuisance controls.

It does not by itself support:

- physiological state names or absolute µV/HbO/HbR claims;
- mechanistic, causal, or clinical interpretations;
- equal semantics for equal EEG/fNIRS token IDs;
- cross-checkpoint ID matching without phenotype matching;
- coupling claims without a frozen endpoint and appropriate null;
- protected-test inspection or selection of a new tokenizer.

The statistical unit for profiles and intervals is the subject. Patch counts
define support but do not give patch-rich subjects extra inferential weight.

## Analysis tiers

| Tier | Contents | Use |
| --- | --- | --- |
| Core | support, hard/soft profiles, patch/channel distributions, assignment and nuisance diagnostics, sequence counts, compact figures | routine completed-checkpoint description |
| Statistical | Core plus subject bootstrap, grouped information ledger, train/validation signature uncertainty | next step for retained T0 |
| Full | Statistical plus frozen lag family and whole-window circular-shift null | conditional coupling question |

The CLI executes the selected tier synchronously. `bootstrap_mode`,
`coupling_null_mode`, and export modes in YAML describe orchestration policy;
they do not create a background job.

## Inputs and identity

Analysis consumes either deterministic checkpoint replay or a versioned token
export. Every patch keeps:

```text
checkpoint/export identity
split, modality, token/codebook identity
dataset, subject, record/session, task and label
window/patch position and real-signal masks
measurement-cache and feature-spec hashes
```

Measurement features are checkpoint-independent and may share a content-hash
cache. Assignment arrays are checkpoint-dependent and keep full posterior,
hard ID, expected/quantized/semantic embeddings, residuals, reconstruction
diagnostics, masks, and provenance. Raw patches are temporary replay inputs,
not long-term Atlas payload.

Protected test never opens merely because a split name is present in YAML.
It requires an independent CLI authorization and a frozen analysis contract;
ordinary train/validation analysis must not use it for support thresholds,
matching, ordering, or narrative selection.

## Feature surface

EEG uses mask-aware, channel-resolved two-second features:

- mean, standard deviation, RMS, slope, endpoint change, and line length;
- Hjorth activity, mobility, and complexity;
- log absolute and relative delta/theta/alpha/beta/low-gamma power;
- spectral entropy and peak frequency.

fNIRS uses two-second HbO/HbR local morphology:

- mean, median, standard deviation, RMS;
- slope, endpoint change, AUC, and derivative spike;
- within-patch HbO–HbR correlation.

These are standardized model-coordinate descriptors. Correlated quantities
such as RMS/activity/power or mean/median/AUC are not independent replications.
A two-second fNIRS patch cannot establish a full HRF shape or peak latency.

## Support and matching

Default support requires:

```text
assigned patch count >= 30
and subject count >= 5
```

Inactive, rare, insufficient-support, and missing estimates remain distinct.
Profiles first average within `subject × token`, then average subjects equally.
Hard and posterior-weighted soft profiles are both retained.

Token IDs are nominal. Train/validation or multi-seed comparison uses
support-gated phenotype matching over common named features. The matching
table records feature overlap, support, cosine, unmatched tokens, and any
subject-bootstrap interval. Static codebook proximity is a model diagnostic,
not a substitute for phenotype matching.

## Reading the figures

Read each split/modalality set in this order:

1. **support** — determine which tokens are interpretable;
2. **phenotype heatmap** — inspect subject-equal standardized enrichment;
3. **codebook PCA** — inspect whether the frozen embedding has a descriptive
   feature gradient;
4. **train/validation match** — verify direction and support;
5. **hard/soft and nuisance tables** — check boundary and confounding
   sensitivity;
6. **bootstrap/null output** — only when the selected tier generated it.

Heatmaps use a zero-centered symmetric scale, but each figure may have a
different limit. Color depth is not comparable until the colorbars are read.
Unsupported estimates use a separate hatch/marker, and codebook plots encode
effect direction redundantly with marker shape. A PCA axis is not a
physiological axis.

The current run retains a complete
[visualization reading guide](../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/runs/t0_seed20260719/analysis/token_physiology_atlas_standard_loader_core_20260730/VISUALIZATION_READING_GUIDE.md)
beside its figures.

## Output contract

An Atlas run atomically publishes:

- top-level manifest and summary;
- token support, profile, distribution, exemplar, assignment, nuisance,
  state-association, sequence, and lag tables;
- stability and information-ledger records;
- PNG/SVG figures as requested;
- per-figure source and manifest data.

The manifest records source hashes, split, tier, thresholds, units/transform,
software identity, protected-test state, wall time, artifact inventory, and
output hashes. Existing output directories are not overwritten unless
explicitly forced.

## Entrypoint

Defaults are versioned in
[`token_physiology_atlas.yaml`](../../experiments/configs/physiology_semantic_tokenizer/token_physiology_atlas.yaml).

Analyze existing exports:

```bash
.venv/bin/python experiments/scripts/analyze_token_physiology_atlas.py \
  --atlas-config experiments/configs/physiology_semantic_tokenizer/token_physiology_atlas.yaml \
  --export train=<exports>/train.npz \
  --export val=<exports>/val.npz \
  --output-dir <run>/analysis/token_physiology_atlas
```

Replay a frozen checkpoint:

```bash
.venv/bin/python experiments/scripts/analyze_token_physiology_atlas.py \
  --atlas-config experiments/configs/physiology_semantic_tokenizer/token_physiology_atlas.yaml \
  --checkpoint <run>/checkpoints/best.pt \
  --model-config <run>/config.yaml \
  --splits train,val \
  --measurement-cache-dir <shared-cache>/token_physiology_measurements \
  --tier statistical \
  --output-dir <new-output-dir>
```

Runtime argument names in `--help` are authoritative; YAML supplies versioned
statistical and output defaults.
