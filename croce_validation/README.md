# Croce-Style SSM Validation Workspace

This directory is intentionally independent from `experiments/`.

It validates a **modified Croce-style state-space model** for EEG-fNIRS neural source
estimation. The core departure from Croce 2017: r(t) has no endogenous dynamics
(no OU process, no random walk). It is proposed from EEG and weighted by fNIRS
likelihood in a particle filter.

## Scope

This workspace is only for:

1. paper-faithful Croce 2017 simulation (reference baseline),
2. local neural source estimation from real EEG-fNIRS data,
3. inverse-stability and identifiability checks,
4. event-locked physiological plausibility analysis,
5. source/observation target separation for downstream tokenizer training.

This workspace is not for:

1. tokenizer training itself,
2. cross-task extensions without prior validation,
3. global EEG-power proxy experiments,
4. full-channel scalar-state reductions that erase local source meaning.

## Design Documents

| Document | Purpose |
|----------|---------|
| [DESIGN.md](FAST_SLOW_DESIGN.md) | **Canonical design** — full mathematical specification |
| [CROCE2017_REAL_DATA_VALIDATION_PLAN.md](CROCE2017_REAL_DATA_VALIDATION_PLAN.md) | Validation plan, metrics, decision rules |

## Layout

```text
croce_validation/
  README.md
  FAST_SLOW_DESIGN.md              # canonical design
  CROCE2017_REAL_DATA_VALIDATION_PLAN.md
  scripts/
    run_local_neighborhood_solver_audit.py  # real-data solver (modified design)
  results/
```

## Conventions

1. Validation scripts live under `scripts/` and are version-controlled.
2. Generated outputs go to `results/<run_name>/` and remain untracked.
3. r(t) is local, signed, and free of endogenous dynamics.
4. Forward models are deterministic — no observation noise in the forward map.
5. Sources are at fNIRS channel positions. EEG forward uses either local (Case A) or whole-brain (Case B) lead fields.
6. Signal units, polarity conventions, and normalization choices must be recorded in every run manifest.
