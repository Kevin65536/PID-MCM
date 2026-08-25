# Croce-Style SSM Validation Workspace

This directory is intentionally independent from `experiments/`.

> **Lifecycle (2026-08-25): stopped / abandoned.** The legacy solver, audits,
> generated caches, and retained diagnostics are stopped evidence surfaces. The
> redesigned Synthetic Phase 1 and Real Phase 2 were never run and are abandoned.
> This directory is for reading and replaying retained evidence only; a future
> clean experiment flow must define its own owner and contract.

It records a **modified Croce-style state-space model** for EEG-fNIRS neural source
estimation. The recorded departure from Croce 2017 is that r(t) has no endogenous dynamics
(no OU process, no random walk). It is proposed from EEG and weighted by fNIRS
likelihood in a particle filter.

The legacy particle-filter (PF) lane is a stopped, independent exploratory lane.
Its result is inconclusive and must not be treated as a qualified teacher, ground
truth, causal estimate, or authorization for tokenizer training. Shared, self, and
direct supervision targets remain competing exploratory choices; no target relation
or information decomposition is frozen. The existing Croce 2017 model and its
modified design are retained as historical records, not as an active validation
plan or a new-run entry point.

## Scope

The stopped workspace records:

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

Existing downstream reuse is not authorized by these records and the legacy PF
result is not promoted into teacher qualification. Any future measured run must
belong to a newly defined protocol and explicit user authorization when it crosses
a protected boundary.

## Design Documents

| Document | Purpose |
|----------|---------|
| [DESIGN.md](DESIGN.md) | Stopped historical design — full mathematical specification |
| [CROCE2017_REAL_DATA_VALIDATION_PLAN.md](CROCE2017_REAL_DATA_VALIDATION_PLAN.md) | Frozen historical validation record, metrics, and decision rules |

## Layout

```text
croce_validation/
  README.md
  DESIGN.md                        # stopped historical design
  CROCE2017_REAL_DATA_VALIDATION_PLAN.md
  scripts/
    run_local_neighborhood_solver_audit.py  # stopped legacy solver (replay only)
  cache/                           # retained generated evidence (ignored)
  analysis/                        # retained audit reports
```

## Conventions

1. Validation scripts live under `scripts/` and are version-controlled; their
   associated experiment lane is stopped.
2. Retained generated outputs live under the existing `cache/` and `analysis/`
   evidence roots and remain untracked. No new output namespace is implied here.
3. r(t) is local, signed, and free of endogenous dynamics.
4. Forward models are deterministic — no observation noise in the forward map.
5. Sources are at fNIRS channel positions. EEG forward uses either local (Case A) or whole-brain (Case B) lead fields.
6. Signal units, polarity conventions, and normalization choices must be recorded in every run manifest.
