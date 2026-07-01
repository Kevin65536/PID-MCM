# Tokenizer Coupling Responsibility Boundary

> **Status (2026-07-01):** The separation advocated here is now formalized in the approved [physiology-semantic target architecture](../../../physiology_semantic_tokenizer/02_TARGET_ARCHITECTURE.md). The primary tokenizer uses independent modality inference; EEG-sequence-to-fNIRS-distribution coupling is evaluated after tokenizer freeze. This file remains the concise responsibility rationale.

## Current Position

The source/observation tokenizer is treated as a discrete physiological representation interface. It is not yet treated as a solved EEG-fNIRS fusion model.

The tokenizer stage should provide:

- stable source/observation reconstruction,
- discrete source tokens that retain physiological state information,
- controlled codebook usage without excessive dead codes,
- no material increase in subject, task, event-phase, or position leakage.

Historical operational default for the audited source/observation runs:

- the audited X3 run has `K=128` and effective EEG/fNIRS source vector dimension `D=48`; its generic `codebook_dim=128` field is shadowed by modality-specific fields,
- observation codebook dims remain branch-specific and are not changed by this default,
- the K128 vector-dim sweep supports `D=128` as the capacity default, but does not show that hard-token cross-modal coupling is solved.

The tokenizer stage should not currently require:

- a task-aware coupling tensor as a default training prior,
- a source-aware or phase-aware manually specified coupling model,
- globally stable EEG-to-fNIRS token predictability across all tasks and datasets.

## Coupling Interpretation

The global coupling tensor is useful as an interpretable diagnostic baseline for `P(fNIRS token | EEG token, lag)`. Current results show that this global view mixes incompatible task/dataset structures. Local residual coupling did not improve hard-token cross-modal predictability in the completed experiments.

Task-aware, source-aware, phase-aware, and position-aware analyses remain diagnostic controls. They can expose upper bounds and nuisance explanations, but they should not be promoted to the main tokenizer architecture without evidence that they improve transfer without memorizing dataset/task marginals.

## What Moves To Token-Sequence Pretraining

The token-sequence pretraining stage is the appropriate place for:

- masked token modeling,
- EEG-to-fNIRS and fNIRS-to-EEG predictive objectives,
- label-free context or mode discovery,
- modeling different coupling patterns across tasks without hand-written task labels,
- downstream label prediction from token sequences.

The preferred next direction is therefore:

1. keep tokenizer coupling-free unless a loss improves nuisance-controlled token semantics,
2. evaluate whether the tokenizer preserves enough information for downstream token models,
3. move flexible cross-modal relationship modeling into token-sequence pretraining.
