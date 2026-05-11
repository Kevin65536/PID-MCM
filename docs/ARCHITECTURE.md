# Current Architecture: Source/Observation Tokenizer

> **Semantics version**: `s2_source_observation_v1`
> **Last updated**: 2026-05-11
> **Current phase**: Phase 1 (Structural Migration) — Complete and Gate1-stable
> **Active phase**: Phase 2 (HRF Source Target) — Next implementation target from locked baseline
> **Config note**: the diagrams below show the nominal source/observation topology; the locked Phase 1 handoff baseline uses uniform32 codebooks and sets `source_coupling_loss` to `0.0`
> **Mainline class**: `SourceObservationLaBraMVQNSP` in [factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py)
> **Changelog**: [architecture_changelog/INDEX.md](architecture_changelog/INDEX.md)

---

## 1. Component Architecture

```mermaid
graph TB
    subgraph Inputs
        EEG[EEG Signal<br/>B x 30ch x 2000]
        FNIRS[fNIRS Signal<br/>B x 36ch x 100]
    end

    subgraph PatchEmbedding
        E_PE[MultiChannelPatchEmbedding<br/>patch=400 → 5 patches]
        F_PE[MultiChannelPatchEmbedding<br/>patch=20 → 5 patches]
    end

    subgraph Encoders
        E_ENC[TransformerEncoder<br/>d=256 depth=8 heads=8]
        F_ENC[TransformerEncoder<br/>d=160 depth=6 heads=4]
    end

    subgraph Projection["Projection Heads (4×)"]
        E_SP[eeg_source_proj<br/>256→48]
        E_OP[eeg_observation_proj<br/>256→64]
        F_SP[fnirs_source_proj<br/>160→48]
        F_OP[fnirs_observation_proj<br/>160→48]
    end

    subgraph Quantizers["Quantizers (4× NormEMAVectorQuantizer)"]
        E_SQ[eeg_source_quantizer<br/>K=128 D=48]
        F_SQ[fnirs_source_quantizer<br/>K=128 D=48]
        E_OQ[eeg_observation_quantizer<br/>K=256 D=64]
        F_OQ[fnirs_observation_quantizer<br/>K=128 D=48]
    end

    subgraph Coupling["Cross-Modal Coupling"]
        COUP_LOGITS[coupling_logits<br/>Parameter: n_lags x K_src x K_src]
        COUP_LOSS[source_coupling_loss<br/>KL div on token distributions]
    end

    subgraph DecodePrep["Decode Input Projection"]
        E_DP[eeg_decode_input_proj<br/>48+64=112 → 256]
        F_DP[fnirs_decode_input_proj<br/>48+48=96 → 160]
    end

    subgraph Decoders
        E_DEC[TransformerDecoder<br/>d=256 depth=4 heads=8]
        F_DEC[TransformerDecoder<br/>d=160 depth=3 heads=4]
    end

    subgraph OutputHeads["Output Heads (4×)"]
        E_AH[eeg_amplitude_head<br/>256 → 30x201]
        E_PH[eeg_phase_head<br/>256 → 30x201]
        F_AH[fnirs_amplitude_head<br/>160 → 36x11]
        F_PH[fnirs_phase_head<br/>160 → 36x11]
    end

    subgraph Reconstruction
        E_REC[EEG Reconstruction<br/>Freq→Time ISTFT]
        F_REC[fNIRS Reconstruction<br/>Freq→Time ISTFT]
    end

    EEG --> E_PE --> E_ENC
    FNIRS --> F_PE --> F_ENC

    E_ENC --> E_SP & E_OP
    F_ENC --> F_SP & F_OP

    E_SP --> E_SQ
    F_SP --> F_SQ
    E_OP --> E_OQ
    F_OP --> F_OQ

    E_SQ --> COUP_LOGITS
    F_SQ --> COUP_LOGITS
    COUP_LOGITS --> COUP_LOSS

    E_SQ --> E_DP
    E_OQ --> E_DP
    F_SQ --> F_DP
    F_OQ --> F_DP

    E_DP --> E_DEC --> E_AH & E_PH
    F_DP --> F_DEC --> F_AH & F_PH

    E_AH & E_PH --> E_REC
    F_AH & F_PH --> F_REC

    style Coupling fill:#e1f5fe
    style Quantizers fill:#fff3e0
    style Projection fill:#f3e5f5
```

## 2. Data Flow (Forward Pass)

```mermaid
sequenceDiagram
    participant EEG as EEG [B,30,2000]
    participant fNIRS as fNIRS [B,36,100]
    participant Patch as PatchEmbedding
    participant Enc as TransformerEncoder
    participant Proj as Projection Heads
    participant Quant as Quantizers (4×)
    participant Coup as Coupling Matrix
    participant Dec as Decoder
    participant Loss as Loss Computation

    EEG->>Patch: split_to_patches (400 stride)
    fNIRS->>Patch: split_to_patches (20 stride)
    Patch->>Enc: embeddings [B,5,D]
    Enc->>Proj: encoded [B,5,D]

    Proj->>Quant: eeg_source [B,5,48]
    Proj->>Quant: fnirs_source [B,5,48]
    Proj->>Quant: eeg_observation [B,5,64]
    Proj->>Quant: fnirs_observation [B,5,48]

    Quant-->>Coup: eeg_source_probs [B,5,128]
    Quant-->>Coup: fnirs_source_probs [B,5,128]
    Coup->>Loss: source_coupling_loss (KL div)

    Quant->>Dec: source_q + observation_q concat
    Dec->>Loss: Amplitude prediction + ISTFT time reconstruction
    Loss->>Loss: rec_loss = amp + time
    Loss->>Loss: vq_loss = commitment losses
    Loss->>Loss: orthogonality_loss (source ⊥ obs)
    Loss->>Loss: codebook_balance_loss (entropy)
    Loss->>Loss: total = rec + vq + coupling + balance + ortho
```

## 3. Loss Composition

```mermaid
graph LR
    TOTAL[total_loss] --> REC[reconstruction]
    TOTAL --> VQ[vq_loss]
    TOTAL --> COUP[source_coupling_loss]
    TOTAL --> BAL[codebook_balance_loss]
    TOTAL --> ORTHO[orthogonality_loss]

    REC --> E_REC[eeg_rec_loss<br/>amp + time]
    REC --> F_REC[fnirs_rec_loss<br/>amp + time]

    VQ --> VQ_S[vq_source_loss<br/>eeg + fnirs]
    VQ --> VQ_O[vq_observation_loss<br/>eeg + fnirs]

    COUP --> COUP_FWD[EEG → fNIRS KL]
    COUP --> COUP_REV[fNIRS → EEG KL<br/>bidirectional]

    BAL --> BAL_S[source_balance_loss<br/>entropy based]
    BAL --> BAL_O[observation_balance_loss<br/>entropy based]

    ORTHO --> O_E[orthogonality_loss<br/>eeg_source ⊥ eeg_obs]
    ORTHO --> O_F[orthogonality_loss<br/>fnirs_source ⊥ fnirs_obs]
```

### Default Structural-Migration Loss Weights

The locked Gate1 handoff baseline overrides `source_coupling_loss` to `0.0` and raises `codebook_balance_loss` to `0.08`; see the locked handoff section below.

| Loss Term | Weight | Purpose |
|-----------|--------|---------|
| `eeg_rec_loss` | 1.0 | EEG full reconstruction (amp 1.0 + time 0.9; no explicit phase supervision) |
| `fnirs_rec_loss` | 1.0 | fNIRS full reconstruction (amp 1.0 + time 1.0; no explicit phase supervision) |
| `vq_loss` | 1.0 | Commitment + EMA codebook loss (all 4 quantizers) |
| `source_coupling_loss` | 0.07 | KL divergence: predicted vs actual source token distributions |
| `codebook_balance_loss` | 0.02 | Entropy-based dead-code prevention (all 4 quantizers) |
| `orthogonality_loss` | 0.01 | Cosine similarity penalty between source and observation within each modality |

## 4. Component Catalog

### Core Tokenizer

| File | Role |
|------|------|
| [src/tokenizers/factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py) | **Mainline tokenizer**: `SourceObservationLaBraMVQNSP` — encoders, projectors, 4 quantizers, coupling, decoders |
| [src/tokenizers/labram_vqnsp.py](../src/tokenizers/labram_vqnsp.py) | **Shared components**: `NormEMAVectorQuantizer`, `TransformerEncoder`, `TransformerDecoder`, `l2norm`, `MultiChannelPatchEmbedding` |
| [src/tokenizers/base.py](../src/tokenizers/base.py) | Abstract `BaseTokenizer` class |
| [src/tokenizers/registry.py](../src/tokenizers/registry.py) | Tokenizer factory: config → constructor mapping, `StandardizedOutput` interface |
| [src/tokenizers/__init__.py](../src/tokenizers/__init__.py) | Tokenizer exports and registration |

### Loss Functions

| File | Role |
|------|------|
| [src/losses/multimodal_tokenizer.py](../src/losses/multimodal_tokenizer.py) | `coupling_kl_loss`, `batch_usage_entropy_loss`, `orthogonality_loss`, `align_pair`, `symmetric_kl_from_logits` |
| [src/losses/reconstruction.py](../src/losses/reconstruction.py) | Multi-STFT and time-domain reconstruction losses |

### Analysis & Visualization

| File | Role |
|------|------|
| [src/visualization/tokenizer_analysis_suite.py](../src/visualization/tokenizer_analysis_suite.py) | **Standardized analysis entry point** — generates full tokenizer report |
| [src/visualization/source_observation_analysis.py](../src/visualization/source_observation_analysis.py) | Source/observation alignment analysis, scorecard generation, Gate 1-4 metrics |
| [src/visualization/tensorboard_logger.py](../src/visualization/tensorboard_logger.py) | TensorBoard metric logging during training |

### Training

| File | Role |
|------|------|
| [experiments/scripts/train_source_observation_tokenizer.py](../experiments/scripts/train_source_observation_tokenizer.py) | **Main training script** — loads config, creates model/dataloaders, runs training loop |
| [experiments/scripts/launch_training_nohup.sh](../experiments/scripts/launch_training_nohup.sh) | Standardized launcher for training runs |

### Configs

| Directory | Purpose |
|-----------|---------|
| [experiments/configs/base.yaml](../experiments/configs/base.yaml) | Dataset, preprocessing, and hardware defaults |
| [experiments/configs/source_observation/phase1/](../experiments/configs/source_observation/phase1/) | Phase 1 configs, including the locked Gate1 baseline and best-current alias |
| [experiments/configs/source_observation/phase2/](../experiments/configs/source_observation/phase2/) | Phase 2 HRF Source Target configs (ready) |
| [experiments/configs/source_observation/phase3/](../experiments/configs/source_observation/phase3/) | Phase 3 Concentration Prior configs (ready) |
| [experiments/configs/source_observation/mechanism_a/](../experiments/configs/source_observation/mechanism_a/) | Mechanism A Smoothness configs (ready) |
| [experiments/configs/source_observation/mechanism_c/](../experiments/configs/source_observation/mechanism_c/) | Mechanism C Asymmetry configs (ready) |

## 5. Quantizer Summary

| Quantizer | Codebook Size | Embedding Dim | Semantics |
|-----------|---------------|---------------|-----------|
| `eeg_source_quantizer` | K=128 | D=48 | EEG neurovascular coupling state |
| `fnirs_source_quantizer` | K=128 | D=48 | fNIRS neurovascular coupling state |
| `eeg_observation_quantizer` | K=256 | D=64 | EEG modality-specific encoding debt |
| `fnirs_observation_quantizer` | K=128 | D=48 | fNIRS modality-specific encoding debt |

All quantizers use EMA updates, kmeans initialization, dead code revival, and cosine-similarity-based assignment (l2-normalized).

The locked Phase 1 Gate1 handoff baseline temporarily uses uniform32 codebooks across all four quantizers to preserve the validated health profile.

## 6. Coupling Mechanism

The coupling matrix `coupling_logits` is an `[n_lags, K_src, K_src]` learned parameter.

**Forward pass** (for each lag):
1. Align EEG and fNIRS source token distributions with lag offset
2. `EEG_probs → [coupling_logits[lag]] → predicted_fNIRS_probs`
3. KL divergence between predicted and actual fNIRS token distributions
4. When bidirectional: also compute `fNIRS → predicted_EEG` and average

**Selection**: Choose lag with minimum coupling loss (`alignment_selection='min'`).

**Current lags**: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`

## 7. Phase Status

| Phase | Name | Status | Key Deliverable |
|-------|------|--------|-----------------|
| Phase 1 | Structural Migration | ✅ Complete | Source/Observation tokenizer running, shared/private removed |
| Phase 2 | HRF Source Target | 🔜 Ready | Double-gamma HRF kernel, fNIRS source target from EEG |
| Phase 2A | Coupling-Aware Quantization | 📋 Planned | Coupling prior guides fNIRS source argmin |
| Phase 3 | Concentration Prior | 📋 Planned | Row entropy regularization on coupling matrix |
| Mechanism A | Coupling Smoothness | 📋 Planned | Local smoothness prior on coupling rows |
| Mechanism C | Causal Asymmetry | 📋 Planned | Independent fwd/rev coupling parameterization |

### Locked Phase1 Handoff

| Artifact | Role |
|----------|------|
| [experiments/configs/source_observation/phase1/gate1_best_current.yaml](../experiments/configs/source_observation/phase1/gate1_best_current.yaml) | Current best Gate1-stable baseline alias |
| [experiments/configs/source_observation/phase1/gate1_baseline_locked_bs128.yaml](../experiments/configs/source_observation/phase1/gate1_baseline_locked_bs128.yaml) | Clean reusable Gate1 baseline outside the historical tuning chain |
| [experiments/runs/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718](../experiments/runs/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718) | Best recorded Gate1 pass; best epoch 278, val_loss 1.6395270029703777 |
| [experiments/runs/archive/source_observation_phase1_gate1_stabilization_20260511/manifest.json](../experiments/runs/archive/source_observation_phase1_gate1_stabilization_20260511/manifest.json) | Formal archive bundle for the full Phase 1 Gate1 search surface |

## 8. Related Documents

| Document | Role |
|----------|------|
| [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) | Implementation order, file migration scope, validation gates |
| [PHYSIOLOGICAL_COUPLING_PLAN.md](PHYSIOLOGICAL_COUPLING_PLAN.md) | Mechanism motivation, math, physiological interpretation |
| [SEMANTIC_TOKEN_SCORECARD.md](SEMANTIC_TOKEN_SCORECARD.md) | 4-Gate evaluation framework |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | Formal experiment conclusions |
| [archive/logs/PHASE1_GATE1_STABILIZATION_20260511.md](archive/logs/PHASE1_GATE1_STABILIZATION_20260511.md) | Formal closure log for the Phase 1 Gate1 search bundle |
| [architecture_changelog/INDEX.md](architecture_changelog/INDEX.md) | Chronological architecture change records |
| [STANDARDIZATION_GUIDE.md](../STANDARDIZATION_GUIDE.md) | Naming conventions, run protocols, artifact standards |
