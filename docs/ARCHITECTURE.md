# Current Architecture: Source/Observation Tokenizer (Dual Decoder + Croce 2017 Physical Model)

> **Semantics version**: `s2_source_observation_v2_phase2b`
> **Last updated**: 2026-05-14
> **Current phase**: Phase 2B (Croce 2017 Physical Model + Coupling Structure Priors) — Architecture stabilized
> **Mainline class**: `SourceObservationLaBraMVQNSP` in [factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py)
> **Changelog**: [architecture_changelog/INDEX.md](architecture_changelog/INDEX.md)

---

## 1. Component Architecture (Current — Phase 2B Stabilized)

```mermaid
graph TB
    subgraph Inputs
        EEG["EEG Signal<br/>B x 30ch x 2000"]
        FNIRS["fNIRS Signal<br/>B x 36ch x 100"]
    end

    subgraph PatchEmbedding
        E_PE["MultiChannelPatchEmbedding<br/>patch=400 → 5 patches"]
        F_PE["MultiChannelPatchEmbedding<br/>patch=20 → 5 patches"]
    end

    subgraph Encoders
        E_ENC["TransformerEncoder<br/>d=256 depth=8 heads=8"]
        F_ENC["TransformerEncoder<br/>d=160 depth=6 heads=4"]
    end

    subgraph Projection["Projection Heads (4×)"]
        E_SP["eeg_source_proj<br/>256→48"]
        E_OP["eeg_observation_proj<br/>256→64"]
        F_SP["fnirs_source_proj<br/>160→48"]
        F_OP["fnirs_observation_proj<br/>160→48"]
    end

    subgraph Quantizers["Quantizers (4× NormEMAVectorQuantizer)"]
        E_SQ["eeg_source_quantizer<br/>K=32 D=48"]
        F_SQ["fnirs_source_quantizer<br/>K=32 D=48"]
        E_OQ["eeg_observation_quantizer<br/>K=64 D=64"]
        F_OQ["fnirs_observation_quantizer<br/>K=64 D=48"]
    end

    subgraph Coupling["Cross-Modal Coupling"]
        COUP_LOGITS["coupling_logits<br/>Parameter: n_lags x K_src x K_src"]
        COUP_LOSS["lag_focus_loss<br/>+ joint_smoothness_loss"]
    end

    subgraph SourceDecoders["Source Decoders (2×)"]
        E_SD["eeg_source_decoder<br/>d=256 depth=4 heads=8"]
        F_SD["fnirs_source_decoder<br/>d=160 depth=3 heads=4"]
    end

    subgraph ObsDecoders["Observation Decoders (2×)"]
        E_OD["eeg_observation_decoder<br/>d=256 depth=4 heads=8"]
        F_OD["fnirs_observation_decoder<br/>d=160 depth=3 heads=4"]
    end

    subgraph OutputHeads["Output Heads (8×)"]
        E_SA["eeg_src_amp_head<br/>256→30x201"]
        E_SPH["eeg_src_phase_head<br/>256→30x201"]
        E_OA["eeg_obs_amp_head<br/>256→30x201"]
        E_OPH["eeg_obs_phase_head<br/>256→30x201"]
        F_SA["fnirs_src_amp_head<br/>160→36x11"]
        F_SPH["fnirs_src_phase_head<br/>160→36x11"]
        F_OA["fnirs_obs_amp_head<br/>160→36x11"]
        F_OPH["fnirs_obs_phase_head<br/>160→36x11"]
    end

    subgraph Reconstruction
        E_SUM["Σ source + observation<br/>= eeg_full_recon"]
        F_SUM["Σ source + observation<br/>= fnirs_full_recon"]
    end

    subgraph Targets["Target Construction (Croce 2017 physical model)"]
        SS["shared neural state<br/>AR-smoothed EEG power @ fNIRS rate"]
        E_ST["EEG source target<br/>signed RMS carrier @ 200Hz"]
        F_ST["fNIRS source target<br/>HRF(shared_state)"]
        E_OT["EEG obs target<br/>= original - source_target"]
        F_OT["fNIRS obs target<br/>= original - source_target"]
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

    E_SQ --> E_SD --> E_SA & E_SPH
    E_OQ --> E_OD --> E_OA & E_OPH
    F_SQ --> F_SD --> F_SA & F_SPH
    F_OQ --> F_OD --> F_OA & F_OPH

    E_SA & E_SPH --> E_SUM
    E_OA & E_OPH --> E_SUM
    F_SA & F_SPH --> F_SUM
    F_OA & F_OPH --> F_SUM

    EEG --> E_ST
    EEG --> E_OT
    FNIRS --> F_OT
    E_ST --> F_ST

    style Coupling fill:#e1f5fe
    style Quantizers fill:#fff3e0
    style Projection fill:#f3e5f5
    style SourceDecoders fill:#e8f5e9
    style ObsDecoders fill:#fce4ec
    style Targets fill:#fff9c4
```


**Key architectural change from Phase 1/2**: Single shared decoder per modality → dual independent decoders (source + observation). Full reconstruction = source_recon + observation_recon (additive in signal space).

## 2. Data Flow (Forward Pass)

```mermaid
sequenceDiagram
    participant EEG as EEG [B,30,2000]
    participant fNIRS as fNIRS [B,36,100]
    participant Enc as Encoders
    participant Proj as Projection Heads
    participant Quant as Quantizers (4×)
    participant Coup as Coupling Matrix
    participant DecS as Source Decoders
    participant DecO as Observation Decoders
    participant Target as Target Construction
    participant Loss as Loss Computation

    EEG->>Target: compute source target (power envelope @ full res)
    fNIRS->>Target: compute source target (HRF of power envelope)
    Target->>Target: obs_target = original - source_target

    EEG->>Enc: EEG encoder
    fNIRS->>Enc: fNIRS encoder
    Enc->>Proj: 4 projection heads
    Proj->>Quant: 4 quantizers (straight-through)

    Quant-->>Coup: eeg_source_probs, fnirs_source_probs
    Coup->>Loss: lag_focus_loss + joint_smoothness_loss

    Quant->>DecS: source_q → source decoder → source_recon
    Quant->>DecO: obs_q → observation decoder → obs_recon

    DecS->>Loss: source_target_loss (source_recon vs source_target)
    DecO->>Loss: observation_loss (obs_recon vs obs_target)
    DecS->>Loss: full_recon_loss ((source_recon + obs_recon) vs original)
    DecO->>Loss: (same, additive)

    Loss->>Loss: vq_loss = commitment (× quantization_strength)
    Loss->>Loss: orthogonality_loss (source ⊥ obs)
    Loss->>Loss: codebook_balance_loss (straight-through hard assign, per-branch temps)
    Loss->>Loss: total = rec + source_target + obs_target + vq + coupling + balance + ortho
```

## 3. Loss Composition (Phase 2A Target)

```mermaid
graph LR
    TOTAL[total_loss] --> REC[full reconstruction]
    TOTAL --> ST[source_target_loss<br/>fNIRS + EEG]
    TOTAL --> OT[observation_loss<br/>fNIRS + EEG]
    TOTAL --> VQ[vq_loss<br/>× quantization_strength]
    TOTAL --> COUP[coupling losses]
    TOTAL --> BAL[codebook_balance_loss]
    TOTAL --> ORTHO[orthogonality_loss]

    REC --> E_REC[eeg_full: source + obs vs original]
    REC --> F_REC[fnirs_full: source + obs vs original]

    ST --> ST_F[fnirs_source_target_loss<br/>source_recon vs HRF target]
    ST --> ST_E[eeg_source_aux_loss<br/>source_recon vs power envelope]

    OT --> OT_F[fnirs_obs_loss<br/>obs_recon vs obs_target]
    OT --> OT_E[eeg_obs_loss<br/>obs_recon vs obs_target]

    VQ --> VQ_S[vq_source_loss<br/>eeg + fnirs]
    VQ --> VQ_O[vq_observation_loss<br/>eeg + fnirs]

    COUP --> COUP_CONC[lag_focus_loss<br/>lag marginal entropy]
    COUP --> COUP_SMTH[joint_smoothness_loss<br/>neighbor JS on lag x token]

    BAL --> BAL_S[source_balance_loss]
    BAL --> BAL_O[observation_balance_loss]

    ORTHO --> O_E[orthogonality_loss<br/>eeg_source ⊥ eeg_obs]
    ORTHO --> O_F[orthogonality_loss<br/>fnirs_source ⊥ fnirs_obs]
```

### Current Target Loss Weights (Phase 2B — Physical Model)

| Loss Term | Weight | Purpose |
|-----------|--------|---------|
| `eeg_rec_loss` | 1.0 (amp 1.0 + time 0.9) | EEG full reconstruction via source + observation sum |
| `fnirs_rec_loss` | 1.0 (amp 1.0 + time 1.0) | fNIRS full reconstruction via source + observation sum |
| `source_target_loss` (fNIRS) | 0.3 | fNIRS source decoder → HRF(shared_state) via Croce model |
| `eeg_source_aux_loss` | 0.3 (weight × aux_weight) | EEG source decoder → signed-RMS-carrier, temporally smoothed |
| `observation_loss` (fNIRS) | 0.15 | fNIRS observation decoder → original − HRF(shared_state) |
| `observation_loss` (EEG) | 0.15 | EEG observation decoder → original − signed-RMS-carrier |
| `vq_loss` | 1.0 × quantization_strength | Commitment + EMA codebook loss (all 4 quantizers) |
| `source_coupling_loss` | `coupling.weight` | `lag_focus_loss + 0.2 * joint_smoothness_loss` when coupling prior is enabled |
| `lag_focus_loss` | internal 1.0 | Normalized entropy of the lag marginal $P(\tau \mid z_{eeg})$ |
| `joint_smoothness_loss` | internal 0.2 | Neighbor JS divergence on $Q(\tau, z_{fnirs} \mid z_{eeg})$ |
| `codebook_balance_loss` | 0.08 | Entropy-based dead-code prevention |
| `orthogonality_loss` | 0.05 | Cosine similarity penalty: source ⊥ observation |

### Coupling Structure Monitoring

Current implementation does not apply a direct EEG-fNIRS KL matching loss. Coupling monitoring therefore focuses on structural priors and matrix geometry:

| Loss | Role | Healthy range | Danger signal |
|------|------|--------------|---------------|
| `lag_focus_loss` | Delay preference concentration | Below the uniform baseline, but not collapsing to 0 | Near 1.0 → lag structure remains uninformative |
| `joint_smoothness_loss` | EEG-neighbor consistency in joint delay-response space | Decreasing, then stable | Increasing while lag focus drops → over-constrained or noisy neighborhoods |

## 4. Component Catalog

### Core Tokenizer

| File | Role |
|------|------|
| [src/tokenizers/factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py) | **Mainline tokenizer**: `SourceObservationLaBraMVQNSP` — encoders, projectors, 4 quantizers, coupling, dual source/observation decoders |
| [src/tokenizers/labram_vqnsp.py](../src/tokenizers/labram_vqnsp.py) | **Shared components**: `NormEMAVectorQuantizer`, `TransformerEncoder`, `TransformerDecoder`, `l2norm`, `MultiChannelPatchEmbedding` |
| [src/tokenizers/base.py](../src/tokenizers/base.py) | Abstract `BaseTokenizer` class |
| [src/tokenizers/registry.py](../src/tokenizers/registry.py) | Tokenizer factory: config → constructor mapping, `StandardizedOutput` interface |

### Loss Functions

| File | Role |
|------|------|
| [src/losses/multimodal_tokenizer.py](../src/losses/multimodal_tokenizer.py) | `batch_usage_entropy_loss`, `straight_through_assignment_probs`, `orthogonality_loss`, `align_pair`, `coupling_lag_focus_loss`, `coupling_eeg_neighbor_smoothness_loss` |
| [src/losses/reconstruction.py](../src/losses/reconstruction.py) | Multi-STFT and time-domain reconstruction losses |

### Spatial & Physiological Priors

| File | Role |
|------|------|
| [src/data/channel_adjacency.py](../src/data/channel_adjacency.py) | 10-10 EEG neighbor table, fNIRS channel name parsing, `mnt.mat` 3D coordinate validation, spatial adjacency matrix construction, per-channel RMS envelope and spatially-weighted fNIRS neural driver |
| [src/inference/neurovascular_smc.py](../src/inference/neurovascular_smc.py) | Sequential Monte Carlo filter for neurovascular state-space model (shared neural state + modality-specific forward models)

### Analysis & Visualization

| File | Role |
|------|------|
| [src/visualization/tokenizer_analysis_suite.py](../src/visualization/tokenizer_analysis_suite.py) | Standardized analysis entry point |
| [src/visualization/source_observation_analysis.py](../src/visualization/source_observation_analysis.py) | Source/observation alignment analysis, Gate 1-4 scorecard |

### Configs

| Directory | Purpose |
|-----------|---------|
| [experiments/configs/source_observation/phase1/](../experiments/configs/source_observation/phase1/) | Phase 1 Gate1 baseline configs (locked) |
| [experiments/configs/source_observation/phase2/](../experiments/configs/source_observation/phase2/) | Phase 2 HRF Source Target configs (historical reference) |
| [experiments/configs/source_observation/phase2a/](../experiments/configs/source_observation/phase2a/) | Phase 2A Dual Decoder + spatial source target configs (**active**) |

## 5. Quantizer Summary

| Quantizer | Codebook Size | Embedding Dim | Semantics |
|-----------|---------------|---------------|-----------|
| `eeg_source_quantizer` | K=32 | D=48 | EEG neurovascular coupling state (shared neural driver) |
| `fnirs_source_quantizer` | K=32 | D=48 | fNIRS neurovascular coupling state (shared neural driver) |
| `eeg_observation_quantizer` | K=64 | D=64 | EEG modality-specific encoding debt |
| `fnirs_observation_quantizer` | K=64 | D=48 | fNIRS modality-specific encoding debt |

All quantizers use EMA updates, kmeans initialization, dead code revival, and cosine-similarity-based assignment (l2-normalized). Phase 2A expands observation codebooks to 64 while keeping source at 32, providing more capacity for modality-specific details.

## 6. Coupling Mechanism

The coupling matrix `coupling_logits` is an `[n_lags, K_src, K_src]` learned parameter.

**Forward pass** (for each lag):
1. Align EEG and fNIRS source token distributions with lag offset
2. Maintain `coupling_logits[lag]` as the lag-indexed EEG→fNIRS mapping scaffold
3. Current implementation does not optimize a direct KL-based EEG-fNIRS matching loss

**Structural priors** (current active design):
- **Lag focus**: for each EEG source token, the lag marginal of the joint distribution
    $$Q_i(\tau, j) = P(\tau, z_{fnirs}=j \mid z_{eeg}=i)$$
    should prefer a few delays. This sharpens delay structure without forcing only a few token-lag pairs overall.
- **Joint smoothness**: nearby EEG tokens in codebook space should have similar joint delay-response distributions $Q_i(\tau, j)$.
    Neighborhoods are computed from detached EEG source codebook geometry rather than raw token indices.

**Selection**: Diagnostics now use the dominant lag under the average lag marginal of $Q_i(\tau, j)$; this is still a structural summary, not a batch-level best-lag search.

**Current lags**: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]`

## 7. Source Target Construction (Croce et al. 2017 Physical Model)

### Design Motivation

Prior Phase 2A design used `EEG_power_envelope` (μV², non-negative) as the EEG
source target. This broke the additive decomposition `original = source + observation`
because power units differ from voltage, and the envelope's non-negativity forced
the observation branch to carry the DC offset and zero-crossing structure.

The revised design adopts Croce et al. 2017's joint EEG-fNIRS state-space model:
a shared latent neural state `s(t)` drives both modalities — EEG observes it
instantaneously, fNIRS observes it through hemodynamic convolution.

### Shared Neural State

```
s_k = α · s_{k-1} + (1 − α) · x_k

where  x_k = channel-averaged EEG power, downsampled to fNIRS rate (10 Hz)
       α   = shared_state_alpha (default 0.90)
```

- α → 1.00: only sub-0.1 Hz hemodynamic fluctuations survive (SMC limit)
- α ≈ 0.90: ~1 s half-life — alpha/beta-band power envelope preserved
- α → 0.00: raw EEG power, no smoothing

### fNIRS Source Target (Croce forward model)

```
s(t) [B,1,100] → HRF convolution (learnable double-gamma) → rescale → [B,36,100]
```
The HRF convolution absorbs the 4–6 s neurovascular delay. The output is
time-synchronous with the original fNIRS (zero-phase alignment).

### EEG Source Target (Croce forward model)

Mode: `signed_rms_carrier` (default in Phase 2B)

```
EEG [B,30,2000] → per-channel RMS envelope (Hann-smoothed, μV units)
                → temporal smoothing with shared α
                → multiply by sign(smoothed voltage waveform)
                → signed, μV units, same physical meaning as raw EEG
```

Key properties:
- Same physical units as EEG (μV, signed)
- Additive decomposition `original = source + observation` is physically meaningful
- Temporal smoothing via shared α removes fast noise while preserving task dynamics

### Observation Target

```
obs_target = original - source_target  (computed independently per modality)
```

## 8. Decoder Modes

Three decoder input modes are explicitly trained:

| Mode | Input to source decoder | Input to obs decoder | Target | Loss |
|------|------------------------|---------------------|--------|------|
| Full | source_q | obs_q | original | full reconstruction loss |
| Source-only | source_q | zeros | source_target | source_target_loss |
| Observation-only | zeros | obs_q | obs_target | observation_loss |

Full reconstruction = source_recon + observation_recon (additive in signal space).

## 9. Phase Status

| Phase | Name | Status | Key Deliverable |
|-------|------|--------|-----------------|
| Phase 1 | Structural Migration | ✅ Complete | Source/Observation tokenizer, shared/private removed |
| Phase 2 | HRF Source Target | ✅ Complete | Double-gamma HRF kernel; Gate 2-4 fail, needed Phase 2A redesign |
| Phase 2A | Branch Target Redesign + Dual Decoder | ✅ Complete | Dual decoder, unified source target, explicit observation target |
| Phase 2B | Croce 2017 Physical Model + Coupling Structure Priors | ✅ **Current** | Shared-state AR-smoothed neural driver, signed-RMS EEG target, HRF fNIRS target, lag focus + joint smoothness |
| Mechanism C | Causal Asymmetry | ❌ Abandoned | See IMPLEMENTATION_PLAN.md §11 |

### Locked Phase1 Handoff

| Artifact | Role |
|----------|------|
| [experiments/configs/source_observation/phase1/gate1_best_current.yaml](../experiments/configs/source_observation/phase1/gate1_best_current.yaml) | Current best Gate1-stable baseline alias |
| [experiments/configs/source_observation/phase1/gate1_baseline_locked_bs128.yaml](../experiments/configs/source_observation/phase1/gate1_baseline_locked_bs128.yaml) | Clean reusable Gate1 baseline |
| [experiments/runs/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718](../experiments/runs/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718) | Best recorded Gate1 pass |

### Phase 2 Diagnostic Baseline

| Artifact | Role |
|----------|------|
| [experiments/runs/s2_phase2_gate2_hrf_target_uniform32_bs128_longrun/](../experiments/runs/s2_phase2_gate2_hrf_target_uniform32_bs128_longrun/) | Phase 2 run with full Gate 1-4 analysis |
| [experiments/runs/s2_phase2_gate2_hrf_target_uniform32_bs128_longrun/analysis/gate_summary.json](../experiments/runs/s2_phase2_gate2_hrf_target_uniform32_bs128_longrun/analysis/gate_summary.json) | Gate scorecard: Gate1=pending, Gate2/3/4=fail |

## 10. Related Documents

| Document | Role |
|----------|------|
| [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) | Implementation order, file migration scope, validation gates |
| [PHYSIOLOGICAL_COUPLING_PLAN.md](PHYSIOLOGICAL_COUPLING_PLAN.md) | Mechanism motivation, math, physiological interpretation |
| [SEMANTIC_TOKEN_SCORECARD.md](SEMANTIC_TOKEN_SCORECARD.md) | 4-Gate evaluation framework |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | Formal experiment conclusions |
| [architecture_changelog/INDEX.md](architecture_changelog/INDEX.md) | Chronological architecture change records |
