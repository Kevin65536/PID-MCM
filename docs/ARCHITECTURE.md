# Current Architecture: Source/Observation Tokenizer

> **Semantics version**: `s2_source_observation_v4_croce_local_highwl`
> **Last updated**: 2026-06-04
> **Current phase**: Croce local cache tokenizer training with highWL-only fNIRS input
> **Mainline class**: `SourceObservationLaBraMVQNSP` in [factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py)
> **Changelog**: [architecture_changelog/INDEX.md](architecture_changelog/INDEX.md)

> **Active note**: repository code still contains legacy proxy target implementations. They are candidate baselines only. They do **not** define the current branch-target contract.

> **Current training contract**: the active tokenizer dataset is `croce_local_cache`. Each sample is one fNIRS spatial anchor, its six-channel local EEG neighbourhood, and explicit source/observation targets from the generated Croce cache. fNIRS uses only `highWL` (`source_fnirs_optical_channel_0` / `obs_fnirs_optical_channel_0`) as an optical measurement-space, HbO-sensitive proxy; `lowWL` remains in cache metadata but is ignored by the current model input.

---

## 1. Architecture Contract

当前主线只把下列语义当作 branch target 合同：

1. source branch 监督对象是两个模态各自的**干净生理观测成分**，而不是单向 proxy 或只在 latent 空间自洽的隐藏变量；
2. clean EEG 与 clean fNIRS 必须由同一个共享生理模型联合约束，二者地位对称；
3. observation target 必须保持逐模态线性残差定义：

   $$
   y^{obs}_{EEG} = y^{raw}_{EEG} - y^{src}_{EEG}, \qquad
   y^{obs}_{fNIRS} = y^{raw}_{fNIRS} - y^{src}_{fNIRS}
   $$

4. source branch 尽量承载真实生理活动，observation branch 尽量承载被试差异、导联接触、仪器噪声和其他观测污染。

下列表述不再属于当前主线语义：

1. 任何把单模态代理量直接当作 source 定义的做法；
2. 任何把 clean EEG 与 clean fNIRS 置于不对等地位的做法；
3. 任何单向构造另一模态 clean target 的做法；
4. 任何只在 latent 中自洽、却不给出测量空间 clean source 的做法。

## 2. Component Architecture

```mermaid
graph TB
    subgraph Inputs
        EEG["Local EEG Signal<br/>B x 6ch x 4000"]
        FNIRS["fNIRS highWL Signal<br/>B x 1ch x 200"]
    end

    subgraph PatchEmbedding
        E_PE["MultiChannelPatchEmbedding<br/>patch=400 → 10 patches"]
        F_PE["MultiChannelPatchEmbedding<br/>patch=20 → 10 patches"]
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
        F_OQ["fnirs_observation_quantizer<br/>K=32 base / 64 sweep D=48"]
    end

    subgraph Coupling["Cross-Modal Coupling"]
        COUP_LOGITS["coupling_logits<br/>Parameter: n_lags x K_src x K_src"]
        COUP_LOSS["lag_focus_loss<br/>+ joint_smoothness_loss<br/>+ optional association_loss"]
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
        E_SA["eeg_src_amp_head<br/>256→6x201"]
        E_SPH["eeg_src_phase_head<br/>256→6x201"]
        E_OA["eeg_obs_amp_head<br/>256→6x201"]
        E_OPH["eeg_obs_phase_head<br/>256→6x201"]
        F_SA["fnirs_src_amp_head<br/>160→1x11"]
        F_SPH["fnirs_src_phase_head<br/>160→1x11"]
        F_OA["fnirs_obs_amp_head<br/>160→1x11"]
        F_OPH["fnirs_obs_phase_head<br/>160→1x11"]
    end

    subgraph Reconstruction
        E_SUM["Σ source + observation<br/>= eeg_full_recon"]
        F_SUM["Σ source + observation<br/>= fnirs_full_recon"]
    end

    subgraph Targets["Branch Target Contract"]
        LAT["shared physiological model<br/>jointly constrained by EEG + fNIRS"]
        E_ST["EEG source target<br/>clean EEG component in measurement space"]
        F_ST["fNIRS source target<br/>clean fNIRS component in measurement space"]
        E_OT["EEG obs target<br/>raw EEG - clean EEG"]
        F_OT["fNIRS obs target<br/>raw fNIRS - clean fNIRS"]
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

    EEG --> LAT
    FNIRS --> LAT
    LAT --> E_ST
    LAT --> F_ST
    EEG --> E_OT
    FNIRS --> F_OT

    style Coupling fill:#e1f5fe
    style Quantizers fill:#fff3e0
    style Projection fill:#f3e5f5
    style SourceDecoders fill:#e8f5e9
    style ObsDecoders fill:#fce4ec
    style Targets fill:#fff9c4
```


**Key architectural invariant**: source/observation 仍然是加法分解架构。唯一被重置的是“source target 究竟如何定义”。

## 3. Data Flow

```mermaid
sequenceDiagram
    participant EEG as EEG [B,6,4000]
    participant fNIRS as fNIRS highWL [B,1,200]
    participant Enc as Encoders
    participant Proj as Projection Heads
    participant Quant as Quantizers (4×)
    participant Coup as Coupling Matrix
    participant DecS as Source Decoders
    participant DecO as Observation Decoders
    participant Target as Target Construction
    participant Loss as Loss Computation

    EEG->>Target: provide raw EEG observation
    fNIRS->>Target: provide raw fNIRS observation
    Target->>Target: read explicit Croce clean EEG + highWL source targets
    Target->>Target: obs_target = raw - source_target (per modality)

    EEG->>Enc: EEG encoder
    fNIRS->>Enc: fNIRS encoder
    Enc->>Proj: 4 projection heads
    Proj->>Quant: 4 quantizers (straight-through)

    Quant-->>Coup: eeg_source_probs, fnirs_source_probs
    Coup->>Loss: lag_focus_loss + joint_smoothness_loss

    Quant->>DecS: source_q → source decoder → source_recon
    Quant->>DecO: obs_q → observation decoder → obs_recon

    DecS->>Loss: source_target_loss (source_recon vs clean source target)
    DecO->>Loss: observation_loss (obs_recon vs obs_target)
    DecS->>Loss: full_recon_loss ((source_recon + obs_recon) vs original)
    DecO->>Loss: (same, additive)

    Loss->>Loss: vq_loss = commitment (× quantization_strength)
    Loss->>Loss: orthogonality_loss (source ⊥ obs)
    Loss->>Loss: codebook_balance_loss (straight-through hard assign, per-branch temps)
    Loss->>Loss: total = rec + source_target + obs_target + vq + coupling + balance + ortho
```

## 4. Loss Contract

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

    ST --> ST_F[fnirs_source_target_loss<br/>source_recon vs clean fNIRS source]
    ST --> ST_E[eeg_source_aux_loss<br/>source_recon vs clean EEG source]

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

### Current Loss Meaning

| Loss Term | Weight | Purpose |
|-----------|--------|---------|
| `eeg_rec_loss` | 1.0 (amp 1.0 + time 0.9) | EEG full reconstruction via source + observation sum |
| `fnirs_rec_loss` | 1.0 (amp 1.0 + time 1.0) | fNIRS full reconstruction via source + observation sum |
| `source_target_loss` (fNIRS) | 0.3 | fNIRS source decoder → clean fNIRS component in measurement space |
| `eeg_source_aux_loss` | 0.3 (weight × aux_weight) | EEG source decoder → clean EEG component in measurement space |
| `observation_loss` (fNIRS) | 0.15 | fNIRS observation decoder → raw fNIRS − clean fNIRS |
| `observation_loss` (EEG) | 0.15 | EEG observation decoder → raw EEG − clean EEG |
| `vq_loss` | 1.0 × quantization_strength | Commitment + EMA codebook loss (all 4 quantizers) |
| `source_coupling_loss` | `coupling.weight` | `lag_focus_loss + 0.2 * joint_smoothness_loss` when coupling prior is enabled |
| `lag_focus_loss` | internal 1.0 | Normalized entropy of the lag marginal $P(\tau \mid z_{eeg})$ |
| `joint_smoothness_loss` | internal 0.2 | Neighbor JS divergence on $Q(\tau, z_{fnirs} \mid z_{eeg})$ |
| `codebook_balance_loss` | 0.08 | Entropy-based dead-code prevention |
| `orthogonality_loss` | 0.05 | Cosine similarity penalty: source ⊥ observation |

当前活动文档只定义这些 loss 的**语义角色**。如果代码里某个 loss 仍通过 legacy proxy 路径来近似实现，应视为待替换的旧求解路径，而不是当前合同。

### Coupling Structure Monitoring

Current implementation does not apply a direct EEG-fNIRS KL matching loss. Coupling monitoring therefore focuses on structural priors and matrix geometry:

| Loss | Role | Healthy range | Danger signal |
|------|------|--------------|---------------|
| `lag_focus_loss` | Delay preference concentration | Below the uniform baseline, but not collapsing to 0 | Near 1.0 → lag structure remains uninformative |
| `joint_smoothness_loss` | EEG-neighbor consistency in joint delay-response space | Decreasing, then stable | Increasing while lag focus drops → over-constrained or noisy neighborhoods |

## 5. Component Catalog

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
| [src/data/croce_local_cache_dataset.py](../src/data/croce_local_cache_dataset.py) | Active Croce local cache adapter: returns `eeg [B,6,4000]`, `fnirs [B,1,200]`, explicit source/observation targets, and Gate0 highWL-only metadata |
| [src/data/channel_adjacency.py](../src/data/channel_adjacency.py) | 10-10 EEG neighbor table, fNIRS channel name parsing, `mnt.mat` 3D coordinate validation, spatial adjacency matrix construction, per-channel RMS envelope and spatially-weighted fNIRS neural driver |
| [src/inference/neurovascular_smc.py](../src/inference/neurovascular_smc.py) | Candidate physical-model inference utilities for joint EEG-fNIRS source estimation; current implementations include legacy Croce-style proxy paths under review |

### Analysis & Visualization

| File | Role |
|------|------|
| [src/visualization/tokenizer_analysis_suite.py](../src/visualization/tokenizer_analysis_suite.py) | Standardized analysis entry point |
| [src/visualization/source_observation_analysis.py](../src/visualization/source_observation_analysis.py) | Source/observation alignment analysis, Gate 0-4 scorecard. Gate0 asserts the highWL-only cache/input contract before semantic metrics are interpreted |

### Configs

| Directory | Purpose |
|-----------|---------|
| [experiments/configs/source_observation/phase1/](../experiments/configs/source_observation/phase1/) | Phase 1 Gate1 baseline configs (locked) |
| [experiments/configs/source_observation/phase2/](../experiments/configs/source_observation/phase2/) | Historical proxy-target configs; not current branch-target contract |
| [experiments/configs/source_observation/phase2a/](../experiments/configs/source_observation/phase2a/) | Historical redesign configs; decoder structure still relevant, target semantics superseded |
| `experiments/configs/source_observation/croce_local/` | Current Croce local highWL-only tokenizer configs. The base config uses canonical cache roots under `croce_validation/cache/croce_local/highwl_v1/` and writes future runs under `experiments/runs/source_observation/croce_local/highwl_v1/` |

## 5. Quantizer Summary

| Quantizer | Codebook Size | Embedding Dim | Semantics |
|-----------|---------------|---------------|-----------|
| `eeg_source_quantizer` | K=32 | D=48 | EEG neurovascular coupling state (shared neural driver) |
| `fnirs_source_quantizer` | K=32 | D=48 | fNIRS neurovascular coupling state (shared neural driver) |
| `eeg_observation_quantizer` | K=64 | D=64 | EEG modality-specific encoding debt |
| `fnirs_observation_quantizer` | K=32 base / K=64 sweep | D=48 | highWL fNIRS modality-specific encoding debt |

All quantizers use EMA updates, kmeans initialization, dead code revival, and cosine-similarity-based assignment (l2-normalized). The active Croce local base keeps source at K=32, EEG observation at K=64, and starts highWL-only fNIRS observation at K=32; the small sweep tests LR=2e-4 and fNIRS observation K=64.

## 6. Coupling Mechanism

The coupling tensor `coupling_logits` is an `[n_lags, K_src, K_src]` learned parameter. `n_lags` is derived from the token window length, so the active 20s Croce local tokenizer with 10 tokens uses valid nonnegative lags `0..9`. There is no user-configured `lag_candidates` list and no selected-lag optimization path.

**Forward pass**:
1. Maintain `coupling_logits` as a full lag-indexed source-state correspondence scaffold
2. Convert it to the EEG-conditioned joint delay-response distribution
    $$Q_i(\tau, j) = P(\tau, z_{fnirs}=j \mid z_{eeg}=i)$$
3. Apply structural coupling losses over all valid lag slices rather than selecting one lag

**Structural priors** (current active design):
- **Lag focus**: for each EEG source token, the lag marginal of the joint distribution
    $$Q_i(\tau, j) = P(\tau, z_{fnirs}=j \mid z_{eeg}=i)$$
    should prefer a few delays. This sharpens delay structure without forcing only a few token-lag pairs overall.
- **Joint smoothness**: nearby EEG tokens in codebook space should have similar joint delay-response distributions $Q_i(\tau, j)$.
    Neighborhoods are computed from detached EEG source codebook geometry rather than raw token indices.
- **Optional association loss**: when enabled, EEG/fNIRS source token distributions are aligned for every valid lag and averaged; it does not choose a best lag.

**Diagnostics**: Analysis reports all-lag tensor views: EEG×fNIRS marginal, EEG×lag marginal, expected fNIRS index by lag, and per-lag conditional slices. It does not report a best or selected lag as a training objective.

## 7. Branch Target Contract

### Required Outputs

当前 branch target 只接受如下输出对：

$$
(\hat y^{src}_{EEG}(t), \hat y^{src}_{fNIRS}(t))
$$

它们必须同时满足：

1. 由同一个共享生理模型联合约束；
2. 分别位于 EEG 与 fNIRS 的测量空间；
3. 保持与原始测量足够同步，使 observation target 可线性定义；
4. 不把被试差异、接触问题和仪器噪声误写成 source semantics。

### What Is No Longer Accepted As Mainline Semantics

以下内容可以作为候选 baseline 保留在代码中，但不再被活动文档当作 branch target 定义：

1. 任何单模态幅值代理直接充当 clean EEG source；
2. 任何单向构造直接充当 clean fNIRS source；
3. 用目标模态自身统计量“制造” clean source；
4. 只凭 coupling 可预测性来替代 clean target 的显式构造。

### Candidate Physical Models

当前允许进入主线评审的物理模型家族包括：

1. **Croce-style joint state-space model**：但必须升级为真正的 joint inference，并输出对称的 clean EEG / clean fNIRS 成分；
2. **Nuisance-augmented local state-space model**：在共享生理状态之外显式建模 contact / device / subject drift；
3. **Simpler dynamic-factor baseline**：如果它能更稳定地满足 clean-source + linear-residual 合同，也允许作为 branch-target baseline。

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
| Phase 2 | Historical Proxy-Target Stages | ⚠️ Historical | Legacy proxy-target experiments, retained only for comparison |
| Phase 2A | Decoder Structure Redesign | ✅ Complete | Dual decoder, explicit observation target, additive reconstruction |
| Phase 2B | Croce Candidate Model Audit | ⚠️ Historical Candidate | Joint state-space tooling and proxy-target baselines introduced |
| Current | Croce Local HighWL Tokenizer Training | ✅ **Active** | Train and evaluate the local highWL-only source/observation tokenizer on generated Croce caches |
| Mechanism C | Causal Asymmetry | ❌ Abandoned | See IMPLEMENTATION_PLAN.md §11 |

### Locked Phase1 Handoff

| Artifact | Role |
|----------|------|
| [experiments/configs/source_observation/phase1/gate1_best_current.yaml](../experiments/configs/source_observation/phase1/gate1_best_current.yaml) | Current best Gate1-stable baseline alias |
| [experiments/configs/source_observation/phase1/gate1_baseline_locked_bs128.yaml](../experiments/configs/source_observation/phase1/gate1_baseline_locked_bs128.yaml) | Clean reusable Gate1 baseline |
| [experiments/runs/archive/source_observation_phase1_gate1_stabilization_20260511/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718](../experiments/runs/archive/source_observation_phase1_gate1_stabilization_20260511/s2_phase1_gate1_health_uniform32_stable_sourceonly_balance_provq_nophase_longwarmup_bs128_20260511_175718) | Best recorded Gate1 pass |

### Phase 2 Diagnostic Baseline

| Artifact | Role |
|----------|------|
| [experiments/runs/archive/pre_croce_local_highwl_20260604/s2_phase2_gate2_hrf_target_uniform32_bs128_longrun/](../experiments/runs/archive/pre_croce_local_highwl_20260604/s2_phase2_gate2_hrf_target_uniform32_bs128_longrun/) | Phase 2 run with full Gate 1-4 analysis |
| [experiments/runs/archive/pre_croce_local_highwl_20260604/s2_phase2_gate2_hrf_target_uniform32_bs128_longrun/analysis/gate_summary.json](../experiments/runs/archive/pre_croce_local_highwl_20260604/s2_phase2_gate2_hrf_target_uniform32_bs128_longrun/analysis/gate_summary.json) | Gate scorecard: Gate1=pending, Gate2/3/4=fail |

## 10. Related Documents

| Document | Role |
|----------|------|
| [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) | Implementation order, file migration scope, validation gates |
| [CROCE2017_REAL_DATA_VALIDATION_PLAN.md](CROCE2017_REAL_DATA_VALIDATION_PLAN.md) | Real-data validation plan for Croce-style forward approximations, inverse stability, and tokenizer integration |
| [PHYSIOLOGICAL_COUPLING_PLAN.md](PHYSIOLOGICAL_COUPLING_PLAN.md) | Mechanism motivation, math, physiological interpretation |
| [SEMANTIC_TOKEN_SCORECARD.md](SEMANTIC_TOKEN_SCORECARD.md) | 4-Gate evaluation framework |
| [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | Formal experiment conclusions |
| [STORAGE_LAYOUT.md](STORAGE_LAYOUT.md) | Canonical generated-data and run-output paths |
| [architecture_changelog/INDEX.md](architecture_changelog/INDEX.md) | Chronological architecture change records |
