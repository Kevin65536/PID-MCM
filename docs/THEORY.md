# Theory: Neuro-Tokenization for EEG/fNIRS

> **Version**: 4.0  
> **Last Updated**: 2026-01-13  
> **Roadmap**: See [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)

---

## Update Notice (2026-05-14)

For the current active development direction, see:

- [docs/PHYSIOLOGICAL_COUPLING_PLAN.md](PHYSIOLOGICAL_COUPLING_PLAN.md) — Active: physiological coupling constraints design
- [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) — Current: implementation status and roadmap
- [docs/ARCHITECTURE.md](ARCHITECTURE.md) — Current: architecture specification
- [docs/notes/2026.3.4_current_problem_of_physiological_foundation_model.md](notes/2026.3.4_current_problem_of_physiological_foundation_model.md) — Background: EEG foundation model challenges

Current branch-target note: active semantics now require jointly inferred clean EEG / clean fNIRS source targets plus linear observation residuals. Historical shared/private and one-way proxy-target interpretations are superseded.

---

## 1. Motivation: Why Tokenization?

EEG 与 fNIRS 是两种互补的神经信号模态：

| 特性 | EEG | fNIRS |
|------|-----|-------|
| 时间分辨率 | 毫秒级 (>100Hz) | 秒级 (~10Hz) |
| 空间分辨率 | 低 (cm级) | 中等 (mm级) |
| 信号来源 | 神经电活动 | 血氧代谢 |
| 主要频带 | 1-100Hz | <0.1Hz |

直接在原始连续信号上进行跨模态分析面临挑战：
- **时标冲突**：采样率差异导致难以对齐
- **分布差异**：信号统计特性完全不同
- **噪声敏感**：原始表示易受预处理差异影响

**核心思想**：将两种模态的信号都映射为**离散 token 序列**（codebook），在 token 空间中进行分析。

---

## 2. Tokenization Framework

### 2.1 基本架构

对每个模态 $m \in \{\text{eeg}, \text{fnirs}\}$：

```
Input x_m [B, T]
    ↓
Encoder E_m
    ↓
Continuous Latent z_m [B, T', D]
    ↓
Quantizer Q_m
    ↓
Token Indices q_m [B, T'] ∈ {1, ..., K_m}
Token Embeddings e_{q_m} [B, T', D]
    ↓
Decoder D_m
    ↓
Reconstruction x̂_m [B, T]
```

### 2.2 训练目标

**主要目标：重构 (Reconstruction)**

$$\mathcal{L}_{rec} = \|x_m - \hat{x}_m\|_2^2 + \lambda_f \mathcal{L}_{freq}(x_m, \hat{x}_m)$$

其中 $\mathcal{L}_{freq}$ 是频域损失（multi-scale STFT），对于 EEG 的频谱特性保真尤为重要。

**辅助目标：Codebook 健康度**

- **Perplexity**：衡量 codebook 使用的丰富度
- **Utilization**：被使用的 code 比例
- **Dead Codes**：从未使用的 code 数量

### 2.3 量化方法

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| **FSQ** | 隐式 codebook，无 collapse 风险 | 快速验证 |
| **VQ-VAE** | 显式 codebook，更灵活 | 正式实验 |
| **RVQ** | 残差量化，更高表达能力 | 高保真重构 |

---

## 3. Design Decisions

### 3.1 Separate Codebooks（推荐）

EEG 与 fNIRS 各自拥有独立的 tokenizer（各自 codebook）。

**优势**：
- 各模态可以学习最适合自己的离散表示
- 采样率差异不会互相干扰
- 避免共享 codebook 被分布差异主导

**架构**：
```
EEG  → Encoder_EEG  → VQ_EEG  → tokens_eeg [K_eeg codes]
fNIRS → Encoder_fNIRS → VQ_fNIRS → tokens_fnirs [K_fnirs codes]
```

### 3.2 未来扩展：Shared Semantic Space

在独立 tokenizer 稳定后，可通过共享 projector 将 token embedding 映射到共同语义空间 $S$：

$$s_m = P(e_{q_m}) \in S$$

这允许在 $S$ 空间中进行跨模态对齐与分析。

### 3.3 What a physiological token should mean

word2vec 的真正启发，不是后续模型依赖某个固定的线性类比技巧，而是说明一个好的表示空间可以把语义变成可计算的几何结构。对 EEG 与 fNIRS 来说，我们不应该把 token 理解成“波形片段的名字”，而应该把它理解成“可复用的局部生理状态标识”。

因此，一个好的生理 token 应该满足：

1. 它对应的窗口在生理机制上近似等价，而不是仅仅在原始形状上相似；
2. 它对未来状态和另一模态的滞后状态具有更强预测力；
3. 它对 subject、device、artifact 等 nuisance factor 尽量不敏感；
4. source token 承载共享生理源对应的 clean measurement 成分，observation token 承载模态特异 residual 与 nuisance。

换句话说，tokenization 的目标不是制造更多同步 token overlap，而是构建一个可用于状态转移建模、滞后耦合建模和下游泛化的离散生理语义空间。

---

## 4. Evaluation Criteria

tokenizer 不应只被当作 reconstruction codec 来评估，而应被当作一个离散生理语义系统来评估。当前推荐的详细指标设计见 [SEMANTIC_TOKEN_SCORECARD.md](SEMANTIC_TOKEN_SCORECARD.md)。高层上，评估应分为四层。

### 4.1 Representation Health

这是进入语义讨论前的保底门槛。

| 指标 | 计算方式 | 作用 |
|------|----------|------|
| Perplexity | $\exp(-\sum_k p_k \log p_k)$ | 检查 codebook 是否系统性 collapse |
| Utilization | $\frac{\text{active codes}}{\text{total codes}}$ | 检查有效使用范围 |
| Gini / Top-k Coverage | usage distribution statistics | 检查是否被少数 code 垄断 |
| Reconstruction Guardrails | MSE / STFT / source+observation additive reconstruction | 保证 token 仍然保留基础信号内容 |
| Branch Ablation Gap | source/observation ablation 后的目标退化差异 | 检查 branch 是否真的分工 |

### 4.2 Physiological Semantic Quality

这一层才真正回答“token 是否在表达状态语义”。

| 指标 | 形式 | 含义 |
|------|------|------|
| Intra-token State Consistency | $\sum_k p_k \mathbb{E}[\|\phi(x)-\mu_k\|^2 \mid z=k]$ | 同一 token 内部的状态是否稳定 |
| Prototype Separation Ratio | between-token distance / within-token variance | 不同 token 是否代表不同状态原型 |
| Transition Predictability Gain | $H(Z_{t+\Delta}) - H(Z_{t+\Delta} \mid Z_t)$ | token 是否保留状态转移结构 |
| Augmentation Consistency | nuisance-preserving augmentation 下的一致率 | token 是否对无关扰动稳定 |

### 4.3 Structured Cross-Modal Value

跨模态价值不应被简化成同步 overlap，而应体现为有时滞的结构化预测能力。

| 指标 | 形式 | 含义 |
|------|------|------|
| Lagged MI Gain | $I(Z^{eeg}_{t-\tau}; Z^{fnirs}_t) - I(Z^{eeg}_t; Z^{fnirs}_t)$ | 检查是否存在更合理的生理时滞耦合 |
| Conditional KL Gain | $D_{KL}(P(Z^{fnirs}\mid Z^{eeg}) \| P(Z^{fnirs}))$ | EEG token 是否改变了 fNIRS token 的条件分布 |
| Cross-modal MTP Gain | masked token prediction gain over shuffled baselines | shared states 是否具有跨模态预测价值 |
| Source Usage Balance | source token usage by modality | 检查 source branch 是否只被单一模态垄断 |

### 4.4 Invariance And Downstream Sanity

foundation representation 必须尽量保留真正的生理信号，同时压低 nuisance factor。

- **跨被试 / 跨 session / 跨设备稳定性**：shared token 的统计结构不应对采集条件过度敏感；
- **subject leakage**：shared states 不应越来越容易恢复 subject identity；
- **task / condition signal**：如果 token 真有语义，轻量 probe 至少应看到弱但稳定的 task-relevant 信号；
- **semantic selectivity**：task-relevant signal 应优于 nuisance-relevant signal，而不是相反。

---

## 5. Connection to Downstream Tasks

Tokenization 完成后，离散 token 可用于多种下游任务：

### 5.1 分类任务

将 token 序列作为输入，训练轻量分类器：
- 任务分类（Motor Imagery, P300, etc.）
- 状态检测（疲劳、注意力等）
- 事件检测

### 5.2 Lag-aware Cross-modal Modeling

对 EEG 与 fNIRS，不应默认要求同一时间窗口的 token 在语义空间中直接接近。更合理的目标是让 shared states 在允许生理时滞的前提下，对另一模态 token 或 future state 具有预测力。

可行方式包括：

- lag-aware mutual information analysis；
- cross-modal masked token prediction；
- shared state 到 delayed opposite-modality target 的预测；
- 仅把 same-time overlap 作为补充诊断，而不是主目标。

当前主线实现把这种 lag-aware correspondence 明确写成 EEG 条件下的联合分布：

$$
Q_i(\tau, j) = P(\tau, z_{fnirs}=j \mid z_{eeg}=i)
$$

其中 $i$ 是 EEG source token，$\tau$ 是候选时延，$j$ 是 fNIRS source token。这个定义的关键点是：

- 模型不再要求一个 EEG token 只对应少数几个 token-lag 组合；
- 对于固定的 lag，允许同一个 EEG token 对应多个 fNIRS token；
- 真正被鼓励集中的是 lag 边际分布

$$
p_i(\tau) = \sum_j Q_i(\tau, j)
$$

也就是“给定 EEG 状态，更偏好哪些时延”，而不是“整个 token-lag 空间只能保留少数几个点”。

因此，当前结构先验分成两层：

1. **lag focus**：压低 $p_i(\tau)$ 的熵，让每个 EEG 状态偏好少数几个生理延迟；
2. **joint smoothness**：如果两个 EEG token 在 EEG source codebook 的嵌入空间里彼此接近，那么它们的 $Q_i(\tau, j)$ 也应相近。

这个设计比旧的“每个 lag 切片内做行熵最小化”更贴近神经血管耦合假设，因为它把“延迟结构”单独建模了出来，同时保留了“每个延迟下可对应多个 fNIRS 状态”的自由度。

### 5.3 可解释性分析

- Token 频率分析：哪些 token 在特定任务/状态下更常出现
- Token 聚类：发现有意义的 token 组合
- 空间模式：token 与电极/探头位置的关系

---

## 6. Long-term Goals (Archived)

以下目标将在 tokenization 和 alignment 稳定后逐步推进：

### 6.1 PID Information Decomposition

在 token 空间进行 Partial Information Decomposition (PID)：
- 定义源变量：$C_{eeg}$（EEG token 序列）、$C_{fnirs}$（fNIRS token 序列）
- 分析冗余 (Redundancy)、唯一性 (Unique)、协同 (Synergy)

详见旧版理论文档：[docs/previous_plan/THEORY_v1_ELP.md](previous_plan/THEORY_v1_ELP.md)

### 6.2 Brain State Modeling

使用 token 序列作为离散状态表示，建模大脑状态转移。

---

## Appendix A: FSQ vs VQ-VAE

| 特性 | FSQ | VQ-VAE |
|------|-----|--------|
| Codebook | 隐式（level 组合） | 显式（embedding table） |
| Collapse 风险 | 低 | 高（需 EMA/reset） |
| 梯度 | 直通 | Straight-through |
| 表达能力 | 受 level 限制 | 灵活 |
| 超参数 | levels 列表 | codebook_size, embedding_dim |

## Appendix B: References

1. van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE), 2017
2. Mentzer et al., "Finite Scalar Quantization" (FSQ), 2023
3. Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec", 2021
4. Défossez et al., "High Fidelity Neural Audio Compression" (EnCodec), 2022
