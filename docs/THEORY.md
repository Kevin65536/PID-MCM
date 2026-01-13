# Theory: Neuro-Tokenization for EEG/fNIRS

> **Version**: 4.0  
> **Last Updated**: 2026-01-13  
> **Roadmap**: See [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)

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

---

## 4. Evaluation Criteria

### 4.1 Codebook Health Metrics

| 指标 | 计算方式 | 期望值 |
|------|----------|--------|
| Perplexity | $\exp(-\sum_k p_k \log p_k)$ | > 30% of codebook size |
| Utilization | $\frac{\text{active codes}}{\text{total codes}}$ | > 20% |
| Dead Codes | $\sum_k \mathbb{1}[p_k = 0]$ | < 30% |

### 4.2 Reconstruction Quality

| 指标 | 适用模态 |
|------|----------|
| Time-domain MSE | 所有模态 |
| Spectral MSE (STFT) | EEG（频谱保真） |
| Smoothness Loss | fNIRS（平滑性） |

### 4.3 Generalization

- **跨被试泛化**：Train on subjects 1-N, test on N+1
- **跨 Session 泛化**：同一被试不同 session
- **期望**：泛化误差 < 2x 训练误差

---

## 5. Connection to Downstream Tasks

Tokenization 完成后，离散 token 可用于多种下游任务：

### 5.1 分类任务

将 token 序列作为输入，训练轻量分类器：
- 任务分类（Motor Imagery, P300, etc.）
- 状态检测（疲劳、注意力等）
- 事件检测

### 5.2 跨模态对齐

使用对比学习（InfoNCE）或其他对齐目标，让同一时间窗口的 EEG 与 fNIRS token 在语义空间中接近。

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

详见旧版理论文档：[docs/THEORY_v1_ELP.md](THEORY_v1_ELP.md)

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
