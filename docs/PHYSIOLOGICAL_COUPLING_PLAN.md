# Physiological Coupling Constraints for EEG-fNIRS Tokenizer

> Created: 2026-04-30 | Last revised: 2026-05-11
> Status: Active development plan — Phase 1 complete, Gate1 stable, Phase 2 source-target implementation is the current blocker
> Supersedes: [archive/plans/NEXT_STAGE_ALIGNMENT_PLAN.md](archive/plans/NEXT_STAGE_ALIGNMENT_PLAN.md) (archived as historical reference)
> Reference implementation: [src/tokenizers/factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py)
> New design specification: Section 2 (Source/Observation Branch Semantics)

---

## 1. Motivation

### 1.1 Problem statement

当前 V6 mainline（codebook-focused factorized tokenizer）已经建立了健康的 shared/private factorization，并通过 lag-aware coupling 矩阵提供 EEG 与 fNIRS token 分布之间的条件概率。但 coupling 矩阵本身是**完全自由的参数化**——一个 `[n_lags, K, K]` 的 `nn.Parameter`，初始化为零，没有任何生理先验指导它应该呈现什么结构。

这导致了创新声明上的薄弱："通过离散化提供 EEG 与 fNIRS 生理模式之间的条件概率"本质上只是一个**被动观察**——我们发现了一个统计量，而不是我们设计了一个机制。在神经信号处理领域，仅仅提供"分析能力"不足以作为核心创新点。

### 1.2 Core idea

在 coupling 机制中引入**生理指导的结构约束**，将创新从"我们发现条件概率存在"转变为**"我们设计了一个体现神经血管耦合原理的离散表示机制"**。

关键设计原则（继承自 V6 reset 的结论）：

1. 约束施加在 coupling 参数上，**不直接触碰 encoder/quantizer 的梯度路径**——不与 reconstruction 形成对抗
2. 约束是**软先验**（小系数正则项），不是硬约束——允许数据覆盖先验
3. 约束表达的是**已知的生理结构**，而不是统计优化的目标

### 1.3 Innovation narrative

| 旧声明 | 新声明 |
|--------|--------|
| "离散化揭示了 EEG-fNIRS 之间的条件概率" | "我们在离散化过程中引入了神经血管耦合的结构先验" |
| shared = temporal smoothing, private = residual | **source branch** = HRF-modeled neurovascular coupling state, **observation branch** = modality-specific encoding debt |
| 单一 shared quantizer 强制跨模态共享离散空间 | 双 source codebook 通过 constrained coupling matrix 建立 state correspondence |
| 自由参数的 coupling matrix | 施加 concentration prior（确定性耦合）+ 可独立验证的 smoothness / asymmetry 先验 |
| 被动观察 | 主动机制设计 |
| 一个统计分析工具 | 一个体现生理原理的表示学习框架 |

---

## 2. Branch Semantics Redesign: From Shared/Private to Source/Observation

> Status: Design specification — serving as implementation blueprint for the branch semantics gate
> Motivation: IMPLEMENTATION_PLAN.md Section 5.3 — branch semantics must be clearly defined before A/C mechanisms
> Reference design: TokenFlow (dual codebook + explicit index mapping), PHYSIOLOGICAL_COUPLING_PLAN (structure prior on coupling)

### 2.1 Diagnosis: Why Current Shared/Private Semantics Are Insufficient

当前 V6 baseline 中 shared/private 分支的语义定义存在四个根本性问题：

**问题 1: shared branch 语义是工程代理，不是生理定义**

当前 shared branch 的训练目标之一是 `shared_eeg_common_loss` 和 `shared_fnirs_common_loss`，其定义是：

```
shared_target = smooth_signal(raw_signal, kernel_size)
```

这是一个纯粹的时间平滑操作——它提取信号的低频成分，然后声称"低频 = 跨模态共性"。这种定义没有任何生理学依据。两个模态共享低频成分可能仅仅是因为两者都包含缓慢的基线漂移，而不是因为存在神经血管耦合。

**问题 2: private branch 语义是残差桶**

当前 private branch 的训练目标之一是 `eeg_private_residual_loss` 和 `fnirs_private_residual_loss`，其定义是：

```
private_target = raw_signal - smooth_signal(raw_signal, kernel_size)
```

这意味着 private branch 被定义为"shared branch 的补集"——任何不被 shared 捕获的信息都被倒入 private。这种定义方式导致：
- private 可能包含大量本应属于跨模态耦合的信息（如果平滑核大小不合适）
- private 必然包含高频噪声——但它是否也包含有意义的模态特异生理信息？无从判断
- private 的语义完全依赖于 shared 的定义——如果 shared 语义错误，private 也必然错误

**问题 3: single shared quantizer 是一个未经检验的强假设**

当前实现中，EEG 和 fNIRS 的 shared latent 共同通过一个 `NormEMAVectorQuantizer`（代码位置 [factorized_labram_vqnsp.py:519-522](src/tokenizers/factorized_labram_vqnsp.py#L519-L522)）：

```python
shared_joint = torch.cat([eeg_shared, fnirs_shared], dim=0)
shared_q_joint, shared_idx_joint, shared_info = self.shared_quantizer(shared_joint)
```

这强制 EEG 和 fNIRS shared latent 在同一离散空间中竞争相同的 codebook vectors。但 EEG（电生理，ms 级动态）和 fNIRS（血流动力学，s 级动态）的潜在状态空间在生理上是不同质的——它们可能有不同的最优离散化粒度和不同的流形结构。将两者压入同一个 codebook 可能迫使模型学习一个"折中"的离散空间，既不是 EEG-friendly 也不是 fNIRS-friendly。

**问题 4: equal token count per window 是工程便利**

当前实现要求 `eeg_n_patches == fnirs_n_patches`（[factorized_labram_vqnsp.py:112-116](src/tokenizers/factorized_labram_vqnsp.py#L112-L116)），即 EEG 2000 samples 和 fNIRS 100 samples 必须产生相同数量的 token。这没有生理学理由——只是为了让 alignment 机制可以在 token 级别一一对应。（此问题的解决延后至 Phase 4，当前仍保留 equal token count 作为工作假设。）

### 2.2 Redesigned Branch Semantics

#### 2.2.1 Naming

| 旧名称 | 新名称 | 命名理由 |
|--------|--------|----------|
| shared branch | **source branch** | 编码跨模态的神经生理**源**——驱动 EEG 电位和 fNIRS 血流响应的共同神经活动状态 |
| private branch | **observation branch** | 编码模态特异的**观测**特性——每个测量通道独特的信号特征、噪声结构、被试特异变异 |

source/observation 命名的生成式直觉：

$$\text{Neural Source State } S \rightarrow \begin{cases} \text{EEG Observation } O_{EEG} = f_{EEG}(S) + \epsilon_{EEG} \\ \text{fNIRS Observation } O_{fNIRS} = f_{fNIRS}(S) + \epsilon_{fNIRS} \end{cases}$$

- **source branch** 编码 $S$：底层神经活动状态（neurogenic driver）
- **observation branch** 编码 $\epsilon_{EEG}$ 和 $\epsilon_{fNIRS}$：每种观测模态特有的、不能从 $S$ 预测的信号成分

#### 2.2.2 Source Branch 语义定义

**Source branch 编码的是"神经血管耦合状态"——即可以通过生理模型从一个模态预测另一个模态的信号成分。**

定量操作化定义：

> 一个 token 是 source token，当且仅当它所编码的信息满足：
> 1. 它在 EEG 侧和 fNIRS 侧存在**有结构的跨模态对应**（通过 coupling matrix 可预测）
> 2. 它可以通过**HRF 卷积模型**重建：source branch decoder 输出的信号应当逼近 HRF 模型从另一模态预测的目标信号

这意味着 source branch 的训练信号来自两个方向：
- **前向（EEG→fNIRS）**：EEG 频带功率与 HRF 卷积 → fNIRS 预测 → 作为 fNIRS source decoder 的目标
- **反向（fNIRS→EEG）**：此方向生理约束弱于前向，初始使用 coarse raw EEG 作为弱辅助目标

**Source branch 不做什么**：
- 不要求 EEG 和 fNIRS 在 source codebook 中取相同的 index
- 不要求 source latent 在欧氏空间中接近
- 不承担 raw signal 的全频带 reconstruction（那是 source + observation 联合的任务）

#### 2.2.3 Observation Branch 语义定义

**Observation branch 编码的是"模态特异重建债务"——即 decoder 重建 raw signal 所需的、但无法从 source representation 和跨模态 coupling 中预测的信号成分。**

定量操作化定义：

> 一个 token 是 observation token，当且仅当：
> 1. 它对 reconstruction 有显著贡献：`recon(source+obs)` 显著优于 `recon(source_only)`
> 2. 它不能被另一模态的信息预测：给定 fNIRS source token，预测 EEG observation token 的准确率应接近随机水平
> 3. 它携带更多的被试特异信息（subject leakage 集中在 observation branch）

**Observation branch 不做什么**：
- 不是 source 的简单补集（不是 `raw - smoothed`）
- 不承担跨模态的共有信息
- 不对应任何特定的预处理操作

### 2.3 Architecture: From Single Shared Quantizer to Dual Coupled Source Codebooks

#### 2.3.1 Core Structural Change

```
旧架构 (V6):
  EEG enc → eeg_shared_proj ─┐
                              ├→ shared_quantizer (single, K=128) → eeg_shared_q / fnirs_shared_q
  fNIRS enc → fnirs_shared_proj┘
  
  EEG enc → eeg_private_proj → eeg_private_quantizer (K=256) → eeg_private_q
  fNIRS enc → fnirs_private_proj → fnirs_private_quantizer (K=128) → fnirs_private_q

新架构:
  EEG enc → eeg_source_proj → eeg_source_quantizer (K_src, D_eeg_src) → eeg_source_q
  fNIRS enc → fnirs_source_proj → fnirs_source_quantizer (K_src, D_fnirs_src) → fnirs_source_q
  
  EEG enc → eeg_obs_proj → eeg_obs_quantizer (K_eeg_obs, D_eeg_obs) → eeg_obs_q
  fNIRS enc → fnirs_obs_proj → fnirs_obs_quantizer (K_fnirs_obs, D_fnirs_obs) → fnirs_obs_q
  
  Coupling: M_lag ∈ R^{K_src × K_src} — bridges two source codebooks
```

核心变更：
1. **单一 shared quantizer 替换为两个独立但有桥接的 source codebook**
2. **耦合发生在 codebook 之间的映射矩阵上**，而不是通过强制共享离散空间
3. **Observation codebook 保持模态独立**（与旧 private 一致，但语义定义更新）

#### 2.3.2 Design Rationale (来自 TokenFlow 的启示)

TokenFlow 的成功模式：
- **semantic codebook** 和 **pixel codebook** 是两个独立的 embedding 空间
- 统一不是通过"同一个 codebook"，而是通过 **shared index mapping**：一个 index 同时从两个 codebook 取出对应的 embedding
- $i^* = \arg\min_i(d_{sem,i} + w_{dis} \cdot d_{pix,i})$ — 联合量化目标

对 EEG-fNIRS 的类比：
- **eeg_source_codebook** 和 **fnirs_source_codebook** 是两个独立的 embedding 空间
- 统一通过 **constrained coupling matrix** $M_{lag} \in \mathbb{R}^{K_{src} \times K_{src}}$ 建立
- 不要求同一个 index，而是学习 index 之间的对应概率

#### 2.3.3 Parameter Specification

| 参数 | 旧值 (V6) | 新值 | 说明 |
|------|----------|------|------|
| `shared_codebook_size` | 128 | — | 替换为 `source_codebook_size` |
| `source_codebook_size` | — | 128 | 两个 source codebook 的共享大小（各自独立参数） |
| `source_codebook_dim` | 48 | 48 (eeg), 48 (fnirs) | 可以不同维度 |
| `eeg_private_codebook_size` | 256 | — | 替换为 `eeg_obs_codebook_size` |
| `eeg_obs_codebook_size` | — | 256 | EEG observation codebook |
| `fnirs_private_codebook_size` | 128 | — | 替换为 `fnirs_obs_codebook_size` |
| `fnirs_obs_codebook_size` | — | 128 | fNIRS observation codebook |
| `coupling_logits` | `[n_lags, K, K]` | `[n_lags, K_src, K_src]` | 不变形状，但连接两个独立 codebook |
| `coupling_asymmetric` | 不存在 | `bool=False` | 机制 C 开关 |
| `coupling_logits_fwd/rev` | 不存在 | 各 `[n_lags, K_src, K_src]` | 机制 C 参数 |

#### 2.3.4 Implementation Notes

**codebook 初始化**：
- 两个 source codebook 独立初始化（各自 kmeans_init）
- 不使用 shared codebook 来初始化任何一个

**前向传播关键差异**：
```python
# 旧: joint quantization through single codebook
shared_joint = torch.cat([eeg_source, fnirs_source], dim=0)
shared_q_joint, _, _ = self.shared_quantizer(shared_joint)
eeg_source_q, fnirs_source_q = torch.split(shared_q_joint, [B, B], dim=0)

# 新: independent quantization through separate codebooks
eeg_source_q, eeg_source_idx, _ = self.eeg_source_quantizer(eeg_source)
fnirs_source_q, fnirs_source_idx, _ = self.fnirs_source_quantizer(fnirs_source)
```

### 2.4 Source Branch Target: HRF Convolution Model

这是本次 redesign 最重要的创新点之一。用神经血管耦合的物理模型替代 `smooth_signal` 代理。

#### 2.4.1 Physiological Basis

神经血管耦合的标准模型：神经活动引起局部代谢需求增加 → 血管舒张 → 脑血流增加 → HbO/HbR 浓度变化。这个过程的时序关系可以用**血流动力学响应函数 (HRF)** 描述：

$$\text{fNIRS}(t) = (\text{Neural Activity} * \text{HRF})(t) + \epsilon(t)$$

其中 $*$ 表示时序卷积，HRF 在 fNIRS/fMRI 文献中有标准参数化形式。

#### 2.4.2 HRF Model Specification

使用标准双 Gamma HRF（SPM canonical HRF 的 fNIRS 适配版本）：

$$\text{HRF}(t; \theta) = A \cdot \left(\frac{t}{\tau_1}\right)^{\alpha_1} \cdot e^{-(t - \tau_1) / \beta_1} - B \cdot \left(\frac{t}{\tau_2}\right)^{\alpha_2} \cdot e^{-(t - \tau_2) / \beta_2}$$

其中 $\theta = \{A, B, \tau_1, \tau_2, \alpha_1, \alpha_2, \beta_1, \beta_2\}$。

默认参数（基于 fNIRS 文献中的典型 HRF 形状）：
- Peak time $\tau_1 \approx 6s$（fNIRS 的 HRF peak 通常比 fMRI BOLD 略晚）
- Undershoot time $\tau_2 \approx 16s$
- 正峰幅值 $A$ 和负峰比 $B$ 由数据尺度决定

考虑到：
1. **个体 HRF 变异大**：被试间、脑区间 HRF 形状显著不同
2. **fNIRS 测量的是 HbO 和 HbR 两个信号**：两者的 HRF 形状不同（HbR 的 peak 通常更晚、更小）
3. **深层/浅层信号混合**：fNIRS 包含 systemic 和 cerebral 两种成分

当前阶段采用**简化方案**：

**方案 A (可学习参数的 HRF Convolution Target)**：
1. 从 EEG 提取宽带功率包络（或直接使用 EEG encoder 的中间表示）
2. 用可学习参数的双 Gamma HRF 核进行时序卷积
3. 卷积结果作为 source branch decoder 的 fNIRS 侧重建目标
4. HRF 核参数受软约束（初始化为典型形状，允许有限偏离）

数学形式：

给定 EEG 信号 $x_{eeg}(t)$ 和当前 batch 的 HRF 核 $h(t; \theta)$，source branch 的 fNIRS 目标为：

$$y_{source\_target}(t) = (x_{eeg}^{power} * h)(t)$$

其中 $x_{eeg}^{power}(t)$ 是 EEG 的瞬时功率包络（通过对 EEG 信号取绝对值 + 低通滤波得到，或直接使用 encoder 中间特征）。

训练时，fNIRS source decoder 的损失为：

$$\mathcal{L}_{source\_target} = \text{MSE}(\text{fNIRS}_{source\_recon}, y_{source\_target})$$

**方案 B (Learned HRF Kernel)** 作为 fallback：如果 HRF 参数化模型因个体差异过大而无法稳定训练，回退到纯学习的时序卷积核（初始化为 HRF 形状但允许更多自由度）。

#### 2.4.3 Alternative: EEG Feature → fNIRS Prediction via Band Power

作为方案 A 的一个具体实现变体，使用 EEG 频带功率作为中间特征：

1. 对 EEG 每个 patch 计算频带功率（delta/theta/alpha/beta/gamma）
2. 对每个频带的功率时间序列分别与 HRF 卷积
3. 加权求和得到 fNIRS 预测
4. 该预测作为 source branch fNIRS decoder 的目标

这种方法的优势在于：
- EEG 频带具有明确的生理意义
- 不同频带的 HRF 形状可能不同（alpha 和 gamma 的血管耦合可能有时序差异）
- 可解释性强：可以分析哪些频带对 fNIRS 预测贡献最大

但当前阶段**不采用**此变体，原因是：
- 频带定义引入了额外的超参数（频带边界、功率计算方法）
- 丢失了 EEG encoder 可能学到的非频带信息
- 增加了计算复杂度

它应作为未来的可解释性分析工具，而非训练目标的一部分。

#### 2.4.4 Where the HRF Target Applies

HRF 模型提供的是**单向**目标（EEG→fNIRS），因为只有这个方向在生理上有明确的卷积模型。因此：

| 目标 | 损失 | 权重 |
|------|------|------|
| fNIRS source decoder → HRF(EEG_power) | MSE | `source_target_weight` |
| EEG source decoder → raw EEG (coarse) | MSE | `source_target_weight * 0.5` |
| Full decoder → raw signal | MSE | 1.0 (main reconstruction) |

EEG source 使用弱辅助目标（raw EEG coarse reconstruction）是为了 prevent source codebook collapse——给它一个独立于 coupling 的训练信号，防止它退化。

### 2.5 Observation Branch Target

Observation branch 的目标从"显式残差回归"转变为"隐式重建债务"。

**旧定义**：
```python
residual_target = raw_signal - smooth_signal(raw_signal, kernel_size)
observation_loss = MSE(obs_recon, residual_target)
```

**新定义**：
Observation branch 不接受独立的显式 target。它的语义通过以下机制隐式定义：

1. **Source-only vs full reconstruction gap**：observation branch 的质量通过 ablation gap 衡量
   ```python
   gap = MSE(recon(source_only), raw) - MSE(recon(source+obs), raw)
   ```
   gap 应该显著 > 0，证明 observation branch 提供了 source 无法提供的信息。

2. **Orthogonality to source**（保留）：
   $$\mathcal{L}_{orth} = \|\text{corr}(z_{source}, z_{obs})\|^2$$
   确保 observation 不复制 source 已编码的信息。

3. **No cross-modal predictability**：从 fNIRS source token 不应能预测 EEG observation token。这是未来的诊断指标，不作为训练损失。

**Observation branch 的 codebook 使用独立的 quantizer**（与 V6 一致，保留不变）。

### 2.6 Coupling Matrix Redesign

#### 2.6.1 From Free Parameter to Constrained Mapping

旧 coupling 是一个完全自由的 `nn.Parameter`：
```python
self.coupling_logits = nn.Parameter(torch.zeros(n_lags, K, K))
```

新 coupling 保持参数化形式，但增加一个**核心生理约束**。

#### 2.6.2 The One Constraint: Concentration Prior

**生理依据**：神经血管耦合在宏观尺度上是**确定性**的——给定某种神经活动状态，血流动力学响应应当是特异的、集中的，而不是均匀分布的。如果一个 EEG source token 等概率地映射到所有 fNIRS source token，说明学到的 coupling 没有任何信息量。

**数学形式**：

设 coupling 矩阵（经 softmax 归一化后）为 $T_{lag} \in \mathbb{R}^{K \times K}$，其中 $T_{ij} = P(\text{fNIRS token}=j \mid \text{EEG token}=i, \text{lag})$。

Concentration loss 定义为行熵的负值（即鼓励每行低熵）：

$$\mathcal{L}_{conc}(T) = \frac{1}{K} \sum_{i=1}^{K} H(T_{i,:}) = -\frac{1}{K} \sum_{i=1}^{K} \sum_{j=1}^{K} T_{ij} \log T_{ij}$$

对所有 lag 取平均：

$$\mathcal{L}_{conc} = \frac{1}{n_{lags}} \sum_{l=1}^{n_{lags}} \mathcal{L}_{conc}(T_l)$$

**为什么选择行熵而不是列熵？**
- 行（EEG→fNIRS）：沿生理因果方向，每行应集中
- 列（fNIRS→EEG）：反向映射可以更分散（生理上合理，因为不同的 EEG 状态可能产生相似的 fNIRS 响应）

**为什么选择 concentration 而不是 smoothness 或 asymmetry？**

| 约束 | 优先级 | 理由 |
|------|--------|------|
| **Concentration** | P0 — 本次实现 | 最基础的生理先验——耦合的确定性。没有 concentration，coupling 可能退化为均匀分布（非信息性），后续所有分析都无意义。数学简单（单标量），实现干净。 |
| Smoothness (A) | P1 — 独立实验 | 连续性假设合理但需要在 concentration baseline 稳定后验证。依赖 codebook 邻居关系有生理意义。文档要求不与 C 同时启用。 |
| Asymmetry (C) | P1 — 独立实验 | 因果不对称性有生理依据，但需要 concentration baseline 先通过。文档要求不与 A 同时启用。 |

**浓度先验的可调参数**：

```yaml
loss:
  coupling:
    concentration_weight: 0.005    # 从小系数开始 sweep: [0.001, 0.005, 0.01]
```

#### 2.6.3 What Is Deliberately Omitted

以下约束被明确排除在当前设计之外：

1. **Coupling smoothness** — 保留为机制 A，作为独立实验
2. **Causal asymmetry parameterization** — 保留为机制 C，作为独立实验
3. **Codebook correspondence loss** — 耦合矩阵可以隐式学习对应关系，额外的 codebook-level 对齐是冗余的
4. **Latent alignment loss** — 在 dual codebook 架构下，EEG 和 fNIRS source latent 处于不同空间，MSE 对齐没有意义
5. **Assignment alignment loss** — TokenFlow analysis 已确认"同 token identity"不应是目标

### 2.7 Complete Loss Function

> **Last revised**: 2026-05-11 — Phase 2A revision: added observation_loss, moved smoothness to Phase 2B, updated weights

#### 2.7.1 Loss Terms Specification

```python
total_loss = (
    # === Gate 1 (Health): Reconstruction ===
    eeg_rec_loss +                         # EEG full reconstruction (source + observation sum)
    fnirs_rec_loss +                       # fNIRS full reconstruction (source + observation sum)
    vq_loss +                              # VQ commitment loss for all quantizers
    
    # === Gate 2 (Semantics): Branch Roles ===
    alpha_source_target * source_target_loss +    # fNIRS source decoder → HRF(EEG_power)
    alpha_source_target * eeg_source_aux_weight * eeg_source_aux_loss +  # EEG source decoder → power envelope
    alpha_obs * observation_loss +                # observation decoder → (original - source_target) (NEW)
    alpha_orth * orthogonality_loss +              # source ⊥ observation
    
    # === Gate 3 (Structure): Coupling with Physiological Prior ===
    alpha_coupling * coupling_kl_loss +            # basic coupling training
    alpha_conc * concentration_loss +              # row entropy penalty (P0)
    alpha_smooth * smoothness_loss +               # neighbor JS divergence (P1, moved from Phase 4)
    
    # === Codebook Health ===
    alpha_balance * codebook_balance_loss          # utilization / perplexity
)
```

**Loss term reference table**:

| 符号 | 损失项 | 来源 | 说明 |
|------|--------|------|------|
| `eeg_rec_loss` | EEG reconstruction | V6 保留 | source + observation sum vs original |
| `fnirs_rec_loss` | fNIRS reconstruction | V6 保留 | source + observation sum vs original |
| `vq_loss` | VQ commitment | V6 保留 | across all 4 quantizers |
| `source_target_loss` | fNIRS source HRF target | Phase 2 新增 | fNIRS source decoder → HRF(EEG_power_envelope) |
| `eeg_source_aux_loss` | EEG source target | Phase 2A 重定义 | EEG source decoder → power envelope @ full resolution |
| `observation_loss` | Observation residual target | **Phase 2A 新增** | observation decoder → (original - source_target) |
| `orthogonality_loss` | Source ⊥ Observation | V6 保留 | cross-correlation penalty |
| `coupling_kl_loss` | Coupling KL | V6 保留 | P(fNIRS token \| EEG token) vs actual |
| `concentration_loss` | Coupling row entropy | Phase 2B 新增 (P0) | encourages sparse/concentrated coupling |
| `smoothness_loss` | Coupling neighbor JS | Phase 2B 新增 (P1) | encourages local smoothness in coupling rows |
| `codebook_balance_loss` | Codebook utilization | V6 保留 | entropy-based balance |

**Removed from V6**:

| 已删除项 | 删除理由 |
|----------|----------|
| `latent_align_loss` | Dual codebook 架构下，source latent 在不同空间，MSE 无意义 |
| `assignment_align_loss` | TokenFlow analysis 确认不应追求同 token identity |
| `hard_assignment_align_loss` | 同上 |
| `shared_entropy_loss` | 功能被 `codebook_balance_loss` 覆盖 |
| `private_entropy_loss` | 功能被 `codebook_balance_loss` 覆盖 |
| `shared_eeg_common_loss` | 被 `source_target_loss` (HRF model) 替代 |
| `shared_fnirs_common_loss` | 被 `source_target_loss` (HRF model) 替代 |
| `eeg_private_residual_loss` | 被 `observation_loss` 替代（但 target 改为 original - source_target，不再是 smooth residual） |
| `fnirs_private_residual_loss` | 被 `observation_loss` 替代（同上） |
| `shared_eeg_recon_loss` | Source decoder 已有 explicit target，不需要重复 loss |
| `shared_fnirs_recon_loss` | 同上 |

**Loss count**: V6 = 12 → New design = 11 (observation_loss 恢复但语义已改为 explicit residual; concentration + smoothness 新增)

#### 2.7.2 Default Weights (Phase 2A/B Target)

```yaml
loss:
  reconstruction:
    eeg_amplitude_weight: 1.0
    eeg_time_weight: 0.9
    fnirs_amplitude_weight: 1.0
    fnirs_time_weight: 1.0
  
  source_target:
    weight: 0.3                      # Phase 2A: 0.15 → 0.3
    eeg_source_aux_weight: 1.0       # Phase 2A: 0.5 → 1.0 (target now meaningful)
    warmup_epochs: 30
  
  observation_target:
    weight: 0.15                     # Phase 2A: NEW
    warmup_epochs: 30
  
  coupling:
    weight: 0.07                     # coupling_kl_loss (retained)
    concentration_weight: 0.01       # Phase 2B: sweep [0.005, 0.01, 0.02]
    smoothness_weight: 0.002         # Phase 2B: sweep [0.001, 0.002, 0.005]
    smoothness_neighbors: 5
    smoothness_warmup_epochs: 30     # enable after concentration stabilizes
    bidirectional: true
  
  branch:
    orthogonality_weight: 0.05       # Phase 2A: 0.01 → 0.05
  
  codebook:
    balance_weight: 0.08
    source_balance_scale: 1.0
    observation_balance_scale: 0.5   # Phase 2A: 0.0 → 0.5
```

#### 2.7.3 Coupling Loss Triplet: Potential Conflicts and Monitoring

**⚠️ 这三个 loss 作用在同一 `coupling_logits` 矩阵上，存在理论张力：**

| Loss | 推向 | 极端后果 |
|------|------|---------|
| `coupling_kl_loss` | 匹配数据中的 token 共现统计 | 均匀分布（如当前 Phase 2 结果） |
| `concentration_loss` | 每行低熵 | 每行坍缩为 one-hot |
| `smoothness_loss` | 相邻行相似 | 所有行坍缩为同一分布 |

**冲突场景：**
- concentration 强 + smoothness 弱 → 每行 one-hot 但相邻行可能指向不同 token（失去平滑结构）
- smoothness 强 + concentration 弱 → 所有行坍缩到同一分布（失去区分度）
- 两者都强 → 所有行坍缩到同一个 one-hot（coupling 彻底退化）
- coupling_kl_loss 过弱 → 数据信号被先验覆盖，学不到真实跨模态结构

**必须监控的指标（训练时实时输出到 TensorBoard/log）：**

1. `concentration_loss` 时间序列 — 应下降后稳定，不应持续下降至零
2. `smoothness_loss` 时间序列 — 应下降后稳定
3. `source_coupling_loss` 时间序列 — **不应显著上升**（如上升说明先验压倒数据）
4. Per-row entropy 分布直方图 — 应集中在 [0.5×logK, 0.8×logK]，不应坍缩到接近零或接近 logK
5. Neighbor JS divergence vs random pair JS divergence — 前者应显著低于后者
6. **constraint_balance_ratio (CBR)** = `concentration_loss / (coupling_kl_loss + 1e-8)` — 健康范围 0.1–2.0；CBR > 5.0 表示 concentration 过强
7. Coupling heatmap — 应呈现可辨识的集中结构

详见 [IMPLEMENTATION_PLAN.md §6.5](../../IMPLEMENTATION_PLAN.md) 中"耦合三项 Loss 的潜在冲突与监控要求"。

### 2.8 Diagnostic Metrics for Branch Semantics Validation

以下诊断指标用于验证 source/observation 分支的重定义是否成功。这些指标不作为训练损失，仅用于监控和 gate decision。

#### 2.8.1 Source Branch Diagnostics

| 指标 | 计算 | 健康范围 |
|------|------|----------|
| **Coupling row entropy** | $H_{row} = -\frac{1}{K}\sum_{i,j} T_{ij}\log T_{ij}$ | [0.5×logK, 0.8×logK]（集中但不坍缩） |
| **Coupling concentration ratio** | $\frac{\max_j T_{ij}}{\text{mean}_j T_{ij}}$ averaged over rows | > 1.5（行有峰值） |
| **Source target reconstruction (fNIRS)** | MSE(fNIRS_source_recon, HRF_target) | 随训练下降 |
| **Source target reconstruction (EEG)** | MSE(eeg_source_recon, power_envelope) | 随训练下降 |
| **Cross-modal token predictability** | P(fNIRS_token \| EEG_token) top-1 accuracy | > 1/K (random baseline) |

#### 2.8.2 Observation Branch Diagnostics

| 指标 | 计算 | 健康范围 |
|------|------|----------|
| **Observation contribution gap** | MSE(source_only) - MSE(source+obs) | > 0（observation 有正贡献） |
| **Observation target reconstruction** | MSE(obs_recon, original - source_target) | 随训练下降 |
| **Observation cross-modal unpredictability** | MI(EEG_obs_token, fNIRS_source_token) | 接近 0（观测是模态特异的） |
| **Subject leakage ratio** | MI(EEG_obs, subject_id) / MI(EEG_source, subject_id) | > 1.0（subject 信息集中在 observation） |

#### 2.8.3 Coupling Health Diagnostics (Phase 2B)

| 指标 | 计算 | 健康范围 |
|------|------|----------|
| **concentration_loss** | Row entropy mean over all lags | 下降后稳定，不接近零 |
| **smoothness_loss** | Neighbor JS divergence | 下降后稳定 |
| **constraint_balance_ratio (CBR)** | concentration_loss / (coupling_kl_loss + 1e-8) | [0.1, 2.0] |
| **Neighbor vs random JS gap** | JS_random - JS_neighbor | > 0（邻居更相似） |
| **Per-row entropy histogram** | Distribution of H(T_i,:) over i | 集中在 [0.5×logK, 0.8×logK] |

#### 2.8.4 Architecture Comparison Metrics

| 指标 | 用于比较 |
|------|----------|
| Source codebook perplexity (EEG vs fNIRS) | 两个 source codebook 各自健康度 |
| Observation codebook perplexity (EEG vs fNIRS) | 扩容后 observation codebook 是否健康 |
| Coupling structure visualization | 热力图：按行熵排序后的 coupling matrix |
| Ablation: source-only vs observation-only vs full | 比较三种模式的 gap 大小 |

### 2.9 Implementation Phases (Revised 2026-05-11)

```
Phase 1: Structural Migration ✅ Complete
  ├── 拆分 shared_quantizer → eeg_source_quantizer + fnirs_source_quantizer
  ├── 重命名参数和方法: shared→source, private→observation
  ├── 删除已废弃的 loss terms
  └── Gate 1 (Health) — 已通过

Phase 2: Source Target Introduction ✅ Implemented (Gate 2-4 fail)
  ├── 实现 HRF 卷积模型
  ├── 新增 source_target_loss (fNIRS source → HRF target)
  ├── 新增 eeg_source_aux_loss (EEG source → coarse raw EEG)
  └── 问题：coupling 均匀坍塌、source target 语义不统一、observation 不可辨识

Phase 2A: Branch Target Redesign + Dual Decoder 🔜 ACTIVE
  ├── 新增 4 个独立 decoder（source/obs 分离）
  ├── 重定义 source target：统一为 EEG power envelope driver
  ├── 新增 observation_loss：explicit residual target
  ├── 扩容 observation codebook（32→64）
  ├── Loss 权重重新平衡
  └── Gate 2 (Semantics) — 当前阻塞目标

Phase 2B: Coupling Structure Priors
  ├── 实现 concentration_loss (row entropy penalty) [P0]
  ├── 实现 smoothness_loss (neighbor JS divergence) [P1, 从 Phase 4 提前]
  ├── 监控 coupling loss triplet 冲突（CBR, per-row entropy histogram, neighbor JS）
  ├── Smoothness warmup：concentration 稳定后启用
  └── Gate 3 (Structure) — coupling row entropy < log(K)/2, concentration_ratio > 1.5

Phase 2C (延后): Cross-Modal Source Target + Coupling-Aware Quantization
  ├── fNIRS→EEG 预测器（让 EEG source target 包含 fNIRS 侧信息）
  ├── Coupling-aware quantization（原 Phase 2A）
  └── Gate 2A (Quantization-Coupling Consistency)

Phase 3 (延后): Mechanism C — Causal Asymmetry
  ├── 独立 fwd/rev coupling 参数化
  └── Gate: asymmetry_ratio > 1.0, Gate 1 不退化

Phase 4 (延后): Structural Assumption Audit
  ├── 评估 equal token count per window 是否应松动
  ├── 评估 patch boundary alignment 是否必要
  └── 评估 source codebook size ratio (EEG vs fNIRS) 的最优配置
```

---

## 3. Mechanism A: Token-Space Coupling Smoothness

> **Status update (2026-05-11)**：本机制已从独立 Phase 4 提前至 Phase 2B，与 concentration prior 在同一阶段实现。以下数学规范和实现细节保留不变，作为 Phase 2B 中 smoothness 部分的技术参考。smoothness_weight 从 0.002 开始 sweep，显著小于 concentration_weight (0.01)。

### 3.1 Physiological basis

神经血管耦合的基本性质：**相近的神经活动状态引起相近的血流动力学响应**。

如果两个 EEG token 在 source codebook 空间中代表相似的神经状态，它们经由 coupling 矩阵映射到的 fNIRS token 分布也应该是相似的。当前自由参数化的 coupling 矩阵不保证这一性质——两个 codebook 向量几乎相同的 token 可能学到完全不同的耦合分布。

### 3.2 Mathematical formulation

设 shared codebook 的归一化权重为 $C \in \mathbb{R}^{K \times D}$。对每个 token $i$，找到其在 codebook 空间中的 $M$ 个最近邻 $\mathcal{N}(i)$（基于余弦相似度）。

对于给定的 lag，coupling 矩阵 $T = \text{softmax}(\text{coupling\_logits}[lag]) \in \mathbb{R}^{K \times K}$（行随机矩阵），定义平滑性损失：

$$\mathcal{L}_{smooth}(T) = \frac{1}{K \cdot M} \sum_{i=1}^{K} \sum_{j \in \mathcal{N}(i)} D_{JS}\big(T_{i,:} \,\|\, T_{j,:}\big)$$

其中 $D_{JS}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)$，$M = (P + Q) / 2$。

**选择 JS 散度而非 L2 的理由**：
- $T_{i,:}$ 是概率分布，JS 散度有界且对称
- JS 散度对低概率区域的微小波动不敏感，避免被噪声 token 主导

**选择局部邻居而非全局平滑的理由**：
- 不假设 codebook 的全局拓扑结构已经学到生理上有意义的组织
- 局部约束更稳健：只要求最相似的 token 有相似耦合行为
- 计算量可控（M=5 时每个 token 只计算 5 个 pair）

### 3.3 Implementation

新文件或修改位置：

**`src/losses/multimodal_tokenizer.py`** — 新增函数：

```python
def coupling_smoothness_loss(
    coupling_logits: torch.Tensor,       # [n_lags, K, K]
    codebook_weight: torch.Tensor,       # [K, D]
    n_neighbors: int = 5,
) -> torch.Tensor:
    """Encourage tokens with similar codebook vectors to have similar coupling profiles."""
    n_lags, K, _ = coupling_logits.shape
    
    # Find local neighbors in codebook space
    normed = F.normalize(codebook_weight, dim=-1)
    sim = normed @ normed.t()
    _, neighbors = sim.topk(n_neighbors + 1, dim=-1)
    neighbors = neighbors[:, 1:]  # [K, n_neighbors], exclude self
    
    total_loss = normed.new_tensor(0.0)
    for lag_idx in range(n_lags):
        T = F.softmax(coupling_logits[lag_idx], dim=-1)        # [K, K]
        T_neighbors = T[neighbors]                              # [K, M, K]
        T_i = T.unsqueeze(1)                                    # [K, 1, K]
        M = 0.5 * (T_i + T_neighbors)                           # [K, M, K]
        js = 0.5 * (
            F.kl_div((T_i + 1e-8).log(), M, reduction='none').sum(dim=-1) +
            F.kl_div((T_neighbors + 1e-8).log(), M, reduction='none').sum(dim=-1)
        )
        total_loss = total_loss + js.mean()
    
    return total_loss / n_lags
```

**`src/tokenizers/factorized_labram_vqnsp.py`** — 新增参数：

```python
# In __init__:
coupling_smoothness_weight: float = 0.0,
coupling_smoothness_neighbors: int = 5,
```

**`compute_factorized_shared_alignment_losses`** — 新增返回：

```python
'smoothness_loss': coupling_smoothness_loss(...) if enabled else zero_tensor
```

### 3.4 Expected behavioral signatures

| 指标 | 预期变化 | 验证方式 |
|------|----------|----------|
| $\mathcal{L}_{smooth}$ | 随训练下降 | 直接监控 |
| Coupling 矩阵可视化 | 按 codebook 相似度排序后呈现平滑结构 | TensorBoard image |
| Token neighborhood coupling consistency | 邻居 token 的耦合分布 JS 散度低于随机基线 | 定量比较 |
| Reconstruction | 无显著变化 | MSE / STFT |
| Codebook health | 无显著退化（可能轻微改善，因为耦合结构更清晰） | Perplexity / utilization |

### 3.5 Failure modes

1. **Codebook 未收敛时无意义**：如果 codebook 本身还在剧烈变化，邻居关系不稳定，平滑性约束会引入噪声。缓解：在 reconstruction 稳定后再 warm-start 此约束。
2. **系数过大导致所有 token 耦合相同**：如果 $\lambda_{smooth}$ 过大，所有行收敛到相同分布。缓解：从小系数（0.005）开始，监控行间方差。
3. **Codebook collapse 时退化为无操作**：如果所有 codebook 向量都相似，邻居没有意义。缓解：此约束假设 Gate 1 (Health) 已通过。

---

## 4. Mechanism C: Causal Direction Asymmetry

### 4.1 Physiological basis

神经血管耦合的因果方向在试次时间尺度（~10s）上是明确的：

- **EEG → fNIRS**：电活动 → 代谢需求 → 血管扩张 → HRF（延迟 2-8s），这是主要的因果通路
- **fNIRS → EEG**：反向因果在此时窗内很弱。血管状态对神经兴奋性的调节（通过 CO₂、pH）发生在更慢的时间尺度上

当前 V6 的 bidirectional coupling 实现中，反向耦合使用前向耦合矩阵的转置（[factorized_labram_vqnsp.py:251](src/tokenizers/factorized_labram_vqnsp.py#L251)）：

```python
reverse_transition = F.softmax(coupling_logits[lag_index].transpose(0, 1), dim=-1)
```

这意味着 $P(\text{fNIRS}_j \mid \text{EEG}_i) \propto P(\text{EEG}_i \mid \text{fNIRS}_j)$——两个方向的耦合共享同一组参数。这在生理上是不合理的：EEG→fNIRS 的预测结构应该比 fNIRS→EEG 更集中、更有组织性。

### 4.2 Design principle: asymmetric prior, not asymmetric loss

不引入显式的"前向必须比反向更集中"的损失项。而是：

1. 为两个方向使用**独立参数矩阵**（`coupling_logits_fwd` 和 `coupling_logits_rev`）
2. 仅在**前向**（EEG→fNIRS）施加结构约束（如机制 A 的平滑性）
3. 反向保持自由参数化，让数据决定其结构
4. 通过诊断指标（asymmetry ratio）观察两个方向的差异

这种"不对等先验"方案比显式不对称损失更干净：它不给优化器增加对抗性约束，而是通过**不对等的参数化自由度和正则化水平**让生理结构自然浮现。

### 4.3 Mathematical formulation

**参数独立化**：

前向（EEG → fNIRS）：
$$T^{fwd}_l = \text{softmax}(W^{fwd}_l), \quad W^{fwd}_l \in \mathbb{R}^{K \times K}$$

反向（fNIRS → EEG）：
$$T^{rev}_l = \text{softmax}(W^{rev}_l), \quad W^{rev}_l \in \mathbb{R}^{K \times K}$$

其中 $W^{fwd}_l$ 和 $W^{rev}_l$ 是独立参数，初始化为零。

**不对等处理**：

前向耦合损失（可加机制 A 平滑约束）：
$$\mathcal{L}_{fwd} = \mathcal{L}_{coupling}(T^{fwd}) + \lambda_{smooth} \cdot \mathcal{L}_{smooth}(T^{fwd})$$

反向耦合损失（自由参数化，无结构约束）：
$$\mathcal{L}_{rev} = \mathcal{L}_{coupling}(T^{rev})$$

总耦合损失：
$$\mathcal{L}_{coupling}^{total} = 0.5 \cdot (\mathcal{L}_{fwd} + \mathcal{L}_{rev})$$

**诊断指标**：

$$\text{asymmetry\_ratio} = \frac{\mathbb{E}_k[H(T^{rev}_{k,:})]}{\mathbb{E}_k[H(T^{fwd}_{k,:})]}$$

其中 $H(\cdot)$ 是行分布的熵。期望 asymmetry_ratio > 1.0（反向比前向更分散）。

### 4.4 Implementation

**`src/tokenizers/factorized_labram_vqnsp.py`** — 参数变更：

```python
# In __init__:
coupling_asymmetric: bool = False,  # toggle for mechanism C

# Replace single coupling_logits:
self.coupling_logits = nn.Parameter(...)  # kept for backward compat when asymmetric=False

# If asymmetric:
self.coupling_logits_fwd = nn.Parameter(
    torch.zeros(len(self.alignment_lag_candidates), shared_codebook_size, shared_codebook_size)
)
self.coupling_logits_rev = nn.Parameter(
    torch.zeros(len(self.alignment_lag_candidates), shared_codebook_size, shared_codebook_size)
)
```

**`src/losses/multimodal_tokenizer.py`** — 修改 `compute_factorized_shared_alignment_losses`：

```python
def compute_factorized_shared_alignment_losses(
    ...,
    coupling_logits: torch.Tensor | None = None,       # legacy shared param
    coupling_logits_fwd: torch.Tensor | None = None,    # mechanism C: forward
    coupling_logits_rev: torch.Tensor | None = None,    # mechanism C: reverse
    coupling_asymmetric: bool = False,
    ...
):
    for lag_index, lag in enumerate(alignment_lag_candidates):
        ...
        # Forward coupling
        if coupling_asymmetric:
            transition_fwd = F.softmax(coupling_logits_fwd[lag_index], dim=-1)
        else:
            transition_fwd = F.softmax(coupling_logits[lag_index], dim=-1)
        pred_fnirs_probs = torch.einsum('bnk,kl->bnl', aligned_eeg_probs, transition_fwd)
        coupling_loss = coupling_kl_loss(pred_fnirs_probs, aligned_fnirs_probs)
        
        # Reverse coupling
        if coupling_bidirectional:
            if coupling_asymmetric:
                transition_rev = F.softmax(coupling_logits_rev[lag_index], dim=-1)
            else:
                transition_rev = F.softmax(coupling_logits[lag_index].transpose(0, 1), dim=-1)
            pred_eeg_probs = torch.einsum('bnk,kl->bnl', aligned_fnirs_probs, transition_rev)
            coupling_loss = 0.5 * (coupling_loss + coupling_kl_loss(pred_eeg_probs, aligned_eeg_probs))
        ...
```

**诊断指标**（在 tokenizer forward 中新增）：

```python
if self.coupling_asymmetric:
    with torch.no_grad():
        T_fwd = F.softmax(self.coupling_logits_fwd[selected_lag_idx], dim=-1)
        T_rev = F.softmax(self.coupling_logits_rev[selected_lag_idx], dim=-1)
        h_fwd = -(T_fwd * (T_fwd + 1e-8).log()).sum(dim=-1).mean()
        h_rev = -(T_rev * (T_rev + 1e-8).log()).sum(dim=-1).mean()
        asymmetry_ratio = h_rev / (h_fwd + 1e-8)
```

### 4.5 Expected behavioral signatures

| 指标 | 预期变化 | 验证方式 |
|------|----------|----------|
| asymmetry_ratio | 稳定 > 1.0 | 直接监控 |
| EEG→fNIRS coupling per-row entropy | 低于反向 | 分布直方图 |
| CMTP EEG→fNIRS vs fNIRS→EEG | 前向优于反向 | 下游预测任务 |
| Reconstruction | 无显著变化 | MSE / STFT |
| Codebook health | 无显著退化 | Perplexity / utilization |

### 4.6 Failure modes

1. **asymmetry_ratio ≈ 1.0**：数据中两个方向的信息流确实对称，或反向参数学到的结构与前向类似。这不是严格意义上的"失败"——它说明数据不支持神经血管耦合的不对称假设。这仍然是有价值的发现。
2. **反向耦合退化**：如果 fNIRS→EEG 的耦合损失变得很大（远大于前向），可能是因为反向参数未被充分优化。缓解：确保两个方向的 coupling loss权重相同，不对反向施加额外的压制。

---

## 5. Experimental Design

### 5.1 Independent experiments (not combined)

在当前研究阶段，机制 A 和机制 C **分别进行实验**，不组合使用。每个机制的实验独立于 V6 baseline 进行比较。

### 5.2 Experiment ladder

```
                        ┌── V6 Baseline (current mainline)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   Exp A1           Exp C1          (future)
   coupling_        coupling_       A + C
   smoothness_      asymmetric=     combined
   weight sweep     True
        │               │
   Exp A2           Exp C2
   A + warm-        C + A
   start schedule   smoothness
```

### 5.3 Exp A: Coupling smoothness

**Config changes** (relative to V6 baseline):

```yaml
loss:
  alignment:
    coupling_smoothness_weight: [0.005, 0.01, 0.02]  # sweep
    coupling_smoothness_neighbors: 5
    coupling_asymmetric: false
```

**Warm-start schedule**: 在 reconstruction 稳定后（通常 epoch 20-30）才启用 smoothness 约束：

```yaml
loss:
  alignment:
    coupling_smoothness_warmup_epochs: 30
    coupling_smoothness_final_weight: 0.01
```

**Comparison metrics vs. V6 baseline**:

1. Gate 1 (Health): reconstruction (full/common/residual), codebook health
2. Gate 2 (Semantics): intra-token consistency, prototype separation ratio
3. Gate 3 (Structure): best-lag MI, conditional KL gain, coupling matrix structure
4. Gate 4 (Utility): subject leakage, task signal

**Decision gate**:

- ✅ Pass: Gate 1 (Health) 不退化 + coupling 矩阵呈现可辨识的平滑结构 + ≥1 项 Gate 3 (Structure) 指标改善
- ❌ Fail: Gate 1 (Health) 退化，或 coupling 矩阵无明显结构改善，或无任何 Gate 2/3 指标改善

### 5.4 Exp C: Causal asymmetry

**Config changes** (relative to V6 baseline):

```yaml
loss:
  alignment:
    coupling_asymmetric: true
    coupling_bidirectional: true  # keep bidirectional, but with separate params
```

**Comparison metrics vs. V6 baseline**:

1. Gate 1 (Health): reconstruction, codebook health
2. Gate 3 (Structure): asymmetry_ratio, forward vs. reverse coupling entropy, CMTP direction comparison
3. Gate 4 (Utility): subject leakage, task signal

**Decision gate**:

- ✅ Pass: asymmetry_ratio 稳定 > 1.0 + Gate 1 (Health) 不退化 + 前向 CMTP 优于反向
- ⚠️ Inconclusive: asymmetry_ratio ≈ 1.0 但 Gate 1/3 (Health/Structure) 不退化（说明数据不支持不对称先验）
- ❌ Fail: Gate 1 (Health) 退化

### 5.5 What NOT to do

- ❌ 同时启用机制 A 和机制 C（当前阶段）
- ❌ 在 shared codebook baseline 上测试这些机制（它们依赖 factorization）
- ❌ 在没有 warm-start 的情况下直接启用以 reconstruction 为主的 run
- ❌ 把 coupling 结构改善当作唯一的成功指标——Gate 1 (Health)（reconstruction/codebook health）是前提

---

## 6. Integration Plan

### 6.1 Code changes summary

| 文件 | 变更 | 机制 |
|------|------|------|
| `src/losses/multimodal_tokenizer.py` | 新增 `coupling_smoothness_loss()` | A |
| `src/losses/multimodal_tokenizer.py` | `compute_factorized_shared_alignment_losses` 支持独立 fwd/rev coupling logits | C |
| `src/tokenizers/factorized_labram_vqnsp.py` | 新增 `coupling_smoothness_weight`, `coupling_smoothness_neighbors` | A |
| `src/tokenizers/factorized_labram_vqnsp.py` | 新增 `coupling_asymmetric`, `coupling_logits_fwd`, `coupling_logits_rev` | C |
| `src/tokenizers/factorized_labram_vqnsp.py` | Forward 中新增 asymmetry_ratio 诊断 | C |
| `src/tokenizers/codebook_focus_factorized_labram_vqnsp.py` | 透传新参数（默认 smoothness=0, asymmetric=False） | A+C |
| Config YAML | 新增 `coupling_smoothness_*` 和 `coupling_asymmetric` 字段 | A+C |

### 6.2 Backward compatibility

- `coupling_smoothness_weight=0.0` → 行为与 V6 完全相同（机制 A 默认关闭）
- `coupling_asymmetric=False` → 行为与 V6 完全相同（机制 C 默认关闭）
- 现有 config 无需修改即可运行

---

## 7. Success Criteria & Promotion Rule

任何机制要进入默认 mainline，必须满足（继承自 V6 reset 的 promotion rule）：

1. Gate 1 (Health) codebook health 不退化
2. Gate 2 (Semantics) branch role quality 不退化，最好有明确提升
3. Gate 3 (Structure) coupling structured value 有明确提升
4. Gate 4 (Utility) invariance / downstream sanity 不出现明显倒退
5. 可以通过 ablation 解释，且不把 source branch 变回第二条全能重建捷径

---

## 8. Relationship to Foundation Model

本计划聚焦于 tokenizer 层面的 coupling 约束。与 foundation model pretraining 的关系：

- Tokenizer 的 coupling 约束提供 token 级的生理结构化先验
- Foundation model 的跨模态目标（当前为 InfoNCE）可随后调整为利用 coupling 先验的 Cross-modal Masked Token Prediction
- Tokenizer 层的约束和 pretraining 层的调整是**正交的**：当前实验先验证 tokenizer 层约束的效果，再决定 pretraining 层是否需要调整

详见后续文档（待 A/C 实验结果后撰写）。

---

## Appendix: Rejected Approaches

### HRF-shaped lag weighting

考虑过使用 SPM  HRF 形状的先验权重替代当前的 `alignment_selection='min'`，使不同时间偏移的耦合损失按 HRF 幅度加权。**放弃理由**：

- 类似物理约束神经网络的经验表明，硬编码的波形形状先验在实际数据上往往不匹配（个体间 HRF 变异性大，被试-被试、试次-试次差异显著）
- 10s 窗口仅 5 patches (lag 0-4)，离散化后的 HRF 先验信息量有限
- 当前 `alignment_selection='min'` 已经允许模型在每个 batch 选择最优 lag，不需要额外的 temporal 先验
- 如果将来需要，可作为扩展方向重新评估

### Explicit asymmetry loss

考虑过显式约束 `H(T_fwd) < H(T_rev)` 的 margin-based loss。**放弃理由**：

- 额外的对抗性约束增加了优化复杂度
- "不对等先验"方案（独立参数 + 仅对前向加结构约束）在概念上更干净
- 如果不对等先验已能产生 asymmetry_ratio > 1.0，显式损失是冗余的
