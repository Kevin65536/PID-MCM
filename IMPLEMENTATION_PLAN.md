# PID-MCM Implementation Plan (Tokenization → Alignment → PID)

> **Last Updated**: 2026-01-05  
> **Status**: Roadmap v2 (Tokenizer-First)  
> **Theory Reference**: [`docs/THEORY.md`](docs/THEORY.md)

---

## 0. Overview

本路线将重心从“直接学习 PID 分解（ELP-first）”调整为：

1) **Neuro-Tokenization**：将 EEG / fNIRS 映射为离散 token（codebook）
2) **Semantic Alignment**：让 token 具备可解释语义（生理节律、Pseudo-Trace、空间模式、潜在脑状态等）
3) **PID-based Analysis & Refinement**：在稳定 codebook 之上做信息分解与解释

关键判断：如果 codebook 本身不稳定或不可复现，任何 PID 结论都会漂移；因此先把“表示”做稳。

---

## 1. Design Decisions (先固定主线，减少发散)

### 1.1 主线架构（推荐）：Separate Codebooks + Shared Semantic Projector

- EEG 与 fNIRS **各自**拥有 tokenizer（各自 codebook / quantizer）
- 通过共享 projector 将各自 token embedding 映射到共同语义空间 $S$，再在 $S$ 上做对齐

**好处**：能自然处理 EEG 与 fNIRS 的采样率/时标差异，并降低“共享 codebook 被分布差异拖垮”的风险。

### 1.2 对照架构（后续）：Shared Codebook + Modality Adapters

- 作为 ablation/探索项保留
- 仅在 Phase 1/2 主线稳定后再尝试

---

## 2. Phase 0: Data Contract & Preprocessing Spec（必须先写清楚）

**目标**：定义跨实验一致的数据接口、窗口化策略与对齐规则，避免模型学到“预处理差异”。

### 2.1 Data Contract（交付物）

- EEG：采样率、通道顺序、参考方式、滤波带宽、artifact 处理、缺失通道策略
- fNIRS：HbO/HbR 转换、滤波、运动伪影处理、通道布局/探头信息
- 同步：时间戳对齐误差处理、窗口中心定义、允许的时间偏移范围

### 2.2 Windowing & Time-Scale Alignment（最小可行方案）

- 统一训练样本单位为 **window**：长度 $W$（建议 2–8s 可配），步长 $H$
- EEG：按窗口切片，可选重采样到统一频率（如 200Hz）
- fNIRS：保留低频特性，允许更低采样率；与 EEG window 在时间上对齐

### 2.3 Augmentations（写进配置）

- EEG：time masking、band-stop/low-pass jitter、channel dropout、轻微 time shift
- fNIRS：motion-like spike、baseline drift、time masking

**Success Criteria**：同一配置下，数据管线可复现（seed 固定），不同实验共享同一输入规范。

---

## 3. Phase 1: Neuro-Tokenization（构建可用 codebook）

**目标**：得到稳定、可泛化的离散 token 表示（EEG tokens / fNIRS tokens）。

### 3.1 Tokenizer Methods（实验对象）

- VQ-VAE（baseline）
- RVQ / Residual VQ（提高表达能力）
- FSQ / Finite Scalar Quantization（实现简单、梯度更稳定，适合起步）

### 3.2 训练目标（从“可测量”开始）

#### A. Reconstruction（必须）

- 时域：MSE / Huber
- 频域：多尺度 STFT loss 或 bandpower loss（EEG 重点在频带能量）
- fNIRS：强调低频形态与平滑性（可加入二阶差分平滑惩罚）

#### B. Codebook Health（必须）

- code usage / perplexity
- top-k usage 覆盖度（避免少数 code 独占）
- collapse 检测：embedding 方差、有效秩、commitment loss 异常

#### C. Stability（建议）

- augmentation 一致性：同一 window 的增强前后 token 分布/embedding 相似
- subject generalization：跨被试 held-out 不崩（至少不发生 codebook collapse）

### 3.3 Phase 1 最小实验矩阵

| Exp | Quantizer | Recon Loss | 频域约束 | 目标 |
| :---: | :---------: | :----------: | :--------: | :----- |
| T0 | FSQ | MSE | 无 | 快速可跑 baseline |
| T1 | VQ-VAE | MSE | MS-STFT | 更强频谱保真 |
| T2 | RVQ | MSE | MS-STFT | 提升表达能力 |

**Success Criteria**（至少满足其一组稳定条件）：

- recon 达到可接受水平（相对基线显著下降）
- perplexity 不塌陷（> 目标阈值；阈值按 codebook size 设定）
- 训练后 3 次不同 seed 的 usage 分布相似（稳定性）

---

## 4. Phase 2: Semantic Alignment（让 token“有意义”）

**目标**：在 token embedding 或其投影空间中，注入可解释的结构。

> 核心原则：先做“最小可验证对齐”，再叠加更强先验（Pseudo-Trace / 生理节律 / 空间）。

### 4.1 对齐对象（明确）

- 主要：token embedding 经 projector 映射到 $S$（共享语义空间）
- 备选：token index 的分布（bag-of-codes / n-gram）

### 4.2 对齐目标分类（您提出的四类）

#### Type A: True Brain Activity / Brain State（弱监督或自监督）

- 如果有任务段/睡眠分期/范式标签：做轻量监督 probe（不反传到 tokenizer 亦可）
- 如果无标签：做跨增强一致性、跨模态一致性（InfoNCE / VICReg-style）

#### Type B: Physiological Rhythm（“对齐”还是“去除”需要区分）

- 方案 1（去除）：把可预测的全局低频/心跳成分建模为 nuisance，并对抗抑制其进入语义空间
- 方案 2（显式编码）：单独开一个 physiological head/token 分支，避免污染 brain-state token

#### Type C: Pseudo-Trace（模板先验，先弱后强）

- fNIRS：canonical HRF / 简化 HRF family 作为弱正则（shape correlation / temporal smoothness）
- EEG：ERP-like 模板仅作为“可选正则”，避免过强错误先验

#### Type D: Spatial Pattern（拓扑一致性）

- EEG：10-20 邻接图平滑 / 通道一致性
- fNIRS：探头邻接图平滑 / ROI-level pooling

### 4.3 Phase 2 最小实验矩阵

| Exp | Alignment | Physiological | Pseudo-Trace | Spatial | 目标 |
| :---: | :---------: | :-------------: | :------------: | :-------: | :----- |
| A0 | InfoNCE (S-space) | 无 | 无 | 无 | 建立跨模态一致性基线 |
| A1 | A0 + nuisance 去除 | 去除 | 无 | 无 | 让 brain token 更纯 |
| A2 | A0 + 弱模板正则 | 无 | 弱 | 无 | 引入 temporal semantics |
| A3 | A0 + 图平滑 | 无 | 无 | 有 | 引入空间结构 |

**Success Criteria**：

- 跨模态检索/匹配准确率提升（同一 window 的 EEG ↔ fNIRS）
- token 使用不塌陷（Phase 1 指标不被破坏）
- 下游线性 probe 有提升或更稳定（少量标签即可）

---

## 5. Phase 3: PID-based Analysis & Refinement（在 token 空间做 PID）

**目标**：把 PID 从“主要训练信号”降级为“分析与结构改造工具”。

### 5.1 PID 的对象与设定（必须写清楚）

- 源变量：$C_{eeg}$（EEG token 序列或统计）、$C_{fnirs}$（fNIRS token 序列或统计）
- 目标变量 $Y$：
  - 优先：任务/状态标签（若有）
  - 备选：Pseudo-Trace 指标（HRF 参数、ERP 幅度）、或自定义伪标签（合成/分段）

### 5.2 分析问题（优先回答）

- 哪些 token 在两模态中呈现**冗余**（高度共享）？
- 是否存在只有 joint 才能解释的 token 组合（**协同**）？
- 加入/去除 physiological nuisance 后，冗余/协同如何变化？

### 5.3 输出与闭环

- 输出：token-level 的 R/U/S 归因报告（按 code 或 code-pair）
- 闭环：据此调整 codebook（合并/拆分/分支），或调整 projector 的对齐权重

---

## 6. Repo / Code Organization（建议的增量结构）

（本阶段先写 roadmap，不强制立刻重构；但建议后续逐步落地）

```text
src/
  tokenizers/         # VQ-VAE / RVQ / FSQ
  alignment/          # projector + alignment objectives
  priors/             # pseudo-trace / physiological / spatial priors
  metrics/            # recon, codebook health, retrieval, PID utilities
```

---

## Next Immediate Steps (本周可做)

1. 写 Phase 0 数据规范（窗口、同步、预处理、增强）并固化到配置
2. 实现/接入一种 tokenizer（建议从 FSQ 开始跑通）并建立 Phase 1 指标看板
3. 加 shared projector + 最小 InfoNCE 对齐（A0）验证不会破坏 codebook health
4. 再引入 physiological / pseudo-trace / spatial 先验做逐项 ablation
