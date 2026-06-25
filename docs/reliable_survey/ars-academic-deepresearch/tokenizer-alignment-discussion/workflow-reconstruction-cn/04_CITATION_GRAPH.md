# 引用图谱：论文→智能体→论点映射

## 论文在两个工作流中的使用方式

每篇论文由特定智能体在特定验证轮次中引用。以下是完整的映射。

---

## 核心论文（同时出现在两个工作流中）

### Huh et al. 2023 — "Straightening Out the Straight-Through Estimator"
- **arXiv**: 2305.08842 | **Venue**: ICML 2023
- **存档 PDF**: `papers/huh2023_straight_through.pdf`
- **引用方**: 第一轮（4 条确认声明中的 3 条）+ 第二轮（1 条确认，2 条被反驳）
- **支持的论点**:
  - [R1+R2 确认] STE 引入梯度间隙 ∝ 量化误差
  - [R1 确认] Commitment loss 是非对称且模式追寻的（Bregman 散度）
  - [R1 确认] 仅被选中的码字获得梯度 → codebook 坍缩
  - [R2 反驳] "STE 梯度间隙是耦合损失失败的根因" —— FSQ 反证削弱了这一声明
- **关键局限**: 分析的是标准 VQ-VAE，使用硬 STE 和梯度下降更新 codebook，而非 EMA 更新的余弦相似度量化和 Gumbel-softmax 软分配

### Liu et al. 2021 — "Cross-Modal Discrete Representation Learning" (CMCM)
- **arXiv**: 2106.05438 | **Venue**: CVPR 2021 / ACL 2022
- **存档 PDF**: `papers/liu2021_cross_modal_discrete.pdf`
- **引用方**: 第一轮（1 条确认）+ 第二轮（2 条确认）
- **支持的论点**:
  - [R2 确认] CMCM 损失在硬量化之前的软编码分配概率上操作 → 绕过了 STE
  - [R2 确认] 跨模态辅助损失可以重塑 VQ token 语义
  - [局限] 需要共享 codebook（本项目使用每个模态独立的 codebook）
  - [局限] 效应量适中（R@1: 46.0 vs 45.2 基线）

### Mentzer et al. 2023 — "Finite Scalar Quantization: VQ-VAE Made Simple" (FSQ)
- **arXiv**: 2309.15505 | **Venue**: ICLR 2024
- **存档 PDF**: `papers/mentzer2023_finite_scalar_quant.pdf`
- **引用方**: 仅第二轮（1 条确认声明）
- **支持的论点**:
  - [R2 确认——决定性证据] FSQ 使用与 VQ 相同的 STE 梯度，却实现了约 100% 的 codebook 利用率，而 VQ 坍缩至 50% 以下
  - 证明了表示几何（有界、低维、固定网格）是上游原因；梯度质量仅是下游中介

### Shekhovtsov 2021 — "Bias-Variance Tradeoffs in Single-Sample Binary Gradient Estimators"
- **arXiv**: 2110.03549 | **Venue**: GCPR 2021
- **存档 PDF**: `papers/shekhovtsov2021_bias_variance.pdf`
- **引用方**: 第一轮（支持性）+ 第二轮（1 条确认，随后被部分反驳）
- **支持的论点**:
  - [R2 确认] 命题 1+2: 对于 Gumbel-Softmax，任何温度设置都无法同时实现低偏差和低方差
  - [R2 反驳] "对于 L 层网络，GS 梯度以 O(τ^L) 数量级消失" —— 分析的是二值 Bernoulli VAE 阶跃函数，而非 VQ 的 argmin-over-K-vectors → 架构不匹配

---

## 第一轮论文（Tokenizer 架构 + 下游方法）

### Zhao et al. 2026 — "Continuous First, Discrete Later: VQ-VAEs Without Dimensional Collapse"
- **arXiv**: 2605.06870
- **存档 PDF**: `papers/zhao2026_continuous_first.pdf`
- **论点**: 训练后的 VQ-VAE 表示坍缩至满秩的 1-2%。AE 预热将有效维度从 3-5 恢复到 17-19。直接解释了 fNIRS 有效秩 = 6-8。

### Wu et al. 2024 — "Learning Granger Causality from Instance-wise Self-attentive Hawkes Processes" (ISAHP)
- **arXiv**: 2402.03726
- **存档 PDF**: `papers/wu2024_granger_hawkes.pdf`
- **论点**: ISAHP 能够从离散事件序列中恢复实例级 Granger 因果关系。首个区分协同事件对与非协同事件对的神经点过程。

### Qiao et al. 2023 — "Structural Hawkes Processes for Learning Causal Structure from Discrete-Time Event Sequences"
- **arXiv**: 2305.05986 | **Venue**: IJCAI 2023
- **存档 PDF**: `papers/qiao2023_structural_hawkes.pdf`
- **论点**: 结构性 Hawkes 过程从离散事件序列中恢复因果图。在事件类型因果发现方面优于基于注意力的方法。

### Baevski et al. 2020 — "wav2vec 2.0"
- **arXiv**: 2006.11477 | **Venue**: NeurIPS 2020
- **存档 PDF**: `papers/baevski2020_wav2vec2.pdf`
- **论点**: 在 Gumbel-softmax 量化潜在变量上的辅助对比损失成功地塑造了离散 token 语义。架构上与本项目设置最接近的前例。

---

## 第二轮论文（对抗验证）

### Fifty et al. 2024 — "Restructuring Vector Quantization with the Rotation Trick"
- **arXiv**: 2410.06424 | **Venue**: ICLR 2025
- **存档 PDF**: 未存档（广泛引用——在各记录中出现 118 次）
- **论点**: STE 忽略了 Voronoi 几何。旋转技巧将利用率从 <2% 提升至 >27%。关键：精确梯度比 STE 表现更差（r-FID 25.4 vs 19.0）。

### Vali et al. 2026 — "DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick"
- **Venue**: ICLR 2026
- **存档 PDF**: 未存档（被引用 56 次）
- **论点**: 来自不同研究组对 STE 梯度病理的独立佐证。消除了所有辅助损失，仅使用重建损失进行端到端训练。

### Lu et al. 2026 — "PCA-VAE: Differentiable Subspace Quantization without Codebook Collapse"
- **arXiv**: 2602.18904
- **存档 PDF**: `papers/lu2026_pca_vae.pdf`
- **论点**: 完全放弃 VQ，转而使用在线 PCA。Codebook 坍缩源于非稳态编码器更新导致的移动目标问题。

### Sadok et al. 2024 — "VQ-MAE-AV: Cross-Modal Discrete Token Alignment"
- **存档 PDF**: 未存档（在日志中被引用）
- **论点**: 最强的存在性证明：独立 codebook + 离散 token 上的联合 MAE + InfoNCE 对齐成功。跨模态损失在已离散化的 token 上操作。

---

## 引用密度图

论文在所有子智能体记录中的引用频率：

```
arXiv:2410.06424  (Fifty/Rotation Trick)         ████████████████████████████████████████ 118
arXiv:2509.26469  (DiVeQ)                        █████████████████ 56
arXiv:2412.19128  (Semantic Residual)             ██████████████ 46
arXiv:2106.05438  (CMCM/Liu)                     █████████████ 45
arXiv:2605.06870  (Continuous First/Zhao)         █████████████ 42
arXiv:2602.18896  (NSVQ/TransVQ)                 █████████████ 42
arXiv:2309.15505  (FSQ/Mentzer)                  ███████ 24
arXiv:2110.03549  (Shekhovtsov bias-variance)    ██████ 22
arXiv:2305.08842  (Huh STE)                      ██████ 21
arXiv:2402.03726  (ISAHP/Wu)                     █████ 16
arXiv:2305.05986  (Structural Hawkes/Qiao)       ████ 14
arXiv:2006.11477  (wav2vec 2.0/Baevski)          ███ 13
```

---

## 论点到论文的依赖关系

### 论点："耦合损失在数学上是不充分的"（R1→R2 部分推翻）
- **主要依据**: Huh et al. 2023 → Shekhovtsov 2021 → Lancucki et al. 2020
- **反证 (R2)**: FSQ (Mentzer 2023) — 相同 STE，无坍缩；CMCM (Liu 2021) — 跨模态损失有效；旋转技巧 (Fifty 2024) — 精确梯度比 STE 更差

### 论点："信息论上限是约束性瓶颈"（R2 提升）
- **主要依据**: Lancucki et al. 2020 (I(X;Z) ≤ log2(K)) → Zhao et al. 2026（维度坍缩）→ arXiv:2604.03191（压缩间隙）
- **支持**: Murugesan 2016（EEG-fNIRS 互信息测量）

### 论点："下游发现是更有前景的路径"（R1+R2 一致）
- **主要依据**: Qiao et al. 2023 (Structural Hawkes) → Wu et al. 2024 (ISAHP) → Shou et al. 2023（影响力感知注意力）
- **支持**: DALL-E (Ramesh 2021) — 独立 tokenizer 训练 + 事后对齐在大规模下有效

### 论点："现有分析存在架构不匹配"（R2 确认）
- **主要依据**: Shekhovtsov 2021（二值 Bernoulli，非 VQ）→ Huh 2023（硬 STE，非软分配）→ Fifty 2024（硬 STE，单模态）
- **关键空白**: 尚无已发表工作对 NormEMAVectorQuantizer 配以余弦相似度、Gumbel-softmax、EMA 更新、独立 codebook 进行测试
