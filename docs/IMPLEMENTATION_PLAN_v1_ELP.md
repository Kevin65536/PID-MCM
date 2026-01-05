# PID-MCM Implementation Plan

> **Last Updated**: 2025-12-22  
> **Status**: Phase 1 - Foundation  
> **Theory Reference**: [`docs/THEORY.md`](docs/THEORY.md)

---

## Overview

This implementation realizes the **Explicit Latent Partitioning (ELP)** framework for PID-guided multimodal pretraining. 

**核心策略**：采用渐进式实验验证，先在合成数据上快速迭代各组件，再迁移到真实 EEG-fNIRS 数据。

### Quick Reference
- **Theory (ELP/PID)**: [`docs/THEORY.md`](docs/THEORY.md)
- **Method Comparison**: [`docs/references/CoMM_FactorCL_analysis.md`](docs/references/CoMM_FactorCL_analysis.md)
- **Data Design**: [`docs/references/synthetic_dataset_design.md`](docs/references/synthetic_dataset_design.md)

---

## 可选组件清单（Baseline → Improvement）

以下列出各模块的基线方案与可选改进，预实验将逐一验证其效果。

### A. 对齐损失（Alignment Loss for $z_r$）

| 组件 | 描述 | 复杂度 | 理论依据 |
|:-----|:-----|:-------|:---------|
| **A1: MSE Align (Baseline)** | $\mathcal{L} = \|z_r^A - z_r^B\|^2$ | 低 | 直接约束两视角冗余 token 相等 |
| A2: InfoNCE Align | 对比学习式对齐，batch 内负样本 | 中 | 防止坍塌，有信息论下界解释 |
| A3: MSE + Variance Reg | MSE + VICReg 方差正则 | 中 | 简单防坍塌 |

### B. 正交损失（Orthogonality Loss）

| 组件 | 描述 | 复杂度 | 理论依据 |
|:-----|:-----|:-------|:---------|
| **B1: Pairwise Cosine (Baseline)** | $\sum_{i<j} |\cos(z_i, z_j)|$ | 低 | 6 对 token 余弦惩罚 |
| B2: Squared Cosine | $\sum_{i<j} \cos^2(z_i, z_j)$ | 低 | 对小相关性更宽容 |
| B3: Covariance Reg (VICReg-style) | 协方差矩阵非对角元惩罚 | 中 | 去相关，更符合高斯假设 |

### C. 协同损失（Synergy Loss for $z_s$）

| 组件 | 描述 | 复杂度 | 理论依据 |
|:-----|:-----|:-------|:---------|
| **C1: Masking Diff (Baseline)** | $-\|z_s^{joint} - z_s^{masked}\|^2$ | 低 | $z_s$ 应随模态缺失而变化（较弱） |
| C2: Prediction Improvement | 加 $z_s$ 后预测性能提升 | 高 | 需要标签，不适合预训练 |
| **C4: Residual Reconstruction** | $z_s$ 重建 $R+U$ 解释不了的残差 | 中 | **推荐**：显式捕捉剩余信息 |
| **C5: Unpredictability** | $z_s$ 不可从单模态预测 | 中-高 | 对抗式，确保协同语义 |

### D. Stop-Gradient 策略

| 组件 | 描述 | 复杂度 | 理论依据 |
|:-----|:-----|:-------|:---------|
| **D1: No Stop-Grad (Baseline)** | 所有 token 均接受所有损失梯度 | 低 | 简单 |
| D2: Phase-Specific SG | Unique 学习时 stop_grad($z_r$) | 中 | 保护已学习的冗余表示 |
| D3: Full Routing SG | 每个 phase 只更新目标 token | 高 | 最严格的梯度路由 |

### E. 重建目标

| 组件 | 描述 | 复杂度 | 理论依据 |
|:-----|:-----|:-------|:---------|
| **E1: Full Reconstruction** | Dec($z_r + z_u + z_s$) 重建全信号 | 低 | 信息完整性 |
| E2: Token-Specific Recon | $z_r$ 重建低频，$z_u$ 重建高频残差 | 中 | 领域先验 |
| E3: Cross-Modal Prediction | 用 $z_r$ 跨模态预测 | 中 | 验证冗余语义 |

---

## Phase 1: 时序合成数据集与快速预实验 (Week 1-2)

### 1.0 基础 Sanity-Check：XOR = 纯协同（必须通过）

**动机**：在经典 PID 场景中，若 $Y = X_1 \oplus X_2$ 且 $X_1, X_2$ 独立同分布，则
\[
I(X_1;Y)=0,\quad I(X_2;Y)=0,\quad I(X_1,X_2;Y)=1\text{ bit}
\]
因此关于目标 $Y$ 的信息应全部落在 **Synergy**。

**实验目标**：验证我们的 token 分解在“显式目标 $Y$”下能够把 XOR 信息主要交给 $z_s$，并且单模态/单 token 不能预测 $Y$。

**实现方式（最小可复现）**：
- 构造二值时间序列 $X_1, X_2 \in \{0,1\}^{T}$，令 $Y = X_1 \oplus X_2$（逐时刻 XOR）。
- 训练一个最小 ELP-style 模型，使 $\hat{Y}$ 仅由 $z_s$ 解码（$z_r, z_{u1}, z_{u2}$ 作为旁路 token 用于正交/对齐约束）。

**评估指标**（必须同时报告）：
- `acc_joint(z_s→Y)`: 用 $z_s$ 预测 $Y$ 的逐点准确率，应接近 1.0
- `acc_x1_only(X1→Y)`: 仅用 $X_1$ 预测 $Y$，应接近 0.5（chance）
- `acc_x2_only(X2→Y)`: 仅用 $X_2$ 预测 $Y$，应接近 0.5（chance）
- `acc_token_ablation`: 冻结 encoder，仅用单个 token（$z_r$ / $z_{u1}$ / $z_{u2}$ / $z_s$）训练线性 probe 预测 $Y$；除 $z_s$ 外都应接近 0.5

**Success Criteria**：
- `acc_joint(z_s→Y) ≥ 0.95`
- `acc_x1_only(X1→Y) ≤ 0.55` 且 `acc_x2_only(X2→Y) ≤ 0.55`
- `acc_probe(z_s) - max(acc_probe(z_r), acc_probe(z_{u1}), acc_probe(z_{u2})) ≥ 0.35`

### 1.1 重新设计：具有明确 PID 语义的时序数据

**设计目标**：生成时间序列数据，其中各 PID 成分有**可验证的物理/频率特征**。

#### 数据生成方案 (`src/data/synthetic.py` 重写)

```
模态 X1 (模拟 EEG):
├── R 成分: 低频正弦波 (0.1-1 Hz) ← 与 X2 共享
├── U1 成分: 高频振荡 (8-30 Hz) ← X1 独有
└── S 成分: R × U1 的调制交互项 ← 协同

模态 X2 (模拟 fNIRS):
├── R 成分: 同样的低频正弦波 (0.1-1 Hz) ← 与 X1 共享
├── U2 成分: 超低频漂移 (< 0.1 Hz) ← X2 独有
└── S 成分: R × U2 的调制交互项 ← 协同
```

**数学定义**：
```python
# 时间轴
t = np.linspace(0, T, num_samples)

# Redundancy: 两模态共享的低频成分
w_r = sin(2π * f_low * t + φ_r)  # f_low ∈ [0.1, 1] Hz

# Unique for X1 (EEG): 高频振荡
w_u1 = sin(2π * f_high * t + φ_u1)  # f_high ∈ [8, 30] Hz

# Unique for X2 (fNIRS): 超低频漂移
w_u2 = trend(t) + slow_oscillation  # < 0.1 Hz

# Synergy: 交互项（乘法调制 = 只有同时观察才能分离）
w_s = w_r * w_u1  # 振幅调制，产生边带频率

# 观测信号
X1 = α1*w_r + β1*w_u1 + γ1*w_s + noise
X2 = α2*w_r + β2*w_u2 + γ2*w_s + noise
```

**验证指标**：
- $z_r$ 的功率谱应集中在 0.1-1 Hz
- $z_{u\_eeg}$ 的功率谱应集中在 8-30 Hz
- $z_s$ 应能预测交互项（如调制深度）

#### 任务标签（用于协同验证）

| 标签类型 | 定义 | 可从哪些成分预测 |
|:---------|:-----|:-----------------|
| y_redundancy | sign(mean(w_r)) | $z_r$ alone |
| y_unique_eeg | sign(power(w_u1, 8-30Hz)) | $z_r + z_{u\_eeg}$ |
| y_synergy | modulation_depth(w_s) | 需要 $z_s$ |

### 1.2 快速预实验设计

**目标**：用最少代码验证各组件的实际效果，每个实验 < 30 分钟。

#### 实验矩阵

| Exp ID | Alignment | Orthogonality | Synergy | Stop-Grad | 验证目标 |
|:-------|:----------|:--------------|:--------|:----------|:---------|
| E0 | A1 (MSE) | B1 (Cosine) | C1 (Diff) | D1 (None) | **Baseline** |
| E1 | A2 (NCE) | B1 | C1 | D1 | 对齐方式对比 |
| E2 | A3 (MSE+Var) | B1 | C1 | D1 | 简单防坍塌 |
| E3 | A1 | B2 (Sq Cos) | C1 | D1 | 正交约束强度 |
| E4 | A1 | B3 (Cov) | C1 | D1 | 协方差正则 |
| E5 | A1 | B1 | C1 | D2 (Phase SG) | 梯度路由效果 |
| **E6** | A1 | B1 | **C4 (Residual)** | D1 | **协同残差重建** |
| **E7** | A1 | B1 | **C5 (Unpred)** | D1 | **协同不可预测性** |
| E8 | A1 | B1 | C4+C5 | D2 | 组合策略 |
| **E9** | (N/A) | B1 | **XOR Supervised Sanity** | (N/A) | **XOR 纯协同验证** |

#### 评估指标

**Phase 1 核心指标**（可在合成数据上精确计算）：

1. **Latent Recovery** (主要)
   - `corr(z_r, w_r)`: 冗余恢复相关性
   - `corr(z_u_eeg, w_u1)`: EEG 独有恢复
   - `corr(z_s, w_s)`: 协同恢复

2. **Disjointness** (次要)
   - `HSIC(z_r, z_u)`: token 间独立性
   - `mean_abs_cosine`: 平均余弦相似度

3. **Collapse Detection**
   - `std(z_r)`: 冗余 token 的方差（防坍塌）
   - `rank(Z)`: 表示矩阵的有效秩

4. **Frequency Validation** (领域特定)
   - `dominant_freq(Dec(z_r))`: 重建的主频率
   - `high_freq_power(Dec(z_u_eeg))`: 高频能量占比

#### 实验脚本结构

```
experiments/
├── configs/
│   ├── baseline.yaml      # E0 配置
│   ├── exp_align_nce.yaml # E1 配置
│   └── ...
├── run_synthetic_exp.py   # 统一入口
└── analyze_results.py     # 指标汇总
```

### 1.3 Loss Functions Module (`src/losses/pid_losses.py`)

**保留现有实现作为 Baseline**，新增可选组件：

**Tasks**:
- [x] `AlignmentLoss` (A1 - MSE, 已实现)
- [x] `OrthogonalityLoss` (B1 - Cosine, 已实现)
- [x] `SynergyLoss` (C1 - Masking Diff, 已实现)
- [x] `ReconstructionLoss` (已实现)
- [ ] `AlignmentLossNCE` (A2 - InfoNCE)
- [ ] `VarianceLoss` (用于 A3)
- [ ] `OrthogonalityLossSquared` (B2)
- [ ] `CovarianceLoss` (B3 - VICReg style)
- [ ] `SynergyLossPredictive` (C2 - 预测式)

**Success Criteria**:
- Baseline (E0) 在合成数据上 `corr(z_r, w_r) > 0.6`
- 至少一个改进组件显著提升 latent recovery

### 1.4 Masking Strategy (`src/data/masking.py`)

**Tasks**:
- [x] Implement `CrossModalMask`: Mask 80% of X1, keep X2 full (已实现)
- [x] Implement `UniModalMask`: Mask 50% of X1, drop X2 entirely (已实现)
- [x] Implement `JointMask`: Random 50% masking on both modalities (已实现)
- [ ] Create `MixedBatchSampler`: 25% cross, 25% uni, 50% joint

**Success Criteria**:
- Batch composition matches intended proportions
- Masking is random and reproducible with seed

### 1.5 时序合成数据集重写 (`src/data/synthetic.py`)

**Tasks**:
- [ ] 重写 `PIDSyntheticDataset` 为时序版本 `PIDTimeSeriesDataset`
- [ ] 实现频率分离的 PID 成分生成
- [ ] 添加任务标签 (y_redundancy, y_unique, y_synergy)
- [ ] 实现频谱验证工具函数

**Success Criteria**:
- 各成分在频域上可分离
- ground truth latents 与生成信号的 PID 语义一致

### 1.6 预实验训练脚本 (`experiments/run_synthetic_exp.py`)

**Tasks**:
- [ ] 实现配置文件驱动的实验入口
- [ ] 支持组件选择 (A1/A2/A3, B1/B2/B3, C1/C2, D1/D2)
- [ ] 自动记录所有评估指标
- [ ] 生成实验对比报告

**Baseline 总损失**:
$$\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{align} + \lambda_2 \mathcal{L}_{orth} + \lambda_3 \mathcal{L}_{syn}$$

初始超参: $\lambda_1=0.5, \lambda_2=0.3, \lambda_3=0.2$

**Success Criteria**:
- 单次实验 < 30 分钟 (小规模数据)
- 可复现，支持随机种子固定

### 1.7 评估与分析 (`notebooks/phase1_analysis.ipynb`)

**Tasks**:
- [ ] 提取各实验的 learned tokens
- [ ] 计算 latent recovery 指标
- [ ] 频谱分析验证
- [ ] 生成组件对比表格

**Success Criteria**:
- Baseline `corr(z_r, w_r) > 0.5` (保守目标)
- 找到至少一个有效改进组件

---

## Phase 2: 从合成到真实数据的迁移 (Week 3-4)

### 迁移策略

**核心原则**：在合成数据上验证有效的组件组合，直接迁移到真实数据，最小化重新调参。

### 2.1 迁移前检查清单

| 检查项 | 合成数据要求 | 真实数据预期 |
|:-------|:-------------|:-------------|
| 冗余恢复 | `corr(z_r, w_r) > 0.6` | $z_r$ 跨模态对齐 |
| 互斥性 | `HSIC < 0.15` | tokens 低相关 |
| 无坍塌 | `std(z_r) > 0.5` | 表示多样性 |
| 频率分离 | 频谱验证通过 | 符合领域先验 |

### 2.2 Dataset Selection
**Candidates**:
1. **OpenBMI** - Motor imagery tasks
2. **Shin et al. (2018)** - Mental arithmetic (N-back)
3. **BIP Datasets** - Standard BCI benchmarks

**Tasks**:
- [ ] Download datasets and verify integrity
- [ ] Inspect data formats (`.mat`, `.fif`, `.npy`)
- [ ] Check alignment between EEG and fNIRS timestamps

### 2.3 Preprocessing Pipeline (`src/data/real_data.py`)

**EEG Preprocessing**:
- [ ] Bandpass filter: 1-50 Hz
- [ ] Downsample to 200 Hz
- [ ] Remove artifacts (ICA or simple thresholding)
- [ ] Epoch into 2-second windows

**fNIRS Preprocessing**:
- [ ] Convert raw intensity to HbO/HbR
- [ ] Bandpass filter: 0.01-0.2 Hz
- [ ] Resample to align with EEG epochs
- [ ] Handle motion artifacts

**Tasks**:
- [ ] Implement `EEGPreprocessor` class
- [ ] Implement `fNIRSPreprocessor` class
- [ ] Create `MultimodalDataset` that loads aligned pairs
- [ ] Split into train/val/test (70/15/15)

**Success Criteria**:
- No data leakage across splits
- Aligned time windows (visual inspection)
- Reasonable SNR (>5 dB for EEG)

### 2.4 迁移验证实验

**目标**：验证合成数据上的最佳配置在真实数据上仍然有效。

| 实验 | 配置 | 验证目标 |
|:-----|:-----|:---------|
| T1 | Phase 1 最佳配置 | 直接迁移效果 |
| T2 | T1 + 领域特定调整 | 频率范围适配 |
| T3 | T2 + 超参微调 | 最终配置 |

---

## Phase 3: 模型训练与调优 (Week 5-6)

### Goal
在真实数据上训练 ELP，使用 Phase 1 验证过的组件配置。

### Deliverables

#### 3.1 Model Enhancements
**Tasks**:
- [ ] Add positional encoding to handle temporal sequences
- [ ] Implement MAE-style masking (learnable `[MASK]` token)
- [ ] Add reconstruction decoder (separate heads for EEG/fNIRS)
- [ ] (可选) Implement stop-gradient for $z_r$ during unique learning

#### 3.2 Hyperparameter Tuning

**基于 Phase 1 结果的调参策略**:
- 如果 Phase 1 中某组件显著优于 baseline，优先使用
- 如果 Phase 1 中某组件效果相当，选择更简单的

**Grid Search**:
- Hidden dim: `[128, 256, 512]`
- Num layers: `[2, 4, 6]`
- Loss weights: 从 Phase 1 最佳配置出发微调

**Tasks**:
- [ ] Create config files (YAML) for each setting
- [ ] Run grid search with early stopping
- [ ] Select best model based on validation reconstruction loss

#### 3.3 Training Script (`experiments/train_real.py`)
**Tasks**:
- [ ] Load `MultimodalDataset`
- [ ] Train for 200 epochs with cosine annealing LR schedule
- [ ] Save checkpoints every 10 epochs
- [ ] Log metrics: loss breakdown, HSIC between tokens

**Success Criteria**:
- Reconstruction MSE < 0.5 (normalized data)
- $\text{HSIC}(z_r, z_u) < 0.15$
- Model generalizes to validation set

---

## Phase 4: Evaluation & Baselines (Week 7-8)

### Goal
Demonstrate ELP superiority over baselines.

### Deliverables

#### 4.1 Baselines
**Implement**:
1. **Vanilla MAE**: Single `[CLS]` token, no latent partitioning
2. **Contrastive**: CLIP-style alignment (captures only $R$)
3. **CCA**: Linear baseline for comparison

**Tasks**:
- [ ] Train each baseline with same data/budget
- [ ] Extract representations
- [ ] Freeze and use for downstream tasks

#### 4.2 Downstream Tasks
**Tasks**:
1. **Mental Arithmetic Classification**: Predict N-back level (0/2/3)
2. **Motor Imagery**: Predict left/right hand movement

**Evaluation**:
- [ ] Freeze ELP encoder and train linear classifier on $[z_r, z_u, z_s]$
- [ ] Compare with baselines on accuracy/F1

#### 4.3 PID Analysis
**Tasks**:
- [ ] Estimate PID using PIDF on extracted representations
- [ ] Compare with PID on raw signals
- [ ] Verify $z_r$ captures high redundancy, $z_s$ captures synergy

**Metrics**:
- Redundancy: $I(z_r^{EEG}; z_r^{fNIRS})$
- Unique: $I(z_{u\_eeg}; X_{eeg} | z_r) - I(z_{u\_eeg}; X_{fnirs} | z_r)$
- Synergy: Improvement in task performance when adding $z_s$

#### 4.4 Ablation Studies
**Tasks**:
- [ ] Train without $\mathcal{L}_{orth}$: Does it hurt disjointness?
- [ ] Train without $\mathcal{L}_{syn}$: Does $z_s$ lose meaning?
- [ ] Use only Cross-Modal masking: Can it learn all components?

---

## Implementation Checklist (Revised)

### Week 1: 合成数据与基线实验
- [x] Set up directory structure
- [x] Implement `PIDSyntheticDataset` (原版)
- [x] Implement `ELPEncoder` skeleton
- [x] Implement baseline loss functions (A1, B1, C1)
- [ ] **重写时序版 `PIDTimeSeriesDataset`**
- [ ] 运行 Baseline 实验 (E0)

### Week 2: 组件消融与最佳配置选择
- [ ] 实现可选组件 (A2, A3, B2, B3, C2)
- [ ] 运行实验矩阵 (E1-E6)
- [ ] 分析结果，确定最佳组件组合
- [ ] 编写 Phase 1 总结报告

### Week 3-4: 数据迁移
- [ ] 下载并预处理真实数据集
- [ ] 创建 `MultimodalDataset`
- [ ] 运行迁移验证实验 (T1-T3)
- [ ] 确认最终配置

### Week 5-6: 真实数据训练
- [ ] 增强模型架构 (positional encoding, decoder)
- [ ] 运行超参数搜索
- [ ] 训练最终模型

### Week 7-8: 评估与消融
- [ ] 实现并训练 baselines
- [ ] 运行下游任务评估
- [ ] 完成 PID 分析和消融实验

---

## Debugging & Validation Strategy

### Common Issues
1. **Gradient Vanishing**: Check loss scales, use gradient clipping
2. **Token Collapse**: All tokens become identical
   - **Fix**: Increase $\lambda_{orth}$, use stop-gradient
3. **Synergy Token Ignored**: Model doesn't use $z_s$
   - **Fix**: Increase $\lambda_{syn}$, add synergy-specific tasks

### Validation Checkpoints
- After every 10 epochs: Visualize token distances (should increase)
- After training: Check if $z_r$ is same for different views of same sample
- Before real data: Ensure synthetic experiment succeeds

---

## Success Metrics Summary

| Phase | Metric | Target | 备注 |
|:------|:-------|:-------|:-----|
| Phase 1 (Baseline) | `corr(z_r, w_r)` | > 0.5 | 保守目标 |
| Phase 1 (Best) | `corr(z_r, w_r)` | > 0.7 | 期望目标 |
| Phase 1 | `HSIC(z_r, z_u)` | < 0.15 | 互斥性 |
| Phase 1 | `std(z_r)` | > 0.5 | 防坍塌 |
| Phase 2 | 迁移成功率 | T1 效果 ≥ 80% of Phase 1 | 配置稳定性 |
| Phase 3 | Reconstruction MSE | < 0.5 | 真实数据 |
| Phase 4 | Classification Accuracy | > Baselines | 下游任务 |

---

## Next Immediate Steps

1. **重写 `src/data/synthetic.py`** → 时序版本，频率分离的 PID 成分
2. **创建实验配置文件** `experiments/configs/baseline.yaml`
3. **实现 `experiments/run_synthetic_exp.py`** → 统一实验入口
4. **运行 Baseline 实验 (E0)** → 验证框架可行性
5. **根据 E0 结果决定是否需要改进组件**
