# Neuro-Tokenization Implementation Plan

> **Last Updated**: 2026-01-13  
> **Status**: Phase 0 - Real Data Tokenization  
> **Theory Reference**: [`docs/THEORY.md`](docs/THEORY.md)

---

## 0. Overview

本项目的核心目标是：将 EEG 与 fNIRS 信号离散化为可复用的 token（codebook），为后续的跨模态分析、下游任务分类、可解释性研究奠定基础。

### 0.1 核心假设

1. EEG 与 fNIRS 信号中存在**可重复出现的模式**，可以被离散 codebook 捕获
2. 良好的 codebook 应具备：
   - **重构能力**：token 序列可重建原始信号
   - **覆盖度**：codebook 中的 code 被均匀使用，无 collapse
   - **泛化性**：在不同被试/session 间保持稳定

### 0.2 当前阶段目标

**Phase 0: Real Data Tokenization**
- 在实际 EEG/fNIRS 数据上验证 tokenizer 的可行性
- 建立 codebook 健康度的评估基准
- 确定后续实验的数据预处理规范

### 0.3 远期目标（存档）

以下目标将在 tokenization 稳定后逐步推进：
- 跨模态对齐（Semantic Alignment）
- 下游任务分类器
- Codebook 可解释性分析
- PID 信息分解（见 [docs/THEORY_v1_ELP.md](docs/THEORY_v1_ELP.md)）

---

## 1. Phase 0: Real Data Tokenization

### 1.1 目标

在真实 EEG/fNIRS 数据上验证：
1. Tokenizer（FSQ / VQ-VAE）能否有效重构信号
2. Codebook 是否会 collapse
3. 不同数据集/被试间的泛化能力

### 1.2 数据准备

#### 1.2.1 数据集选择

| 数据集 | 模态 | 任务类型 | 备注 |
|--------|------|----------|------|
| TBD | EEG | Motor/Cognitive | 待确定公开数据集 |
| TBD | fNIRS | Motor/Cognitive | 待确定公开数据集 |
| TBD | EEG+fNIRS | 同步采集 | 理想情况，待寻找 |

#### 1.2.2 数据预处理规范

需要在实验前确定以下规范：

**EEG 预处理**
- 采样率：重采样到统一频率（如 200Hz）
- 滤波：带通滤波范围（如 0.5-45Hz）
- 参考方式：平均参考 / 特定电极
- 伪迹处理：ICA / 阈值剔除
- 通道选择：全通道 / ROI 子集

**fNIRS 预处理**
- 信号类型：HbO / HbR / 两者
- 滤波：低通滤波（如 0.1Hz）
- 运动伪迹：样条插值 / 剔除
- 通道处理：单通道 / 多通道

**窗口化**
- 窗口长度：$W$（建议 2-8s，按任务设计）
- 步长：$H$（可选重叠）
- 对齐策略：时间戳同步

### 1.3 实验设计

#### P0: 单模态 Tokenizer 验证

**目标**：分别在 EEG 和 fNIRS 上验证 tokenizer 基本功能

| 实验 ID | 数据 | Tokenizer | 目标 |
|---------|------|-----------|------|
| P0-EEG-FSQ | EEG | FSQ | 验证 FSQ 在 EEG 上的重构与 codebook 健康度 |
| P0-EEG-VQ | EEG | VQ-VAE | 对比 VQ-VAE，选择更优方案 |
| P0-fNIRS-FSQ | fNIRS | FSQ | 验证 FSQ 在 fNIRS 上的表现 |
| P0-fNIRS-VQ | fNIRS | VQ-VAE | 对比 VQ-VAE |

**评估指标**

| 指标 | 说明 | 期望 |
|------|------|------|
| Reconstruction MSE | 时域重构误差 | 越低越好 |
| Spectral MSE | 频域重构误差（对 EEG 重要） | 越低越好 |
| Perplexity | Codebook 使用丰富度 | > 50% of codebook size |
| Code Utilization | 被使用的 code 比例 | > 20% |
| Dead Codes | 从未使用的 code 数量 | < 30% |

#### P1: 跨被试泛化验证

**目标**：验证 codebook 在不同被试间的稳定性

| 实验 ID | 设置 | 目标 |
|---------|------|------|
| P1-Cross-Subject | Train: Subject 1-N, Test: Subject N+1 | 泛化误差 vs 训练误差 |
| P1-Session | Train: Session 1, Test: Session 2 | 同被试跨 session 稳定性 |

### 1.4 Success Criteria

| 阶段 | 指标 | 阈值 |
|------|------|------|
| P0 | Reconstruction MSE | 相对 baseline（无量化）增加 < 50% |
| P0 | Perplexity | > 30% of codebook size |
| P0 | Training Stability | Loss 单调下降，无震荡 |
| P1 | Generalization Gap | Test MSE / Train MSE < 2.0 |

---

## 2. 代码结构

### 2.1 当前结构

```text
src/
  tokenizers/           # Tokenizer 实现
    __init__.py
    base.py            # 基类：BaseTokenizer, Conv1dEncoder/Decoder
    fsq.py             # Finite Scalar Quantization
    vqvae.py           # VQ-VAE
  data/                 # 数据加载
    __init__.py
    (待新增) eeg_dataset.py      # EEG 数据加载
    (待新增) fnirs_dataset.py    # fNIRS 数据加载
  metrics/              # 评估指标（待新增）
    __init__.py
    codebook_health.py  # perplexity, usage, dead codes
    reconstruction.py   # MSE, spectral loss
  models/               # 旧 ELP 模型（存档）
  losses/               # 旧 PID 损失（存档）
  utils/                # 工具函数
```

### 2.2 实验结构

```text
experiments/
  configs/
    base.yaml                  # 基础配置
    phase0/                    # Phase 0 实验配置
      P0_eeg_fsq.yaml
      P0_eeg_vqvae.yaml
      P0_fnirs_fsq.yaml
      P0_fnirs_vqvae.yaml
      P1_cross_subject.yaml
  scripts/
    train_tokenizer.py         # 通用训练脚本
    evaluate_tokenizer.py      # 评估脚本
    visualize_codebook.py      # Codebook 可视化
  runs/                        # 实验运行记录
    {exp_name}_{timestamp}/
      config.yaml              # 实验配置快照
      metrics.json             # 训练指标
      checkpoints/             # 模型检查点
      figures/                 # 可视化图表
  results/                     # 汇总结果
    comparison.csv             # 实验对比表
    figures/                   # 汇总图表
```

### 2.3 实验命名规范

```
{Phase}_{Modality}_{Tokenizer}_{Variant}_{Timestamp}
```

示例：
- `P0_EEG_FSQ_baseline_20260113_140000`
- `P0_fNIRS_VQVAE_ema_20260113_150000`
- `P1_EEG_FSQ_cross_subject_20260115_100000`

---

## 3. 配置设计

### 3.1 Base Config

```yaml
# experiments/configs/base.yaml

experiment:
  name: "base"
  seed: 42
  device: "cuda"

data:
  modality: "eeg"  # eeg | fnirs
  dataset: "TBD"
  preprocessing:
    resample_rate: 200
    bandpass: [0.5, 45]
  window:
    length: 512    # samples
    stride: 256    # samples

model:
  type: "fsq"      # fsq | vqvae
  encoder:
    hidden_dims: [64, 128, 256]
    kernel_size: 7
    stride: 2
  quantizer:
    # FSQ specific
    levels: [8, 8, 8, 8]
    # VQ-VAE specific
    # codebook_size: 512
    # embedding_dim: 64
    # commitment_cost: 0.25
  decoder:
    hidden_dims: [256, 128, 64]

loss:
  reconstruction:
    weight: 1.0
    type: "mse"
  spectral:
    weight: 0.1
    type: "multi_stft"
    fft_sizes: [64, 128, 256]

training:
  epochs: 100
  batch_size: 64
  learning_rate: 1e-3
  weight_decay: 1e-4
  scheduler: "cosine"
  warmup_epochs: 5

logging:
  log_every_n_steps: 100
  save_checkpoint_every: 10
  metrics:
    - reconstruction_mse
    - spectral_mse
    - perplexity
    - code_utilization
    - dead_codes
```

### 3.2 Phase 0 实验配置

每个 Phase 0 实验配置继承 base.yaml 并覆盖特定参数：

```yaml
# experiments/configs/phase0/P0_eeg_fsq.yaml

experiment:
  name: "P0_EEG_FSQ"
  description: "FSQ tokenizer on EEG data"

data:
  modality: "eeg"
  dataset: "TBD"

model:
  type: "fsq"
  quantizer:
    levels: [8, 8, 8, 8]  # 4096 codes
```

---

## 4. 下一步行动

### Week 1（当前）

1. [ ] **数据集调研**：寻找合适的公开 EEG/fNIRS 数据集
2. [ ] **数据加载模块**：实现 `src/data/eeg_dataset.py`
3. [ ] **评估指标模块**：实现 `src/metrics/codebook_health.py`
4. [ ] **训练脚本**：实现 `experiments/scripts/train_tokenizer.py`

### Week 2

1. [ ] 运行 P0-EEG-FSQ 实验
2. [ ] 运行 P0-EEG-VQ 实验
3. [ ] 对比分析，选择更优方案

### Week 3

1. [ ] 在 fNIRS 数据上重复 P0 实验
2. [ ] 运行 P1 跨被试泛化实验
3. [ ] 总结 Phase 0 结论，规划 Phase 1

---

## 5. 待讨论事项

以下事项将在实验推进过程中逐步确定：

1. **数据集选择**：具体使用哪个公开数据集？
2. **预处理细节**：滤波参数、伪迹处理方法？
3. **模型超参数**：codebook 大小、encoder 深度？
4. **下游任务**：使用什么任务验证 token 质量？
5. **跨模态对齐**：何时开始、使用什么对齐目标？

---

## Appendix A: 旧实验归档

以下实验属于 ELP-first 方案（直接学习 PID 分解），已归档但保留代码供参考：

| 实验 | 配置 | 状态 | 结论 |
|------|------|------|------|
| E0_baseline | `configs/E0_baseline.yaml` | 已完成 | 失败：PID 成分恢复接近随机 |
| E6_synergy_residual | `configs/E6_synergy_residual.yaml` | 未运行 | - |
| E7_synergy_unpred | `configs/E7_synergy_unpred.yaml` | 未运行 | - |

**ELP-first 失败分析**：
- 连续空间的 PID 估计对表示分布敏感
- 正交性约束无法保证信息分离
- 需要先建立稳定的离散表示

相关代码：
- `src/models/elp_encoder.py` - ELP 编码器
- `src/losses/pid_losses.py` - PID 损失函数

PID 分析将作为**远期目标**，在 tokenization 和 alignment 稳定后重新引入。详见 [docs/THEORY_v1_ELP.md](docs/THEORY_v1_ELP.md)。

---

## Appendix B: Tokenizer 对比

| 特性 | FSQ | VQ-VAE |
|------|-----|--------|
| Codebook | 隐式（level 组合） | 显式（embedding table） |
| Collapse 风险 | 低 | 高（需 EMA/reset） |
| 梯度 | 直通 | Straight-through |
| 表达能力 | 受 level 限制 | 灵活 |
| 推荐场景 | 起步验证 | 正式实验 |

## Appendix C: Codebook 健康度指标

```python
def compute_codebook_health(indices: torch.Tensor, codebook_size: int) -> dict:
    """
    计算 codebook 健康度指标
    
    Args:
        indices: Token indices [B, T] or [N]
        codebook_size: Total number of codes
        
    Returns:
        dict with perplexity, utilization, dead_codes
    """
    flat = indices.flatten()
    usage = torch.bincount(flat, minlength=codebook_size).float()
    usage_prob = usage / usage.sum()
    
    # Perplexity
    entropy = -(usage_prob * torch.log(usage_prob + 1e-10)).sum()
    perplexity = torch.exp(entropy)
    
    # Utilization
    active_codes = (usage > 0).sum()
    utilization = active_codes / codebook_size
    
    # Dead codes
    dead_codes = (usage == 0).sum()
    
    return {
        'perplexity': perplexity.item(),
        'utilization': utilization.item(),
        'dead_codes': dead_codes.item(),
        'active_codes': active_codes.item(),
    }
```
