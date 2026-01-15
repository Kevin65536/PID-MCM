# Neuro-Tokenization Implementation Plan

> **Last Updated**: 2026-01-15  
> **Status**: Phase 0+ - Multi-channel Tokenizer & Classification Pipeline  
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
3. 两种模态的 token 序列之间可能存在**时序耦合或因果关系**，值得探索

### 0.2 两条推进路线

根据 2026-01-15 的讨论，确定两条并行路线：

| 路线 | 目标 | 输出 |
|------|------|------|
| **路线 A: 工程验证** | 快速完成全数据流可行性验证 | 端到端 pipeline + baseline 性能 |
| **路线 B: 理论分析** | 观察跨模态 token 关系，指导对齐设计 | 跨模态 token 关系的 empirical 发现 |

### 0.3 阶段规划

```
Phase 0+: Tokenizer 完善 (Week 1-2)
    ├── P0.1: 多通道输入设计
    ├── P0.2: 序列长度优化
    └── P0.3: 跨被试泛化验证

Phase 1A: 全数据流验证 (Week 2-3) ──┐
    ├── P1A.1: EEG tokens → MI 分类   │  并行
    ├── P1A.2: fNIRS tokens → MI 分类  │
    └── P1A.3: 简单多模态融合          │
                                      │
Phase 1B: 跨模态 Token 分析 (Week 2-3)─┘
    ├── P1B.1: Token 序列统计分析
    ├── P1B.2: 时序耦合分析
    └── P1B.3: 因果关系探索

Phase 2: 对齐策略设计 (Week 4+)
    └── 基于 Phase 1B 发现设计对齐方法
```

---

## 1. Phase 0+: Tokenizer 完善

### 1.1 目标

在 Phase 0 基础实验之上，完善 tokenizer 以支持：
1. **多通道输入**：利用空间信息提升重构质量
2. **合理序列长度**：根据模态特性和任务需求优化
3. **跨被试泛化**：验证 codebook 的稳定性

### 1.2 数据准备（已完成）

#### 1.2.1 数据集选择

**选定数据集**: `EEG+NIRS Single-Trial` (TU Berlin Open Access Dataset)

**数据集对比分析** (2026-01-14):

| 数据集 | 被试 | 任务类型 | EEG采样率 | fNIRS采样率 | 格式 | 同步方式 |
|--------|------|----------|-----------|-------------|------|----------|
| Visual Cognitive Motivation | 16 | 视觉记忆动机 | 原始EDF | CSV | .edf+.csv+.mat | DC9触发器 |
| **EEG+NIRS Single-Trial** ✅ | 29 | MI + MA | 200Hz | 10Hz | MATLAB .mat | Parallel port |
| REFED | 32 | 情绪识别 | 1000Hz | 47.62Hz | MATLAB .mat | 时间对齐 |
| Simultaneous EEG&NIRS (Cognitive) | 26 | N-back/DSR/WG | 200Hz | 10Hz | MATLAB .mat | Parallel port |

**选择 EEG+NIRS Single-Trial 的理由**:

1. **任务经典**: Motor Imagery (左/右手想象) 是 BCI 领域标准任务，信号模式清晰可辨
2. **格式统一**: MATLAB .mat 格式，已下采样 (EEG 200Hz, NIRS 10Hz)，无需复杂预处理
3. **同步完善**: 通过 parallel port 同时发送触发器到 EEG 和 NIRS 设备，时间同步精确
4. **文档完整**: 使用 BBCI Toolbox 数据结构，有详细说明
5. **被试充足**: 29 名被试，可进行跨被试泛化实验

**数据集详细信息**:

| 属性 | 值 |
|------|------|
| 来源 | TU Berlin (doc.ml.tu-berlin.de/hBCI) |
| EEG 电极 | 30 通道 (10-5 系统) + EOG/ECG |
| fNIRS 通道 | 36 通道 (14 sources + 16 detectors) |
| 任务 A | Motor Imagery: 左手 vs 右手想象 (marker 16/32) |
| 任务 B | Mental Arithmetic: 心算 vs 静息 (marker 16/32) |
| 试次结构 | 2s指示 + 10s任务 + 15-17s休息 |
| 数据路径 | `data/EEG+NIRS Single-Trial/` |

**数据结构**:
```
EEG_01-29/subject XX/
  cnt.mat          # 连续EEG数据 (1x6 cells: MI/MA 交替 x 3 sessions)
  mrk.mat          # 事件标记 (1x6 cells)
  mnt.mat          # 电极位置

NIRS_01-29/subject XX/
  cnt.mat          # 连续NIRS光强数据 (需转换为 HbO/HbR)
  mrk.mat          # 事件标记 (marker 1/2)
  mnt.mat          # 光极位置
```

**同步检查结果** (2026-01-14):

通过对全部 29 名被试的 marker 对齐分析，验证了 EEG 和 fNIRS 的同步质量：

| 指标 | 结果 | 说明 |
|------|------|------|
| 偏移量 (NIRS - EEG) | 52.2 ± 2.4 s | NIRS 包含更长的预实验静息期 |
| 事件间隔差异 | mean ≈ 0 ms | 两模态事件间隔高度一致 |
| 事件间隔最大差异 | 64-78 ms | 在可接受范围内 (< 100ms) |
| 标签匹配 | 100% | 所有被试所有 session 标签完全一致 |

**结论**: 虽然存在固定时间偏移（因录制起点不同），但事件触发器精确对齐，可安全使用 marker 时间戳进行跨模态窗口裁剪。

**数据加载器**: `src/data/eeg_fnirs_dataset.py`
- `BBCIDataLoader`: 低层数据访问，包含同步检查功能
- `EEGfNIRSDataset`: 单模态 PyTorch Dataset
- `MultiModalEEGfNIRSDataset`: 双模态同步 Dataset
- `create_dataloaders()`: 创建 train/val/test 分割

#### 1.2.2 数据预处理规范

基于 EEG+NIRS Single-Trial 数据集特点的预处理规范：

**EEG 预处理**
- 采样率：200Hz (已由数据集提供方下采样)
- 滤波：带通滤波 0.5-45Hz (需实现)
- 参考方式：已 re-reference 到 linked mastoids
- 伪迹处理：数据集提供 artifact 数据可选择跳过
- 通道选择：30 通道全使用，或选择运动区 ROI (C3, C4, Cz 周围)

**fNIRS 预处理**
- 采样率：10Hz (已由数据集提供方下采样)
- 信号转换：原始数据为光强，需转换为 HbO/HbR (Modified Beer-Lambert Law)
- 滤波：低通滤波 0.1Hz (去除心跳等高频噪声)
- 通道选择：36 通道全使用，或选择运动区 ROI (C3, C4 周围各 4 通道)

**窗口化设计**
- 任务时长：10s (每个 trial)
- EEG 窗口：512 samples = 2.56s @ 200Hz
- fNIRS 窗口：26 samples ≈ 2.5s @ 10.4Hz
- 步长：50% 重叠
- 对齐策略：以 marker 时间戳为基准，裁剪对应时段

### 1.3 实验设计

#### 1.3.1 Phase 0 基础实验（已完成）

| 实验 | 模态 | Tokenizer | Test MSE | Perplexity | Utilization | 状态 |
|------|------|-----------|----------|------------|-------------|------|
| P0-EEG-FSQ | EEG | FSQ (4096) | 0.0767 | 727.4 | 53.5% | ✅ |
| P0-EEG-VQ | EEG | VQ-VAE (512) | **0.0742** | 382.3 | **100%** | ✅ |
| P0-fNIRS-FSQ | fNIRS | FSQ (512) | **0.0538** | 329.2 | 85.7% | ✅ |
| P0-fNIRS-VQ | fNIRS | VQ-VAE (256) | 0.0613 | 189.1 | 95.3% | ✅ |

**结论**：
- EEG: VQ-VAE 重构更好，codebook 无死码
- fNIRS: FSQ 重构更好，perplexity 更高
- 问题：当前使用单通道、固定序列长度

#### 1.3.2 P0+.1: 多通道输入设计

**目标**：利用多通道空间信息提升表示质量

**EEG 多通道方案**：

| 方案 | 输入 shape | 描述 | 优势 | 劣势 |
|------|------------|------|------|------|
| A1: 通道平均 | [B, T] | 空间平均后单通道 | 简单，已验证 | 丢失空间信息 |
| A2: 运动区 ROI | [B, 6, T] | C3/C4/Cz 及周围 | 任务相关，降低维度 | 需要先验知识 |
| A3: 全通道独立 | [B, 30, T] | 每通道独立 token 化 | 保留全部信息 | 计算量大 |
| **A4: 全通道共享** | [B, 30, T] → [B, T, 30] | 多通道作为特征维度 | 学习空间模式 | Encoder 需适配 |

**推荐方案**: A4（全通道共享 Encoder）
- 输入：[B, C, T] = [B, 30, 512]
- Encoder：将通道视为特征维度，Conv1d 沿时间轴
- 或使用 2D Conv：时间×通道

**fNIRS 多通道方案**：

| 方案 | 输入 shape | 描述 |
|------|------------|------|
| B1: 单通道 | [B, T] | 当前方案 |
| B2: HbO+HbR | [B, 2, T] | 双通道（同位置） |
| **B3: 运动区 ROI** | [B, 8, T] | C3/C4 周围 8 通道 |

**推荐方案**: B3（运动区 ROI）
- 选择运动皮层上方的 8 个通道
- 包含 HbO 或仅 HbO（HbR 信噪比较低）

#### 1.3.3 P0+.2: 序列长度优化

**当前问题**：
- EEG: 512 samples @ 200Hz = 2.56s，可能包含任务无关段
- fNIRS: 25 samples @ 10Hz = 2.5s，血流动力学响应未完全展开

**优化方案**：

| 模态 | 当前 | 建议 | 时长 | 理由 |
|------|------|------|------|------|
| EEG | 512 | **400** | 2.0s | 聚焦任务开始后的关键时段 |
| fNIRS | 25 | **80** | 8.0s | 覆盖 HRF 峰值（~5-6s） |

**验证方法**：
- 训练不同长度的 tokenizer，比较：
  1. 重构质量
  2. 下游分类准确率
  3. Token 序列的判别性

#### 1.3.4 P0+.3: 跨被试泛化验证

| 实验 | 训练 | 测试 | 目标 |
|------|------|------|------|
| Within-subject | Subject 1-20, Session 1-2 | Subject 1-20, Session 3 | Session 泛化 |
| Cross-subject | Subject 1-20 | Subject 21-29 | 被试泛化 |
| Leave-one-out | Subject 2-29 | Subject 1 | 极端泛化 |

**评估指标**：
- Generalization Gap = Test MSE / Train MSE（期望 < 2.0）
- Token 分布一致性（KL divergence）

### 1.4 Success Criteria

| 阶段 | 指标 | 阈值 |
|------|------|------|
| P0+ | Multi-channel MSE | < 单通道 MSE × 0.9 |
| P0+ | Cross-subject Gap | Test MSE / Train MSE < 2.0 |
| P0+ | Token Perplexity | > 30% of codebook size |

---

## 2. Phase 1A: 全数据流验证（分类任务）

### 2.1 目标

验证 token 表示能否支持下游分类任务，建立端到端 baseline。

### 2.2 任务定义

**Motor Imagery 二分类**：
- 类别 0: 左手想象 (LMI)
- 类别 1: 右手想象 (RMI)
- Chance level: 50%

### 2.3 分类器设计

#### 2.3.1 输入形式

| 输入类型 | Shape | 描述 |
|----------|-------|------|
| Token indices | [B, T'] | 离散 token 序列 |
| Quantized latent | [B, T', D] | 量化后的连续表示 |
| Pre-quantized | [B, T', D] | 量化前的 encoder 输出 |

**推荐**：使用 Quantized latent，保留连续信息且包含 codebook 约束

#### 2.3.2 分类器架构

| 方案 | 架构 | 复杂度 | 描述 |
|------|------|--------|------|
| C1: Simple | Pool + Linear | 低 | 全局平均池化 → 线性层 |
| C2: Temporal | LSTM/GRU | 中 | 建模时序依赖 |
| C3: Attention | Transformer | 高 | 自注意力机制 |

**推荐起步方案**: C1（Simple Pool + Linear）
```
z_q: [B, T', D] → AvgPool → [B, D] → Linear → [B, 2]
```

### 2.4 实验设计

| 实验 ID | 输入 | 分类器 | 目标 |
|---------|------|--------|------|
| P1A-EEG-simple | EEG tokens | Pool+Linear | EEG 单模态 baseline |
| P1A-fNIRS-simple | fNIRS tokens | Pool+Linear | fNIRS 单模态 baseline |
| P1A-Fusion-concat | EEG + fNIRS | Concat+Linear | 简单特征融合 |
| P1A-Fusion-late | EEG + fNIRS | 双流 + 融合 | 后期决策融合 |

### 2.5 Baseline 对比

| Baseline | 描述 | 预期 |
|----------|------|------|
| Raw signal | 原始信号 → Pool+Linear | 参考点 |
| Pre-tokenizer | Encoder 输出（无量化） | 量化损失评估 |
| Literature | 公开论文报告性能 | ~60-80% |

### 2.6 Success Criteria

| 指标 | 阈值 | 说明 |
|------|------|------|
| Token classification | > 55% | 显著高于 chance |
| vs Raw baseline | > 0.9× | Token 不显著劣于原始信号 |
| Fusion gain | > single best | 融合应有增益 |

---

## 3. Phase 1B: 跨模态 Token 分析

### 3.1 目标

在 tokenizer 稳定后，分析 EEG 和 fNIRS token 序列之间的关系，为对齐策略设计提供 empirical 依据。

### 3.2 分析维度

#### 3.2.1 Token 序列统计分析 (P1B.1)

| 分析项 | 描述 | 工具 |
|--------|------|------|
| Token 频率分布 | 各 code 使用频率对比 | Histogram, KL divergence |
| 序列熵 | Token 序列的信息量 | Entropy, normalized entropy |
| 转移概率矩阵 | Token 间转移模式 | Markov chain analysis |
| 稀有 token | 低频 token 的语义 | 可视化对应时段 |

**关键问题**：
- EEG 和 fNIRS 的 token 分布是否相似？
- 高频 token 是否对应相似的生理状态？

#### 3.2.2 时序耦合分析 (P1B.2)

| 分析项 | 描述 | 工具 |
|--------|------|------|
| Cross-correlation | 两序列的线性相关 | `scipy.signal.correlate` |
| Mutual Information | 不同 lag 的互信息 | `sklearn.metrics.mutual_info_score` |
| Phase coupling | 相位同步 | Hilbert transform |

**关键问题**：
- 两模态 token 序列是否存在时滞相关？
- 最大相关出现在什么 lag？（反映神经血管耦合）

#### 3.2.3 因果关系探索 (P1B.3)

| 分析项 | 描述 | 工具 |
|--------|------|------|
| Granger Causality | 一序列是否预测另一序列 | `statsmodels.tsa.stattools` |
| Transfer Entropy | 信息流方向 | `pyinform` 或自实现 |
| Convergent Cross Mapping | 非线性因果 | `pyEDM` |

**关键问题**：
- EEG token 是否能"预测" fNIRS token？
- 信息流是单向还是双向？

### 3.3 预期发现与设计指导

| 发现模式 | 对齐策略建议 |
|----------|--------------|
| 强时序耦合（固定 lag） | 时间对齐 + lag 补偿 |
| 语义相关但时序不同步 | 对比学习 / CCA 对齐 |
| 单向因果关系 | 预测模型（EEG→fNIRS） |
| 弱/无关联 | 独立处理，后期融合 |

### 3.4 实验设计

| 实验 ID | 分析内容 | 输出 |
|---------|----------|------|
| P1B-stat | Token 统计分布对比 | 分布图、熵值表 |
| P1B-xcorr | 时序相关性分析 | 相关曲线、最优 lag |
| P1B-causal | 因果关系检验 | Granger/TE 结果 |
| P1B-viz | 同步 trial 可视化 | 双模态 token 序列对比图 |

---

## 4. Phase 2: 对齐策略设计（规划中）

### 4.1 目标

基于 Phase 1B 的 empirical 发现，设计合适的跨模态对齐策略。

### 4.2 候选方法

| 方法 | 适用场景 | 复杂度 |
|------|----------|--------|
| 时间平移对齐 | 固定 lag 耦合 | 低 |
| CCA 对齐 | 线性语义相关 | 中 |
| 对比学习 | 非线性语义相关 | 高 |
| Cross-attention | 动态对齐 | 高 |

### 4.3 时间线

Phase 2 将在 Phase 1A/1B 完成后开始，预计 Week 4+。

---

## 5. 代码结构

### 5.1 当前结构

```text
src/
  tokenizers/           # Tokenizer 实现
    __init__.py
    base.py            # 基类：BaseTokenizer, Conv1dEncoder/Decoder
    fsq.py             # Finite Scalar Quantization ✅
    vqvae.py           # VQ-VAE ✅
  data/                 # 数据加载
    __init__.py
    eeg_fnirs_dataset.py    # EEG+NIRS 真实数据加载 ✅
    synthetic_timeseries.py # 合成数据
  metrics/              # 评估指标
    __init__.py
    codebook_health.py  # perplexity, usage, dead codes ✅
    reconstruction.py   # MSE, spectral loss ✅
  classifiers/          # 下游分类器 (待实现)
    __init__.py
    simple_classifier.py    # Pool + Linear
    sequence_classifier.py  # LSTM/Transformer
  analysis/             # 跨模态分析 (待实现)
    __init__.py
    token_statistics.py     # Token 分布、熵
    temporal_coupling.py    # 时序相关性分析
    causal_analysis.py      # Granger/Transfer Entropy
  models/               # 旧 ELP 模型（存档）
  losses/               # 旧 PID 损失（存档）
  utils/                # 工具函数
  visualization/        # 可视化
    tokenizer_plots.py  # Tokenizer 可视化 ✅
```

### 5.2 实验结构

```text
experiments/
  configs/
    base.yaml                  # 基础配置
    phase0/                    # Phase 0 实验配置
      P0_eeg_fsq.yaml
      P0_eeg_vqvae.yaml
      P0_fnirs_fsq.yaml
      P0_fnirs_vqvae.yaml
    phase0plus/                # Phase 0+ 多通道实验 (待添加)
      P0plus_eeg_multichannel.yaml
      P0plus_fnirs_multichannel.yaml
    phase1a/                   # Phase 1A 分类实验 (待添加)
      P1A_eeg_classification.yaml
      P1A_fnirs_classification.yaml
      P1A_fusion.yaml
    phase1b/                   # Phase 1B 分析配置 (待添加)
      P1B_token_analysis.yaml
  scripts/
    train_tokenizer.py         # 通用训练脚本 ✅
    run_tokenizer_comparison.py # 对比实验脚本 ✅
    train_classifier.py        # 分类器训练 (待实现)
    analyze_tokens.py          # Token 分析 (待实现)
  runs/                        # 实验运行记录
    comparison_20260114_*/     # Tokenizer 对比实验 ✅
  results/                     # 汇总结果
```

---

## 6. 下一步行动

### ✅ 已完成

1. [x] **数据集调研**：寻找合适的公开 EEG/fNIRS 数据集 ✅ 2026-01-14
   - 选定 EEG+NIRS Single-Trial 数据集 (TU Berlin)
2. [x] **数据加载模块**：实现 `src/data/eeg_fnirs_dataset.py` ✅ 2026-01-14
   - 解析 BBCI Toolbox .mat 格式
   - 实现窗口化和标签提取
   - 验证 29 名被试的 EEG/fNIRS 同步对齐
3. [x] **评估指标模块**：完善 `src/metrics/codebook_health.py` ✅ 已实现
4. [x] **训练脚本**：完善 `experiments/scripts/train_tokenizer.py` ✅ 已实现
5. [x] **Phase 0 实验**：FSQ/VQ-VAE × EEG/fNIRS 对比 ✅ 2026-01-14
   - 完成 4 组对比实验
   - 结论：VQ-VAE 适合 EEG，FSQ 适合 fNIRS

### 🔄 当前：Phase 0+

**P0+.1：多通道 Tokenizer**（优先）

1. [ ] 修改 `src/tokenizers/base.py` 支持多通道输入
   - EEG: input_channels=30 (原有30通道作为特征维度)
   - fNIRS: input_channels=8 (C3/C4 区域 ROI)
2. [ ] 创建 `experiments/configs/phase0plus/` 配置目录
3. [ ] 运行多通道实验，对比单通道结果

**P0+.2：序列长度优化**

4. [ ] EEG 窗口从 100 扩展到 400 samples (2.0s)
5. [ ] fNIRS 窗口从 100 扩展到 80 samples (8.0s @ 10Hz)
6. [ ] 评估不同序列长度的重建质量和 token 利用率

### 📋 下阶段：Phase 1A（分类验证）

1. [ ] 实现 `src/classifiers/simple_classifier.py`
2. [ ] 创建分类实验配置 `experiments/configs/phase1a/`
3. [ ] 训练 Token 分类器，与 Raw 信号基线对比
4. [ ] 实现融合分类器（concat / late fusion）

### 📋 下阶段：Phase 1B（跨模态分析）

1. [ ] 实现 `src/analysis/token_statistics.py`
2. [ ] 实现 `src/analysis/temporal_coupling.py`
3. [ ] 运行 token 统计分析（分布、熵、转移概率）
4. [ ] 运行时序耦合分析（cross-correlation, MI）
5. [ ] 运行因果关系检验（Granger causality）

### 📋 远期：Phase 2（对齐策略）

- 基于 Phase 1B 发现设计对齐策略
- 候选方法：时间对齐、CCA、对比学习

---

## 7. 待讨论事项

以下事项将在实验推进过程中逐步确定：

1. ~~**数据集选择**~~：✅ 已选定 EEG+NIRS Single-Trial
2. ~~**Tokenizer 选择**~~：✅ VQ-VAE 用于 EEG，FSQ 用于 fNIRS
3. **多通道策略细化**：30 通道共享 vs 按区域分组？
4. **fNIRS ROI 选择**：C3/C4 区域具体包含哪些 optode？
5. **序列长度与 codebook 大小权衡**：更长序列需要更大 codebook？

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
