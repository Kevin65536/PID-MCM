# Archived Pre-Experiments Log

> This document contains the records of pre-experiment explorations. These experiments focused on tokenizer validation and initial downstream classification baselines.
> **Note on Bottlenecks:** The downstream Motor Imagery (MI) classification task suffered from a lack of cross-subject generalization (~50% accuracy). The models tended to overfit to subject-specific features. These runs have been archived to reset focus for the First Stage experiments.

---

## Experiment Index

| Date | ID | Phase | Description | Status |
|------|-----|-------|-------------|--------|
| 2026-01-30 | EXP-017 | P1B | EEG Token + Data Augmentation | ✅ 55.83% |
| 2026-01-30 | EXP-016 | P1B | Raw Signal Baselines (EEG/fNIRS/Multi) | ⚠️ ~50% |
| 2026-01-30 | EXP-015 | P1B | Unified Downstream MI Classification | ⚠️ ~50% |
| 2026-01-29 | EXP-014 | P0+ | LaBraM VQNSP v2: Freq-only Loss Ablation | ❌ Failed |
| 2026-01-27 | EXP-013 | P0+ | fNIRS LaBraM VQNSP Tokenizer | ✅ Complete |
| 2026-01-26 | EXP-012 | P0+ | EEG LaBraM VQNSP Tokenizer | ✅ Complete |
| 2026-01-15 | EXP-011 | P1A | Dual-Modality (4s MI window) | ⚠️ 46.7% |
| 2026-01-15 | EXP-010 | P1A | fNIRS Classification (4s MI window) | ⚠️ 49.2% |
| 2026-01-15 | EXP-009 | P1A | EEG Classification (4s MI window) | ⚠️ 55.0% |
| 2026-01-15 | EXP-008 | P0+ | LaBraM-style Tokenizers (4s window) | ✅ Complete |
| 2026-01-15 | EXP-007 | P1A | Dual-Modality Multi-Lead Classification | ⚠️ Chance level |
| 2026-01-15 | EXP-006 | P1A | fNIRS Multi-Lead Classification (5s) | ⚠️ Chance level |
| 2026-01-15 | EXP-005 | P1A | EEG Multi-Lead Classification (5s) | ⚠️ ~54% |
| 2026-01-15 | EXP-004 | P0+ | Time-aligned Tokenizers (5s window) | ✅ Complete |
| 2026-01-15 | EXP-003 | P1A | fNIRS Token Classification (single-channel) | ⚠️ Chance level |
| 2026-01-15 | EXP-002 | P1A | EEG Raw Baseline Classification | ⚠️ Chance level |
| 2026-01-15 | EXP-001 | P1A | EEG Token Classification (single-channel) | ⚠️ Chance level |
| 2026-01-14 | EXP-000 | P0 | Tokenizer Comparison (FSQ vs VQ-VAE) | ✅ Complete |

---

## EXP-017: EEG Token Classification with Data Augmentation (2026-01-30)

### Objective
通过数据增强和正则化技术改善跨被试运动想象分类的泛化能力。

### Motivation
EXP-015/016的分析显示：
1. 被试内分类准确率可达40%-83%，但跨被试泛化仅~50%
2. 个体差异极大，需要数据增强来提高鲁棒性
3. 需要更强的正则化防止过拟合

### Configuration

**Run Directory:** `experiments/runs/mi_eeg_token_aug_20260130_222111`

**增强策略:**
| Augmentation | Setting |
|--------------|---------|
| Time Shift | ±40 samples (±200ms @ 200Hz) |
| Channel Dropout | 10% probability |
| Gaussian Noise | std=0.05 |
| Amplitude Scaling | [0.9, 1.1] |
| Mixup | alpha=0.2 |

**正则化:**
| Parameter | Baseline | Augmented |
|-----------|----------|-----------|
| Dropout | 0.1 | **0.3** |
| Weight Decay | 0.0001 | **0.001** |
| Learning Rate | 0.001 | **0.0005** |
| Label Smoothing | 0 | **0.1** |
| Early Stopping Patience | 20 | **30** |
| Max Epochs | 100 | **150** |

### Results

| Metric | EEG Token (Baseline) | EEG Token + Aug | Improvement |
|--------|---------------------|-----------------|-------------|
| **Accuracy** | 51.25% | **55.83%** | **+4.58%** |
| Balanced Accuracy | 51.25% | 55.83% | +4.58% |
| Precision | 55.17% | 56.73% | +1.56% |
| Recall | 13.33% | 49.17% | +35.84% |
| **F1 Score** | 21.48% | **52.68%** | **+31.20%** |
| **Cohen's Kappa** | 2.50% | **11.67%** | **+9.17% (4.7x)** |
| ROC-AUC | 53.69% | 53.03% | -0.66% |

**Confusion Matrix:**
```
Baseline:     TN=107, FP=13, FN=104, TP=16  (预测偏向Left 87.9%)
Augmented:    TN=75,  FP=45, FN=61,  TP=59  (预测更平衡)
```

### Analysis

1. **准确率提升显著**: 从51.25%提升到55.83%，超过随机基线5.83个百分点
2. **预测偏差大幅改善**: 基线模型严重偏向预测"Left"类别，增强后预测更加平衡
3. **F1分数大幅提升**: 从21.48%提升到52.68%，增强使模型学到了更有意义的特征
4. **Cohen's Kappa提升4.7倍**: 表明一致性显著提高，不再是纯粹的随机预测

### Conclusion

✅ **数据增强有效改善跨被试泛化**

**关键洞察:**
- 时间偏移和Mixup帮助模型学习更鲁棒的时序特征
- 通道dropout防止模型过度依赖特定通道
- Label smoothing配合Mixup提供软标签监督，改善泛化

---

## EXP-016: Raw Signal Baseline Classification (2026-01-30)

### Objective
建立原始信号分类基线，验证Token-based方法的有效性。

### Motivation
EXP-015中Token-based方法表现接近随机，需要排除是否是tokenizer的问题：
- 如果Raw baseline表现更好，说明tokenizer损失了判别信息
- 如果Raw baseline表现相近，说明问题在于数据本身

### Implementation

**新增模块:**
- `src/data/augmentation.py`: 数据增强模块
- `src/classifiers/multi_lead.py`: 添加 `RawMultiLeadClassifier` 和 `RawDualModalityClassifier`

**Raw分类器架构:**
```
Input [B, C, T] 
  → Conv1D(1→32, k=25, s=4) + BN + GELU
  → Conv1D(32→64, k=11, s=2) + BN + GELU  
  → Conv1D(64→128, k=5, s=2) + BN + GELU
  → AdaptiveAvgPool1D(4)
  → FC(128*4 → 128)
  → Attention/Transformer Aggregation
  → Classifier
```

### Results

| Model | Accuracy | ROC-AUC | Kappa | Trainable Params |
|-------|----------|---------|-------|------------------|
| EEG Token | 51.25% | 53.69% | 2.50% | 33,667 |
| **EEG Raw** | **50.83%** | 48.76% | 1.67% | **155,971** |
| fNIRS Token | 49.17% | 51.47% | -1.67% | 8,643 |
| **fNIRS Raw** | **50.83%** | 49.40% | 1.67% | **67,267** |
| Multi Token | 52.08% | 50.25% | 4.17% | 62,852 |
| **Multi Raw** | **52.50%** | **55.60%** | **5.00%** | **294,404** |

### Analysis

**关键发现: Token和Raw表现几乎相同**

1. **Token ≈ Raw**: 两种方法测试准确率都在49-53%范围内
2. **Tokenizer未损失信息**: 如果tokenizer丢失了判别信息，Raw应该表现更好
3. **问题在数据层面**: 跨被试MI信号本身可区分性有限

### Conclusion

⚠️ **Raw baseline确认问题不在tokenizer，而在于跨被试泛化的固有困难**

---

## EXP-015: Unified Downstream MI Classification (2026-01-30)

### Objective
使用统一的下游任务训练框架，系统评估Token-based运动想象分类能力。

### Motivation
之前的分类实验（EXP-005~011）使用不同的脚本和配置，难以公平比较。需要：
1. 统一的训练脚本（类似train_tokenizer.py）
2. 统一的配置系统（YAML继承）
3. 标准化的评估指标和可视化

### Implementation

**新增文件:**
```
experiments/scripts/train_downstream.py          # 统一训练脚本 (989行)
experiments/configs/downstream/base_downstream.yaml    # 基础配置
experiments/configs/downstream/mi_eeg_token.yaml       # EEG Token配置
experiments/configs/downstream/mi_fnirs_token.yaml     # fNIRS Token配置  
experiments/configs/downstream/mi_multimodal_token.yaml # 多模态Token配置
experiments/configs/downstream/mi_eeg_raw_baseline.yaml     # EEG Raw配置
experiments/configs/downstream/mi_fnirs_raw_baseline.yaml   # fNIRS Raw配置
experiments/configs/downstream/mi_multimodal_raw_baseline.yaml # 多模态Raw配置
```

**实验设置:**
| Parameter | Value |
|-----------|-------|
| Task | Motor Imagery (Left vs Right) |
| Window | 4s (BCI standard) |
| EEG | 30 channels @ 200Hz = 800 samples |
| fNIRS | 36 HbO channels @ 10Hz = 40 samples |
| Train Subjects | 1-20 |
| Val Subjects | 21-25 |
| Test Subjects | 26-29 |
| Batch Size | 32 |
| Epochs | 100 |

**使用的Tokenizer checkpoints:**
- EEG: `eeg_labram_vqnsp_20260129_192557` (7.4M params, 64D latent)
- fNIRS: `fnirs_labram_vqnsp_20260130_170132` (1.25M params, 32D latent)

### Results

| Experiment | Accuracy | Balanced Acc | ROC-AUC | Kappa | Training Time |
|------------|----------|--------------|---------|-------|---------------|
| EEG Token | 51.25% | 51.25% | 53.69% | 2.50% | 13.1 min |
| fNIRS Token | 49.17% | 49.17% | 51.47% | -1.67% | 4.8 min |
| Multimodal Token | 52.08% | 52.08% | 50.25% | 4.17% | 24.8 min |

**所有结果接近50%随机基线。**

### Deep Dive: Within-Subject Analysis

为了理解为什么跨被试分类失败，我们对每个被试进行了被试内LDA 5-fold CV分析：

| Subject Split | Avg Within-Subject Acc | Range |
|---------------|------------------------|-------|
| Train (S01-S20) | 53.2% ± 9.1% | 40.0% - 75.0% |
| Val (S21-S25) | 53.3% ± 10.5% | 38.3% - 66.7% |
| Test (S26-S29) | 59.2% ± 16.9% | 41.7% - 83.3% |

**个体差异极大:**
- S26: 83.3% (MI信号强)
- S02: 75.0% (MI信号强)
- S28: 41.7% (MI信号弱或噪声大)
- S20: 40.0% (MI信号弱或噪声大)

### Root Cause Analysis

1. **被试间差异是核心障碍**: 
   - 被试内可达83%，跨被试仅~50%
   - 模型学到的是被试特异性特征而非通用MI特征

2. **多模态融合未能帮助**:
   - 早期融合策略可能引入噪声
   - EEG和fNIRS的时间尺度差异大

3. **数据集特性**:
   - 29个被试中只有少数有高质量MI信号
   - 类别完全平衡 (50%/50%)，标签正确

### Conclusion

⚠️ **跨被试运动想象分类是公认的难题，需要更先进的domain adaptation技术**

---

## EXP-014: LaBraM VQNSP v2 - Frequency-Only Loss Ablation (2026-01-29)

### Objective
测试完全移除时域损失（遵循LaBraM原论文设计）对重建质量的影响。

### Motivation
EXP-012/013的v1实验中，码本使用率起初低于50%。用户希望：
1. 移除时域损失（LaBraM原论文仅使用频域损失）
2. 降低β值（commitment loss权重）
3. 增大batch size以改善训练稳定性

### Configuration Changes (v1 → v2)

| Parameter | v1 EEG | v2 EEG | v1 fNIRS | v2 fNIRS |
|-----------|--------|--------|----------|----------|
| time_weight | 0.5 | **0.0** | 1.0 | **0.0** |
| beta | 1.0 | **0.25** | 1.0 | **0.25** |
| batch_size | 128 | **256** | 256 | **512** |
| lr | 0.0003 | **0.0005** | 0.0002 | **0.0005** |
| use_smooth_l1 | false | **true** | false | **true** |

### Training Results

**EEG v2 Run:** `experiments/runs/eeg_labram_vqnsp_v2_20260129_172418`
- Early stopped at epoch 112 (best at epoch 82)
- Best val_loss: 0.3752

**fNIRS v2 Run:** `experiments/runs/fnirs_labram_vqnsp_v2_20260129_172449`
- Completed 150 epochs (best at epoch 146)
- Best val_loss: 0.1468

### Results Comparison (Test Set)

| Modality | Version | Time Corr | Spectral Corr | Utilization |
|----------|---------|-----------|---------------|-------------|
| **EEG** | v1 (time_weight=0.5) | **0.7441** | **0.8744** | 100% |
| EEG | v2 (time_weight=0.0) | 0.0522 | 0.3462 | 100% |
| | **Δ Change** | **-0.6919** | **-0.5282** | - |
| **fNIRS** | v1 (time_weight=1.0) | **0.8295** | **0.8184** | 100% |
| fNIRS | v2 (time_weight=0.0) | 0.0454 | 0.2304 | 100% |
| | **Δ Change** | **-0.7841** | **-0.5879** | - |

### Analysis

**关键发现：移除时域损失导致重建质量灾难性下降**

1. **时间域相关性骤降**：
   - EEG: 0.74 → 0.05 (降低93%)
   - fNIRS: 0.83 → 0.05 (降低95%)
   - 接近随机重建水平

2. **频谱相关性也大幅下降**：
   - EEG: 0.87 → 0.35 (降低60%)
   - fNIRS: 0.82 → 0.23 (降低72%)
   - 即使仅优化频域损失，频谱重建也不如v1

3. **为什么v2失败？**
   - 纯频域损失无法约束时域相位一致性
   - 解码器可能找到了"欺骗"频域损失的解
   - amplitude + phase loss并不等同于完美时域重建

4. **LaBraM原设计的考量**：
   - LaBraM使用BERT-style masked prediction，不需要完美重建
   - 其目标是学习语义表示，而非信号重建
   - 我们的目标是tokenizer，需要高质量重建

### Conclusion

❌ **实验失败** - 移除时域损失严重损害重建质量

**建议**：
- ✅ 继续使用v1配置 (time_weight > 0)
- ✅ v1已达到100%码本使用率（通过dead code revival）
- ✅ v1重建质量优秀：EEG 0.74, fNIRS 0.83 时间相关性

**下一步**：使用v1 tokenizer进行下游分类任务

---

## EXP-012: LaBraM VQNSP Tokenizer Implementation (2026-01-26)

### Objective
实现 LaBraM 风格的 VQNSP tokenizer 作为 NeuroRVQ 的替代方案。NeuroRVQ 由于其复杂的多分支多层 RVQ 设计难以训练，重建损失收敛较差。

### Motivation
NeuroRVQ 训练结果分析 (eeg_neurorvq_20260123_100117):
- 150 个 epoch 后 val_rec_loss = 1.57 (较高)
- 时间域重建相关性 ≈ 0 (非常差)
- 复杂的多分支设计导致优化困难

### Solution: LaBraM VQNSP
基于 LaBraM 论文的 VQNSP 架构，关键简化：
1. **简单架构**: Transformer Encoder → Single VQ → Transformer Decoder
2. **NormEMA VQ**: L2 归一化 + EMA 更新的 codebook（更稳定）
3. **频域重建**: 分别预测幅度和相位（跟随 LaBraM）
4. **单层 VQ**: 不使用多层 RVQ，更容易训练

### Implementation
新增文件：
- `src/tokenizers/labram_vqnsp.py`: 主要实现
- `experiments/configs/phase0/P0_eeg_labram_vqnsp.yaml`: EEG 配置
- `experiments/configs/phase0/P0_fnirs_labram_vqnsp.yaml`: fNIRS 配置

注册的 tokenizer 类型：
- `labram_vqnsp`: 基础版本
- `labram_vqnsp_eeg`: EEG 优化版 (200Hz, 200 samples/patch)
- `labram_vqnsp_fnirs`: fNIRS 优化版 (10Hz, 40 samples/patch)

### Architecture Summary
```
Input [B, T] → Split to Patches [B, N, P]
                    ↓
              FFT Features (amp + phase)
                    ↓
              Patch Embedding [B, N, D]
                    ↓
              Transformer Encoder (6 layers)
                    ↓
              Project to Codebook Dim
                    ↓
              NormEMA VQ (8192 codes x 64D)
                    ↓
              Project to Decoder Dim
                    ↓
              Transformer Decoder (3 layers)
                    ↓
              Amplitude Head + Phase Head
                    ↓
              iFFT → Reconstructed [B, T]
```

### Key Features
- **K-means 初始化**: Codebook 使用 k-means 初始化而非随机初始化
- **L2 归一化**: 输入和 codebook 都进行 L2 归一化，使用余弦相似度
- **EMA 更新**: Codebook 通过 EMA 更新，不需要梯度传播
- **频域损失**: amplitude_loss + phase_loss + (optional) time_loss

### Model Size
- EEG version: ~7.4M parameters
- fNIRS version: ~1.3M parameters

### Usage
```python
from src.tokenizers import LaBraMVQNSP_EEG, create_tokenizer

# Direct usage
model = LaBraMVQNSP_EEG(
    patch_size=200,
    seq_length=800,
    codebook_size=8192,
)

# Via registry
config = {'model': {'type': 'labram_vqnsp_eeg', ...}}
model = create_tokenizer(config)

# Training
python train_tokenizer.py --config phase0/P0_eeg_labram_vqnsp.yaml
```

### Status
✅ 训练完成，效果优于 NeuroRVQ

### Training Results (2026-01-26)
**Run:** `experiments/runs/eeg_labram_vqnsp_20260126_212630`

**配置:**
- Epochs: 150
- Batch Size: 128
- Learning Rate: 3e-4 (cosine schedule)
- Codebook: 8192 codes × 64D
- 新增: 死码复活机制 (dead_code_threshold=10)

**最终指标:**
| Metric | Val Set | Test Set | vs NeuroRVQ |
|--------|---------|----------|-------------|
| Loss | **0.9675** | **0.9745** | ↓40% (1.6261) |
| Time Correlation | 0.6737 | **0.7441 ± 0.15** | ↑显著 (~0) |
| Spectral Correlation | 0.8324 | **0.8744 ± 0.10** | 新指标 |
| Code Utilization | 100% | **100%** | ↑625× (0.16%) |

**关键改进:**
1. ✅ **死码复活机制完美工作** - 全部8192个码字都被使用
2. ✅ **重建质量大幅提升** - 时间相关性从~0提升到0.67
3. ✅ **频谱保真度优秀** - 频谱相关性达到0.83
4. ✅ **训练稳定** - 150 epoch平稳收敛

### Comparison with NeuroRVQ

| Aspect | NeuroRVQ | LaBraM VQNSP |
|--------|----------|--------------|
| Architecture | Multi-branch Inception + 8-layer RVQ | Transformer + Single VQ |
| Training | 难以优化，损失较高 | 简单稳定，快速收敛 |
| Reconstruction | 时间域相关性~0 | 时间域相关性0.67 |
| Codebook | 严重崩塌 (0.16%) | 完全利用 (100%) |
| Parameters | ~10M | 7.4M |

### Next Steps
1. ✅ ~~使用真实 EEG/fNIRS 数据训练~~ (完成)
2. ✅ ~~与 NeuroRVQ 对比重建质量~~ (完成，显著更优)
3. ✅ ~~fNIRS版本训练~~ (完成，见 EXP-013)
4. 用于下游分类任务 (P1A phase)

---

## EXP-013: fNIRS LaBraM VQNSP Tokenizer (2026-01-27)

### Objective
为fNIRS信号训练LaBraM VQNSP tokenizer，测试其在低采样率血氧信号上的重建效果。

### Configuration
**Run:** `experiments/runs/fnirs_labram_vqnsp_20260127_124919`

```yaml
Model: LaBraMVQNSP_fNIRS
- Sampling Rate: 10Hz
- Window: 40 samples (4s)
- Patch Size: 10 samples (1s)
- Patches per window: 4

Architecture:
- Encoder: 4 layers, 128D, 4 heads
- Decoder: 2 layers, 128D, 4 heads
- Codebook: 4096 codes × 32D
- Parameters: 1.25M

Loss Weights:
- Amplitude: 1.0
- Phase: 0.5 (less important for fNIRS)
- Time: 1.0 (more important for fNIRS)

Training:
- Epochs: 150
- Batch Size: 256
- Learning Rate: 3e-4 (cosine)
```

### Results

**验证集 (Best @ Epoch 128):**
| Metric | Value |
|--------|-------|
| Val Loss | 0.6017 |
| Amp Loss | 0.1963 |
| Phase Loss | 0.1764 |
| Time Loss | 0.3159 |
| Code Utilization | 100% |

**测试集 (4 subjects, 240 windows × 36 channels = 8640 samples):**
| Metric | Value |
|--------|-------|
| Test Loss | **0.6071** |
| Time Correlation | **0.8295 ± 0.15** |
| Spectral Correlation | **0.8184 ± 0.22** |
| Code Utilization | **100%** |

### Comparison: EEG vs fNIRS LaBraM VQNSP

| Metric | EEG | fNIRS |
|--------|-----|-------|
| Parameters | 7.4M | 1.25M |
| Codebook Size | 8192 | 4096 |
| Val Loss | 0.97 | 0.60 |
| Test Loss | 0.97 | 0.61 |
| Time Correlation | 0.74 | **0.83** |
| Spectral Correlation | **0.87** | 0.82 |
| Training Time | ~66 min | ~13 min |

### Conclusions
1. ✅ **fNIRS重建质量更高** - 时间相关性0.83 > EEG的0.74（fNIRS信号更平滑，更易重建）
2. ✅ **模型更小更快** - 1.25M参数，训练仅13分钟
3. ✅ **码本完全利用** - 死码复活机制同样有效
4. ✅ **较小码本足够** - 4096个码字对fNIRS足够（信号动态范围较小）

---

## EXP-011: Dual-Modality Classification with 4s MI Window (2026-01-15)

### Objective
使用4秒MI标准窗口和LaBraM风格tokenizer进行双模态融合分类。


### Configuration
```yaml
Run: experiments/runs/P1A_both_attention_early_20260115_233642/
Tokenizers: 
  - EEG: VQVAE (800 samples = 4s @ 200Hz), 100 tokens
  - fNIRS: VQVAE (40 samples = 4s @ 10Hz), 10 tokens
Window Offset: 500ms (MI response delay)
Classifier: DualModalityMultiLeadClassifier
  - EEG leads: 32
  - fNIRS leads: 36
  - Aggregation: attention
  - Fusion: early
```

### Results
```
| Metric | Value |
|--------|-------|
| Accuracy | 46.7% |
| Precision | 47.6% |
| Recall | 67.5% |
| F1 | 55.9% |

Confusion Matrix (Test):
[[31, 89]
 [39, 81]]
```

### Conclusion
- ⚠️ 双模态融合性能反而下降（46.7% < 55.0%）
- 可能原因：两个模态的噪声叠加，或融合层需要更多训练数据

---

## EXP-010: fNIRS Classification with 4s MI Window (2026-01-15)

### Objective
使用4秒MI窗口和对齐的fNIRS VQVAE tokenizer进行分类。

### Configuration
```yaml
Run: experiments/runs/P1A_fnirs_attention_20260115_233533/
Tokenizer: VQVAE_fNIRS_Aligned (4s window, 10 tokens, dim=64)
Window Offset: 500ms
Classifier: MultiLeadClassifier
  - Leads: 36
  - Aggregation: attention
```

### Results
```
| Metric | Value |
|--------|-------|
| Accuracy | 49.2% |
| Precision | 49.5% |
| Recall | 89.2% |
| F1 | 63.7% |

Confusion Matrix (Test):
[[11, 109]
 [13, 107]]
```

### Conclusion
- ⚠️ fNIRS分类仍接近chance level
- 高recall但极低precision说明模型偏向预测一个类别
- fNIRS对MI任务可能不敏感

---

## EXP-009: EEG Classification with 4s MI Window (2026-01-15)

### Objective
使用4秒MI标准窗口和LaBraM风格tokenizer进行EEG分类。这是最接近标准MI-BCI实验设置的实验。

### Configuration
```yaml
Run: experiments/runs/P1A_eeg_attention_20260115_232738/
Tokenizer: VQVAE_EEG_LaBraM
  - Input: 800 samples (4s @ 200Hz)
  - Output: 100 tokens
  - Codebook: 1024, dim=64
  - Test MSE: 0.0186, Utilization: 99%
Window Offset: 500ms (standard MI response delay)
Classifier: MultiLeadClassifier
  - Leads: 32
  - Aggregation: attention
  - Hidden dim: 128
  - Params: 33,667
```

### Results
```
| Metric | Value |
|--------|-------|
| Accuracy | 55.0% |
| Precision | 54.5% |
| Recall | 60.0% |
| F1 | 57.1% |

Confusion Matrix (Test):
[[60, 60]
 [48, 72]]
```

### Conclusion
- ⚠️ 准确率55%，略高于chance level但仍不理想
- **这是本阶段最好的结果**
- 与文献报告的MI-BCI准确率(60-80%)仍有差距
- **关键问题**: 冻结的tokenizer可能不包含任务判别特征

---

## EXP-008: LaBraM-style Tokenizers with 4s Window (2026-01-15)

### Objective
按照LaBraM/NeuroLM等EEG foundation model的标准做法，训练4秒窗口的VQ-VAE tokenizer。

### Code Changes
- `experiments/scripts/train_labram_tokenizers.py`: 新增LaBraM风格tokenizer训练脚本

### Configuration
```yaml
Run: experiments/runs/LaBraM_tokenizers_20260115_232415/

EEG Tokenizer:
  - Input: 800 samples (4.0s @ 200Hz)
  - Encoder: (64, 128, 256), kernel=7, stride=2
  - Output: 100 tokens, dim=64
  - Codebook: 1024, commitment=0.25, EMA decay=0.99

fNIRS Tokenizer (Aligned):
  - Input: 40 samples (4.0s @ 10Hz)
  - Encoder: (64, 128), kernel=5, stride=2
  - Output: 10 tokens, dim=64
  - Codebook: 1024, commitment=0.25, EMA decay=0.99

Training: 100 epochs, batch_size=64, lr=0.001
```

### Results
```
| Model | MSE | Perplexity | Utilization | Dead Codes |
|-------|-----|------------|-------------|------------|
| VQVAE_EEG_LaBraM | 0.0186 | 760.2 | 99.0% | 10/1024 |
| VQVAE_fNIRS_Aligned | 0.0066 | 374.4 | 49.7% | 515/1024 |
```

### Conclusion
- ✅ EEG tokenizer表现优秀：低MSE，高perplexity，99%利用率
- ✅ fNIRS tokenizer重建质量好，但codebook利用率较低(50%)
- ✅ 两模态token维度对齐(dim=64)，时间对齐(4s)
- **下一步**: 使用这些tokenizer进行下游分类

---

## EXP-007: Dual-Modality Multi-Lead Classification (2026-01-15)

### Objective
测试 EEG + fNIRS 双模态融合是否能提升 Motor Imagery 分类性能。

### Code Changes
- `src/classifiers/multi_lead.py`: 新增 `DualModalityMultiLeadClassifier`

### Configuration
```yaml
Run: experiments/runs/P1A_both_attention_early_20260115_175050/
Tokenizers: VQVAE_EEG_5s + FSQ_fNIRS_5s (frozen)
Classifier: DualModalityMultiLeadClassifier
  - EEG leads: 30
  - fNIRS leads: 36
  - Aggregation: attention
  - Fusion: early
  - Hidden dim: 128
Training:
  - Epochs: 24 (early stop)
  - Batch size: 32
  - Learning rate: 0.001
```

### Results
```
| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | 52.8% | 48.0% | 50.4% |
| Precision | - | - | 50.3% |
| Recall | - | - | 80.8% |
| F1 | - | - | 62.0% |

Confusion Matrix (Test):
[[24, 96]
 [23, 97]]
```

### Conclusion
- ⚠️ 双模态融合未带来性能提升
- 模型仍倾向于预测单一类别
- 可能原因：两模态的 token 表示都缺乏任务判别性

---

## EXP-006: fNIRS Multi-Lead Classification (2026-01-15)

### Objective
使用多导联 fNIRS 信号进行 Motor Imagery 分类。

### Code Changes
- `experiments/scripts/train_multilead_classifier.py`: 多导联分类训练脚本

### Configuration
```yaml
Run: experiments/runs/P1A_fnirs_attention_20260115_174936/
Tokenizer: FSQ_fNIRS_5s (frozen, 5s window, 50 samples @ 10Hz)
Classifier: MultiLeadClassifier
  - Leads: 36
  - Aggregation: attention
  - Hidden dim: 128
Training:
  - Epochs: 30
  - Batch size: 32
  - Learning rate: 0.001
```

### Results
```
| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | 51.4% | 47.3% | 48.8% |
| Precision | - | - | 49.4% |
| Recall | - | - | 95.8% |
| F1 | - | - | 65.2% |

Confusion Matrix (Test):
[[2, 118]
 [5, 115]]
```

### Conclusion
- ⚠️ fNIRS 分类性能仍接近 chance level
- 模型几乎全部预测为同一类别
- 可能原因：fNIRS token 表示（latent_dim=3）过于简单

---

## EXP-005: EEG Multi-Lead Classification (2026-01-15)

### Objective
使用多导联 EEG 信号进行 Motor Imagery 分类，测试空间信息是否有帮助。

### Code Changes
- `src/classifiers/multi_lead.py`: 新增 `MultiLeadClassifier`
- `experiments/scripts/train_multilead_classifier.py`: 多导联分类训练脚本

### Configuration
```yaml
Run: experiments/runs/P1A_eeg_attention_20260115_174603/
Tokenizer: VQVAE_EEG_5s (frozen, 5s window, 1000 samples @ 200Hz)
Classifier: MultiLeadClassifier
  - Leads: 30
  - Aggregation: attention
  - Hidden dim: 128
  - Trainable params: 33,667
Training:
  - Epochs: 22 (early stop)
  - Batch size: 32
  - Learning rate: 0.001
```

### Results
```
| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | 52.2% | 50.7% | 53.8% |
| Precision | - | - | 54.8% |
| Recall | - | - | 42.5% |
| F1 | - | - | 47.9% |

Confusion Matrix (Test):
[[78, 42]
 [69, 51]]
```

### Conclusion
- ⚠️ 相比单通道平均 (50.4%), 多导联方法略有提升 (53.8%)
- 但仍然接近 chance level (50%)
- 可能原因：
  1. Tokenizer 学习的是重构最优表示，不是判别性表示
  2. Attention 聚合可能无法学习 C3/C4 空间模式
  3. 需要更长训练或更复杂的分类器

---

## EXP-004: Time-aligned Tokenizers (2026-01-15)

### Objective
训练时间对齐的 tokenizer，使两模态的 token 序列对应相同时长的原始信号 (5.0s)。

### Code Changes
- `experiments/scripts/train_aligned_tokenizers.py`: 对齐训练脚本
- `experiments/configs/phase0plus/P0plus_eeg_vqvae_5s.yaml`
- `experiments/configs/phase0plus/P0plus_fnirs_fsq_5s.yaml`

### Configuration
```yaml
EEG (VQ-VAE):
  - Window: 1000 samples = 5.0s @ 200Hz
  - Encoder: [32, 64, 128], stride=2 → T'=125 tokens
  - Codebook: 512, embedding_dim=64

fNIRS (FSQ):
  - Window: 50 samples = 5.0s @ 10Hz
  - Encoder: [32, 64], stride=2 → T'=13 tokens
  - Levels: [8, 8, 8] = 512 codes

Training: 50 epochs, batch_size=64, lr=0.001
```

### Results
```
Run: experiments/runs/P0plus_aligned_20260115_174100/

| Model | Test MSE | Perplexity | Utilization | Dead Codes |
|-------|----------|------------|-------------|------------|
| VQVAE_EEG_5s | **0.0211** | 369.3 | 99.6% | 2 |
| FSQ_fNIRS_5s | **0.0087** | 112.2 | 34.4% | 336 |
```

### Conclusion
- ✅ EEG tokenizer: 优秀重构 (MSE 0.021), 高 codebook 利用率 (99.6%)
- ✅ fNIRS tokenizer: 优秀重构 (MSE 0.009), 但 codebook 利用率较低 (34.4%)
- ✅ 两模态 token 时间对齐 (5.0s window)
- **注意**: fNIRS latent_dim=3 (FSQ levels 数量)，可能过于简单

---

## EXP-003: fNIRS Token Classification (2026-01-15)

### Objective
验证 fNIRS token 表示是否能支持 Motor Imagery 二分类任务。

### Code Changes
- 无额外代码更改（复用 EXP-001 框架）

### Configuration
```yaml
Config: experiments/configs/phase1a/P1A_fnirs_classification.yaml
Tokenizer: FSQ (pre-trained from EXP-000)
Classifier: EndToEndClassifier (Pool + Linear)
Data: 
  - Modality: fNIRS
  - Window: 25 samples (2.5s @ 10Hz)
  - Single channel (channel average)
  - Train: subjects 1-20, Val: 21-25, Test: 26-29
Training:
  - Epochs: 20
  - Batch size: 32
  - Learning rate: 0.001
```

### Results
```
Run directory: experiments/runs/P1A_fnirs_classification_20260115_170237/

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | 51.0% | 47.7% | 50.0% |
| Precision | - | - | 0.0% |
| Recall | - | - | 0.0% |
| F1 | - | - | 0.0% |

Confusion Matrix (Test):
[[120   0]
 [120   0]]
```

### Conclusion
- ⚠️ 分类器完全预测为单一类别，无法区分左右手
- fNIRS 单通道平均同样丢失空间信息
- 需要选择运动区 ROI 通道

---

## EXP-002: EEG Raw Baseline Classification (2026-01-15)

### Objective
建立 Raw EEG 信号分类的基线性能，用于与 Token-based 方法对比。

### Code Changes
- 无额外代码更改

### Configuration
```yaml
Config: experiments/configs/phase1a/P1A_eeg_raw_baseline.yaml
Classifier: RawSignalClassifier (Encoder + Pool + Linear)
Data:
  - Modality: EEG  
  - Window: 512 samples (2.56s @ 200Hz)
  - Single channel (channel average)
  - Train: subjects 1-20, Val: 21-25, Test: 26-29
Training:
  - Epochs: 20
  - Batch size: 32
  - Learning rate: 0.001
```

### Results
```
Run directory: experiments/runs/P1A_eeg_raw_baseline_20260115_170149/

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | 55.9% | 49.3% | 50.8% |
| Precision | - | - | 50.5% |
| Recall | - | - | 82.5% |
| F1 | - | - | 62.7% |

Confusion Matrix (Test):
[[23 97]
 [21 99]]
```

### Conclusion
- ⚠️ Raw 信号基线同样接近 chance level
- 说明问题不在 tokenization，而在**单通道平均丢失空间信息**
- 验证了多通道输入的必要性

---

## EXP-001: EEG Token Classification (2026-01-15)

### Objective
验证 EEG token 表示是否能支持 Motor Imagery 二分类任务，建立端到端 classification pipeline。

### Code Changes
新增文件：
- `src/classifiers/__init__.py` - 模块初始化
- `src/classifiers/simple_classifier.py` - TokenClassifierHead, SimpleTokenClassifier, RawSignalClassifier
- `src/classifiers/end_to_end.py` - EndToEndClassifier, MultiModalClassifier
- `experiments/scripts/train_classifier.py` - 分类器训练脚本
- `experiments/scripts/test_classifier_pipeline.py` - 测试脚本
- `experiments/configs/phase1a/P1A_eeg_classification.yaml`
- `experiments/configs/phase1a/P1A_eeg_raw_baseline.yaml`
- `experiments/configs/phase1a/P1A_fnirs_classification.yaml`

### Configuration
```yaml
Config: experiments/configs/phase1a/P1A_eeg_classification.yaml
Tokenizer: VQ-VAE (pre-trained from EXP-000)
  - Codebook: 512
  - Embedding dim: 64
  - Encoder: [32, 64, 128]
Classifier: EndToEndClassifier
  - Pool: mean
  - Hidden: [128]
  - Freeze tokenizer: True
Data:
  - Modality: EEG
  - Window: 512 samples (2.56s @ 200Hz)
  - Single channel (channel average)
  - Train: subjects 1-20, Val: 21-25, Test: 26-29
Training:
  - Epochs: 20
  - Batch size: 32
  - Learning rate: 0.001
```

### Results
```
Run directory: experiments/runs/P1A_eeg_classification_20260115_165247/

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Accuracy | 53.1% | 48.0% | 50.4% |
| Precision | - | - | 50.4% |
| Recall | - | - | 60.0% |
| F1 | - | - | 54.8% |

Confusion Matrix (Test):
[[49 71]
 [48 72]]
```

### Conclusion
- ⚠️ 分类性能接近 chance level (50%)
- **关键发现**: 单通道平均丢失了 MI 任务的空间信息
- Motor Imagery 的核心特征是 C3/C4 区域的 mu/beta 功率不对称
- **下一步**: 必须使用多通道或 ROI 选择策略

---

## EXP-000: Tokenizer Comparison (2026-01-14)

### Objective
比较 FSQ 和 VQ-VAE tokenizer 在 EEG/fNIRS 数据上的重建质量和 codebook 健康度。

### Code Changes
- `src/tokenizers/fsq.py` - FSQ Tokenizer
- `src/tokenizers/vqvae.py` - VQ-VAE Tokenizer  
- `src/data/eeg_fnirs_dataset.py` - 真实数据加载
- `experiments/scripts/run_tokenizer_comparison.py` - 对比实验脚本

### Configuration
```yaml
Data: EEG+NIRS Single-Trial (TU Berlin)
  - EEG: 200Hz, 512 samples (2.56s), single channel
  - fNIRS: 10Hz, 25 samples (2.5s), single channel
  - Train: subjects 1-20, Val: 21-25, Test: 26-29

Models:
  FSQ_EEG: levels=[8,8,8,8], codebook=4096
  VQVAE_EEG: codebook=512, embedding_dim=64
  FSQ_fNIRS: levels=[8,8,8], codebook=512
  VQVAE_fNIRS: codebook=256, embedding_dim=32

Training: 50 epochs
```

### Results
```
Run directory: experiments/runs/comparison_20260114_183311/

| Experiment | Test MSE | Perplexity | Utilization | Dead Codes |
|------------|----------|------------|-------------|------------|
| FSQ_EEG | 0.0767 | 727.4 | 53.5% | - |
| VQVAE_EEG | **0.0742** | 382.3 | **100%** | 0 |
| FSQ_fNIRS | **0.0538** | 329.2 | 85.7% | 73 |
| VQVAE_fNIRS | 0.0613 | 189.1 | 95.3% | - |
```

### Conclusion
- ✅ **EEG**: VQ-VAE 表现更好（更低 MSE，100% 利用率，无死码）
- ✅ **fNIRS**: FSQ 表现更好（更低 MSE，更高 perplexity）
- 推荐配置：EEG 用 VQ-VAE，fNIRS 用 FSQ
- **下一步**: 验证 token 表示的下游分类性能

---

## Summary: Motor Imagery Classification Experiments (2026-01-30)

### Overall Results Table

| Experiment | Method | Modality | Accuracy | ROC-AUC | Kappa | Status |
|------------|--------|----------|----------|---------|-------|--------|
| EXP-017 | Token + Aug | EEG | **55.83%** | 53.03% | **11.67%** | ✅ Best |
| EXP-015 | Token | EEG | 51.25% | 53.69% | 2.50% | ⚠️ |
| EXP-016 | Raw | EEG | 50.83% | 48.76% | 1.67% | ⚠️ |
| EXP-015 | Token | fNIRS | 49.17% | 51.47% | -1.67% | ⚠️ |
| EXP-016 | Raw | fNIRS | 50.83% | 49.40% | 1.67% | ⚠️ |
| EXP-015 | Token | Multi | 52.08% | 50.25% | 4.17% | ⚠️ |
| EXP-016 | Raw | Multi | 52.50% | 55.60% | 5.00% | ⚠️ |
| Random | - | - | 50.00% | 50.00% | 0.00% | Baseline |

### Key Findings

1. **Data Augmentation Works**: EXP-017证明数据增强可将准确率从51.25%提升到55.83%，Kappa提升4.7倍

2. **Cross-Subject Generalization is the Core Challenge**:
   - Within-subject accuracy: 40%-83% (average 55%)
   - Cross-subject accuracy: ~50% (chance level)
   - Individual differences dominate task-related signals

3. **Token ≈ Raw**: Token-based方法与Raw信号基线表现相当，说明tokenizer未损失判别信息，问题在数据层面

4. **Multimodal Fusion Not Helping**: 当前早期融合策略未带来显著提升

### Recommendations for Future Work

1. **Subject Adaptation**: 使用少量目标被试数据进行微调
2. **Domain Adaptation**: 实现对抗性领域适应减少被试间分布差异
3. **Better Fusion**: 尝试交叉注意力、门控融合等更复杂策略
4. **Hyperparameter Search**: 系统优化增强参数和模型架构

---

## Template for New Experiments

```markdown
## EXP-XXX: [Title] (YYYY-MM-DD)

### Objective
[实验目的]

### Code Changes
[新增或修改的文件列表]

### Configuration
```yaml
[关键配置参数]
```

### Results
```
Run directory: experiments/runs/[run_name]/

[结果表格或关键指标]
```

### Conclusion
[实验结论和下一步行动]
```
