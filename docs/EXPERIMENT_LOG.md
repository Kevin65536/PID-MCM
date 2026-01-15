# Experiment Log

> 实验记录文档，按时间倒序记录每次实验的配置、结果和结论。

---

## Experiment Index

| Date | ID | Phase | Description | Status |
|------|-----|-------|-------------|--------|
| 2026-01-15 | EXP-007 | P1A | Dual-Modality Multi-Lead Classification | ⚠️ Chance level |
| 2026-01-15 | EXP-006 | P1A | fNIRS Multi-Lead Classification (5s) | ⚠️ Chance level |
| 2026-01-15 | EXP-005 | P1A | EEG Multi-Lead Classification (5s) | ⚠️ ~54% |
| 2026-01-15 | EXP-004 | P0+ | Time-aligned Tokenizers (5s window) | ✅ Complete |
| 2026-01-15 | EXP-003 | P1A | fNIRS Token Classification (single-channel) | ⚠️ Chance level |
| 2026-01-15 | EXP-002 | P1A | EEG Raw Baseline Classification | ⚠️ Chance level |
| 2026-01-15 | EXP-001 | P1A | EEG Token Classification (single-channel) | ⚠️ Chance level |
| 2026-01-14 | EXP-000 | P0 | Tokenizer Comparison (FSQ vs VQ-VAE) | ✅ Complete |

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
