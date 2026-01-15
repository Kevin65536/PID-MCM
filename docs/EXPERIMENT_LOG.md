# Experiment Log

> 实验记录文档，按时间倒序记录每次实验的配置、结果和结论。

---

## Experiment Index

| Date | ID | Phase | Description | Status |
|------|-----|-------|-------------|--------|
| 2026-01-15 | EXP-003 | P1A | fNIRS Token Classification (single-channel) | ⚠️ Chance level |
| 2026-01-15 | EXP-002 | P1A | EEG Raw Baseline Classification | ⚠️ Chance level |
| 2026-01-15 | EXP-001 | P1A | EEG Token Classification (single-channel) | ⚠️ Chance level |
| 2026-01-14 | EXP-000 | P0 | Tokenizer Comparison (FSQ vs VQ-VAE) | ✅ Complete |

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
