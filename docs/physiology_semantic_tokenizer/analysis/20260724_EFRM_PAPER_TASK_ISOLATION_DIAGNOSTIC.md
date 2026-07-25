# EFRM 论文任务隔离快速诊断

_日期：2026-07-24；结论范围：public train/validation；protected test 未开启。_

## 结论

在只保留 EFRM 论文实际用于 paired CLIP 的
`eeg_fnirs_single_trial:motor_imagery` 任务后，**三目标 EFRM 仍未实现
held-out subject 上的 exact-pair CLIP 泛化**。

- 三目标主 run 可以拟合训练配对：最终训练 CLIP 相对精确机会基线改善
  `-0.05477`。
- 验证 CLIP 的最好值仍是机会水平，最终相对机会基线反而为
  `+0.00233`。
- 最佳 CLIP checkpoint 和最终 checkpoint 在全部 240 个 public-validation
  配对上的双向 Top-1、MRR、AUC 和 permutation evidence 均不支持
  exact-pair alignment。
- CLIP-only capacity control 出现过弱的总体正负分离，但信号只存在于
  cross-subject negatives；同被试、同 record 内均为机会水平，同时两模态
  embedding 退化到近一维。因此这不是同步窗口 identity 的泛化。

该结果是一个**任务隔离后的负向快速诊断**，不是对论文全部结果的数值反驳。

## 1. 论文条件核对

论文 Table 2 和 paired-dataset 描述表明，paired EEG-fNIRS 预训练语料只来自
Shin 等人的单试次数据集：

- 任务：motor imagery；
- 被试：29；
- paired recording time：15.5 h；
- EEG/fNIRS 同步采集。

论文没有把本项目 `simultaneous_eeg_nirs` 数据集中的 WG、n-back 和 DSR
三个任务共同用于 paired CLIP。上一轮四数据集、多任务训练因此不是对论文
paired-domain 选择的隔离复测。

## 2. 诊断设计

### 2.1 单变量隔离

主诊断与上一轮保持以下条件不变：

- 两个独立 ViT-base MAE，`221,459,034` 参数；
- encoder `D=768, depth=12, heads=12`；
- decoder `D=512, depth=8, heads=16`；
- mask ratio `0.5`；
- batch size `32`；
- AdamW、LR `1e-4`、cosine schedule；
- source-faithful fixed `0.1 × cosine` logits；
- EEG/fNIRS patch 物理尺度 `0.25 s / 2 s`；
- EEG/fNIRS/CLIP 三项 loss 权重均为 1。

只改变 paired corpus：

```text
dataset = eeg_fnirs_single_trial
task namespace = eeg_fnirs_single_trial:motor_imagery
```

### 2.2 Public boundary

| Split | Pairs | Subjects | Records |
| --- | ---: | ---: | ---: |
| Train | 1,260 | 21 | 63 |
| Validation | 240 | 4 | 12 |
| Total admitted task | 1,740 | 29 | 87 |

Train/validation subject 和 window 均无重叠。4 个 reserved/protected subjects
未读取。

### 2.3 两条 run

1. `20260724_efrm_paper_mi_diag_v1`：论文式三目标训练；
2. `20260724_efrm_paper_mi_clip_only_v1`：仅用于判断架构 capacity 的
   CLIP-only control，两个 reconstruction 权重设为 0。

两条 run 都从零初始化，固定 seed 42，完整运行 30 epochs。每个 epoch
包含 40 个 train batches 和 8 个 validation batches；每条 run 约 18 分钟。

## 3. 三目标主结果

训练和验证的精确随机 CE 基线分别为：

```text
train:      3.45639467  (39×32 + 1×12)
validation: 3.41952609  ( 7×32 + 1×16)
```

| 指标 | Epoch 1 | Epoch 30 | 判断 |
| --- | ---: | ---: | --- |
| Train CLIP | 3.45639392 | 3.40162662 | 训练配对可拟合 |
| Train CLIP − chance | `-0.00000075` | `-0.05476805` | 明确下降 |
| Validation CLIP | 3.41952868 | 3.42185434 | 未泛化 |
| Validation CLIP − chance | `+0.00000258` | `+0.00232825` | 最终恶化 |

验证 CLIP 最低点位于 epoch 3，但相对机会基线仍为
`+0.00000117`，不是改善。

### 3.1 全 public-validation exact-pair 证据

| 指标 | Chance | Best CLIP epoch 3 | Latest epoch 30 |
| --- | ---: | ---: | ---: |
| EEG→fNIRS Top-1 | 0.00417 | 0.00417 | 0.00417 |
| fNIRS→EEG Top-1 | 0.00417 | 0 | 0 |
| EEG→fNIRS MRR | 0.02525 | 0.02529 | 0.02337 |
| fNIRS→EEG MRR | 0.02525 | 0.02248 | 0.02057 |
| Positive-vs-negative AUC | 0.5 | 0.49928 | 0.49326 |
| Permutation p, one-sided | — | 0.71633 | 0.61004 |
| Positive − negative cosine | >0 | `-0.000005` | `-0.010865` |
| Positive − hardest negative | >0 | `-0.04935` | `-0.92276` |

主 run 的 best 和 latest 均不支持 exact-pair 泛化。

## 4. CLIP-only capacity control

CLIP-only 的最佳 validation loss 位于 epoch 22：

```text
best validation CLIP − exact chance = -0.00544080
```

表面上存在弱总体分离：

| 指标 | Best CLIP epoch 22 | Latest epoch 30 |
| --- | ---: | ---: |
| EEG→fNIRS Top-1 | 0.00417 | 0.00833 |
| fNIRS→EEG Top-1 | 0.00417 | 0 |
| Positive-vs-negative AUC | 0.52462 | 0.52578 |
| Permutation p | 0.01600 | 0.27367 |
| Positive − negative cosine | 0.07526 | 0.02853 |
| Positive − hardest negative | -0.55634 | -0.75993 |
| EEG effective rank | 1.189 | 1.180 |
| fNIRS effective rank | 1.273 | 1.085 |
| EEG first-axis energy | 96.55% | 96.75% |
| fNIRS first-axis energy | 94.51% | 98.71% |

这个信号不满足 exact-pair 泛化：

| Negative stratum, best epoch 22 | Positive-vs-negative AUC |
| --- | ---: |
| Same subject | 0.49688 |
| Same record | 0.50114 |
| Cross subject | 0.53371 |
| Same condition | 0.52467 |
| Different condition | 0.52457 |

因此总体 AUC 和 permutation 显著性主要来自 subject-level 粗分组。控制在
同一被试或同一 record 后，正配对不可区分；双向全局 Top-1 也恰好等于
`1/240` 机会值。近一维表示进一步说明模型利用了低秩 nuisance geometry，
而不是学习稳定的同步窗口 identity。

## 5. 判定

### 主问题

> 在论文挑选的 motor-imagery paired task 上，EFRM 三目标架构是否能实现
> CLIP exact-pair alignment 的 held-out 泛化？

本次快速诊断的答案是：**没有证据支持；结果仍为失败。**

任务隔离让训练 CLIP 比四数据集 run 更早下降，但没有让验证同步改善。
这说明上一轮失败不能仅归因于混合了多个数据集或任务。

### Capacity control

EFRM encoder 在 CLIP-only 下可以形成弱跨被试几何，但该几何：

- 不支持同被试/同 record 内的 pair identity；
- exact retrieval 处于机会水平；
- 不稳定，末 checkpoint 的 permutation p 不显著；
- 伴随接近 rank-one 的双模态表示。

因此不能把它解释为 CLIP 对齐泛化的正结果。

## 6. 结论边界

本诊断仍不是论文数值复现：

- 论文还有 868 h EEG-only 和约 364 h fNIRS-only 语料供 MAE reconstruction；
  本诊断只使用 paired motor-imagery 数据同时供应三项目标；
- 本项目使用 200/10 Hz、真实可变通道且不复制；论文使用 128/16 Hz，并把
  EEG/fNIRS 固定到 24/64 个位置；
- 本诊断为 30 epochs、1,200 optimizer steps；论文报告约 1.5M iterations；
- 论文没有报告 subject-held-out pretraining retrieval、CLIP curve、AUC 或
  collapse geometry；
- 本诊断只有一个 seed 和 4 个 public-validation subjects。

所以可成立的表述是：

> 在本项目保持上一轮处理/架构不变、仅隔离论文 paired motor-imagery 任务的
> 快速 public-development 诊断中，EFRM 未实现 exact-pair CLIP 泛化；即使
> 去掉重构竞争，留下的也只是低秩 subject-level 分组，而非配对 identity。

不能据此否定论文的下游 few-shot accuracy，也不能推断 EEG-fNIRS 生理耦合
不存在。

## 7. 证据与复现入口

- 主配置：
  `comparative_methods/EFRM-PyTorch/configs/diagnostic_paper_motor_imagery.yaml`
- CLIP-only control：
  `comparative_methods/EFRM-PyTorch/configs/diagnostic_paper_motor_imagery_clip_only.yaml`
- 主 run：
  `comparative_methods/EFRM-PyTorch/runs/pretraining/20260724_efrm_paper_mi_diag_v1`
- Control run：
  `comparative_methods/EFRM-PyTorch/runs/pretraining/20260724_efrm_paper_mi_clip_only_v1`
- 全验证 checkpoint 评估：
  `comparative_methods/EFRM-PyTorch/evaluate_pretrain_checkpoint.py`
- 主 best/latest metrics：
  `20260724_efrm_paper_mi_diag_v1/analysis/checkpoints/`
- Control best/latest metrics：
  `20260724_efrm_paper_mi_clip_only_v1/analysis/checkpoints/`
- 论文：
  `docs/paper_pdf/Jung和An - 2025 - EFRM A Multimodal EEG–fNIRS Representation-learning Model for few-shot brain-signal classification.pdf`
