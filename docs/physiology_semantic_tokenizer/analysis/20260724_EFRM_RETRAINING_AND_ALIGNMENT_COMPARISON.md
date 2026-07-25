# EFRM 复现、重训练与 EEG–fNIRS 对齐分析

_分析日期：2026-07-24；主 run：`20260722_efrm_sync_dev_v5`；范围：public train/validation，protected test 未开启。_

## 结论摘要

1. **工程复现与重训练已完成，但不是论文数值复现。** 当前实现保留了 EFRM 的双 ViT-base MAE、双向 batch contrastive loss、0.5 mask ratio、0.25 s EEG patch、2 s fNIRS patch、batch 32、AdamW 和 cosine schedule。训练在第 84 epoch 正常早停，最佳总验证损失位于第 69 epoch。
2. **重构成功，对齐失败。** 从首个到最终 epoch，验证 EEG/fNIRS 重构损失分别下降 `80.07%/79.08%`；验证 CLIP loss 反而上升 `0.24%`。最终保存 batch 的双向 Top-1 都是 `1/32=3.125%`，正对负余弦差为 `-0.2861`，AUC 为 `0.3852`。
3. **训练 pattern 是“先只学重构，后拟合训练配对，但不泛化”。** 前 45 epoch 的训练 CLIP 与精确随机基线几乎相同；第 46–49 epoch 突然下降，最终比训练随机基线低 `0.0840`，但验证 CLIP 没有同步改善，最终验证−训练 CLIP gap 为 `0.0649`。
4. **最终对齐空间发生近一维双极塌缩。** EEG/fNIRS centered effective rank 分别为 `1.025/1.014`，第一轴解释 `99.73%/99.86%` 方差。最佳 checkpoint 也出现同类现象，因此不是只由最后 15 个 patience epoch 造成。
5. **论文没有报告预训练 loss、检索率或正负 pair 分离，无法对这些指标做数值复现比较。** 论文的主要数值证据是不同下游数据集上的 few-shot accuracy/F1；本服务器当前只有完成的 EFRM 预训练 run，没有 EFRM 下游 linear-probe/full-finetune 结果目录，所以还不能声称复现了论文 Table 3。
6. **EFRM CLIP 与本项目对齐回答的是不同问题。** EFRM 判断“同一 8 s 采集窗是不是 batch 对角线配对”；本项目目标判断“EEG 历史在控制 fNIRS 自身历史、边际和 nuisance 后，是否增加对未来 fNIRS token 分布的预测信息”。后者更接近延迟神经血管耦合，但当前只是已注册的 E7–E9 机制；E2 没有 admit semantic row，E7–E9 尚未形成实证结果，因此不能把设计优势写成性能优势。

## 1. 可比性边界

### 1.1 高保真复现部分

| 项目 | 论文/官方实现 | 当前同步数据实现 | 判断 |
| --- | --- | --- | --- |
| 主体结构 | 两个独立 ViT-base MAE | 两个独立 variable-channel ViT-base MAE | 高度一致 |
| Encoder | `D=768`, 12 blocks, 12 heads | 相同 | 一致 |
| Decoder | `D=512`, 8 blocks, 16 heads | 相同 | 一致 |
| Mask ratio | 0.5 | 0.5 | 一致 |
| EEG patch | 128 Hz 下 32 点，即 0.25 s | 200 Hz 下 50 点，即 0.25 s | 物理尺度一致 |
| fNIRS patch | 16 Hz 下 32 点，即 2 s | 10 Hz 下 20 点，即 2 s | 物理尺度一致 |
| 对齐目标 | pooled embedding 双向交叉熵 | 相同 | 一致 |
| logit scale | 官方源码乘以 `0.1` | 乘以 `0.1` | 源码忠实 |
| 优化 | LR `1e-4`, Adam `(0.9,0.95)`, cosine, batch 32 | 相同，并加 weight decay/早停审计 | 核心一致 |
| 可训练参数 | 官方 `221,466,720` | `221,459,034` | 相差 `7,686`，约 `0.0035%` |

### 1.2 不可直接数值复现部分

| 维度 | 论文 | 当前 run | 后果 |
| --- | --- | --- | --- |
| 预训练语料 | 约 1,247.3 h、918 人；EEG-only 868 h、fNIRS-only 363.8 h、paired 15.5 h | 四个项目内同步数据集；train/validation 共 17,734 个 8 s 样本入口 | 数据分布完全不同 |
| 目标供数 | 单模态数据供重构，仅 paired 数据供 CLIP | 每个样本同时供双重构和 CLIP | CLIP 梯度占比和重复曝光不同 |
| 通道策略 | 裁剪或重复至 EEG 24、fNIRS 64 | 保留真实测量通道，不复制 | 更符合本项目数据契约，但不是论文输入 |
| 训练长度 | 1,500,000 iterations，约 8 天，A6000 | 63,168 optimizer steps，84 epochs，完整 epoch 累计约 24.5 h，4090 | 当前更新数约为论文的 `4.21%` |
| 停止规则 | 固定 iterations | public validation 总损失早停 | checkpoint 选择逻辑不同 |
| 下游任务 | 论文自有 EEG/fNIRS/paired 三类数据 | 计划七任务共享协议 | 任务、split、shot 均不等价 |

因此，当前最准确的名称是 **source-faithful objective retraining on synchronized in-domain data**，不是 paper-number reproduction。

## 2. 完整训练指标

### 2.1 Run 完整性

| 指标 | 结果 |
| --- | ---: |
| 状态 | `completed`，exit code 0 |
| 完整 epoch | 84 / 100 |
| 最佳 epoch | 69 |
| 最佳验证总损失 | 3.889639 |
| optimizer steps | 63,168 |
| 最终 patience | 15 |
| 最佳/末 checkpoint | 不同 |
| protected test | 未开启 |
| GPU peak allocated/reserved | 15.84 / 18.16 GiB |
| epoch 平均耗时 | 17.51 min |

### 2.2 首末变化

| 指标 | Epoch 1 | Epoch 84 | 相对变化 | 解释 |
| --- | ---: | ---: | ---: | --- |
| Train total | 6.03398 | 3.69371 | −38.78% | 总目标收敛 |
| Validation total | 5.70879 | 3.90041 | −31.68% | public validation 改善 |
| Validation EEG reconstruction | 1.38868 | 0.27683 | −80.07% | EEG MAE 学习明确 |
| Validation fNIRS reconstruction | 0.89109 | 0.18640 | −79.08% | fNIRS MAE 学习明确 |
| Train CLIP | 3.45629 | 3.37228 | −2.43% | 后期仅训练集改善 |
| Validation CLIP | 3.42902 | 3.43718 | +0.24% | 无泛化改善 |
| 最终 total generalization gap | — | 0.20669 | — | 后期过拟合 |
| 最终 CLIP generalization gap | — | 0.06491 | — | 对齐分支过拟合更明确 |

梯度裁剪前 norm 的 median/p95 为 `1.197/5.444`，`5.65%` 的 steps 超过阈值 5；最大值为 `1116.23`（epoch 29）。这说明优化总体可运行，但存在少量大梯度事件。它不是对齐失败的充分解释，因为核心 pattern 是系统性的训练/验证分叉。

## 3. EEG–fNIRS 对齐 pattern 的训练阶段变化

### 阶段 A：Epoch 1–45，重构主导，对齐未启动

- 训练 CLIP 相对每个实际 batch 大小的精确随机交叉熵基线仅改善约 `10^-6–10^-4`。
- 同期 EEG/fNIRS 重构快速下降，总验证损失主要由重构改善驱动。
- 解释：固定乘数 `0.1` 将所有 cosine logits 压在 `[-0.1,0.1]`。batch 32 下随机 CE 为 `3.465736`；即使理想化地令所有正 pair cosine 为 `+1`、负 pair 为 `−1`，CE 下界也只有 `3.272631`，整个可见动态范围仅 `0.193105`。

### 阶段 B：Epoch 46–49，训练对齐突变

| Epoch | Train CLIP | 相对精确随机基线 | Validation CLIP |
| ---: | ---: | ---: | ---: |
| 45 | 3.455744 | −0.000287 | 3.427412 |
| 46 | 3.453823 | −0.002207 | 3.425628 |
| 47 | 3.435091 | −0.020939 | 3.426739 |
| 48 | 3.418999 | −0.037031 | 3.426837 |
| 49 | 3.409673 | −0.046652 | 3.430642 |

重构梯度逐渐减弱后，CLIP 分支开始改变 encoder；但验证没有同步收益。这个时间点支持“模型开始拟合训练 pair/record/source 结构”的解释，而不支持“共享生理空间开始形成”。

### 阶段 C：Epoch 50–84，训练 CLIP 继续改善，验证恶化

- Epoch 69 达到最佳**总**验证损失，但该选择主要受重构项驱动。
- Train CLIP 最终相对随机基线改善 `0.0840`；Validation CLIP 从首 epoch 到末 epoch 上升。
- Train/validation CLIP 的 epoch 相关方向相反：训练随 epoch 下降，验证总体随 epoch 上升。
- 早停后 15 epochs 没有修复该 gap。

最符合证据的 pattern 是：

```mermaid
flowchart LR
    recon["前期：双 MAE 快速学习"] --> weak["CLIP 受 0.1 logit multiplier 抑制"]
    weak --> onset["Epoch 46 左右：CLIP 开始主导剩余可塑性"]
    onset --> trainfit["训练 pair loss 下降"]
    onset --> valflat["验证 pair loss 不降"]
    trainfit --> rank1["近一维双极表示"]
    valflat --> chance["验证 exact-pair retrieval 机会水平"]
```

这是根据时间顺序和几何结果作出的机制推断，不是仅凭曲线即可证明的唯一因果解释。

## 4. 最佳与最终 checkpoint 的对齐复核

为避免只看早停末模型，在同一个最终 public-validation batch（32 pairs，Visual Cognitive Motivation，S06/S07）上重新评估了 epoch 69 best 和 epoch 84 latest。

| 指标 | Chance | Best epoch 69 | Latest epoch 84 |
| --- | ---: | ---: | ---: |
| EEG→fNIRS Top-1 | 0.03125 | 0.03125 | 0.03125 |
| fNIRS→EEG Top-1 | 0.03125 | 0.03125 | 0.03125 |
| EEG→fNIRS MRR | 0.12683 | 0.08931 | 0.09781 |
| fNIRS→EEG MRR | 0.12683 | 0.10373 | 0.10629 |
| EEG→fNIRS mean rank | 16.50 | 20.25 | 20.72 |
| fNIRS→EEG mean rank | 16.50 | 18.72 | 19.50 |
| Positive−negative cosine | >0 | −0.20884 | −0.28613 |
| Positive-vs-negative AUC | 0.50 | 0.41737 | 0.38524 |
| Hard-negative margin | >0 | −1.10703 | −1.24646 |
| EEG effective rank | — | 1.152 | 1.025 |
| fNIRS effective rank | — | 1.046 | 1.014 |

最终 checkpoint 的附加证据：

- positive cosine mean `−0.24946`，negative mean `+0.03667`；
- identity-pair permutation one-sided `p=0.9753`；
- EEG/fNIRS 第一轴能量 `99.73%/99.86%`；
- EEG 第一轴在该 batch 中几乎是 subject 分隔轴：S06 有 15/16 位于一侧，S07 有 16/16 位于另一侧；
- fNIRS 第一轴形成另一个粗粒度双极分组，与 EEG 配对第一轴相关仅 `r=0.332`，sign agreement `62.5%`；
- same-subject negative cosine 为 `−0.2660`，cross-subject cosine 反而为 `+0.3204`，说明两个 modality 的粗粒度轴没有形成一致的 subject 或 exact-window 几何。

这不是“所有 embedding 都变成同一个向量”，而是**两个 modality 各自退化到近一维、但两条轴没有按真实 pair 一致对应的双极解**。

证据限制：上述 retrieval/几何来自一个导出的 32-pair batch，不能外推为四数据集总体 retrieval；但 280 个 validation batches 聚合得到的 CLIP loss 同样没有改善，且 best/latest 两个 checkpoint 的 batch 级结论一致，因此足以形成严重的 public-development failure warning。

## 5. 与论文结果对比

### 5.1 论文下游结果

论文 Table 3 的最大 shot 设置如下：

| 下游数据 | EFRM 论文结果 | 主要论文基线 |
| --- | ---: | --- |
| EEG binary Alertness/Sleep，8-shot | `0.74 ± 0.11` accuracy | MAE `0.70±0.12`; SSCL `0.71±0.11`; Conformer 800-shot `0.77±0.08` |
| EEG ternary，64-shot | `0.65 ± 0.11` | MAE `0.64±0.14`; SSCL `0.53±0.09`; Conformer 6400-shot `0.76±0.06` |
| fNIRS Mental Arithmetic/Rest，8-shot | `0.94 ± 0.06` | MAE `0.75±0.10`; SSCL `0.69±0.22`; fNIRS-T 24-shot `0.88±0.09` |
| Paired EEG–fNIRS Alertness/Sleep，6-shot | `0.80 ± 0.10` | SleepFM `0.71±0.10`; EFNet 600-shot `0.86±0.10`; BimodalNet `0.85±0.08` |

论文用 Figure 3 报告 F1 boxplot，但没有提供可直接抄录的完整数值表。论文还报告 EEG band 变宽时 embedding 间估计 mutual information 从 `0.486` 增至 `0.769`，并将其解释为 shared domain 增加。

### 5.2 当前服务器能与不能比较的内容

| 指标族 | 论文是否报告 | 当前 run 是否有 | 可比性 |
| --- | --- | --- | --- |
| Pretraining reconstruction loss | 否 | 有 | 不能数值对比 |
| Pretraining CLIP loss 曲线 | 否 | 有 | 不能数值对比 |
| Cross-modal retrieval/MRR/AUC | 否 | 有 | 只能作为新增诊断 |
| Embedding effective rank/collapse | 否 | 有 | 只能作为新增诊断 |
| Embedding MI | 有，但 estimator 细节不足 | 未按论文流程计算 | 不能直接对比 |
| Few-shot accuracy/F1 | 有 | 当前无下游 run | 尚未复现 |

因此不能用当前良好的重构 loss 代替论文 Table 3，也不能因为当前 CLIP 失败就直接否定论文的 few-shot accuracy。反过来，论文的下游 accuracy 也不能证明其 exact-window CLIP 真正学到了延迟 EEG–fNIRS 生理耦合。

论文 shared-domain 论证还有一个重要混杂：通过改变 EEG bandpass 宽度来改变所谓 shared information，同时也改变了 EEG 自身可用信息和优化难度。`I(E;F)` 与下游准确率同向并不能单独识别“共享域增加导致性能提升”；论文也没有报告 held-out retrieval、fNIRS-history baseline、lag profile 或 time-shift null。

## 6. EFRM CLIP 对齐与本项目对齐的本质区别

| 维度 | EFRM CLIP | 本项目目标对齐 |
| --- | --- | --- |
| 核心问题 | EEG 与 fNIRS 是否来自同一 8 s window | EEG 历史是否在 fNIRS 自历史之外增加对未来 fNIRS 的预测信息 |
| 方向 | 对称 EEG↔fNIRS retrieval | 非对称 EEG history→future fNIRS |
| 时间 | 同窗、零显式 lag | 初始 lag `0..16 s` |
| 表示 | 两个 pooled 768-D 连续向量 | 独立 EEG/fNIRS hard ID、posterior、prototype、residual 序列 |
| 正例 | batch 对角线 exact instance | 同一 causal history 下的未来 fNIRS 分布 |
| 负例/基线 | batch 其余样本全作 negatives | matched `q0`: fNIRS history+nuisance；`q1`: q0+EEG history |
| 主要统计量 | symmetric cross-entropy/retrieval | held-out incremental proper likelihood `q1-q0` |
| 生理延迟 | 未建模 | 显式 lag/horizon profile |
| 私有信息 | 被共同空间压力压缩 | modality-private residual 明确保留 |
| 混杂控制 | 无显式 history/marginal/source 控制 | subject/source/task prevalence、history、marginal controls |
| Null | batch negatives | EEG shuffle、circular shift、频率保持 permutation、spatial/null family |
| 梯度边界 | 两个 encoder 可共同适应 | preservation 阶段只让梯度进入 EEG semantic tokenizer；fNIRS/teacher/baseline detached |
| 独立证书 | 训练 head 本身就是结果 | 丢弃训练 shaper，冻结后用 fresh/cross-fitted evaluator |
| 可解释含义 | “配对共现的连续几何” | “有控制的延迟条件分布”，仍不自动等于因果 NVC |

一句话概括：

> **EFRM 优化的是 co-occurrence identity；本项目要估计的是 history-controlled directional information。**

EFRM 的优势是简单、可扩展到通用连续表示和下游 few-shot；其代价是把同一 task/subject 的相似窗口当作 false negatives，并把跨模态关系压成无方向、无延迟的一个点积。对 EEG–fNIRS 这种慢血流响应与快电活动的关系，exact-window identity 不是最自然的生理 estimand。

本项目设计更符合 delayed coupling 的问题结构，但更难训练和验证。当前状态必须明确：

- E2 的 T1−T0 三个 matched seeds 为 `−0.0271/−0.0413/+0.0065`；
- T2−T0 为 `−0.0343/−0.0560/−0.0324`；
- bootstrap mean 分别为 `−0.0326`（95% CI `[-0.0770,0.0042]`）和 `−0.0575`（`[-0.1107,-0.0147]`）；
- 决策是 `no_semantic_row_admitted_retain_T0`；
- E7 preservation、E8 discovery、E9 independent certificate 尚未通过。

所以当前可成立的表述是“本项目定义了比 CLIP 更有生理针对性的对齐 estimand 和证据链”，不能表述为“我们的对齐已经优于 EFRM”。

## 7. 最终判断与下一步

### 对当前 EFRM run 的判定

- **架构/目标复现：通过。**
- **同步数据重训练完成：通过。**
- **MAE 重构学习：通过。**
- **public-development exact-pair CLIP 泛化：失败警告。**
- **论文 few-shot 数值复现：未完成，因为没有下游结果。**
- **生理耦合结论：不支持，也不能由当前失败反推“不存在耦合”。**

### 最小必要后续

1. 用 best epoch 69 checkpoint 在全部 public validation 上做 dataset/subject-stratified retrieval，而不是只保存最后一个 batch。
2. 从零训练一个明确命名的 temperature ablation（learned logit scale 或除以 `τ≈0.07`），同时保留当前 `×0.1` source-faithful baseline。
3. 增加 false-negative sensitivity：同 subject/condition/record negatives 分层报告，并加入 lagged-positive sensitivity。
4. 完成 EFRM 的 public-development downstream linear probe/full fine-tune 后，才与论文 Table 3 或本项目下游结果做表格比较；由于 task/split 不同，应分为“paper-reference”和“shared-protocol”两张表。
5. 本项目不应跳过 E2/G2/G3 直接进入漂亮的 coupling heatmap；E7–E9 必须继续按 preserve–discover–certify 顺序执行。

## 证据路径

- EFRM 最终自动审计：`comparative_methods/EFRM-PyTorch/runs/pretraining/20260722_efrm_sync_dev_v5/analysis/REPORT.md`
- Epoch 指标表：`comparative_methods/EFRM-PyTorch/runs/pretraining/20260722_efrm_sync_dev_v5/analysis/tables/epoch_metrics.csv`
- 最终 alignment evidence：`comparative_methods/EFRM-PyTorch/runs/pretraining/20260722_efrm_sync_dev_v5/figure_data/clip_alignment_evidence.npz`
- EFRM 配置与 manifest：`comparative_methods/EFRM-PyTorch/runs/pretraining/20260722_efrm_sync_dev_v5/resolved_config.yaml`、`manifest.json`
- EFRM 论文：`docs/paper_pdf/Jung和An - 2025 - EFRM A Multimodal EEG–fNIRS Representation-learning Model for few-shot brain-signal classification.pdf`
- 本项目理论与对齐定义：`docs/physiology_semantic_tokenizer/03_THEORETICAL_FOUNDATIONS.md`
- 本项目 E7–E9 方案：`docs/physiology_semantic_tokenizer/05_EXPERIMENT_DESIGN.md`
- 当前 E2 决策：`experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/decision/summary.md`
