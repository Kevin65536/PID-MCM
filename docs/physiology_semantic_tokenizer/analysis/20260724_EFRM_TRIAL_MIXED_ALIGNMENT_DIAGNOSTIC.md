# EFRM 混合被试 trial-split CLIP 对齐诊断

_日期：2026-07-24；结论范围：25 个 public subjects；protected test 未开启。_

## 结论

放开跨被试泛化要求、让同一批 public subjects 同时出现在训练和验证中后，
结果不再是完全的零信号，但需要区分论文式三目标模型和 CLIP-only
capacity control：

1. **论文式三目标 EFRM 仍未给出可信的 exact-pair 泛化证据。**
   最佳 checkpoint 的全验证 AUC 为 0.522，置换检验
   `p=0.0911`；在同 record、同 condition 的二选一检验中，双向 Top-1
   为 51.7%/53.3%，组内配对检验 `p=0.1431`。
2. **CLIP-only 可以学到一个很弱的 within-subject/within-record 配对信号。**
   最佳 checkpoint 的全验证 AUC 为 0.571，整体置换
   `p<1e-4`；在最严格的同 record、同 condition 二选一检验中，双向
   Top-1 为 53.3%/55.7%，配对优势 0.0899，组内检验
   `p=0.0257`。latest checkpoint 仍有相近结果。
3. **这个部分正结果不能解释为有用的全局 CLIP 检索泛化。**
   CLIP-only 在 300 个候选上的 EEG→fNIRS/fNIRS→EEG Top-1 仅为
   0%/1%；两个 embedding 的有效秩只有 1.11/1.23，第一主轴能量达到
   98.3%/95.2%。大部分总体分离来自跨被试、跨 record 的粗粒度结构。

最准确的判定是：

> 在极宽松的混合被试 trial split 下，EFRM encoder 在去掉重构竞争后具有
> 学习微弱 trial-level 对应的容量；但论文式三目标训练仍未通过，而
> CLIP-only 的信号强度、检索效果和表示坍缩都不足以支持“实现了可靠
> CLIP 对齐泛化”。

## 1. 诊断设计

### 1.1 唯一改变

继续只使用论文 paired CLIP 所选的：

```text
dataset = eeg_fnirs_single_trial
task = motor_imagery
conditions = LMI, RMI
```

模型、采样率、patch 尺度、固定 `0.1 × cosine` logits、batch size、优化器、
30 epochs 和 seed 42 均与上一轮任务隔离诊断保持一致。模型为完整的两个
ViT-base MAE，共 `221,459,034` 参数。

唯一核心改变是 split：

```text
subject-disjoint public split
        ↓
within-subject, record-condition-stratified trial split
```

### 1.2 Split 边界

每个 `(subject, record, condition)` stratum 恰好包含 10 个 trials，按固定
hash 和 seed 42 划分为 8 train + 2 validation。

| Split | Trials | Subjects | Records | LMI / RMI |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,200 | 25 | 75 | 600 / 600 |
| Validation | 300 | 25 | 75 | 150 / 150 |

- subject overlap：25；
- record overlap：75；
- exact trial overlap：0；
- 每个 subject：48 train + 12 validation；
- 4 个 nonpublic/protected subjects
  (`subject_11/15/19/23`) 未打开。

因此“所有 trial”在这里严格指全部 **public-admitted** trials，而不是打开
项目预留的 protected subjects。

### 1.3 重要的宽松边界

本数据缓存采用 full-record normalization；同一 record 的 train/validation
trials 共享该归一化状态。这正是本轮有意放宽的条件之一，但意味着：

- 本结果只能称为 within-subject/within-record trial generalization；
- 不能外推为新被试泛化；
- 不能把共享 record-level 状态带来的信号全部解释为跨模态生理语义。

Boundary hash：
`0a7cb568292f6331b58308dbab2f17cf712e52f696274ec8294e08d3927f1231`。

## 2. 训练曲线

批内随机 CE 的精确基线为：

```text
train:      3.45649394  (37×32 + 1×16)
validation: 3.42650273  ( 9×32 + 1×12)
```

### 2.1 论文式三目标 EFRM

| 点位 | Train CLIP | Train − chance | Validation CLIP | Validation − chance |
| --- | ---: | ---: | ---: | ---: |
| Epoch 1 | 3.456496 | +0.000002 | 3.426504 | +0.000001 |
| Best alignment, epoch 25 | 3.413607 | -0.042887 | 3.423121 | -0.003381 |
| Latest, epoch 30 | 3.405633 | -0.050861 | 3.424803 | -0.001700 |

混合被试后验证 loss 的确低于随机基线，但改善远小于训练端，且 epoch 25
之后回退。loss 下降本身需要由完整检索矩阵验证。

### 2.2 CLIP-only capacity control

| 点位 | Train CLIP | Train − chance | Validation CLIP | Validation − chance |
| --- | ---: | ---: | ---: | ---: |
| Epoch 1 | 3.456505 | +0.000011 | 3.426502 | -0.000000 |
| Best alignment, epoch 27 | 3.442217 | -0.014277 | 3.410327 | -0.016176 |
| Latest, epoch 30 | 3.440740 | -0.015754 | 3.411303 | -0.015200 |

CLIP-only 的验证改善在末期保持，说明它不是单个 epoch 的偶然低点。但验证
改善大于训练改善也提示 batch composition 和共享 nuisance structure 对
该 loss 有明显影响。

## 3. 完整 300-trial 验证矩阵

### 3.1 三目标模型

| 指标 | Chance | Best epoch 25 | Latest epoch 30 |
| --- | ---: | ---: | ---: |
| EEG→fNIRS Top-1 | 0.00333 | 0 | 0.00333 |
| fNIRS→EEG Top-1 | 0.00333 | 0.00333 | 0.00333 |
| EEG→fNIRS MRR | 0.02094 | 0.02287 | 0.02327 |
| fNIRS→EEG MRR | 0.02094 | 0.02286 | 0.01932 |
| Positive-vs-all-negative AUC | 0.5 | 0.52207 | 0.51537 |
| Identity permutation p | — | 0.09109 | 0.16948 |
| Positive − negative cosine | 0 | 0.04496 | 0.03314 |
| EEG / fNIRS effective rank | — | 3.34 / 3.26 | 3.34 / 3.07 |

最佳与最终 checkpoint 的全矩阵置换检验均不显著，全局检索也在机会水平。

### 3.2 CLIP-only

| 指标 | Chance | Best epoch 27 | Latest epoch 30 |
| --- | ---: | ---: | ---: |
| EEG→fNIRS Top-1 | 0.00333 | 0 | 0.00667 |
| fNIRS→EEG Top-1 | 0.00333 | 0.01000 | 0.00667 |
| EEG→fNIRS MRR | 0.02094 | 0.02484 | 0.02807 |
| fNIRS→EEG MRR | 0.02094 | 0.03673 | 0.03127 |
| Positive-vs-all-negative AUC | 0.5 | 0.57070 | 0.56971 |
| Identity permutation p | — | <0.0001 | <0.0001 |
| Positive − negative cosine | 0 | 0.20266 | 0.19957 |
| EEG / fNIRS effective rank | — | 1.106 / 1.228 | 1.105 / 1.208 |
| EEG / fNIRS first-axis energy | — | 98.29% / 95.21% | 98.32% / 95.73% |

总体正负分离显著且在 latest 保留，但 300-way exact retrieval 几乎没有实际
提升，同时表示接近 rank one。

## 4. 信号来源拆解

### 4.1 Negative strata：最佳 checkpoint

| Positive-vs-negative AUC | 三目标 | CLIP-only |
| --- | ---: | ---: |
| All negatives | 0.52207 | 0.57070 |
| Same subject negatives | 0.51672 | 0.52006 |
| Same record negatives | 0.52976 | 0.52452 |
| Same record + same condition negatives | 0.52284 | 0.52877 |
| Cross subject negatives | — | 0.57264 |

CLIP-only 从 all-negative AUC 0.571 降至 same-subject 0.520，说明大部分
容易的分离来自 subject/record 层级。控制这些因素后仍有约 0.52–0.53 的
微弱残余。

### 4.2 逐级收紧候选池：最佳 checkpoint

| 检索范围 | Chance Top-1 | 三目标 EEG→fNIRS / fNIRS→EEG | CLIP-only EEG→fNIRS / fNIRS→EEG |
| --- | ---: | ---: | ---: |
| Full validation，300 candidates | 0.33% | 0% / 0.33% | 0% / 1.00% |
| Within subject，12 candidates | 8.33% | 10.67% / 9.67% | 9.67% / 9.67% |
| Within record，4 candidates | 25% | 27.67% / 28.00% | 27.67% / 28.67% |
| Same record + condition，2 candidates | 50% | 51.67% / 53.33% | 53.33% / 55.67% |

最严格的 150 个二选一组中，额外比较 identity mapping 与组内 swap：

| 指标 | 三目标 best | CLIP-only best | CLIP-only latest |
| --- | ---: | ---: | ---: |
| Identity − swap cosine mean | 0.04719 | 0.08992 | 0.09009 |
| Sign-flip p, one-sided, 100k | 0.14310 | 0.02567 | 0.03013 |

这项检验是本轮最重要的细化：

- 三目标模型未通过；
- CLIP-only 的 exact trial pairing 在完全控制 subject、record 和 condition
  后仍有弱但显著的残余；
- 53–56% 的二选一效果距离可用的跨模态检索仍很远。

## 5. 与 subject-disjoint 结果的关系

上一轮 subject-disjoint 诊断中：

- 三目标模型的 held-out-subject alignment 完全失败；
- CLIP-only 的弱分离只存在于 cross-subject negatives；
- same-subject 和 same-record AUC 均约为 0.5。

本轮放宽边界后，CLIP-only 的 same-record-condition AUC 提高到 0.529，
并在二选一配对检验中达到 `p=0.0257`。因此可以认为：

> EFRM 不是在任何条件下都无法学习 paired EEG-fNIRS；它可以在训练见过
> 相同 subjects 和 records 的条件下记住/利用足以产生微弱 trial 配对优势
> 的结构。但该结构没有转化为 subject-disjoint 泛化，也没有形成健康的
> 高秩全局检索空间。

这解释了为什么只看 CLIP loss 或总体 AUC 容易得到过度乐观的结论。

## 6. 最终判定

如果问题是：

> 放开跨被试条件，混合所有 public-subject trials 后，验证 CLIP loss
> 能否低于随机？

答案是 **可以**，特别是 CLIP-only。

如果问题是：

> 论文式三目标 EFRM 是否因此实现了可信的 exact-pair CLIP 泛化？

答案仍是 **没有**。

如果问题是：

> EFRM 架构本身是否具备任何 within-subject exact-pair 学习能力？

答案是 **有一点证据支持，但只在 CLIP-only、极宽松边界下，效应很弱且
伴随严重表示坍缩**。

基于这三轮递进诊断，不建议继续通过放宽 split、挑选 loss 低点或延长同一
配置训练来证明 CLIP 对齐。若后续仍研究 EFRM，优先级应转向修正目标本身：
可学习温度、去坍缩正则、subject/record-balanced negatives，以及明确区分
instance alignment 与 subject-level nuisance。

## 7. 限制

- 单 seed、30 epochs；
- source-faithful logit multiplier 固定为 0.1；
- 使用 200/10 Hz 和本项目真实通道处理，而非论文的 128/16 Hz 与固定位置；
- 没有论文的 EEG-only/fNIRS-only 大规模 MAE 预训练语料；
- train/validation 共享 subject、record 和 full-record normalization state；
- 论文没有报告可用于对标的 CLIP retrieval、AUC、permutation 或 collapse
  指标。

因此本报告评估的是本项目 EFRM 实现的快速诊断，不否定论文的下游
few-shot classification 结果。

## 8. 证据与复现入口

- 三目标配置：
  `comparative_methods/EFRM-PyTorch/configs/diagnostic_paper_motor_imagery_trial_mixed.yaml`
- CLIP-only 配置：
  `comparative_methods/EFRM-PyTorch/configs/diagnostic_paper_motor_imagery_trial_mixed_clip_only.yaml`
- Split manifest：
  `comparative_methods/EFRM-PyTorch/runs/pretraining/20260724_efrm_paper_mi_trial_mixed_v1/boundary_manifest.json`
- 三目标 run：
  `comparative_methods/EFRM-PyTorch/runs/pretraining/20260724_efrm_paper_mi_trial_mixed_v1`
- CLIP-only run：
  `comparative_methods/EFRM-PyTorch/runs/pretraining/20260724_efrm_paper_mi_trial_mixed_clip_only_v1`
- Checkpoint 全验证评估器：
  `comparative_methods/EFRM-PyTorch/evaluate_pretrain_checkpoint.py`
- 上一轮 subject-disjoint 报告：
  `docs/physiology_semantic_tokenizer/analysis/20260724_EFRM_PAPER_TASK_ISOLATION_DIAGNOSTIC.md`
