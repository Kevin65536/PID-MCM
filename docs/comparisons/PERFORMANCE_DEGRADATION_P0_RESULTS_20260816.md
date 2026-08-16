# 对比方法性能劣化：P0 首轮机制分析结果

_版本：v1.0（2026-08-16）；状态：首轮完成、探索性；主协议：_
[`PERFORMANCE_DEGRADATION_ANALYSIS_PLAN_20260816.md`](PERFORMANCE_DEGRADATION_ANALYSIS_PLAN_20260816.md)

## 1. 结论先行

首轮证据不支持把原论文与本项目之间的巨大性能 gap 归结为单一因素。更符合现有结果的解释是：**任务本身的跨被试可迁移信号、评价对象变化、方法原生输入/训练预算被统一协议改变，以及部分创新机制没有转化为实例级或任务相关信息，共同造成了劣化。**

其中，证据最强的五点是：

1. **低结果首先按任务聚集。** 在冻结的完整五方法 × 六分类任务平衡面板上，原始 macro-F1 的描述性平方和有 91.57% 位于任务项、4.51% 位于方法项；按每个单元格的 B0 中心化后，任务、方法和残差分别占 51.78%、25.82% 和 22.40%。Visual 六方法均值低于 B0 约 0.021，而 MA、WG 均稳定高于 B0。这说明存在跨方法共享的任务瓶颈。
2. **“严格跨被试”只解释一部分差距，且影响高度依赖任务。** STA-Net 的统一 subject-level bridge 中，strict cross-subject 相对 trial-random 在 MI 为 +0.004、n-back 为 −0.004，但在 MA、DSR 约下降 0.049 和 0.046。三协议结果也不呈统一单调下降，因此不能把全部 gap 简化为 split 差异。
3. **EFRM 存在可检测的平均配对分离，但没有形成稳健的实例级共享表征。** sample-ID 去重后的双向 exact-pair AUC 约 0.543–0.565，但 MRR 仅 0.0027–0.0047、Recall@1 至多 0.00071、hardest-negative margin 为 −0.74 至 −0.84；两模态 centered effective rank 仅 1.06–1.38，第一主轴解释 95.98%–99.38% 能量。这支持“弱平均分离 + 高度低秩几何”，不支持“强同步实例对齐”。
4. **表征可保留很强的被试身份指纹，却只有弱 MI 跨被试任务信息。** 在相同 public MI cache 上，subject-held-out task probe 的 macro-F1 为 0.508–0.553；closed-set row-split subject-ID probe 在 BIOT/REVE 达到 0.946/0.980。后者只是 identity-retention 上界，不是 shortcut 的因果证明，但它说明“表征有信息”与“信息能跨被试服务任务”是两回事。
5. **当前输入和适配预算下，原论文创新点没有自动产生优势。** CBraMod 的单折固定预算 pilot 中，预训练 frozen 头未优于同架构 random-init，对 last-block/full 使用统一较高学习率则发生优化塌缩；BrainFusion 中 MI 的 stack 低于 EEG-only 约 0.047，8 s 内 NVC 类别 AUC 接近机会水平。这些是明确的后续机制靶点，而不是足以替换主表的调优结果。

因此，论文中更稳健的表述不是“所有方法因为跨被试而下降”，而是：

> 在统一严格跨被试、共同通道和观测预算下，多个方法面对共同的任务侧泛化瓶颈；协议改变对部分任务贡献明显，但不足以解释全部差距。进一步的 public-only 机制实验表明，部分预训练、对齐和多模态融合创新未在当前域形成稳定的任务相关收益。

## 2. 证据边界

本报告同时包含三种证据，不能混为同一等级：

| 证据类型 | 数据范围 | 可回答的问题 | 不能回答的问题 |
| --- | --- | --- | --- |
| 冻结描述性证据 | 已封存 protected aggregate 或既有 STA-Net 预测产物 | 低结果按任务/方法如何聚集；既有协议结果是否敏感 | 不能据此重新选模型、窗口、层或超参数；不能把 fold 当独立被试做推断 |
| public-only 探索性机制证据 | public development、outer-train/public-validation export | alignment、identity、输入窗、组件与适配容量是否显示预期机制 | 不能直接晋升为新的 protected 主结果或确认性因果结论 |
| 质量控制/能力审计 | split、manifest、cache、官方代码/权重/数据入口 | 数据合同是否一致、source anchor 是否可验证 | `pass` 不代表方法应有高性能；不可运行不代表方法无效 |

P0 没有打开新的 protected labels、indices 或 signals，也没有授权新的 protected endpoint。若 P1 结果需要进入论文定量主张，必须先冻结变体、选择规则和统计方案，再单独授权一次评估。

## 3. 首轮证据矩阵

| 问题 | 首轮结果 | 证据强度与状态 | 权威产物 |
| --- | --- | --- | --- |
| 低结果是方法特异还是任务聚集？ | B0-centered 描述性分解中任务占 51.78%；Visual 为共同低谷 | 冻结 post-hoc 描述性；非 ANOVA | [global diagnostics](../../comparative_methods/runs/performance_analysis/20260816_p0/global_diagnostics/) |
| 数据/标签/split 是否有系统性污染？ | 35 个 task-fold 的 subject、subject-record、join-key、trial-group 隔离通过；EFRM export 有重复 validation rows | QC `pass_with_findings` | [data audit](../../comparative_methods/runs/performance_analysis/20260816_p0/data_audit/REPORT.md) |
| 跨被试是否足以解释 STA-Net gap？ | cross 与 trial-random 的差在 MI/n-back 近零，在 MA/DSR 约 −0.05；无统一单调桥 | 既有证据的 subject-level 描述性重聚合 | [STA-Net bridge](../../comparative_methods/runs/performance_analysis/20260816_p0/stanet_bridge/REPORT.md) |
| EFRM alignment 是否为实例级？ | AUC 略高于机会，但 MRR/Recall 极低、hard margin 为负、几何近 rank-one | public validation、duplicate-aware、探索性 | [EFRM hierarchical](../../comparative_methods/runs/performance_analysis/20260816_p0/efrm_hierarchical/summary.md) |
| EFRM 的训练目标是否同步改善？ | 四组正式 Stage-A 重建损失下降约 75%–87%，CLIP 标量仅变化约 −0.24% 至 +0.13%；缺少 epoch-wise alignment export | public summary 轨迹；缺失值不插补 | [EFRM trajectory](../../comparative_methods/runs/performance_analysis/20260816_p0/efrm_trajectory/REPORT.md) |
| 能否识别生理 lag？ | 现有 evidence 缺绝对 modality-clock 窗口起点和 event identity；仅能形成 relative crop-offset proxy | fail-closed capability；`physical_lag_identifiable=false` | [EFRM lag](../../comparative_methods/runs/performance_analysis/20260816_p0/efrm_lag/README.md) |
| 表征是标签主导还是身份主导？ | MI task probe 约 0.51–0.55；BIOT/REVE closed-set subject-ID 约 0.95/0.98 | public cache、探索性；identity 上界非因果 | [identity probes](../../comparative_methods/runs/performance_analysis/20260816_p0/identity_probes/REPORT.md) |
| 经典 EEG 信号是否也受同一瓶颈影响？ | 逐被试 macro-F1：MI 0.531、n-back 0.350、Visual 0.189；Visual 同样明显低 | public strict-cross-subject 探索性 benchmark | [classical baselines](../../comparative_methods/runs/performance_analysis/20260816_p0/classical_baselines/REPORT.md) |
| CBraMod 是否只是冻结头过弱？ | 单折小样本 pilot 中 frozen MLP 无增益；last/full 在不合适的统一高 LR 下塌缩；random controls 不弱 | public 单折 fixed-budget pilot；只用于生成 P1 假设 | [CBraMod ladder](../../comparative_methods/runs/performance_analysis/20260816_p0/cbramod_ladder/PILOT_SUMMARY.md) |
| BrainFusion 的 NVC/stack 是否生效？ | MI stack 比 EEG-only 低 0.0469；MA 仅高于最好单模态 0.0126；8 s NVC 类别 AUC 近机会 | public 五折组件诊断；不是 source reproduction | [BrainFusion bridge](../../comparative_methods/runs/performance_analysis/20260816_p0/brainfusion_bridge/README.md) |
| 能否直接恢复原论文数值作正控制？ | 7 个方法均缺少至少一个不可替代 source-level 条件；其中 BIOT/CBraMod/REVE 为条件可运行 | 只读能力审计；无数值复现声明 | [source anchors](../../comparative_methods/runs/performance_analysis/20260816_p0/source_anchors/REPORT.md) |

## 4. 共同任务瓶颈

冻结 aggregate 中，36 个 macro-F1 单元格有 27 个高于各自 B0、8 个低于 B0、1 个 unsupported。按任务汇总的 `value − B0` 为：MA +0.0948、WG +0.0837、DSR +0.0439、n-back +0.0439、MI +0.0244、Visual −0.0213。

这一模式排除了“统一 adapter 让所有模型普遍失效”的简单解释：若 adapter 是唯一主因，预计方法间或所有任务会同步崩溃；实际 MA/WG 在多个方法上仍保留稳定增益，而 Visual 是跨方法共同失败面。更合理的优先级是：

1. Visual 首先审计标签语义、类不平衡、事件到窗映射、subject-held-out 学习曲线和经典基线；
2. MI/n-back 同时检查跨被试信号强度与身份/会话偏移；
3. 只有在同一任务上经典基线明显强于 foundation adapter 时，才把主要问题定位到 adapter/表示接口。

这里的平方和分解只是冻结面板的描述性压缩，不是因果效应、显著性检验或方差成分模型。

## 5. 数据合同：总体通过，但 EFRM 分析定义需要修正

G0 审计覆盖 7 个任务 × 5 folds、100,648 个样本清单和 614 个 public manifests。所有 35 个 task-fold 在 subject、subject-record、join-key 和 trial-group 上保持 train/validation 隔离；原始 `record_id` 可以跨被试复用，因此不被单独当作污染键。审计未发现需要 fail-closed 的 split 泄漏。

EFRM 的两个 complete-validation export 则暴露出一个独立问题：

| Stage-A export | 原始行数 | unique sample IDs | 重复行超额 | 最大重复次数 |
| --- | ---: | ---: | ---: | ---: |
| exclude single-trial | 7,559 | 4,787 | 2,772 | 21 |
| exclude simultaneous | 5,632 | 2,825 | 2,807 | 16 |

重复 sample-ID 的 EEG/fNIRS embedding 完全一致。原因与 balanced validation sampler 对较小数据集循环取样相容；这不是 train/test leakage。但是，旧分析采用 diagonal-only positive mask，会把重复的同一正样本放在 off-diagonal negative 中。因此：

- 旧的 full-matrix retrieval 不能继续作为实例级 alignment 证据；
- P0 的主分析先按 stable sample ID 去重；
- `raw_duplicate_aware` 只作为排除同 ID 假负样本的敏感性视图；
- checkpoint trajectory 中旧值明确标记为 `row_weighted_duplicate_unaware_existing_export`。

## 6. 协议变化不是统一的单调惩罚

STA-Net 是目前唯一拥有可重聚合三协议结果的比较方法。以 subject 为统计单位、10,000 次 subject bootstrap 后，主要结果为：

| 任务 | Trial-random | Group-safe within-subject | Strict cross-subject | Cross − trial |
| --- | ---: | ---: | ---: | ---: |
| MI macro-F1 | 0.5448 | 0.5009 | 0.5486 | +0.0038 |
| MA macro-F1 | 0.6869 | 0.6237 | 0.6381 | −0.0488 |
| WG macro-F1 | 0.6015 | 0.5495 | 0.5769 | −0.0246 |
| n-back macro-F1 | 0.3744 | 0.3059 | 0.3708 | −0.0036 |
| DSR macro-F1 | 0.6276 | 0.4965 | 0.5812 | −0.0464 |
| Visual macro-F1 | 0.2622 | NA | 0.2396 | −0.0226 |
| REFED CCC | 0.1058 | −0.1052 | 0.0728 | −0.0330 |

MI 和 n-back 的 strict 结果几乎没有低于 trial-random，MA 和 DSR 则下降约五个百分点。Group-safe within-subject 也不总位于两者之间。这个结果说明 protocol sensitivity 是真实且任务依赖的，但没有支持“被试内 → 跨被试必然造成同幅巨大下降”的 H1 强版本。

该 bridge 也不能作 split 的纯因果效应：three-protocol 运行的训练支持和配置哈希不同，within-subject aggregate 只暴露 count、没有 sample IDs。Visual 因仅覆盖 11/16 subjects 被显式置为 NA；target-subject fine-tuning 仅作 context，不混入 bridge。

## 7. EFRM：alignment 可检测，但不可检索且高度低秩

### 7.1 训练轨迹

四个正式 Stage-A run 中，EEG validation reconstruction 下降 82.6%–86.6%，fNIRS reconstruction 下降 74.7%–79.9%，而 CLIP 标量变化只有 −0.235% 至 +0.128%。现有文件没有逐 epoch 的 AUC、MRR、margin 或 effective-rank，因此不能把最终 checkpoint alignment 值复制到每个 epoch，也不能证明 reconstruction 改善带来 shared-domain 改善。

### 7.2 Duplicate-aware 分层检索

| Exclusion run | 方向 | exact AUC | MRR | Recall@1 | Recall@5 | Hardest margin |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Exclude single-trial | EEG→fNIRS | 0.5650 | 0.002742 | 0.000209 | 0.001880 | −0.7423 |
| Exclude single-trial | fNIRS→EEG | 0.5637 | 0.002823 | 0.000627 | 0.001671 | −0.7657 |
| Exclude simultaneous | EEG→fNIRS | 0.5630 | 0.004748 | 0.000708 | 0.003540 | −0.8229 |
| Exclude simultaneous | fNIRS→EEG | 0.5434 | 0.003251 | 0.000000 | 0.001770 | −0.8356 |

AUC 表明平均 exact-pair score 并非完全随机；subject/record-block permutation 也在部分 AUC、score 和 margin 指标上观察到偏离 null。然而，MRR 与 Recall 极低，每个 query 的真配对通常不能战胜 hardest negative；部分 same-dataset、cross-dataset 或 same-class 负池的 AUC 与 exact-pair 接近或更高。由于负池可以重叠、置换指标多且未作多重性校正，这些 p 值保持探索性，不作确认性显著声明。

### 7.3 几何

unique-sample embedding 的 centered effective rank 和第一主轴能量为：

| Run | EEG rank / axis 1 | fNIRS rank / axis 1 |
| --- | ---: | ---: |
| Exclude single-trial | 1.379 / 95.98% | 1.206 / 96.60% |
| Exclude simultaneous | 1.231 / 97.65% | 1.060 / 99.38% |

在 768 维 embedding 中，这属于接近 rank-one 的几何压缩。它能解释为何总体 cosine 分布仍可有小幅均值差，却缺乏可用的实例排序结构。P0 因而支持：**当前 EFRM checkpoint 学到弱的平均跨模态分离，但未形成稳健、丰富的同步实例共享域。**

### 7.4 Lag 能力边界

三组 export 可计算同 record 的 relative crop-offset proxy，但 `crop_start_s` 是 event/source-window-relative offset，现有 evidence 无法把 sample ID 反解到 event identity 和两模态的绝对采集起点。0 s 还包含 identity/self-pair baseline。因此本轮明确记录 `physical_lag_identifiable=false`，不从 proxy 曲线声称生理血氧延迟峰。

## 8. 身份保留强于任务迁移

在 MI public feature cache 上，固定 probe 预算得到：

| 方法 | Task macro-F1 | Session macro-F1 | Closed-set subject-ID macro-F1 |
| --- | ---: | ---: | ---: |
| BIOT | 0.5095 ± 0.0409 | 0.3442 ± 0.0061 | 0.9455 ± 0.0150 |
| CBraMod | 0.5190 ± 0.0348 | 0.2940 ± 0.0250 | 0.6626 ± 0.0342 |
| REVE | 0.5530 ± 0.0285 | 0.3902 ± 0.0455 | 0.9797 ± 0.0099 |
| NormWear | 0.5082 ± 0.0426 | 0.3239 ± 0.0199 | 0.4320 ± 0.0325 |

Task 和 session probe 均按 subject group holdout；subject-ID 则必须是 closed-set row split，因为 held-out 新 subject 的 ID 类别不可能在训练类集合中出现。该 row split 未隔离同一 record/session，故只能作为 identity-retention 上界。它不能单独证明模型在下游使用了被试 shortcut，也不能与跨被试 task probe 作严格相同 estimand 的数值差检验。

即便如此，BIOT/REVE 的结果仍说明低任务分并非表示“没有信息”，而是可重复身份信息远强于当前可线性迁移的 MI 标签信息。下一步应在 outer-train 内拟合 identity nuisance direction，再对 untouched public dev subjects 测试标签性能，才可接近因果判断。

## 9. 经典 EEG 基线重现相同的任务困难顺序

G1 使用固定 16 通道、Welch 频带功率（1–4、4–8、8–13、13–30、30–45 Hz）、train-only 标准化和 shrinkage LDA，没有针对任务搜索特征或超参数。所有结果来自 strict-cross-subject public outer folds。

public validation folds 之间会重复出现 subject 和 sample ID，不能把 5 个 fold 的窗口直接拼接后称为 OOF。主 estimand 因而是：先在每个 fold 内计算每位 subject 的 macro-F1，再对同一 subject 跨其 validation folds 求均值，最后以 subject 为统计单位作 10,000 次 bootstrap。

| 任务 | Subjects | Subject mean macro-F1 | Subject SD | 95% bootstrap CI | Fold mean（次要） |
| --- | ---: | ---: | ---: | ---: | ---: |
| MI | 17 | 0.5305 | 0.1012 | [0.4839, 0.5775] | 0.5491 |
| n-back | 16 | 0.3496 | 0.1343 | [0.2859, 0.4139] | 0.3979 |
| Visual | 8 | 0.1888 | 0.0355 | [0.1686, 0.2145] | 0.2189 |

MI 的简单频带模型约为 0.53，处于当前 foundation comparator 的同一量级；n-back 的 subject mean 仅略高于三分类 1/3，Visual 则低于四分类 1/4。由于 public 与 frozen protected 主表并非同一结果面，这里不作方法胜负比较；但三项结果清楚显示，简单、可解释的 EEG 模型也没有在 Visual/n-back 上恢复出接近原论文高分的跨被试信号。

这加强了“任务/域瓶颈是共同原因”的解释，同时保留一个重要差异：MI/Visual 的 subject SD 和 CI 说明 subject 异质性不可忽略，fold pooled 指标会高估或掩盖这一点。

## 10. 适配容量和多模态机制的受控 pilot

### 10.1 CBraMod

单个 MI public outer fold、每类 64 train + 64 validation、固定 5 epochs 的 ladder 结果为：frozen linear/MLP 均 0.5152，random linear 0.5271，random MLP 0.5816，last-block/full fine-tune 均 0.3333。后两者使用统一 `1e-3` 学习率，而上游 fine-tune 默认更接近 `1e-4`、非零 weight decay、gradient clipping 和分组学习率。

因此，本 pilot 只支持两个后续判断：

1. 当前小样本、固定预算下没有观察到正的 pretrained-transfer gain；
2. last/full 的失败首先是优化不匹配警报，不能当作 fine-tune 上限。

P1 应在 public 数据上冻结 source-like optimizer schedule、多个 folds/seeds 和不依赖 validation 选 epoch 的预算后再判断 H3/H4。

### 10.2 BrainFusion

五个 public outer folds、固定 seed 17 的组件诊断为：

| 任务 | EEG | HbO | HbR | NVC | Grouped-OOF stack |
| --- | ---: | ---: | ---: | ---: | ---: |
| MI | 0.5727 | 0.4942 | 0.4836 | 0.5093 | 0.5258 |
| MA | 0.5117 | 0.5206 | 0.5001 | 0.5124 | 0.5332 |

MI 中 stack 比 EEG-only 低 0.0469；MA 中 stack 比最好单模态 HbO 高 0.0126。已有 cache 只含刺激起点后的 8 s，不能观察完整 fNIRS 后延；2/4/6/8 s prefix 的 MI mean-absolute-NVC 类别 AUC 为 0.508/0.488/0.505/0.506，MA 和 WG 的 8 s AUC 为 0.438/0.414。短窗 absolute correlation 还会被 sample-wise min–max 与 HRF 边界放大。

所以本轮只说明原论文中的 NVC/stacking 优势没有在当前严格跨被试、8 s 预算下自动出现；它不说明 stacking 普遍无效，也不能替代 source-case 复现。P1 需要 time-reversed/random HRF、同被试 pair shuffle、逐被试 stack delta 和真实长后延 fNIRS。

## 11. Source-anchor 可验证性

本地只读盘点没有找到任何可立即、完整建立“原论文数值正控制”的方法：

- BIOT、CBraMod、REVE 有可识别的官方代码/权重入口，但缺外部数据、完整 split manifest 或受控权重使用条件，属于条件可运行；
- NormWear 原预训练模态不含 fNIRS，当前结果必须称 `NormWear EEG-fNIRS adapted`；
- EFRM 缺官方 pretrained checkpoint 和源数据；
- STA-Net runner 依赖未发布数据和硬编码本地路径；
- BrainFusion 公开代码没有论文 CSP/AutoML/stacking case 的完整执行路径。

这意味着论文中应把“source numerical reference”“source-aligned adaptation”和“本项目 independent reimplementation”分列。不能因为 adapter 已通过输入输出审计，就把当前结果称为原论文数值复现；也不能因为 source anchor 不可运行，就反向断言原论文结果不可相信。

## 12. 综合归因与论文可用表述

| 候选归因 | P0 判定 | 理由 |
| --- | --- | --- |
| 代码/切分普遍错误 | **不支持为主因** | 35 个 public task-fold 的 contamination-relevant group keys 均隔离；运行完成度和 cache 审计总体一致。EFRM 的确有分析 mask 问题，但不是 split leakage |
| 跨被试 estimand 改变 | **部分支持、任务依赖** | STA-Net 在 MA/DSR 有约五点下降，在 MI/n-back 几乎没有下降；不呈统一单调桥 |
| 任务跨被试信号弱 | **较强支持** | task-centered 分解占比最高；Visual 跨方法共同低；经典 EEG 基线也未普遍远超 foundation 方法 |
| 冻结/适配容量不合适 | **CBraMod 初步支持，尚未定量** | 单折 pilot 暴露 optimizer mismatch，但没有 source-like 多折容量曲线 |
| 跨模态 alignment 弱或塌缩 | **EFRM 较强支持** | duplicate-aware retrieval 很弱、hard margin 负、几何近 rank-one |
| 被试身份主导表示 | **相关性支持，因果未证实** | BIOT/REVE closed-set identity decodability 极高；仍需 nuisance intervention |
| 多模态 NVC/stack 未生效 | **当前 8 s BrainFusion 设置下支持** | MI stack 低于 EEG-only，prefix NVC AUC 接近机会；长后延数据不可用 |
| 原论文数值可直接比较 | **不支持** | estimand、通道、窗口、训练边界和 source artifacts 均存在方法特异差异 |

论文可以写：

- “严格跨被试协议是 gap 的一部分，但其影响是任务依赖的，不能解释所有方法和任务。”
- “多方法共同在 Visual、MI/n-back 等任务上接近基线，且经典基线呈相似困难，支持任务/域侧泛化瓶颈。”
- “在 EFRM public validation export 上，duplicate-aware 分析发现弱平均配对分离、极低实例检索和近 rank-one 表征几何。”
- “部分表示保留强被试身份信息，但其与低跨被试任务性能的关系仍是相关性证据。”

论文不应写：

- “原论文都没有跨被试，所以低分完全由 protocol 导致”；
- “EFRM 完全没有 alignment”或“已识别出 fNIRS 生理 lag”；
- “subject-ID probe 证明模型使用了身份 shortcut”；
- “CBraMod full fine-tune 无效”或“BrainFusion stacking 普遍无效”；
- “当前 adapter 分数复现了原论文结果”。

## 13. P1 优先级

在不重新打开 protected test 的前提下，建议按下列顺序推进：

1. **CBraMod source-like 适配阶梯。** 固定较小 backbone LR、weight decay、clip、multi-LR、多个 public folds/seeds；预训练与 random-init 共用预算。
2. **STA-Net 因果注意扰动。** fNIRS modality shuffle、空间注意置换、EEG 时间反转/lag shuffle，并报告逐被试预测一致率和 paired delta。
3. **BrainFusion NVC null 与长窗能力。** 正确/反转/random HRF、同被试和跨被试 pair shuffle；只有取得真实 post-stimulus fNIRS 后才做长窗 dose。
4. **BIOT/REVE/NormWear 原创新点干预。** BIOT channel semantics + random-init；REVE trainable query + coordinate permutation；NormWear modality/CWT/liaison ablation。
5. **Identity nuisance intervention。** 只在 outer-train 拟合 subject/session directions，对 untouched public dev subject 测 task probe；避免把 identity probe 当因果终点。
6. **EFRM controlled mechanism variants。** reconstruction-only、shuffled-pair、temperature/variance-covariance regularization；必须同时改善几何、负对照区分和 public downstream 才算机制修复。
7. **物理 lag 再导出。** 每个样本提供 event identity、EEG/fNIRS 绝对 window start/end、record clock 和前后 context；否则继续 fail closed。
8. **Source numerical anchors。** 只在取得官方数据/checkpoint/split 的不可替代条件后运行，不用 project adapter 冒充 source reproduction。

## 14. 复现与产物索引

P0 冻结 manifest 为
[`p0_experiment_manifest.yaml`](../../comparative_methods/performance_analysis/p0_experiment_manifest.yaml)，
分析入口位于
[`comparative_methods/performance_analysis`](../../comparative_methods/performance_analysis/)
及各方法目录。每个主图包均提供 PNG/PDF、source CSV 或 JSON、alt text 和 manifest；缺失或不可比较结果保留为 NA/capability，而未被插值。

推荐优先查看：

- [全局 B0-centered 热图](../../comparative_methods/runs/performance_analysis/20260816_p0/global_diagnostics/value_minus_b0_heatmap.png)
- [STA-Net 协议桥](../../comparative_methods/runs/performance_analysis/20260816_p0/stanet_bridge/bridge_protocols.png)
- [EFRM 分层检索：exclude-ST](../../comparative_methods/runs/performance_analysis/20260816_p0/efrm_hierarchical/efrm_lodo_full_target_fivefold_v2__exclude_eeg_fnirs_single_trial__stage_a_seed42/hierarchical_alignment.png)
- [EFRM checkpoint 轨迹](../../comparative_methods/runs/performance_analysis/20260816_p0/efrm_trajectory/trajectory.png)
- [MI identity/task/session probes](../../comparative_methods/runs/performance_analysis/20260816_p0/identity_probes/figures/identity_probe_macro_f1.png)
- [逐被试经典 EEG 基线](../../comparative_methods/runs/performance_analysis/20260816_p0/classical_baselines/subject_macro_f1_public_fold_average.png)
- [CBraMod adaptation ladder](../../comparative_methods/runs/performance_analysis/20260816_p0/cbramod_ladder/figures/adaptation_ladder_epoch_trajectory.png)

本报告是 P0 的综合解释层；数值以各目录的 machine-readable CSV/JSON 和 manifest 为准。既有 protected 主表、准入状态和 540-job traceability 继续以
[`PROTECTED_CAMPAIGN_RESULTS_20260814.md`](PROTECTED_CAMPAIGN_RESULTS_20260814.md)
为唯一权威，不被本轮探索性结果替换。
