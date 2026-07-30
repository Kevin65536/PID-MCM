# 对比方法最终性能数字准入规则

_适用于后续对比实验结果获取和论文表格填数，更新于 2026-07-30_

## 1. 这份规则只管最终数字

本文只回答一个问题：

> 某个 `method × task × track × metric` 的结果，能否作为该对比方法的正式性能
> 数字填入论文表格？

它不规定训练、调参、数据划分、代码发布、日志公开、工件哈希或 protected-test
管理方式，也不要求为准入额外运行重型统计流程。已有方法特定实验协议继续支配
其运行过程；本文只验收其最终汇总结果。

目标是同时做到：

1. 在现有资源下，数字足以代表该方法的合理高水平，而不是同口径候选中的异常低值；
2. 没有数据、实现或方法适配失败时，不把低于无信息基线的结果当作正常性能；
3. 在任务和指标可比时，与原论文的绝对水平和相对关系形成合理对照；
4. 数字一旦达到准入要求，就已经足够用于论文；协议不要求为更小的增益追加实验。

机器可读版本见
[`comparison_metric_targets_v1.yaml`](../../comparative_methods/comparison_metric_targets_v1.yaml)。

## 2. 每个待填数字只需要七项结果信息

无需建立新的公开工件体系。内部验收表每格只记录：

| 字段 | 含义 |
| --- | --- |
| `value` | 按该表统一口径聚合的最终点估计 |
| `uncertainty` | 已有重复、fold 或 subject 汇总产生的 SD/CI；没有时标 `not_available` |
| `B0` | 同一数据支持和同一指标上的简单无信息基线 |
| `target` | 该格的 `minimum_admissible` 与更理想的 `preferred_target`；可选记录异常高值检查线 |
| `source_target` | 原论文数值及其可比性；不可比时为 `not_applicable` |
| `relation_status` | 原论文任务顺序/模态关系是否适用及是否保持 |
| `decision` | 第 6 节定义的唯一准入状态 |

这些信息只用于决定表格填数，不自动构成对外公开要求。

## 3. 四个硬性结果门

### 3.1 数字本身有效且可比

正式表格中的数字必须同时满足：

- 有限、位于指标合法范围内，标签顺序、metric 方向和缩放无误；
- 与同一表内其他方法使用相同的 task、split、输入模态、label budget、metric
  和 aggregation；linear probe、full fine-tune、外部预训练等不同 track 分表；
- 使用完整汇总值，不用最好 seed、最好 fold、最好 subject 或最好 checkpoint
  的 test 数值替代整体结果；
- 既有结果定义中预期的 subject/fold/seed 单元缺失，或分母因某个方法运行失败而
  缩小时，该格为 `INVALID_VALUE`，不能用剩余容易样本计算一个较高数字；本文不
  新增 seed/fold 数量；
- 均值、SD/CI 和样本单位一致。fold SD、seed SD 和 subject SD 不混称；
- 小数精度与结果不确定性相称，默认百分比保留两位小数、`[0,1]` 指标保留四位。

任一项失败时先修正结果汇总；不能通过换指标或换一组更有利的样本解决。

### 3.2 必须高于正确的无信息基线

定义方向归一化增益：

\[
\Delta_{\mathrm{base}} =
\begin{cases}
M-B_0, & \text{越大越好}\\
B_0-M, & \text{越小越好}.
\end{cases}
\]

正式 named-method 性能要求 `Δ_base > 0`。基线按指标选择：

| 指标 | `B0` |
| --- | --- |
| Accuracy | 同一评估支持上的 train-fold majority；同时给出 `1/K` 作为参考 |
| Balanced Accuracy | `1/K` |
| Macro-F1 | majority 与 train-prior random 的期望中较强者；不能直接写死为 `1/K` |
| Cohen's Kappa | `0` |
| CCC / Pearson / Spearman | `0` 表示无正向关联；常数预测的相关系数不得伪造 |
| R² | `0`，并同时检查 train-mean predictor 的实际 held-out R² |
| MAE / RMSE | train-target-mean predictor 的实际 held-out error |

若已有 fold/seed/subject 重复结果，沿用该表既定的汇总单位报告 uncertainty；
不同单位结论混杂时加稳定性说明，但不另造一个“多数 fold/seed/subject”硬门。
已有 95% CI 完全高于 `B0` 是强证据，但不是为了填表必须额外运行
bootstrap/permutation 的前置条件。

低于或等于 `B0` 时：

- 有明确的数据、实现或域适配失败：记 `FAILURE_RESULT`；
- 没有明确失败解释：记 `REJECTED_VALUE`，不能把该值当作正常性能；
- 两种情况都不能把该值裁到 chance、替换为论文值或人工抬高。

### 3.3 必须像一个“资源内合理高水平”结果

这里不要求证明理论上界。每格使用两级、方向感知的结果阈值：

- `minimum_admissible`：一个数字进入主表的最低合理水平；
- `preferred_target`：足以支持“接近现有资源下合理高水平”的目标。

对越大越好的指标使用 `value ≥ threshold`，对 MAE/RMSE 等越小越好的指标使用
`value ≤ threshold`。

先定义不依赖历史候选的轻量 fallback `F`：

- 所有越大越好的指标：`F = min(legal_max, B0 + 0.02)`；
- MAE/RMSE：`F = 0.98 × B0`。

若存在同一 `method × task × track × metric` 单元、且 dataset/evaluation
support、split、输入模态、label budget、aggregation 和资源量级相容的已知候选，
令 `H` 为其中**最强的有效结果**。不能拿其他方法或任务的强值作 `H`，不能挑
较弱的 `H` 降低门槛，也不能逐任务混用不兼容 variant。默认的 80% 最低线和
90% 理想线为：

\[
\begin{aligned}
T_{80} &=
\begin{cases}
B_0 + 0.80(H-B_0), & \text{越大越好}\\
B_0 - 0.80(B_0-H), & \text{越小越好},
\end{cases}\\
T_{90} &=
\begin{cases}
B_0 + 0.90(H-B_0), & \text{越大越好}\\
B_0 - 0.90(B_0-H), & \text{越小越好}.
\end{cases}
\end{aligned}
\]

存在 `H` 时，`minimum_admissible` 取 `F` 与 `T80` 中性能方向上更严格者，
`preferred_target` 取 `minimum_admissible` 与 `T90` 中更严格者。高度可比的
source track 还要求至少 80%/90% source recovery；若 source 与 `H` 同时适用，
仍取性能方向上更严格的线。

若既没有可用的 `H`，也没有高度可比的 source target，则令
`preferred_target = minimum_admissible`，该格最高为 `TABLE_READY_WITH_NOTE`，
脚注说明“缺少同口径能力参考，未单独证明接近上界”。这给 UMAP 等暂无可信同域
历史值的方法一个确定性目标，同时不要求新增实验来制定目标。

异常高值检查是可选诊断，不是每格必填字段。只有结果明显超出指标合法范围、
可比论文值或可信同口径结果时，才先排除泄漏、量纲和聚合错误。

恒定预测或完整类塌缩按上述 baseline 和 minimum 线直接判定，不再增加主观的
“明显离群”门。Linear probe 只代表冻结表示质量；它可以进入 linear-probe 表，
但不能被写成端到端方法性能上界。

### 3.4 与原论文的对照必须合理

只有 dataset、task、split、metric、模态、label budget、aggregation 和资源量级
基本可比时，原论文数值才形成硬对照。此时使用 above-baseline recovery：

\[
R_{\mathrm{source}}
=
\frac{M_{\mathrm{project}}-B_{\mathrm{project}}}
     {M_{\mathrm{paper}}-B_{\mathrm{paper}}}
\]

（越小越好指标反向计算）。

- `R_source ≥ 0.90`：达到默认预期，可记 `TABLE_READY`；
- `0.80 ≤ R_source < 0.90`：可记 `TABLE_READY_WITH_NOTE`，说明资源或实现差异；
- `R_source < 0.80`：默认不作为该方法的正常最终水平，记 `REJECTED_VALUE`；
  有明确失败模式时记 `FAILURE_RESULT`；
- 论文值、baseline 或评估口径不可比时，不计算 recovery，标
  `not_applicable`，改用 shared benchmark 的 above-baseline 与稳定性门。

默认 practical tolerance 直接由指标给出：越大越好的指标为原生量纲
`0.02`（例如两个百分点或 `0.02` CCC），MAE/RMSE 为 `0.02 × |B0|`。
只有事先已有任务公认容差时才覆盖这个默认值，不为某个观察结果临时调整。

若论文的 above-baseline gain 非正，或不大于本格的 practical tolerance，
该比率也记 `not_applicable`，避免近零分母制造夸张恢复率。

90%/80% 是高度可比单元格的项目默认工程线，不是跨方法通用的“复现成功”标准。
若已知域差异或资源量级差异使 source 不再高度可比，则把 source comparison 标成
`not_applicable`，改用同 track 的 `H` 或通用 fallback；不能在看到候选低值后
临时下调目标。

原论文的任务顺序或模态关系只在比较对象可比、差异超过上述 practical
tolerance 时约束结果。缺少 paired
不确定性本身不抹掉一个远大于 practical-tie 的清楚均值顺序；此时把它当作
描述性 sanity target，而不声称统计复现。论文差异落在 practical-tie 内或已有
区间明显重叠时，才记 `tie/not_testable`。
若出现稳健反转，必须满足至少一种情况：

- 数据或任务定义改变；
- 模态、label budget、split 或 aggregation 改变；
- 有理论上合理的域迁移解释；
- 找到明确的失败模式。

若差异落在论文不确定性或 practical-tie 范围内，记 `tie/not_testable`，不降低
数字准入状态；超出 tie 范围的稳健、无解释反转为 `REJECTED_VALUE`。

## 4. 当前方法的具体对照目标

### STA-Net

当前可审计的原论文 balanced-binary mean Accuracy 如下。表内两条目标线假设
project `B0` 也为 50%；若实际 majority baseline 不同，按第 3.4 节公式重算：

| 任务 | 论文值 | 80% 最低线 | 90% 理想线 |
| --- | ---: | ---: | ---: |
| MI | 69.65% | 65.72% | 67.69% |
| MA | 85.14% | 78.11% | 81.63% |
| WG | 79.03% | 73.22% | 76.13% |

原论文描述性顺序是 `MA > WG > MI`，不是示例中的 `MI > WG`。该数值目标和顺序
只约束论文兼容的 subject-specific/session track；在没有明确域差异或失败模式
时，source-compatible 最终数字应保持这一顺序。Shared cross-subject 表只要求
同口径 above-baseline、稳定和合理，不要求复刻这些绝对值。新的 shared
cross-subject 候选可直接用 2026-07-27 同口径可信 aggregate 的 above-baseline
gain 90% 作为轻量 `preferred_target`，无需为设置目标另跑实验。

### EFRM

原论文任务、few-shot regime 与本项目七任务不完全对应，因此不设置虚假的
MI/WG 数值或顺序目标。项目结果应分清：

- frozen-backbone linear probe：表示质量；
- full fine-tune：端到端适配水平；
- target-excluded 与 target-seen pretraining：不同数据条件。

这些 track 不混表。EFRM v2 保留其已经冻结的 fold-mean primary；只有使用相同
subjects、folds、seeds、metric、aggregation、modality 和 transfer setting 的
结果才能直接排名。历史 v1 与 v2 不是同一 track，不能冒充 `H` 或成为 v2 的
硬下界；它只提供上下文参考。v2 使用通用 fallback，只有证明某个候选与 v2
口径相容后，才能把它作为 `H` 设置 80%/90% 目标。

### UMAP

原论文是 SEED-IV/V/VII 上的 EEG+眼动情绪分类，本项目替换为 EEG+fNIRS 并改变
任务，因此原论文绝对值不适用。`paired > EEG-only > eye-only` 只作为原域参考，
不能强制本项目 MI/WG 顺序。历史上重复查看 test 后得到的 best-test 值不能作为
当前正式性能数字。

## 5. 审稿人最容易质疑的结果模式

以下情况不能无说明地出现在主表：

- 某个完整方法低于 majority/chance，而更简单基线明显更高；
- 同一行混用不同 split、模态、label budget、probe/fine-tune 或聚合口径；
- 只报最好 seed/fold，或每个任务从不同 variant 中挑最高值；
- 原论文可比任务恢复率明显低于 80%，却声称“成功复现”；
- 原论文中稳定的任务/模态关系强烈反转，却没有任务差异或失败解释；
- mean 与误差条来自不同单位，或把 fold 数当作独立受试者数；
- 缺失困难 subject/seed 后仍用缩小分母给出正常排名；
- 用 `NR`、chance clipping 或论文原值隐藏一次有效但不利的运行结果。

## 6. 轻量准入状态与使用规则

每格只取一个状态，按表中从上到下的优先级判定：

| 状态 | 结果条件 | 表格处理 |
| --- | --- | --- |
| `INVALID_VALUE` | 数值、分母、metric、aggregation、结果完整性错误，或异常高值尚未排除泄漏/口径错误 | 不填数值，先核验计算 |
| `FAILURE_RESULT` | 数字有效，但至少一个正常结果门因已确认的数据、方法或域适配失败而未通过 | 若该方法必须出现，以失败结果单列/脚注，不冒充正常性能上界 |
| `TABLE_READY` | 有效、同口径、有相容的 `H` 或 source target、达到 `preferred_target`，且适用的 source/relation 目标通过 | 可直接填主表并参与同口径排名 |
| `TABLE_READY_WITH_NOTE` | 尚未命中前三种状态，达到 `minimum_admissible` 和所有适用的 source/relation 最低线，且存在明确脚注原因：未达 preferred、source recovery 为 80–90%、缺少 `H`/source target 或可解释偏差 | 可填主表，加简短脚注 |
| `REJECTED_VALUE` | 数字有效但未通过正常结果门，且没有能解释该失败的已确认失败模式 | 不作为正常性能数字填主表；内部保留真实值和拒绝原因 |

`TABLE_READY` 与 `TABLE_READY_WITH_NOTE` 表示现有数字已经足够填表，无需为
准入继续优化。
其他状态只说明数字当前不能作为正常性能；本规则不规定是否、如何或运行多少次
后续实验。任何状态都禁止篡改、裁剪或捏造数值。

同一方法不能只删除 `REJECTED_VALUE` 的低分任务而保留高分任务。可选方法要么整条
可比 track 排除并说明，要么保留状态；预先承诺的核心基线则必须让失败状态可见。

这套规则的目标不是让每个结果都漂亮，而是用最少额外工作得到一个审稿人看来
有效、可比、符合方法合理能力边界的最终数字。
