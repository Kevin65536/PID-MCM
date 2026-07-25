# E2 语义目标全面复核与 E0 状态校正

_2026-07-24 · development-only · protected subjects 24–29 保持关闭_

## 结论

本轮复核支持上一轮结论，正式决策仍为：

`no_semantic_row_admitted_retain_T0`

没有发现结果文件缺失、checkpoint 损坏、运行未完成、保护测试泄漏、梯度
路由违规或量化器塌缩。9/9 个 T0/T1/T2 × 3 seed 运行均完成 462 steps，
18/18 个 best/last checkpoint 的 SHA-256 与 manifest 一致，sidecar 和权重
校准文件哈希一致。因此，T1/T2 未被接纳是生理语义约束没有转化为可泛化
token 结构，而不是软件或产物完整性故障。

更具体地说，E2 当前同时存在四个主问题：

1. **EEG required 状态没有进入可泛化表示。** T0/T1/T2 的 EEG hard-token
   seed 均值分别为 `-0.109/-0.163/-0.201`；连续 latent、posterior 和
   codebook embedding 也都为负。所有 9 个 EEG row×seed 结果均低于
   shuffled-target q95。
2. **EEG 语义监督出现明显训练—验证分离。** T1 EEG state 训练损失平均
   下降 `50.5%`，验证仅下降 `0.7%`；T2 训练下降 `39.7%`，验证反而上升
   `3.0%`。最终 state generalization gap 约为 `0.635/0.631`。
3. **EEG 可用目标支持显著少于 fNIRS。** 虽然 sidecar 接纳了 `230/230`
   target trials，但训练日志中 sample-level auxiliary target coverage 仅约
   `16.7%`；信号有效性掩码后，EEG required probe 只使用
   `797/1800=44.3%` train patches 和 `178/500=35.6%` validation patches，
   而 fNIRS 是 `100%/100%`。
4. **fNIRS 有基础可解码性，但不是 T1/T2 新增的语义收益。** fNIRS
   hard-token 在所有 row/seed 中都高于 null，但 T1/T2 相对 T0 没有稳定
   增益，且跨 seed prototype stability 从 T0 的 `0.9318` 降至
   `0.9226/0.9247`。

复核还发现原 subject bootstrap 与注册的 run-level 主终点口径不完全一致。
按被试聚类、跨 seed 保持被试身份并对两模态等权的敏感性分析后，T1/T2
均值分别为 `-0.0578` 和 `-0.0926`，95% exact cluster bootstrap 区间为
`[-0.1573, 0.0062]` 和 `[-0.2343, 0.0082]`。方向仍不支持 T1/T2，但原报告
中的区间不应解释为严格的被试聚类不确定性；尤其不应把 T2 原区间不跨零
表述为稳健的统计显著劣化。

## 复核范围与证据

本轮没有重训模型，也没有打开 24–29 号保护被试。复核只读取冻结的：

- 9 个正式 run 的配置、manifest、checkpoint 哈希、训练/验证轨迹和量化器健康；
- `state_decoding.json`、`prototype_signatures.csv`、
  `prototype_stability.json`、`objective_ablation.csv`；
- 原 `decision.json`、配对被试差异和坐标差异；
- train-only 权重梯度校准；
- v4 E0 revalidation 的目标可观测性、K=128 transmissibility、物理重建和
  posterior calibration。

重新生成的机器摘要、7 张数据表和 4 组多面板图位于：

`experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/`

复核脚本为：

`experiments/review_e2_semantic_objective_results.py`

## 门禁复核

| 层级 | 结果 | 解释 |
| --- | --- | --- |
| 9-run 完整网格 | Pass | T0/T1/T2 × 3 seeds 全部存在 |
| 训练完成 | Pass | 9/9 为 `training_complete`，均为 462 steps、14 epochs |
| Checkpoint 完整性 | Pass | 18/18 best/last SHA-256 匹配 |
| Split/sidecar/weight hash | Pass | 所有运行共享同一 split；sidecar 与校准哈希匹配 |
| Protected test | Pass | suite、evaluation 与 E0 均未打开 24–29 |
| Gradient entry routing | Pass | 9/9 严格 allowlist 通过 |
| Quantizer health | Pass | E1 health、revival stop、effective rank 均通过 |
| E0 required local target observability | Pass | required `r` 与 HbO/HbR 坐标通过 |
| E0 target-space K=128 transmissibility | Pass | EEG `R²=0.7344`，fNIRS `R²=0.8791` |
| E0 complete sign-calibrated physical teacher | **Pass** | SSM 生理信息（含 fNIRS）完全可接受 |
| E2 required hard-token improvement | **Fail** | T1/T2 不具 seed-consistent improvement |
| E2 hard-token null | **Fail** | EEG 0/9 高于 null；fNIRS 9/9 高于 null |
| Prototype non-decrease | **Fail** | EEG 提升，但 fNIRS T1/T2 均低于 T0 |
| T2 optional `s` 与 required non-decrease | **Fail** | optional 不稳定，required 三 seed 均下降 |
| G3/E6/G2/coupling promotion | Not eligible | 当前仅为 development evidence |

运行时 manifest 均记录 `dirty_worktree=true`。9 个
`implementation_snapshot.json` 的核心源文件哈希映射一致，因此横向比较
仍可复现；但单独依赖 git commit 不能完整恢复当时状态，后续正式阶段应以
clean worktree 或归档 source bundle 消除这一 provenance 限制。

## 主终点复核

![E2 primary endpoint re-audit](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig01_primary_endpoint_reaudit.png)

图 1A–B 同时显示 observed hard-token `R²` 与各自 shuffled-target q95；
图 1C 是注册的 seed-matched 两模态 pooled endpoint；图 1D 是新增的
被试聚类、模态等权敏感性分析。误差线为 5 名 validation subjects 的
exact cluster bootstrap 95% 区间，seed 被视为固定初始化重复，而不是新的
生物学样本。

### 各 row 的 hard-token 水平

| Row | EEG mean R² ± seed SD | EEG mean null margin | EEG above null | fNIRS mean R² ± seed SD | fNIRS mean null margin | fNIRS above null |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 | `-0.1092 ± 0.0381` | `-0.0624` | 0/3 | `0.1420 ± 0.0319` | `+0.1650` | 3/3 |
| T1 | `-0.1626 ± 0.0398` | `-0.0901` | 0/3 | `0.1542 ± 0.0094` | `+0.1805` | 3/3 |
| T2 | `-0.2006 ± 0.0213` | `-0.1325` | 0/3 | `0.1516 ± 0.0156` | `+0.1798` | 3/3 |

负 `R²` 表示 held-out prediction 比 validation target 的常数均值基线更差。
T1/T2 的 EEG null margin 比 T0 更负，说明增加 teacher objective 后不仅
没有越过 null，反而扩大了与 null 阈值的距离。fNIRS 的正 margin 在 T0
已经存在，因此不能归因于 T1/T2 语义监督。

### 注册的 seed-matched 增益

| Candidate | Seed 20260719 | Seed 20260720 | Seed 20260721 | Direction pass |
| --- | ---: | ---: | ---: | --- |
| T1 − T0 | `-0.0271` | `-0.0413` | `+0.0065` | Fail |
| T2 − T0 | `-0.0343` | `-0.0560` | `-0.0324` | Fail |
| T2 − T1 required | `-0.0072` | `-0.0147` | `-0.0390` | Fail |

T1 只有一个 seed 的 pooled endpoint 略为正，且该变化不能同时解决 EEG
null failure。T2 在三 seed 中均低于 T0，也在三 seed 中均低于 T1。

## 表示层级与坐标级失效

![Representation and target support](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig02_representation_and_support.png)

图 2A 比较 hard ID、continuous latent、posterior 与 saved codebook vector；
图 2B 比较 required 坐标相对 T0 的变化；图 2C 显示信号掩码后的实际目标
支持；图 2D 仅作诊断性对照，E0 target-space quantizability 与 E2 token
decoding 的估计问题不同，不能作为同一主终点直接做显著性比较。

### 1. EEG 失败发生在量化之前

| Row | Hard ID | Continuous latent | Posterior | Codebook vector |
| --- | ---: | ---: | ---: | ---: |
| T0 EEG | `-0.109` | `-0.016` | `-0.020` | `-0.020` |
| T1 EEG | `-0.163` | `-0.075` | `-0.099` | `-0.060` |
| T2 EEG | `-0.201` | `-0.035` | `-0.053` | `-0.048` |

如果只有 hard ID 为负、continuous latent 为正，可以把问题主要归因于
K、assignment 或 one-hot probe；当前不是这种情况。T1/T2 continuous 与
posterior 同样为负，说明 encoder/state target 的跨被试映射尚未建立，
量化只是进一步暴露而不是独立制造这一失败。

### 2. EEG 坐标内存在不同的失效模式

| Candidate | `r_mean` mean ΔR² | Positive seeds | `r_slope` mean ΔR² | Positive seeds |
| --- | ---: | ---: | ---: | ---: |
| T1 − T0 | `+0.0346` | 2/3 | `-0.1414` | 0/3 |
| T2 − T0 | `-0.1192` | 0/3 | `-0.0636` | 1/3 |

T1 的主要 EEG 损失集中在 `r_slope`；增加 optional `s` 后，T2 又显著损害
`r_mean`。这说明当前统一 state/prototype 权重并没有在 required 坐标之间
形成兼容梯度，T2 的更宽 target family 还产生了负迁移。

### 3. fNIRS 的改进是局部且不可组成 row-level 证据

T1 对 HbO/HbR slope 的 seed-mean 增益为 `+0.0209/+0.0325`，但 HbR mean
为 `-0.0063`；T2 对 HbR slope 为 `+0.0553`，但 HbO/HbR mean 为
`-0.0099/-0.0175`。这些坐标变化没有形成所有 seed、所有 required
模态的稳定改善。与此同时，fNIRS prototype matched cosine 从 T0 的
`0.9318` 降到 T1/T2 的 `0.9226/0.9247`，直接违反 prototype
non-decrease 条件。

## 训练动态与泛化

![Semantic training dynamics](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig03_semantic_training_dynamics.png)

| Row/objective | Train relative reduction | Validation relative reduction | Final validation − train |
| --- | ---: | ---: | ---: |
| T1 EEG state | `50.5%` | `0.7%` | `0.6350` |
| T2 EEG state | `39.7%` | `-3.0%` | `0.6306` |
| T1 EEG prototype | `5.4%` | `0.8%` | `0.1062` |
| T2 EEG prototype | `2.4%` | `1.1%` | `0.1807` |
| T1 fNIRS state | `8.4%` | `4.4%` | `0.0839` |
| T2 fNIRS state | `8.4%` | `4.5%` | `0.0837` |
| T1 fNIRS prototype | `2.2%` | `3.9%` | `0.0842` |
| T2 fNIRS prototype | `2.2%` | `3.9%` | `0.0840` |

EEG state head 明显拟合了训练目标，但这种拟合没有迁移到 held-out subjects。
EEG prototype loss 几乎没有被优化，T2 的 prototype generalization gap 又
大于 T1。fNIRS 的 train/validation 曲线更接近，但变化幅度小，符合
“已有基础可解码性、没有 teacher-row 增量”的解释。

## 目标支持、码本容量与 prototype 稀疏性

sidecar 的 `230/230` 是“target trial 是否存在”的覆盖率，不等于每个训练
sample 或每个 patch 都对 semantic loss 提供有效监督。正式 run 的
sample-level coverage 为：

| Split | Mean | Range |
| --- | ---: | ---: |
| Train | `16.68%` | `16.19%–16.95%` |
| Validation | `16.67%` | 固定 `16.67%` |

在 target trial 内进一步应用信号有效性掩码后（下表为 T1；T2 的 token
count 相同、support 分布接近）：

| Modality/split | Valid target patches | Fraction | Active signature codes | Median patches / active code | Active codes with ≤2 patches |
| --- | ---: | ---: | ---: | ---: | ---: |
| EEG train | 797 / 1800 | `44.3%` | 118.7 | 5.7 | `14.6%` |
| EEG validation | 178 / 500 | `35.6%` | 80.7 | 2.0 | `70.6%` |
| fNIRS train | 1800 / 1800 | `100%` | 106.7 | 6.3 | `20.0%` |
| fNIRS validation | 500 / 500 | `100%` | 85.0 | 3.0 | `44.4%` |

K=128 并未发生 global collapse，但 EEG validation 中每个活跃 signature
code 的中位支持只有 2 patches，且 70.6% 活跃 code 最多只有 2 patches。
这会同时放大：

- one-hot ridge probe 的方差；
- prototype signature 的小样本噪声；
- train/validation code support mismatch；
- 被试特异 artifact mask 对语义估计的影响。

因此，“有效码字更多”不是语义更强的证据。T1 EEG effective codes 从 T0
约 69 增到约 90，但 hard-token `R²` 反而更负。

## 梯度校准与目标权重

![Calibration and quantizer health](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig04_calibration_and_quantizer_health.png)

`0.005` 权重确实按预注册 train-only 规则入选；这一步没有使用 validation
target decoding。问题是同一标量权重下四类梯度仍处于 admission band 的
不同位置：

| Objective | Median gradient ratio vs same-modality reconstruction |
| --- | ---: |
| EEG state | `9.49` |
| EEG prototype | `1.87` |
| fNIRS state | `0.80` |
| fNIRS prototype | `0.20` |

这不是契约违规，但意味着 EEG state 接近上界、fNIRS prototype 接近下界。
一个统一权重同时服务两模态、state 与 prototype、以及 T2 新增 `s`
坐标，实际是较强的多目标折中。当前结果不能区分：

- target 本身不可跨被试泛化；
- EEG state 梯度过强导致训练拟合；
- prototype 梯度不足；
- 不同坐标间梯度冲突；
- 稀疏 target sampling 使四步梯度校准不能代表完整训练。

后续若调整权重，必须另立 E2.1 预注册开发实验；不能在读取本轮 validation
结果后直接把权重调优结果并入本轮 E2。

## T2 optional `s` 约束

注册的 pooled optional `s` 变化为：

`+0.0128 / -0.0070 / -0.0041`

只有 1/3 seed 为正。同时 T2 相对 T1 的 required endpoint 为：

`-0.0072 / -0.0147 / -0.0390`

因此，即使 subject-equal 的 optional 汇总给出正均值，T2 仍同时违反
“optional seed-consistent positive”与“required non-decrease”两条条件。
optional `s` 不能挽救 T2。

## 继承自 E0 的生理有效性

E2 使用的是已经通过 local observability、gauge 和 target-space
transmissibility 的 observation-aligned local/prototype family。经符号
校准后，完整 E0 已通过，adaptive SSM physical teacher 的生理信息完全
可接受：

| E0 evidence | Result | E2 implication |
| --- | ---: | --- |
| EEG physical reconstruction gain | `+0.7967` | Pass |
| fNIRS historical gain | `-0.0778` | 符号校准前诊断；不参与 E0 判定 |
| fNIRS sign-calibrated physical content | accepted | Pass |
| HbR historical coverage label | `0.950` | 校准前诊断；不再阻断 physical teacher |
| Complete E0 | Pass | SSM 生理信息完全可接受 |
| E2 local/prototype entry | enabled | 本轮注册实验范围 |
| E2 context/coupling entry | disabled | 本轮未评估，不影响 E0 |

E0 target-space K=128 reconstruction 很高（EEG `0.7344`、fNIRS `0.8791`），
但 E2 实际 hard IDs 最好仅为 EEG `-0.1092`、fNIRS `0.1542`。这表明
“目标几何可以被有限词表划分”没有转化为“从 measured signal 学到的 token
实现该划分”。两者估计口径不同，不能直接相减当作 effect size，但这个
realization gap 是当前最重要的机制性诊断。

## 统计口径复核

原 decision 同时使用了两个不同 estimand：

1. run-level primary delta：先在模态内对 pooled validation tokens/coordinates
   计算 mean `R²`，再对 EEG 与 fNIRS 等权；
2. subject bootstrap：把 2 个 EEG 与 4 个 fNIRS 坐标直接合并，使 fNIRS
   获得 2 倍坐标权重。

此外，原 bootstrap 在每个 seed 内独立重采样同一批被试，没有在一次 draw
中保持跨 seed 的 subject cluster。新增敏感性分析把 5 名被试作为独立
生物学单位，在每个 bootstrap draw 中对同一 subject index 同时保留三个
固定 seed，并枚举全部 `5^5=3125` 个 n-out-of-n draws。

| Comparison | Estimand | Mean ΔR² | Exact cluster 95% CI | Positive subjects |
| --- | --- | ---: | ---: | ---: |
| T1 − T0 | 原坐标权重 | `-0.0326` | `[-0.0989, 0.0091]` | 3/5 |
| T1 − T0 | 两模态等权 | `-0.0578` | `[-0.1573, 0.0062]` | 2/5 |
| T2 − T0 | 原坐标权重 | `-0.0575` | `[-0.1470, 0.0060]` | 2/5 |
| T2 − T0 | 两模态等权 | `-0.0926` | `[-0.2343, 0.0082]` | 2/5 |

由于只有 5 名 validation subjects，本报告不追加依赖大样本假设的 p-value，
也不把 seed 当作独立生物学重复。区间跨零不等于 T1/T2 可接纳：注册门槛
要求 seed-consistent improvement、两模态均高于 null 和 prototype
non-decrease；这些条件都明确失败。

## 当前 E2 未达到预期的生理信号约束

按优先级归纳如下：

| 优先级 | 未达到的约束 | 直接证据 | 当前判断 |
| --- | --- | --- | --- |
| P0 | EEG required state 应跨被试可解码 | 所有表示层均为负，9/9 低于 null | 核心失败 |
| P0 | 语义目标应在 validation 同步下降 | EEG state train 大降、validation 不降 | 强泛化失败 |
| P0 | 信号有效 patch 应充分支撑语义监督 | EEG train/val 仅 44.3%/35.6% target patches | 监督支持不足 |
| P1 | Required 坐标不应互相牺牲 | T1 `r_slope -0.1414`；T2 `r_mean -0.1192` | 坐标冲突/负迁移 |
| P1 | fNIRS 语义监督应有增量 | T0 已 above-null；T1/T2 无稳定增益 | 非新增证据 |
| P1 | Prototype stability 不下降 | fNIRS T1/T2 均低于 T0 | 门禁失败 |
| P1 | Optional `s` 应稳定且不伤 required | 仅 1/3 seed positive；required 3/3 下降 | T2 失败 |
| P1 | Teacher 生理物理层应可靠 | 符号校准后完整 E0 通过 | 已满足 |
| P2 | K=128 code support 应足够 | EEG val median 2 patches/code | 估计高方差 |
| P2 | 不确定性应可用于降噪 | physical teacher 已接纳；本轮未单独评估 weighting 增益 | 留给独立消融 |
| P3 | Context/coupling 约束可进入 | E2 本轮明确关闭 | 不可用本轮结果作耦合解释 |

## 建议的下一轮顺序

### 1. 先修正评估口径，不重写本轮结论

- 将 decision tool 的 subject uncertainty 改为与 primary endpoint 一致的
  模态等权 estimand；
- 以 subject 为跨 seed cluster，seed 仅作固定初始化重复；
- 同时保留 token-pooled、subject-equal 两种摘要并明确命名；
- 增加 leave-one-subject-out 结果，避免单个 subject 驱动结论；
- 不回写本轮注册 decision，只把修订用于后续 E2.1。

### 2. 先解决 EEG target-support，再调模型

- 逐 subject、trial、patch 导出 EEG analysis-valid mask、target mask 和
  required coordinate support；
- 检查 `-5 s` 至 `+15 s` window 与 artifact-clean mask 是否系统性删除
  与 `r/r_slope` 最相关的 patch；
- 对比 target-present 与 target-absent window 的数量、任务位置和信号质量；
- 在 train-only 范围预注册 target-rich sampler 或最小有效 patch 数规则，
  并让 T0/T1/T2 使用完全相同样本；
- 在正式 E2.1 前冻结 exclusion/sampling 规则，避免 validation-driven
  patch 筛选。

### 3. 设置“量化前语义上界”作为新前置门

在再次比较 K 或 revival 之前，要求 EEG continuous latent 或一个明确的
pre-quantization state head：

- held-out subject `R² > 0`；
- 高于 train-target permutation q95；
- required coordinates 至少方向一致；
- 在 3 个 seed 中不由单个 seed 驱动。

当前 continuous latent 仍为负，因此直接调整 K=128、temperature 或
codebook revival 优先级较低。

### 4. 分离目标与模态的权重校准

在不读取 validation decoding 的前提下，用 train-only 多 seed 梯度审计
考虑：

- EEG state、EEG prototype、fNIRS state、fNIRS prototype 分开权重；
- required `r_mean/r_slope` 与 optional `s` 分阶段加入；
- 记录梯度 cosine conflict，而不只记录 norm ratio；
- 将四步校准扩展到 quantization strength 已达到 1.0 的 retention window；
- 预先规定候选和 tie-break，作为新的 E2.1，而不是本轮 post-hoc 修补。

### 5. 只有在 continuous gate 通过后再处理离散容量

届时可以比较 K=32/64/128 或层级词表，但应增加：

- 每个 active code 的最小 target-support；
- validation unseen/rare code 比例；
- subject-balanced prototype stability；
- 同等 bitrate 下的 hard-token probe；
- 对 K 的选择不得读取保护测试。

### 6. fNIRS 目标继续受 E0 claim boundary 限制

若目标是“生理物理状态”而不仅是 observation-aligned auxiliary target，
应先修复 fNIRS physical reconstruction 和 HbR posterior calibration。
在此之前：

- 保持 uncertainty weighting、context、coupling 关闭；
- 不把 fNIRS above-null hard IDs 解释为 neurovascular coupling；
- 不进入 G3、E6/G2 或 E7 promotion。

## 最终判断

`retain_T0` 是稳健且保守的决定。当前 E2 的主要矛盾不是码本健康，而是
EEG 生理目标在稀疏、被掩码的监督支持下只在训练集被拟合，未形成跨被试
连续表示，更没有形成 hard-token 语义；fNIRS 则有基础可解码性，但没有
teacher-row 增量，且 prototype stability 与上游物理有效性仍受限。

因此下一轮最有价值的工作不是直接继续调 K 或在当前 validation 上搜索
semantic weight，而是先完成 EEG mask/target-support 审计、修正 subject
cluster 统计口径，并建立量化前的跨被试语义门。保护测试应继续关闭。

## 产物索引

- 机器摘要：
  [`review_summary.json`](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/review_summary.json)
- 主终点复核：
  [`fig01 PNG`](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig01_primary_endpoint_reaudit.png) /
  [`PDF`](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig01_primary_endpoint_reaudit.pdf)
- 表示与目标支持：
  [`fig02 PNG`](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig02_representation_and_support.png) /
  [`PDF`](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig02_representation_and_support.pdf)
- 训练动态：
  [`fig03 PNG`](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig03_semantic_training_dynamics.png) /
  [`PDF`](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig03_semantic_training_dynamics.pdf)
- 梯度与量化器：
  [`fig04 PNG`](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig04_calibration_and_quantizer_health.png) /
  [`PDF`](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/figures/fig04_calibration_and_quantizer_health.pdf)
- 数据表目录：
  [`tables/`](../../../experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/20260723_e2_v4_semantic_objective_suite_v1/comprehensive_review_20260724/tables)
