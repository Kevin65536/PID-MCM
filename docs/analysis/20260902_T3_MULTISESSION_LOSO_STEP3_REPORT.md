# T3 实验第三步：三 session 留一 session 与名义恢复期详细报告

_实验日期：2026-09-02；报告状态：完整 exploratory fit-only experiment_

## 结论

第三步已完成，但“一个稳定的被试级适应坐标可以跨 session 改善恢复期预测”这一方向性
假设在本合同下**失败**。正式 v2 运行将每名被试的两个 session 用于拟合，把第三个
session 整段留出，并循环三次。主要比较为
`M1_kappa_session_nuisance − M0_fixed`；负值才有利于候选模型。18 名被试等权的三折
平均 ΔNLL 为 **+5.288570**，95% subject-block bootstrap CI
**[+1.319282, +9.417574]**，中位数为 **+5.329969**。三折中位数也全部为正。因此
预设判定是 `exploratory_directional_failure`，科学结论是
`no_conclusive_cross_session_support_for_subject_trait`。

共享 κ 模型相对 M0 更差：ΔNLL **+6.621874**，95% CI
**[+3.016225, +10.634917]**。训练 session 各自拟合有效 κ、再以其几何中心应用到留出
session，确实比单个共享 κ 略好：`nuisance − shared` 为 **−1.333304**，95% CI
**[−2.763149, −0.038879]**；但这只收回了部分损失，最终候选的方向性得分仍低于 M0。它最多
说明允许这种 session-dependent 拟合形式相对 shared 模型带来探索性分数改善，不能证明
数据中存在独立的 session 生理成分，更不能证明其来源是紧张、疲劳、心率或呼吸。

参数稳定性 screen 也失败：只有 6/18 名被试达到注册的跨折 log-range 阈值，7/18 名
被试至少有一个有效 κ 接近数值边界。由于各折的 EEG PCA、fNIRS 通道、`P0/Q0` 和噪声
尺度不同，这个 κ screen 只是 fold-dependent operational diagnostic，不能当作同一物理量
的跨 session 方差估计。当前结果不支持把 κ 称为血管衰减率、健康 trait 或 teacher 标签，
也不开放 validation/protected，不构成 physical-teacher 资格或 tokenizer promotion 证据。

正式证据由 v2 [manifest](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/manifest.json)、
[summary](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/summary.json)
和逐层 CSV 共同支持；manifest 是运行状态、边界、输入身份、split、行数与工件哈希的唯一
owner。v1 是工程失败记录，不参与任何科学统计。

## 1. 运行身份与问题定义

| 字段 | 冻结值 |
| --- | --- |
| 正式 suite/run | `physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2` |
| schema | `t3_multisession_loso_v1` |
| 范围 | `fit_only_multisession_measured_exploratory` |
| 状态 | `status=exploratory_complete`; `run_state=complete`; `completion_status=complete`; `stage=complete` |
| UTC 时间 | 2026-09-02 15:01:28.411 至 15:36:21.655 |
| Asia/Shanghai 时间 | 2026-09-02 23:01:28.411 至 23:36:21.655 |
| 墙钟时长 | 2093.244 秒，即 34 分 53.2 秒 |
| 主 estimand | 整 session 留出、目标遮挡的恢复段 HbO/HbR variance-matched Gaussian NLL |
| 候选 / 参考 | `M1_kappa_session_nuisance` / `M0_fixed` |
| 差值方向 | candidate − M0；负值有利于候选 |
| 推断单位 | subject；三折先在被试内平均，再对 18 名被试等权聚合 |
| bootstrap | 10,000 次 percentile subject-block bootstrap；95%；seed `20260902` |
| 资格边界 | `qualification_eligible=false`; `decision_eligibility=false` |

证据：[resolved config](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/resolved_config.yaml)、
[manifest](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/manifest.json)、
[summary](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/summary.json)。

本实验回答的是：在第二步已经不支持联合恢复原始 `beta/kappa/tau` 的前提下，只保留一个
受限的 effective κ 适应坐标，是否能在三个 session 的整段留出恢复期上优于固定 M0，
同时保持参数和推断 driver 的操作性稳定。它不是临床 trait 研究，也不是新队列确认实验。

## 2. 冻结模型与拟合合同

### 2.1 三个比较模型

| 模型 | 每名被试、每折的处理 | 留出 session 使用值 |
| --- | --- | --- |
| `M0_fixed` | κ 固定为 0.64 | 0.64 |
| `M1_kappa_shared` | 两个训练 session 的 20 个 trial 共同拟合一个 κ | 同一共享 κ |
| `M1_kappa_session_nuisance` | 两个训练 session 各拟合一个 effective κ；其 log 均值定义 subject center，两个 log 偏差严格零和 | 两个训练 κ 的几何中心；未知留出偏差固定为 0 |

`beta=1.0`、`tau=2.0`、`gamma=0.32`、`alpha=0.32` 和 `E0=0.32` 全部固定；第三步
没有恢复第二步失败的三参数联合拟合。κ 的范围是 `[0.20, 1.50]`，注册中心 0.64、
尺度 0.20。共享拟合和每个 session
拟合的目标均为训练 predictive NLL 加 κ penalty，因此应称为正则化或 MAP-like 拟合，
不是纯 likelihood 参数恢复。session-specific 拟合没有额外的 session-effect penalty。

每个 session 拟合先用 9 点 transformed grid，再做有界标量 refinement，最多 35 次迭代。
三折共生成 162 个 optimizer 行：54 个共享 κ 拟合和 108 个单训练-session 拟合。162/162
均成功，训练物理布尔失败为 0。两个 log session 偏差的零和误差最大绝对值为
`1.11e-16`；留出参数拟合调用为 0。

证据：[config](../../experiments/configs/physiology_semantic_tokenizer/t3_multisession_loso_v1.yaml)、
[runner](../../experiments/evaluate_t3_multisession_loso.py)、
[optimizer diagnostics](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/optimizer_diagnostics.csv)、
[parameter estimates](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/parameter_estimates.csv)。

### 2.2 每折防泄漏操作

每折的 active HbO/HbR 通道、EEG PCA loading 与尺度、`P0/Q0` observation gauge、
`P0/Q0` 相关尺度和观测噪声均只由两个训练 session 决定。留出 session 的通道选择、
calibration、参数拟合和 warm-state 调用均为 0。留出窗口的 fNIRS 在任务开始 `t=0` 至
窗口末端 `t=25 s` 全部遮挡；只有 `[-5,0)` baseline fNIRS 和完整 EEG 进入 smoother。
主分数仅在 `[10,25)` 的 150 个 fNIRS 点上计算。

这是 fixed-interval target-masked reconstruction：它可使用留出窗口中未来时刻的 EEG，
不是 causal forecast。上游 canonical loader 还会对每条完整记录做 median/MAD robust
standardization；该步骤早于本 LOSO runner，并会使用留出记录自身的全记录分布。因此本轮
证明的是预标准化记录上的 target-masked LOSO 泛化，不是无需目标记录上下文的部署预测。

主分数用 smoother 的 `total_variance` 构造逐点 Gaussian NLL，再按 10 trial × 2 chromophore
在 subject×fold 内平均。它是 variance-matched Gaussian approximation，不是精确的
Student-t observation noise 与 latent-state uncertainty 卷积。

证据：[fold calibration](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/fold_calibration.json)、
[held-out metrics](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/heldout_metrics.csv)、
[subject-fold metrics](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/subject_fold_metrics.csv)、
[canonical loader](../../src/data/unified_physiology.py)。

## 3. 数据边界与恢复窗口

### 3.1 数据身份和实际计数

- 数据集与任务：`eeg_fnirs_single_trial`、MA only。
- 被试：只物化 `subject_01–18`；共 18 名。
- cache record：`session_01/03/05`；它们分别对应源 metadata 的 zero-based
  `session_idx=1/3/5`，不是把索引重写为连续的第 1/2/3 次记录。
- 每名被试、每个 record 有 10 个 MA trial，共 54 条记录、540 个唯一目标 trial。
- 原事件表在这 54 条记录中共有 1080 个事件，即 MA 与非目标事件合计；非 MA 不进入拟合。
- 每折 360 个拟合 trial、180 个整 session 留出 trial；三折累计 1080 次拟合使用和
  540 次留出评分。每个唯一 trial 恰好在两折中拟合、在一折中留出。
- `trial_inventory.csv` 共 1620 行；逐折拟合与留出 trial identity 交集为 0。

共享索引可以枚举全局 registry metadata，但数组访问只发生在获准的 01–18、三个指定
record。`subject_19–23` validation 与 `subject_24–29` protected 的数组访问计数均为 0，
对应 `enabled/opened` 也均为 false；CSV 中没有 19–29 的被试行。

证据：[metadata boundary](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/metadata_boundary.csv)、
[trial inventory](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/trial_inventory.csv)、
[manifest split proof](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/manifest.json)。

### 3.2 “完整恢复期”的实际实现边界

共同安全窗口为事件相对 `[-5,+25) s`：EEG 为 6000 点（200 Hz），fNIRS 为 300 点
（10 Hz）。名义任务区间是 `[0,+10) s`，主恢复评分区间是 `[+10,+25) s`，因此比第二步
`+15 s` 的窗口多取得 10 秒 post-task 信息。

54/54 个 alignment report 均通过。下一事件的最小剩余裕量为 EEG 491 点/2.455 秒、
fNIRS 24 点/2.4 秒，均超过注册的 0.5 秒 guard；记录末端最小裕量为 EEG 5.0 秒、
fNIRS 66.3 秒。24 个目标 MA event 是其记录的最后一个 event，没有下一事件可作休息终点。

事件索引的 `duration` 为空，无法知道每个 trial 的精确 rest end。因此本报告把
`[+10,+25)` 严格称为 **15 秒名义恢复包络**：它是在所有记录上共同安全、未跨下一事件的
恢复窗，但不能声称已根据原始事件标注证明“完整休息段”恰好结束于 +25 秒。

证据：[summary metadata boundary](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/summary.json)、
[metadata rows](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/metadata_boundary.csv)。

## 4. 合成软件前置检查

合成预检在任何 measured metadata 或数组读取前运行并通过，用时 57.702 秒。它使用 1 名
合成被试、3 个 session、每 session 2 个独立 reset trial；真值不传给拟合器。训练
`session_01/03`、留出 `session_05` 时：

| 项目 | κ |
| --- | ---: |
| truth session_01 | 0.781698 |
| truth session_03 | 0.523988 |
| truth session_05 | 0.640000 |
| estimated session_01 | 0.840843 |
| estimated session_03 | 0.560327 |
| estimated geometric center | 0.686401 |
| estimated shared κ | 0.687756 |

三个拟合目标均不劣于注册 prior 起点，优化器均成功，物理失败为 0，恢复分数有限。
合成 M0 和候选的 Gaussian NLL 分别为 −2.646157 和 −2.620921；候选并未优于 M0，
但预检从未把预测优越性作为软件 gate。该单案例只验证代码路径、遮挡、零和分解和有限性，
不是 simulation-based calibration、参数 recovery 资格或 measured practical margin。

证据：[synthetic preflight](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/synthetic_preflight.json)。

## 5. 三个 LOSO fold

下表的 CI 均以 18 名被试为 bootstrap 单位；负 ΔNLL 才有利于候选。

| 留出 session | 训练 sessions | 拟合/留出 trial | 候选−M0 mean | 95% CI | 中位数 | 候选较好被试 | optimizer fail | session κ 边界被试 | 用时 |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `session_01` | `03 + 05` | 360 / 180 | +5.029604 | [+1.284746, +8.776304] | +3.918269 | 5/18 | 0 | 0/18 | 624.64 s |
| `session_03` | `01 + 05` | 360 / 180 | +5.769328 | [+2.083368, +9.325536] | +4.632540 | 3/18 | 0 | 5/18 | 650.34 s |
| `session_05` | `01 + 03` | 360 / 180 | +5.066778 | [−1.686790, +13.198234] | +2.010402 | 7/18 | 0 | 2/18 | 679.56 s |

第 1、2 折的 CI 完全在 0 以上，明确不利于候选；第 3 折区间跨 0，单折是不确定，但
点估计和中位数仍不利于候选。三折中位数全部为正，所以不满足“每折中位数均小于 0”的
方向性支持条件。三个折的训练物理布尔失败总数均为 0。

拆分模型比较为：

| 留出 session | shared−M0 mean [95% CI] | nuisance−shared mean [95% CI] |
| --- | --- | --- |
| `session_01` | +6.292497 [+3.114604, +9.511130] | −1.262893 [−3.473700, +0.704763] |
| `session_03` | +6.504075 [+3.648657, +9.615236] | −0.734747 [−3.089842, +0.900150] |
| `session_05` | +7.069049 [−0.739381, +16.126006] | −2.002271 [−4.316077, +0.319105] |

折级证据：[session_01 report](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/fold_session_01_report.json)、
[session_03 report](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/fold_session_03_report.json)、
[session_05 report](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/fold_session_05_report.json)、
[fold table](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/fold_summary.csv)。

## 6. 跨折主要与次要结果

在每名被试内先平均三个 fold，再对被试等权聚合。恢复段绝对 NLL 的 54 个
subject×fold 等权均值为：M0 67.653882、共享 κ 74.275756、候选 72.942452。

| 比较 | 被试等权 mean | 95% CI | 中位数 | 解释 |
| --- | ---: | --- | ---: | --- |
| 候选 − M0 | +5.288570 | [+1.319282, +9.417574] | +5.329969 | `exploratory_directional_failure` |
| shared − M0 | +6.621874 | [+3.016225, +10.634917] | +6.432246 | CI 在 0 以上，得分不及固定 M0 |
| nuisance − shared | −1.333304 | [−2.763149, −0.038879] | −0.570319 | session-specific 训练拟合有探索性改善，但不足以追上 M0 |

候选−M0 的三折平均在 4/18 名被试为负、14/18 为正。CI 下界已高于 0，故根据冻结规则
归类为方向性失败，而不是“无显著差异”或“模型等价”。同时，本合同没有由独立 synthetic
technical repeats 冻结 practical margin；即使结果方向相反，也最多只能称 exploratory
support，不能升级为正式确认。

证据：[subject summaries](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/subject_summary.csv)、
[subject-fold metrics](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/subject_fold_metrics.csv)、
[summary primary result](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/summary.json)。

## 7. 被试级结果

`κ center` 是三折 session-center κ 的几何均值；`log-range/span` 是三折 center log-κ
极差除以注册 log 边界跨度。这里的“稳定”只指 `range/span ≤ 0.10` 的 operational screen。

| 被试 | 候选−M0 | shared−M0 | nuisance−shared | κ center | log-range/span | κ稳定 | 任一κ边界 |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| subject_01 | -5.45333 | -0.32649 | -5.12684 | 0.571100 | 0.0784 | 是 | 否 |
| subject_02 | 3.04886 | 4.74379 | -1.69493 | 0.769727 | 0.2168 | 否 | 否 |
| subject_03 | 9.07773 | 8.25181 | 0.82592 | 0.763306 | 0.3057 | 否 | 否 |
| subject_04 | 9.62614 | 8.21885 | 1.40729 | 0.882471 | 0.2176 | 否 | 是 |
| subject_05 | 10.50190 | 11.09449 | -0.59259 | 0.949448 | 0.3610 | 否 | 是 |
| subject_06 | 18.60694 | 18.45853 | 0.14841 | 1.022430 | 0.0117 | 是 | 否 |
| subject_07 | 0.07896 | 1.02620 | -0.94724 | 0.656350 | 0.1216 | 否 | 否 |
| subject_08 | 10.25859 | 9.08625 | 1.17234 | 0.820252 | 0.1139 | 否 | 否 |
| subject_09 | 3.15649 | 3.29119 | -0.13469 | 0.753080 | 0.3416 | 否 | 是 |
| subject_10 | 8.61034 | 11.36438 | -2.75404 | 0.949167 | 0.2020 | 否 | 否 |
| subject_11 | 27.27391 | 30.82605 | -3.55214 | 1.231926 | 0.0823 | 是 | 是 |
| subject_12 | -5.34746 | -9.06521 | 3.71775 | 0.555336 | 0.4822 | 否 | 是 |
| subject_13 | 0.12341 | 7.30869 | -7.18528 | 0.670540 | 0.3606 | 否 | 否 |
| subject_14 | -2.05937 | 1.29506 | -3.35443 | 0.614834 | 0.1032 | 否 | 否 |
| subject_15 | 7.50344 | 5.55581 | 1.94764 | 0.805546 | 0.0759 | 是 | 否 |
| subject_16 | -9.52825 | -1.90162 | -7.62663 | 0.450103 | 0.0205 | 是 | 是 |
| subject_17 | 8.74060 | 9.28865 | -0.54805 | 0.949812 | 0.2913 | 否 | 是 |
| subject_18 | 0.97536 | 0.67732 | 0.29804 | 0.671473 | 0.0484 | 是 | 否 |

被试级差异很大，例如 subject_16 的候选−M0 为 −9.53，而 subject_11 为 +27.27；这不应
被重新解释为临床亚型。当前是同一历史 fit-only cohort、没有人口学分层、没有新确认队列，
且每折的 observation gauge 并不相同。

## 8. 参数与 driver 稳定性

### 8.1 κ operational screen

注册规则要求至少 75% 被试的 `center log-range / registered log-span ≤ 0.10`，且任一有效
κ 边界被试比例不超过 25%。实际只有 6/18（33.3%）达到 range 条件，7/18（38.9%）
至少有一个 shared 或 session κ 进入边界带；总体 screen 为 `False`。三折 κ center 几何
均值的被试分布最小/中位/最大为 0.450103 / 0.766517 / 1.231926。

边界带定义为距 `[0.20,1.50]` 的 log-span 任一端不超过 1%。10 个 optimizer 边界行为：

| fold | 被试 | 边界拟合 |
| --- | --- | --- |
| holdout 03 | subject_04 | training session_05 κ = 1.500000 |
| holdout 03 | subject_05 | shared κ = 1.470476；training session_01 κ = 1.500000 |
| holdout 03 | subject_09 | training session_05 κ = 1.500000 |
| holdout 03 | subject_16 | training session_05 κ = 0.200000 |
| holdout 03 | subject_17 | training session_05 κ = 1.491728 |
| holdout 05 | subject_11 | shared κ = 1.484854；training session_01 κ = 1.500000 |
| holdout 05 | subject_12 | shared κ = 0.200000；training session_01 κ = 0.200000 |

这个失败不能单独证明“真实参数不稳定”。各折使用不同训练 session 重新选择通道并建立
PCA/`P0/Q0`/噪声 gauge，κ 又是带 prior 的 effective coordinate；跨折 center 不是已证明
同量纲的物理测量。因此最小结论只是：当前整套 fit-and-gauge pipeline 没有产生满足注册
稳定性 screen 的 κ 坐标。

### 8.2 r(t) screen

同一 fold、同一 held-out trial、同一 gauge 下比较候选与 M0 的 r(t)，1080 个
model-contrast 行的中位 NRMSE 为 `3.91e-05`、中位 correlation 为
`0.9999999993`；18/18 名被试均通过 NRMSE≤0.10、correlation≥0.95 的 screen。
这说明改变 κ 的这两个模型几乎没有改变同一次 smoother 中的 driver 解，但不是独立的
跨 session repeatability 证据，也可能反映共同 EEG 输入、先验和尺度对 driver 的主导。

另行比较三个 held-out session 各自 10 个 trial 的平均 r(t) 时，54 个 session-pair 行的
中位 NRMSE 为 1.005935、correlation 为 0.543056。每个 session 的 driver 来自不同折的
PCA 与 observation gauge，因此该比较明确 `gauge_invariant=false`，只能描述，不能据此
宣布 latent state 跨 session 不稳定或稳定。

证据：[driver stability](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/driver_stability.csv)、
[mean trajectories](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/driver_mean_trajectories.csv)、
[subject summaries](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/subject_summary.csv)。

## 9. Fold gauge 与通道差异

| 留出 session | active fNIRS pair | `P0` gauge | `Q0` gauge | EEG PC scale |
| --- | --- | ---: | ---: | ---: |
| session_01 | `FC3FC1` | 22.821893 | 7.987662 | 4.491670 |
| session_03 | `FC3FC5` | 28.942496 | 10.129874 | 4.488301 |
| session_05 | `FC3FC1` | 21.561968 | 7.546689 | 4.486479 |

三折均使用全部 30 个 scalp EEG 通道建立 fold-specific PCA，但 holdout 03 选择了不同的
fNIRS pair。每个主 ΔNLL 都是同一折、同一通道和 gauge 内的配对模型差值，因此候选−M0
方向仍可比较；跨折 pooled ΔNLL 则平均了不同 local pair 和不同 scale。它不能被解释为
同一固定脑区参数的三次重复测量，这也是为何本轮不支持 trait claim。

证据：[fold calibration](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/fold_calibration.json)。

## 10. Session nuisance 与辅助生理边界

本轮的 nuisance 只是两个训练 session 的 log effective-κ 零和偏差；未知留出 session 的
偏差固定为 0。当前 canonical API 没有向 runner 暴露 ECG 或 respiration，本地 EEG 路径
只提供 30 个 scalp channels；VEOG/HEOG 只在底层清理路径作为 auxiliary。因而：

- 不能把 nuisance 改善归因于心率、呼吸、紧张或疲劳；
- 不能检验 κ 偏差是否与 ECG/呼吸同步；
- 不能把全阵列低秩成分命名为 systemic physiology；
- 计划第五步仍需要一个单独、带 provenance 的 auxiliary 数据合同，不能由本结果替代。

本轮 nuisance 的因果归因字段为 false。它只是一个 generic session residual coordinate。

证据：[summary session nuisance](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/summary.json)、
[loader contracts in manifest](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/manifest.json)。

## 11. 工程失败记录与正式重跑

第一次正式尝试 `20260902_step3_multisession_loso_v1` 在完成 holdout-session_01 的 18 名
被试计算后，于折级 bootstrap 聚合处失败。根因是 NumPy 2.4 不能直接把
`dict_values` 转为 `float64`；错误为
`TypeError: float() argument must be a string or a real number, not 'dict_values'`。
该 run 的 manifest 正确标记：

| 字段 | v1 失败记录 |
| --- | --- |
| 状态 | `status=incomplete_failed`; `run_state=failure`; `completion_status=incomplete` |
| 失败位置 | `failed_after_stage=holdout_session_01_running` |
| 用时 | 757.075 秒 |
| 已物化拟合 trial | 540 |
| validation / protected 数组访问 | 0 / 0 |
| 部分行 | metadata 540；inventory 1620；parameters 18；optimizers 54；metrics 2160；subject-fold 108；drivers 360；trajectories 16200；fold summary 0 |

修复只发生在共享 bootstrap helper：先把 iterable 显式转成 list，再调用
`np.asarray(..., dtype=float64)`；回归测试直接传入 `dict_values`。v1 目录保持不可变，未
resume、未覆盖，也未把部分结果合入正式统计。修复后从头运行全三折，产生本报告唯一正式
证据 v2。

证据：[v1 failure manifest](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v1/manifest.json)、
[regression test](../../tests/test_t3_multisession_loso.py)。

## 12. 工件与独立审计

### 12.1 行数合同

| 工件 | data rows | 行单位 |
| --- | ---: | --- |
| `metadata_boundary.csv` | 540 | approved unique trial |
| `trial_inventory.csv` | 1620 | trial×fold use |
| `parameter_estimates.csv` | 54 | subject×fold |
| `optimizer_diagnostics.csv` | 162 | fit role×subject×fold |
| `heldout_metrics.csv` | 6480 | trial×model×mask×chromophore |
| `subject_fold_metrics.csv` | 324 | subject×fold×model×mask |
| `driver_stability.csv` | 1134 | driver comparison |
| `driver_mean_trajectories.csv` | 48600 | subject×session×model×timepoint |
| `fold_summary.csv` | 3 | fold |
| `subject_summary.csv` | 18 | subject |

18/18 个 manifest 声明工件全部存在，文件大小和 SHA-256 现场重算均匹配；9/9 个当前
源码/配置 SHA-256 与 manifest 匹配；25/25 个输入 hash 也匹配。关键身份是 runner
`1d3aa32c49bbe95e733c1b62af160c60c071de430cfc3915539b7463f6b4d679`、launch config
`d843335902e16c7de19e7464c5acae3341b5e8913a12ee26adc08bea39a3532a` 和 Balloon model
`4221b4a53e9b2041d6db5e0274e4d5b509ab8108291e004c01ab67d9054326e4`。

输入 hash 的声明范围是 7 个 cache/metadata owner 加 18 个 manifest-declared fit-subject
原始 `cnt.mat` 来源；它没有逐个完整哈希全部选中数组内容。这个限制已写入 manifest，
不能把 25/25 匹配扩大解释为全数组 content-addressed snapshot。

### 12.2 评分与 split 独立重算

只读审计从 `heldout_metrics.csv` 重新按 trial→subject×fold→subject 三层聚合，复现了
三个 fold、主要和两个次要比较的均值、中位数与 10,000 次 bootstrap CI；与 summary 的
差异仅为约 `1e-15` 的浮点末位。另核对：

- 6480/6480 个 NLL 有限；
- recovery 行均有 150 点，response 行均有 250 点；
- 目标 HbO/HbR 对 smoother 全部标记为未观察；
- held-out parameter fit 调用总数为 0；
- held-out 评分与训练物理失败总数均为 0；
- 324/324 个 subject-fold 支持行均 `support_valid=true`；
- 每个唯一 trial 恰有 2 个 fit use 和 1 个 held-out use；三折 overlap 为 0。

### 12.3 软件验证

- `.venv/bin/python -m pytest -q tests/test_t3_multisession_loso.py tests/test_t3_identifiability.py tests/test_t3_measured_reconstruction_null.py`：30 passed，4.14 秒。
- Step3 定向测试单独覆盖 closed-subject fail-closed、metadata-only 无数组读取、三折无交集、
  250/150 点 mask、零和 session 分解、留出目标遮挡/冻结 apply 和 `dict_values`
  bootstrap 回归。
- 本轮只声称运行了这些直接相关的 T3 测试，不声称执行全仓测试套件。

## 13. 限制

1. 01–18 是已参与历史开发的 fit-only cohort，不是新的确认队列；本轮不能给出独立的
   population-generalization 或临床 trait 结论。
2. 只有三个 cache records，而且 record/session 顺序可能与时间、熟悉、疲劳等因素混杂；
   没有随机化 session-level exposure 的因果设计。
3. 精确 trial/rest duration 不在事件索引中；`+10` 至 `+25 s` 只是共同安全的名义恢复
   包络，不是已标注的每 trial 完整 rest period。
4. κ 是第二步否定性结果后保留的一维 effective adaptation coordinate。它带 prior、受
   observation gauge 和模型失配影响，不能直接映射为真实血管衰减率。
5. 各折重新选择 fNIRS pair，并重建 EEG PCA、`P0/Q0` 和噪声尺度。折内模型差值仍是配对
   的，但 κ/r 的跨折绝对比较不具已证明的共同 gauge。
6. upstream canonical median/MAD normalization 使用完整记录，包括留出记录自身的分布；
   smoother 又使用未来 EEG。因此这是 target-masked fixed-interval analysis，而非严格
   causal/deployment forecast。
7. fNIRS 每折只使用一个 active HbO/HbR pair；没有保留多通道观测，也没有通道扰动、
   geometry permutation 或全阵列低秩 nuisance 检验。
8. canonical runner 没有 ECG/呼吸输入，不能区分 neural、systemic、motion 与心理状态来源。
9. within-fold r(t) screen 比较的是两个模型在同一 trial/gauge 下的结果，不是同一 latent
   coordinate 的跨 session 重测；跨 session 描述本身又是 gauge-dependent。
10. Gaussian NLL 是 variance-matched approximation；没有精确 Student-t-plus-state
    convolution、校准敏感性分析或本实验专属的 cross-modal null。
11. synthetic preflight 只有一个被试、两个 trial/session 和一组 truth；没有
    simulation-based calibration，也没有生成可冻结的 practical margin。
12. 输入身份可审计，但 selected array contents 没有形成逐数组完整哈希快照。

## 14. 决策与下一步边界

- 第三步到此完成；最小可辩护结论是：**在当前预标准化、fold-specific gauge 和三个
  session 的 target-masked LOSO 合同下，subject/session effective-κ 候选没有改善 M0，
  也没有产生满足注册稳定性 screen 的被试坐标。**
- `nuisance − shared` 的探索性改善可作为“session residual 值得继续建模”的动机，但不能
  被解释成已经分离出 ECG、呼吸、紧张或疲劳。
- 不把 κ 或当前 r(t) 写入 teacher/token label，不把被试级差异描述为血管健康分层，不因
  本轮结果开放 19–23 或 24–29。
- 若用户决定进入第四步，不能直接把当前 κ 放入 hierarchical partial pooling。最低前置
  条件应是：冻结一个共同可比的 observation/driver gauge 和固定 local channel endpoint，
  只测试一到两个由敏感度或综合参数化预先定义的 gain/time-scale 方向，并先建立独立的
  practical margin。
- ECG、呼吸和低秩全阵列 nuisance 仍属于计划第五步，需要新的数据 provenance 与边界
  合同；本轮没有替它提前执行。
- `research_state/registry.json`、teacher qualification、validation/protected authorization
  和 tokenizer promotion 状态均不因本实验改变。

第三步的负结果与第二步互相一致但不重复：第二步说明原始 `beta/kappa/tau` 不能被当前
观测可靠联合识别；第三步进一步说明，即使只保留一个受限 κ 适应坐标并增加三 session
及更长恢复窗，它在整 session 留出评分上仍不优于固定 M0。当前证据因此支持继续降低参数
解释强度、先统一 gauge 和 nuisance 观测，而不是扩大自由生理参数集合。
