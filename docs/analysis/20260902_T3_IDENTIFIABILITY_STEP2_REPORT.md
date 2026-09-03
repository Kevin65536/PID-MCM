# T3 实验第二步：fit-only 可辨识性详细报告

_实验日期：2026-09-02；报告状态：完整 exploratory diagnostic_

## 结论

第二步已完成，但预设的实际可辨识性假设**未得到支持**。主终点要求每个案例的
`beta`、`kappa`、`tau` 三条 profile 网格均完整，似然支持有限、连续且不接触注册
边界；合成案例和 3 个测量代表案例均为 `False`，因此总体结果为
`primary_practical_identifiability_hypothesis_not_supported`。

测量结果更直接：低残差代表 `subject_03` 的 `kappa/tau` profile 接触注册边界，
中位代表 `subject_13` 的 `kappa/tau` 接触边界，高残差代表 `subject_10` 的三个参数
全部接触边界。扩大边界后，后两例的似然明显降低，但预测轨迹不满足等价阈值，不能据此
证明“参数不同而状态相同”；它们按合同保留为 `inconclusive`。`subject_03` 和合成案例
存在彼此有实质参数差异、预测等价且驱动稳定的近优解，故只支持
`parameters_nonidentifiable_but_state_stable` 这一案例级描述。

这不是 teacher 资格证据。运行本身标记为 `qualification_eligible=false`、
`decision_eligibility=false`，不能覆盖已有 P0 失败，不能开放验证集或保护集，也不能推动
tokenizer promotion。以上结论由正式运行的
[manifest](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/manifest.json)
和 [summary](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/summary.json)
共同支持；manifest 是状态、输入哈希和工件清单的唯一 owner。

## 1. 运行身份与问题定义

| 字段 | 冻结值 |
| --- | --- |
| 正式 suite/run | `physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3` |
| schema | `t3_identifiability_v1` |
| 状态 | `exploratory_diagnostic_complete`; `completion_status=complete`; `stage=complete` |
| UTC 时间 | 2026-09-02 12:15:14.383 至 13:06:14.188 |
| Asia/Shanghai 时间 | 2026-09-02 20:15:14.383 至 21:06:14.188 |
| 墙钟时长 | 3059.804 秒，即 50 分 59.8 秒 |
| estimand | `fit_only_practical_identifiability_beta_kappa_tau_v1` |
| 假设 | `beta_kappa_tau_are_practically_identifiable_on_fit_only_T3a_observations` |
| operator | `likelihood_only_profile_with_companion_parameters_and_latent_states_reoptimized` |
| null | `no_cross_modal_null_operator_diagnostic_only`；本 suite 未运行 cross-modal null |
| Git 快照 | commit `3692b23ce0f4e092fed420370586f6450a32ea26`，dirty 状态逐项写入 manifest |

证据：[manifest](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/manifest.json)、
[resolved config](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/resolved_config.yaml)。
父运行 `t3_measured_reconstruction_null/20260828_subject_parameter_fit_v2` 仅作上下文；
manifest 明确记录 `arrays_consumed=false`，其 SHA-256 为
`960d5bab17159cbf26768b45dbe5b4c5fd829aa8a8f25e27bff1dc003c8296a1`。

## 2. 冻结方法与判定合同

本轮仅分析 diagnostic-only 的 M2：拟合 `beta/kappa/tau`，固定
`gamma=alpha=E0=0.32`。注册范围为 `beta=[0.25,4]`、
`kappa=[0.20,1.50]`、`tau=[0.50,5.00]`；优化坐标对正参数取 log。
每个案例包含：

1. 16 个确定性变换空间起点：注册范围以 prior 起点开头，扩边范围以参考解 warm start
   开头，二者各再加 15 个 seeded Latin-hypercube 起点；L-BFGS-B 最多 60 次迭代，
   `ftol=1e-7`。
2. 对每个参数使用 9 个网格点；每点固定目标参数，以 2 个起点重新优化另外两个参数。
   每个未缓存的唯一参数向量都会重新运行 smoother，因此候选 latent state 也被条件重估。
3. 在变换空间两端各扩大原 span 的 25% 后，再运行 16 个起点。实际扩边范围是
   `beta=[0.125,8]`、`kappa=[0.120855,2.482313]`、
   `tau=[0.281171,8.891397]`。
4. 对 `beta/kappa/tau/gamma/alpha/E0` 计算固定 driver 的条件前向白化 Jacobian
   与 SVD；相对奇异值阈值为 0.05，并另报 active `beta/kappa/tau` 子矩阵。

优化目标只含 predictive likelihood；CSV 中的 prior penalty 只作诊断，不参与选解。
profile likelihood 支持阈值为 `delta NLL <= 1.920729410347062`，profile 最小值与
无约束参考解的一致性容差为 0.01 NLL。参数边界判定带宽和“实质参数差异”阈值均为
注册变换 span 的 1%。预测等价要求 observation whitened RMSE 不超过 0.10；驱动稳定
要求 NRMSE 不超过 0.10 且相关系数不低于 0.95。

完整合同见
[t3_identifiability_v1.yaml](../../experiments/configs/physiology_semantic_tokenizer/t3_identifiability_v1.yaml)，
执行入口见
[evaluate_t3_identifiability.py](../../experiments/evaluate_t3_identifiability.py)。本轮未把
M2 恢复为 recommendation-eligible，也未运行 M3--M5。

## 3. 数据边界与代表选择

### 3.1 数据身份

- 数据集/条件：`eeg_fnirs_single_trial`、`single_trial_ma_session_01`、任务标签 `MA`。
- 每个窗口为相对事件 `-5 s` 至 `+15 s`，在 10 Hz 坐标上为 20 秒；要求完整有限支持。
- EEG 分支为 `raw_with_ocular_artifact`；当前坐标是 10 Hz log-power、fit-fold PCA
  proxy，不是 200 Hz 原始电位波形。
- fNIRS 使用规范化 HbO/HbR 坐标和 `FC3FC5_HbO/FC3FC5_HbR` 这一配对；
  `P0/Q0` 是 observation-space gauge，不是绝对浓度。
- 01--18 号被试的 `trial_position=4,9`（0-based）只作为内部留出标记，本轮既不进入
  gauge、M0 选择，也不进入 M2 分析。

证据：[data contracts](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/data_contracts.json)、
[calibration](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/calibration.json)、
[trial inventory](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/trial_inventory.csv)。

### 3.2 实际访问计数

| 数据动作 | 实际计数 |
| --- | ---: |
| 加载并物化的 fit trial | 180 = 18 名被试 × 10 trial |
| 参与 gauge 的 fit trial | 144 = 18 × 8 |
| 参与 M0 代表选择的 fit trial | 144 = 18 × 8 |
| 进入 M2 的测量 trial | 24 = 3 × 8 |
| 进入 M2 的测量被试 | 3 |
| 19--23 validation trial/数组加载 | 0 |
| 24--29 protected trial/数组加载 | 0 |

共享 loader 构建规范数据集索引元数据和 window references，可能枚举 fit-only 选择之外
的 registry record；但 19--23 和 24--29 的数组均未解引用、窗口样本均未物化。
`validation_data_opened=false`、`protected_data_opened=false`。计数由
[manifest](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/manifest.json)
和逐 trial 的 [inventory](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/trial_inventory.csv)
交叉核对。

### 3.3 低/中/高残差代表

选择指标为固定 M0 pooled predictive NLL / finite observation；排序使用 01--18 每人
8 个 fit trial。18 人完整排名见
[representative_selection.csv](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/representative_selection.csv)。

| 角色 | 被试 | 排名 | M0 NLL/观测 | event indices | finite observations |
| --- | --- | ---: | ---: | --- | ---: |
| low | `subject_03` | 1/18 | -0.4514167221 | 1, 3, 4, 7, 10, 13, 14, 17 | 4800 |
| median | `subject_13` | 9/18 | 0.3573731803 | 0, 3, 4, 7, 10, 12, 14, 17 | 4800 |
| high | `subject_10` | 18/18 | 2.3843962142 | 0, 3, 4, 7, 10, 13, 15, 16 | 4800 |

这些标签只是当前 fit-only M0 分数的代表位置，不是临床、生理或总体人群分层。

## 4. 合成已知真值结果

合成案例使用 replicate 0、seed `20260902`、prior-centre 真值和 P0 noisy clean
scenario。拟合参数的误差如下；后三个固定参数的零误差是按构造固定，并非被数据恢复。

| 参数 | 是否拟合 | 真值 | 估计 | 相对误差 |
| --- | --- | ---: | ---: | ---: |
| beta | 是 | 1.000000 | 1.22527784 | +22.5278% |
| kappa | 是 | 0.640000 | 0.61653988 | -3.66564% |
| tau | 是 | 2.000000 | 1.83936483 | -8.03176% |
| gamma | 否 | 0.320000 | 0.320000 | 0%（固定） |
| alpha | 否 | 0.320000 | 0.320000 | 0%（固定） |
| E0 | 否 | 0.320000 | 0.320000 | 0%（固定） |

证据：[parameter_recovery.csv](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/parameter_recovery.csv)。
本合同未预设 recovery pass tolerance，因此这些误差只作描述，不能单独构成通过判定。

## 5. Multistart 与 profile likelihood

### 5.1 优化收敛与参考解

| 案例 | 注册起点有效 | 扩边起点有效 | 参考 NLL | finite obs | beta | kappa | tau |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic | 16/16 | 14/16 | -2079.676515 | 1440 | 1.225278 | 0.616540 | 1.839365 |
| `subject_03` | 16/16 | 16/16 | -3416.298306 | 4800 | 1.805536 | 0.200000 | 5.000000 |
| `subject_13` | 16/16 | 16/16 | -1859.296450 | 4800 | 0.869041 | 0.200000 | 5.000000 |
| `subject_10` | 16/16 | 15/16 | 5339.471361 | 4800 | 0.250000 | 1.500000 | 5.000000 |

证据：[multistart_results.csv](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/multistart_results.csv)
及 [summary cases](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/summary.json)。
NLL 只在同一案例的候选解之间解释，不跨案例比较。

### 5.2 离散 profile 支持

下表报告 9 点网格上的似然支持集合范围，不把它解释为连续参数置信区间。

| 案例 | finite/profile 点 | beta 支持 | kappa 支持 | tau 支持 | 接触注册边界 | 主终点 |
| --- | ---: | --- | --- | --- | --- | --- |
| synthetic | 26/27 | [1.225278, 1.225278] | [0.616540, 0.704601] | [1.839365, 1.839365] | 无 | False：网格不完整 |
| `subject_03` | 27/27 | [1.805536, 1.805536] | [0.200000, 0.257284] | [5.000000, 5.000000] | kappa 下界、tau 上界 | False |
| `subject_13` | 27/27 | [0.869041, 0.869041] | [0.200000, 0.200000] | [5.000000, 5.000000] | kappa 下界、tau 上界 | False |
| `subject_10` | 27/27 | [0.250000, 0.250000] | [1.500000, 1.500000] | [5.000000, 5.000000] | beta 下界、kappa/tau 上界 | False |

四案的支持集合均连续，三个参数各自的 profile 最小值与无约束参考 NLL 差异也都小于
0.01；失败原因不是 reference mismatch。合成案例的 `beta=4.0` 上界网格点由优化器
返回 `success=True`，但 likelihood 为 `1e12` 数值哨兵，因此严格 finite gate 将其排除，
只计 26/27。对应 CSV 还把 `latent_state_reoptimized=True` 写成了 optimizer success 的
镜像；该字段不能证明这个哨兵点产生了有效 latent 解。这一记录问题不改变主终点，因为
finite gate 已经 fail closed。

证据：[profile_likelihood.csv](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/profile_likelihood.csv)
和 [summary profile support](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/summary.json)。

## 6. 扩大边界诊断

| 案例 | 原注册边界 | 扩边解越过原界 | 仍贴扩边外界 | expanded beta/kappa/tau | ΔNLL | obs white RMSE | driver NRMSE | driver corr |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| synthetic | 无 | 无 | 无 | 1.225278 / 0.616540 / 1.839365 | 0.000 | 5.95e-11 | 9.24e-11 | 1.000000 |
| `subject_03` | kappa, tau | tau | 无 | 2.191604 / 0.309761 / 6.843508 | -50.4969 | 0.07133 | 0.07974 | 0.996863 |
| `subject_13` | kappa, tau | kappa, tau | kappa, tau | 1.669751 / 0.120855 / 8.891397 | -1028.7642 | 0.30509 | 0.19496 | 0.984378 |
| `subject_10` | beta, kappa, tau | beta, kappa, tau | beta, tau | 0.125000 / 2.325388 / 8.891397 | -2812.9079 | 1.05949 | 0.17797 | 0.990585 |

证据：[expanded_bounds.csv](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/expanded_bounds.csv)。
负的 ΔNLL 表示扩边解有更低的 likelihood NLL。合成案例满足预测和 driver 稳定阈值；
在三个测量案例中，只有 `subject_03` 同时满足这些阈值。其扩边最优解已越过原 tau
上界，但没有继续贴住新外界。`subject_13` 与 `subject_10` 虽显著降低 NLL，却改变了
预测轨迹并超过 driver NRMSE 阈值，所以合同不把它们判为预测等价补偿脊线，而保留为
不确定结果。

## 7. 条件敏感度 SVD

| 案例 | 六参数奇异值（降序） | rank/6 | condition | active beta/kappa/tau 奇异值 | active rank/3 | active condition |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| synthetic | 152.284, 78.485, 58.201, 22.481, 15.925, 5.458 | 5 | 27.901 | 114.486, 55.273, 24.123 | 3 | 4.746 |
| `subject_03` | 3998.869, 1918.248, 1037.439, 588.633, 365.735, 96.784 | 5 | 41.317 | 2706.325, 908.604, 510.386 | 3 | 5.303 |
| `subject_13` | 4221.549, 1731.747, 1132.355, 627.984, 316.283, 109.507 | 5 | 38.550 | 2923.356, 960.901, 472.987 | 3 | 6.181 |
| `subject_10` | 1099.135, 404.863, 296.970, 57.183, 26.566, 7.858 | 4 | 139.873 | 952.908, 243.104, 96.184 | 3 | 9.907 |

证据：[sensitivity_svd.csv](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/sensitivity_svd.csv)
和 [summary](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/summary.json)。
这里的 condition 使用所有数值非零奇异值，而 rank 使用 0.05 相对阈值。

active 子矩阵在四案均为局部 rank 3，并不推翻 profile/边界结果：SVD 固定 driver，只问
给定状态下前向预测对参数的局部微分敏感度；profile 则在每次 likelihood 评估中重新估计
latent state，能看到全局非线性补偿和硬边界。它不是联合参数--状态可辨识性证明。

## 8. 近优解的状态稳定性

| 案例 | 已评估状态行（含 reference） | 预测等价替代解 | 其中实质参数不同 | 最坏合格 driver NRMSE | 最低合格 corr | 案例解释 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| synthetic | 21 | 20 | 1 | 0.04249 | 0.999115 | `parameters_nonidentifiable_but_state_stable` |
| `subject_03` | 19 | 18 | 2 | 0.07974 | 0.996863 | `parameters_nonidentifiable_but_state_stable` |
| `subject_13` | 19 | 17 | 0 | 不适用 | 不适用 | `inconclusive` |
| `subject_10` | 2 | 0 | 0 | 不适用 | 不适用 | `inconclusive` |

证据：[state_stability.csv](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/state_stability.csv)
及 summary 中的 diagnostic flags。`subject_13` 的 17 个预测等价候选都未达到注册
变换 span 1% 的“实质参数差异”阈值，不能用几乎重复的解证明状态稳定。
`subject_03` 的两个关键替代解是：

- 扩边最优解：参数距离 0.21712 span，observation whitened RMSE 0.07133，driver
  NRMSE 0.07974，相关 0.996863。
- `profile_kappa_1`：参数距离 0.12500 span，observation whitened RMSE 0.00947，
  driver NRMSE 0.02222，相关 0.999764。

合成案例唯一的实质不同等价解为 `profile_kappa_5`：参数距离 0.06626 span，
observation whitened RMSE 0.02996，driver NRMSE 0.04249，相关 0.999115。
这些只是单案例、当前窗口和当前 observation operator 下的条件稳定性。

## 9. 主终点与科学解释

| 案例 | 主终点 | 直接原因 | 合同解释 |
| --- | --- | --- | --- |
| synthetic | False | beta 上界数值哨兵导致 profile 仅 26/27 finite | 参数不唯一但当前状态稳定；合成全范围数值验证仍不完整 |
| `subject_03` | False | kappa/tau profile 接触注册边界 | 参数不可辨识，但本案例状态稳定 |
| `subject_13` | False | kappa/tau profile 接触边界；扩边预测不等价 | 不确定，不能声称状态等价 |
| `subject_10` | False | beta/kappa/tau profile 全部接触边界；扩边预测不等价 | 不确定，不能声称状态等价 |
| 全体 | **False** | 四案均未满足完整的 all-case endpoint | **实际可辨识性假设未得到支持** |

测量案例显示的边界命中首先应解释为可辨识性不足、受限解或模型失配信号，而不是被试
处于可解释的生理极值。尤其是 `subject_13/10` 扩边后 NLL 大幅下降且预测发生实质变化，
说明原范围强烈约束了解；本轮不能在“补偿脊线”和“模型结构/观测失配”之间进一步区分。

`subject_03` 的状态稳定结果最多使当前 `r(t)` 成为后续 state-only 稳定性检验的候选坐标；
它不能被称为恢复出的真实神经活动，也不能仅凭一个 session、一个局部通道对和一个案例
进入 teacher 或 tokenizer promotion。`subject_13/10` 连这一条件性结论也未建立。

## 10. 质控运行与工件审计

### 10.1 排除的中止运行

| run | 状态 | 中止原因 | 证据用途 |
| --- | --- | --- | --- |
| `20260902_step2_identifiability_v1` | `incomplete_aborted_pre_result` | 首个案例完成前停止，以修正扩边分类 | 仅质控失败记录；无结果 |
| `20260902_step2_identifiability_v2` | `incomplete_aborted_after_median` | synthetic、03、13 后停止，加入“参数至少相差 1% span”要求 | 仅部分诊断；不是最终 Step 2 证据 |
| `20260902_step2_identifiability_v3` | `complete` | 无 | 本报告唯一结果 owner |

中止原因分别保存在
[v1 summary](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v1/summary.json)
和 [v2 summary](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v2/summary.json)。
本报告不合并其数值，也不把它们加入 retained evidence。

### 10.2 正式工件完整性

| 工件 | data rows |
| --- | ---: |
| `representative_selection.csv` | 18 |
| `trial_inventory.csv` | 180 |
| `multistart_results.csv` | 128 = 4 × (16 注册 + 16 扩边) |
| `profile_likelihood.csv` | 108 = 4 × 3 × 9 |
| `expanded_bounds.csv` | 12 = 4 × 3 |
| `sensitivity_svd.csv` | 24 = 4 × 6 |
| `state_stability.csv` | 61 |
| `parameter_recovery.csv` | 6 |

manifest 声明的 14 个工件全部存在且顶层没有额外文件。只读审计现场重算了 9/9 个
`source_sha256`、7/7 个 `input_hashes` 和父 manifest SHA-256，全部匹配；summary 与
manifest 的 endpoint、状态和数据计数一致。关键源码哈希为 runner
`548917d498c215d227bb885bfe89ab0a99564d1879270b999bd00a8ea9c2e863`、Balloon model
`4221b4a53e9b2041d6db5e0274e4d5b509ab8108291e004c01ab67d9054326e4`、measured config
`09317a7fd6eb50b44c829d1ad3f2e5a4319a2fe29e16544448d44095801a939e`、synthetic config
`f8343378c00cb8e0237aba6db82a4bebdf383816ca1f70daccb480c23ce16e31`。

### 10.3 软件与状态验证

- `.venv/bin/python -m pytest -q tests/test_t3_identifiability.py tests/test_t3_measured_reconstruction_null.py tests/test_t3a_balloon_robust_p0.py`：
  29 passed，7.25 秒。
- `.venv/bin/python experiments/scripts/project_state.py validate`：通过；registry schema
  `research_state_registry_v1`，72 records、32 current entities。
- 本报告的本地链接逐项存在；CSV 行数、案例分组、finite profile gate、数据访问计数、
  endpoint 对等和全部已声明 provenance hash 均经只读断言复核。
- 本轮执行的是上述三组直接相关的 T3 回归测试，未声称运行了全仓测试套件。

## 11. 限制

1. 只有一个固定 prior-centre 的 noisy synthetic replicate；没有 simulation-based
   calibration，也没有覆盖多个真值、噪声和输入谱。
2. profile 每参数只有 9 个离散点；单点支持可能反映粗网格，不能当作精确连续区间。
3. 合成 truth 使用 `solve_ivp`，候选 likelihood 与敏感度使用 RK4 前向器；两者不是同一
   数值积分器。合成 beta 上界还出现一个有限性失败点。
4. 测量臂只有 session 01、三个有目的选择的代表和每人 8 个 fit trial；没有总体抽样推断、
   leave-one-session-out、ECG/呼吸 nuisance 或 session-level 随机效应。
5. 选择和分析复用了同一 fit-only cohort；M0 选择分数只用于覆盖低/中/高难度，不是独立
   validation，也不能据此估计泛化性能。
6. EEG 是 10 Hz log-power PCA proxy；HbO/HbR、`P0/Q0` 均为标准化 observation
   坐标。`beta/kappa/tau` 是模型中的有效或 lumped 系数，不能直接解释为分子机制、健康
   指标、绝对 OEF 或绝对 CMRO2。
7. 六参数 SVD 是固定 driver 的局部条件前向分析，不是参数和 latent state 的联合
   identifiability test；其 full/active rank 不能覆盖全局 profile 和边界失败。有限差分
   使用参考点两侧各 0.01 transformed step，未裁剪到注册范围；因此像 `subject_10`
   这样参考解就在边界的案例会有一侧扰动越过注册范围，其 rank 只能作局部诊断。
8. 当前空间观测只有一个预选 HbO/HbR 对和 fit-fold EEG PCA；没有检验多通道空间扰动。
9. 本 suite 没有 cross-modal null、held-out predictive qualification 或 protected
   evaluation；也不能覆盖先前 P0 的负结果。

## 12. 决策与下一步边界

- 原始 `beta/kappa/tau` 三参数组合保持 diagnostic-only；不得以当前最优值作为被试生理
  参数或 teacher 标签。
- 不因较低训练 NLL 放宽参数范围，也不把边界值解释为生理极值。
- 第三步未启动。若继续多 session 实验，应先冻结一个新的、仍属 nonprotected 的合同：
  将 `r(t)` 稳定性作为诊断终点，加入 session-level nuisance，并明确原始三参数不具备
  promotion 资格；更稳妥的参数路线是先改为一到两个 gain/time-scale 综合方向。
- 在多 session、nuisance 和空间扰动下尚未复现前，`state-only` 只是一项后续候选解释，
  不是已获准的 teacher。
- `research_state/registry.json` 未因本实验改变；validation 与 protected boundary 保持关闭。

这一步的最小可辩护结论是：**当前 fit-only T3a 观测不足以支持把原始
`beta/kappa/tau` 作为可解释、可推广的被试特异生理参数；一个代表案例中的 latent
trajectory 对等价参数较稳定，但证据不足以晋升为 physical teacher。**
