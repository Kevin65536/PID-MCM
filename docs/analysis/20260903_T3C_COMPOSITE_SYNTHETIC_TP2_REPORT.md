# T3c 实验第四步：综合参数合成 T-P2 详细报告

_实验日期：2026-09-03；报告状态：formal synthetic C1 screen 完成，C2 按门控未运行_

## 结论

第四步的已注册综合参数合成 `T-P2` 屏幕已完整运行。正式 v1 的机器判定为
`BLOCKED_C1_COMPOSITE_IDENTIFIABILITY`：`C1_G` 在 19 项 gate 中通过 13 项、失败
6 项；`C1_T` 通过 14 项、失败 5 项。两个一维方向都没有通过，因此
`C2_GT=NOT_RUN_C1_GATE_NOT_MET`，没有启动二维联合拟合，measured hierarchical arm
仍为 `blocked_prerequisite_not_started`。

这是对当前“综合坐标 + EKF likelihood 拟合”的明确负结果，不是运行失败。数值优化、
边界、局部 SVD 和 held-out 预测门均通过，但两个方向同时出现低 profile/
posterior coverage、定向偏差、SBC 失配以及 60/60 个存在预测等价替代参数的重复。
因此，良好的 held-out 预测不能被解释为参数可辨识。

这一结果不否定被试间可能存在真实神经血管动力学差异；它只说明，当前观测合同与
推断器在一个已知真值、没有 measured-domain 复杂性的有利合成环境中，仍不能稳定恢复
`log_gain_relative` 或 `log_time_relative`。因而不能进入 measured partial pooling。

正式证据由 [manifest](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/manifest.json)
统一记录运行状态、边界、source hash、artifact hash 和行数合同。结论摘要见
[summary](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/summary.json)，
逐门数值见 [gates](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/gates.csv)。

## 1. 运行身份与数据边界

| 字段 | 正式值 |
| --- | --- |
| suite/run | `t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1` |
| schema | `t3c_composite_synthetic_t2_v1` |
| 运行状态 | `status=complete`; `run_state=complete`; `completion_status=complete` |
| 顶层判定 | `BLOCKED_C1_COMPOSITE_IDENTIFIABILITY` |
| C2 状态 | `NOT_RUN_C1_GATE_NOT_MET` |
| UTC 时间 | 2026-09-03 04:11:41.582 至 05:13:07.707 |
| Asia/Shanghai 时间 | 2026-09-03 12:11:41.582 至 13:13:07.707 |
| 墙钟时长 | 3686.1247 秒，约 61 分 26.1 秒 |
| 独立重复 | `C1_G` 60；`C1_T` 60 |
| scope | `synthetic_known_truth_only` |
| measured metadata / array | 未打开 / 未打开 |
| validation / protected array access | 0 / 0 |
| truth 传入 fitter | `false` |
| qualification / decision eligibility | `false` / `false` |

本轮不读取 measured、validation 或 protected 数组，也不调用 measured loader。这些边界在
[manifest](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/manifest.json)
和 [resolved config](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/resolved_config.yaml)
中均为 fail-closed。

## 2. 综合坐标与已冻结问题

对参考参数

\[
(\beta_*,\kappa_*,\gamma_*,\tau_*,\alpha_*,E_{0*})
=(1.0,0.64,0.32,2.0,0.32,0.32),
\]

定义相对 gain 和 time-scale 坐标

\[
g=\log\frac{(\beta/\gamma)}{(\beta_*/\gamma_*)},\qquad
t=\log\frac{1/\sqrt{\gamma}}{1/\sqrt{\gamma_*}}.
\]

固定 `zeta`、`Tv`、`alpha`、`E0`、光学观测 gauge 和 EEG loading 后，逆映射为

\[
\beta=\beta_*e^{g-2t},\qquad
\gamma=\gamma_*e^{-2t},\qquad
\kappa=\kappa_*e^{-t},\qquad
\tau=\tau_*.
\]

任何诱导出的 raw 参数超出 `beta=[0.25,4]`、`kappa=[0.20,1.50]`、`gamma=[0.10,1]`、
`tau=[0.50,5]`、`alpha/E0=[0.10,0.80]` 都直接拒绝，不做 clipping。数学和边界的可执行 owner
是 [config](../../experiments/configs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2_v1.yaml)
与 [runner](../../experiments/evaluate_t3c_composite_synthetic_t2.py)。

| 候选 | 激活坐标 | 固定坐标 | truth 分布 | truth support | 注册域 | 扩展域 |
| --- | --- | --- | --- | --- | --- | --- |
| `C1_G` | `g=log_gain_relative` | `t=0` | 截断 Normal(0, 0.20) | `[-0.40,0.40]` | `[-0.60,0.60]` | `[-0.80,0.80]` |
| `C1_T` | `t=log_time_relative` | `g=0` | 截断 Normal(0, 0.15) | `[-0.40,0.40]` | `[-0.50,0.50]` | `[-0.55,0.55]` |

`C1_G` 和 `C1_T` 是两个独立一维问题。只有两者都通过，`C2_GT` 才能在另一个新的、
已注册 run 中联合放开 `g,t`；本 runner 本身不会内联执行 C2。

## 3. 合成设计与 truth–fitter 隔离

每个方向独立抽取 60 个 truth panel。每个 panel 含 3 个拟合 trial 和 1 个 held-out
trial，每个 trial 都独立重置，长 40 秒、4 Hz、160 个时点。truth trajectory 使用独立
`solve_ivp` 生成，不调用拟合器的离散状态转移。观测噪声是同方差 Student-t，`df=5`，
EEG/HbO/HbR scale 分别为 `0.080/0.025/0.015`。数据设计见
[truth table](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/truth_parameters.csv)
和 [inventory](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/synthetic_inventory.csv)。

`FitDataset` 只携带 candidate、replicate ID、噪声训练观测、optimizer-only seed 和删除已实现
truth/driver/generation seed 的拟合合同。truth parameter、clean driver 和 held-out observations
不进入 worker payload；只在全部拟合结束后再做真值连接和评分。每个 truth row 的
`truth_passed_to_fitter=false`，480 个 inventory row 的观测/driver hash 均已按种子重生成核对，
0 个 mismatch。

## 4. 拟合、不确定性与 held-out 合同

- point fit 与 profile 使用 likelihood-only EKF/IRLS/RTS smoother。每个 panel 在注册域 16 个起点、
  扩展域 16 个起点，共 32 个 L-BFGS-B 拟合。
- 一维 profile 使用 21 点注册网格，95% support 阈值为 `delta NLL=1.920729410347062`。
- SBC 主方法是截断 Normal prior 下的 `EKF_Laplace_truncated_normal`。若其未通过预冻结
  calibration gate，只允许切换为 81 点 `exact_1d_grid_quadrature_under_EKF_likelihood`。
- held-out 评分固定 fitted coordinate，EEG 全程可见，HbO/HbR 仅 12 秒前可见；评分 12–40 秒
  的 masked HbO/HbR，即每个 panel `112 time points x 2=224` 个有限目标。
- SVD 是在已知 truth driver 上计算的局部、白化、两列 gain/time forward Jacobian。活跃
  C1 rank=1 是硬门，联合 rank=2 只是诊断。

## 5. 结果概览

| 指标 | `C1_G` | `C1_T` |
| --- | ---: | ---: |
| 候选判定 | `FAIL` | `FAIL` |
| gate 通过 | 13/19 | 14/19 |
| 估计偏差（log 坐标） | +0.114422 | -0.079825 |
| 偏差 95% bootstrap CI | [0.101840, 0.126903] | [-0.090325, -0.069240] |
| RMSE | 0.125041 | 0.090099 |
| profile truth coverage | 17/60 = 0.2833 | 27/60 = 0.4500 |
| 选定 SBC posterior | 81 点 grid fallback | 81 点 grid fallback |
| 选定 SBC rank mean | 0.062994 | 0.903931 |
| 预测等价替代参数 panel | 60/60 | 60/60 |
| held-out candidate - M0 mean NLL/obs | -0.102453 | -0.060245 |
| held-out candidate - oracle mean NLL/obs | -0.024950 | -0.003651 |

数值来自 [summary](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/summary.json)、
[recovery](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/parameter_recovery.csv)
和 [calibration](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/calibration.csv)。

## 6. 逐项 gate 结果

下表将 artifact 中的 `passed=True/False` 保守显示为 `PASS/FAIL`。精确原始值保留在
[gates.csv](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/gates.csv)。

| Gate | 预冻结阈值 | `C1_G` | `C1_T` |
| --- | --- | --- | --- |
| independent SBC replicates | >=60 | 60 `PASS` | 60 `PASS` |
| solver failure fraction | <=0 | 0 `PASS` | 0 `PASS` |
| estimate boundary fraction | <=0 | 0 `PASS` | 0 `PASS` |
| multistart success fraction | >=1 | 1.0 `PASS` | 1.0 `PASS` |
| max multistart NLL spread/obs | <=0.10 | 6.676e-9 `PASS` | 5.473e-9 `PASS` |
| max multistart parameter spread/span | <=0.10 | 0.000181 `PASS` | 0.000268 `PASS` |
| expanded outside registered near-optimal count | <=0 | 0 `PASS` | 0 `PASS` |
| profile complete and contiguous | `true` | `true PASS` | `true PASS` |
| max profile-reference difference NLL | <=0.01 | 0.294116 `FAIL` | 0.337098 `FAIL` |
| profile boundary fraction | <=0.05 | 0 `PASS` | 0.033333 `PASS` |
| profile truth coverage | [0.90,0.99] | 0.283333 `FAIL` | 0.450000 `FAIL` |
| selected SBC calibration | `true` | `false FAIL` | `false FAIL` |
| absolute bias/span | <=0.05 | 0.095351 `FAIL` | 0.079825 `FAIL` |
| RMSE/span | <=0.20 | 0.104201 `PASS` | 0.090099 `PASS` |
| max absolute bias-CI endpoint/span | <=0.10 | 0.105752 `FAIL` | 0.090325 `PASS` |
| material prediction-equivalent panel count | <=0 | 60 `FAIL` | 60 `FAIL` |
| active composite sensitivity rank | `true` | `true PASS` | `true PASS` |
| held-out excess-oracle CI upper | <=0.10 | -0.007329 `PASS` | 0.005378 `PASS` |
| held-out noninferiority CI upper | <=+0.05 | -0.062074 `PASS` | -0.029482 `PASS` |

### `C1_G` 失败项

`profile_reference_difference_nll`、`profile_truth_coverage`、
`sbc_selected_posterior_calibration`、`absolute_bias_fraction`、
`bias_ci_extent_fraction` 和 `material_prediction_equivalent_count`。

### `C1_T` 失败项

`profile_reference_difference_nll`、`profile_truth_coverage`、
`sbc_selected_posterior_calibration`、`absolute_bias_fraction` 和
`material_prediction_equivalent_count`。

## 7. SBC 和 coverage

| 方向 / posterior | KS D（critical=0.175575） | rank mean | 95% coverage | Clopper–Pearson 95% CI | resolution | 判定 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| G / EKF-Laplace | `NaN` | `NaN` | 17/60 = 0.2833 | [0.1745,0.4144] | `false` | `FAIL` |
| G / EKF-grid | 0.763320 | 0.062994 | 18/60 = 0.3000 | [0.1885,0.4321] | `false` | `FAIL` |
| T / EKF-Laplace | `NaN` | `NaN` | 33/60 = 0.5500 | [0.4161,0.6788] | `false` | `FAIL` |
| T / EKF-grid | 0.682676 | 0.903931 | 33/60 = 0.5500 | [0.4161,0.6788] | `true` | `FAIL` |

Laplace 在 `C1_G` replicate 46 命中 truth-support 上边界，在 `C1_T` replicate 12 命中下边界；
这两行的 posterior SD、区间和 rank 按合同记为 `NaN`，使 Laplace calibration 失败并触发
grid fallback。G-grid 还有 1/60 个 posterior SD 小于两个网格步长，因此 resolution 失败；
T-grid 的 resolution 通过，但 rank、KS 和 coverage 都严重失配。G/T grid KS p-value 分别为
`5.06e-37` 和 `2.72e-28`。所有合计区间都不包含名义 coverage 0.95。

`exact_1d_grid_quadrature_under_EKF_likelihood` 的“exact”只指一维网格数值积分；它仍依赖近似
EKF likelihood，不是对全部潜状态的 exact Bayesian posterior。若未来重新考虑正结果，计划所要求的
exact latent-posterior SBC 仍属未执行项；当前 grid 失败只能作为当前 EKF 推断管线的负证据。

## 8. Recovery、profile、multistart 和边界

`C1_G` 的 mean error 是 `+0.114422`，95% bootstrap CI `[0.101840,0.126903]`；注册跨度为
1.2，所以偏差比例是 0.095351。`C1_T` 的 mean error 是 `-0.079825`，CI
`[-0.090325,-0.069240]`；注册跨度为 1.0。两个偏差都有明确方向，不是无偏误差增大。

3840/3840 个 multistart 全部成功，solver failure fraction 为 0，两方向的注册起点结果几乎
一致。没有任何 expanded-domain best 在注册域外仍近似最优，也没有 point estimate 命中边界。
这表明负结果不是简单的 optimizer 未收敛或 raw-bound clipping 造成。

21 点 profile 都是有限且 support 连续；G 方向 0/60 触边，T 方向 2/60 触边，低于 5%
阈值。但 truth 仅被 profile support 覆盖 17/60 和 27/60。

`profile_reference_difference_nll` 需要单独解读：它比较连续 multistart optimum 与固定 21 点
profile grid 的最小值，而网格不强制包含连续 optimum。G 最大差值的 optimum 为 0.510388，
最近 profile minimum 在 0.54；T 分别为 -0.025072 和 -0.05。因此 0.294116/0.337098
的 gate 失败混合了网格离散误差，不应被单独解释为 optimizer 不一致。即使不把该项作为
科学失败证据，两方向仍因 SBC、coverage、bias 和 confounding 中的多项独立门而失败。

使用已写盘的 81 点 likelihood grid 做不改变正式 gate 的敏感性复核后，G/T truth coverage 仅从
17/60、27/60 上升到 19/60、30/60，仍远低于 `[0.90,0.99]`。在 continuous best 落入 81 点
truth support 的 58/60 个 panel 中，细网格 min-vs-best 最大差值降为 G 0.006395、T
0.013448。这确认 21 点 reference gate 主要是分辨率问题，但细网格不能挽救 truth coverage。

## 9. SVD 与参数/状态 confounding

两个 C1 的 active singular value 都是有限正数，60/60 个 active rank 都为 1。联合两列诊断也在
120/120 行中 rank=2；第二奇异值相对第一奇异值的范围为 G `[0.6287,0.9111]`、
T `[0.6171,0.9015]`，高于 0.05 诊断阈值。这只证明在 truth driver 已知、其他量固定时局部
forward map 有一阶敏感度，不能证明拟合后验在全局上可辨识。原始值见
[sensitivity SVD](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/sensitivity_svd.csv)。

更直接的 confounding 检查在每个 panel 中搜索至少相差 1% 注册跨度、但仍位于
`delta NLL<=1.920729` 支持域内的替代参数。若其相对 best fit 的观测白化 RMSE `<=0.10`，
则标记为 prediction-equivalent。G 和 T 均在 60/60 个 panel 中找到至少一个：

| 指标 | `C1_G` | `C1_T` |
| --- | ---: | ---: |
| parameter distance/span 范围 | 0.0252–0.0991 | 0.0251–0.1099 |
| whitened prediction RMSE 范围 | 0.0127–0.0371 | 0.0163–0.0431 |
| driver NRMSE 最大值 | 0.0791 | 0.0779 |
| driver correlation 最小值 | 0.99914 | 0.99703 |
| driver-stable panel | 60/60 | 60/60 |

因此更准确的解读是：在当前合成设计中，参数并不唯一，但平滑后的 driver 仍可稳定。
这只支持 state-level 稳定描述，不支持把 G/T 作为个体生理 trait。本检查每个 panel 在
找到第一个预测等价解后即停止，所以“60”是至少有一个替代解的 panel 数，不是对所有替代解的
穷尽计数。详见 [state confounding](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/state_confounding.csv)。

## 10. Held-out 预测

| 方向 | candidate - M0 mean NLL/obs | 95% bootstrap CI | candidate - oracle mean NLL/obs | 95% bootstrap CI |
| --- | ---: | --- | ---: | --- |
| `C1_G` | -0.102453 | [-0.147365,-0.062074] | -0.024950 | [-0.042972,-0.007329] |
| `C1_T` | -0.060245 | [-0.096637,-0.029482] | -0.003651 | [-0.012994,0.005378] |

差值为负表示 candidate NLL 更低。两个方向的 non-inferiority 与 excess-oracle gate 都通过，且
120 个 panel 的 224 个目标都是有限值。但 oracle 参数只是生成真值，评分仍使用近似 EKF smoother；
候选估计可以通过补偿近似推断与特定噪声实现而得到更低 NLL。因此预测通过与参数恢复失败
并不矛盾，也不能用前者覆盖后者。详见
[held-out scores](../../experiments/runs/physiology_semantic_tokenizer/t3c_composite_synthetic_t2/20260903_step4_composite_t2_v1/heldout_scores.csv)。

这一 endpoint 是“EEG 全程条件 + fNIRS 后段遮挡”的 smoother 得分，不是无未来 EEG 的因果预测。

## 11. Artifact 完整性与可复现性

| Artifact | 行单位 | 预期 | 实际 |
| --- | --- | ---: | ---: |
| `truth_parameters.csv` | candidate-replicate truth | 120 | 120 |
| `synthetic_inventory.csv` | candidate-replicate-trial | 480 | 480 |
| `multistart_results.csv` | candidate-replicate-variant-start | 3840 | 3840 |
| `profile_likelihood.csv` | candidate-replicate-profile point | 2520 | 2520 |
| `posterior_grid.csv` | candidate-replicate-posterior point | 9720 | 9720 |
| `posterior_diagnostics.csv` | candidate-replicate-method | 240 | 240 |
| `parameter_recovery.csv` | candidate-replicate | 120 | 120 |
| `state_confounding.csv` | candidate-replicate | 120 | 120 |
| `heldout_scores.csv` | candidate-replicate | 120 | 120 |
| `sensitivity_svd.csv` | candidate-replicate | 120 | 120 |
| `calibration.csv` | candidate-method | 4 | 4 |
| `gates.csv` | candidate-gate | 38 | 38 |

已重算 manifest 列出的 15 个 artifact SHA-256、size 和 CSV 行数，全部精确匹配；所有自然主键
组合完整且无重复。480/480 个 synthetic trial 都是 `physical_valid=true`，trial seed 唯一。

| 冻结源 | SHA-256 |
| --- | --- |
| runner | `596fe24fd1925f950e69f79b14acf28ff7e19afbc313825b3990e2ef0344c55e` |
| config | `78f9ee1c06225dec9d664ef6ca984d4f194139c24f3aed9f693219d6ad2dc2f8` |
| resolved config | `178480eb9c3dc742b7e24cec6dd946dff3d1fd41750d1ad4e4876fa2dad3988b` |
| Balloon model | `4221b4a53e9b2041d6db5e0274e4d5b509ab8108291e004c01ab67d9054326e4` |
| T3a P0 reference | `f8343378c00cb8e0237aba6db82a4bebdf383816ca1f70daccb480c23ce16e31` |
| T3c admission contract | `6ce049571bd268f5e68e7f3f1b1e43a66f875b06e3b056b28e36c7ec77bec0b1` |

运行环境记录为 Python 3.12.4、NumPy 2.4.1、SciPy 1.17.0。本次最终相关回归为
`33 passed`；两个 runner 通过 `py_compile`。测试 owner 是
[composite tests](../../tests/test_t3c_composite_synthetic_t2.py) 和
[admission tests](../../tests/test_t3c_hierarchical_admission.py)。

## 12. 审计限制与合同偏差

1. 全局计划要求每个 gate 显式记录 `PASS/FAIL/INCONCLUSIVE/INVALID`，而 v1
   `gates.csv` 只记录布尔 `passed`。本轮完整运行且所有判定均有数值支持，所以
   `True/False -> PASS/FAIL` 的保守映射不改变 blocked 结论；但在任何未来可能得到正结果的新
   run 前，应在新版本中补充四态字段和 reason，不应回写本 v1。
2. profile-reference gate 受 21 点粗网格影响，应把它视为数值诊断偏保守；本轮的负结论不依赖
   这一项。
3. 14 个 `NaN` 字段由两个 Laplace 边界 invalid case 触发，manifest/artifact 未损坏；
   有限 grid fallback 已被使用，但两方向仍失败。
4. SVD 在已知 driver 下条件化，confounding 只比较 posterior trajectory mean/driver，不穷尽
   完整 latent posterior 等价性。
5. manifest 顶层未重复 `C2_GT_policy` 和 `expected_artifact_row_counts`；前者由已哈希的
   resolved config 拥有，后者已逐 artifact 出现在 manifest `row_count_contract` 中。manifest 仍唯一拥有
   C2 当前状态和实际/预期行数；该顶层便利字段缺失不影响结论，但应在未来版本中补齐。
6. 本轮是有利的 known-truth synthetic test，不是 measured external validation。不能从中作出任何被试 trait、
   teacher qualification 或 tokenizer promotion 声明。

## 13. 第四步状态与后续边界

按预冻结决策树：

1. `C1_G=FAIL`；
2. `C1_T=FAIL`；
3. 因两者未同时通过，`C2_GT` 不运行；
4. synthetic composite T-P2 不满足 measured hierarchy 的参数辨识前置条件；
5. measured hierarchical partial pooling、validation 与 protected 数据继续关闭。

因此本次“按计划执行”在预注册的 stop rule 处正常结束，不会为了进入 C2 而修改阈值或追加
未注册分析。若未来继续，应先以新版本合同修正 gate schema/profile 参考评估，并选择“改善推断器/
重新参数化”或“只保留 state-level 目标”之一，然后再注册新的 synthetic screen。本 v1 负结果保持不可改写。
