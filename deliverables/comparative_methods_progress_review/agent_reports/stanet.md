# STA-Net 进度审查（审计日期：2026-08-11）

## 结论

STA-Net-PyTorch 的项目适配、训练/评估流水线和正式结果工件均已完成并冻结：正式协议为 `7 tasks × 2 protocols × 5 outer folds = 70/70` 个 job，70 个训练均通过收敛规则，70 个评估摘要和 70 个预测工件均存在。该结论由 `comparative_methods/STA-Net-PyTorch/runs/fivefold/20260727_sta_net_no_artifact_mask_converged_5fold_v1/` 的 supervisor、freeze、training、evaluation 和 aggregate 工件核验得到；未读取其中的 protected prediction 数组。

但是，现有数字不能直接进入当前 support-matched 总表。统一计划明确指出，STA-Net 的默认分类 observation budget 是 EEG 3 s + fNIRS 13 s（DSR 为 EEG 2 s + fNIRS 13 s），而 EFRM 参照为 8/8 s（DSR 2/2 s），所以现有结果只能以 `method_native_context_reference` 保留（`comparative_methods/EXPERIMENT_PLAN.md:160-163`）。项目状态也将 STA-Net 标成 “Complete; context reference”，而非 support-matched direct-table evidence（`docs/comparisons/STATUS.md:8-10`）。

## 加权评分

| 维度 | 分数 | 依据（证据级别） |
| --- | ---: | --- |
| `code_components`（30%） | 93 | [已验证] 5 个核心 `sta_net_pytorch/*.py`，25 个顶层 runner，5 个 YAML 配置，2 个测试文件；模型、adapter、split registry、trainer、freeze/evaluate、aggregate、visualize、smoke 均存在。 |
| `input_adaptation`（20%） | 84 | [已验证] 7 个任务规格、16×16 投影、valid-mask、bad-channel 和 train-only REFED scaler；[已验证] 25 个方法测试通过。因 method-native observation budget 与公共直接比较不一致，扣除 support-match 适配分。 |
| `output_adaptation`（15%） | 92 | [已验证] 6 个分类 head（Macro-F1 主端点）+ 1 个 REFED `[2,20]` 连续序列 head（CCC 主端点），per-coordinate target mask、native-coordinate error 和预测保存均实现。 |
| `result_generation`（25%） | 94 | [已验证] `70/70` 训练、`70/70` evaluation summaries、`70/70` `protected_predictions.npz`、8 个 aggregate 输出文件；[缺失] 正式 run 下原生 PNG/SVG 为 `0/0`，需由总报告另行绘图。 |
| `evidence_reproducibility`（10%） | 88 | [已验证] upstream revision、配置/代码/模型/adapter/split/checkpoint hashes、3 lane 状态和冻结协议齐全；[缺失] method manifest 没有 B6/table-admissibility 字段，且上游无许可证、不能宣称 source-level numerical equivalence。 |

加权总分为 `0.30×93 + 0.20×84 + 0.15×92 + 0.25×94 + 0.10×88 = 90.80/100`（四舍五入 **91/100**）。

## 1. 方法、上游和计划边界

- [已验证] `comparative_methods/STA-Net-PyTorch/README.md:3-18` 将实现定义为独立 PyTorch adaptation，不声称源代码级数值等价；`sources/method_manifest.yaml:1-39` 固定方法为 `sta_net_eeg_fnirs_supervised`，上游 revision 为 `b6db8bb5eb2f6491a13f0938880ee70e32162ee7`，并记录 `no_license_file_observed_blocks_redistribution`。
- [已验证] 未修改的 TensorFlow 上游位于 `comparative_methods/STA-Net/`（310 行 `sta.py`、124 行 `run_sta_net.py`、5 个 preprocessing 文件共 571 行）；上游 runner 使用预生成 NPZ、每被试三 session 的 subject-specific 二分类，只有 MI/MA/WG 论文任务。`comparative_methods/STA-Net/sta.py:249-306` 的模型输入为 EEG `[16,16,600,1]`、fNIRS `[11,16,16,30,2]`，并含 FGSA/EGTA/融合权重。
- [已验证] 统一计划把 STA-Net 列入多模态监督深度融合，正式轨为 strict cross-subject 五折端到端训练，状态为“正式五折已完成，保留为固定参考”（`comparative_methods/EXPERIMENT_PLAN.md:54-71`）。计划禁止为统一新方法回写、调参或重新选择该冻结结果（同文件 `:68-71`）。
- [声明] `sources/method_manifest.yaml` 的 B0–B4 均为 pass/文档化状态；formal aggregate 另有完整 B5 工件，但 manifest 本身没有 B5/B6 字段，不能把 manifest 声明单独当作 table admission。

## 2. 必要代码组件进度

| 组件 | 精确计数 | 状态与证据 |
| --- | ---: | --- |
| `sta_net_pytorch/data.py` | 472 行；7 个 `STANetTaskSpec` | [已验证] 统一 loader、task filtering、16×16 spatial projector、valid-mask、HbO/HbR 分解、REFED scaler、sample identity 和 collate。 |
| `sta_net_pytorch/model.py` | 407 行；9 个 class | [已验证] `SamePadConv3d`、2 个时空卷积/FGSA block、Keras-style attention/EGTA、fusion gates、classification/regression heads、mask-aware objective。 |
| `sta_net_pytorch/splits.py` | 384 行；11 个函数 | [已验证] cross-subject、sample-random、single-subject registry，public validation 检查和 hash 写入。 |
| `sta_net_pytorch/metrics.py` | 78 行 | [已验证] classification metrics 和 finite/improvement helpers。 |
| 顶层 runner | 25 个 `.py` | [已验证] `train.py` 829 行、`tune.py` 450 行、`visualize_results.py` 821 行、`aggregate_fivefold.py` 430 行、`evaluate_protocol.py` 157 行及 launch/worker/freeze/smoke 等配套。 |
| 配置/测试 | 5 个 YAML；2 个测试文件；25 个 test functions | [已验证] smoke、训练、tuning、split、指标、绘图和五折 registry 覆盖。 |

## 3. EEG–fNIRS 输入合同与适配

`sta_net_pytorch/data.py:241-279` 由采样率和 task spec 计算窗口，不消费旧的 `analysis_valid_mask`，仅使用真实记录 `valid_mask`；坏通道在空间插值前排除。`data.py:322-339` 将 mask policy、坐标模式和 train-only target scaler 写入 adapter manifest。方法 native 输入预算如下：

| 任务 | 类型/类别 | EEG budget | fNIRS budget | adapter 输出（去 batch） | 适配状态 |
| --- | --- | ---: | ---: | --- | --- |
| MI | 二分类 LMI/RMI | 3 s = 600 samples | 11 × 3 s lag = 13 s = 130 samples | EEG `[1,16,16,600]`; fNIRS `[11,2,16,16,30]` | [已验证] formal 10 folds |
| MA | 二分类 MA/BL | 3 s = 600 | 13 s = 130 | 同上 | [已验证] formal 10 folds |
| WG | 二分类 WG/BL | 3 s = 600 | 13 s = 130 | 同上 | [已验证] formal 10 folds |
| n-back | 三分类 0/2/3-back | 3 s = 600 | 13 s = 130 | 同上（3-class head） | [已验证] formal 10 folds；项目适配 |
| DSR | 二分类 Go/No-go | 2 s = 400 | 13 s = 130 | EEG `[1,16,16,400]`; fNIRS `[11,2,16,16,30]` | [已验证] formal 10 folds；EEG-primary/context variant |
| Visual | 四分类 RR/RF/FF/FR | 3 s = 600 | 13 s = 130 | 同 MI | [已验证] formal 10 folds；项目适配 |
| REFED | 连续 valence/arousal | 20 s = 4000 | 18 × 3 s lag = 20 s = 200 samples | EEG `[1,16,16,4000]`; fNIRS `[18,2,16,16,30]`; target `[2,20]` + mask | [已验证] formal 10 folds；项目 regression adapter |

- [已验证] 官方坐标清单为 28 EEG、36 fNIRS locations（HbO/HbR 共 72 components）；官方 inventory 命中时使用 source 16×16 grid，否则使用统一 geometry normalized grid。对应代码为 `data.py:16-63,222-239`，方法测试在 `tests/test_sta_net_pytorch.py:139-217`。
- [已验证] test `test_official_wg_adapter_emits_released_sta_net_tensor_shapes` 和 `test_adapter_ignores_legacy_artifact_gated_analysis_valid_mask` 分别验证 `[1,16,16,600]`/`[11,2,16,16,30]` 形状、finite output、`artifact_mask_consumed=False` 与 `validity_source=valid_mask_only`。
- [已验证] REFED scaler 由训练窗口有效 target values 拟合（`data.py:396-424`）；formal training manifest 的 `target_scaler`/fit scope 对应训练主体。
- [已验证] split registry summary 有 7 task：cross-subject outer folds `5`、public inner folds `15`；single-subject public/test folds 依任务为 `44–160`。正式 run 的 70 jobs 划分为 strict `35` + sample-random `35`。
- [已验证] 统一合同要求 observation anchor、relative interval、channel identity 和 canonical branch 一致；但该冻结 STA-Net 结果明确属于 native context，不能视作 support-matched（`comparative_methods/EXPERIMENT_PLAN.md:138-163`）。

## 4. 输出适配与指标

- [已验证] `sta_net_pytorch/model.py:224-331` 返回 `prediction`、EEG auxiliary prediction、fusion/fNIRS predictions、fusion weights、lag/spatial attention 和 alignment losses；分类输出为概率，回归输出为 `[B,2,20]`。
- [已验证] `model.py:334-400` 对分类使用 main + EEG auxiliary + alignment loss，对 REFED 使用 `target_valid_mask` 加权 Smooth-L1/MSE；遮蔽坐标不会进入 regression loss。
- [已验证] `aggregate_fivefold.py:88-110,172-224` 将分类主端点设为 outer-fold Macro-F1，REFED 主端点设为 outer-fold CCC；同时保留 accuracy、balanced accuracy、Kappa、ROC-AUC、MAE/RMSE/R²/Pearson 等 companion metrics。
- [缺失] formal aggregate `summary.json` 的 keys（12 个）没有 `table_admissible`、support profile 或 direct-comparison disposition 字段；当前可否入表必须依统一计划/状态文档另行判断。

## 5. 正式结果生成与冻结状态

### 5.1 矩阵完整性

`protocol_freeze_manifest.json` 声明 7 task、seed `42`、3 lanes、最大安全预算 300 epochs，并明确 `strict_cross_subject` 按 subject 隔离、`sample_random` 不隔离 dependency groups。精确工件计数：

- [已验证] `job_count=70`；按 protocol 为 strict `35`、sample-random `35`；按 task 每项 `10`（2 protocol × 5 folds）。
- [已验证] supervisor `completed_jobs=70`、`completed_lanes=3/3`，lane job counts 为 `26 + 19 + 25`，状态文件为 `3 completed`（`supervisor_status.json`、`status/lane_00–02.json`）。
- [已验证] 每一 protocol 有 35 个 `training` 目录、35 个 `evaluation` 目录、35 个 `checkpoint_best.pt`、35 个 `checkpoint_latest.pt`、35 个 `freeze_manifest.json`、35 个 evaluation `summary.json` 和 35 个 `protected_predictions.npz`，总计 70/70。
- [已验证] 70/70 training status 为 `completed`，aggregate convergence audit `all_folds_converged=true`，停止规则为至少 40 epochs 且至少 30 个 validation epochs 无 improvement；`training_convergence.csv` 为 70 行数据 + header。

### 5.2 严格 cross-subject 主端点（均值 ± 5-fold sample SD）

| 任务 | 主端点 |
| --- | ---: |
| MI Macro-F1 | 56.40 ± 1.58% |
| MA Macro-F1 | 62.84 ± 4.25% |
| WG Macro-F1 | 62.11 ± 3.13% |
| n-back Macro-F1 | 37.52 ± 2.32% |
| DSR Macro-F1 | 60.69 ± 2.38% |
| Visual Macro-F1 | 25.01 ± 0.77% |
| REFED CCC | 0.081 ± 0.048 |

这些值与 `docs/comparisons/STATUS.md:37-54` 和 aggregate `summary.json` 一致；它们是已完成 benchmark artifact，但不是原论文 subject-specific 复现。

### 5.3 sample-random 诊断端点

| 任务 | 主端点 |
| --- | ---: |
| MI Macro-F1 | 53.20 ± 3.53% |
| MA Macro-F1 | 68.87 ± 2.17% |
| WG Macro-F1 | 63.18 ± 1.13% |
| n-back Macro-F1 | 37.07 ± 4.93% |
| DSR Macro-F1 | 65.29 ± 1.42% |
| Visual Macro-F1 | 27.74 ± 1.38% |
| REFED CCC | 0.126 ± 0.038 |

该 split 的 `dependency_group_isolation=false`，只能标作信息可见上界/诊断，不能支撑新主体泛化（`aggregate/paper_table.md:40`、`protocol_freeze_manifest.json`）。

### 5.4 结果冻结与 protected 边界

- [已验证] training/`manifest.json` 与 freeze manifest 在评估前均写 `protected_test_opened=false`；70 个 evaluation `summary.json` 写 `protected_test_opened=true`，表示该历史正式协议确实完成了一次显式 protected 评估。
- [已验证] `protocol_freeze_manifest.json` 的 `protected_open_authorization` 为显式用户请求；aggregate `summary.json` 的 `protected_test_opened=true`。本次审计没有加载或解析 `protected_predictions.npz` 数组。
- [声明] 当前统一状态要求 STA-Net 只保留冻结 context reference，不得重跑或重新挑选；本审计未执行任何训练、protected evaluation、aggregate 重算或 checkpoint 选择。

## 6. 可视化和最终交付风险

- [已验证] `visualize_results.py` 821 行，能够输出 validation prediction/metric report、confusion/reliability/attention/fusion diagnostics 和 SVG/PNG；tuning analysis 目录保留 `86` 个 PNG + `86` 个 SVG。
- [缺失] 正式 run 目录本身没有 PNG/SVG（精确计数 `formal_png=0`, `formal_svg=0`）；formal aggregate 有 `8` 个文件：`summary.json`、5 个 CSV、`paper_table.md`、`paper_table.tex`。主报告 PPT 需要基于本审计 JSON/aggregate CSV 单独生成进度图和结果图，不能假设 formal run 已带图。

## 7. 阻塞项与风险

1. **support-matched 入表阻塞（已验证）**：`comparative_methods/EXPERIMENT_PLAN.md:160-163` 的 3+13 s（DSR 2+13 s）与 EFRM 8/8 s（DSR 2/2 s）不同；所有 7 个正式 task 只能标 `method_native_context_reference`，不能直接排进 support-matched direct table。
2. **source-level 复现边界（已验证/声明）**：上游只有 MI/MA/WG 的 subject-specific binary runner；项目增加 n-back、DSR、Visual 和 REFED regression，README 明确这些是 adapted variants，不可写成原论文数值复现。
3. **许可证与再分发（已验证）**：`method_manifest.yaml` 记录上游没有 license file，禁止把上游代码/权重当作可再分发资产；本项目随机初始化训练，没有 upstream checkpoint（manifest checkpoint status 为 not applicable）。
4. **数字准入字段缺失（已验证缺失）**：aggregate 没有 `table_admissible`、cell-level source/support disposition；需由统一 `METRIC_ACCEPTANCE.md`/总表流程外部标注。
5. **样本随机诊断不可泛化（已验证）**：sample-random 35 folds 不隔离 subject/record/trial dependency groups，不能与 strict 结果合并排名。
6. **formal 原生图缺失（已验证缺失，非科学阻塞）**：正式目录 0 PNG/0 SVG；主 PPT 必须重新制作每个项目的完整进度可视化。

## 8. 测试与证据质量

- [已验证] 运行 `.venv/bin/python -m pytest -q comparative_methods/STA-Net-PyTorch/tests/test_sta_net_pytorch.py comparative_methods/STA-Net-PyTorch/tests/test_fivefold.py`：**25 passed in 5.36 s**。
- [已验证] 只读核对 7 task spec、split registry summary、freeze manifest、3 lane status、70 个训练/评估目录、aggregate row counts 和数值范围；没有访问 protected prediction 数组。
- [已验证] `aggregate.log` 记录 `status=completed`, `job_count=70`, `task_count=7`；`paper_table.md` 明确 hyperparameters 在 outer-test 前冻结、70 folds 全部按 convergence rule 结束。
- [声明] 统一计划/状态文档将该结果定为 context reference；该审计不把“工件完整”推断成“支持匹配”或“原论文数值复现”。
- [推断] 代码覆盖和 hash pinning 足以支撑同一冻结工件的重审，但因未执行 protected 数组级复算，不能在本审计中声称重新验证每个预测数值。

## 建议交付标签

`STA-Net = completed_frozen_context_reference; formal_70_of_70; native_budget_3s+13s (DSR 2s+13s); not_support_matched_direct_table; no_new_protected_run`。
