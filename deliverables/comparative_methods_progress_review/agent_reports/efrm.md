# EFRM-PyTorch 全面进度审查

审计日期：2026-08-11（Asia/Shanghai）
审计对象：EFRM-PyTorch 独立适配（正式项目：comparative_methods/EFRM-PyTorch）
审计范围：只读核查协议、代码、配置、测试、证据包和 runs/formal；未读取 protected 数组/身份，未解锁门控，未启动训练，也未修改方法实现。

## 执行结论

**状态：public pipeline 完成，protected evaluation 锁定，v2 最终结果不可用。**

- [已验证] EFRM LODO v2 的 Stage A target-excluded selection 为 4/4，Stage B full non-target refit 为 4/4；四个排除目标及所选 epoch 分别为 Single-Trial 45、REFED 90、Simultaneous 89、Visual 35。证据：runs/formal/efrm_lodo_full_target_fivefold_v2/status.json、protocol/selections/（4 个 JSON）、protocol/final_refits/（4 个 JSON）。
- [已验证] 七任务 downstream public matrix 为 105/105（7×5 outer folds×3 seeds），0 failed、0 retries；所有 retained public outputs 均标记 protected=false、target_dataset_exposure=false、table_admissible=false。证据：evidence/public_development_v2/matrix_completion_summary.json、runs/formal/efrm_lodo_full_target_fivefold_v2/downstream_public_v2/a7_complete_matrix/completed_public_audit.json。
- [已验证] alignment v2 为 7/7 cells 通过 A0–A8；public unique sample count 合计 25,162（分类 22,442，REFED 2,720）。证据：evidence/alignment_v2/summary_final.json 及各 cell JSON。
- [已验证] protected_evaluation_authorized=false、protected_test_opened=false；v2 formal root 没有 aggregate/、protected/ 或正式最终聚合。不能把 public diagnostics 写成最终比较表数字。
- [历史来源，不可混用] efrm_resource_bounded_dual_protocol_v1 曾完成 70 public+70 protected，但其估计量和协议不同，不能替代当前 v2；v1 历史数值仅作来源证据，不作为本审计结果。

## 统一维度评分

权重固定为 code_components 30%、input_adaptation 20%、output_adaptation 15%、result_generation 25%、evidence_reproducibility 10%。分数为本次审计判断，不是仓库原生字段。

| 维度 | 分数 | 判断依据 |
|---|---:|---|
| code_components | **94/100** | [已验证] 数据/模型/任务/协议、LODO runner、public matrix builder/runner/finalizer、审计器和 3 个测试文件均存在；[扣分] 源码是独立 retraining，官方 checkpoint 不可得，且 v2 aggregate 尚未实现。 |
| input_adaptation | **98/100** | [已验证] EEG 200 Hz、fNIRS 10 Hz，physical patches 50/20 samples，variable-channel、no copy/mirror、invalid-support mask、七任务窗口和 target-exclusion guard 均有配置与代码；[扣分] protected input 尚未开放。 |
| output_adaptation | **94/100** | [已验证] encoder means→768-d paired fusion→LayerNorm/Dropout/Linear probe，REFED coordinate mask 与 partial-input mask 均保留；[扣分] public outputs 明确 development-only，尚无 protected final prediction/aggregate。 |
| result_generation | **60/100** | [已验证] Stage A/B 8 个 LODO jobs 和 105 个 public jobs 均完成；[缺失] v2 protected evaluation、OOF final aggregate、C6/table-admission 数字尚未生成。 |
| evidence_reproducibility | **82/100** | [已验证] queue、manifests、checkpoint hashes、105 run reports、A0–A8 summary 和 34 个 targeted tests 可复核；[风险] 8 个 run-level status.json 仍显示 stale running，和 completed manifests/queue 不一致；pretraining quality comparison 仅覆盖 2 个 Stage-A runs。 |
| **overall（加权）** | **84.8/100** | 0.30×94 + 0.20×98 + 0.15×94 + 0.25×60 + 0.10×82 = **84.8**。这表示 public implementation ready，不表示 final-table ready。 |

## 任务与输入适配

EFRM v2 预注册 7 个 canonical tasks，均有 public cell；每个 cell 15 jobs，共 105。

| task | 数据/窗口 | public sample count | LODO refit epoch | 输入适配状态 |
|---|---|---:|---:|---|
| motor_imagery | eeg_fnirs_single_trial，0–8 s | 1,740 | exclude Single-Trial：45 | [已验证] EEG+fNIRS paired，variable channels，mask |
| mental_arithmetic | eeg_fnirs_single_trial，0–8 s | 1,740 | exclude Single-Trial：45 | [已验证] 同上 |
| wg | simultaneous_eeg_nirs，0–8 s | 1,560 | exclude Simultaneous：89 | [已验证] 同上 |
| nback | simultaneous_eeg_nirs，0–8 s | 702 | exclude Simultaneous：89 | [已验证] 同上 |
| dsr | simultaneous_eeg_nirs，0–2 s EEG + synchronized fNIRS context | 8,980 | exclude Simultaneous：89 | [已验证] short EEG window 与同步 fNIRS 适配 |
| visual | visual_cognitive_motivation，0–8 s | 7,720 | exclude Visual：35 | [已验证] paired EEG+fNIRS |
| refed_regression | REFED，0–20 s sequence regression | 2,720 | exclude REFED：90 | [已验证] masked coordinate regression，partial input mask |

证据：comparative_methods/EFRM-PyTorch/README.md；configs/pretrain_sync.yaml；configs/downstream_public_v2.yaml；evidence/alignment_v2/summary_final.json。
输入关键约束：[已验证] EEG 200 Hz、fNIRS 10 Hz；patch size EEG 50 samples（0.25 s）、fNIRS 20 samples（2 s）；通道数按真实记录变化，不复制或镜像；无效支持由 mask 传播。
任务支持结论：[已验证] supported_tasks=7，unsupported_tasks=0。这里的 supported 表示 public pipeline 能按冻结合同执行，不等于 protected performance 已通过。

## 必要代码组件完成度

| 组件 | 状态 | 精确计数/路径 |
|---|---|---|
| upstream/source boundary | [已验证] 有界 | sources/method_manifest.yaml：upstream revision a62bf3d4c092ac3022b6c0bad90ec3993d5a5720；upstream checkout 11 个 tracked files，无 LICENSE；official checkpoint unavailable，random-init retraining |
| data and identity adapter | [已验证] 完成 | efrm_pytorch/data.py，672 lines；variable-channel sample construction、rate conversion、mask/identity checks |
| model | [已验证] 完成 | efrm_pytorch/model.py，447 lines；dynamic MAE/CLIP/transfer，embedding 768 |
| tasks/metrics | [已验证] 完成 | efrm_pytorch/tasks.py，310 lines（7 tasks）；metrics.py，179 lines（classification + regression/CCC） |
| protocol and guards | [已验证] 完成 | efrm_pytorch/protocol.py，547 lines；public/protected boundary、target exposure 和 fold guards |
| pretraining runner | [已验证] 完成 | train_pretrain.py 611 lines；run_lodo_pretraining.py 417 lines；LODO queue/protocol manifests |
| downstream public runner | [已验证] 完成 | run_downstream_public_v2.py 1,489 lines；frozen checkpoint, outer fold, linear probe/refit, mask-aware REFED |
| matrix build/run/finalize/audit | [已验证] 完成 public scope | build_downstream_public_matrix_v2.py 116；run_downstream_public_matrix_v2.py 329；finalize_downstream_public_matrix_v2.py 202；audit_downstream_public_run_v2.py 242 |
| package/test inventory | [已验证] | 22 top-level Python scripts、9 package modules、3 test files；总计约 13,459 lines（8,630 + 3,745 + 1,084） |

## 模型输入/输出适配

### 输入

- [已验证] pretraining config 固定 sample rates 200/10、window 8 s、physical patches 50/20、mask ratio 0.5、contrastive multiplier 0.1、seed 42、100 epochs、effective batch 32（micro-batch 8）。
- [已验证] downstream config 固定七任务、outer folds 0–4、seeds 17/42/73、expected_public_jobs=105，target scaling 只在 outer-train/refit 内拟合；protected 默认 locked。
- [已验证] protocol/lodo_manifests/ 恰有 4 个 target-exclusion manifests；每个 manifest 写明 target exposure false。shared_full_target_fold_registry/registry_manifest.json 含 70 个 public/protected fold metadata entries（7 tasks×2 protocols×5 folds）；本次仅读元数据，未 dereference protected arrays。
- [已验证] REFED cell 记录 partial_input_sample_count=480，并保留 valence-arousal 2×20 coordinate mask；无伪造完整支持。

### 输出

- [已验证] EFRM encoder 输出 embedding_dim=768；paired fusion 是 elementwise_sum_before_trainable_layer_norm；下游 head 为 trainable LayerNorm→Dropout→Linear probe only；每个 cell 的 metadata 都标记 nonconstant_features=true。
- [已验证] 七个 cell 的 output adapter identity、checkpoint/selection report、prediction NPZ 和 status/manifest 均保留；REFED 输出为 masked_native_coordinate_ccc。
- [边界] [缺失] 当前 public output 的 table_admissible=false；没有 protected predictions、OOF aggregate、最终 table-ready confidence interval。

## 结果生成阶段

~~~
Stage A: 4/4 target-excluded selections
   -> Stage B: 4/4 full non-target refits (terminal checkpoint)
   -> downstream public: 105/105, 0 failure, 0 retry
   -> alignment v2: 7/7 cells, A0-A8 pass
   -> protected unlock: 未授权/未打开
   -> v2 aggregate/final table: 不存在
~~~

### Stage A/B 精确核查

- [已验证] selections 目录恰有 4 个 JSON，schemas 为 efrm_lodo_stage_a_selection_v2；selected epochs 为 45、90、89、35；每份 target exposure=false。
- [已验证] final_refits 目录恰有 4 个 JSON，schemas 为 efrm_lodo_stage_b_refit_freeze_v2；对应 terminal checkpoint SHA 字段且 target exposure=false。
- [已验证] formal status.json：selection_completed=4、final_refit_completed=4、lodo_selection_jobs=4、lodo_final_refit_jobs=4、status=lodo_pretraining_completed、protected_test_opened=false。
- [已验证] pretraining_queue/state.json：schema efrm_lodo_pretraining_queue_v2、status=completed、selection_completed=4、final_refit_completed=4、last_child_exit_code=0。
- [已验证] 8 个 runs/pretraining/efrm_lodo_full_target_fivefold_v2__*stage_{a,b}_seed42/manifest.json 均 status=completed、protected=false、target exposure=false；Stage B terminal checkpoint 文件存在。
- [风险] 与上述证据并存的是 8 个 run 目录下的 status.json 仍为 status=running（stale epoch/无 schema）。这是 telemetry inconsistency，不足以推翻 queue、manifest、formal status 的完成结论；应在后续维护中统一状态源。

### Public 105-job 核查

- [已验证] matrix_completion_summary.json：status=pass、completed_job_count=105、job_count=105、failed_job_count=0、automatic_retry_count=0、max_concurrent_jobs=1、protected_evaluation_authorized=false、protected_test_opened=false、target_dataset_exposure=false、table_admissible=false。
- [已验证] 7 个 task 目录各有 15 个 manifest.json + 15 个 status.json，合计 105；所有 manifest status=completed、mode=public_selection_and_refit、protected=false、target exposure=false、table=false。
- [已验证] completed_public_audit.json 的 run_reports 长度为 105；全部 status=pass、table_admissible=false、protected=false。

## Protected 与最终结果可用性

| 项目 | 状态 | 证据/含义 |
|---|---|---|
| public pipeline | **complete** | 105/105 jobs，7/7 A0-A8 cells pass |
| protected authorization | **false** | evidence/alignment_v2/summary_final.json；不能自行解锁 |
| protected test opened | **false** | 同上；本次未访问 protected arrays |
| v2 protected predictions | **unavailable_locked** | formal v2 下无 protected 输出目录 |
| v2 aggregate/oof | **missing** | runs/formal/efrm_lodo_full_target_fivefold_v2/aggregate/ 不存在 |
| final table numeric cells | **unavailable** | public table_admissible=false；metric acceptance 不允许代填 |
| historical v1 | **completed but non-comparable** | v1 70 public+70 protected 是不同 estimand，不能当 v2 final |

## 阻塞与风险

### Blockers

1. [已验证] protected gate 未授权且未打开；需要统一 evidence/joint review 与独立一次性 protected evaluation。
2. [缺失] v2 aggregate/OOF/final-number admission 尚未生成；现有 aggregate_formal_results.py 是历史 v1、固定 70-job 计数且要求 protected status=true，不能直接生成 v2 final。
3. [声明/来源边界] official checkpoint unavailable，method 是 source-faithful independent retraining；不能声称 original checkpoint reproduction 或 physiological causality。

### Risks

- [已验证] 8 个 run-level status.json stale running 与 completed manifests/queue 不一致，可能误导进度可视化。
- [已验证] pretraining_quality_audit_20260729/comparison_summary.json 只列出 2 个 Stage-A runs；它不能代表四个 target exclusions 的完整质量审计。
- [推断] public validation diagnostic 不是 protected new-subject generalization；任何 macro-F1/CCC 直接进总表都会越过 table-admission 边界。
- [声明] 上游目录无 LICENSE 文件（method_manifest.yaml 已记录），分享/再分发边界需继续保留。

## 测试与轻量验证

- [已验证] 命令：PYTHONPATH=.:comparative_methods/EFRM-PyTorch .venv/bin/pytest -q comparative_methods/EFRM-PyTorch/tests/test_efrm_pytorch.py comparative_methods/EFRM-PyTorch/tests/test_efrm_formal_protocol.py comparative_methods/EFRM-PyTorch/tests/test_efrm_downstream_public_v2.py → **34 passed in 14.13 s**。
- [已验证] 命令：PYTHONPATH=.:comparative_methods/EFRM-PyTorch .venv/bin/python -m compileall -q comparative_methods/EFRM-PyTorch/efrm_pytorch comparative_methods/EFRM-PyTorch/*.py → **PASS**。
- [已验证] 命令：PYTHONPATH=. .venv/bin/python comparative_methods/audit_adapter_alignment.py → **PASS**（schema adapter_alignment_audit_report_v2，contract v2，gate IDs A0-A8，7 tasks/profile checks，protected=false）。
- [备注] 首次 pytest 使用 PYTHONPATH 仅指向 EFRM 目录，因环境缺少顶层 src 而 collection 失败；修正 PYTHONPATH=.:comparative_methods/EFRM-PyTorch 后通过。该失败是测试环境 invocation 问题，不是方法测试失败。
- [遵守范围] 未运行训练、完整 public matrix、全仓测试或任何 protected data dereference。

## 证据质量

**public pipeline：高；final-result：中低。**

- [已验证] 协议 freeze、4+4 LODO manifests、8 completed run manifests、queue state、105-job completion summary、105 run reports、7-cell A0-A8 alignment summary、checkpoint/prediction artifacts 和 targeted tests 均可复核。
- [声明] source-faithful retraining、selected epoch 和 output adapter identity 由 manifest/config 宣布并与代码路径一致。
- [缺失] protected arrays/predictions、v2 aggregate/OOF、C6 metric-admission decision、四个 Stage-A 的完整统一质量 comparison。

## Executive summary

**[已验证]** EFRM-PyTorch v2 已完成必要代码组件、输入/输出适配、4/4 Stage A、4/4 Stage B 和 105/105 public jobs；7/7 alignment cells 通过 A0-A8。
**[已验证]** public evidence 仍明确 protected locked、target exposure false、table_admissible=false；没有可用于最终比较表的 v2 数字。
**[缺失/阻断]** 下一步不是重跑 public pipeline，而是等待统一 protected unlock review，执行一次性 protected evaluation、生成 v2 aggregate/OOF 并通过指标准入。历史 v1 的 70+70 结果不能回填 v2。
