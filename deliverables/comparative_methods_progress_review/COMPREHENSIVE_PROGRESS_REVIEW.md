# 对比方法全面进度审查

证据快照：2026-08-11。评分表示工程与比较准备度，不等于最终论文数字可用性。

| 方法 | 必要代码 | 输入 | 输出 | 结果生成 | 证据复现 | 加权总分 | Public jobs | 最终数字 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BIOT | 90 | 92 | 80 | 45 | 82 | 76.9 | 90/90 | 不可用（protected locked） |
| CBraMod | 97 | 97 | 88 | 52 | 88 | 83.5 | 90/90 | 不可用（protected locked） |
| REVE-base | 92 | 91 | 84 | 58 | 86 | 81.5 | 90/90 | 不可用（protected locked） |
| EFRM LODO v2 | 94 | 98 | 94 | 60 | 82 | 84.8 | 105/105 | 不可用（protected locked） |
| NormWear adapted | 96 | 94 | 90 | 68 | 88 | 86.9 | 90/90 | 不可用（protected locked） |
| BrainFusion NVC–CSP | 95 | 98 | 96 | 75 | 95 | 90.8 | 75/75 | 不可用（protected locked） |
| STA-Net | 93 | 84 | 92 | 94 | 88 | 90.8 | 70/70 | 有冻结 aggregate；仅 context reference |

## 总体判断

六个新增或适配方法均已完成 public A0–A8 证据链，但 protected evaluation 尚未授权，public summary 不能进入最终排名。STA-Net 已有冻结五折 aggregate，但观察预算属于 method-native context reference，不是当前 support-matched 主表证据。

## 复核命令

- `.venv/bin/python comparative_methods/build_joint_protected_unlock_candidate_v2.py --check`
- `.venv/bin/python -m pytest -q tests/test_joint_protected_unlock_candidate_v2.py tests/test_adapter_alignment_gate_contract.py`

## 方法审查明细

### BIOT

BIOT is public-delivery complete for six EEG-only classification cells: A0-A8 pass across 22,442 unique public samples and 90/90 serial selection/refit jobs with zero failures/retries. The frozen official PREST-16 encoder is safely hash-verified and strict-loaded, and the native-electrode positional-transfer deviation is documented. REFED regression is explicitly unsupported before scoring because partial terminal time support cannot be represented. Protected evaluation remains locked and no final aggregate or metric-acceptance decision exists; therefore BIOT cannot yet supply a final table number despite a 76.9 weighted implementation-progress score.

- Public：verified: six classification cells public-complete; A0-A8 pass; 90/90 public jobs complete; REFED predeclared unsupported
- Protected：verified: locked; protected_evaluation_authorized=false and protected_test_opened=false throughout retained BIOT evidence; zero protected jobs
- Final：missing: no protected predictions, final fold-seed aggregate, metric-acceptance decisions, or BIOT runs/aggregate artifact; public completion is table_admissible=false
- 详细报告：`agent_reports/biot.md`

### CBraMod

CBraMod 已完成官方上游/权重固定、200 Hz EEG 输入适配、pre-proj_out 200-d latent 平均池化、冻结 linear-probe head、A0-A8 alignment 和 90/90 public development jobs。六个支持分类任务的 22442 个唯一公开样本均完成 full coverage，90 个 job 全部通过且无重试；REFED 按 CBRAMOD_NO_PARTIAL_TIME_MASK_CONTRACT 事前 unsupported。当前 protected evaluation 仍锁定，public validation 与六任务开发均值明确 table_admissible=false，且没有正式 runs/aggregate 或 metric-acceptance 结果。因此工程/public 进度评分 83.5，但最终可用性能结果仍 unavailable；下一步只能在独立授权后执行冻结 protected 评估并生成正式聚合/准入。

- Public：complete
- Protected：locked_not_authorized
- Final：unavailable_no_protected_or_metric_accepted_aggregate
- 详细报告：`agent_reports/cbramod.md`

### REVE-base

REVE 的 public 适配链路已完成：六个 EEG-only classification cells 的 A0-A8 全部通过，22,442 个 unique public samples、90/90 public fold-seed jobs、模型/坐标 identity、mask/geometry、预测 schema 和 replay evidence 齐全；MI/MA 必须标为 Shin2017A overlap track，REFED 事前 unsupported。protected evaluation 从未授权或打开，public validation 结果全部 table_admissible=false，未形成可入最终论文表的 aggregate/metric-acceptance 数字。工程加权进度 81.5/100，但最终结果状态仍为 unavailable。

- Public：complete_public_development_6_of_6_supported_cells_90_of_90_jobs_a0_a8_pass_refed_unsupported
- Protected：locked_not_authorized_not_opened
- Final：unavailable_no_protected_or_metric_accepted_aggregate
- 详细报告：`agent_reports/reve.md`

### EFRM LODO v2

[已验证] EFRM-PyTorch LODO v2 已完成必要代码组件、输入/输出适配、Stage A 4/4、Stage B 4/4、7/7 A0-A8 cells 和 105/105 public jobs，0 failure/0 retry。[已验证] 所有 public outputs 均 target exposure false、protected false、table_admissible=false；protected_test_opened=false，v2 没有最终 protected aggregate。[缺失/阻断] 当前可交付的是 public pipeline/evidence，不是最终比较表数字；下一步需经统一 review 授权 protected evaluation，再生成 v2 aggregate/OOF 并通过指标准入。

- Public：complete_public_development_A0_A8_pass
- Protected：locked
- Final：protected_final_aggregate=unavailable_no_v2_aggregate; final_table=unavailable_metric_admission_not_run
- 详细报告：`agent_reports/efrm.md`

### NormWear adapted

NormWear EEG-fNIRS adapted public delivery is complete and reproducible through A8: pinned source/checkpoint, frozen EEG/HbO/HbR input contract, full public replay of 22442 identities, and 90/90 serial linear-probe jobs with zero failures/retries. Six classification cells pass A0-A8; REFED is explicitly unsupported for the fixed mask/regression boundary. Protected evaluation remains locked and no final/aggregate metric is available, so public validation means must not be reported as final table values or as an original-paper fNIRS reproduction.

- Public：complete: six classification cells A0-A8 pass; 22442 unique public samples replayed; 90/90 serial public selection/refit jobs complete; 0 failures and 0 automatic retries
- Protected：locked: protected_evaluation_authorized=false and protected_test_opened=false
- Final：unavailable: no protected evaluation, no table-admissible metric-acceptance aggregate, and no NormWear runs/aggregate directory
- 详细报告：`agent_reports/normwear.md`

### BrainFusion NVC–CSP

[已验证] BrainFusion 独立 NVC-CSP stacking 重实现已完成 v2 public delivery：五个 8 秒 support-matched direct 分类 cell A0-A8 全通过，13,462 个 public identity、75/75 serial jobs 和 75 个可 reload artifact 均通过审计；DSR/REFED 在 target performance 前声明 unsupported。[已验证] 输入、fold-local NVC/CSP/feature selection/stacking 与输出适配完整。[缺失/阻断] protected evaluation 未授权且未打开，public metrics 明确 table_admissible=false，尚无 C6 final-number 或 protected final result；当前可交付的是 public pipeline/evidence，不是最终论文数字或原论文 numerical reproduction。

- Public：complete_public_development_A0_A8_pass
- Protected：locked
- Final：protected_predictions=unavailable_locked; final_table=unavailable_C6_metric_admission_not_run
- 详细报告：`agent_reports/brainfusion.md`

### STA-Net

STA-Net-PyTorch 已完成 7 任务、strict 与 sample-random 两条五折协议共 70/70 个正式 job；70 个训练均通过至少 40 epoch + 30 个无改进 validation 的收敛规则，70 个评估摘要和 8 个 aggregate 工件均在位，25 个方法测试全部通过。输入/输出适配覆盖 EEG+HbO/HbR、16×16 空间投影、valid-mask、六分类与 REFED mask-aware sequence regression。由于 EEG 3 s + fNIRS 13 s（DSR 2+13 s）是 method-native context，和当前 support-matched EFRM 8/8 s（DSR 2/2 s）不一致，结果必须标为 method_native_context_reference，不能直接进入 support-matched 总表；不得重跑、重选或重新开放 protected 结果。

- Public：complete_and_tested; 7/7 smoke tasks, 25/25 method-local tests, tuning/freeze/training/evaluation/aggregation code present
- Protected：historical_formal_evaluation_opened_and_completed_70_of_70; no_new_protected_evaluation_authorized_or_run
- Final：aggregate_available=true; support_matched_table=not_eligible; strict_status=available_as_method_native_context_reference
- 详细报告：`agent_reports/stanet.md`
