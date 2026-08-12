# BrainFusion NVC–CSP Stacking 进度审查

审计日期：2026-08-11（Asia/Shanghai）
审计对象：`brainfusion_nvc_csp_stacking_reimplementation`
审计目录：`comparative_methods/BrainFusion-NVC-CSP-Stacking`

## 结论

**当前状态：public delivery 已完成，protected evaluation 锁定；没有可填入最终比较表的 BrainFusion 数字。**

- [已验证] 五个 8 秒分类 cell（MI、MA、WG、n-back、Visual）均为 `support_matched_direct`、`public_complete`、A0–A8 全 `pass`；2 个 cell（DSR、REFED regression）在 target performance 前静态声明 `unsupported`。证据见 `evidence/alignment_v2/summary_final.json:20-98` 及各 cell JSON。
- [已验证] 13,462 个支持任务的 unique public identities 已通过 production data view 审计一次：MI 1,740、MA 1,740、WG 1,560、n-back 702、Visual 7,720。`ALIGNMENT_V2_STATUS.md:10-24`、`summary_final.json:2-18`。
- [已验证] public matrix 为 75/75 jobs（5 tasks × 5 outer folds × 3 seeds），0 failures、0 retries、串行 GPU `cuda:1`；每个 job 的 membership、targets、macro-F1、cache/raw 一致性和 checkpoint reload 均被 retained audit 重算。`evidence/public_development_v2/matrix_completion_summary.json:1-20`；`runs/public_development_v2/matrix_v2/completed_public_audit.json` 含 75 条 audit reports。
- [已验证] 所有 NVC pair selection、CSP、feature standardization、base-estimator selection、OOF stacking/meta-estimator 都绑定 outer-training identities；代码与 checkpoint audit 均写明该边界。`OBSERVATION_BUDGET_AUDIT.md:49-55`、`adapters/brainfusion_gpu/features.py:197-346`、`stacking.py:110-343`、`pipeline.py:32-154`。
- [声明/边界] 官方 checkout 只公开 scalar NVC；GUI 的 `NVC CSP`/`Integrated Model` 是 simulation placeholder，因此本项目必须使用独立重实现名称，不能声称复现论文 95.5% 数字。`README.md:3-18`、`sources/SOURCE_FIDELITY.md:3-17`、`sources/method_manifest.yaml:29-41`。
- [已验证] public validation 均保留为 development-only diagnostics，`table_admissible=false`；protected evaluation `authorized=false` 且 `opened=false`。`configs/public_development_v2.yaml:37-41`、`matrix_completion_summary.json:16-20`、`summary_final.json:9-18`。

## 统一维度评分（审计判断）

权重为 code components 30、input adaptation 20、output adaptation 15、result generation 25、evidence/reproducibility 10；分数是本次审计判断，不是仓库原生字段。

| 维度 | 分数 | 依据与未完成项 |
|---|---:|---|
| code_components | **95/100** | [已验证] NVC CPU reference/GPU kernel、dynamic contributions、NVC selector、四路 CSP、四路 base learner、grouped OOF、linear-SVM meta、public loader、cache、runner、matrix controller/finalizer 均有实现和测试；[声明] CSP/stacking 是独立重实现而非官方 case pipeline。 |
| input_adaptation | **98/100** | [已验证] canonical EEG 200 Hz/fNIRS 10 Hz、三模态、0–8 s、真实 channel inventory、recorded/analysis masks、无复制/填充；13,462/13,462 public identity audited。扣分仅因 DSR/REFED 不支持且 protected 未开放。 |
| output_adaptation | **96/100** | [已验证] 四个 CSP views→selected NVC CSP→OOF linear-SVM stack；75/75 checkpoint/prediction artifacts reload exact。扣分因无 masked sequence-regression output（REFED unsupported）。 |
| result_generation | **75/100** | [已验证] 75/75 public-development jobs、每任务 15 jobs、均值/SD 和 75 run reports 已生成；[缺失] protected final predictions、跨方法 table-admissible 数字和 C6 metric-admission decision 尚不存在。 |
| evidence_reproducibility | **95/100** | [已验证] A0–A8 schema audit pass、源码/config/manifest/hash、75 audits、38 个本地测试通过；[风险] run-level report 未显式保存 `device` 字段，矩阵级 summary 才记录 `cuda:1`；另有 pair-count 字段命名风险（见下）。 |
| **overall（加权）** | **90.85/100** | `0.30×95 + 0.20×98 + 0.15×96 + 0.25×75 + 0.10×95 = 90.85`。状态仍是“public complete / protected locked”，不是最终结果 ready。 |

## 任务、输入/输出适配与结果状态

### 已支持任务（`support_matched_direct`）

| task | public unique samples | 输入（每 item） | 输出（每 item） | public validation macro-F1（15 jobs，诊断） |
|---|---:|---|---|---:|
| motor_imagery | 1,740 | EEG `[30,1600]`；HbO/HbR `[36,80]` | 四路 CSP `[4]`；stack decision `[2]` | 0.52686 ± 0.01355 |
| mental_arithmetic | 1,740 | EEG `[30,1600]`；HbO/HbR `[36,80]` | 四路 CSP `[4]`；stack decision `[2]` | 0.52697 ± 0.02750 |
| wg | 1,560 | EEG `[28,1600]`；HbO/HbR `[36,80]` | 四路 CSP `[4]`；stack decision `[2]` | 0.53157 ± 0.03051 |
| nback | 702 | EEG `[28,1600]`；HbO/HbR `[36,80]` | 四路 CSP `[6]`；stack decision `[3]` | 0.35467 ± 0.03066 |
| visual | 7,720 | EEG `[30,1600]`；HbO/HbR `[24,80]` | 四路 CSP `[8]`；stack decision `[4]` | 0.24029 ± 0.02362 |

数值均来自 `evidence/public_development_v2/matrix_completion_summary.json:21-97`，只能作为 public-development diagnostics；不得作为 final table performance。

### 预注册不支持任务

- [已验证/声明] **DSR**：2 s synchronized block context 小于配置的 `minimum_supported_nvc_interval_s: 8.0`，不能扩展观察区间；只跑 modality-specific CSP 会改变注册方法。`configs/alignment_v2.yaml:25-33,83-87`、`evidence/alignment_v2/dsr.json`（`BRAINFUSION_NVC_TWO_SECOND_CONTEXT_UNSUPPORTED`）。
- [已验证/声明] **REFED regression**：源 case 是 classification stack，缺乏 masked continuous regression 和 partial-terminal-support 合同；不能把 mask 或 padded time points 强转成分类。`configs/alignment_v2.yaml:93-97`、`evidence/alignment_v2/refed_regression.json`（`BRAINFUSION_CLASSIFICATION_ONLY_AND_NO_PARTIAL_MASK_CONTRACT`）。

## 必要代码组件完成度

| 组件 | 状态 | 证据（相对路径与行数） |
|---|---|---|
| upstream/source boundary | **完成（component fidelity）** | pinned revision `1d9dcf4026f237efed7f0dd44ba44ef0bf87915b`；NVC source 11,140 bytes、SHA256 在 `sources/method_manifest.yaml:11-41`；官方 CSP/stacking case execution 缺失。 |
| public data/identity adapter | **完成** | `alignment_data.py` 378 lines；config/registry/mask/channel checks `alignment_data.py:66-124,165-218,244-378`。 |
| NVC | **完成** | CPU reference + CUDA `avg_raw` `adapters/brainfusion_gpu/nvc.py:30-146`；dynamic Pearson contributions `:149-215`。 |
| feature selection/CSP | **完成，fold-local** | NVC pair selector、regularized one-vs-rest CSP、state identity `adapters/brainfusion_gpu/features.py:35-194,197-346`。配置为 32 selected pairs、2 components/class、regularization 0.1，`configs/public_development_v2.yaml:15-18`。 |
| stacking | **完成，fold-local** | per-view SVM/RF candidates、5 grouped inner folds、OOF scores、linear-SVM meta、state hash `adapters/brainfusion_gpu/stacking.py:110-343`。 |
| serialization/reload | **完成** | feature `.pt` + estimator `.joblib` + hash-bound manifest，`adapters/brainfusion_gpu/pipeline.py:90-154`；75 个 matrix run 各有 `run_report.json`、`checkpoint/manifest.json`、`public_validation_predictions.json`。 |
| public runner/cache/matrix/finalizer | **完成 public scope** | cache identity and raw replay `run_public_development_v2.py:234-373`；runner `:401-573`；75-job matrix `build_public_job_matrix_v2.py:92-165` 与 `run_public_matrix_v2.py`；finalizer `finalize_public_matrix_v2.py:54-269`。 |

## fold-local 与模型 I/O 审计

- [已验证] **NVC**：每样本每通道 min-max → causal SPM HRF（32 s kernel，full convolution 后 crop 到 observed 8 s）；dynamic contributions 沿时间求和等于 Pearson coefficient，`tests/test_fold_local_features.py` 覆盖该不变量。
- [已验证] **feature selection/CSP**：selector、CSP filters、每 view feature states 使用同一 `fit_sample_identity_sha256`；`features.py:274-319` 的 audit/state_dict 复核一致性；重复训练 synthetic test deterministic。
- [已验证] **stacking**：5 个 `StratifiedGroupKFold` inner folds 无 group overlap，validation membership 对 outer-train 样本恰好覆盖一次；candidate selection 仅使用 inner OOF macro-F1，meta 只看 selected base OOF scores，`stacking.py:138-269`。
- [已验证] **输入**：`support_matched_direct`，anchor 为 canonical registry window start，pre/post extra context 均 0；EEG 200 Hz、fNIRS 10 Hz，三模态 required，`configs/alignment_v2.yaml:17-49`。
- [已验证] **输出**：CSP log-normalized variance features → per-view standardized SVM/RF scores → linear-SVM stacking decision；predictions 与 decision scores 以 `brainfusion_public_predictions_v2` 保存，protected flag 为 false。
- [声明] 无 pretrained checkpoint 是有意的 supervised fold-local 边界，不是缺失的模型权重；`README.md:3-5`、`sources/method_manifest.yaml:21-23`。

## 最终结果生成阶段

1. [已验证] smoke pilot：`evidence/public_development_v2/pilot_audit.json`，36 train / 16 validation，status pass。
2. [已验证] full-fold pilot：`full_fold_pilot_audit.json`，900 train / 480 validation，cache↔raw、target/metric、checkpoint reload 均 pass。
3. [已验证] terminal public matrix：5 tasks × 5 folds × 3 seeds = **75/75**，0 failed、0 retries、serial `cuda:1`；每 task 15 jobs；`matrix_completion_summary.json:1-20`。
4. [已验证] A8 freeze：`summary_final.json` status `public_development_complete_A0_A8_pass_protected_locked`，matrix identity 与 completion hash 均保留。
5. [缺失/阻断] protected evaluation 未授权、未打开；没有 protected predictions 或 protected aggregate。
6. [缺失] C6/final-number admission（baseline、minimum/preferred target、source relation、decision）未生成；public summary 明确 `table_admissible=false`。

## 阻塞、风险与边界

### 阻塞项

- [已验证] `protected_evaluation_authorized=false`、`protected_test_opened=false`，见 `evidence/alignment_v2/summary_final.json:9-18`；必须等待统一 unlock review，不能由 public evidence 自行解锁。
- [已验证] DSR/REFED 是终态 `unsupported`，不是待补跑的 failed job；证据中没有 public inventory dereference，也没有 target performance。
- [声明] 官方公开源无法恢复 paper-case CSP/ensemble，因此原论文数字 reproduction 永久不允许，除非另有完整 source release；`sources/SOURCE_FIDELITY.md:7-17`。

### 风险（不改变当前 A0–A8 public 状态）

- [已验证] 元数据命名风险：`audit_alignment_v2.py:225-230` 将 `nvc_unselected_pair_count` 计算为 `EEG×fNIRS×2` 总 pair 数；因此 evidence 显示 MI/MA 2160、WG/n-back 2016、Visual 1440，而真实未选数应分别为 2128、1984、1408（各自仍选 32）。实际 selector 使用 `features.py:156-194`，本风险不等同于 feature leakage，但会误导 pair-count 可视化。
- [已验证] 运行级复现元数据：`matrix_completion_summary.json:5` 明确记录 `device: cuda:1`，但单个 `run_report.json` 未保存 device 字段；建议后续报告把 device 写进每个 run manifest，避免仅依赖 controller summary。
- [推断] public validation 均值（例如 Visual 0.24029）不可解释为新主体泛化或论文复现；它们是 outer public validation 上的 development diagnostics，且尚未经过 protected/C6 准入。
- [声明/文档漂移] `tests/README.md:18` 仍写着 full-public A0–A8 “pending”，与 `summary_final.json` 的最终 `A8 pass` 不一致；权威状态应以 retained final evidence 为准，后续可只做文档同步。

## 测试与独立审计记录

- [已验证] 执行：`.venv/bin/python -m pytest -q comparative_methods/BrainFusion-NVC-CSP-Stacking/tests/{test_fold_local_features.py,test_fold_local_stacking.py,test_pipeline_reload.py,test_gpu_nvc.py,test_alignment_audit_v2.py,test_alignment_data_v2.py,test_public_runner_v2.py}` → **38 passed in 8.96 s**。
- [已验证] 执行：`.venv/bin/python comparative_methods/audit_adapter_alignment.py comparative_methods/BrainFusion-NVC-CSP-Stacking/evidence/alignment_v2/{motor_imagery,mental_arithmetic,wg,nback,visual,dsr,refed_regression}.json` → `status=pass`；7 cell reports、9 gates、5 direct groups（每组 1 member）检查通过，protected flag false。
- [已验证] retained full-fold pilot：`evidence/public_development_v2/full_fold_pilot_audit.json` → `status=pass`、`cached_validation_matches_raw_adapter=true`、`checkpoint_reload_exact=true`。
- [缺失] 本次没有重跑完整 13,462-sample A0–A7 production audit 或 75-job训练；为遵守“轻量验证、禁止高成本训练”，使用仓库 retained evidence 做一致性审查。

## 证据质量

**高（public scope）/中（final-result scope）。** [已验证] 证据包同时保留 cell declarations、comparison fields、source/config hashes、全量 public replay summary、cache identity、75 run audits、checkpoint/prediction artifacts 与 freeze flags；[缺失] protected arrays/predictions、最终 metric-admission records、官方完整 CSP/stacking source。故可审计 public pipeline，但不能产出最终正式数字或原论文 numerical reproduction。
