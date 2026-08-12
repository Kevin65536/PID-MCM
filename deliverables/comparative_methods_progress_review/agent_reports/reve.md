# REVE 对比方法进度审查

审计日期：2026-08-11
审计边界：只读仓库文件与已保留机器工件；未读取 protected manifest/array，未解锁门控，未启动训练。计划文档用于解释目标，结论以当前代码、状态和工件为准。

## 结论先行

REVE 的六个支持分类单元（motor imagery、mental arithmetic、WG、n-back、DSR、Visual）已经完成 public development：A0–A8 均为 pass，22,442 个公开样本各审计一次，90/90 个 public selection/refit jobs 完成，0 失败、0 自动重试、串行并发上限为 1。Single-Trial 的两个任务必须留在 `open_world_pretrained_with_target_corpus_overlap`，不能宣称 target-excluded；REFED 在看分数前登记为 unsupported，因为冻结的 REVE 快照不能把 REFED 的部分 terminal 时间支持从 transformer attention/query pooling 中屏蔽。

这不等于有可填入最终论文表格的数字：所有现有 REVE 结果仍明确标记 `table_admissible: false`，没有 protected evaluation、最终 protected aggregate 或 METRIC_ACCEPTANCE 准入记录。因此当前状态是“public pipeline complete；protected locked；final result unavailable”。

## 加权评分

评分维度按统一权重 `30/20/15/25/10` 计算；overall 为 `0.30×92 + 0.20×91 + 0.15×84 + 0.25×58 + 0.10×86 = 81.5`。分数只描述工程进度，不能越过 protected 或最终数字门控。

| 维度 | 分数 | 审计判断 |
| --- | ---: | --- |
| code_components | 92 | 上游 revision、base checkpoint、official position bank、坐标感知 adapter、public runner/controller/finalizer、A0–A8 与 public-run audit、单元测试均存在并有 retained evidence；正式 protected runner/aggregate 未实现为可用终态，且若干 README/manifest 仍写 pending。 |
| input_adaptation | 91 | 六个分类 cell 的 support-matched direct 输入合同和 geometry/identity 均通过；200 Hz、16 个真实通道、mask/branch/split/hash 可追溯；REVE 原生不支持部分 terminal temporal mask，REFED 因此 unsupported；8 s/2 s 窗口会丢弃不完整末 patch。 |
| output_adaptation | 84 | final latent token→冻结 query attention→512-d embedding→仅线性 head 的边界清楚，分类预测和失败码可审计；没有 REFED masked-regression 输出路径，也没有最终 metric-acceptance/aggregate 输出。 |
| result_generation | 58 | public 矩阵 90/90 及每 job checkpoint/prediction/manifest/feature cache 完整；只有 public validation development diagnostics，未开放 protected，未生成 table-admissible final aggregate。 |
| evidence_reproducibility | 86 | source/config/runner/registry/cache/checkpoint/prediction hash 链、deterministic replay、completion summary 和测试齐全；current queue guard 阻止历史 runner 直接 dry-run，文档状态与终态工件不一致。 |
| **overall** | **81.5** | 不掩盖 protected/final gates；最终结果仍不可用。 |

## 门控与任务覆盖

| 项目 | 当前状态 | 证据与计数 |
| --- | --- | --- |
| B0/source identity | `pass_with_gated_weight_access`（已验证工件；manifest 为线索） | 上游 Git revision `06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7`；`reve-base` snapshot `fa9a2163…944dc`；`reve-positions` snapshot `befa5b57…0bace`。见 `comparative_methods/REVE/sources/method_manifest.yaml`、`IDENTITY_AND_REPRESENTATION_AUDIT.md`。 |
| B1/A0–A7 | 六个支持 cell 通过 | `evidence/alignment_v2/{motor_imagery,mental_arithmetic,wg,nback,dsr,visual}.json`；每 cell A0–A7=pass，public_complete。 |
| A8/freeze | 六个支持 cell 已通过；REFED N/A | `summary_final.json` 为 `public_development_complete_A0_A8_pass_protected_locked`，每个支持 cell A8=pass；REFED A8=`not_applicable`。 |
| B2 smoke | pass（局部验证） | `tests/test_reve_smoke.py` 的有限 forward/backward、optimizer step、冻结、reload、坐标拒绝逻辑；本次 4/4 通过。 |
| B3 source fidelity | `official_reve_base_position_bank_linear_probe_v1` | 保留了官方 patch/overlap/attention pooling/512-d boundary；不是原论文任务数值的精确复现。见 `sources/SOURCE_FIDELITY.md`。 |
| B4 formal protocol/unlock | **pending/未授权** | public config 明确 `public_development_only`，`protected_test_default: locked`；无 protected manifest 解引用。旧 manifest 的 `B4_protocol_freeze: pending_formal_unlock...` 与最终 public A8 evidence 需在后续文档同步。 |
| B5 public execution | complete | `matrix_completion_summary.json`、`runs/public_development_v2/matrix_v2/controller_status.json`：90 planned/90 completed，0 failed，0 retries，6 tasks×5 folds×3 seeds。 |
| B6 final numeric acceptance | **not started** | 没有 REVE aggregate/metric-acceptance artifact；completion summary 明确 `table_admissible: false`。 |

### 任务和 job 计数

本报告的 `planned_job_count=90` 专指已授权 public development 矩阵（支持的 6 个任务×5 outer folds×3 seeds）；protected job 尚未获授权，未虚构其数量。原统一七任务矩阵中的 REFED 被事前 unsupported 处置，不进入 90 个 REVE public jobs。

| task | 支持 | track | public unique samples | planned/completed public jobs |
| --- | --- | --- | ---: | ---: |
| motor_imagery | 是 | `open_world_pretrained_with_target_corpus_overlap` | 1,740 | 15/15 |
| mental_arithmetic | 是 | `open_world_pretrained_with_target_corpus_overlap` | 1,740 | 15/15 |
| wg | 是 | `single_modal_eeg_official_pretrained_linear_probe` | 1,560 | 15/15 |
| nback | 是 | `single_modal_eeg_official_pretrained_linear_probe` | 702 | 15/15 |
| dsr | 是 | `single_modal_eeg_official_pretrained_linear_probe` | 8,980 | 15/15 |
| visual | 是 | `single_modal_eeg_official_pretrained_linear_probe` | 7,720 | 15/15 |
| refed_regression | 否 | preregistered unsupported | n/a（未解引用） | 0/0 |
| **合计（支持 cell）** | 6 | 2 overlap + 4 official-probe | **22,442** | **90/90** |

## code_components（必要组件进度）

- `[已验证] upstream fixed`：`comparative_methods/REVE/upstream/` 为 pinned official checkout；`sources/method_manifest.yaml` 记录 MIT、revision 和 source entrypoints。
- `[已验证] model assets`：`checkpoints/reve-base/model.safetensors` 与 `checkpoints/reve-positions/model.safetensors` 有尺寸/SHA-256/trusted-code 校验；base 为 512 dim、patch 200、overlap 20；position bank 为 543 个一对一名称/坐标。`reve-large` 仅是 secondary gated asset，未进入本轮 primary path。
- `[已验证] adapter`：`adapters/reve.py` 实现 local hash verification、200 Hz/shape/finite/support/unknown-coordinate fail-closed、official position lookup、final latent token、frozen pretrained query attention pooling、仅线性 head 训练。
- `[已验证] data/input boundary`：`alignment_data.py` 复用 method-neutral inventory/branch fingerprint，并强制 16-channel real panel、canonical 200 Hz、REFED unsupported；`configs/alignment_v2.yaml` 固定 registry SHA-256 `2a10b36d…11106`。
- `[已验证] train/eval and artifacts`：`run_public_development_v2.py` 的 outer-train-only standardizer、候选选择、train+public-validation refit、weights-only reload、prediction/manifest 写出；`run_public_matrix_v2.py` 串行 controller；`finalize_public_matrix_v2.py` 仅生成 public completion/A8 evidence，未生成 protected final aggregate。
- `[已验证] audit/test`：`audit_alignment_v2.py`、`audit_public_run_v2.py`、仓库级 `audit_adapter_alignment.py` 和 17+4 个 REVE tests 均存在；见下方测试记录。
- `[文档冲突]`：`README.md` 仍写 “B1-B4 remain pending”，`adapters/README.md` 写 “Pending”，`configs/README.md` 写 “No ... formal configuration is frozen”，`sources/method_manifest.yaml` B4 仍 pending；这些不能覆盖 `summary_final.json` 的机器终态，但必须在出最终 PPT/表格前同步。

## input_contract（模态、shape、通道、采样/窗口、mask/geometry、split）

- `[已验证] 模态/采样`：只消费 EEG + registered electrode coordinates；禁止 fNIRS；200 Hz canonical EEG，1–45 Hz record-wise robust-standard-deviation coordinate。每个支持 cell `modality_identity=["eeg"]`。
- `[已验证] 通道/shape`：每任务 16 个真实 measured channels，禁止复制、镜像、padding；各 cell 有 delivered order hash，position bank 543 names，unknown names fail closed。motor imagery/mental arithmetic、WG、n-back、DSR、Visual 的面板和 exact 13 个 direct-comparison fields 与 BIOT/CBraMod peer evidence 对齐。
- `[已验证] 时间/窗口`：registry anchor `canonical_registry_window_start`；8 s（1,600 samples）用于 MI/MA/WG/n-back/Visual，DSR 为 2 s（400 samples）。REVE patch 是 200 samples、20 overlap、step 180；8 s 只覆盖 1,460/1,600，2 s 只覆盖 380/400，剩余真实 tail 被官方 patching 规则丢弃，不用 padding 或复制填充。此为 method-native effective support 差异，不能假装与 nominal window 完全相同。
- `[已验证] mask/recorded support`：支持 cell 的 `recorded_support_mask` 是 all-true recorded support/no padding，public audit 检查每个样本完整 support；REFED 的 semantically real partial terminal support 不能进入该 adapter 的 attention/query pooling，故 A4/A7 unsupported。
- `[已验证] split/fold`：strict cross-subject、outer folds 0–4，shared registry fingerprint 为 `2a10b36d…11106`；public selection/refit 采用每 task 15 个 fold/seed jobs（17/42/73）。
- `[已验证] target corpus overlap/identity`：REVE pretraining exhaustive list 含 `Shin2017A`，而 Single-Trial 对应该 corpus；MI/MA 明确使用 `open_world_pretrained_with_target_corpus_overlap`。WG/n-back/DSR/Visual 的 manifest 声明在已发表清单中无目标语料重叠；不能外推为绝对无未披露重叠。

## output_contract（head、预测 schema、任务/指标、失败码、聚合口径）

- `[已验证] representation/head`：`final_transformer_latent_tokens_after_identity_final_layer` → frozen pretrained `cls_query` attention pooling → `(batch,512)`；`REVELinearProbe` 输出 `(batch,n_classes)` logits。encoder、position bank 和 query 均 frozen；只训练 downstream linear head weight/bias。
- `[已验证] 任务/primary endpoint`：六个支持任务都是 classification，primary metric 为 macro-F1；每 job 保留 logits、target、dataset_index、subject、sample_id，且 audit 从 prediction artifact 重新计算 validation macro-F1。
- `[已验证] failure/unsupported codes`：wrong sample rate、unknown electrode、duplicate/missing channels、non-finite input、padded support 会 fail closed；REFED 固定 `REVE_NO_PARTIAL_TIME_MASK_CONTRACT`。这些不是把失败数字裁成 chance 的机制。
- `[文档/实现边界]`：上游 downstream “linear probing”会让 query token 可训练；本项目显式冻结 512-parameter query，以保证 static feature-cache/equal-capacity linear-head 边界。这是可追溯的 source deviation，不是官方原 LP 数值复现。
- `[缺失]`：没有 masked `[2,T]` REFED head/prediction schema；没有受 METRIC_ACCEPTANCE 约束的 protected fold-mean aggregate、uncertainty/seed dispersion final table artifact。现有 public validation means 只能作 development diagnostic。

## result_stages（最终结果生成进度）

1. `[已验证] public feature/alignment`：6 个 supported cells A0–A8 pass；每个 feature cache shape 为 `(N,512)`，512/512 coordinates nonconstant，identical-input replay bitwise deterministic；7 comparison groups 的 13 exact direct fields schema audit pass。
2. `[已验证] public matrix`：90/90 job manifests/status、90/90 public refit checkpoints、90/90 prediction NPZ 和对应 feature caches retained；`completed_job_count=90`、`failed_job_count=0`、`automatic_retry_count=0`、`max_concurrent_jobs=1`。
3. `[已验证] public aggregate-like diagnostics`：`evidence/public_development_v2/matrix_completion_summary.json` 给出每 task 15-job public validation macro-F1 mean/SD，但每行 `claim_boundary=public_development_only_not_table_admissible`。均值仅供进度可视化，不能当 final result。

   | task | public validation macro-F1 mean ± SD（development only） |
   | --- | ---: |
   | motor_imagery | 0.57123 ± 0.01483 |
   | mental_arithmetic | 0.60132 ± 0.01806 |
   | wg | 0.60840 ± 0.01544 |
   | nback | 0.41695 ± 0.02505 |
   | dsr | 0.66466 ± 0.01415 |
   | visual | 0.27902 ± 0.01596 |

4. `[未完成/门控] protected evaluation`：`protected_evaluation_authorized=false`、`protected_test_opened=false`；没有 protected fold/seed prediction 或 final metric。
5. `[未完成/缺失] metric acceptance/final reporting`：未见 REVE 专属 aggregate、`METRIC_ACCEPTANCE` decision、table-ready/with-note artifact；不能以 public 90/90 completion 掩盖该门。

## evidence_quality（结论证据分级）

- `verified`：`summary_final.json`、六个 cell JSON、`matrix_completion_summary.json`、`controller_status.json`、90 个 run manifests/status/checkpoint/prediction/cache；本次 17 alignment tests、4 smoke tests、仓库级 v2 contract/cell audit 通过。
- `documented`：source revision、checkpoint/position SHA-256、Shin2017A overlap、official patch rule、upstream-vs-project query freezing deviation，来自 `sources/method_manifest.yaml`、`SOURCE_FIDELITY.md`、`IDENTITY_AND_REPRESENTATION_AUDIT.md` 与配置。
- `inferred`：本报告的五维分数、整体 81.5 和“未具备最终表格数字”的工程判定（后者由 `table_admissible=false`、protected lock、无 aggregate 共同支持）。
- `missing`：protected manifest/arrays、protected final aggregate/metric-acceptance artifact、REFED mask-aware adapter/head；未声称它们存在或已失败。

## blockers 与 risks

- **硬 blocker — protected locked**：没有 unlock authority；不得读取或执行 protected evaluation。当前 `final_result_availability=unavailable`。
- **硬 blocker — final aggregate/acceptance 缺失**：public completion summary 不满足 METRIC_ACCEPTANCE 的完整 fold/seed final table 口径，且明确不可入表。
- **能力 blocker — REFED**：没有 partial temporal mask 参数，不能 truthful 地支持 REFED terminal windows；保持 unsupported 比删样本或 padding 更符合契约。
- **重放风险 — queue state drift**：当前 `adapter_alignment_gate_contract_v2.yaml` 的 active delivery method 为 `none_public_delivery_queue_complete`，而历史 REVE runner 要求 active=`reve`；本次 `run_public_matrix_v2.py --dry-run` 因此被 PermissionError 拒绝。保留工件仍可被静态审计，若需重放必须另建经审查的 archival replay contract，不能改写当前队列或直接绕过 guard。
- **文档风险 — stale pending labels**：README、adapter/config README、method manifest/test README 与 A8 terminal evidence 不一致；出 PPT 时若不标明来源和日期，容易误读为 B1–B4 尚未完成或误读为 protected 已完成。
- **方法解释风险 — tail discard**：REVE nominal 8 s/2 s 输入分别只被 patch grid 覆盖 1,460/1,600 与 380/400 samples；应在比较图中标注 method-native effective support，不能只展示 tensor shape。
- **方法解释风险 — target overlap/source deviation**：MI/MA 必须单列 overlap track；冻结 query 和省略 upstream task-dependent scale 是项目适配偏离，不能写成 exact upstream LP reproduction。

## 本次运行的快速验证

| 命令 | 结果 |
| --- | --- |
| `.venv/bin/python -m pytest -q comparative_methods/REVE/tests/test_reve_alignment_v2.py` | **17 passed in 6.86s** |
| `.venv/bin/python -m pytest -q comparative_methods/REVE/tests/test_reve_smoke.py` | **4 passed in 10.60s** |
| `.venv/bin/python comparative_methods/audit_adapter_alignment.py` | **pass**；v2 contract schema/gates/profiles/tasks 校验通过；未读取 protected。 |
| `.venv/bin/python comparative_methods/audit_adapter_alignment.py comparative_methods/REVE/evidence/alignment_v2/{motor_imagery,mental_arithmetic,wg,nback,dsr,visual,refed_regression}.json` | **pass**；6 supported cell pass、REFED unsupported，direct fields schema 校验通过。 |
| `.venv/bin/python comparative_methods/REVE/run_public_matrix_v2.py --launch comparative_methods/REVE/configs/public_matrix_launch_v2.yaml --dry-run` | **拒绝（预期的当前队列保护）**：`PermissionError: REVE is not the active serial delivery method: 'none_public_delivery_queue_complete'`；没有启动任何 job。 |

## 最终审计标签

- `public_pipeline_status`: **complete for 6/6 supported classification cells; 90/90 public jobs; A0–A8 pass; REFED unsupported**。
- `protected_status`: **locked / not authorized / not opened**。
- `final_result_availability`: **unavailable**（仅有 development diagnostics，所有 retained public runs `table_admissible=false`）。
- `status_label`: **public_development_complete_A0_A8_pass_protected_locked_final_unavailable**。
