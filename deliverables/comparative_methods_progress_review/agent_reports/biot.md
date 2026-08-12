# BIOT 全面进度审查

审计日期：2026-08-11
方法：BIOT（官方预训练 EEG、冻结编码器 linear probe）
审计边界：只读仓库与机器工件；未读取 protected 数组/身份、未解锁门控、未启动训练。

## 结论先行

BIOT 的 public delivery 已完成，但最终结果尚未产生。六个分类任务的 adapter v2 full-public replay 与 A0–A8 均有机器证据，公共矩阵为 6 tasks × 5 outer folds × 3 seeds = 90/90 jobs，0 failures、0 retries；REFED regression 在看分数前按 `BIOT_NO_PARTIAL_TIME_MASK_CONTRACT` 事前 unsupported。所有工件均声明 `protected_test_opened=false`，protected evaluation 仍 locked。

因此，BIOT 当前应标为 **public pipeline complete / protected locked / final result unavailable**。90 个 public validation 分数只能作为 development evidence，不能进入最终表格；仓库中没有 BIOT 的 protected predictions、fold/seed final aggregate、metric-acceptance decision 或 `runs/aggregate` 工件。

评分按用户指定权重计算（code 30%、input 20%、output 15%、result 25%、evidence 10%）：

| 维度 | 分数 | 权重 | 依据摘要 |
| --- | ---: | ---: | --- |
| code_components | 90 | 30% | 上游 revision、3 个 checkpoint、adapter、public train/eval、审计脚本、测试均可运行；正式 protected/最终聚合组件尚未交付 |
| input_adaptation | 92 | 20% | 六个分类 cell A0–A8 public-complete；16 个真实 EEG 通道、200 Hz、窗口、mask、branch/split hash 均有证据；REFED 不支持 |
| output_adaptation | 80 | 15% | 256-d frozen embedding、线性 head、macro-F1/logits/identity schema 与 train-only fit scope 完整；masked-CCC/最终准入输出不存在 |
| result_generation | 45 | 25% | 90/90 public development jobs 完成且可审计；protected 0 jobs，最终 aggregate/metric acceptance 0 |
| evidence_reproducibility | 82 | 10% | manifest、hash、checkpoint/cache、prediction、status 与测试证据齐全；README/旧 summary 与 final summary 状态不一致 |
| **overall** | **76.9** | **100%** | 加权值；不覆盖 protected/final 门控结论 |

## 状态与任务覆盖

| 项目 | 审计结论 | 精确计数/证据 |
| --- | --- | --- |
| planned_tasks | 已登记 7 个统一任务 | `motor_imagery`, `mental_arithmetic`, `wg`, `nback`, `dsr`, `visual`, `refed_regression`；`docs/comparisons/PROTOCOL.md`、`comparative_methods/BIOT/configs/alignment_v2.yaml` |
| supported_tasks | 6 个分类任务 | 6 个 alignment evidence cell，唯一 public samples 合计 22,442：1,740 + 1,740 + 1,560 + 702 + 8,980 + 7,720；`comparative_methods/BIOT/evidence/alignment_v2/summary_final.json` |
| unsupported_tasks | REFED regression 1 个 cell | `BIOT_NO_PARTIAL_TIME_MASK_CONTRACT`；partial terminal support 无法由 frozen adapter truthful 传播；`comparative_methods/BIOT/evidence/alignment_v2/refed_regression.json` |
| planned_job_count | 105（统一合同上限） | 7 tasks × 5 folds × 3 seeds；REFED 15 个单元为事前 unsupported，不应算作失败缺失 |
| planned_public_job_count | 90（当前支持 scope） | `comparative_methods/BIOT/configs/public_development_v2.yaml:job_matrix.expected_public_jobs` |
| completed_job_count | 90 | 6 tasks × 5 folds × 3 seeds；每 task 15 jobs；`runs/public_development_v2/matrix_v2/controller_status.json`、`completed_public_audit.json` |
| public_pipeline_status | **complete for six classification cells** | `summary_final.status=public_development_complete_A0_A8_pass_protected_locked`；每个支持 cell A0–A8 pass |
| protected_status | **locked / unopened** | launch/config/evidence/status 均为 `protected_evaluation_authorized=false`、`protected_test_opened=false`；未见 protected 路径或 true flag |
| final_result_availability | **unavailable** | 无 protected run、final prediction、aggregate、metric acceptance；public completion summary 明确 `table_admissible=false` |

## 代码组件进度

| 组件 | 状态 | 已核实证据 |
| --- | --- | --- |
| 上游固定与资产 | verified complete | `comparative_methods/BIOT/sources/method_manifest.yaml:10-50` 固定 revision `d138e32634e52ae9fa6ec98ac9c4087b14ca869a`、MIT、3 个官方 checkpoint；`upstream/.git` HEAD 同 revision；PREST-16 SHA-256 `40f55f5d23e83796495616c8145c8336fcff2b901c42e8ba5115223081c2ad70` |
| 模型实现 | verified complete | `upstream/model/biot.py` 原始 `BIOTEncoder`；`adapters/biot.py:78-156` 以 `weights_only=True`、尺寸/hash 检查、strict state-dict 加载，冻结并 eval |
| adapter | verified public-complete | `adapters/biot.py:159-246` 检查 `[B,C,T]`、200 Hz、唯一通道名、容量、finite、full support，输出 `[B,256]`；`alignment_v2/*.json` 六个 cell A0–A8 pass |
| train/eval | verified public-complete, final pending | `run_public_development_v2.py` 做 feature extraction、outer-train-only standardization、2×2 probe selection、train+public-validation refit、weights-only reload；无 protected runner/最终 refit |
| 审计脚本 | verified | `audit_alignment_v2.py`、`audit_public_run_v2.py`、`build_public_job_matrix_v2.py`、`run_public_matrix_v2.py`、`finalize_public_matrix_v2.py`；代表性 run audit 与 launch dry-run 均 pass |
| 测试 | verified | `tests/test_biot_smoke.py`、`tests/test_biot_alignment_v2.py`；本次命令共 14 passed |
| 文档/状态一致性 | blocked for clean handoff | `README.md:4`、`adapters/README.md:3-4` 仍写 B1–B4/pending；`ALIGNMENT_V2_STATUS.md:11-16` 写 A8 pending，但同文件 `:26-28` 与 `summary_final.json` 写 A8 pass；`public_development_v2.yaml` 仍指向旧 `summary.json` 的 A8-pending status |

## 输入适配合同

| 项目 | 状态 | 证据与计数 |
| --- | --- | --- |
| 模态边界 | verified | EEG-only；`alignment_v2.yaml:data`、`method_manifest.yaml:56-59` 禁止 fNIRS、teacher/derived feature、synthetic channel copy；六个 cell 的 `modality_identity=["eeg"]` |
| 采样/shape/window | verified | 200 Hz canonical EEG；PREST-16 固定 16 channels；MI/MA/WG/n-back/Visual 8 s（1,600 samples），DSR 2 s（400 samples）；六 cell feature shape 为 `[N,256]`，N=1,740/1,740/1,560/702/8,980/7,720 |
| 通道 identity/order | verified with documented deviation | 每 task 16 个 unique measured native electrodes；`BIOTPublicView` 按身份选择/reorder，不复制/填充；上游 PREST bipolar montage 到 native-electrode positional transfer 的偏离见 `sources/SOURCE_FIDELITY.md:13-26`，不能称原论文输入复现 |
| support/mask/geometry | verified | `alignment_data.py:167+` 只接受全 true `valid_mask`、拒绝 bad channel/non-finite/缺 geometry；六 cell `recorded_support_mask.mask_semantics=all_true_recorded_support_no_padding`、full unique inventory replay；REFED partial terminal support 明确 unsupported |
| split/sample identity | verified | method-neutral registry SHA-256 `2a10b36db85dba6ec5543edc7810ff85d978ea5af8c79fda3d38a1e5cfd11106`；5 outer folds；canonical join-key/event/offset sample IDs；每 cell `outer_fold_count=5`，public matrix 90 jobs |
| canonical branch | verified | 1–45 Hz、robust-standard-deviation coordinate；single-trial v4、simultaneous EOG-clean v1、visual raw branch 的 branch/hash fingerprints 在 alignment evidence 中保留 |
| pretraining exposure | documented/verified for declared corpus | 三个 checkpoint 的 declared corpora 与 target overlap 状态写入 `method_manifest.yaml`/cell identity；不把“未声明重叠”扩大成绝对无重叠证明 |
| REFED input contract | unsupported (pre-score) | `BIOT_NO_PARTIAL_TIME_MASK_CONTRACT`；不读取 REFED public signal/identity 来伪造支持或删样本 |

## 输出适配合同

| 项目 | 状态 | 证据与计数 |
| --- | --- | --- |
| encoder output | verified | upstream mean over channel-frequency tokens；adapter identity `output_layer=upstream_biot_encoder_mean_embedding`、`output_shape=[batch,256]`；六 cell deterministic replay exact，256/256 coordinates non-constant |
| task head | verified for classification | `BIOTLinearProbe` 为 frozen 256-d encoder + trainable `nn.Linear(256,K)`；K=2（MI/MA/WG/DSR）、K=3（n-back）、K=4（Visual）；90 jobs 均保留 `logits` |
| target/metric schema | verified public | canonical labels/sample IDs、`classification_metrics`、primary `macro_f1`；每 run prediction arrays 为 `logits,target,dataset_index,subject,sample_id`；public validation report 与 prediction 可重算（代表性 run 已审计） |
| fit/aggregation scope | verified public | selection standardizer fit outer-train-only；probe search 2 learning rates × 2 weight decays；refit standardizer fit train+public-validation；seed/fold/job manifests retained；protected/final aggregate absent |
| regression output | unsupported | no truthful masked sequence head/target-valid-mask path for REFED；因此不存在 masked CCC output/metric result |
| final acceptance output | missing | no per-cell `B0`, `minimum_admissible`, `preferred_target`, decision (`TABLE_READY*`/`FAILURE_RESULT`/`REJECTED_VALUE`) or final table aggregate |

## 结果生成进度

### 已完成的 public development（不能当最终结果）

| 任务 | public jobs | unique public samples | public validation macro-F1 mean ± job SD | recomputed B0（majority/prior） | 解释 |
| --- | ---: | ---: | ---: | ---: | --- |
| motor_imagery | 15/15 | 1,740 | 0.5299 ± 0.0177 | 0.3333 | development-only；90/90 matrix 内 |
| mental_arithmetic | 15/15 | 1,740 | 0.6095 ± 0.0260 | 0.3333 | development-only；90/90 matrix 内 |
| wg | 15/15 | 1,560 | 0.5946 ± 0.0340 | 0.3333 | development-only；90/90 matrix 内 |
| nback | 15/15 | 702 | 0.4011 ± 0.0235 | 0.1667 | development-only；90/90 matrix 内 |
| dsr | 15/15 | 8,980 | 0.5399 ± 0.0090 | 0.4118 | development-only；90/90 matrix 内 |
| visual | 15/15 | 7,720 | 0.2665 ± 0.0225 | 0.1287 | development-only；90/90 matrix 内 |
| REFED | 0/15（事前 unsupported） | 未审计 signal | — | — | `BIOT_NO_PARTIAL_TIME_MASK_CONTRACT` |

这些均值/SD来自 `evidence/public_development_v2/matrix_completion_summary.json`（macro-F1）与 90 个 run manifest 的 baseline 字段重算。90/90 public jobs 的每一个 validation macro-F1 都高于该 run 的 majority/prior B0，但这只说明 public development 的 above-baseline sanity，不构成 protected final metric acceptance。

### 尚缺失的结果阶段

| 阶段 | 状态 | 计数/证据 |
| --- | --- | --- |
| smoke/pilot | verified pass | smoke 与 MI/outer0/seed17 public pilot 各有保留工件；均 `table_admissible=false` |
| A0–A8 adapter gates | verified pass for 6 cells | 6/6 public-complete cells，22,442 unique samples；REFED unsupported，A8 N/A |
| public selection/refit matrix | verified complete | 90/90 jobs，6 tasks × 5 folds × 3 seeds，0 failures，0 retries，serial `max_concurrent_jobs=1` |
| public aggregate | verified development-only | 6 task means/SD + fold seed means；`table_admissible=false` |
| metric acceptance | missing | 0 个 BIOT final cell decision；没有 acceptance audit artifact |
| protected evaluation | locked | 0 jobs、0 protected predictions、0 protected aggregate；需要独立授权 |
| final result package | missing | 未找到 `comparative_methods/BIOT/runs/aggregate`、final table、OOF aggregate 或 protected manifest |

## 主要 blockers

| 严重度 | 项目 | 证据 | 下一步（需授权时才做） |
| --- | --- | --- | --- |
| high | protected evaluation 未授权/未执行 | `configs/public_matrix_launch_v2.yaml:43`、`summary_final.json`、90 run manifests 均 `protected_evaluation_authorized=false`/`protected_test_opened=false`；0 protected jobs | 由项目负责人单独签发 unlock manifest；沿用冻结 config、fold、seed、unsupported REFED 边界，完成 protected 预测与审计 |
| high | 最终 aggregate/metric acceptance 缺失 | 无 `runs/aggregate`；public completion `table_admissible=false`；未见 `comparison_metric_targets_v1.yaml` 对应 BIOT decision rows | protected 完成后生成完整 fold/seed aggregate、B0、uncertainty、acceptance decision；不得用 public validation 均值替代 |
| medium | 状态源与说明文档漂移 | `evidence/alignment_v2/summary.json` 为 A0–A7 pending、`summary_final.json` 为 A0–A8 pass；README/adapter README 仍 pending；public runner config hash 固定旧 summary | 在任何后续 unlock 前明确唯一 authoritative summary（建议保留 immutable old summary、在新 protocol version 中显式指向 `summary_final`），更新文档/manifest/hash；当前不要重跑或覆盖旧 artifacts |
| medium | named-method 与项目输入存在 positional-transfer 偏离 | `sources/SOURCE_FIDELITY.md:21-26` 明确 native electrodes ≠ original PREST bipolar montage | 结果表使用 `official_pretrained_biot_encoder_native_electrode_positional_transfer_v1`/adapted 标签，不声称原论文 downstream 数值复现 |
| medium | REFED 覆盖缺口 | `refed_regression.json` 事前 unsupported；BIOT adapter 只接受 full time support | 只有实现并审计 truthful `target_valid_mask`/partial terminal support 后才创建新协议；否则保持 unsupported，不删样本、不伪造 mask |

## 风险

- 高：public validation 数字容易被误写成最终性能；所有 public manifests 与 completion summary 都明确 `table_admissible=false`，必须保留该边界。
- 中：16 个 native electrode token 顺序沿用 PREST positional table，是有记录的算法/信息适配偏离，不是原始 bipolar montage 复现。
- 中：REFED 缺失意味着 BIOT 不能覆盖统一七任务表；总体进度不能用 6/7 的 public 覆盖掩盖这一点。
- 中：`summary.json`/`summary_final.json` 和 README 的状态漂移降低了复现入口的单一性；若未经 protocol version/hash 处理直接新跑，可能读到 A8-pending 旧门控。
- 低：三类官方 checkpoint 都是 PyTorch pickle 容器；当前 loader 已 hash 检查并 `weights_only=True`，但任何未来改动仍须保持这一安全边界。

## 本次运行的快速测试/审计

1. `.venv/bin/python -m pytest -q comparative_methods/BIOT/tests` → **14 passed in 5.38s**（含三 checkpoint hash/strict load、GPU frozen probe forward/backward/optimizer/reload、alignment/REFED unsupported、job matrix boundary）。
2. `.venv/bin/python comparative_methods/BIOT/audit_public_run_v2.py comparative_methods/BIOT/runs/public_development_v2/matrix_v2/motor_imagery/outer0/seed17 --config comparative_methods/BIOT/configs/public_development_v2.yaml --output /tmp/biot_public_run_audit.json` → **pass**；MI/outer0/seed17，selection+refit，`validation_macro_f1=0.541583264649319`，`table_admissible=false`，protected false。
3. `.venv/bin/python comparative_methods/BIOT/run_public_matrix_v2.py --launch comparative_methods/BIOT/configs/public_matrix_launch_v2.yaml --dry-run` → **pass**；`job_count=90`、`max_concurrent_jobs=1`、`automatic_retry_count=0`、protected false。
4. 只读静态工件扫描 → **90/90 manifest-status 对齐，6×15 jobs，0 missing artifact/mismatch，0 protected true flag**；`completed_public_audit.json` 保留 90 个唯一 job reports，全部 pass。

未运行 `audit_alignment_v2.py` 的全量重新提取，也未执行任何 protected/final run；本报告使用已保留的 `summary_final.json`、cell evidence、90-run audit 与快速验证结果。

## 证据质量与结论类型

- **已验证（verified）**：本次命令通过的 14 项测试、代表性 run audit、matrix dry-run、90 个 run manifest/status/artifact 静态扫描，以及 `summary_final` 中六个 A0–A8 cell 的机器字段。
- **文档/工件声明（documented）**：官方 checkpoint corpus exposure、上游论文/仓库语义、source-fidelity positional-transfer 边界；其证据来自 `method_manifest.yaml`、`SOURCE_FIDELITY.md` 与上游 revision，而非本次重新训练。
- **推断（inferred）**：overall 评分、public scores above B0 的 development 解释、文档漂移对未来复现入口的影响；不把这些推断写成 final performance。
- **缺失（missing）**：protected predictions/identities、final aggregate、metric acceptance decisions、REFED masked-CCC 结果、唯一统一的 post-A8 README/config 状态。

## 关键证据路径

- `comparative_methods/BIOT/sources/method_manifest.yaml`
- `comparative_methods/BIOT/sources/SOURCE_FIDELITY.md`
- `comparative_methods/BIOT/upstream/model/biot.py`
- `comparative_methods/BIOT/adapters/biot.py`
- `comparative_methods/BIOT/alignment_data.py`
- `comparative_methods/BIOT/configs/alignment_v2.yaml`
- `comparative_methods/BIOT/configs/public_development_v2.yaml`
- `comparative_methods/BIOT/configs/public_matrix_launch_v2.yaml`
- `comparative_methods/BIOT/evidence/alignment_v2/summary_final.json`
- `comparative_methods/BIOT/evidence/alignment_v2/{motor_imagery,mental_arithmetic,wg,nback,dsr,visual,refed_regression}.json`
- `comparative_methods/BIOT/evidence/public_development_v2/matrix_completion_summary.json`
- `comparative_methods/BIOT/runs/public_development_v2/matrix_v2/controller_status.json`
- `comparative_methods/BIOT/runs/public_development_v2/matrix_v2/completed_public_audit.json`
- `comparative_methods/BIOT/run_public_development_v2.py`
- `comparative_methods/BIOT/audit_public_run_v2.py`
- `comparative_methods/BIOT/tests/test_biot_smoke.py`
- `comparative_methods/BIOT/tests/test_biot_alignment_v2.py`
