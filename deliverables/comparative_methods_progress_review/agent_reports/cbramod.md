# CBraMod 对比方法进度审查

审计日期：2026-08-11
审计范围：只读仓库/机器工件审查；未读取 protected manifest/array，未解锁门控，未启动训练。

## 结论先行

CBraMod 的实现、输入/输出适配、六个支持分类 cell 的 A0–A8 门控，以及 public development 矩阵已经完成：6 个任务覆盖 22,442 个唯一公开样本，90/90 个 selection/refit job（5 outer folds × 3 seeds × 6 tasks）完成，0 失败、0 自动重试。`evidence/alignment_v2/summary_final.json` 将状态记为 `public_development_complete_A0_A8_pass_protected_locked`。

这不等于已有最终可填表性能。所有 90 个 run 的 `table_admissible` 均为 `false`；CBraMod 没有 `runs/aggregate/` 正式聚合目录，也没有 metric-acceptance 产物，protected evaluation 未授权且未打开。因此当前结论是“public pipeline complete / final result unavailable”，不能把 public validation macro-F1 当作论文主表数字。

## 分数（统一权重）

| 维度 | 权重 | 得分 | 依据 |
| --- | ---: | ---: | --- |
| `code_components` | 30 | 97 | 官方上游、哈希校验/strict load、冻结 encoder、linear-probe runner、alignment/public-run audit、finalizer 与 17 个方法级测试均存在并可审查 |
| `input_adaptation` | 20 | 97 | EEG-only、200 Hz、200-sample patch、每任务 16 个真实通道、full recorded support、共享五折 public inventory；REFED 有事前 unsupported code |
| `output_adaptation` | 15 | 88 | 200-d latent + linear head、prediction schema、macro-F1/baselines、failure policy 完整；最终 aggregate/acceptance schema 尚缺 |
| `result_generation` | 25 | 52 | smoke、A0–A8、90/90 public matrix 完成；但没有 protected-final、正式 aggregate 或 metric acceptance |
| `evidence_reproducibility` | 10 | 88 | config/source/checkpoint/data/cache/run hashes、manifest、prediction、weights-only checkpoint 和审计包齐全；独立日志/最终聚合缺失，且队列完成后旧 replay CLI 被 active-method guard 阻止 |
| **overall（加权）** | **100** | **83.5** | 工程/public 进度分，不覆盖 protected/final 门控 |

## 状态门控

- `status_label`: `public_development_complete_A0_A8_pass_protected_locked_final_unavailable`
- `public_pipeline_status`: `complete`（6/6 supported cells；22,442 unique public samples；90/90 jobs）
- `protected_status`: `locked_not_authorized`（所有相关工件 `protected_test_opened=false`，`protected_evaluation_authorized=false`）
- `final_result_availability`: `unavailable`（没有 protected predictions/metrics，也没有 table-admissible aggregate）
- REFED regression：事前 `unsupported`，原因码 `CBRAMOD_NO_PARTIAL_TIME_MASK_CONTRACT`；不应计入 90 个 public job 分母。

## 代码组件进度

| 组件 | 状态 | 证据（仓库相对路径） |
| --- | --- | --- |
| 官方上游与许可 | 已验证 | `comparative_methods/CBraMod/sources/method_manifest.yaml`（GitHub revision `0ff6be918985689e7df679bc731ffb70e6c6224f`、MIT）；`upstream/models/cbramod.py`、`upstream/models/criss_cross_transformer.py`、`upstream/LICENSE` |
| checkpoint 获取与安全加载 | 已验证 | `adapters/cbramod.py:83-146`：size/SHA-256、`torch.load(weights_only=True)`、全 tensor state、strict state-dict；manifest pin `19775842` bytes / SHA-256 `0792cb…5718` |
| 表征边界 | 已验证 | `REPRESENTATION_LAYER_AUDIT.md`；加载后将 `proj_out` 替换为 `Identity`，输出 pre-reconstruction 200-d latent token |
| EEG adapter | 已验证 | `adapters/cbramod.py:149-227`：200 Hz、正的 200-sample 整倍窗口、唯一 channel names、finite、全真 channel/sample masks、mean over channel/patch tokens |
| 数据边界/库存 | 已验证 | `alignment_data.py:58-166,169-239`；registry SHA、五 outer folds、逐任务 panel/duration、真实通道/geometry、full recorded support、sample identity |
| linear-probe train/eval | 已验证 | `run_public_development_v2.py:535-745`：outer-train 标准化、macro-F1 选择、inverse-frequency class weights、train+public-validation refit、weights-only reload |
| public matrix/finalizer | 已验证 | `configs/public_development_v2.yaml`、`configs/public_matrix_launch_v2.yaml`、`finalize_public_matrix_v2.py`；串行（max concurrency=1）、无自动重试 |
| audit/test | 已验证 | `audit_alignment_v2.py`、`audit_public_run_v2.py`、`tests/test_cbramod_smoke.py`、`tests/test_cbramod_alignment_v2.py`；本次 17 passed |

## 输入契约进度

| 输入项 | 状态 | 证据/计数 |
| --- | --- | --- |
| 模态 | 已验证 | `configs/alignment_v2.yaml:1-74`：仅 EEG；fNIRS 不允许；六个分类任务 supported |
| 采样/窗口/patch | 已验证 | 200 Hz；MI/MA/WG/N-back/Visual 为 8 s（1600 samples、8 patches），DSR 为 2 s（400 samples、2 patches）；adapter 拒绝非 200 Hz 或非 200 整倍窗口 |
| measured channels | 已验证 | 每个 supported task 冻结 16 个唯一真实通道；`comparison_fields.measured_channel_identity_set` 与 delivered order/hash 均保留；没有复制/镜像/补齐 |
| masks/support | 已验证 | `CBraModPublicView` 要求 `valid_mask` 全真、无 padding、无 bad measured channel、finite EEG、geometry available；六 cell A4/A7 pass |
| signal branch/coordinate | 已验证 | canonical recordwise robust-SD coordinate；1–45 Hz branch 与 cache/event/geometry/EEG branch fingerprints 见各 cell JSON；上游 raw `/100` 省略被 `REPRESENTATION_LAYER_AUDIT.md` 明确记录为 coordinate adaptation |
| split/identity | 已验证 | method-neutral registry 五 folds；每 task 的 sample/split/target/observation/channel/branch fields 与 BIOT exact-match；6 cell A0–A8 pass |
| public inventory | 已验证 | MI 1,740；MA 1,740；WG 1,560；N-back 702；DSR 8,980；Visual 7,720；合计 **22,442 unique samples**，每 cell `outer_fold_count=5`、`nonconstant_coordinate_count=200` |
| REFED | 事前 unsupported | `evidence/alignment_v2/refed_regression.json`：partial terminal windows 无 truthful partial-time-mask contract；A4/A7 unsupported，不删样本伪装支持 |

## 输出契约进度

| 输出项 | 状态 | 证据 |
| --- | --- | --- |
| 表征输出 | 已验证 | latent tensor `(B,C,patch_count,200)`，pool 后 `(B,200)`；`output_layer=encoder_latent_before_pretraining_proj_out`，`pooling=official_avgpooling_patch_reps` |
| prediction/head | 已验证 | `adapters/cbramod.py:230-239` 的 `nn.Linear(200, output_dim)`；每 run 保存 `logits,target,dataset_index,subject,sample_id` 五类核心数组 |
| 任务/主指标 | 已验证（开发口径） | 6 分类任务使用 macro-F1；run selection report 同时保存 majority/train-prior baselines、candidate history、standardizer hash |
| failure/retry | 已验证 | `public_development_v2.yaml`：retain failed status、automatic retry=0、completed overwrite forbidden；90 run 均 `status=completed` |
| 聚合/准入 | 未完成 | `matrix_completion_summary.json` 仅给 public-development cell mean/SD 并明确 `table_admissible=false`；未见 `CBraMod/runs/aggregate/`、comparison metric target 审计或最终表格 schema |

## 结果生成阶段

| 阶段 | 状态 | 计数 | 证据 |
| --- | --- | ---: | --- |
| smoke/connectivity | 已通过 | 2 个保留 pilot run（mini + full MI/outer0/seed17）；本次测试 17 passed | `evidence/public_development_v2/pilot_audit.json`、`runs/public_development_v2/pilot_*` |
| alignment A0–A8 | 已通过 | 6/6 supported cells；22,442 samples；各 cell 9/9 gate pass | `evidence/alignment_v2/summary_final.json`、六个 task JSON |
| public selection/refit matrix | 已完成 | **90/90** = 6 tasks × 5 folds × 3 seeds；0 fail、0 retry、max concurrency 1；90 manifest/status/prediction/report/checkpoint 全部存在 | `evidence/public_development_v2/matrix_completion_summary.json`、`runs/public_development_v2/matrix_v2/completed_public_audit.json` |
| development aggregate | 仅开发用途 | 6 task summaries（每 task 15 jobs，均值/SD 已保存） | `matrix_completion_summary.json`；每行 `claim_boundary=public_development_only_not_table_admissible` |
| metric acceptance | 未运行/缺失 | 0 accepted final cells | CBraMod 目录无 metric-acceptance/target-check aggregate；主规则见 `docs/comparisons/METRIC_ACCEPTANCE.md` |
| protected-final | locked | 0 protected jobs / 0 opened arrays | `summary_final.json`、completion summary、90 run manifests 全部 `protected_test_opened=false` |

### Public development 分数（仅诊断，不是最终表格数字）

| Task | 15-job validation macro-F1 mean ± SD | 备注 |
| --- | ---: | --- |
| Motor imagery | 0.5375 ± 0.0141 | development-only |
| Mental arithmetic | 0.6099 ± 0.0196 | development-only |
| WG | 0.6014 ± 0.0147 | development-only |
| N-back | 0.4114 ± 0.0258 | development-only |
| DSR | 0.5793 ± 0.0072 | development-only |
| Visual | 0.2626 ± 0.0221 | development-only |

## 阻塞项与下一步

| 严重度 | 项目 | 证据 | 下一步（需授权时停止） |
| --- | --- | --- | --- |
| critical | protected final unavailable | `summary_final.json`：`protected_evaluation_authorized=false`、`protected_test_opened=false`；所有 run 同样为 false | 由统一协议签发独立 protected unlock manifest 后，按冻结 config 执行；本审计不代为解锁 |
| high | no final aggregate/metric acceptance | 无 `comparative_methods/CBraMod/runs/aggregate/`；completion summary 的 `table_admissible=false` | 在授权 protected 结果后生成 fold/seed/OOF aggregate，并按 `METRIC_ACCEPTANCE.md`/machine targets 逐 cell 审核；不能直接复用 public validation mean |
| medium | historical replay guard | 本次运行 `audit_public_run_v2.py` 对一个已完成 public run 被 `PermissionError` 拦截：当前 `active_delivery_method=none_public_delivery_queue_complete` | 保留现有 completion/audit hash；如需重放，应增加不改变结果的历史只读审计入口或显式历史 contract，不重新激活队列 |
| medium | 文档/配置时点漂移 | `README.md` 仍写 B1–B4 pending；`configs/public_development_v2.yaml` 指向旧 `summary.json`（A0–A7 pending），而 `summary_final.json` 已 A0–A8 complete | 发布前更新 README/config provenance 或明确“历史生成配置”，避免将已完成与待门控状态混读 |

## 风险

1. `public_validation_macro_f1` 来自 public validation，可能因选择/开发而偏乐观；在 protected 或 acceptance 前不得声称泛化性能。
2. REFED 整格 unsupported，CBraMod 只能覆盖 6/7 统一任务；跨方法汇总必须保留缺失状态和原因码。
3. 省略上游 raw-array `/100` 是有意的 canonical-coordinate 决策，虽有哈希/文档支撑，仍需在最终论文脚注中说明，避免被误读为原始预处理复现。
4. 90 个 run 保留 manifest、prediction、report、checkpoint 和 cache，但没有独立训练日志目录；重放主要依赖 hash-complete manifest 和 retained history。
5. checkpoint 的官方 upstream code revision（`0ff6be…`）与 Hugging Face artifact revision（`500543…`）分开 pin，符合 manifest 设计，但发布时需同时展示，避免只报告一个 revision。

## 审计证据质量

等级：`strong_public_machine_evidence_with_final_gap`。

- 已验证：官方代码/权重 hash、strict/weights-only load、input/output rejection、exact replay checks、六 cell full-public coverage、90/90 artifact re-audit、protected=false invariant。
- 文档声明并有工件佐证：A0–A8 public-complete、22,442 unique samples、matrix 0 failure/0 retry、unsupported REFED reason。
- 推断：没有 `runs/aggregate/` 与 acceptance artifact，因此不能提供最终表格数字；这是目录/工件检查结论，不是把缺失数字当作零。
- 缺失：protected predictions/metrics、正式 aggregate、metric acceptance decision、独立完整训练日志。

## 可复核命令

```bash
.venv/bin/python -m pytest -q \
  comparative_methods/CBraMod/tests/test_cbramod_smoke.py \
  comparative_methods/CBraMod/tests/test_cbramod_alignment_v2.py
# 结果：17 passed in 7.71s
```

```bash
.venv/bin/python comparative_methods/CBraMod/audit_public_run_v2.py \
  comparative_methods/CBraMod/runs/public_development_v2/matrix_v2/motor_imagery/outer0/seed17
# 当前队列已进入 none_public_delivery_queue_complete；脚本按设计拒绝历史重放：
# PermissionError: CBraMod is not the active serial delivery method
```

以上失败没有读取 protected 数据，也没有修改任何方法实现；它暴露的是历史 replay 的 active-method guard，而非已完成 run 的 artifact mismatch。
