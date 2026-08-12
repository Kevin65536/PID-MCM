# NormWear（`normwear_eeg_fnirs_adapted`）进度审查

审计日期：2026-08-11（CST）。本审查只读核对当前仓库文件、机器工件和轻量测试；未访问 protected 数据、未解锁门控、未启动训练或修改 NormWear 方法实现。相对路径均相对于仓库根目录 `/SSD_2/pid-mcm-implementation`。

## 结论

NormWear 的公开交付链已完成：六个分类 cell 的 A0–A8 门控均为 `pass`，22,442 个公开身份已生产回放，90 个公开 selection/refit job 全部完成且失败/自动重试为 0。方法身份应始终写为 `normwear_eeg_fnirs_adapted`；上游预训练没有 fNIRS，因此不能写成原论文 fNIRS 复现。

这不等于已有最终性能数字。当前没有 protected evaluation、没有可准入论文表的 NormWear aggregate，也没有 `runs/aggregate` 目录；90 个 job 的数值都明确标为 public-development-only，`table_admissible=false`。REFED 被预注册为 unsupported，因为固定 NormWear encoder 没有 truthful temporal support-mask 接口，也没有 masked sequence-regression 输出合同。

## 五维评分

评分维度和权重遵循任务给定的 `30/20/15/25/10`。`已验证` 表示由测试或机器工件直接核对；`声明` 表示 README/config/manifest 的冻结声明；`推断` 表示从实现和工件结构推得；`缺失` 表示当前没有足够工件。

| 维度 | 分数 | 权重 | 审计依据 |
| --- | ---: | ---: | --- |
| `code_components` | 96 | 30 | `upstream/` 有 45 个 Python 文件；manifest 对 7 个 source-fidelity 文件给出 hash；10 个方法级运行/审计脚本、1 个 adapter、4 个测试模块齐全；checkpoint strict match、encoder freeze 和 head-only update 均验证。残余是清理/汇总工件，而非实现缺口。 |
| `input_adaptation` | 94 | 20 | 六个 support-matched direct cell 的 A0–A4 全通过；22,442 个输入全有限、非恒定、真实通道且完整 recorded/analysis-valid support；EEG/HbO/HbR、rate、window、CWT、channel order、split/hash 均冻结。REFED partial-support 不能忠实消费，故显式 unsupported；DSR 的 fNIRS 只作同步 block context。 |
| `output_adaptation` | 90 | 15 | final layer-norm token、mean-token/channel-concat 特征、outer-train linear probe、logits/target/sample identity schema、macro-F1 已在 smoke 和 90 个 public run 中工作；但没有 masked sequence regression head（REFED unsupported），也没有最终跨 fold aggregate 输出。 |
| `result_generation` | 68 | 25 | A7 全量 feature cache 和 A8 的 90/90 public selection/refit 已完成，保留每 job 的 manifest/status/report/checkpoint/prediction；但 protected-final、metric acceptance 和 `runs/aggregate` 缺失，public validation 数值不能进最终表。 |
| `evidence_reproducibility` | 88 | 10 | 17 个 NormWear evidence JSON、源/配置/运行 hash、split fingerprint、cache identity、controller/completed audit 和 35 个测试通过；但保留了 1 个旧 `state=validating` n-back cache，`summary_final.json` 仍有 1 条旧 visual pending `task_reports`，增加解释歧义。 |

加权总分：`96×0.30 + 94×0.20 + 90×0.15 + 68×0.25 + 88×0.10 = 86.9/100`。

## 状态三元组

- `public_pipeline_status`：**complete**；`summary_final.json` 报告 `public_development_complete_A0_A8_pass_protected_locked`，6/6 支持任务 A0–A8 通过，22,442/22,442 公开样本回放，90/90 public jobs 完成，0 失败、0 自动重试。
- `protected_status`：**locked**；`public_matrix_launch_v2.yaml`、`matrix_completion_summary.json` 和 `summary_final.json` 均为 `protected_evaluation_authorized=false`、`protected_test_opened=false`。
- `final_result_availability`：**unavailable**；当前只有 public-development validation，`matrix_completion_summary.json` 为 `table_admissible=false`，没有 NormWear protected/final aggregate。

## 任务、输入和公开作业矩阵

| 任务 | 数据集 / 标签 | 窗口 | 公开样本 | 真实通道（EEG + HbO/HbR） | 特征宽度 | 当前门控 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| motor imagery | `eeg_fnirs_single_trial`; LMI/RMI | 8 s | 1,740（870/870） | 30 + 36 + 36 = 102 | 78,336 | A0–A8 pass，protected locked |
| mental arithmetic | `eeg_fnirs_single_trial`; MA/BL | 8 s | 1,740（870/870） | 30 + 36 + 36 = 102 | 78,336 | A0–A8 pass，protected locked |
| WG | `simultaneous_eeg_nirs`; WG/BL | 8 s | 1,560（780/780） | 28 + 36 + 36 = 100 | 76,800 | A0–A8 pass，protected locked |
| n-back | `simultaneous_eeg_nirs`; 0/2/3-back | 8 s | 702（234/类） | 28 + 36 + 36 = 100 | 76,800 | A0–A8 pass，protected locked |
| DSR | `simultaneous_eeg_nirs`; EEG-native Go/No-go | 2 s | 8,980（2,694/6,286） | 28 + 36 + 36 = 100 | 76,800 | A0–A8 pass；fNIRS 为同步 block context |
| Visual | `visual_cognitive_motivation`; RR/RF/FF/FR | 8 s | 7,720（FF 1,378；FR 2,726；RF 860；RR 2,756） | 30 + 24 + 24 = 78 | 59,904 | A0–A8 pass，protected locked |
| REFED regression | `refed`; masked valence/arousal sequence | 20 s | 未回放 | 部分 terminal support | — | 预注册 `unsupported`；不是失败数字 |

公开矩阵计划为 6 个支持任务 × 5 outer folds × 3 seeds（17、42、73）=`90` 个 job；实际完成 `90/90`。每个 job 保留 5 个核心文件（manifest、status、public selection report、public refit checkpoint、validation predictions），因此矩阵目录有 `90×5=450` 个 job 工件。

## 必要代码组件进度

| 组件 | 当前状态与精确证据 |
| --- | --- |
| 上游固定 / source fidelity | **已验证**：`comparative_methods/NormWear/sources/method_manifest.yaml` 固定 revision `07517fcb13def8c89cb586128359cec02f86ec8d`；identity audit 对 7 个源文件 hash、upstream clean 和 strict model match 均通过。上游仓库共 45 个 Python 文件。 |
| 官方模型 / checkpoint | **已验证**：`checkpoints/normwear_pretrain_ckpt.pth` 为 544,579,503 bytes，SHA-256 `36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561`；weights-only `OrderedDict` 261 entries、136,116,425 tensor elements；严格匹配 pinned `NormWear`。主轨 encoder 222 entries/128,118,528 parameters，decoder 39 entries 排除；12 层 encoder、768 dim、9×5 patch。 |
| EEG-fNIRS adapter | **已验证**：`adapters/normwear.py` 负责 rate conversion、CWT、channel identity/support 校验、chunked encoder 和 linear-probe wrapper；chunked vs upstream 的 bitwise/数值界限在 smoke evidence 中通过。 |
| 数据边界 / join / cache | **已验证**：`alignment_data.py` 读取 method-neutral registry，逐 task 验证 public split hash、canonical sample identity、真实 EEG/HbO/HbR 和 masks；6 个完成 cache 每个有 `features.npy`、`metadata.npz`、`identity.json`、`status.json`。 |
| train/eval / public matrix | **已验证**：`run_public_development_v2.py` 仅训练 outer-training linear head；`run_public_matrix_v2.py` 为单 GPU、单 job、0 retry；`finalize_public_matrix_v2.py` 生成 completion summary 并把 A8 写入最终 cell evidence。 |
| 审计脚本 / 测试 | **已验证**：identity、data-boundary、alignment、adapter-smoke、public-run、matrix-finalize 六类审计脚本存在；4 个测试模块共 35 个 pytest case（含参数化 case）通过。 |

## 输入合同核对

1. **身份和信息预算（已验证）**：报告名为 `normwear_eeg_fnirs_adapted`，profile 为 `support_matched_direct`；5 个共享任务与 BrainFusion 的 13 个 method-neutral comparison fields 做 exact equality 检查，DSR 另有完整 direct evidence。
2. **模态和采样（已验证）**：canonical EEG 200 Hz、fNIRS 10 Hz，显式区分 `fnirs_hbo` 与 `fnirs_hbr`；确定性 `scipy.signal.resample_poly` 到模型 65 Hz，EEG `[13,40]`、fNIRS `[13,2]`，无 target 信息、未叠加 upstream `basic_preproc`。
3. **窗口和语义（已验证）**：anchor 是 `canonical_registry_window_start`，无 pre/post extra context；MI/MA/WG/n-back/Visual 为同步 8 s offline full-window，DSR 为同步 2 s，不能声称 event-level fNIRS response。
4. **通道和几何（已验证/声明边界）**：所有通道来自冻结真实 inventory，顺序 EEG→HbO→HbR，禁止 copy/padding/mirror；geometry manifest hash 被记录，但 NormWear 没有额外几何编码，不把模板坐标说成 subject digitization。
5. **CWT / tokenization（已验证）**：pinned optimized PyTorch Ricker CWT，3 个 signal/一阶差分/二阶差分 plane，scale 0.1–64、step 1、65 scales，最大 wavelet length 640；9×5 non-overlap patch、pretrained grid 43×13、bicubic position interpolation、12 层 encoder，channel-attention chunk size 16 且完整 CLS 融合。
6. **mask / 数据边界（已验证）**：adapter 只接受全部 recorded 且 analysis-valid 的 support；任何缺失/填充 channel 或时间点直接拒绝，不把 artifact/QC 当成 observed signal。REFED 的真实 partial terminal support 因此保持整个 cell unsupported。

## 输出合同核对

- **表示**（已验证）：`final_encoder_layer_norm_tokens` 为 `[B, real_channels, patches+CLS, 768]`；patch 维 mean（含 channel CLS），再按冻结 delivered order 拼接 channel vectors，不做跨模态平均。
- **head / 任务**（已验证）：encoder 全冻结，仅 outer-training linear probe 可训练；输出维度分别为二类（MI、MA、WG）、三类（n-back）、二类（DSR）、四类（Visual）。
- **预测 schema**（已验证）：每个 public run 的 `.npz` 含 `logits`, `target`, `dataset_index`, `subject`, `sample_id`；manifest/checkpoint 还记录 split、feature cache identity、standardizer fit membership 和 hash。
- **指标 / 聚合**（已验证到 public development）：主指标是 classification `macro_f1`；每 task 有 5 folds × 3 seeds 的 validation cell metrics，保留 mean/SD/min/max。**缺失**：protected test prediction、outer-fold final aggregate、metric-acceptance decision，因此不能把 validation mean 当最终数字。

## 结果生成阶段

| 阶段 | 状态 | 证据与计数 | 边界 |
| --- | --- | --- | --- |
| A0 identity/cell registration | **pass（已验证）** | 6 supported + 1 unsupported；identity summary status `pass` | unsupported REFED 在看分数前冻结 |
| A1–A4 data boundary | **pass（已验证）** | 6 个 cell 全量 public；22,442 unique samples；所有输入有限/非恒定/真实且 support 完整 | 早期 `data_boundary_summary.json` 的 `A0_A4_pass_A5_A8_pending` 是历史快照 |
| A5/A6 executable smoke + source fidelity | **pass（已验证）** | `adapter_smoke_v2/summary.json`；upstream unchunked bitwise exact；chunked max token diff 0.002161、mean diff 0.0000621、pooled max diff 0.0001994、cosine 0.99999988；encoder gradient count 0 | smoke 证明连通性和边界，不是最终性能 |
| A7 production replay | **pass（已验证）** | 6 个 task cache；22,442 rows；每个 cache first/full replay exact；feature dtype float32 | public cache，不含 protected |
| A8 public development/freeze | **pass（已验证）** | `90/90` job，0 failed，0 retries，max concurrency 1；每 task 15 jobs | `matrix_completion_summary.table_admissible=false` |
| final metric acceptance / protected | **未完成（缺失/受锁）** | 无 protected evaluation authorization、无 final aggregate、无 `runs/aggregate` | 不应从 public validation 生成论文数字 |

仅供开发诊断的 validation macro-F1 cell mean（不是最终结果）为：MI `0.55529`、MA `0.62159`、WG `0.61829`、n-back `0.45331`、DSR `0.58932`、Visual `0.27721`；来源 `evidence/public_development_v2/matrix_completion_summary.json`，该文件明确 `claim_boundary=public_development_only_not_table_admissible`。

## 证据质量、风险和阻断

### 已验证证据

- `evidence/identity_v2/summary.json`：source revision、源文件 hash、checkpoint 结构、6/1 cell registration。
- `evidence/adapter_smoke_v2/summary.json`：GPU smoke、CWT/feature shape、bitwise replay、linear head update、encoder gradient 0。
- `evidence/alignment_v2/{motor_imagery,mental_arithmetic,wg,nback,dsr,visual}.json`：每个 cell A0–A8、sample count、feature shape/hash、channel/mask/branch fingerprints。
- `evidence/alignment_v2/summary_final.json`：最终当前状态、A0–A8 cell reports、completed public job count 90、protected locked。
- `evidence/public_development_v2/matrix_completion_summary.json`、`runs/public_development_v2/matrix_v2/controller_status.json`、`completed_public_audit.json`：90/90 completion、0 failure、0 retry、每 job artifact audit。
- `comparative_methods/NormWear/tests`：执行得到 `35 passed, 1 warning`。

### 阻断（需外部授权或新合同）

1. Protected evaluation 仍锁定，且当前交付队列状态为 `none_public_delivery_queue_complete`；没有授权不能生成 final/protected 数字。
2. REFED 需要 temporal support-mask 和 masked sequence-regression 合同；不能通过 padding 或删样本“补齐”。
3. NormWear 没有 `runs/aggregate` 或 metric-acceptance final artifact；即使 public job 全部完成，也不能直接进论文表。

### 风险 / 清理项

- **已发现的残余 cache（机器事实）**：`runs/public_feature_cache_v2/nback/987e74b108e98fd6acfc8b02fc9a051b4b6ff71e220e24f0edc736c9e50a18cd` 仍为 `state=validating`（702/702 rows），另有最终使用的 `f82d2ef5…` `state=complete` cache。当前 job manifest 指向后者，未见 protected flag；发布前应标注或清理旧 cache，避免自动发现歧义。
- **汇总字段残余（机器事实）**：`summary_final.json.task_reports` 仍保留 1 条 Visual `A0_A7_pass_A8_pending` 历史条目，但 `summary_final.json.tasks` 的六个 supported task 全为 `A0-A8_pass...`，且顶层 status 为 complete；发布图表应以 `tasks`/cell 文件为准并标记该残余。
- **运行器重放保护（已观察）**：在当前 queue state 下直接执行 public matrix `--dry-run` 或 `audit_public_run_v2.py` 会因 `active_delivery_method=none_public_delivery_queue_complete` 拒绝；这符合交付完成后的串行保护，不推翻 retained evidence，但若要重审需使用只读/隔离的审计上下文。
- **公开 exposure 声明（声明）**：manifest 写明 target-corpus overlap 为 `none_by_declared_dataset_identity`；这是 provenance 声明，不等同于重新审计上游预训练全集。

## 轻量测试 / 审计记录

| 命令 | 结果 |
| --- | --- |
| `.venv/bin/python -m pytest -q comparative_methods/NormWear/tests` | **35 passed, 1 warning, 15.53 s**；GPU adapter、identity、data boundary、A7/A8 schema tests 均执行通过。 |
| `.venv/bin/python comparative_methods/NormWear/audit_identity_v2.py` | **pass**；revision、upstream clean、checkpoint 261 entries/136,116,425 elements、strict model match、cell registration 均重算通过。 |
| `.venv/bin/python comparative_methods/NormWear/run_public_matrix_v2.py --launch comparative_methods/NormWear/configs/public_matrix_launch_v2.yaml --dry-run` | **按预期拒绝**：`active_delivery_method='none_public_delivery_queue_complete'`；未启动任何 job，未写 protected 工件。 |
| `.venv/bin/python comparative_methods/NormWear/audit_public_run_v2.py .../pilots/nback_outer0_seed17_full` | **按预期拒绝**：同一完成队列保护；保留的 full-pilot audit JSON 已在测试和最终 evidence 中验证。 |

## 最终判断

NormWear 是当前仓库中“公开适配交付完成、protected/final 数字尚不可用”的方法：代码、输入边界、适配器、公开 replay 和 90-job public-development matrix 已具备复现条件；结果生成尚停在 public-development，不得写成最终性能或原论文 fNIRS 复现。下一步只应由独立授权流程决定 protected unlock 与 cell-level metric acceptance，并先处理旧 cache/旧汇总字段的可见性；本审计不授权这些动作。
