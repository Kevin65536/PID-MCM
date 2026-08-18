# 对比方法目录完整阅读指南

本目录是统一对比实验的代码、协议、来源审计、public-development 证据和
protected campaign 控制面。它不是一个可直接遍历全部文件的普通源码目录：
`upstream/`、`checkpoints/` 和大部分 `runs/` 是本地资产，Git 中保留的是可审计合同、
必要代码、摘要证据和小型 shadow 工件。

本指南不再维护当前状态。统一的 execution/scientific-verdict 投影见
[`docs/PROJECT_STATUS.md`](../docs/PROJECT_STATUS.md)；本目录中的日期、计数和表格
属于相应协议或历史执行记录。2026-08-14 campaign 的完整主指标、fold SD、终态和证据哈希见
[`docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md`](../docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md)。

## 一、推荐阅读顺序

### 1. 十分钟掌握当前结论

依次阅读：

1. 本文件：了解目录分层和事实优先级。
2. [`docs/PROJECT_STATUS.md#对比实验`](../docs/PROJECT_STATUS.md#对比实验)：由统一
   registry 生成的当前执行状态和科研判定。
3. 本地忽略的 `evidence/protected_campaign/orr_preflight_v1.json`：正式执行前的
   机器可读 `GO` 快照；共享仓库以结果报告中的哈希和结论为准。
4. [`docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md`](../docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md)：
   完整 42-cell 结果、准入终态和证据绑定。
5. [`ROUND_ARTIFACTS_20260812.md`](ROUND_ARTIFACTS_20260812.md)：正式运行前的历史
   `NO-GO` 快照；它保留原始时间语义，不再代表当前状态。

不要从某个方法的旧 README、历史 ORR 或单一结果文件推断全局状态；冻结 candidate、
sealed campaign status、unblind、aggregate 和 cell-level acceptance 必须联合解释。

### 2. 三十分钟理解科学比较口径

依次阅读：

1. [`docs/comparisons/PROTOCOL.md`](../docs/comparisons/PROTOCOL.md)：数据、split、
   任务、主指标和比较边界。
2. [`ADAPTER_ALIGNMENT_GATES_V2.md`](ADAPTER_ALIGNMENT_GATES_V2.md)：A0–A8 对齐门控。
3. [`adapter_alignment_gate_contract_v2.yaml`](adapter_alignment_gate_contract_v2.yaml)：
   上述门控的机器合同。
4. [`comparison_metric_targets_v1.yaml`](comparison_metric_targets_v1.yaml)：最终数字的
   baseline、minimum、preferred target 和准入终态。
5. [`docs/comparisons/METRIC_ACCEPTANCE.md`](../docs/comparisons/METRIC_ACCEPTANCE.md)：
   指标准入的解释性说明。
6. [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)：方法队列、资源调度、失败与停止规则。

这一步回答“比较是否公平、主指标是什么、哪些结果只能作 context/overlap、什么数字能
进入论文表格”，不回答某次运行是否已经完成。

### 3. 审计实现和证据

先看联合门控，再进入方法目录：

1. [`audit_adapter_alignment.py`](audit_adapter_alignment.py)
2. [`evidence/joint_protected_unlock_candidate_v2.json`](evidence/joint_protected_unlock_candidate_v2.json)
3. 每个方法的 `sources/`、`configs/alignment_v2.yaml`、`alignment_data.py` 或 adapter、
   `evidence/alignment_v2/summary_final.json`、public matrix summary、runner 和 tests。

推荐在“来源身份 → 输入/适配合同 → public 证据 → 执行代码 → 测试”的顺序内阅读，
不要先看模型代码再反推它获得了哪些数据。

## 二、事实来源优先级

同一问题出现不一致时，按以下顺序判定：

1. **冻结机器合同和精确 SHA**：release candidate、authorization、lane、unblind、
   metric target、adapter gate contract。
2. **运行终态证据**：`completed_public_audit.json`、`summary_final.json`、matrix
   completion summary、ORR 和 job status/checksums。
3. **受控实现**：builder、worker、controller、aggregator、方法 runner 和 adapter。
4. **说明文档**：本指南、`docs/comparisons/`、方法 README、source-fidelity 说明。
5. **本地重资产**：`runs/`、`checkpoints/`、`upstream/`。它们必须由前四层的路径和
   hash 解释，不能凭目录存在就视为有效结果。

Git 未保存完整 public feature cache 或 checkpoint；release candidate 保存了它们的
路径、大小、角色和 SHA-256。缺少本地资产时应重新获取或重建并核对 hash，不应修改
candidate 迁就当前机器。

## 三、2026-08-14 冻结方法与任务矩阵

统一任务为 MI、MA、WG、n-back、DSR、Visual 和 REFED。正式候选共 42 cells，
36 supported、6 unsupported；每个 supported cell 为 5 folds × 3 seeds，即 15 jobs。

| 方法 | 正式任务 | Jobs | 阅读时必须保留的边界 |
| --- | --- | ---: | --- |
| BIOT | 六个分类任务 | 90 | EEG-only；REFED unsupported |
| CBraMod | 六个分类任务 | 90 | EEG-only；REFED unsupported |
| REVE | 六个分类任务 | 90 | MI/MA 为 target-corpus-overlap 附表；REFED unsupported |
| EFRM | 六分类 + REFED | 105 | synchronized in-domain adaptation；线性 probe 为主 track |
| NormWear | 六个分类任务 | 90 | `normwear_eeg_fnirs_adapted`；REFED unsupported |
| BrainFusion | MI/MA/WG/n-back/Visual | 75 | 独立 NVC-CSP stacking 重实现；DSR/REFED unsupported |
| STA-Net | 本轮 0 个新 job | 0 | 只引用冻结 strict cross-subject aggregate，属于 context reference |

REVE 的 MI/MA 两个 overlap cells 包含在 36 supported cells 内，但不进入 34-cell
support-matched direct 主表。禁止把不同任务覆盖数压成“方法总分”。

## 四、顶层文件地图

| 文件 | 用途 |
| --- | --- |
| `ASSET_STATUS.md` / `audit_assets.py` | 上游源码和权重资产边界 |
| `audit_public_preflight.py` | public-only 输入与运行前检查 |
| `ADAPTER_ALIGNMENT_GATES_V2.md` | A0–A8 人类可读门控 |
| `adapter_alignment_gate_contract_v2.yaml` | A0–A8 机器合同 |
| `audit_adapter_alignment.py` | 六方法 42-cell 联合对齐审计 |
| `comparison_metric_targets_v1.yaml` | 最终数字准入合同 |
| `EXPERIMENT_PLAN.md` | 方法队列和执行计划 |
| `build_joint_protected_unlock_candidate_v2.py` | 构建 42-cell 非授权联合候选 |
| `build_joint_protected_release_candidate.py` | 构建 540-job immutable release candidate |
| `protected_campaign_common.py` | schema、hash、候选/授权/lane/unblind 校验 |
| `protected_campaign_worker.py` | 单 job 盲化推理；不计算或打印指标 |
| `benchmark_protected_campaign_shadow.py` | CPU 重复性、双 GPU 等价、显存/耗时与 lane 冻结 |
| `protected_campaign_controller.py` | `preflight`、`execute`、`status` 和 append-only audit |
| `prepare_protected_authorization.py` | 生成严格非授权的双签模板 |
| `aggregate_protected_campaign.py` | 仅在 SEALED_COMPLETE + 双签 unblind 后聚合 |

## 五、每个方法目录怎么读

所有新方法大体遵循相同结构：

```text
README / ALIGNMENT 状态
sources/                 来源、revision、权重和命名边界
configs/                 输入、适配、public queue 的冻结合同
adapters/ 或模型包       方法特有实现
alignment_data.py        共享数据合同到方法输入的映射
audit_*.py               来源、数据、运行与 reload 审计
run_*.py                 单 job runner
build_* / finalize_*     job matrix 构建与完成封存
evidence/                可提交的小型摘要证据
runs/                    本地大工件，默认不进 Git
tests/                   方法级测试
```

### BIOT

阅读链：

`BIOT/README.md` → `BIOT/sources/method_manifest.yaml` →
`BIOT/sources/SOURCE_FIDELITY.md` → `BIOT/configs/alignment_v2.yaml` →
`BIOT/alignment_data.py` / `BIOT/adapters/biot.py` →
`BIOT/evidence/alignment_v2/summary_final.json` →
`BIOT/evidence/public_development_v2/matrix_completion_summary.json` →
`BIOT/run_public_development_v2.py` → `BIOT/tests/`。

重点检查 PREST-16 权重身份、真实测量 EEG channel 映射、不得消费 fNIRS，以及
public refit head/standardizer 的 fold-local fitting。

### CBraMod

阅读链与 BIOT 相同，并补读 `CBraMod/REPRESENTATION_LAYER_AUDIT.md`。重点检查
200 Hz/200-sample patch 合同、representation layer、padding/mask 和冻结 probe。

### REVE

阅读链与 BIOT 相同，并补读 `REVE/IDENTITY_AND_REPRESENTATION_AUDIT.md`。重点检查
encoder/position bank 身份、电极坐标来源，以及 `Shin2017A` 导致 MI/MA 只能进入
overlap track。

### EFRM

EFRM 是独立 PyTorch 重实现和 LODO 流程，建议按以下顺序：

1. `EFRM-PyTorch/README.md`
2. `EFRM-PyTorch/sources/20260727_LODO_FULL_TARGET_FIVEFOLD_PROTOCOL_FREEZE.md`
3. `EFRM-PyTorch/sources/lodo_full_target_fivefold_v2.yaml`
4. `EFRM-PyTorch/efrm_pytorch/` 中的 data/model/protocol/training/tasks
5. `build_formal_protocol.py`、`run_lodo_pretraining.py`
6. `run_downstream_public_v2.py` 与两个 public matrix 脚本
7. `EFRM-PyTorch/evidence/alignment_v2/summary_final.json`
8. `EFRM-PyTorch/evidence/public_development_v2/matrix_completion_summary.json`
9. `EFRM-PyTorch/tests/`

重点区分历史 resource-bounded v1 与当前 LODO v2，区分 frozen-backbone probe 与
full fine-tuning，并保留 REFED native-coordinate mask-aware contract。

### NormWear

阅读链：`NormWear/README.md` → `sources/` →
`IDENTITY_AND_ADAPTATION_AUDIT.md` → `configs/alignment_v2.yaml` → adapters/data →
adapter smoke、data boundary、identity 和 final alignment evidence → public matrix → tests。

重点：它是 EEG-fNIRS adaptation，不是原论文 fNIRS reproduction；REFED 因当前无法
提供诚实的 partial-time-mask regression contract 而 unsupported。

### BrainFusion NVC-CSP Stacking

阅读链：`BrainFusion-NVC-CSP-Stacking/README.md` → `sources/method_manifest.yaml` →
`OBSERVATION_BUDGET_AUDIT.md` → `configs/alignment_v2.yaml` →
`adapters/brainfusion_gpu/` 的 features/nvc/stacking/pipeline → public runner → final evidence → tests。

重点：公开源码只有 NVC 参考，完整 paper-case pipeline 未公开，因此必须使用
`brainfusion_nvc_csp_stacking_reimplementation` 名称；所有 CSP、base learner 和 stacking
必须在 outer-training support 内拟合。

### STA-Net

`STA-Net-PyTorch/` 是既有独立重实现。先读 README，再读 active protocol freeze、
模型/adapter、five-fold aggregate 和 tests。2026 campaign 未重新运行 STA-Net，也未打开历史
predictions；只把冻结 aggregate 作为 `method_native_context_reference`。

`STA-Net/` 和 EFRM 的原始上游 checkout 含嵌套 `.git`，属于本地来源镜像，不是本仓库
提交入口。

### `single_modal_eeg/`

这是 BIOT、CBraMod、REVE 的共享早期 public runner。它只接受 public manifest，不能
接受 protected manifest。冻结六方法正式 public matrix 的权威实现仍以各方法 v2 runner
及 release candidate 冻结 artifact 为准；阅读它主要用于理解共享 EEG-only 输入合同和
早期 270-job 队列设计。

## 六、protected campaign 阅读与执行链

### 执行前审查层

1. `audit_adapter_alignment.py` 汇总方法证据。
2. `build_joint_protected_unlock_candidate_v2.py` 生成 42-cell eligibility candidate；
   它不包含 measured-data 执行状态。
3. `build_joint_protected_release_candidate.py` 冻结 540 jobs、source/config/environment、
   split fingerprint、checkpoint/pipeline/cache/input-contract hashes 和失败策略。

### shadow 与 lane 层

1. `protected_campaign_worker.py --surface shadow` 走完整 public 推理路径。
2. 两次 CPU shadow 必须逐字段、dtype、shape bitwise 相同。
3. 原双 GPU 等价路线曾 fail-closed；随后经具名双负责人明确授权，切换为固定单 GPU
   self-consistency 路线。
4. benchmark 记录 UUID、driver、CUDA/ECC、显存峰值和中位耗时，并把全部 540 jobs
   固定到同一 GPU UUID，禁止自动迁移。

上述 shadow、单 GPU lane freeze 和正式运行均已完成。旧的双 GPU `NO-GO` 是
2026-08-12 的历史状态，不应再作为统一 registry 的当前状态。

### 历史签署、运行和揭盲层

1. lane 冻结后重新构建 release candidate；已完成。
2. 两名不同负责人在独立 authorization 记录中签署 GO；已完成。
3. controller `preflight` 零 reasons 后执行正式队列；已完成。
4. worker 只写 prediction、identity、必要 target/mask、状态和 checksums，不计算指标；
   540/540 jobs 已 sealed complete。
5. 相同责任角色生成双签 unblind；已完成。
6. aggregator 复核每个 job 的 candidate/auth/input/artifact/device/determinism/checksum/
   coverage 绑定后，计算 seed、fold、OOF companion 和 cell 终态；已完成。

attempt-2 只允许在同一冻结 GPU UUID 上作一次技术恢复。assigned GPU 不可用时进入
`INCOMPLETE_TECHNICAL`，必须生成新 candidate 并重新双签，不能隐式迁移 lane。

## 七、2026 campaign 保留证据

- Release candidate：`evidence/protected_campaign/joint_release_candidate_v1.json`
- 历史签署记录：`evidence/protected_campaign/authorization_template_v1.json`（保留旧文件名，
  工作区内容是该 campaign 的已签署执行记录）
- Pre-run ORR：`evidence/protected_campaign/orr_preflight_v1.json`
- CPU shadow：`evidence/protected_campaign/shadow_cpu_pass_v1/` 和
  `shadow_cpu_pass_v1_repeat/`
- Local sealed status：`runs/protected_campaign/joint-comparison-protected-20260813-v3-single-gpu/`
- Local unblind：`evidence/protected_campaign/unblind_manifest_v1.json`
- Tracked result report：
  [`docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md`](../docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md)

`ROUND_ARTIFACTS_20260812.md` 只记录运行前历史快照。最终 SHA、完成计数和结果来源
见新结果报告。CPU shadow 仍只是 public validation 工件，不是 protected 性能数字。

## 八、安全阅读规则

- 已完成的双签揭盲只记录 aggregator 在冻结范围内的历史处理边界，不意味着应人工逐个打开
  `protected_predictions.npz`。
- 后续任何新 candidate 或新协议重新从 protected closed 状态开始，不能继承本轮签署记录。
- 可以读取 candidate 中已登记的 path、hash、sample count 和 split fingerprint。
- 可以读取本轮两个 `shadow_cpu_*` 目录，因为它们明确为 public surface。
- 性能阅读以聚合报告为准，不从 worker/controller stdout、单 job status 或 audit log
  搜索 sample-level payload。

## 九、常用只读验证命令

```bash
.venv/bin/python comparative_methods/build_joint_protected_unlock_candidate_v2.py --check
.venv/bin/python comparative_methods/build_joint_protected_release_candidate.py --check
.venv/bin/python -m pytest -q \
  tests/test_joint_protected_unlock_candidate_v2.py \
  tests/test_adapter_alignment_gate_contract.py \
  tests/test_protected_campaign_v1.py
```

六方法测试应分别运行，避免不同方法目录中同名模块污染同一 pytest 进程：

```bash
for suite in BIOT CBraMod REVE EFRM-PyTorch NormWear BrainFusion-NVC-CSP-Stacking; do
  .venv/bin/python -m pytest -q "comparative_methods/${suite}/tests"
done
```

查看已完成 campaign 的 sealed 状态时使用 controller 的只读 `status`：

```bash
.venv/bin/python comparative_methods/protected_campaign_controller.py status \
  --candidate comparative_methods/evidence/protected_campaign/joint_release_candidate_v1.json \
  --output-root comparative_methods/runs/protected_campaign/joint-comparison-protected-20260813-v3-single-gpu
```

不应为复现文档状态而再次调用 `execute`。`orr_preflight_v1.json` 是执行前快照，
`campaign_status.json` 和 aggregate 才描述执行后终态。

## 十、Git 与本地资产边界

Git 应包含：合同、受控源码、测试、source-fidelity 文档、alignment/public completion
摘要，以及不含 sample-level payload 的汇总结果报告。正式 campaign 的精确身份由
tracked 报告中的哈希绑定，不要求把执行控制文件本身提交到 Git。

Git 默认不包含：上游 checkout、模型权重、完整 feature cache、训练 checkpoint、大型
public/formal runs、protected manifests、protected predictions、run-specific candidate、
authorization、ORR、具名签署/揭盲记录和完整 aggregate/traceability。任何需要长期保存
的正式工件应先生成 checksum/manifest，并按项目归档策略保存，而不是通过取消
`.gitignore` 直接塞入提交。
