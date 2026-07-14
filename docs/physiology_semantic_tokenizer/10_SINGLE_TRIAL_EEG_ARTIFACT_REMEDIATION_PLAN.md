# Single-Trial EEG 伪影修复与准入计划

_Planned: 2026-07-14_

_Execution status: candidate implemented and fully audited; not admitted_

## 当前判定

Single-Trial EEG 当前不能被标记为 artifact-clean。统一 loader 仍从发布目录
`with occular artifact/cnt.mat` 读取任务记录，只排除非 EEG channel，并执行
1–45 Hz band-pass、200 Hz resample 与 full-record robust standardization；这些步骤
不会去除传播到额区 EEG 的眨眼/眼动成分，也不会充分处理 30–45 Hz 的肌电污染。

现有证据包括：

- 数据发布目录明确标为 `with occular artifact`；
- S19 session 2 的既有检查发现 11/30 EEG channels 与 EOG 明显相关；
- 当前四数据集质量报告的 EEG PSD 仍出现显著异常；
- 每个被试另有 `cnt_artifact.mat`、`mrk_artifact.mat`、`mnt_artifact.mat` 控制伪影记录，包含 EOG、EMG、眨眼、咬牙和张口条件，可用于验证检测器，但不是任务数据的“clean 版本”。

因此修复目标是生成可审计的 cleaned branch 和 artifact mask，同时保留 raw branch；
不能用滤波或 robust-SD 后振幅看似接近作为清理成功证据。

## 分阶段实现

### A0 — 先冻结 raw/controlled-artifact 对照

对 29 名被试逐 session 建立 `single_trial_eeg_artifact_audit_v2`：记录通道类型、
原始采样率、EOG/ECG/呼吸参考、事件时刻、PSD、line-noise ratio、低频大幅偏转、
30–45 Hz 高频比例、flatline、极端振幅、channel correlation 与 EOG→EEG coupling。
`cnt_artifact.mat` 的五类控制条件用于检测器灵敏度/特异性校准，任务标签不参与阈值拟合。

### A1 — 新增 provenance-preserving 双分支

统一 loader 为 Single-Trial 增加显式配置：

- `eeg_signal_branch=raw_with_ocular_artifact`：保留当前行为，用于回归对照；
- `eeg_signal_branch=single_trial_eeg_artifact_clean_v2`：新主候选；
- 两个分支使用相同事件、时钟、channel geometry 和 sample IDs；
- 输出 `artifact_mask`, `bad_channel_mask`, `cleaning_state` 与处理参数/代码 hash；
- 原始数据不被覆盖，cleaned cache 使用新 schema/version。

### A2 — 清理候选流水线

按完整 recording、在裁窗之前执行：

1. channel typing 与非有限值修复；
2. 1–45 Hz band-pass，并审计本地电源频率；仅当 50 Hz 污染会通过过渡带混叠时加入明确记录的 notch；
3. 使用 flatline、robust variance、邻道相关、频谱偏离和高频 burst 建立 bad-channel mask；
4. 在保留 VEOG/HEOG reference 的阶段执行 robust EOG regression 或 ICA ocular-component removal；
5. 使用控制伪影记录验证 BSS-CCA/ICA/ASR 肌电候选，不能仅凭任务数据上 PSD 变平选择算法；
6. 基于 `mnt` geometry 插值可恢复坏道，记录无法插值的 subject 14 geometry 缺失；
7. average/reference 策略、200 Hz resample、full-record robust standardization；
8. 最后切出统一的 20 秒窗口，并传播 sample/channel artifact masks。

算法选择通过训练被试和控制伪影记录完成。阈值使用分被试/分 session 的 robust
分布、训练-only calibration 和版本化 evidence，而不是硬编码一个跨数据集绝对振幅阈值。

## 防止“清理即删信号”的必要对照

每个候选同时报告 raw、cleaned、仅 EOG regression、仅 muscle correction 和
sham-component removal。必须检查：

- EOG reference 与额区 EEG 的相关/相干是否下降；
- blink-locked 大幅低频偏转是否下降；
- 30–45 Hz burst 与控制 EMG template 相似度是否下降；
- 8–13 Hz alpha peak、非额区谱形、事件边界和通道拓扑是否保留；
- cleaned-minus-raw 是否主要落在被检测的 artifact 时间/成分，而非全记录生理信号；
- 标签、EEG/fNIRS offset、window count 和 split hash 是否完全不变。

任务分类提升不是清理成功的必要条件，也不能单独证明去伪影正确；它只作为次级
signal-preservation 检查。

## 准入 gate

`single_trial_eeg_artifact_clean_v2` 进入正式训练前必须同时满足：

1. deterministic loader/branch/schema 测试通过；
2. 29 subjects 的 audit 完整，无 silent channel drop 或 sample loss；
3. controlled-artifact sensitivity 明确优于 sham/null，并报告被试级不确定性；
4. EOG/EMG 污染相对 raw 分支下降，且关键 EEG 谱形/拓扑保留；
5. 事件标签、EEG/fNIRS 对齐、20 秒窗口和 geometry contract 不漂移；
6. adaptive QC calibration、算法版本和所有排除原因进入 manifest；
7. 先通过 S19 smoke，再通过全数据 dry-run/short-formal，之后才切换 registry 的
   `default_eeg_signal_branch`。

在 gate 通过前，Single-Trial 可用于 loader correctness 和 raw-vs-cleaning 诊断，
但正式 tokenizer 训练必须排除该数据集或显式标记为 artifact-contaminated ablation。

## 计划代码与产物

- 保留 `experiments/inspect_eeg_artifacts.py` 作为旧的单 session 诊断；新增
  `experiments/audit_single_trial_eeg_artifact_v2.py` 承担全被试批量审计和控制伪影校准；
- 新建 `src/data/eeg_artifact_preprocessing.py`：无标签、record-level、可版本化的清理流水线；
- 扩展 `src/data/unified_physiology.py`：signal branch、masks 和 cleaning provenance；
- 新建 `tests/test_eeg_artifact_preprocessing.py`：synthetic blink/EMG、mask、reload 和 alignment invariance；
- 产物目录：`experiments/runs/physiology_semantic_tokenizer/data_quality_audit/single_trial_eeg_artifact_v2/`；
- 汇总文件：`artifact_calibration.json`, `subject_session_qc.jsonl/.csv`,
  `raw_cleaned_psd_comparison.*`, `signal_preservation.json`, `admission_decision.yaml`。

## 2026-07-14 执行结果

已完成 A0、A1 和保守版 A2：

- 新增 `single_trial_eeg_artifact_clean_v2` record-level branch；raw 分支不覆盖；
- raw/clean 使用同一 join key、事件、标签、EEG/fNIRS 时钟、20 秒窗口和 geometry；
- loader 保留 VEOG/HEOG，在裁窗前完成 1–45 Hz filter、robust lagged EOG
  regression、adaptive bad-channel consensus、geometry-aware interpolation 和
  高频 burst mask；
- 高频肌电当前只进入 `artifact_mask`，没有在未验证前被强行消除；
- `valid_mask` 仍只表达 record boundary，`analysis_valid_mask` 额外排除 artifact
  samples，`bad_channel_mask` 独立传播；
- registry 默认仍为 `raw_with_ocular_artifact`，候选状态为
  `artifact_clean_v2_candidate_not_admitted`；
- 统一质量报告支持 `--eeg-signal-branch`，并报告 branch/schema、artifact、bad
  channel 和 analysis-valid 比例。

首轮 29-subject audit 使用“任一 QC 指标异常即坏道”的规则，坏道中位数触及上限
6，alpha topology 5% 分位仅 0.396，因此该规则被否决并保留为失败校准证据。
正式候选改为“至少两个指标一致异常，或单项达到极端异常；flatline 直接异常”。

第二轮覆盖 29 subjects × 6 sessions = 174 records：

| 检查 | 结果 |
| --- | ---: |
| EEG/EOG source sample/channel loss | 0 |
| artifact-mask fraction | median 0.154; 5–95% 0.043–0.238 |
| bad-channel count | median 3; 5–95% 0–6 |
| median EOG correlation | raw 0.531 → clean 0.028 |
| clean/raw EOG-correlation ratio | median 0.058 |
| alpha-power ratio | median 0.877 |
| non-frontal alpha topology correlation | median 0.967; 5–95% 0.635–0.997 |
| negative non-frontal alpha-topology records | 0 / 174 |
| controlled-artifact availability | 28 / 29 subjects; subject 14 missing |

受控记录给出正确的相对方向：Eye Blinking 的低频比例高于 Teeth Clenching，
Teeth Clenching 的 30–45 Hz 比例高于 Eye Blinking。这只验证检测特征方向，
不等价于肌电修正算法已通过 sham/null 对照。

真实 unified-loader S19 对照确认 raw/clean 的 window count、join key、event、label、
alignment、EEG shape、fNIRS values 和 boundary valid masks 全部相同。cleaned-branch
统一质量报告的既有 contract checks 全部通过。

审计产物位于 gitignored run 目录：

```text
experiments/runs/physiology_semantic_tokenizer/data_quality_audit/
  single_trial_eeg_artifact_v2/full_29_subject_consensus_20260714/
  single_trial_eeg_artifact_v2/unified_report_clean_v2_20260714/
```

### 当前准入决定

**不准入正式训练。** 已通过全数据覆盖、无 silent loss、EOG reduction、控制记录
隔离、控制特征方向、坏道饱和和非额区 alpha 保留检查；尚未通过
`muscle_correction_validated_against_sham`。因此不物化正式 cleaned training cache，
也不切换 registry 默认 branch。下一步只需围绕受控 EMG/咬牙/张口条件完成
mask sensitivity、sham/null 与信号保留对照；通过后再生成版本化 cache 并复跑 gate。
