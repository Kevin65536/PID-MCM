# Single-Trial EEG 伪影修复与准入计划

_Planned: 2026-07-14_

_Execution status: v3 implemented, cached, fully audited and admitted_

> **Authority update, 2026-07-25:** the clean branch remains available, but
> artifact detection no longer has token-invalidity authority. Current
> `valid_mask` is boundary/finite measurement validity; artifact fields are QC
> annotations and sensitivity strata. See
> [`2026-07-25_disable_eeg_artifact_mask_authority.md`](../project_changelog/2026-07-25_disable_eeg_artifact_mask_authority.md).

## 最终判定

Single-Trial 原始 EEG 仍必须标记为 `raw_with_ocular_artifact`，但经过受控伪影
记录校准、循环移位 sham 对照和 29-subject 审计后，
`single_trial_eeg_artifact_clean_v3` 已准入为统一 loader 的默认分支。原始分支和
历史 v2 mask-only 分支均保留，任何结果都可以回退或做 raw/clean ablation。

现有证据包括：

- 数据发布目录明确标为 `with occular artifact`；
- S19 session 2 的既有检查发现 11/30 EEG channels 与 EOG 明显相关；
- 当前四数据集质量报告的 EEG PSD 仍出现显著异常；
- 每个被试另有 `cnt_artifact.mat`、`mrk_artifact.mat`、`mnt_artifact.mat` 控制伪影记录，包含 EOG、EMG、眨眼、咬牙和张口条件，可用于验证检测器，但不是任务数据的“clean 版本”。

不能用滤波或 robust-SD 后振幅看似接近作为清理成功证据。本次准入依据是任务记录
上的信号保留检查、受控动作相对等大小循环移位 sham 的特异性检查，以及统一
loader 的事件/时钟/标签/fNIRS 不变性检查。

## 发布页给出的 `cnt_artifact` 合同

[TU Berlin hBCI 发布说明](https://doc.ml.tu-berlin.de/hBCI/contactthanks.php)
明确说明，Dataset C 的 motion-artifact experiment 在所有 MI/MA 任务记录之后执行，
artifact experiment 没有前后 resting period；EEG `cnt_artifact` 是五个独立记录：

1. EOG；
2. EMG；
3. eye blinking；
4. teeth clenching；
5. mouth opening。

相应的 `mrk_artifact` 保存 task-onset markers，`mnt_artifact` 保存 montage。发布数据
除 MATLAB 转换、EEG/EEG-fNIRS 降采样及 linked-mastoid reference 外未做伪影清理。
原始研究论文为 [Shin et al., IEEE TNSRE 2018](https://doi.org/10.1109/TNSRE.2016.2628057)。

这里有两条严格边界：

- 五个 `cnt_artifact` 是受控校准/验证记录，不是第五个数据集，也不是任务 EEG 的
  clean target，绝不能逐点从任务记录相减；
- 发布页只给出 onset marker，没有给出每次动作的精确持续时间。因此 v3 用每个
  文件的 marker 间隔自适应确定事件邻域，并把这一点记录为实现推断，不声称是论文
  规定的动作时长。

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

## 2026-07-14 v2 执行结果（历史、已被 v3 取代）

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

### v2 准入决定

**不准入正式训练。** 已通过全数据覆盖、无 silent loss、EOG reduction、控制记录
隔离、控制特征方向、坏道饱和和非额区 alpha 保留检查；尚未通过
`muscle_correction_validated_against_sham`。因此不物化正式 cleaned training cache，
也不切换 registry 默认 branch。下一步只需围绕受控 EMG/咬牙/张口条件完成
mask sensitivity、sham/null 与信号保留对照；通过后再生成版本化 cache 并复跑 gate。

## 2026-07-14 v3 最终实现与准入

v3 在 v2 的 EOG regression、坏道检测/插值和 artifact mask 基础上，只在自适应
高频 burst mask 内对 30–45 Hz 分量做 Hann taper 的门控衰减。它不删除样本、不删除
通道、不对全记录做宽带抑制；修正后的时间点仍保留在 `artifact_mask`，下游可选择
使用 `analysis_valid_mask` 排除它们。受控验证对每种动作使用相同大小的循环移位
mask 作为 sham，并在被试级聚合，避免仅凭“PSD 变平”准入算法。

全量结果覆盖 29 subjects × 6 sessions = 174 task records：

| 检查 | v3 结果 |
| --- | ---: |
| EEG sample/channel loss | 0 |
| artifact-mask fraction | median 0.154; 5–95% 0.043–0.238 |
| bad-channel count | median 3; 5–95% 0–6 |
| median EOG correlation | raw 0.531 → v3 0.029 |
| alpha-power ratio | median 0.877 |
| non-frontal alpha topology correlation | median 0.967; 5–95% 0.635–0.997 |
| negative non-frontal alpha-topology records | 0 / 174 |
| masked 30–45 Hz energy reduction | median 0.977; 5% quantile 0.524 |
| controlled-artifact availability | 28 / 29 subjects; subject 14 missing |

受控动作相对 sham 的结果：

| 条件 | event HF reduction | sham reduction | target > sham subjects | event coverage > circular null subjects |
| --- | ---: | ---: | ---: | ---: |
| EMG | 0.273 | 0.059 | 96.4% | 64.3% |
| Teeth Clenching | 0.963 | 0.004 | 100% | 100% |
| Mouth Opening | 0.350 | 0.026 | 78.6% | 67.9% |

这些结果支持“受控条件对准的保守高频修正”，不支持“肌电已经被完全移除”。眼动
污染继续由 EOG regression 处理；Eye Blinking 不被要求通过 30–45 Hz muscle gate。

最终 admission gates 全部通过，registry 和 `UnifiedPhysiologyWindowDataset` 默认值
已切换到 `single_trial_eeg_artifact_clean_v3`。版本化缓存位于：

```text
data/cache/physiology_semantic_clean_v1/eeg_artifact_clean_v3/
  cache_manifest.json
  subject_01/session_00.npz
  ... 174 records
```

缓存 schema 为 `single_trial_eeg_artifact_cache_v3`，逐记录校验 join key、原始文件
size/mtime、处理配置与代码 hash；不匹配时拒绝使用陈旧缓存。缓存与现场计算在真实
S19/session_00 上 EEG 最大绝对差为 0，artifact/bad-channel masks、fNIRS 和统一
窗口数量完全一致。

最终审计与默认四数据集报告位于：

```text
experiments/runs/physiology_semantic_tokenizer/data_quality_audit/
  single_trial_eeg_artifact_v3/full_29_subject_controlled_sham_cache_20260714/
  final_four_dataset_check_v3_default_20260714/
```

重建命令：

```bash
.venv/bin/python experiments/audit_single_trial_eeg_artifact_v2.py \
  --workers 4 \
  --cache-root data/cache/physiology_semantic_clean_v1/eeg_artifact_clean_v3 \
  --output-dir experiments/runs/physiology_semantic_tokenizer/data_quality_audit/\
single_trial_eeg_artifact_v3/full_29_subject_controlled_sham_cache_20260714
```
