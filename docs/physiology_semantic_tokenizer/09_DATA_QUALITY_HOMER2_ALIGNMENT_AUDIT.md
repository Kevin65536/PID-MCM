# 数据规范化、HOMER2 对齐与统一缓存规范

_Created: 2026-07-08_
_Unified: 2026-07-14_

## 当前结论

`data/cache/physiology_semantic_clean_v1` 与原始 EEG bridge 已推进到四数据集
统一规范加载的 P1 阶段：

- fNIRS signal records、event rows、alignment reports 已共享同一套
  canonical join key。
- 四个原始数据集中的通道/电极位置资料已经审计，并生成统一
  `channel_geometry` sidecar。
- 已实现 `CleanPhysiologyCacheIndex`、fNIRS-first
  `CleanPhysiologyAlignedWindowDataset` 与四数据集 multimodal
  `UnifiedPhysiologyWindowDataset`。
- EEG 不复制进 8.4 GB clean cache，而是由统一 loader 通过 manifest/join key
  回读原始 MATLAB/EDF record；fNIRS 继续读取 clean-cache
  `homer2_aligned_fnirs`。这避免重复存储，同时保留原始文件 provenance。

因此当前支持：

- 四数据集 event-aligned EEG-fNIRS loading；
- EEG 200 Hz、fNIRS 10 Hz 的统一时基；
- EEG 与 fNIRS 均为无量纲 `robust_standard_deviation` 数值单位，同时保留
  native unit metadata；
- fNIRS 统一交错 HbO/HbR component contract；
- canonical task label 与统一 channel geometry row schema；
- raw-native vs HOMER2-aligned fNIRS 对照；
- 四数据集统一事件/标签审计；
- geometry-aware loader 和后续空间邻接扩展；
- teacher-free fNIRS reconstruction/VQ smoke。
- Single-Trial EEG 的 provenance-preserving raw/v2/v3 branches、EOG
  regression、bad-channel/artifact/analysis-valid masks 与 29-subject audit；v3
  已通过受控伪影循环移位 sham 对照并准入为默认分支。

当前不支持：

- 把不同任务、不同空间覆盖的四个数据集误称为物理上完全同质；当前统一的是
  数值单位、成分、时基、预处理、标签 schema 和 geometry schema，不是抹除
  dataset/task provenance；
- physical-teacher-supervised tokenizer training；
- protected formal run；
- 声称 REFED、Visual、Simultaneous 已按完整 HOMER2 optical pipeline
  重新处理。

## 1. 数据规范化总 contract

### 1.1 Canonical Join Key

所有 signal record、event row、alignment report 使用：

```text
join_key = dataset_id|canonical_subject_id|base_record_id
```

字段含义：

- `dataset_id`: 数据集 ID。
- `canonical_subject_id`: 去除数据集特有 modality suffix 后的被试 ID。
- `base_record_id`: 去除 signal branch suffix 后的任务/视频/会话 ID。
- `signal_branch`: 同一个 canonical record 下的信号分支。

已解决的原始不一致：

| 数据集 | 旧 signal/event 不一致 | 新规范 |
| --- | --- | --- |
| REFED | signal `video_1_hbo_hbr` / event `video_1` | `base_record_id=video_1`, branch=`hbo_hbr` |
| REFED absorbance | signal `video_1_absorbance_780_805_830` / event `video_1` | 同一 join key, branch=`absorbance_780_805_830` |
| Simultaneous | signal subject `VP001-NIRS` / event subject `VP001` | `canonical_subject_id=VP001` |
| Single-Trial | subject text `subject 01` | `canonical_subject_id=subject_01` |

实现位置：

- `src/data/clean_physiology_cache.py`
- `experiments/normalize_clean_cache_metadata.py`
- `experiments/build_clean_eeg_fnirs_cache.py`
- `experiments/build_clean_event_index.py`

### 1.2 Signal Branches

每条 `.npz` 至少包含：

- `native_input_fnirs`
- `raw_native_fnirs`
- `homer2_aligned_fnirs`
- `time_s`
- `native_channel_names`
- `homer2_channel_names`

`raw_native_contract` 保留数据集原始测量语义，并做 full-record robust
standardization。它回答的问题是：不虚构同一物理单位时，数据能否进入统一数值坐标。

`homer2_aligned_contract` 是 best-effort HOMER2 对齐分支。它回答的问题是：
尽可能靠近 HOMER2 处理坐标后，teacher 失败是否缓解，以及哪些步骤由于原始文件缺失
无法执行。

### 1.3 Event And Alignment Contract

事件索引位于：

```text
data/cache/physiology_semantic_clean_v1/event_index/
  event_manifest.json
  events.jsonl
  alignment_reports.jsonl
```

事件支持四类粒度：

- `trial`
- `session_block`
- `video_segment_with_continuous_labels`
- `fnirs_csv_mark`

alignment report 记录 EEG/fNIRS 事件数、label match、offset mean/std、
piecewise offset blocks、skipped marker、drift slope 等。Simultaneous 中
`wg`、`nback`、`dsr` 的事件粒度不能强行视为同一种 trial。

## 2. HOMER2 对齐状态

HOMER2/Homer 常见链条为：

1. raw intensity quality/channel pruning；
2. intensity -> optical density；
3. motion detection/correction；
4. band-pass filtering；
5. optical density -> HbO/HbR via MBLL；
6. stimulus rejection、block average 或 GLM/HRF；
7. optional short-channel/global physiology regression。

当前四数据集适配度：

| 数据集 | 当前入口 | 可执行步骤 | 无法完整标准化的缺失 |
| --- | --- | --- | --- |
| `eeg_fnirs_single_trial` | raw 760/850 intensity, `V` | OD、motion suppression、bandpass、MBLL | short-channel regression、subject-specific DPF、完整 HOMER2 pruning policy |
| `simultaneous_eeg_nirs` | MATLAB oxy/deoxy, `mmol/L` | post-conversion motion suppression、bandpass | raw `wl1/wl2` intensity、OD replay、MBLL replay、short channels |
| `refed` | HbO/HbR/HbT + Abs780/805/830 export | post-conversion motion suppression、bandpass | raw light intensity、声明物理单位、short channels；absorbance export 不能恢复 raw intensity |
| `visual_cognitive_motivation` | ETG-7100 Oxy/Deoxy CSV export | post-conversion motion suppression、bandpass | raw 695/830 intensity、source-detector geometry、声明物理单位、short channels |

结论：只有 Single-Trial 具备接近完整 HOMER2 fNIRS optical pipeline 的
必要 raw intensity 输入。其他三个数据集只能做 post-conversion 对齐与 provenance
标记，不能被称为完整 HOMER2-clean。

## 3. 原始伪影与单位状态

fNIRS 单位/物理语义已通过 `src/data/fnirs_standardization.py` 的 measurement
contract 进入代码，但这是“保留原始测量语义并映射到可比较无量纲坐标”，不是把所有
数据宣称为同一 HbO/HbR 物理单位。

TU Berlin 两个数据集不能视为已去伪影：

- Single-Trial 原始 EEG 来自 `with occular artifact` 文件；2026-07-08 S19
  inspection 显示 11/30 EEG channels 有 EOG 相关污染。统一 loader 仍保留该 raw
  provenance，但默认使用版本化 `single_trial_eeg_artifact_clean_v3` 缓存。
- Simultaneous MATLAB 发布说明显示 MATLAB 数据主要是 downsample + format
  conversion，不能等同于已完成 motion/ocular artifact cleaning。

此前 TRTD/teacher 诊断仍保留：

| 数据 | TRTD in-sample R2 | LOO R2 | self-persistence R2 |
| --- | ---: | ---: | ---: |
| Single-Trial S19 session 2 | 0.021923 | -1.89 | 0.997342 |
| Simultaneous VP001 WG | 0.003606 | -0.615709 | 0.991695 |

这些结果不支持当前 physical teacher 直接进入 tokenizer supervision。

## 4. Channel Geometry Sidecar

原始数据中已确认存在位置/通道资料：

| 数据集 | 原始位置资料 | 当前规范化方式 |
| --- | --- | --- |
| Single-Trial | NIRS `mnt.mat`; EEG `mnt_artifact.mat`/`mnt.mat` | per-subject EEG electrode 与 fNIRS channel midpoint |
| Simultaneous | per-subject/task `mnt_{task}.mat` for EEG/NIRS | per-subject per-task EEG electrode 与 fNIRS channel midpoint |
| REFED | `fNIRS_coordinates.csv` | global fNIRS channel midpoint + source/detector index |
| Visual | `Location.ced`; `fNIRS_to_EEG_channel_reference.xlsx` | EEG global CED 坐标；fNIRS channel-to-EEG reference label |

输出位置：

```text
data/cache/physiology_semantic_clean_v1/channel_geometry/
  geometry_manifest.json
  channels.jsonl
```

当前全量 sidecar 统计：

| 数据集 | 条目数 | 说明 |
| --- | ---: | --- |
| `eeg_fnirs_single_trial` | 1884 | 1044 fNIRS + 840 EEG |
| `refed` | 51 | fNIRS coordinates |
| `visual_cognitive_motivation` | 55 | 31 EEG CED + 24 fNIRS-to-EEG reference |
| `simultaneous_eeg_nirs` | 5148 | 2808 fNIRS + 2340 EEG |

注意：Visual fNIRS 当前是 EEG 位置参考标签，不是真实 3D optode/source-detector
坐标；不能用于精确 source-detector distance 或 MBLL replay。

montage 完整性：

- Single-Trial NIRS: 29/29 subjects have `mnt.mat`。
- Single-Trial EEG: 28/29 subjects have `mnt.mat` or `mnt_artifact.mat`;
  `subject 14` currently has no EEG montage file in the local raw tree。
- Simultaneous EEG: 26/26 subjects have all three `mnt_dsr.mat`,
  `mnt_nback.mat`, `mnt_wg.mat` files。
- Simultaneous NIRS: 26/26 subjects have all three task montage files。
- REFED and Visual use dataset-global geometry/reference files rather than
  per-subject montage files。

实现位置：

- `src/data/channel_geometry.py`
- `experiments/build_clean_channel_geometry.py`

## 5. Clean Cache Structure

当前规范目录：

```text
data/cache/physiology_semantic_clean_v1/
  cache_manifest.json
  event_index/
    event_manifest.json
    events.jsonl
    alignment_reports.jsonl
  channel_geometry/
    geometry_manifest.json
    channels.jsonl
  eeg_fnirs_single_trial/
    subject_01/
      session_00.npz
      session_00.manifest.json
  simultaneous_eeg_nirs/
    VP001-NIRS/
      cnt_wg.npz
      cnt_wg.manifest.json
  refed/
    1/
      video_1_hbo_hbr.npz
      video_1_hbo_hbr.manifest.json
      video_1_absorbance_780_805_830.npz
      video_1_absorbance_780_805_830.manifest.json
  visual_cognitive_motivation/
    S01/
      S01_Part1_Probe1.npz
      S01_Part1_Probe1.manifest.json
```

当前全量 join coverage：

- signal records: 1267
- canonical record join keys: 787
- events: 14545（Visual 由 1/2/3 三类原始 mark 改为与 EEG DC9 对齐的
  stimulus-onset trial；不再把三类 mark 当三个训练样本）
- alignment reports: 787
- `record_keys_without_events`: 0
- `event_keys_without_records`: 0
- `record_keys_without_alignment_reports`: 0

signal branch 统计：

- REFED `hbo_hbr`: 480
- REFED `absorbance_780_805_830`: 480
- Single-Trial `homer2_wavelength_pair`: 174
- Simultaneous `oxy_deoxy`: 78
- Visual `oxy_deoxy`: 55

## 6. Unified Loader Contract

当前 loader 实现：

- `CleanPhysiologyCacheIndex`：signal/event/alignment join；
- `CleanPhysiologyAlignedWindowDataset`：兼容 fNIRS-first 路径；
- `UnifiedPhysiologyWindowDataset`：四原始数据集的 EEG-fNIRS 统一路径；
- `DatasetQualityReporter` / `visualize_dataset_quality.py`：只审计四原始数据集，
  Croce cache 仅记录为派生监督目标。

registry 已与实现同步：

| Registry ID | `loader_status` | `primary_loader` | 已声明 interface |
| --- | --- | --- | --- |
| `eeg_fnirs_single_trial` | `implemented` | `UnifiedPhysiologyWindowDataset` | unified、legacy、continuous visualization |
| `refed` | `implemented` | `UnifiedPhysiologyWindowDataset` | unified |
| `visual_cognitive_motivation` | `implemented` | `UnifiedPhysiologyWindowDataset` | unified |
| `simultaneous_eeg_nirs` | `implemented` | `UnifiedPhysiologyWindowDataset` | unified、legacy、continuous visualization |
| `croce_local_cache` | `implemented` | `CroceLocalCacheDataset` | derived cache、legacy；`resource_kind=derived_supervision_cache` |

interface 分开记录是为了不把 REFED/Visual 的统一加载成功误报成旧
`create_continuous_visualization_dataset` 已实现。Visual registry 的 native sampling
与通道信息也已按本地 EDF/CSV 同步为 EEG 500 Hz、fNIRS 10 Hz、30 EEG channels
和 24 个 fNIRS base channels；统一输出仍为 200/10 Hz 和 48 个 HbO/HbR component
channels。

统一输出：

- `eeg`, `fnirs`: channel-first event window；
- `sample_rate_hz={"eeg": 200, "fnirs": 10}`；
- `unit={"eeg": "robust_standard_deviation", "fnirs":
  "robust_standard_deviation"}`；
- `component_roles.fnirs`: 仅 HbO/HbR；
- `label`: `canonical_task_label_v1`，显式分离 namespace、task、condition、
  class index 与 event role；
- `alignment`: 分别使用 `eeg_time_ms` 和 `fnirs_time_ms`，不把固定/分段 offset
  当成同一 clock；
- 默认只接纳 `stable_fixed_offset`、`piecewise_constant_offset`、
  `skip_aligned_piecewise_constant_offset` 与
  `shared_segment_index_no_marker_stream`；continuous-drift/不稳定 record 留在
  sidecar 中供诊断，但不进入统一训练窗口；
- `channel_geometry`: EEG/fNIRS 均返回 `canonical_channel_geometry_v1` rows，
  缺失位置保留 null/provenance，不虚构坐标；
- native unit、原始路径、full-record robust location/scale、filter/resample state。

统一 loader 的默认观测窗为 **20 秒**：EEG `(C_E, 4000)`，fNIRS
`(C_F, 200)`。20 秒用于覆盖较慢的血流动力学响应并与现有 pilot 配置一致；
2 秒 patch 只是模型内部划分。0.01 Hz 频率边缘的 PSD/质量估计仍必须使用至少
100 秒的 record-level 片段，不能用单个 20 秒训练窗估计。

2026-07-10 8 秒首窗 smoke（历史报告，证明当时的 shape/contract；不再代表默认窗）：

| 数据集 | paired windows | EEG shape | fNIRS shape | finite/full-window |
| --- | ---: | --- | --- | --- |
| Single-Trial | 3480 | `(30, 1600)` | `(72, 80)` | pass |
| REFED | 480 | `(64, 1600)` | `(102, 80)` | pass |
| Visual | 3250（31/55 records 因不稳定 alignment 排除） | `(30, 1600)` | `(48, 80)` | pass |
| Simultaneous | 2711（1/78 records 排除） | `(30, 1600)` | `(72, 80)` | pass |

仍需在训练入口补：

1. split manifest 与 protected-test lock；
2. 除 Single-Trial v3 外，其余数据集的 dataset-specific
   bad-channel/window rejection masks；
3. physical teacher targets、uncertainty、valid mask，仅在科学 gate 允许后加入。

## 7. Rebuild Commands

全量 fNIRS signal cache：

```bash
.venv/bin/python experiments/build_clean_eeg_fnirs_cache.py \
  --subjects-per-dataset 1000 \
  --records-per-subject 1000 \
  --include-refed-absorbance \
  --output-dir data/cache/physiology_semantic_clean_v1 \
  --overwrite
```

全量 event index：

```bash
.venv/bin/python experiments/build_clean_event_index.py \
  --subjects-per-dataset 1000 \
  --records-per-subject 1000 \
  --output-dir data/cache/physiology_semantic_clean_v1/event_index \
  --overwrite
```

修复/刷新既有 manifest 的 canonical 字段：

```bash
.venv/bin/python experiments/normalize_clean_cache_metadata.py \
  --cache-root data/cache/physiology_semantic_clean_v1
```

生成 channel geometry sidecar：

```bash
.venv/bin/python experiments/build_clean_channel_geometry.py \
  --output-dir data/cache/physiology_semantic_clean_v1/channel_geometry \
  --overwrite
```

生成四数据集统一质量报告：

```bash
.venv/bin/python experiments/scripts/visualize_dataset_quality.py --all \
  --samples-per-dataset 4 \
  --window-duration-s 20 \
  --output-dir experiments/runs/physiology_semantic_tokenizer/data_quality_audit/final_four_dataset_check_20260714_window20s
```

审计并物化 Single-Trial EEG v3：

```bash
.venv/bin/python experiments/audit_single_trial_eeg_artifact_v2.py \
  --workers 4 \
  --cache-root data/cache/physiology_semantic_clean_v1/eeg_artifact_clean_v3 \
  --output-dir experiments/runs/physiology_semantic_tokenizer/data_quality_audit/\
single_trial_eeg_artifact_v3/full_29_subject_controlled_sham_cache_20260714

.venv/bin/python experiments/scripts/visualize_dataset_quality.py \
  --all \
  --window-duration-s 20 \
  --output-dir experiments/runs/physiology_semantic_tokenizer/data_quality_audit/\
final_four_dataset_check_v3_default_20260714
```

`data/` 仍是 gitignored 本地 artifact。代码、测试和文档进入 git；缓存本体不进入 git。

## 8. 当前训练判定

支持：

- P1 数据合同、事件合同、geometry sidecar、四数据集 multimodal unified loader
  已落地；
- raw-native 与 HOMER2-aligned 分支可对照；
- 四数据集 format/unit/component/preprocessing/label/timing/geometry schema 的
  2026-07-10 final check 全部通过。

不支持：

- 因此直接声称四数据集 scientific equivalence 或 cross-dataset validity；
- 把 post-conversion 数据称为完整 HOMER2-clean；
- 在 split lock、teacher-valid masks 与对应科学 gate 未补齐前启动
  physical-teacher-supervised training；Single-Trial v3 的软件/数据准入不等价于
  physical teacher 获得科学准入。

Single-Trial EEG 的污染处理不是通过把 PSD 异常“标准化掉”来解决。分阶段修复、
对照分支、adaptive QC 和准入条件见
[`10_SINGLE_TRIAL_EEG_ARTIFACT_REMEDIATION_PLAN.md`](10_SINGLE_TRIAL_EEG_ARTIFACT_REMEDIATION_PLAN.md)。
当前 `single_trial_eeg_artifact_clean_v3` 已完成 29 subjects / 174 task records
审计，并以 28 subjects 的 EMG、咬牙、张口受控记录完成循环移位 sham/null 验证。
registry 默认已切换到 v3；raw 与 v2 仍保留用于诊断和消融。发布页没有给出动作
持续时间，因此验证事件邻域来自 `mrk_artifact` marker 间隔的自适应估计，而不是
未经证实的固定时长。
