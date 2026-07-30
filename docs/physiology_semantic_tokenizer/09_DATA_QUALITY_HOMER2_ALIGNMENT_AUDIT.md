# 数据规范化、HOMER2 对齐与统一缓存规范

_Created: 2026-07-08_
_Unified: 2026-07-18_

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
- Simultaneous EEG 已使用 HEOG/VEOG 辅助回归修复并在输出前排除两个眼动通道；
  DSR 已按 EEG 16/32 标记恢复为 Go/No-go stimulus 事件，仍服从统一 alignment gate。
- Visual EEG DC9 已按原始三事件语义解析，统一入口当前接纳 54/55 records、
  7,750 windows 和 16/16 subjects；S06 Part1 继续隔离。
- REFED EEG 的 64 个标准 10–10 标签已映射到固定版本的 FieldTrip
  `standard_1005` template，并物化仅用于 EEG 内部邻接的 64-node/168-edge
  Delaunay graph；template provenance 与非实测坐标边界保留在 sidecar。

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

### 1.4 DSR 恢复与标记契约（2026-07-18）

此前的 `unified_training_hard_exclusion_v1` 已被下列可审计契约取代：

```text
forbidden_task_policy = no_hard_exclusions_dsr_restored_v2
forbidden_task_namespaces = []
task = simultaneous_eeg_nirs:dsr
labels = {16: Go, 32: No-go}
```

边界如下：

- EEG 原始 code 48 与 fNIRS code 3 只作为 block 同步锚点；EEG code 16/32
  才是 stimulus label source；
- 每个 symbol 使用所属已对齐 block 的局部 offset 映射到 fNIRS clock。没有
  fNIRS 锚点的 block 整体不生成 symbol window，禁止插值猜测；
- fNIRS 是 synchronized hemodynamic context，不是独立的 symbol-level marker
  或 2 秒级分类 ground truth；
- 数据论文描述 180 个 symbol trials/participant，但发布 marker stream 对 26/26
  participants 均给出 360 个 16/32 markers。event index 保留发布文件的 360，
  并显式记录该 paper/release discrepancy，不擅自抽半；
- VP001 缺一个 fNIRS block anchor，因此生成 340/360 个事件；其余 25 人生成
  360 个。raw event index 合计 9,340（Go 2,802；No-go 6,538）；
- 默认 alignment gate 排除 `continuous_drift` 的 VP005 DSR，最终准入 25 人、
  8,980 windows（Go 2,694；No-go 6,286）。设置 diagnostic alignment 模式可
  查看 VP005，但这不是绕过科学准入；
- DSR 推荐 EEG epoch 为 2 秒（刺激显示 0.5 秒、约每 2 秒一次）；统一 loader
  的 20 秒默认窗仍只是通用 multimodal context，不应直接冒充 ERP trial。

### 1.5 Visual 原始时序标记契约

Visual 原始说明和数据论文一致声明：每个 trial 有三个事件；EEG 文本 sidecar
用相同的 `DC9` 依次记录刺激出现、3 秒后的刺激消失和参与者反应，而 fNIRS CSV
的 `Mark=1/2/3` 分别显式记录三者。[^1] 因此跨模态 anchor 是 EEG 中“其后约
3,000 ms 出现下一 DC9”的事件与 fNIRS `Mark=1`，而不是机械选取 EEG sidecar
的每第三行。

当前解析规则为：

1. 删除相同时间戳的重复 DC9 行；这修复 S15 Part1 的成对重复 annotation；
2. 在 `3,000 ± 10 ms` 内紧随另一个 DC9 的事件识别为 stimulus onset；
3. 与 fNIRS `Mark=1` 序列执行既有 skip-aware 对齐；
4. 仍按既有 `stable_fixed_offset` 阈值准入，不放宽 alignment gate；
5. 将原始、去重、候选 onset 与 duplicate counts 写入 alignment report。

`event_manifest.json.dataset_timing_contracts.visual_cognitive_motivation` 固化
上述规则及版本 `visual_dc9_stimulus_timing_v1`，避免重建时静默退回行号规则。

重建后 55 条 Visual reports 中 54 条为 `stable_fixed_offset`，offset standard
deviation 约为 28–37 ms；唯一未准入的 S06 Part1 Probe1 只有 108 个可识别
EEG onset、但有 125 个 fNIRS onset，继续保留为 `continuous_drift` 诊断记录。

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
  conversion，不能等同于已完成 motion/ocular artifact cleaning。2026-07-18
  新增 `simultaneous_eeg_eog_clean_v1`：HEOG/VEOG 仅作 0.5–15 Hz robust nuisance
  regressors，逐通道移除方差上限为可配置的 0.5；该分支不做坏道插值，也不做
  muscle-band attenuation。完整 78-record audit 的全通道 median EOG correlation
  从 0.4517 降至 0.0221，眼周通道从 0.7972 降至 0.2209；眼动 mask 外 waveform
  correlation 中位数为 0.9277，15–45 Hz variance ratio 中位数为 0.9965。
  这些结果支持清理/保真软件契约，不证明所有眼动已消失或科学性能必然提升。

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
| REFED | `EEG_channels.csv`、官方 `Figure_1.png`、`fNIRS_coordinates.csv` | EEG 标准-template topology；global fNIRS channel midpoint + source/detector index |
| Visual | `Graphical_recording_head_model.pdf`; `Location.ced`; `fNIRS_to_EEG_channel_reference.xlsx`; raw CSV `Mode,4x4` | EEG global CED 坐标；双侧 4×4 fNIRS 图示/CED template projection + shared-optode adjacency |

输出位置：

```text
data/cache/physiology_semantic_clean_v1/channel_geometry/
  geometry_manifest.json
  channels.jsonl
  refed_eeg_adjacency.json
  visual_fnirs_adjacency.json
```

当前全量 sidecar 统计：

| 数据集 | 条目数 | 说明 |
| --- | ---: | --- |
| `eeg_fnirs_single_trial` | 1884 | 1044 fNIRS + 840 EEG |
| `refed` | 115 | 64 EEG template coordinates + 51 fNIRS coordinates |
| `visual_cognitive_motivation` | 79 | 31 EEG CED + Probe1/Probe2 各 24 个 fNIRS graphical-template coordinates |
| `simultaneous_eeg_nirs` | 5148 | 2808 fNIRS + 2340 EEG |

Visual 的 112 个原始 fNIRS CSV 均声明 `Mode,4x4`，每个 probe 含 CH1–CH24。
原始 `fNIRS_to_EEG_channel_reference.xlsx` 对每侧 24 个通道给出 14 个 EEG
位置锚点、10 个 `-`；`Graphical_recording_head_model.pdf` 给出双侧红/蓝 optode
相对头皮与 EEG 电极布局。当前规范化采用固定的 4×4 shared-optode channel
topology：14 个已标注通道直接继承 `Location.ced` 坐标，10 个未标注通道在
24-channel line graph 上做 graph-Laplacian harmonic interpolation，并投影回 CED
head radius。每侧生成 24-node/52-edge、连通、对称、无 self-loop 的邻接图。
CH1–CH24 的 4×4 index topology 采用公开的 Hitachi 4×4 channel layout；它只定义
共享 optode 关系，实际头皮位置仍完全由本数据集 PDF/xlsx/CED 决定。[^5]

工作簿 Probe2 CH13 的原始标签为不存在于 `Location.ced` 和图示中的 `FP4`；根据
Probe1 CH13=`FC3`、双侧图示与 CED 中的 `FC4` 显式修正为 `FC4`，原值与修正规则
均写入 metadata。loader 现在还会按 record 后缀选择 Probe1/Probe2 geometry，
避免旧实现把两侧同名 CH 通道混为一套坐标。

该补全把 Visual fNIRS loader position availability 从 `54.17%` 提升到 `100%`，
但它仍是 dataset-global graphical/CED template projection，不是真实逐被试 3D
optode/source-detector digitization。只允许用于 within-fNIRS adjacency、图模型输入、
可视化和 coarse EEG-fNIRS alignment；不得用于精确 source-detector distance、
MBLL replay、个体源定位或 exact co-registration。数据论文也只把 PDF 定义为记录
位置图、把 xlsx 定义为 fNIRS-channel/EEG-label mapping，而未声明个体三维测量。[^1]

REFED EEG 不提供逐被试 digitization。本项目只需要通道邻接，因此采用固定
FieldTrip commit `462487e4` 的 `standard_1005.elc` MNI-template 坐标：64 个
REFED 标签中 62 个精确同名匹配；官方 Figure 1 中额外的 `CB1/CB2` 分别按
`PO7–O1`、`PO8–O2` 的三维算术中点显式插值。[^2][^3] 顶视 `x/y`
Delaunay 产生 168 条无向边，图连通，degree range 为 3–8。该坐标和邻接只允许
用于 within-EEG topology、图模型和可视化；不得解释为被试实测电极位置、源定位
精度或 EEG-fNIRS 共配准。标准 template montage 与个体 digitization 的区别也与
MNE 的 montage 定义一致。[^4]

montage 完整性：

- Single-Trial NIRS: 29/29 subjects have `mnt.mat`。
- Single-Trial EEG: 28/29 subjects have `mnt.mat` or `mnt_artifact.mat`;
  `subject 14` currently has no EEG montage file in the local raw tree。
- Simultaneous EEG: 26/26 subjects have all three `mnt_dsr.mat`,
  `mnt_nback.mat`, `mnt_wg.mat` files。
- Simultaneous NIRS: 26/26 subjects have all three task montage files。
- REFED EEG uses a versioned standard-template topology proxy；REFED fNIRS 与
  Visual 使用 dataset-global geometry/reference files，而非 per-subject montage。

实现位置：

- `src/data/channel_geometry.py`
- `src/data/channel_adjacency.py`
- `src/data/assets/refed_standard_1005_montage_v1.csv`
- `src/data/assets/visual_fnirs_4x4_topology_v1.csv`
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
- events: 14547（Visual 由 1/2/3 三类原始 mark 改为与 EEG DC9 对齐的
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
| `refed` | `implemented` | `UnifiedPhysiologyWindowDataset` / `REFEDContinuousSequenceDataset` | unified classification context / continuous sequence regression |
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
- DSR 使用 EEG-native Go/No-go labels 与 block-anchor clock projection，并继续
  受普通 alignment admission gate 约束；
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

2026-07-18 DSR 恢复并重建 event index 后的 20 秒默认入口 window-reference contract：

| 数据集 | 准入 windows | Subjects | 排除说明 |
| --- | ---: | ---: | --- |
| Single-Trial | 3480 | 29 | 无 |
| REFED | 480 | 32 | 无 |
| Visual | 7750 | 16 | S06 Part1 Probe1 原始触发缺失 |
| Simultaneous | 11242 | 26 | 含 8,980 个 DSR Go/No-go windows；VP005 DSR 因 continuous drift 排除 |

重建后的全入口共 22,952 windows；DSR 对外计数为 8,980。以上是 loader-reference 与数据
协议验证，不是模型性能或科学有效性结果。

同日 DSR 恢复前的历史全量 audit 曾对 13,972 个 20 秒窗口完成逐任务统计；下表
保留作为恢复前快照，不再代表当前入口。这里的 sample 数就是 loader window 数；幅值矩按模型实际看到的窗口加权，
重叠窗口中的原始时间点会被重复计数，不应解释为去重后的 raw-record 总体估计。
EEG 使用 `analysis_valid_mask`，fNIRS 使用 `valid_mask`；单位均为 canonical
`robust_standard_deviation`：

该段描述的是 2026-07-18 历史 audit。自 2026-07-25 起，当前统一 loader 的
`analysis_valid_mask` 与 `valid_mask` 相同，只表达边界/数据存在性；
`artifact_mask` 为全 false 兼容字段，不再标记或排除样本。

| 数据集 / 任务 | Subjects | Records | Samples | EEG / fNIRS channels | 标签分布 | EEG std / var | fNIRS std / var |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Single-Trial / mental arithmetic | 29 | 87 | 1,740 | 30 / 72 | BL 870；MA 870 | 1.0431 / 1.0880 | 1.0211 / 1.0427 |
| Single-Trial / motor imagery | 29 | 87 | 1,740 | 30 / 72 | LMI 870；RMI 870 | 1.0291 / 1.0591 | 1.0208 / 1.0420 |
| REFED / emotion video | 32 | 480 | 480 | 64 / 102 | 5 个 video categories 各 96 | 1.2359 / 1.5275 | 1.1146 / 1.2423 |
| Simultaneous / n-back | 26 | 26 | 702 | 30 / 72 | 0/2/3-back 各 234 | 1.3676 / 1.8703 | 1.0786 / 1.1633 |
| Simultaneous / WG | 26 | 26 | 1,560 | 30 / 72 | BL 780；WG 780 | 1.4006 / 1.9617 | 1.0616 / 1.1269 |
| Visual / cognitive motivation | 16 | 54 | 7,750 | 30 / 48 | FF 1,378；FR 2,726；RF 860；RR 2,756；unknown 30 | 1.4301 / 2.0451 | 1.1799 / 1.3922 |

全量信号值均 finite，所有任务内部的通道数与通道签名稳定，且所有被试都覆盖
各任务的全部已知类别。幅值差异仍然明显：EEG global variance 从 1.0591 到
2.0451，fNIRS 从 1.0420 到 1.3922；这要求 split 后、仅以 train subjects
拟合的变换与数据集 provenance，不能据 canonical scaling 宣称物理同质。

正式多数据集训练尚未准入，新增阻断证据如下：

1. Visual 仍有 30 个 `unknown` 窗口；在恢复 source-backed 语义前必须在 split
   generation 之前拒绝。
2. Visual 的 Probe1/Probe2 形成 3,875 个共享 EEG trial/label、不同 fNIRS
   hemisphere 的成对组；7,750 个 loader samples 不是 7,750 个独立 trial。
   两个 probe 必须同 split，并采用 fusion 或显式 trial weighting。
3. REFED 虽有 480/480 个合法 valence/arousal streams（长度 60–170，median
   103），当前每个 video 只输出一个 20 秒 signal window，且 canonical label
   仍是 video category；连续 regression 的 window/sequence target 与 mask 尚未实现。
4. Visual 有 54 个 EEG 与 54 个 fNIRS 边界 padding 窗口。loader 已提供 mask，
   但正式四数据集训练 adapter 尚未证明 loss/投影会消费这些 mask。
5. REFED EEG topology 缺口已关闭：position availability 为 100%，其中 62 个
   template exact、2 个 Figure-1-backed interpolation；64-node/168-edge 邻接通过。
   这只准入 within-EEG adjacency，不准入个体坐标或跨模态物理距离声明。
6. Visual fNIRS geometry 已达到 100% position availability：每侧 14 个 CED
   anchors + 10 个 graph-harmonic interpolations，并生成 24-node/52-edge 邻接。
   这只关闭 spatial-adapter 的缺失输入，不升级为个体 digitization 或精确物理距离。
7. Single-Trial 使用 artifact-clean v3；Simultaneous 使用只针对眼动的
   `simultaneous_eeg_eog_clean_v1` 并输出 28 个 scalp channels。REFED 与 Visual
   仍是 `raw_with_ocular_artifact` 且 bad-channel mask 为默认零值，不能解释为无伪影。
8. 以任务内 log(record std) 的 robust z、`|z| > 3.5` 自适应规则标记了 28/760
   个 records；它们是复核队列，不是通用删除阈值。尤其需复查 REFED subject 30
   video 15 fNIRS 与多个 Visual Probe1 的尺度异常。

完整 Markdown/HTML、逐记录 CSV、逐标签/被试覆盖表、channel signature 与图位于
[`final_unified_loader_audit_20260718`](../../experiments/runs/physiology_semantic_tokenizer/data_quality_audit/final_unified_loader_audit_20260718/quality_report.md)。

DSR/EOG 契约更新后又对当前入口的 **22,952** 个窗口和 **7** 个 task
namespaces 完成了全量复审：DSR 为 25 subjects / 25 records / 8,980 windows，
标签 Go 2,694、No-go 6,286；Simultaneous 的 nback、wg、dsr 均稳定输出 28 EEG
channels 和 `simultaneous_eeg_eog_clean_v1`。所有任务的载入幅值有限，DSR
restoration check 为 pass；整体 readiness 仍为 7 pass / 7 block / 1 warn，未因
恢复 DSR 自动解除 Visual labels/probe dependence、REFED/Visual QC、shared split
和 adapter/mask blockers。当前证据位于
[`final_unified_loader_audit_post_dsr_20260718`](../../experiments/runs/physiology_semantic_tokenizer/data_quality_audit/final_unified_loader_audit_post_dsr_20260718/quality_report.md)。
报告记录 cache/event/alignment/geometry/adjacency/template/audit-script hashes；这些
检查证明数据与软件 contract，不证明模型性能、跨模态耦合或生理有效性。

### 6.6 REFED continuous regression adapter follow-up

上述全量审计中的 REFED target 阻断项随后由
`REFEDContinuousSequenceDataset` 关闭。原始 `*_label.mat` 实际数组为
`[time, 2]`；两列按数据集说明依次解释为 valence/arousal。480 个 stream 的点数
与视频时长秒数对应，原生频率范围为 0.999687–1.000013 Hz；loader 采用
event-relative normalized video time 处理名义 47.62 Hz 带来的微小时长误差。

版本化 schema 为 `refed_continuous_va_sequence_v1`。默认 20 秒无重叠窗口把
480 个视频事件展开为 2,720 个 regression samples，目标形状固定为 `[2, 20]`；
480 个末尾 partial windows 保留，target-valid fraction 为 0.902941。mask 按
coordinate/time 排除非 finite 标注以及缺少任一 EEG/fNIRS 支持的时点，无效值
置零。这样覆盖了全部 paired annotation support，同时没有引入一个固定删除阈值。
视频 category 仅保存在 `video_context_label` 中，不再充当回归标签。

全量 49,120 个 native annotation points 均 finite，valence/arousal 的观测范围均
为 1–255；分别有 255/251 个 unique values。480 个视频中 464 个 valence stream
与 456 个 arousal stream 非常数，逐秒 change fraction 分别为 0.0838/0.0898，
说明 joystick 序列包含较长平台段。因而不能逐窗口计算后再平均可能对常数窗口
未定义的 CCC；正式评估应先按 held-out subject/video 拼接 valid support，再报告
CCC，并同时报告 MAE/RMSE 与各坐标 coverage。
训练批必须使用 `collate_refed_continuous_sequences`：它只堆叠定长 signal、
signal mask、target 与 target mask，并把包含 nullable geometry 的 provenance
保留为逐样本列表，避免 PyTorch default collator 把元数据误当 tensor。

该结果只关闭 target construction/data-contract 缺口。正式回归仍必须先按 subject
分割，并保持同一 subject/video 的所有窗口同 split；任何 target scaling 只能由
train subjects 拟合。STA-Net/EFRM 等方法还必须证明其 regression head 同时消费
signal mask 与 target mask，才可进入正式训练。

仍需在训练入口补：

1. split manifest、probe grouping 与 protected-test lock；
2. unknown-label rejection，以及 REFED candidate regression loss 对 target mask
   的强制消费与测试；
3. channel/geometry adapter，并证明 time/channel/QC masks 被训练消费；
4. 除 Single-Trial v3 外，其余数据集的 dataset-specific
   bad-channel/window rejection masks；
5. physical teacher targets、uncertainty、valid mask，仅在科学 gate 允许后加入。

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
当前 Single-Trial 分支和 mask 决策已合并到
[`DATA_CONTRACT.md`](../DATA_CONTRACT.md)。
当前 `single_trial_eeg_artifact_clean_v3` 已完成 29 subjects / 174 task records
审计，并以 28 subjects 的 EMG、咬牙、张口受控记录完成循环移位 sham/null 验证。
registry 默认已切换到 v3；raw 与 v2 仍保留用于诊断和消融。发布页没有给出动作
持续时间，因此验证事件邻域来自 `mrk_artifact` marker 间隔的自适应估计，而不是
未经证实的固定时长。

## References

[^1]: Phukhachee, T., et al. (2024). “A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults.” _Data in Brief_, 53, 110260. https://pmc.ncbi.nlm.nih.gov/articles/PMC10964074/
[^2]: FieldTrip. “Template 3-D electrode sets: standard_1005.elc.” https://www.fieldtriptoolbox.org/template/electrode/
[^3]: REFED dataset README, “The channel distribution of the joint EEG-fNIRS acquisition.” [`data/REFED-dataset/README.md`](../../data/REFED-dataset/README.md)
[^4]: MNE-Python. “Working with sensor locations.” https://mne.tools/stable/auto_tutorials/intro/40_sensor_locations.html
[^5]: Iso, N., et al. (2021). “Hemodynamic Signal Changes During Motor Imagery Task Performance Are Associated With the Degree of Motor Task Learning,” Figure 2, standard Hitachi 4×4 24-channel layout. https://pmc.ncbi.nlm.nih.gov/articles/PMC8081959/
