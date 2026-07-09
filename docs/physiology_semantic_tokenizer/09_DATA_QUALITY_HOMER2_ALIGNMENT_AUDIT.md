# 数据规范化、HOMER2 对齐与统一缓存规范

_Created: 2026-07-08_
_Unified: 2026-07-09_

## 当前结论

`data/cache/physiology_semantic_clean_v1` 已从“fNIRS 清洗缓存 +
事件 sidecar”推进到统一规范缓存的 P1 阶段：

- fNIRS signal records、event rows、alignment reports 已共享同一套
  canonical join key。
- 四个原始数据集中的通道/电极位置资料已经审计，并生成统一
  `channel_geometry` sidecar。
- 已实现 `CleanPhysiologyCacheIndex` 与 fNIRS-first
  `CleanPhysiologyAlignedWindowDataset`。
- 当前缓存仍不是完整 EEG-fNIRS multimodal training cache，因为 EEG signal
  arrays 尚未进入 `physiology_semantic_clean_v1`。

因此当前支持：

- fNIRS-only loading；
- raw-native vs HOMER2-aligned fNIRS 对照；
- 四数据集统一事件/标签审计；
- geometry-aware loader 和后续空间邻接扩展；
- teacher-free fNIRS reconstruction/VQ smoke。

当前不支持：

- 把四个数据集作为等价完整 EEG-fNIRS 样本直接训练；
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

- Single-Trial 本地 EEG loader 与诊断读取 `with occular artifact` 文件；
  2026-07-08 S19 artifact inspection 显示 11/30 EEG channels 有 EOG 相关污染。
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
- events: 30270
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

- `CleanPhysiologyCacheIndex`
- `CleanPhysiologyAlignedWindowDataset`

返回样本包含：

- `fnirs`: channel-first window array；
- `eeg`: 当前为 `None`；
- `modality_available`: `{"fnirs": true, "eeg": false}`；
- label 与 event metadata；
- `dataset_id`, `subject`, `canonical_subject_id`, `record_id`,
  `base_record_id`, `signal_branch`, `join_key`；
- sample rate 与 sample slice。

当前 2 秒窗口 smoke：

- `branch_preference="hbo_hbr"` 生成 30270 个窗口；
- 首个样本为 `eeg_fnirs_single_trial|subject_01|session_00`，
  shape `(72, 20)`。

这说明统一 fNIRS-first loader 已可工作，但完整 multimodal loader 仍需补：

1. EEG signal cache 或原始 EEG bridge；
2. split manifest 与 protected-test lock；
3. bad-channel/window rejection masks；
4. Visual/REFED EEG alignment 的进一步解析；
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

`data/` 仍是 gitignored 本地 artifact。代码、测试和文档进入 git；缓存本体不进入 git。

## 8. 当前训练判定

支持：

- P1 数据合同、事件合同、geometry sidecar、fNIRS-first loader 已落地；
- raw-native 与 HOMER2-aligned 分支可对照；
- 四数据集 event/signal join coverage 已清零缺口。

不支持：

- 把当前缓存称为完整 EEG-fNIRS multimodal training cache；
- 把 post-conversion 数据称为完整 HOMER2-clean；
- 在 EEG signal、artifact masks、split lock、teacher-valid masks 未补齐前启动
  physical-teacher-supervised training。
