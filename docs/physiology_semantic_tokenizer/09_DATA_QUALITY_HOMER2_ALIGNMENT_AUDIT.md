# 数据质量与 HOMER2 对齐审计

_Created: 2026-07-08_

## 结论

当前证据支持三点：

1. fNIRS 单位/物理语义对齐已经在仓库中以 measurement contract 的形式落实，但落实的是“保留原始测量语义并映射到可比较的无量纲坐标”，不是把所有数据宣称为同一 HbO/HbR 物理单位。
2. 当前处理流程与经典 HOMER2 流程仍有实质差异，尤其是 raw intensity 到 optical density、运动伪影检测/校正、Beer-Lambert 浓度转换、短程/全局生理回归和标准 HRF 估计链条。现有差异足以影响最终数据形态，因此不能再把当前 cache 当作 HOMER2 等价数据使用。
3. 两个 TU Berlin 数据集的发布数据不能视为已完成伪影清理。Single-Trial 任务数据明确使用 `with occular artifact` EEG 文件，Simultaneous EEG&NIRS MATLAB 说明明确除下采样和格式转换外未做其他信号处理。此前幅度/波动检查和 TRTD 诊断显示可见影响仍存在。

因此下一步方向是：先构建 HOMER2-aligned 的统一数据缓存和对照审计，再重新判断 physical teacher 是否仍然失败。E0 teacher-supervised tokenizer training 应继续阻塞。

## 1. fNIRS 单位与物理语义

仓库当前合同是正确方向：

- `eeg_fnirs_single_trial`: `lowWL_760nm` / `highWL_850nm`, native unit `V`, measurement family `optical_intensity`。
- `simultaneous_eeg_nirs`: `Oxy` / `Deoxy`, native unit `mmol/L`, measurement family `chromophore_concentration`。
- `refed`: HbO/HbR 与 Abs780/805/830 是不同 signal family，不能混成一个单位。
- `visual_cognitive_motivation`: Oxy/Deoxy export 未声明统一单位。

`src/data/fnirs_standardization.py` 采用 full-record robust linear detrend + robust scale，并记录 contract、baseline、slope、scale 和 repaired non-finite count。`src/data/registry.py` 默认注入 `measurement_standardization`；Single-Trial 和 Simultaneous loaders 在启用 record-level standardization 后跳过旧的 session/window fNIRS z-score。

这说明“单位对齐”在数值入口层已落实，但它只是数值/元数据合同，不是 HOMER2 生理处理链，也不证明 physical teacher target 有效。

## 2. 与 HOMER2 标准流程的差异

HOMER2/Homer 常见处理链包括：

1. raw intensity quality/channel pruning；
2. intensity -> optical density；
3. motion artifact detection and correction，例如 spline、wavelet、TDDR、PCA 等；
4. band-pass filtering；
5. optical density -> HbO/HbR concentration via modified Beer-Lambert law；
6. stimulus rejection、block average 或 GLM/HRF 估计；
7. 可选短程通道/全局生理回归。

当前仓库 active path 的差异：

| 环节 | 当前仓库状态 | 影响 |
| --- | --- | --- |
| 原始单位保留 | 已落实 measurement contract | 防止错把 V、Abs、mmol/L 混成一个单位 |
| full-record baseline/drift/scale | 已落实 | 减少 crop-dependent normalization，但不等价于生理预处理 |
| intensity -> OD | 只在 `evaluate_lin2024_raw_session_trtd.py` 诊断里近似做过，不是统一 loader/cache 合同 | Single-Trial optical input 仍可能不是 HOMER2 HbO/HbR |
| motion artifact correction | active loader/cache 未见 HOMER2 等价步骤 | 运动/眼动伪影会进入 teacher 与 tokenizer 输入 |
| MBLL HbO/HbR | Simultaneous MATLAB 已提供 oxy/deoxy；Single-Trial active contract 仍保留 optical domain | 跨数据集无法直接比较 HbO/HbR 物理量 |
| short-distance regression | 两个 TU Berlin 数据集当前可见配置没有短程通道 | 无法清除 superficial/systemic physiology |
| HRF/GLM | 只在诊断脚本中做 active-channel/TRTD 检查 | 未成为统一缓存的可复现处理层 |

所以当前差异不是小的实现细节。它们会改变信号的幅度、低频漂移、运动尖峰、HbO/HbR 相位和任务锁定形态，足以解释一部分 physical teacher 与 fNIRS observation 不匹配。

## 3. 原始数据集伪影状态

### EEG+NIRS Single-Trial

原始 HTML 说明任务数据与 Dataset C motion artifact 数据分开；Dataset C 是专门诱发的 EOG、EMG、眨眼、咬牙、张嘴等伪影记录。NIRS 任务数据被描述为 continuous NIRS light intensity data。EEG loader 与 Lin-style 诊断均读取 `EEG_01-29/subject XX/with occular artifact/cnt.mat`。

本地 2026-07-08 S19 artifact inspection 显示：

- EOG 相关污染：11/30 EEG channels，36.7%，severity `moderate`；
- muscle artifact 指标：1/30 channels，3.3%，severity `mild`；
- 报告结论：无 artifact-free version，dataset authors 未在该数据副本中移除伪影。

因此 Single-Trial 不能被当作已去伪影数据。

### Simultaneous EEG&NIRS

MATLAB 官方说明写明发布的 MATLAB 数据 downsampled to 200 Hz EEG / 10 Hz NIRS，并且除 EEG/NIRS MATLAB-compatible format conversion 外未做其他 signal processing。BrainVision/NIRx 说明显示 vendor raw NIRS 是 `wl1=760 nm`、`wl2=850 nm`，并提示替换 BBCI 中 `proc_BeerLambert.m` 才能正确执行 MBLL。当前本地 MATLAB 文件已经是 `oxy/deoxy`，但并不等于已完成运动伪影校正或 HOMER2 质量控制。

2026-07-08 VP001 WG raw TRTD diagnostic 中，TRTD leave-one-trial R2 为 `-0.615709`，self-persistence R2 为 `0.991695`。这说明 fNIRS 自身低频连续性极强，而 EEG-derived TRTD component 对 held-out trial fNIRS 的解释有限。

## 4. 既有幅度/波动诊断的含义

Single-Trial S19 session 2：

- TRTD in-sample upper bound R2 最高 `0.021923`；
- leave-one-trial R2 约 `-1.89`；
- self-persistence R2 `0.997342`；
- optimized TRTD amplitude ratio 约 `0.15`。

Simultaneous VP001 WG：

- TRTD in-sample R2 `0.003606`；
- leave-one-trial R2 `-0.615709`；
- self-persistence R2 `0.991695`；
- amplitude ratio 约 `0.06`。

这支持两个判断：

- 当前 physical teacher / TRTD 类 shared neural drive 不能解释主要 fNIRS 波动，至少在当前处理链下不支持进入实际 tokenizer supervision。
- 伪影和非 HOMER2 处理可能是重要混杂，但不是唯一解释；即便 in-sample upper bound 也很低，说明 teacher family 或目标粒度仍需重新审查。

## 5. 下一步执行方向

### A. 建立 HOMER2-aligned preprocessing adapter

优先实现两个路径，而不是直接替换现有 contract：

1. `raw_native_contract`: 保留当前 full-record standardization，作为可回溯 baseline。
2. `homer2_aligned_contract`: 尽可能复现 HOMER2 处理链，并记录每一步参数和不可实现项。

对 Single-Trial：

- 使用 `lowWL/highWL` 生成 OD；
- 做 channel quality/finite/SNR 审计；
- 加入 motion detection + 至少一种 correction，建议 TDDR 或 wavelet/spline 作为主路径，另保留 no-correction 对照；
- 用明确 extinction coefficients、DPF/PPF 和 source-detector distance 做 MBLL；
- 输出 HbO/HbR canonical cache，同时保留 OD cache。

对 Simultaneous：

- 若只使用 MATLAB `oxy/deoxy`，明确标记为 post-conversion chromophore input；
- 若需要 optical-domain 对齐，必须回到 vendor NIRx raw `wl1/wl2` 或做显式 forward projection，不能把 oxy/deoxy 当作 760/850 raw intensity；
- 加入同样的 motion/quality audit，并把 correction 前后差异保存。

### B. 制作统一干净缓存

建议新增 cache schema：

```text
clean_eeg_fnirs_cache_v1/
  dataset_id/
    subject_id/
      record_id.npz
      record_manifest.json
  cache_manifest.json
```

每个 `record_manifest.json` 至少包含：

- original files and hashes；
- native measurement family/unit；
- preprocessing branch: `raw_native_contract` or `homer2_aligned_contract`；
- EEG artifact handling: EOG regression/ICA/BSS-CCA/TDDR-equivalent status；
- fNIRS steps: OD conversion, motion detection/correction, MBLL, bandpass, baseline, optional short-channel regression；
- rejected channels/windows/trials；
- retained channel labels and coordinates；
- transform parameters needed for reproducibility；
- quality summary before/after preprocessing；
- split/protected-test boundary。

### C. 先做可见影响审计，再重开 teacher 判断

每个数据集先运行同一组对照：

1. raw-native vs HOMER2-aligned waveform overlay；
2. per-channel amplitude/drift/motion spike change；
3. task-locked HbO/HbR average and active-channel map；
4. EEG artifact removal before/after PSD and EOG correlation；
5. same TRTD/shared-state diagnostic on both branches；
6. teacher physical observation check on both branches。

只有当 HOMER2-aligned cache 显著改善 fNIRS observation fit、posterior calibration 或 shared-state diagnostic，才重新设计 E0-v3。否则应把 physical teacher 降级为弱 auxiliary/context target，并把 tokenizer 主训练转向 reconstruction + residual/private branch + downstream/frozen-probe validation。

## 6. 已落地的代码路径

本轮已新增两个执行入口：

- `src/data/homer2_preprocessing.py`: HOMER2-aligned contract、四数据集适配度 manifest、OD 转换、TDDR-like derivative suppression、bandpass、显式 MBLL 近似转换。
- `experiments/build_clean_eeg_fnirs_cache.py`: 统一 clean cache 构建入口，同时保存 `raw_native_fnirs` 与 `homer2_aligned_fnirs`，并为每条记录写入 manifest。

注意：`homer2_aligned_contract` 是 alignment/audit contract，不是完整 HOMER2 复刻。只有 `eeg_fnirs_single_trial` 具备 raw intensity，可以进入 `intensity -> OD -> motion/filter -> MBLL`。其他三个数据集当前只能作为 post-conversion trace 做运动/滤波清理，并在 manifest 中标记缺失 raw intensity / pre-conversion OD / MBLL replay。

### 6.1 raw native contract

`raw_native_contract` 继续使用 `src/data/fnirs_standardization.py`：

- 保留 dataset-specific native measurement family/unit；
- full-record non-finite interpolation；
- robust linear drift/baseline；
- robust channel scaling；
- 输出无量纲 native-semantics deviation；
- 用作所有后续实验的可回溯 baseline。

它回答的问题是：在不虚构物理单位的情况下，当前数据能否被稳定放入统一数值坐标。

### 6.2 HOMER2-aligned contract

`homer2_aligned_contract` 尽可能贴近 HOMER2 处理顺序：

1. raw intensity 可用时执行 intensity -> optical density；
2. 执行 derivative-based motion suppression；
3. 执行 0.01-0.2 Hz fNIRS bandpass；
4. raw intensity + wavelength axis 可用时执行 MBLL 输出 HbO/HbR；
5. 所有不可执行步骤进入 `missing_inputs` / `skipped_steps`。

它回答的问题是：如果把可用数据推到更接近 HOMER2 的处理坐标，teacher 失败是否明显缓解；以及哪些数据因为原始文件缺失，无法完成标准链条。

### 6.3 四数据集当前适配度

| 数据集 | 当前入口 | 可执行 HOMER2-aligned 步骤 | 缺失导致无法完整标准化的部分 |
| --- | --- | --- | --- |
| `eeg_fnirs_single_trial` | raw 760/850 intensity, `V` | OD、motion suppression、bandpass、MBLL | short-channel regression、subject-specific DPF、精确 HOMER2 channel pruning policy |
| `simultaneous_eeg_nirs` | MATLAB oxy/deoxy, `mmol/L` | post-conversion motion suppression、bandpass | raw `wl1/wl2` intensity、raw OD replay、MBLL replay、short channels |
| `refed` | HbO/HbR/HbT + Abs780/805/830 export | post-conversion motion suppression、bandpass、reservation metadata 可用于后续 channel mask | raw light intensity、声明物理单位、short channels；absorbance export 不能恢复 raw intensity |
| `visual_cognitive_motivation` | ETG-7100 Oxy/Deoxy CSV export | post-conversion motion suppression、bandpass | raw 695/830 intensity、source-detector geometry、声明物理单位、short channels |

结论：四个数据集中只有 Single-Trial 具备接近完整 HOMER2 fNIRS 处理链的必要 raw intensity 输入；其他三个都存在原始光强缺失，不能按照标准完整 HOMER2 流程重跑，只能做 post-conversion 对齐和 provenance 标记。

### 6.4 已验证 smoke 命令

```bash
.venv/bin/python experiments/build_clean_eeg_fnirs_cache.py \
  --subjects-per-dataset 1 \
  --records-per-subject 1 \
  --max-samples 2000 \
  --output-dir experiments/runs/physiology_semantic_tokenizer/data_quality_homer2_clean_cache_smoke \
  --overwrite
```

该 smoke 已生成 4 条真实记录：

- `eeg_fnirs_single_trial`: applied `intensity_to_optical_density`, `robust_derivative_motion_suppression`, `bandpass`, `modified_beer_lambert`。
- `refed`: applied `robust_derivative_motion_suppression`, `bandpass`; skipped OD/MBLL。
- `visual_cognitive_motivation`: applied `robust_derivative_motion_suppression`, `bandpass`; skipped OD/MBLL。
- `simultaneous_eeg_nirs`: applied `robust_derivative_motion_suppression`, `bandpass`; skipped OD/MBLL。

全量缓存可使用同一脚本去掉 `--max-samples`，并把 `--output-dir` 指向 `data/cache/physiology_semantic_clean_v1`。REFED 如需同时保存 absorbance export，可加 `--include-refed-absorbance`。

## 当前判定

支持：

- fNIRS native unit/semantic metadata 已经进入代码和文档；
- 当前数据处理与 HOMER2 存在可见且重要差异；
- 原始发布数据不能视为已完成伪影清理；
- 现有 poor teacher evidence 不能简单归因于模型训练失败。

不支持：

- 不支持把当前 cache 称为 HOMER2-clean；
- 不支持在当前数据质量疑虑未扫清前启动 physical-teacher-supervised tokenizer training；
- 不支持用 lower loss、self-persistence 或全局 shared-state coupling 结果替代数据质量和 teacher endpoint 验证。
