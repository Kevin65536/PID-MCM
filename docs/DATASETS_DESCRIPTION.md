# 数据集说明文档

> **重要提示**：使用任何数据集前，请务必先阅读本文档以及对应数据集目录中的原始说明文件。

---

## 数据集总览

| 数据集名称 | 包含的模态 | 被试数量 | 刺激素材类型 | 任务类型 | 采样率 | 标签类型 | 说明文件 |
|-----------|-----------|---------|-------------|---------|--------|---------|---------|
| EEG+NIRS Single-Trial | EEG (30ch) + fNIRS (72ch: 36 lowWL + 36 highWL) + EOG, ECG, 呼吸 | 29 | 视觉指令 (箭头/数字) | Motor Imagery (左右手), Mental Arithmetic | EEG: 200Hz, fNIRS: 10Hz | Left/Right MI, MA/Baseline | `.html` 文档 |
| REFED-dataset | EEG (64ch) + fNIRS (51ch, 6信号类型) | 32 | 情绪视频 (15个) | 情绪诱发 | EEG: 1000Hz, fNIRS: 47.62Hz | 实时动态 Valence + Arousal | `README.md` |
| Visual Cognitive Motivation | EEG + fNIRS (Oxy/Deoxy CSV 导出, 共享位置) | 16 | 场景图片 (250个) | 视觉认知动机决策 | EEG: 高采样, fNIRS: Hitachi ETG-7100 原始导出 | RF/RR/FF/FR (记忆动机) | `readme.txt` |
| Simultaneous EEG&NIRS | EEG + fNIRS (MATLAB 导出为 oxy/deoxy) | 26 | 认知任务 | N-back, 心算等认知任务 | EEG: 高采样, fNIRS: 10Hz (MATLAB导出) | 认知负荷等级 | PDF 文档 |

---

## fNIRS 原始单位与幅值核对

下表基于原始说明文件和代表性原始文件的直接检查，用于回答两个问题：
1. 当前数据目录里保存的是 HbO/HbR，还是 highWL/lowWL / Abs 波长信号。
2. 不同数据集的绝对幅值是否处于同一量级，能否直接混合解释。

| 数据集 | 当前保存形式 | 原始单位/标注来源 | 代表性幅值（单被试全部原始 fNIRS 文件） | 稳健结论 |
|--------|-------------|------------------|----------------------------------------|---------|
| EEG+NIRS Single-Trial | 72通道，36个空间位置的 lowWL/highWL 成对保存，`wavelengths=[760,850]` | 样例 `cnt.mat` 内 `yUnit='V'`, `signal='NIRS (low wavelength, high wavelength)'` | lowWL: median=0.210, P01-P99=0.009-0.817, max=0.878; highWL: median=0.321, P01-P99=0.014-1.087, max=1.216 | 当前保存的是波长通道/光强样式，不是已转换的 HbO/HbR 浓度 |
| REFED-dataset | 同一张量内混合 6 类信号: HbO, HbR, HbT, Abs780, Abs805, Abs830 | README 明确给出 6 类信号，但未提供统一单位；原始 `.mat` 也未附单一 `yUnit` | HbO/HbR 约在 `[-5, 5]`; HbT 约在 `[-2, 3]`; Abs 信号约在 `[0.6, 4.5]` | 不能假定存在单一原始单位；缓存前必须先显式选择 signal type |
| Visual Cognitive Motivation | 以 Oxy/Deoxy CSV 分文件保存，CSV 头仍保留 695/830nm 设备信息 | `readme.txt` 说明为 oxyhemoglobin / deoxyhemoglobin，同时说明文件是 Hitachi ETG-7100 raw export；CSV 未显式写单位 | Oxy: median=0.029, P01-P99=-26.410-54.135, max=61.670; Deoxy: median=-0.034, P01-P99=-13.624-29.934, max=33.646 | 保存语义已是 Oxy/Deoxy，不是 highWL/lowWL；但单位未显式标注，且绝对幅值与 mmol/L 数据集不在同一量级 |
| Simultaneous EEG&NIRS | MATLAB 文件按 `cnt_{task}.oxy` / `cnt_{task}.deoxy` 保存 | 样例 `cnt_nback.mat` 内 `yUnit='mmol/L'`, `signal='NIRS (oxy, deoxy)'` | oxy: median≈0, P01-P99=-0.0085-0.0098, max=0.0747; deoxy: median≈0, P01-P99=-0.0045-0.0050, max=0.0423 | 当前保存已经是 HbO/HbR 浓度，且单位明确为 mmol/L |

**说明**:
- 幅值统计分别取单个代表被试的全部原始 fNIRS 文件: Single-Trial `subject 01`, REFED `subject 1`, Visual `S01`, Simultaneous `VP001`。
- 这里的幅值只用于判断量级和语义是否一致，不代表所有被试的完整总体分布。
- 单位与幅值联合判断后，可将四个数据集分成三类: `波长/optical-domain 通道`、`混合信号类型`、`已导出为 Oxy/Deoxy 浓度语义`。

## 光强可用性与波段对照

以下表格只回答统一 optical measurement space 所需的两个问题：当前仓库里是否已经有直接可用的 optical-domain 通道，以及这些通道的波段是否与 EEG+NIRS Single-Trial 的 `760/850 nm` 基准一致。

| 数据集 | 当前文件里是否有直接可用的 optical-domain 通道 | 当前可见的 optical-domain 形式 | 相对 Single-Trial `760/850 nm` 的差异 | 进入统一 optical cache 前的要求 |
|--------|--------------------------------------------|------------------------------|-------------------------------------|----------------------------------|
| EEG+NIRS Single-Trial | 是 | `lowWL/highWL`, `760/850 nm`, `V` | 基准，不存在差异 | 可直接进入统一 optical cache |
| REFED-dataset | 是 | `Abs780/Abs805/Abs830` 三路 optical-domain 通道 | 波段不同，且是三波段而不是 `760/850` 二波段 | 必须先显式选择/投影到统一的两通道 optical contract |
| Visual Cognitive Motivation | 否 | 当前文件只给 `Oxy/Deoxy`; CSV 头保留 `Wave[nm]=695,830` 设备元数据 | 仪器波段与 `760/850` 不同，但当前导出里没有直接 optical-domain 通道 | 不能直接进入统一 optical cache；若要对齐，需拿到上游 optical export 或做显式前向投影 |
| Simultaneous EEG&NIRS | 否 | 当前 MATLAB 导出只给 `oxy/deoxy`, `mmol/L` | 已检查导出字段里未暴露 optical-domain 波段，无法和 `760/850` 直接比较 | 不能直接进入统一 optical cache；若要对齐，需做显式前向投影 |

---

## 详细描述

### 1. EEG+NIRS Single-Trial (TU Berlin)

**目录**: `data/EEG+NIRS Single-Trial/`

**说明文件**: `Open access dataset for simultaneous EEG and NIRS Brain-Computer Interfaces (BCIs).html`

#### 基本信息
- **来源**: TU Berlin Machine Learning Group
- **被试**: 29人 (健康成人)
- **数据格式**: MATLAB (.mat)

#### 模态详情

| 模态 | 通道数 | 采样率 | 覆盖区域 | 数据格式 |
|------|-------|--------|---------|---------|
| EEG | 30 | 200 Hz (原1000Hz下采样) | 全脑 (10-5系统) | `.mat` |
| fNIRS | 72 (36 lowWL + 36 highWL) | 10 Hz (原12.5Hz下采样) | 前额、运动区、视觉区 | `.mat` |
| EOG | 4 | 1000 Hz | 眼电 | 包含在EEG文件 |
| ECG | 2 | 1000 Hz | 心电 | 包含在EEG文件 |
| 呼吸 | 1 | 1000 Hz | 胸带 | 包含在EEG文件 |

#### 实验范式

**Dataset A - Motor Imagery (运动想象)**
- 任务: 左手/右手握拳想象
- 试次结构: 2s指令 + 10s任务 + 15-17s休息
- 每session: 20次重复 (每类10次)
- 共3个session

**Dataset B - Mental Arithmetic (心算)**
- 任务: 连续减法 vs 休息基线
- 试次结构: 同上
- 共3个session

**Dataset C - Motion Artifacts**
- 用于运动伪迹研究

#### 标签类型
- Motor Imagery: `marker 16 = left`, `marker 32 = right`
- Mental Arithmetic: `marker 16 = MA`, `marker 32 = baseline`

#### fNIRS 通道语义
- 原始 `cnt.mat` 里的 fNIRS 通道按空间位置成对出现，总共 72 通道。
- 样例字段明确给出 `signal = NIRS (low wavelength, high wavelength)`，`yUnit = V`，`wavelengths = [760, 850]`。
- 通道名中的 `highWL` / `lowWL` 应视为波长通道标签，而不是已经转换完成的 HbO/HbR 浓度对。
- 因此在统一缓存前，这个数据集应先标记为“波长通道输入”，不能直接和 mmol/L 的浓度型数据按相同语义对齐。
- 当前 Croce source/observation cache 的统一语义应固定在 optical measurement space：Single-Trial 直接使用 `highWL` / `lowWL`；其他已经是 HbO/HbR 的数据集在进入统一缓存前，需要先显式前向投影到一对 optical channels，而不是反过来把 Single-Trial 误标成 HbO/HbR。

#### 适用场景
✅ Motor Imagery BCI  
✅ EEG-fNIRS融合研究  
✅ 多模态脑-机接口  
✅ 运动伪迹分析

---

### 2. REFED-dataset (Real-time Dynamic Labeled)

**目录**: `data/REFED-dataset/`

**说明文件**: `README.md`

#### 基本信息
- **来源**: NeurIPS 2025 Datasets Track (CC BY-NC-SA 4.0)
- **被试**: 32人 (18-34岁, 21男11女)
- **数据格式**: MATLAB (.mat)

#### 模态详情

| 模态 | 通道数 | 采样率 | 信号类型 | 数据格式 |
|------|-------|--------|---------|---------|
| EEG | 64 | 1000 Hz | 标准10-20扩展系统 | `.mat` (channel × time) |
| fNIRS | 51 | 47.62 Hz | HbO, HbR, HbT, Abs 780/805/830nm | `.mat` (signal_type × channel × time) |

#### fNIRS 原始保存语义与单位
- README 明确说明原始 fNIRS 张量同时包含 `HbO`, `HbR`, `HbT`, `Abs 780 nm`, `Abs 805 nm`, `Abs 830 nm` 六类信号。
- 由于同一张量同时混合浓度类信号和吸光度类信号，原始保存形式不存在单一统一单位；后续缓存必须显式记录所选 `signal_type`。
- 从幅值上看，HbO/HbR 大致在 `[-5, 5]`，而 Abs 三路主要落在 `[0.6, 4.5]`，也支持其语义并不相同。

#### 实验范式
- **刺激**: 15个情绪诱发视频 (60-170秒不等)
- **情绪类型**:
  - HVHA (High Valence, High Arousal): 快乐
  - HVLA (High Valence, Low Arousal): 放松
  - LVHA (Low Valence, High Arousal): 恐惧
  - LVLA (Low Valence, Low Arousal): 悲伤
  - MVMA (Medium): 中性

#### 标签类型
- **实时动态标注**: 被试使用摇杆实时标注 Valence 和 Arousal
- **SAM评分**: 每个视频后的主观情绪评分
- **PANAS**: 实验前后的正负情绪评估

#### 数据结构
```
REFED-dataset/
├── data/{subject_id}/
│   ├── EEG_baselines.mat    # 基线期EEG
│   ├── EEG_videos.mat       # 视频观看期EEG
│   ├── fNIRS_baselines.mat  # 基线期fNIRS
│   └── fNIRS_videos.mat     # 视频观看期fNIRS
└── annotations/{id}_label.mat  # 实时标注 (2 × time)
```

#### 适用场景
✅ 情感脑-机接口  
✅ 动态情绪识别  
✅ 神经血管耦合研究  
✅ 多模态情感计算

---

### 3. Visual Cognitive Motivation Study

**目录**: `data/A simultaneous EEG-fNIRS dataset of the visual cognitive motivation study in healthy adults/`

**说明文件**: `readme.txt`

#### 基本信息
- **来源**: Kyushu University (Data in Brief)
- **被试**: 16人 (健康成人)
- **数据格式**: EEG - EDF (原始) + MATLAB (预处理), fNIRS - Hitachi 原始格式

#### 模态详情

| 模态 | 设备 | 采样率 | 位置系统 | 格式 |
|------|------|--------|---------|------|
| EEG | - | 高采样 | 国际10-20系统 | `.edf` / `.mat` |
| fNIRS | Hitachi NIRS ETG-7100 | 设备原生 | 与EEG共享位置 | Oxy/Deoxy CSV 原始导出 |

#### 实验范式
- **刺激**: 250个不重复场景图片
- **试次结构**: 3s刺激呈现 + 9s决策期 (共12s/试次)
- **任务**: 决定是否想记住呈现的刺激
- **验证**: 实验后进行500张图片的再认测试

#### 标签类型
根据认知实验决策和再认测试结果组合:

| 标签 | 实验期决策 | 再认测试结果 | 含义 |
|------|-----------|-------------|------|
| RR | 想记住 | 记住了 | 高动机+成功记忆 |
| RF | 想记住 | 忘记了 | 高动机+记忆失败 |
| FR | 不想记住 | 记住了 | 低动机+意外记忆 |
| FF | 不想记住 | 忘记了 | 低动机+正常遗忘 |

#### Trigger说明
- **EEG (DC9/DC09通道)**: 1=刺激出现, 2=刺激消失, 3=被试响应
- **fNIRS (Mark)**: 1=刺激出现, 2=刺激消失, 3=被试响应

#### fNIRS 原始保存语义与单位
- 原始说明文件写明记录的是 `oxyhemoglobin` 和 `deoxyhemoglobin`，目录中的文件也按 `*_Oxy.csv` / `*_Deoxy.csv` 分开保存。
- 但同一说明文件同时指出这些 CSV 仍是 Hitachi ETG-7100 的 raw export without further processing；CSV 头部保留了 `Wave[nm]=695,830` 等设备元数据。
- 因此该数据集的当前保存语义应视为 Oxy/Deoxy 导出值，而不是 highWL/lowWL；不过原始文件未显式标出统一浓度单位，绝对幅值也明显大于 mmol/L 数据集，不能跨数据集直接按绝对数值对齐。

#### 适用场景
✅ 记忆编码研究  
✅ 认知动机与注意力  
✅ 同步EEG-fNIRS记录方法学  
✅ 事件相关范式

---

### 4. Simultaneous EEG&NIRS (Cognitive Tasks)

**目录**: `data/Simultaneous EEG&NIRS/`

**说明文件**: 
- `Dataset description_BrainVision and NIRx.pdf`
- `Dataset description_MATLAB.pdf`
- `brain_image_data_classification/README.md`

#### 基本信息
- **来源**: Nature Scientific Data (sdata20183)
- **被试**: 26人 (健康成人)
- **数据格式**: BrainVision (.vhdr, .vmrk, .dat) 或 MATLAB

#### 模态详情

| 模态 | 设备 | 采样率 | 通道数 |
|------|------|--------|-------|
| EEG | BrainVision | 高采样 | 多通道 |
| fNIRS | NIRx | 10.4 Hz 原始采集, 10 Hz MATLAB导出 | 36个空间位置的 oxy/deoxy |

#### 实验范式
- **任务类型**: N-back、心算等认知任务
- **认知负荷**: 不同难度等级

#### 数据结构
```
Simultaneous EEG&NIRS/
├── VP001-EEG/  # 被试1的EEG数据
├── VP001-NIRS/ # 被试1的fNIRS数据
├── VP002-EEG/
├── VP002-NIRS/
... (共26个被试)
└── brain_image_data_classification/  # 分类代码示例
```

#### fNIRS 原始保存语义与单位
- MATLAB 版原始文件 `cnt_{task}.mat` 直接以 `oxy` 和 `deoxy` 两个字段保存 fNIRS。
- 样例文件中 `yUnit = mmol/L`，`signal = NIRS (oxy, deoxy)`，说明当前目录下的 MATLAB 数据已经是 HbO/HbR 浓度表示。
- 其幅值集中在约 `10^-3` 到 `10^-2 mmol/L` 的量级，和 Single-Trial 的 `V` 量级、Visual 的 ETG-7100 原始导出量级明显不同。

#### 适用场景
✅ 认知负荷评估  
✅ 工作记忆研究  
✅ EEG-fNIRS同步采集方法学

---

## 数据使用注意事项

### 通用要求
1. **阅读原始文档**: 使用任何数据集前必须阅读对应目录中的说明文件
2. **引用要求**: 使用数据集请按要求引用原始论文
3. **许可证**: 注意各数据集的使用许可 (如 CC BY-NC-SA)
4. **预处理**: 了解数据是原始还是预处理过的

### 各数据集特殊注意事项

| 数据集 | 特殊注意事项 |
|--------|-------------|
| EEG+NIRS Single-Trial | 数据已下采样; 使用前需要BBCI Toolbox; fNIRS 当前保存为 `lowWL/highWL` 波长通道，不要直接按 HbO/HbR 解释 |
| REFED | 需同意非商业使用; fNIRS有6种信号类型且不共享单一单位，必须分别处理 |
| Visual Cognitive Motivation | EEG预处理数据已去除眼动伪迹epoch; fNIRS 为 Oxy/Deoxy 原始导出，绝对幅值不可直接与 mmol/L 数据集比较 |
| Simultaneous EEG&NIRS | 请参考PDF文档了解完整实验协议; MATLAB 版 fNIRS 已是 `oxy/deoxy`, 单位为 `mmol/L` |

---

## 快速参考

### 按任务类型选择

| 任务类型 | 推荐数据集 |
|---------|-----------|
| Motor Imagery BCI | EEG+NIRS Single-Trial |
| 情感识别 | REFED-dataset |
| 认知记忆 | Visual Cognitive Motivation |
| 认知负荷 | Simultaneous EEG&NIRS |

### 按采样率选择

| 需求 | 数据集 | EEG采样率 | fNIRS采样率 |
|-----|--------|----------|------------|
| 高时间分辨率EEG | REFED | 1000 Hz | 47.62 Hz |
| 标准BCI采样 | EEG+NIRS Single-Trial | 200 Hz | 10 Hz |

---

*最后更新: 2026-05-29*
