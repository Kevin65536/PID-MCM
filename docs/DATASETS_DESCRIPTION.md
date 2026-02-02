# 数据集说明文档

> **重要提示**：使用任何数据集前，请务必先阅读本文档以及对应数据集目录中的原始说明文件。

---

## 数据集总览

| 数据集名称 | 包含的模态 | 被试数量 | 刺激素材类型 | 任务类型 | 采样率 | 标签类型 | 说明文件 |
|-----------|-----------|---------|-------------|---------|--------|---------|---------|
| EEG+NIRS Single-Trial | EEG (30ch) + fNIRS (36ch) + EOG, ECG, 呼吸 | 29 | 视觉指令 (箭头/数字) | Motor Imagery (左右手), Mental Arithmetic | EEG: 200Hz, fNIRS: 10Hz | Left/Right MI, MA/Baseline | `.html` 文档 |
| REFED-dataset | EEG (64ch) + fNIRS (51ch, 6信号类型) | 32 | 情绪视频 (15个) | 情绪诱发 | EEG: 1000Hz, fNIRS: 47.62Hz | 实时动态 Valence + Arousal | `README.md` |
| Visual Cognitive Motivation | EEG + fNIRS (共享位置, 10-20系统) | 16 | 场景图片 (250个) | 视觉认知动机决策 | EEG: 高采样, fNIRS: Hitachi ETG-7100 | RF/RR/FF/FR (记忆动机) | `readme.txt` |
| Simultaneous EEG&NIRS | EEG + fNIRS | 26 | 认知任务 | N-back, 心算等认知任务 | EEG: 高采样, fNIRS: ~7.81Hz | 认知负荷等级 | PDF 文档 |

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
| fNIRS | 36 | 10 Hz (原12.5Hz下采样) | 前额、运动区、视觉区 | `.mat` |
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
| fNIRS | Hitachi NIRS ETG-7100 | 设备原生 | 与EEG共享位置 | 设备原始格式 |

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
| fNIRS | NIRx | ~7.81 Hz | 多通道 |

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
| EEG+NIRS Single-Trial | 数据已下采样; 使用前需要BBCI Toolbox |
| REFED | 需同意非商业使用; fNIRS有6种信号类型需分别处理 |
| Visual Cognitive Motivation | EEG预处理数据已去除眼动伪迹epoch |
| Simultaneous EEG&NIRS | 请参考PDF文档了解完整实验协议 |

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

*最后更新: 2026-02-02*
