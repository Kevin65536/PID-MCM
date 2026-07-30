# Token Physiology Atlas 可视化阅读指南

本文说明如何阅读 Atlas `core` 层级生成的全部 summary figures。它是
[Token Physiology Atlas 分析契约](TOKEN_PHYSIOLOGY_ATLAS.md)的配套文档；
表格字段、缓存身份和 protected-test policy 仍以分析契约与运行 manifest
为准。

Atlas 图展示的是 **token-conditioned measurement phenotype**：在当前
checkpoint、split、预处理和支持度门槛下，被分到某个 token 的 2 秒原始
信号 patch 具有哪些描述性测量特征。它不把 token 自动命名为生理状态，
也不提供因果解释。

## 1. Core 图集包含什么

每个 `split × modality` 组合生成三张图。默认分析 `train,val` 和
`eeg,fnirs`，因此共有 12 张 PNG：

| 文件模式 | 图的作用 | 首要问题 |
| --- | --- | --- |
| `<split>_<modality>_token_support.png` | 全部 128 个 token 的分配支持度 | 哪些 token 有足够 patch 和被试支持，可以继续解释？ |
| `<split>_<modality>_phenotype_heatmap.png` | 支持度最高的 24 个 hard token 的被试等权表型 | 每个 token 相对该 split 的总体测量分布高在哪里、低在哪里？ |
| `<split>_<modality>_codebook_<feature>.png` | 静态 codebook 的二维 PCA 几何，以一个预选生理特征着色 | 学到的 embedding 几何是否呈现该测量特征的连续梯度或局部聚集？ |

每张图旁边还有：

- `.alt.txt`：图的简短无障碍描述；
- `.manifest.json`：输入 export 与 measurement cache 的路径和 SHA-256、
  split、modality、support threshold、坐标轴、软件版本、DPI、输出文件
  大小和 SHA-256；heatmap 另记录实际选择的 token ID、排序规则与色标，
  codebook 图另记录 embedding shape、投影方法、着色字段与色标；
- 顶层 `manifest.json`：整个 Atlas 的 artifact inventory 与
  `protected_test_opened`；
- `tables/*.csv`：用于复算图中数值和进行精细比较的长表。

图的推荐阅读顺序是：**support → phenotype heatmap → codebook geometry →
train/val 复现 → hard/soft 与 nuisance 诊断**。跳过 support 图直接解释颜色，
很容易把稀有 token 的波动写成生理模式。

## 2. 阅读前必须固定的四个口径

### 2.1 Token ID 是名义标签

`0…127` 只是当前 codebook 的索引。数字大小、相邻 ID 和 EEG/fNIRS 的
同号 ID 都没有生理含义。同一 checkpoint 的 train/val 图可以按相同 ID
对照；不同 seed、不同训练或重新排列的 codebook 必须先做 signature
matching，不能直接按 ID 对齐。

### 2.2 输入不是物理单位

图中的 patch 是 tokenizer 实际接收的 `canonical_robust_sd` 信号。幅值、
RMS、斜率和功率等量均不再是 µV 或血红蛋白浓度。图可以回答标准化输入中
的相对表型，不可用来恢复绝对物理幅值。

### 2.3 估计单位是被试，不是 patch

hard-token profile 先在每个 `subject × token` 内对 patch 求均值，再让有该
token 的被试等权进入总体。patch 数量仍用于 support，但 patch 多的被试
不会在 profile 均值中自动获得更大权重。

### 2.4 Core 是描述层，不含区间推断

`core` 把 bootstrap 和 coupling null 的迭代数设为 0，只生成 PNG。热图颜色
不是显著性、置信度或 posterior probability。需要被试 bootstrap CI、
information ledger 或时延 null 时，运行 `statistical` 或 `full` 层级，并
结合对应表格解释。

## 3. Token support 柱状图

### 3.1 图形编码

- 横轴：全部 codebook token ID；ID 是分类标签，不是连续量。
- 纵轴：通过 token mask 的 assigned patch count。
- 蓝色实心柱：同时通过 `min_count` 和 `min_subjects`。
- 灰色斜线柱：至少有一个支持度门未通过。
- 水平虚线：只画出 `min_count`，默认是 30 patches。
- 缺失 support 值：绘图 API 使用深灰交叉标记；标准 Atlas 的完整
  `token_support.csv` 通常不会缺失。
- 零高度：该 split 中 inactive 的 token，不等于“测得了零生理效应”。

默认充分支持需要：

```text
count >= 30 AND subject_count >= 5
```

因此，某根柱即使超过虚线，仍可能因为覆盖不足 5 个被试而带斜线。最终状态
以颜色/斜线和 `tables/token_support.csv` 的 `support_status` 为准，不能只
看虚线。

### 3.2 可以读出什么

- active codebook coverage：有多少 token 真正在该 split 被使用；
- usage concentration：是否只有少数 token 占据大部分 patch；
- train 到 val 的支持度收缩：哪些 token 在独立被试中仍有足够覆盖；
- subject domination：图本身不画出，由 `subject_count`、
  `effective_subjects`、`max_subject_fraction` 和
  `normalized_subject_entropy` 补充判断。

### 3.3 不能怎样比较

train 与 val 的总 patch 数不同，柱高不可直接当作使用率比较。需要比较
occupancy 时，应以各 split 的 valid patch 总数归一化；各图的纵轴也独立
缩放，不能只比较像素高度。低柱不等于 token “无意义”：它只表示当前数据
不足以支撑稳定的生理叙述。

## 4. Hard-token phenotype heatmap

### 4.1 行、列和排序

- 每行是一个 token；
- 默认只展示 24 行；
- 排序先把 support 充分的 token 放前面，再按 assigned count 从高到低；
  count 相同时较小 ID 在前；
- 每列是一个明确命名的原始 patch 特征；
- 图使用 hard assignment；soft profile 不叠加在图上。

如果某个组合少于 24 个充分支持 token，剩余行仍会保留，但整行显示灰色
斜线。支持充分但某个特征无法计算时，该单元格显示 `×`，不会用 0 填充。

### 4.2 颜色究竟表示什么

每个单元格是：

\[
E_{t,f} =
\frac{\mu^{\mathrm{subject\text{-}equal}}_{t,f}
      - \mu^{\mathrm{subject\text{-}equal}}_{f}}
     {s^{\mathrm{subject\text{-}equal}}_{f}}
\]

其中：

- \(\mu_{t,f}\) 是 token \(t\) 的被试内均值再被试等权；
- \(\mu_f\) 是每个被试在全部有效 patch 上的均值再被试等权；
- \(s_f\) 是围绕该 marginal mean 的 patch 离差，经被试等权汇总后的
  RMS scale。

因此：

- 红色：该 token 的特征均值高于当前 split 的 marginal mean；
- 蓝色：低于 marginal mean；
- 白色附近：接近 marginal mean；
- 色条单位是 “marginal subject-equal scale”，不是原始单位、p 值、
  posterior probability 或标准误。

每张热图都使用以 0 为中心的对称色标，但绝对上限按该图中支持充分的可见
数值单独计算。比较 train/val 或 EEG/fNIRS 时，必须先读各自 colorbar；
不能仅凭“颜色更深”判断效应更大。精确比较使用
`tables/token_profiles.csv` 的 `marginal_standardized_effect`。

### 4.3 EEG 的 42 个特征

EEG 每个基础特征出现两次：

- `channel_mean/...`：该 patch 所选有效 EEG channels 的平均特征；
- `channel_sd/...`：这些 channels 之间的离散程度。

所选 EEG channel 可以随样本变化，因此 `local_eeg_channel_0` 不是一个固定
头皮位置。需要 channel-specific 解释时，应查看
`tables/token_channel_feature_distributions.csv` 中保存的真实逐样本
channel identity。

基础特征包括：

- 时域：`mean`、`std`、`rms`、`slope`、`endpoint_delta`、
  `line_length`；
- Hjorth：`activity`、`mobility`、`complexity`；
- log absolute power：delta、theta、alpha、beta、low-gamma；
- log relative power：相对 `[1,45) Hz` 参考带的同五个频带；
- `spectral_entropy` 与 `peak_frequency`。

默认频带是 delta `[1,4)`、theta `[4,8)`、alpha `[8,13)`、beta
`[13,30)`、low-gamma `[30,45)` Hz。power 使用去均值、symmetric Hann
单边 periodogram；`log` 是自然对数。`log_relative_power_alpha` 的正向
enrichment 表示 alpha 在参考带中的**对数占比相对更高**，不表示绝对
alpha 功率或放松状态必然更高。

具体而言，`channel_mean/log_relative_power_alpha` 是先在每个 channel
计算 `ln(P[8,13) / P[1,45))`，再对所选有效 channels 求算术平均；它不是
先合并 channels 后计算的线性 alpha 占比。

`std`、`rms`、Hjorth activity 与 absolute power，以及相邻频带的相对功率
并非独立证据。阅读时优先识别成组模式，不应把一组高度相关的红色格子当成
多次独立验证。

### 4.4 fNIRS 的 17 个特征

HbO 和 HbR 各保留八个 2 秒局部形态特征：

- `mean`、`median`、`std`、`rms`；
- `slope`、`endpoint_delta`；
- `auc`；
- `derivative_spike`。

最后一列 `HbO_HbR/within_patch_correlation` 是同一 patch 内 HbO 与 HbR
的相关。热图展示的仍是该相关量的标准化 enrichment，因此颜色范围不必落在
`[-1,1]`；原始相关分布应从 distribution table 读取。它不是 EEG–fNIRS
相关、跨被试相关或长时窗神经血管 coupling。

2 秒 fNIRS patch 不提供 band power、慢波、HRF 峰值时间或完整响应形状。
`mean`、`median`、`auc` 以及 `slope`、`endpoint_delta` 通常高度相关，
同样应按形态簇解释，不可视为独立重复证据。

## 5. Codebook PCA + physiological colour 图

### 5.1 几何与颜色

- 每个点代表一个 codebook token；
- 对大于二维的静态 codebook 向量先中心化，再用确定性的未缩放 PCA/SVD
  投到二维；
- 坐标轴括号给出 PC1、PC2 各自解释的 codebook variance；
- EEG 默认颜色是
  `channel_mean/log_relative_power_alpha` 的 hard-profile enrichment；
- fNIRS 默认颜色是 `HbO/slope` 的 hard-profile enrichment；
- 上三角表示高于 0，下三角表示低于 0，圆点表示数值近似等于 0；
- 灰色大 `X` 表示 feature 有数值但 token support 不充分；
- 灰色小 `x` 表示 feature estimate 缺失；
- embedding 坐标缺失的 token 不放在虚构位置，图内会报告 omitted 数量。

颜色与三角方向是冗余编码，使方向不只依赖红蓝色。色标同样以 0 为中心，
但 train/val 的范围单独计算。

### 5.2 几何可以支持的描述

可以描述：

- 相邻 codebook 区域是否出现平滑的颜色梯度；
- 某个表型是否集中在局部几何分支；
- 高支持 token 是否只占据 codebook 的一部分；
- 同一 checkpoint 的 train/val 着色模式是否在相同几何上复现。

不能据此断言：

- PCA 图上相邻就等于完整 64 维 codebook 中最近；
- PC1/PC2 是某个生理轴；
- 红蓝分区就是离散生理状态；
- 不同 checkpoint 的 PCA 方向或正负号可以直接比较；
- EEG 与 fNIRS 的相似图形意味着同号 token 同义。

二维图只保留坐标轴标出的方差比例，其余维度被省略。默认图不标 128 个
token ID，以避免文字遮挡；它是 topology-level 诊断。具体 token 的精确
数值使用 heatmap 和 `token_profiles.csv`；需要带 ID 的探索图时，可调用
`plot_codebook_embedding_colored(..., annotate_token_ids=True)`。

## 6. Train/validation 的正确对照流程

同一 checkpoint 下，建议按以下顺序：

1. 在 support 图中找出 train 和 val 都充分支持的 token ID；
2. 对这些共同支持 token 比较 heatmap 中特征方向和成组模式；
3. 从各图 colorbar 或 `token_profiles.csv` 比较数值，避免按色深比较；
4. 查看 `hard_soft_profile_differences.csv`，判断模式是否依赖硬边界；
5. 查看 `posterior_normalized_entropy_subject_equal_mean` 与
   `posterior_margin_subject_equal_mean`，判断 assignment 是否清晰；
6. 检查 metadata/nuisance association，排除 subject、task、dataset、
   label 或 token position 主导；
7. 对候选结论运行 subject-bootstrap；若是跨模态时延结论，再运行
   within-window circular-shift null。

validation 支持度通常比 train 少，因为被试和 patch 数更少。一个 train
模式在 val 中没有显示，可能只是没通过 support gate，不应自动记为方向
相反或复现失败。

## 7. 图与数值表的对应关系

| 想核对的问题 | 首选文件 |
| --- | --- |
| 柱高、被试覆盖、inactive/rare 状态 | `tables/token_support.csv` |
| 热图 hard-profile 数值与边际 scale | `tables/token_profiles.csv` |
| soft profile 与 hard profile 的边界敏感性 | `tables/hard_soft_profile_differences.csv` |
| 原始 patch 分布与 5/25/50/75/95% 分位数 | `tables/token_feature_distributions.csv` |
| 带真实 channel identity 的分布 | `tables/token_channel_feature_distributions.csv` |
| posterior、重建与 latent/code 距离诊断 | `tables/assignment_diagnostics.csv` |
| state 双向条件概率、lift、NMI | `tables/state_associations.csv` |
| nuisance/context 条件关联 | `tables/metadata_associations.csv` |
| train/val phenotype matching | `stability.json` |
| transition、dwell、Markov 泛化 | `sequence_summary.json` 与 `arrays/*_sequence_counts.npz` |
| 每张图的输入与输出 hash | 同名 `.manifest.json` |

## 8. 可写入报告的最小表述

推荐：

> 在当前 checkpoint 的 development split 中，token 17 在至少 30 个
> valid patches 和 5 个被试的支持度门下，呈现较高的被试等权 alpha
> 相对功率 enrichment；该方向在 validation 的共同支持 token 中复现。

不推荐：

> Token 17 是 alpha/放松状态。

前一种表述明确了 checkpoint、split、支持门、测量、估计方式和复现范围；
后一种表述把描述性条件表型越级成了状态身份。
