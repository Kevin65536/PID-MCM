# 对比方法性能劣化的机制分析实验设计

_版本：v1.0（2026-08-16）；状态：已冻结；P0 首轮已完成，结果见
[`PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md`](PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md)_

## 1. 目的与结论先行

本协议不以“把对比方法调到更高分”为目标，而是回答一个可被审稿人检验的问题：**为什么多个已发表方法在本项目的统一、严格跨被试评估下显著低于其原论文结果，这一差距来自评估对象变化、输入与训练容量变化、预训练迁移失效、跨模态机制失效，还是当前任务本身缺少稳定的跨被试可判别信号？**

现有证据不支持用单一理由解释全部 gap。更可信的总体假设是四类因素叠加：

1. **estimand 改变**：部分原论文评价的是被试内识别或较弱的被试隔离，而本项目评价新被试泛化；
2. **统一公平协议压缩了方法原生容量**：16 个共同 EEG 通道、统一时间窗和冻结线性探针，与部分论文的全量通道、长 fNIRS 上下文、非线性头或端到端微调不同；
3. **预训练或跨模态机制没有迁移成任务相关信息**：模型可能编码数据集/被试身份，或仅学到可检测但低秩、不可检索的跨模态对齐；
4. **任务侧存在共同瓶颈**：Visual、n-back 和部分 MI 上几乎所有方法同时接近基线，说明问题不能只归于某一个 adapter。

因此，论文中最有说服力的证据不应是又一轮无边界调参，而应是一条完整的“差距归因链”：

> 原始任务锚点可复现性 → 协议桥接 → 输入/适配容量阶梯 → 表征与对齐机制诊断 → 因果扰动与负对照 → 跨被试逐被试统计。

## 2. 当前证据基线

### 2.1 已完成结果中最值得解释的模式

下表由已封存的 2026-08-14 联合 protected campaign 汇总而来。范围是同时支持六个分类任务的 BIOT、CBraMod、EFRM、NormWear 和 REVE；数值仅作**描述性诊断**，不新增任何 protected 结论。

| 任务 | 五方法 macro-F1 均值 | 方法间 SD | 相对 1/K 的直观位置 | 初步含义 |
| --- | ---: | ---: | --- | --- |
| DSR | 0.5439 | 0.0916 | 总体高于二分类机会水平，但方法差异大 | 方法表征与适配仍有作用 |
| Mental arithmetic | 0.6037 | 0.0234 | 所有方法稳定高于基线 | 任务存在可迁移信号 |
| Motor imagery | 0.5194 | 0.0363 | 仅略高于二分类机会水平 | 跨被试运动表征迁移弱 |
| n-back | 0.3860 | 0.0212 | 略高于三分类机会水平 | 共性任务瓶颈明显 |
| Visual | 0.2246 | 0.0246 | 接近或低于四分类机会水平 | 首要排查标签、窗口、类条件与跨被试信号 |
| Word generation | 0.5919 | 0.0327 | 稳定高于基线 | 统一协议并非普遍使模型失效 |

此处 1/K 仅用于帮助阅读任务难度，不替代已冻结的正式 `B0`；正式接纳继续使用同一 support 上“多数类基线与按训练先验随机预测的期望 macro-F1 中较强者”。

在完整的 5 方法 × 6 任务平衡面板上，对原始 macro-F1 做无推断含义的双因素平方和分解，任务、方法和残差分别约占 **91.57%、4.51% 和 3.91%**。改用按机会水平归一化的增益后，三者约为 **51.78%、25.82% 和 22.40%**。这只用于确定分析优先级，不能写成因果解释或显著性检验。

另外，42 个注册单元中有 22 个 `TABLE_READY_WITH_NOTE`、12 个 `REJECTED_VALUE`、2 个 overlap-only 和 6 个 unsupported；540/540 个作业完整完成，无技术失败。因此，当前主要问题是**有效低结果的科学解释**，而不是运行不完整。

主要本地证据：

- `docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md`
- `docs/comparisons/PROTOCOL.md`
- `docs/comparisons/METRIC_ACCEPTANCE.md`
- `comparative_methods/ADAPTER_ALIGNMENT_GATES_V2.md`
- `comparative_methods/comparison_metric_targets_v1.yaml`

### 2.2 原论文评价协议不能统一表述为“不跨被试”

| 方法 | 原论文/代码中与 gap 解释最相关的事实 | 对论文叙述的约束 |
| --- | --- | --- |
| BIOT | 核心优势是跨数据预训练、频谱 token、通道/位置编码和变长/缺失通道适应；下游通常允许任务适配或微调 | 当前冻结线性探针只检验静态表征，不等价于完整论文训练方案 |
| CBraMod | 原文多个下游任务明确按被试划分；且论文消融显示固定预训练参数相对全量微调可出现很大下降 | 不能用“原文不跨被试”解释其全部 gap；必须直接测试微调容量 |
| REVE | 包含被试隔离下游评估，强调 4D 位置编码、规模化预训练及冻结表征；项目 MI/MA 又存在 target-corpus overlap | 需要同时诊断空间坐标、冻结 query、通道数和 overlap，不可混成单一结果 |
| NormWear | 若有被试 ID，原文使用被试分层；原预训练模态包括 EEG 等，但不包括 fNIRS | 项目中的 fNIRS 使用应明确称为跨模态适配，而非原论文原生 fNIRS 能力 |
| EFRM | 原文强调 EEG–fNIRS 重建与对比学习的共享域、少样本探针；上游代码按有序数据/被试目录固定 train/test，但公开材料不足以核实所有数据集的身份映射 | 不应声称已完全核实原文 split；应重点检验对齐是否为任务相关、实例级对齐 |
| STA-Net | 原文主要报告被试特异 MI/MA/WG，并将优势归因于 fNIRS 引导空间注意和 EEG 引导时延注意 | 协议桥接是解释巨大 gap 的核心，同时必须用干预验证注意是否真的被使用 |
| BrainFusion | 原 MI 案例按参与者分别建模并做 10 折选择/stacking，且使用较完整的 fNIRS/NVC 时间信息 | 原文约 0.955 与本项目约 0.549 不是同一 estimand；需拆分 split、时间窗和重实现边界 |

## 3. 待检验的机制假设

所有假设都应在实验开始前绑定主指标、比较方向和判定规则。

| ID | 假设 | 可证伪预测 | 若不成立的含义 |
| --- | --- | --- | --- |
| H1 | gap 主要由被试内到跨被试的 estimand 改变造成 | 在相同数据和模型上，性能随 `trial-random → 被试内 group-safe → 跨被试` 单调下降；主体下降发生在最后一步 | 不能再以 split 作为主解释 |
| H2 | 统一 16 通道和短上下文丢失了方法原生信息 | 增加原生通道或 fNIRS 后延窗口后，公共开发集上有稳定、逐被试一致的增益 | 输入预算不是主要瓶颈 |
| H3 | 冻结线性探针低估了需要任务适配的方法 | `linear → MLP → last-block/parameter-efficient → full fine-tune` 出现有序增益，尤其 CBraMod | 预训练表示本身或域匹配更可疑 |
| H4 | 预训练没有贡献任务相关迁移 | 预训练权重不优于同架构随机初始化，或只提高被试/数据集身份探针 | 模型架构而非预训练知识贡献性能 |
| H5 | 跨模态 alignment 弱、塌缩或依赖捷径 | 精确配对检索接近机会水平、hardest-negative margin < 0、有效秩低；同被试负样本会消除表面优势 | 若 alignment 强，则需检查它是否与标签相关 |
| H6 | 跨模态模块没有被下游真正使用 | 打乱另一模态、置换注意或门控后，预测和性能几乎不变 | 多模态优势主张在当前域未生效 |
| H7 | 表征携带强被试/数据集身份而非标签不变量 | 身份探针显著强于任务探针；移除身份方向后跨被试标签性能提高或不降 | 主因更可能是信号不足或适配不足 |
| H8 | Visual/n-back 等任务的跨被试标签信号本身很弱或时间窗错配 | 学习模型和经典强基线均接近基线；标签/事件/窗口审计通过但 subject-held-out 学习曲线早早饱和 | 若经典方法明显更强，应回查 adapter/表征 |

## 4. 总体实验架构

### 4.1 数据治理：先避免二次“看 test 调实验”

1. 新机制实验默认只在现有 **public train/dev 或 outer-train 内嵌交叉验证**上开发；不得反复读取 protected test 选择窗口、层、学习率或图形口径。
2. 将本文件冻结为 v1 后，为每组实验生成配置哈希、数据清单哈希、代码提交和随机种子清单。
3. 最终需要把新增数字写入论文时，先冻结所有决策，再申请一次新的、独立授权的评估；如果不重新授权，则只报告 public/dev 机制证据，并保持既有 protected 主结果不变。
4. 原始任务锚点、统一 direct track、method-native context 和 target-overlap track 分表报告，禁止挑每个方法的最佳不兼容设置组成总排名。

### 4.2 三层实验面板

#### 层 A：共同诊断面板

| 实验 | 设计 | 主输出 | 主要回答 |
| --- | --- | --- | --- |
| G0 数据/标签/窗口审计 | 每任务逐被试检查类数、样本数、事件到窗映射、重叠、坏道、幅度/频谱、fNIRS HbO/HbR 合理性；训练集内做 label permutation | 审计表、事件对齐图、permutation null | 是否存在任务定义或数据管线问题 |
| G1 经典强基线 | EEG：CSP/FBCSP、Riemannian；fNIRS：均值/斜率/GLM/CSP；早期拼接与晚期融合；同一 outer split | 逐被试 macro-F1/CCC、CI | 任务是否包含可跨被试利用的基础信号 |
| G2 协议桥接 | 固定模型/输入，依次评估 trial-random、被试内 record/session-grouped、严格跨被试 | gap waterfall | gap 中有多少与 estimand 有关 |
| G3 输入预算 | 共同 16 通道 vs 方法原生可用通道；统一窗 vs 方法原生窗/后延窗；只在兼容方法上做 | 通道/时间剂量曲线 | 公平协议是否切断原论文机制 |
| G4 适配容量阶梯 | 线性、等参数量 MLP、只训 query/归一化、last block 或 PEFT、全量微调；统一开发预算 | 预训练增益与适配增益 dumbbell | gap 是否源于冻结策略 |
| G5 预训练对照 | 官方预训练 vs 同架构随机初始化；训练预算相同；必要时加域内自监督 | `Δpretrain` 及逐被试差 | 预训练知识是否迁移 |
| G6 表征可诊断性 | 逐层标签、被试 ID、数据集 ID、session ID 探针；有效秩、谱、CKA/SVCCA；仅在训练数据拟合探针 | identity–label 平面、层曲线 | 表征是否被身份/域主导 |
| G7 因果扰动 | 通道/坐标/模态/配对/时间顺序置换，缺失模态与噪声剂量，保持类和被试结构 | 性能下降和预测一致率 | 被声称模块是否真正参与决策 |
| G8 学习曲线 | 每个 outer-train 使用 10/25/50/100% 被试或标签量，固定超参选择规则 | 性能–被试数曲线 | 是数据不足、容量不足还是已饱和 |

#### 层 B：原始任务锚点

对每个有官方代码、权重和可得下游数据的方法，选择**一个**论文中信息最完整的任务，运行最小 source-anchor：原始通道、预处理、split、头和训练预算尽量一致，并同时记录无法一致的项目。锚点不是新的主比较，而是安装/权重/调用链的正控制。

- 若锚点能恢复论文合理区间而本项目低：支持“实现基本正确，域/协议改变导致下降”；
- 若锚点也显著失败：优先修复 source fidelity，暂不能把本项目低分解释为泛化失败；
- 若原始数据或代码不可得：明确标记 `not verifiable`，不可用自建近似结果声称复现成功。

#### 层 C：方法机制面板

只选择能检验原论文核心创新点的少量实验，详见第 5 节。每个方法至少包含一个正对照、一个负对照和一个因果扰动，避免只展示 UMAP 或注意图后做机制归因。

## 5. 分方法实验设计

### 5.1 BIOT：频谱 token、通道语义与跨数据预训练是否迁移

**论文主张。** BIOT 将每个通道的短时频谱片段编码为 token，并加入通道与相对位置编码，以统一处理变长、异构和缺失通道的生物信号；跨数据预训练是性能来源之一。

**当前关键边界。** 项目使用 PREST-16 官方权重、16 个实测原生电极、8 s 分类窗/2 s DSR 窗和冻结标准化线性头；原文常用 200 Hz、1 s token/0.5 s overlap、特定归一化与任务适配。项目设置是可信的“位置适配迁移”，不是原论文数值复现。

| 实验 | 对照与指标 | 判读 |
| --- | --- | --- |
| B1 预训练贡献 | PREST-16 vs 随机初始化同架构；线性和相同微调预算各一组 | 仅微调后有增益说明静态表征不线性可用；始终无增益说明域不匹配 |
| B2 通道语义干预 | 正确电极名/位置、随机置换通道 ID、固定同一通道 ID、缺失通道剂量 | 正确语义应显著优于置换；否则通道编码优势未被使用 |
| B3 原生预处理桥接 | 当前 canonical coordinate 与论文式分通道 95 分位幅值缩放、原始 token 参数分别运行 | 确认输入统计与频谱分块是否造成 gap |
| B4 层与池化 | 各层 mean/attention pooling 线性探针，配合标签和 subject-ID probe | 找到标签可分信息是否存在但在最终池化丢失 |
| B5 频谱充分性 | 从 BIOT token 频段能量解码任务标签；与 FBCSP 基线比较 | 若 FBCSP 强而 token 弱，问题定位到频谱 token/预处理接口 |

主要图：`Δpretrain` dumbbell、通道置换剂量曲线、逐层 label-vs-subject probe 曲线。

### 5.2 CBraMod：首先检验“冻结”而不是重复调线性头

**论文主张。** CBraMod 用 criss-cross transformer 分别建模空间和时间依赖，ACPE 支持可变通道/时间格式，时域与频域 patch 共同表征 EEG。

**最关键原文证据。** 原文消融直接显示 frozen 参数可相对 full fine-tune 大幅下降，例如其表中 FACED balanced accuracy 约从 0.5509 降至 0.3146，PhysioNet-MI 从约 0.6417 降至 0.3845。原文多个下游数据也采用严格被试划分。因此 CBraMod 的首要解释假设是**当前冻结边界与论文最优使用方式不同**，而非“原论文只做被试内”。

| 实验 | 对照与指标 | 判读 |
| --- | --- | --- |
| C1 适配阶梯（最高优先级） | frozen linear → 官方式 MLP/flatten head → 只训 LN/ACPE → last block/PEFT → full fine-tune；统一超参预算 | 恢复曲线直接量化冻结造成的 gap |
| C2 随机初始化架构 | 对 C1 每个主要容量点加入 random-init | `pretrained - random` 分离预训练与架构贡献 |
| C3 criss-cross 模块干预 | 空间支路、时间支路、频域支路分别置零/置换；保持参数尺度 | 判断创新模块在当前任务的实际边际贡献 |
| C4 ACPE/通道稳健性 | 正确坐标、通道置换、坐标噪声、16→原生通道 | 若坐标扰动无影响，ACPE 优势在当前 adapter 下未生效 |
| C5 逐层探针 | 各 block label/subject/session 线性探针与 effective rank | 确认最终均值池化是否丢失中间层信息 |

主要图：适配容量—性能阶梯；预训练和微调增益的两段式 waterfall；空间/时间/频域支路消融森林图。

### 5.3 REVE：4D 坐标、query 训练边界、规模与 target overlap

**论文主张。** REVE 用 4D Fourier 时空位置编码适应不同 montage/长度，依托大规模多数据预训练和辅助全局重建提升可冻结表征，并报告规模效应与线性探针优势。

**当前关键边界。** 项目调用官方位置库与最终 latent token，再用预训练 query attention 得到 512 维表征；但上游“linear probing”仍允许 `cls_query_token` 训练，而项目静态特征缓存使其冻结。当前采用 base、16 通道；MI/MA 与 REVE 预训练语料存在目标语料 overlap，已单独列轨。

| 实验 | 对照与指标 | 判读 |
| --- | --- | --- |
| R1 query 边界 | 冻结 query、可训练 query、MLP、last-block/full FT | 若仅放开 query 即恢复，gap 是抽取边界而非 backbone 失效 |
| R2 位置编码因果检验 | 真坐标、坐标置换、坐标噪声、统一假坐标、通道名错配 | 真坐标优势是 4D PE 在当前数据生效的必要证据 |
| R3 通道剂量 | 原生全通道逐步降到 32/16/8，固定被试 split | 检验统一 16 通道是否超出论文声称的稀疏稳健区间 |
| R4 规模/模型汤（低优先级） | base vs large；只有在固定预算与可得权重下才加入 soup | 检验规模效应，不作为救分式无限搜索 |
| R5 overlap 敏感性 | overlap 任务与非 overlap 任务分表；能得到排除目标语料权重时再作差 | 防止把 target exposure 误写为跨域泛化 |
| R6 身份与层诊断 | 各层 label/subject/dataset 探针、pooling 对照 | 区分空间域捷径和任务相关表征 |

主要图：坐标扰动剂量曲线、query 训练前后 paired subject effect、overlap/non-overlap 分面图。

### 5.4 NormWear：fNIRS 是适配域，重点检验 CWT 与跨通道融合

**论文主张。** NormWear 对原信号及一/二阶导数做多尺度 CWT tokenization，用共享单通道编码器和周期性 inter-channel CLS liaison 支持任意传感器组合，并强调通道排列稳健性。

**当前关键边界。** 官方预训练模态含 EEG、PPG、ECG、GSR、PCG、IMU，但不含 fNIRS。项目将 200 Hz EEG 和 10 Hz fNIRS 统一至 65 Hz，以 Ricker CWT 和完整通道输入，并拼接最终每通道 token 后训练线性头。因此结果应称 `NormWear adapted`，不能视为论文已验证的 EEG–fNIRS 模型。

| 实验 | 对照与指标 | 判读 |
| --- | --- | --- |
| N1 模态贡献 | EEG-only、HbO-only、HbR-only、EEG+HbO/HbR；再做模态 shuffle/missing | 多模态是否有真实边际增益、哪一模态被使用 |
| N2 预训练贡献 | 官方预训练 vs random-init，在 EEG-only 和 multimodal 分别比较 | fNIRS 适配是否稀释了 EEG 预训练收益 |
| N3 CWT/导数消融 | raw、raw+一阶、raw+一/二阶、完整 CWT；保持头容量 | 核心多尺度设计是否在目标域有效 |
| N4 频率占用审计 | EEG 与 10→65 Hz fNIRS scalogram 的能量、稀疏度和预训练统计距离 | 判断上采样后的 fNIRS token 是否落在模型支持域外 |
| N5 融合与排列 | 绕过 CLS liaison、只保留单通道编码；通道排列前后比较 embedding/预测 | 验证跨通道融合及论文声称的排列稳健性 |
| N6 表征条件数 | 逐层有效秩、特征维度/样本比、标准化前后线性头 | 排除“通道×768 高维拼接使探针病态”的实现性原因 |

主要图：模态消融 paired forest、EEG/fNIRS scalogram 分布、通道排列前后预测一致率。

### 5.5 EFRM：将“alignment 是否生效”变成可证伪的主分析

**论文主张。** EFRM 用 EEG/fNIRS 专属 masked autoencoder 学习模态内结构，再以对比目标构造共享域；论文认为共享域与宽频 EEG 输入提高少样本下游表现。

**现有直接证据。** 项目已经保存两组完整公共验证集 alignment 报告：

- 排除 `eeg_fnirs_single_trial` 的 Stage-A：重建损失显著改善，正负配对 AUC = 0.5413，双向 MRR 略高于机会；但 hardest-negative margin = -0.8312，EEG/fNIRS centered effective rank 约 1.33/1.20，第一轴能量均超过 96%；
- 排除 `simultaneous_eeg_nirs` 的 Stage-A：AUC = 0.5508，但一个检索方向低于机会，hardest-negative margin = -0.8283；两模态接近 rank-one，第一轴能量超过 98%/99%。

这支持一个谨慎结论：**已存在可检测的平均配对分离，但不是稳健的实例级匹配，而且几何高度压缩。** 不能仅凭 CLIP 标量损失近似不变判断 alignment 失败，因为 source-faithful 的 0.1 固定 logit multiplier 将 logits 限制在很窄范围；检索、margin、秩和负对照更有判别力。

#### E1 分层负样本检索（最高优先级、可复用现有 checkpoint）

对每个 EEG query 分别构造：

1. 精确同步 fNIRS 正样本；
2. 同被试、不同时间负样本；
3. 同数据集、不同被试负样本；
4. 不同数据集负样本；
5. 类别相同但实例不同负样本。

双向报告 Recall@1/5/10、MRR、median rank、positive-minus-negative cosine、positive-minus-hardest-negative margin 和配对 AUC。以**被试/record 为置换块**获得 null；不能把窗口当独立样本。

关键判读：若只优于“不同数据集”负样本，却无法区分同被试错时或同数据集其他被试，则 alignment 主要是数据集/身份捷径，而非同步脑状态共享域。

#### E2 时间滞后扫描

固定 EEG 窗，对同被试 fNIRS 做预注册的 lag grid（范围由可用前后上下文决定，不能越界补伪数据），计算配对相似度、AUC/MRR 和下游性能随 lag 的变化；与同被试随机 circular shift、time reversal 和跨试次同类替换比较。

预期图为“相似度/检索—lag 曲线 + 被试块置换的 95% null envelope”。如果峰值稳定落在生理合理的血氧延迟且 time reversal 破坏它，才可支持时序跨模态关联；单个 t-SNE/UMAP 聚团不足以支持该结论。

#### E3 几何与身份捷径

- 每层、每 checkpoint 计算 centered effective rank、奇异值谱、各向同性、CKA/SVCCA 和 whitened Procrustes；
- 用相同容量探针预测 task label、subject、dataset、session；
- 在 outer-train 内拟合 subject/dataset nuisance directions，投影去除后再在 untouched dev-subject 上测试标签探针。

若身份探针强而标签探针弱、去除身份方向后标签泛化不降甚至提高，说明共享域被域身份主导。

#### E4 checkpoint 轨迹与下游相关性

对保存 checkpoint 同时画：EEG/fNIRS 重建、alignment AUC/MRR/margin、effective rank 和公共 dev 下游分数。预先定义 Spearman 相关及 subject-bootstrap CI。

- 重建持续改善而 alignment/下游不改善：总 loss 不是合适的任务相关选择标准；
- alignment 改善但下游不改善：学到的对齐不具任务相关性；
- rank 与下游同步恢复：塌缩可能是主要机制。

#### E5 小规模受控重训（在 E1–E4 完成后）

仅在公共数据上比较 reconstruction-only、原始权重/温度、可学习或校准温度、shuffled-pair 负对照、variance/covariance regularization，以及配对数据比例剂量。所有变体采用相同训练步数和选择规则。

这些结果必须称为“EFRM mechanism variants”，不能悄然替换主表中的 source-faithful EFRM。只有当某个改动同时提高 alignment 几何、负对照区分和跨被试下游，才能认为修复了机制，而不是又一次调分。

#### E6 频段与跨模态共享信息

原论文将更宽 EEG 频带与更高的 EEG–fNIRS 共享信息联系起来。因而在不改变其他处理的条件下，预注册 broad band 与 delta/theta/alpha/beta/low-gamma 等频带，对每组同时报告 alignment AUC/MRR/effective rank 和跨被试下游变化。若宽频只提高重建或数据集身份探针、却不提高同步检索和标签性能，则不能据此声称共享的任务信息增加。滤波器阶数、过渡带和边缘裁剪必须固定，避免把滤波伪迹解释为频带作用。

主要图组：分层相似度矩阵（按 dataset/subject/time 排序）、双向层级检索、奇异值谱、lag profile、checkpoint 四轨迹、alignment gain 与 downstream gain 的逐被试散点。

### 5.6 STA-Net：空间/时延注意需要干预证据

**论文主张。** FGSA 用 fNIRS 引导 EEG 空间注意；EGTA 用 EEG 引导时间 cross-attention，自适应处理 fNIRS 延迟。原文主要被试特异 MI/MA/WG 准确率约为 69.65%/85.14%/79.03%。项目现有严格跨被试 context reference 的 macro-F1 为 56.40%/62.84%/62.11%，两者评价对象和指标并不完全相同。

| 实验 | 对照与指标 | 判读 |
| --- | --- | --- |
| S1 协议桥接 | 同一数据/代码：trial-random → 被试内 session/record-grouped → 跨被试 | 直接量化原文被试特异与本项目泛化差距 |
| S2 模块消融 | EEG-only、fNIRS-only、无 FGSA、无 EGTA、固定 lag、完整模型 | 两个创新模块在当前域的边际贡献 |
| S3 时延干预 | fNIRS 按 lag 平移、time reversal、同被试错配、跨被试错配 | EGTA 是否利用真实时序关系 |
| S4 空间干预 | FGSA map 置换、均匀化、左右翻转；和 contralateral motor ROI 的预定义重叠比较 | 空间图是否具有功能和解剖一致性 |
| S5 稳定性 | 每被试/任务/session 的注意熵、峰值 lag、空间图 ICC/重测一致性 | 漂移很大的图不能作为稳定机制证据 |
| S6 融合门控 | 门控分布、模态置换前后门控与预测变化 | 排查融合权重塌到单模态 |

可视化必须同时包含真实模型与置换 null envelope。注意热图只作为结果描述；性能对注意/时延的干预响应才是机制证据。

### 5.7 BrainFusion：分解被试内建模、血氧时间预算与 NVC

**论文主张。** 原 MI 案例对 29 名参与者分别训练，EEG/HbO/HbR/NVC 使用 CSP 和分类器，再以 stacking 融合；报告参与者间平均 EEG、HbO、HbR、NVC、ensemble 准确率约为 0.570、0.906、0.901、0.934 和 0.955。其 NVC 将 EEG PSD 经 canonical HRF 卷积预测血氧响应。

**当前关键边界。** 论文案例完整 CSP/stacking 代码并未公开，项目是独立可信重实现；统一 direct track 使用 8 s time-zero 支持并裁剪 HRF，不包含原方法可能依赖的较长血氧后延信息。因此不能把项目结果称为数值复现。

| 实验 | 对照与指标 | 判读 |
| --- | --- | --- |
| F1 协议桥接 | source-like 被试内 10-fold、被试内 session/record-grouped、严格跨被试 | 估计 source estimand 带来的份额，并暴露 trial leakage 风险 |
| F2 时间窗/HRF 剂量 | 8 s 同步窗、增加不同 fNIRS post-stimulus 窗、完整可用 HRF；所有窗预定义 | 若长后延窗恢复 HbO/HbR/NVC，gap 与 observation budget 强相关 |
| F3 组件消融 | EEG、HbO、HbR、NVC、三者融合、stacking；报告增量 | 验证 ensemble 是否真的超越最好单模态 |
| F4 NVC 负对照 | 正确 HRF、time-reversed/random HRF、配对打乱；类/被试结构保持 | 正确 NVC 必须优于无生理意义卷积 |
| F5 预处理桥接 | 当前 canonical 与论文式 EEG 2–50 Hz、notch、重参考/可行伪迹处理分表 | 衡量源预处理差异，不污染统一主表 |
| F6 身份与 stack 诊断 | 单模态 subject-ID probe；out-of-fold stack 权重与逐被试贡献 | 识别被试特异模式和 stack 是否过拟合 |

主要图：source value → source-like within-subject → group-safe within-subject → cross-subject → support-matched 的 gap waterfall；时间窗剂量曲线；单模态到 stack 的逐被试增量。

## 6. 统计设计与判定规则

### 6.1 统计单位

- 跨被试评估以**被试**为主要统计单位；如果 outer fold 包含多个被试，bootstrap/置换首先在被试层进行，再汇总到 fold。
- 被试内桥接以 session/record 为 group，禁止将高度重叠窗当作独立样本。
- 方法 A/B 必须使用相同 held-out 样本做 paired comparison；报告均值差、95% CI 和预定义效应量，而不仅是各自均值。
- 主指标沿用现有协议：分类为 macro-F1，REFED 为 masked CCC；accuracy/balanced accuracy 只在复现原论文锚点时按原定义补充，不与主表横向混用。

### 6.2 推断方法

1. 主比较：subject-blocked paired permutation 或 hierarchical bootstrap 95% CI；随机种子在 fold 内先平均，避免将 seed 伪装成独立重复。
2. 学习曲线/剂量曲线：使用预定义单调趋势检验或混合效应模型，subject 为随机截距；样本量不足时只给 bootstrap 区间和原始点。
3. alignment：双向检索分别报告；配对置换必须在 subject/record block 内；Top-1 必须同时给机会值，不能只给绝对百分比。
4. 多重比较：按假设家族校正，而非对整篇所有探索统一校正。建议四个家族：协议、适配/预训练、alignment、模态/因果扰动；confirmatory 使用 Holm，探索性结果清楚标注。
5. 机制成立最低条件：主性能变化方向正确、CI 排除预注册的无实际意义区间，并且至少一个相应负对照失败。只有“图变好看”不算成立。

### 6.3 防止事后解释

- 预先冻结每个实验的主任务。建议：CBraMod 以 MI + 一个高信号任务（MA/WG）；EFRM 以已有 target-excluded 两组公共验证；STA-Net/BrainFusion 以 MI 为 source-case、MA 为复核；Visual 作为共同失败任务而非调参主任务。
- 每个方法最多一个主要窗口方案、一个主要微调预算和一个主层；其余为探索性。
- 不按 protected 结果为不同任务选择不同方法变体；不裁剪到 chance；不隐藏负结果。

## 7. 论文用可视化交付物

建议最终主文保留 3–4 张图，其余放补充材料。

1. **全局任务×方法热图**：色值为 `value - B0` 或标准化 above-baseline recovery；格内叠加 CI/terminal 状态，unsupported 留空。它展示失败按任务聚集。
2. **gap waterfall / bridge plot**：原文值、source-anchor、目标域被试内 group-safe、严格跨被试、support-matched frozen。每一步都标明 metric/split/input 差异，不能把不兼容数值画成连续因果分解而不加说明。
3. **预训练–适配二维图**：横轴 `pretrained - random`，纵轴 `fine-tune - frozen`。四象限能区分预训练有效、适配不足、架构贡献和域失配。
4. **identity–label probe 图**：每层/方法同时显示新被试标签探针和身份探针；身份强、标签弱是域捷径证据。
5. **EFRM alignment 机制图组**：层级检索矩阵、奇异值谱、lag 曲线及 null envelope、alignment 与 downstream 的 checkpoint/subject 对应关系。
6. **STA-Net 注意干预图**：真实/置换的 lag 分布与空间图，旁边放置相应性能下降；注意图和因果影响必须成对出现。
7. **模态贡献森林图**：EEG、fNIRS、融合、模态 shuffle 的逐被试 paired effect，适用于 EFRM、NormWear、STA-Net、BrainFusion。

可视化规范：显示所有 subject/fold 点；色盲安全配色；同一指标固定轴范围；不以 UMAP/t-SNE 作为主要量化证据；所有降维图固定参数并配合原空间指标。

## 8. 优先级与执行顺序

### P0：复用现有产物，先形成最便宜且最强的证据

1. 冻结本协议和结果字段 schema；
2. 制作任务×方法 above-baseline 热图和逐被试/折分布；
3. 完成 G0 数据/标签/窗口审计；
4. 用已有 EFRM checkpoint 完成 E1–E4，不先重训；
5. 做 label/subject/dataset/session 探针与 effective-rank/层诊断；
6. 运行经典强基线，尤其 Visual、n-back、MI。

**P0 成功标准：** 能回答低分是共同任务瓶颈、身份捷径还是 EFRM alignment 几何问题，并形成至少两张有置换 null 的机制图。

### P1：最能改变解释的受控训练

1. CBraMod C1/C2 适配容量阶梯；
2. STA-Net S1 协议桥接与 S2/S3 模块/时延干预；
3. BrainFusion F1/F2/F3 split—时间窗—组件分解；
4. NormWear N1/N2/N3 模态、预训练与 CWT 消融；
5. BIOT B1/B2 和 REVE R1/R2。

**P1 成功标准：** 对每个主要方法给出一个可重复的性能下降主因，并量化其 paired effect，而不是仅给定性猜测。

### P2：昂贵或仅补强的实验

- 完整 source-anchor reproduction；
- REVE large/model soup；
- EFRM 多变体受控重训；
- 原生全通道和长 fNIRS context 的全面剂量实验；
- 更大规模 label/subject learning curve。

只有当 P0/P1 无法区分关键假设时才启动 P2，避免把资源消耗在对论文防御价值低的全面网格搜索上。

## 9. 结果到结论的决策表

| 观察结果 | 允许的结论 | 不允许的过度结论 |
| --- | --- | --- |
| source-anchor 可恢复，gap 主要出现在跨被试一步 | 当前实现有正控制；主要挑战是新被试泛化 | “所有原论文都数据泄漏” |
| source-anchor 也失败 | source fidelity/依赖/调用链尚需处理 | “目标数据太难” |
| full fine-tune 大幅优于 frozen，pretrained 又优于 random | 预训练有效，但线性可读性/任务适配不足 | “基础模型无效” |
| full fine-tune 有效，但 pretrained≈random | 架构/监督训练有效，预训练迁移证据弱 | “预训练一定带来优势” |
| 经典与所有深度方法在 Visual 均近基线 | 当前窗和 split 下缺少稳定跨被试信号，需审计任务定义/时序 | “Visual 任务本身不可解” |
| 经典强、预训练方法弱 | adapter、池化或预训练域失配更可疑 | “严格跨被试必然导致低分” |
| EFRM 仅能区分跨数据集负样本且有效秩极低 | 存在域级捷径和压缩几何，实例级 alignment 弱 | “完全没有学到任何跨模态关系” |
| alignment 指标提高但下游不提高 | 对齐不是任务相关对齐 | “alignment 越强性能必然越高” |
| 模态/注意置换不降低性能 | 声称的多模态机制在当前模型/域未被决策使用 | “该模块在所有数据上无效” |
| 真实坐标优于位置置换且差异跨被试稳定 | 空间编码在当前数据上有功能贡献 | 仅凭注意/位置可视化声称神经机制 |

## 10. 建议的论文叙述结构

最终论文可按以下顺序组织，而不是围绕“没有达到原论文分数”被动解释：

1. 主表首先明确它是统一 support、统一严格跨被试 estimand 下的比较，不是原论文数值复现；
2. 用全局热图证明劣化具有任务聚集性，且 WG/MA 等任务并非全部失效；
3. 用 source-anchor 和协议 bridge 量化原文设置到当前设置的主要差异；
4. 用 CBraMod 适配阶梯说明 frozen representation 与 full fine-tuning 是不同问题；
5. 用 EFRM 检索、秩、lag 和负对照说明 alignment 在当前域是“可检测但弱/捷径化”，而不是只展示嵌入图；
6. 用 STA-Net/BrainFusion 的时延和模态干预说明 fNIRS observation budget 对其原生机制的重要性；
7. 以限制段落承认无法完全复现的代码/数据和 REVE overlap，并公开所有有效低结果。

一个可预先采用的中性表述是：

> “这些结果衡量统一观测预算与严格新被试泛化条件下的方法行为，而非对原论文被试内、方法原生输入或端到端训练结果的数值复现。机制实验进一步将差距分解为评估协议、适配容量、输入时间/空间预算与表征迁移因素；所有开发决策均在非 protected 数据上完成。”

具体结论必须等待实验结果，不得提前把上句中的任一因素写成已证实主因。

## 11. 证据来源与可追溯性

### 项目内

- 联合结果：`docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md`
- 统一协议：`docs/comparisons/PROTOCOL.md`
- 指标接纳：`docs/comparisons/METRIC_ACCEPTANCE.md`
- adapter gates：`comparative_methods/ADAPTER_ALIGNMENT_GATES_V2.md`
- BIOT：`comparative_methods/BIOT/sources/SOURCE_FIDELITY.md`
- CBraMod：`comparative_methods/CBraMod/sources/SOURCE_FIDELITY.md`、`comparative_methods/CBraMod/REPRESENTATION_LAYER_AUDIT.md`
- REVE：`comparative_methods/REVE/IDENTITY_AND_REPRESENTATION_AUDIT.md`
- NormWear：`comparative_methods/NormWear/IDENTITY_AND_ADAPTATION_AUDIT.md`
- EFRM 完整 public validation 报告：
  - `comparative_methods/EFRM-PyTorch/runs/pretraining/efrm_lodo_full_target_fivefold_v2__exclude_eeg_fnirs_single_trial__stage_a_seed42/analysis/REPORT.md`
  - `comparative_methods/EFRM-PyTorch/runs/pretraining/efrm_lodo_full_target_fivefold_v2__exclude_simultaneous_eeg_nirs__stage_a_seed42/analysis/REPORT.md`
- STA-Net context：`comparative_methods/STA-Net-PyTorch/runs/fivefold/20260727_sta_net_no_artifact_mask_converged_5fold_v1/aggregate/paper_table.md`

### 原始/官方来源

- BIOT: Yang et al., NeurIPS 2023, [arXiv:2305.10351](https://arxiv.org/abs/2305.10351), [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/f6b30f3e2dd9cb53bbf2024402d02295-Paper-Conference.pdf)
- CBraMod: Wang et al., [arXiv:2412.07236](https://arxiv.org/abs/2412.07236)
- REVE: Cui et al., [arXiv:2510.21585](https://arxiv.org/abs/2510.21585)
- NormWear: [arXiv:2412.09758](https://arxiv.org/abs/2412.09758)
- EFRM: Jung and An, *Computers in Biology and Medicine* (2025), [DOI:10.1016/j.compbiomed.2025.111292](https://doi.org/10.1016/j.compbiomed.2025.111292)；项目内保存论文 PDF
- STA-Net: [DOI:10.1016/j.inffus.2025.103023](https://doi.org/10.1016/j.inffus.2025.103023)
- BrainFusion: Lee et al., *Advanced Science* (2025), [DOI:10.1002/advs.202417408](https://doi.org/10.1002/advs.202417408), [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12407257/)

### 尚未完全核实的边界

- EFRM 上游代码显示按有序数据/被试目录固定训练和测试集合，但缺少全部原始目录身份映射，不能据此断言每项论文实验都严格 subject-disjoint；
- BrainFusion 论文案例的完整 CSP/AutoML/stacking 实现未公开，本项目只能做独立重实现的机制桥接，不能声称逐数值复现；
- REVE MI/MA 的目标语料 overlap 必须继续保持单独轨道；
- source-anchor 的可复现性需待原始数据许可、依赖和算力核实后更新。

## 12. 预期最小交付包

实验进入执行阶段后，每个实验单元至少输出：

1. 冻结配置、数据/代码/checkpoint hash 与运行 manifest；
2. sample/subject/fold 支持表和 exclusion reason；
3. 主指标、paired effect、95% CI、null/permutation 结果；
4. 逐被试原始点和机器可读表；
5. 至少一个正对照与一个负对照；
6. 图形生成脚本和图源数据；
7. 一段严格按第 9 节决策表生成、区分 confirmatory 与 exploratory 的解释。

这套交付物足以把“分数低”从一个防御性问题，转化为对跨被试泛化、适配容量和跨模态表征机制的系统实证分析。
