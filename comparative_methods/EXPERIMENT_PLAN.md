# 对比方法未来实验计划

_计划快照：2026-07-31；本文只规定对比方法，不授权开启新的 protected
evaluation 或高成本训练。_

本文是 `comparative_methods/` 下对比实验的规划入口。现阶段正式方法集合固定为：

- 单模态 EEG：**BIOT、CBraMod、REVE**；
- 多模态 EEG–fNIRS：**NormWear、EFRM、BrainFusion NVC–CSP
  Stacking、STA-Net**。

方法进入此队列表示“计划实现或保留为正式比较”，不表示已经完成复现、通过
source-fidelity 检查或可以填入最终论文表格。现有数据、划分、指标和
protected-test 边界继续服从
[`docs/DATA_CONTRACT.md`](../docs/DATA_CONTRACT.md)、
[`docs/comparisons/PROTOCOL.md`](../docs/comparisons/PROTOCOL.md) 和
[`docs/comparisons/METRIC_ACCEPTANCE.md`](../docs/comparisons/METRIC_ACCEPTANCE.md)。

## 1. 固定方法队列

### 1.1 单模态：EEG-only

| 方法 | 在本项目中的作用 | 计划输入 | 计划主评测轨 | 当前状态 |
| --- | --- | --- | --- | --- |
| BIOT | 通用生理信号 foundation model 代表；检验跨数据预训练的 EEG 表征迁移 | 仅 EEG | 官方预训练权重 + 冻结编码器 linear probe | 待获取并固定官方实现、权重和许可证 |
| CBraMod | EEG foundation model 代表；检验通道–时间结构化预训练表征 | 仅 EEG | 官方预训练权重 + 冻结编码器 linear probe | 待获取并固定官方实现、权重和许可证 |
| REVE | 几何感知 EEG foundation model 代表；检验真实电极坐标编码的跨布局迁移 | EEG + 已登记电极坐标 | 官方预训练权重 + 冻结编码器 linear probe | 待适配；预训练数据重叠需单列 |

本轨只回答“EEG 单模态表征能达到什么水平”。不得把 fNIRS、EEG–fNIRS
融合或本项目的 derived teacher 特征输入这三个模型，也不得据此声称
fNIRS-only 性能。

REVE 的官方预训练集合包含 `Shin2017A`。因此，在 Single-Trial 数据上的官方
checkpoint 结果必须标为 `open_world_pretrained_with_target_corpus_overlap`，
不能放入“目标数据集完全排除”的 inductive 表。若未来获得明确排除该语料的
checkpoint，才增加独立的 clean target-excluded track。BIOT 和 CBraMod 的
预训练语料与四个目标数据集是否重叠，在代码接入前仍须逐项核验；未核验不能
默认写成无重叠。

### 1.2 多模态：EEG–fNIRS

| 方法 | 方法角色 | 计划输入 | 计划主评测轨 | 当前状态 |
| --- | --- | --- | --- | --- |
| NormWear | 通用多变量生理信号 foundation model；检验统一时频 token 化能否迁移到 EEG–fNIRS | 同步 EEG + HbO/HbR | 项目适配版、冻结编码器 linear probe | 待获取上游实现并完成 fNIRS 适配 |
| EFRM | 直接面向 EEG–fNIRS 的多模态表示学习基线 | 同步 EEG + HbO/HbR | target-dataset-excluded LODO 预训练 + linear probe | 已有独立实现；正式 v2 流程进行中 |
| BrainFusion NVC–CSP Stacking | 传统显式融合基线；检验 NVC/CSP 手工结构与 stacking 的竞争力 | 同步 EEG + HbO/HbR | fold 内特征拟合 + 监督分类/回归 | 待固定原文、实现范围和超参数协议 |
| STA-Net | 任务监督式深度时空融合基线 | 同步 EEG + HbO/HbR | strict cross-subject 五折端到端训练 | 正式五折已完成，保留为固定参考 |

NormWear 原始方法面向更广泛的可穿戴生理信号，而不是原生 EEG–fNIRS
专用模型。本项目必须将其报告为 `normwear_eeg_fnirs_adapted`：保留其上游
tokenization/backbone 原则，但单独记录 fNIRS 的 HbO/HbR 表示、采样率、
时频参数、通道身份和所有代码偏离。适配结果不能表述为原论文 fNIRS 数值复现。

EFRM 继续使用现有
[`EFRM-PyTorch`](EFRM-PyTorch/README.md) 的 LODO v2 方案；STA-Net 继续使用
现有 [`STA-Net-PyTorch`](STA-Net-PyTorch/README.md) 的正式五折结果。不得为了
统一新方法而回写、调参或选择这两项已经冻结的 protected 结果。

BrainFusion NVC–CSP Stacking 的 NVC 时延/响应核设定、CSP 滤波器数量、特征
选择和 stacking 学习器必须只在 outer-training 内拟合或用 inner validation
选择。若原文没有公开实现，项目版本必须标为独立重实现，并列出无法从原文确定
的细节；不得用测试折选择最有利的 NVC 时延或特征组合。

## 2. 明确不进入正式队列的工作

### fNIRS Foundation Model for Few-Shot Based fNIRS Classification

该 BCI 2025 四页短文与 EFRM 来自同一团队，且同样采用 fNIRS masked
autoencoder/few-shot 路线。现阶段没有足够材料证明它应当作为独立于 EFRM 的
第二个正式方法；将其同时计入会带来方法重复和证据不对称。因此：

- 不为该短文新增正式方法行；
- 不把同团队后续 EFRM 实现反向宣称为这篇短文的官方代码或精确复现；
- 如将来作者提供独立 checkpoint、完整数据清单、split 和实现，并能证明其与
  EFRM fNIRS 分支存在足够方法差异，再重新评审；
- 在此之前，只能将其思想视为 EFRM/fNIRS-MAE 路线的背景，不进入最终排名。

### UMAP

仓库中已有 UMAP 历史诊断材料继续保留，但 UMAP 不在本次固定方法集合中。
不再为其安排新的正式重跑，历史上反复查看 test 后得到的结果也不进入新的
跨方法主表。

## 3. 统一比较面

### 3.1 任务与主指标

在数据合同允许且方法输入可用时，统一覆盖七个任务：

| 数据集 / 任务 | 单模态表输入 | 多模态表输入 | 主指标 |
| --- | --- | --- | --- |
| Single-Trial / motor imagery | EEG | EEG + fNIRS | macro-F1 |
| Single-Trial / mental arithmetic | EEG | EEG + fNIRS | macro-F1 |
| Simultaneous / word generation | EEG | EEG + fNIRS | macro-F1 |
| Simultaneous / n-back | EEG | EEG + fNIRS | macro-F1 |
| Simultaneous / DSR | EEG | EEG + 同步 fNIRS context | macro-F1 |
| Visual / cognitive motivation | EEG | EEG + fNIRS | macro-F1 |
| REFED / valence–arousal sequence | EEG | EEG + fNIRS | masked CCC |

Accuracy、balanced accuracy、class-wise F1、MAE/RMSE、runtime、峰值显存和
参数量作为伴随指标。分类与回归不合成为单一排名。

### 3.2 必须分开的结果表

最终至少分为以下结果轨，不允许混表：

1. `single_modal_eeg_official_pretrained_linear_probe`；
2. `single_modal_eeg_official_pretrained_full_finetune`（资源允许时的二级轨）；
3. `multimodal_target_excluded_linear_probe`；
4. `multimodal_supervised_end_to_end`；
5. `open_world_pretrained_with_overlap`；
6. `adapted_or_independent_reimplementation`。

主论文中的 foundation-model 表优先使用冻结编码器 linear probe，以比较表示
质量；full fine-tuning 只能作为单独的端到端适配表。STA-Net 和 BrainFusion
属于监督融合轨，不与 frozen-probe 数字混成一个 foundation-model 排名。

full-label strict cross-subject 是统一主矩阵。Few-shot 曲线属于二级分析，
必须在运行前冻结每类 shot/label budget、抽样单位、重复 seed 和缺类处理规则；
在这些字段未冻结前，不启动 few-shot 正式实验，也不从观察结果反推最有利预算。

### 3.3 输入与适配边界

- 四个数据集使用相同 eligible subjects、样本清单、outer folds、标签、mask
  和任务窗口；每个方法另存 split fingerprint。
- 单模态表只加载 EEG。多模态表只使用真实同步 EEG–fNIRS pair，缺失模态样本
  不通过复制或生成伪装成 observed pair。
- Single-Trial fNIRS 必须从双波长强度按数据合同转换为 HbO/HbR；不得把两条
  wavelength 直接重命名为 HbO/HbR。
- 不复制、镜像或随机补齐通道。模型要求固定通道数时，优先使用 mask-aware
  adapter；无法做到时将该 cell 标为不支持，而不是改变 measured support。
- REVE 只消费 registry 中有来源的电极坐标。缺失或模板投影坐标必须保留其
  provenance，不能声称为被试级精确配准。
- 所有 normalization、时频变换、NVC/CSP 特征、target scaling、head 和
  threshold 均在 outer-training 内拟合。

## 4. 每个新方法的实施门

新方法按以下顺序推进，未通过前一门不得开始高成本正式矩阵：

| 门 | 必须产生的证据 | 通过标准 |
| --- | --- | --- |
| B0 — 来源固定 | 论文、官方仓库、revision、许可证、checkpoint、预训练语料清单 | 来源可追溯；目标语料重叠状态不再是 `unknown` |
| B1 — 输入合同 | adapter 说明、shape/rate/channel/geometry/mask assertions | 不伪造模态、通道、坐标或有效支持 |
| B2 — 软件 smoke | finite forward/backward、一次 optimizer step、checkpoint reload | 数值有限且可重复 |
| B3 — source fidelity | 原任务或最小官方示例、关键模块和参数偏离表 | 能解释项目实现与 named method 的边界 |
| B4 — 协议冻结 | folds、seeds、主指标、checkpoint rule、label budget、unlock rule | 测试数据保持不可访问 |
| B5 — 正式执行 | 全部计划 fold/seed 完成，失败单元保留 | 不用最好 seed/fold 替代完整矩阵 |
| B6 — 数字准入 | cell-level metric acceptance 审计 | 只把通过的同口径数字写入最终表 |

若官方代码或 checkpoint 不可用，B0 可以以“独立重实现”继续，但报告名必须加
`reimplementation`，且 B3 不得宣称数值复现。若许可证不允许所需使用或发布，
该方法保持 blocked。

## 5. 实施顺序

### 第一批：复用已完成或正在执行的多模态基线

1. 保留 STA-Net strict five-fold aggregate，不重新选择结果；
2. 让 EFRM LODO v2 按其冻结队列完成；
3. 将两者的最终结果按 track、任务和 metric 进入 cell-level 准入。

### 第二批：单模态 EEG foundation models

1. BIOT：完成 B0–B3，随后冻结与共享 outer folds 一致的 linear-probe 协议；
2. CBraMod：使用同一 EEG 输入支持和 linear-probe head 规则；
3. REVE：先完成坐标 sidecar 与预训练重叠审计，再分别报告 clean/overlap
   可用的轨。

三种方法的 linear-probe 主矩阵完成后，再决定是否投入 full fine-tuning。
不能因某个 linear-probe 结果较低而只为该方法开放额外标签或更宽调参预算。

### 第三批：补齐多模态方法谱系

1. BrainFusion NVC–CSP Stacking：先实现传统、可审计、计算成本较低的
   fold-local pipeline；
2. NormWear：最后进行 EEG–fNIRS 适配，因为其 fNIRS 输入契约和时频表示需要
   额外 source-fidelity 审计；
3. 在四个多模态方法都有同口径可用 cell 后，形成
   `foundation/generalist`、`direct EEG–fNIRS representation`、
   `traditional fusion`、`supervised deep fusion` 四类对照。

## 6. 计划状态

| 项目 | 状态 |
| --- | --- |
| 方法集合 | **已固定** |
| STA-Net 正式五折 | **已完成** |
| EFRM LODO v2 | **按既有冻结协议执行** |
| BIOT / CBraMod / REVE | **计划中；尚未通过 B0** |
| BrainFusion NVC–CSP Stacking | **计划中；尚未通过 B0** |
| NormWear EEG–fNIRS adaptation | **计划中；尚未通过 B0** |
| fNIRS Few-Shot Foundation Model | **不单列，等待足以证明独立性的材料** |
| UMAP 新正式实验 | **不再计划** |
| 新 protected evaluation | **未由本文授权** |

实现时，每个新方法在 `comparative_methods/<method>/` 下独立保存上游
revision、adapter、config、tests、source-fidelity 说明和运行工件，不向主方法
的 `experiments/runs/` 写入结果。
