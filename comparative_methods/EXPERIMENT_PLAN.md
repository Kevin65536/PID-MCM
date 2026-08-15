# 对比方法实验计划与完成记录

_原计划冻结于 2026-07-31；执行终态更新至 2026-08-14。本文记录已执行的
comparison campaign，但不授权任何新的 protected evaluation 或高成本训练。_

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

尚未冻结的 adapter 和 comparison cell 还必须服从
[`ADAPTER_ALIGNMENT_GATES_V2.md`](ADAPTER_ALIGNMENT_GATES_V2.md) 及其
[机器合同](adapter_alignment_gate_contract_v2.yaml)。旧 manifest 中的
`B1_input_contract: pass...` 不自动提升为 v2 通过；v2 按
`method × task × track × alignment_profile` 判定，而不是按整个方法判定。

新方法按 BIOT、CBraMod、REVE、BrainFusion、NormWear 的冻结顺序完成了 adapter
实现与审核、public preflight/development、freeze 和正式执行；事前 unsupported
处置也已保留。当前没有 active delivery method。EFRM LODO v2 与上述五种方法共同
进入完成的联合 protected campaign；STA-Net 仍只引用既有冻结结果。

## 1. 固定方法队列

### 1.1 单模态：EEG-only

| 方法 | 在本项目中的作用 | 计划输入 | 计划主评测轨 | 当前状态 |
| --- | --- | --- | --- | --- |
| BIOT | 通用生理信号 foundation model 代表；检验跨数据预训练的 EEG 表征迁移 | 仅 EEG | 官方预训练权重 + 冻结编码器 linear probe | 90/90 protected jobs 完成；3 ready-with-note、3 rejected；REFED unsupported |
| CBraMod | EEG foundation model 代表；检验通道–时间结构化预训练表征 | 仅 EEG | 官方预训练权重 + 冻结编码器 linear probe | 90/90 protected jobs 完成；4 ready-with-note、2 rejected；REFED unsupported |
| REVE | 几何感知 EEG foundation model 代表；检验真实电极坐标编码的跨布局迁移 | EEG + 已登记电极坐标 | 官方预训练权重 + 冻结编码器 linear probe | 90/90 protected jobs 完成；3 ready-with-note、1 rejected、MI/MA overlap-only；REFED unsupported |

本轨只回答“EEG 单模态表征能达到什么水平”。不得把 fNIRS、EEG–fNIRS
融合或本项目的 derived teacher 特征输入这三个模型，也不得据此声称
fNIRS-only 性能。

REVE 的官方预训练集合包含 `Shin2017A`。因此，在 Single-Trial 数据上的官方
checkpoint 结果必须标为 `open_world_pretrained_with_target_corpus_overlap`，
不能放入“目标数据集完全排除”的 inductive 表。若未来获得明确排除该语料的
checkpoint，才增加独立的 clean target-excluded track。BIOT 和 CBraMod 的
公开预训练语料已逐项核对，未发现与四个目标数据集同一语料的声明；这一结论
按 [`ASSET_STATUS.md`](ASSET_STATUS.md) 和各方法 manifest 的证据边界报告，
不外推为“绝对不可能存在任何未披露重叠”。

### 1.2 多模态：EEG–fNIRS

| 方法 | 方法角色 | 计划输入 | 计划主评测轨 | 当前状态 |
| --- | --- | --- | --- | --- |
| NormWear | 通用多变量生理信号 foundation model；检验统一时频 token 化能否迁移到 EEG–fNIRS | 同步 EEG + HbO/HbR | 项目适配版、冻结编码器 linear probe | 90/90 protected jobs 完成；5 ready-with-note、1 rejected；REFED unsupported |
| EFRM | 直接面向 EEG–fNIRS 的多模态表示学习基线 | 同步 EEG + HbO/HbR | target-dataset-excluded LODO 预训练 + linear probe | 105/105 protected jobs 完成；4 ready-with-note、3 rejected |
| BrainFusion NVC–CSP Stacking | 传统显式融合基线；检验 NVC/CSP 手工结构与 stacking 的竞争力 | 同步 EEG + HbO/HbR | fold 内特征拟合 + 监督分类/回归 | 75/75 protected jobs 完成；3 ready-with-note、2 rejected；DSR/REFED unsupported |
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

UMAP 不在本次固定方法集合中，也不再安排正式重跑。原工作区已从 active tree
移除，历史实现和诊断结果仍可从 Git 历史追溯；历史上反复查看 test 后得到的
结果不进入新的跨方法主表。

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
  和任务窗口；每个方法另存 split fingerprint。声称 direct comparison 时还必须
  精确对齐每模态的 observation anchor/relative interval、真实通道身份集合和
  canonical branch hash，而不只是 window 名称或 tensor shape。
- 单模态表只加载 EEG。多模态表只使用真实同步 EEG–fNIRS pair，缺失模态样本
  不通过复制或生成伪装成 observed pair。
- Single-Trial fNIRS 必须从双波长强度按数据合同转换为 HbO/HbR；不得把两条
  wavelength 直接重命名为 HbO/HbR。
- 不复制、镜像或随机补齐通道。模型要求固定通道数时，优先使用 mask-aware
  adapter；无法做到时将该 cell 标为不支持，而不是改变 measured support。
- REVE 只消费 registry 中有来源的电极坐标。缺失或模板投影坐标必须保留其
  provenance，不能声称为被试级精确配准。
- 除数据合同已固定、对所有方法一致执行的 record-wise offline canonical
  measurement transform 外，所有群体/特征 normalization、可学习时频变换、
  NVC/CSP 特征、target scaling、head 和 threshold 均在 outer-training 内拟合。

adapter 之后的 patch/token grid、method-native 通道顺序、几何编码和池化可以不同，
因为它们属于方法定义；但共享 channel set 与实际 delivered order 必须分别保存
hash。固定的无标签 source-declared sample transform 可以作为 adapter 语义保留，
不得通过目标任务分数选择。使用不同数量通道的结果只能进入
`native_capacity_secondary`，不能和 support-matched 主表直接排名。

现有 STA-Net 冻结结果不回写或重跑，但其默认分类 observation budget 是 EEG 3 s +
fNIRS 13 s，DSR 是 2 s + 13 s；这与 EFRM 的 8/8 s、DSR 2/2 s 不同。因此现有
STA-Net 结果按 `method_native_context_reference` 保留，不能仅凭相同 sample/split
声称为同步 support-matched direct comparison。

## 4. 每个新方法的实施门

新方法按以下顺序推进，未通过前一门不得开始高成本正式矩阵：

下表保留项目级 B0–B6 执行阶段；其中 B1 的实际准入由 adapter v2 的 A0–A7
逐 cell 门控给出，B4 unlock 对应 A8。synthetic 或 public mini smoke 不能形成
full-public A4/A7 结论。

| 门 | 必须产生的证据 | 通过标准 |
| --- | --- | --- |
| B0 — 来源固定 | 论文、官方仓库、revision、许可证、checkpoint、预训练语料清单 | 来源可追溯；目标语料重叠状态不再是 `unknown` |
| B1 — 输入合同 | adapter v2 A0–A7 evidence bundle | scientific information budget 对齐；所有 planned cell 在 full public scope 下 resolved |
| B2 — 软件 smoke | finite forward/backward、一次 optimizer step、checkpoint reload | 数值有限且可重复 |
| B3 — source fidelity | 原任务或最小官方示例、关键模块和参数偏离表 | 能解释项目实现与 named method 的边界 |
| B4 — 协议冻结 | folds、seeds、主指标、checkpoint rule、label budget、unlock rule | 测试数据保持不可访问 |
| B5 — 正式执行 | 全部计划 fold/seed 完成，失败单元保留 | 不用最好 seed/fold 替代完整矩阵 |
| B6 — 数字准入 | cell-level metric acceptance 审计 | 只把通过的同口径数字写入最终表 |

若官方代码或 checkpoint 不可用，B0 可以以“独立重实现”继续，但报告名必须加
`reimplementation`，且 B3 不得宣称数值复现。若许可证不允许所需使用或发布，
该方法保持 blocked。

## 5. 已执行的实施顺序

以下顺序保留为本轮实际执行记录。它不构成再次运行或新协议的授权。

### 第一批：复用已完成或正在执行的多模态基线

1. 保留 STA-Net strict five-fold aggregate，不重新选择结果；
2. EFRM LODO v2 冻结预训练与 public downstream 队列均已完成；
3. 将两者的最终结果按 track、任务和 metric 进入 cell-level 准入。

### 第二批：单模态 EEG foundation models

1. BIOT：完成 B1–B3，随后冻结与共享 outer folds 一致的 linear-probe 协议；
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
| B0 资产审计 | **7/7 方法通过；可获取的本地权重已做非反序列化哈希核验** |
| EFRM LODO v2 | **105/105 protected jobs 完成；4 ready-with-note、3 rejected** |
| BIOT / CBraMod | **各 90/90 protected jobs 完成；REFED unsupported** |
| REVE | **90/90 protected jobs 完成；MI/MA overlap-only；REFED unsupported** |
| BrainFusion NVC–CSP Stacking | **75/75 protected jobs 完成；DSR/REFED unsupported** |
| NormWear EEG–fNIRS adaptation | **90/90 protected jobs 完成；REFED unsupported** |
| fNIRS Few-Shot Foundation Model | **不单列，等待足以证明独立性的材料** |
| UMAP 新正式实验 | **不再计划** |
| 联合 protected campaign | **540/540 完成；双签揭盲和聚合完成；42-cell 终态已生成** |
| 新 protected evaluation | **未由本文授权；必须新建协议和授权** |

实现时，每个新方法在 `comparative_methods/<method>/` 下独立保存上游
revision、adapter、config、tests、source-fidelity 说明和运行工件，不向主方法
的 `experiments/runs/` 写入结果。

## 7. 已执行的运行合同

以下字段在 v2 A8 / B4 冻结并用于本轮正式执行，现作为完成记录保留：

- 主协议为 strict cross-subject five-fold；outer seed 为 `42`，inner seed 为
  `43 + outer_fold`，下游 seeds 为 `[17, 42, 73]`；
- 复用当前 method-neutral registry，其 manifest SHA-256 为
  `2a10b36db85dba6ec5543edc7810ff85d978ea5af8c79fda3d38a1e5cfd11106`；
- 每个方法最多产生 `7 tasks × 5 outer folds × 3 seeds = 105` 个正式结果单元；
  inner selection 和 full-outer refit 是每个单元的两个训练阶段，不是额外结果；
- classification 的 primary endpoint 为 fold-mean macro-F1，REFED 为
  fold-mean native-coordinate masked CCC；seed 先在 outer fold 内平均，再汇总
  五个 folds，并单独报告 seed dispersion；
- 所有新方法先完成全部 public inner selection、full-outer refit 和 sanity
  gates，再通过一个联合 unlock manifest 一次性开放 protected folds；
- sample-random、few-shot 和 full fine-tuning 均不在第一轮主矩阵中。它们只有
  在 strict frozen-probe 主矩阵聚合后，才能以新版本协议单独授权。

计划文档本身不创建 unlock 权限。即使 protected manifest 已存在，训练代码也
不得读取其中的样本身份或数组。

### 7.1 主 checkpoint 决策

| 方法 | 第一轮主资产 | 决策边界 |
| --- | --- | --- |
| BIOT | `EEG-PREST-16-channels` | public preflight 发现 REFED partial terminal windows 无法满足当前 encoder 的 truthful time support，故按事前规则全局回退 PREST-16；六个分类任务使用各自冻结的 16 个真实电极，REFED v1 事前标为 unsupported。原 checkpoint 的 bipolar montage 与项目 native-electrode positional transfer 偏离必须显式报告 |
| CBraMod | `pretrained_weights.pth` | 固定 200 Hz、200-sample patch；token pooling/head 偏离必须先通过 B3 |
| REVE | `reve-base` | `reve-large` 仅作为主矩阵完成后的预注册二级规模分析；Single-Trial cell 固定进入 overlap track |
| NormWear | `normwear_pretrain_ckpt.pth` | MSiTF text-alignment checkpoint 不进入第一轮 frozen-backbone 主轨 |
| EFRM | 四个 target-excluded Stage-B checkpoints | 严格服从既有 LODO v2 freeze，不与任何官方预训练轨混写 |
| BrainFusion | 无预训练权重 | NVC、CSP、feature selection、base/meta estimators 全部 fold-local |
| STA-Net | 已冻结的 2026-07-27 formal aggregate | 不重跑、不根据新方法结果重新选 checkpoint |

BIOT、CBraMod 和 NormWear 的 PyTorch pickle 权重先完成静态来源审查，并优先用
`torch.load(..., weights_only=True)` 读取；不得为了适配旧 checkpoint 而直接
执行其中任意 pickled object。REVE 使用本地 safetensors，但仍受其 Responsible
Use Agreement 和禁止再分发边界约束。

## 8. 分批实施与退出条件

### Batch 0 — 共享 preflight（已完成）

本批只做 CPU/小显存工作，不触碰 protected data：

1. 从 method-neutral registry 生成只含 public identity、计数和 hashes 的共享
   comparison contract，并逐 task 核对 subjects、records、labels 和 masks；
2. 形成七任务的 EEG channel/geometry coverage 表，先解决 BIOT 的全局
   16/18-channel 决策，再把同一真实 EEG support 提供给 CBraMod 和 REVE；
3. 固定 Single-Trial intensity-to-HbO/HbR、其余数据集 chromophore 分支、
   REFED target mask 和 DSR synchronized-context 入口；
4. 固定统一的 prediction/metric schema、majority/prior baselines、run ID 和
   failure codes；
5. 为每个方法生成 checkpoint hash、source revision、data branch hash 和 split
   fingerprint 的 preflight report。

退出条件是所有 planned cell 都明确为 `supported` 或有事前理由的
`unsupported`，不存在 unknown channel、coordinate、mask 或 target semantics，
并且 A4/A7 报告覆盖全部 unique public inventory。带
`max_records_per_task` 或 `max_samples_per_record` 截断的报告只能是
`public_mini`，不得宣称退出 Batch 0。

### Batch 1 — B1/B2/B3 adapter 与 smoke（已完成）

按依赖顺序实施：

1. BIOT：先解决固定 channel token 与真实通道映射；
2. CBraMod：复用同一 EEG support，固定 patching、pooling 和 linear head；
3. REVE-base：接入带 provenance 的电极坐标 sidecar；
4. BrainFusion reimplementation：完成 fold-local NVC/CSP/stacking 单元测试；
5. NormWear adapted：最后固定 EEG/HbO/HbR identity、CWT 和模态 mask。

每个方法必须依次通过：

- 一个公开 mini-batch 的 finite forward；
- classification 或 regression head 的 finite backward 和一次 optimizer step；
- 只含 allowed training state 的 checkpoint save/reload；
- 同 seed 重跑的 shape、mask、sample identity 和输出一致性；
- 上游最小示例或 source-task 结构检查，以及逐项 deviation table；
- 一个公开 outer fold、一个 seed 的 selection/refit dry run。

Smoke 只证明连通性。不得用 smoke 分数决定 checkpoint、task support 或正式
超参数。

### Batch 2 — public development 与 B4 freeze（已完成）

对每种方法使用统一规模的、事前列出的轻量搜索空间。允许 inner validation
选择 learning rate、weight decay、batch size、epoch 数和 class weighting，但
linear probe 不得解冻 backbone，也不得搜索输入通道、数据分支、checkpoint 或
结果 track。

资源 pilot 必须记录每样本特征大小、峰值 GPU memory、wall time、data-loader
吞吐和预计总磁盘量。随后冻结：

- 完整 `method × task × fold × seed` job matrix；
- resolved config 和 checkpoint rule；
- selection metric、tie-break、epoch cap 和 full-outer refit 规则；
- retry、OOM batch-size fallback 和失败保留规则；
- protected unlock manifest 的先决 hashes。

任何影响 estimand、输入支持或模型容量的修改都创建新 protocol version，不能
作为同一 frozen run 的“修复”。

### Batch 3 — 单模态正式主矩阵（已完成）

BIOT、CBraMod、REVE-base 共同通过 adapter v2 并冻结后再启动。v1 因 REFED 对三种方法均事前
unsupported，上限为 270 个结果单元；冻结 encoder feature 应在不做 target-wide
拟合变换的前提下按 checkpoint/data-branch 缓存并跨 seeds 复用。

所有 public jobs 和 gates 完成后，三种方法使用同一个 unlock 事件完成最多
270 次 protected evaluation。任一方法的低分不会触发额外调参，也不会允许另外
两种方法提前查看 protected 结果。REVE 的 Single-Trial 两项任务进入 overlap
表，其余可用任务保留 open-world checkpoint 身份但不标为已知目标语料重叠。

### Batch 4 — 多模态补齐（已完成）

BrainFusion 在 EFRM、BIOT、CBraMod 和 REVE 依序到达各自冻结 scope 的终态后
晋级。NVC、CSP 和 stacking 的 GPU 实现先与可审计的小型 CPU reference 完成数值
等价性检查；NormWear 随后完成。两者均进入同一个冻结单 GPU campaign，没有因
前序方法结果改变方法定义。

两种新方法上限为 210 个结果单元、420 个 public selection/refit job。它们完成
后，与 frozen STA-Net 和完成后的 EFRM v2 按各自 track 汇总；BrainFusion 与
NormWear 均保留 `reimplementation/adapted` 名称，不能伪装成原论文原域复现。

### Batch 5 — 聚合和数字准入（已完成）

1. 先核对 expected 105-cell support 或事前冻结的 unsupported cells；
2. 生成 fold mean、fold SD、pooled OOF、seed dispersion、per-subject companion
   metrics、runtime、显存和参数量；
3. 在完全相同 folds 上做 subject-cluster bootstrap 和 paired differences；
4. 按 `comparison_metric_targets_v1.yaml` 为每个 cell 赋予
   `TABLE_READY`、`TABLE_READY_WITH_NOTE`、`INVALID_VALUE`、
   `FAILURE_RESULT` 或 `REJECTED_VALUE`；
5. classification 与 REFED、frozen probe 与 supervised/adapted、overlap 与
   target-excluded 分表，不生成跨指标总排名。

## 9. 历史资源调度

2026-07-31 的以下只读资源快照仅保留为历史调度记录：

- GPU0 正在运行 EFRM Stage-B Single-Trial final refit，项目进程约占
  16.6 GiB；在该队列结束前不向 GPU0 加入新方法任务；
- GPU1 约有 23.4 GiB 可用，当时可供 active delivery method BIOT 的 public
  audit/development 使用；不得借空闲资源提前启动 CBraMod 或更后方法；
- `/SSD_2` 约有 1.5 TiB 可用。运行目录继续忽略，不复制 raw data、upstream
  checkout 或相同 checkpoint；feature cache 在生成前必须先给出体积估计。

REVE-large、full fine-tuning、sample-random 和 few-shot 不进入当前资源队列。
EFRM 的既有冻结 GPU0 队列与新方法 delivery queue 分开管理。新方法侧每次只晋级
一个 method；GPU0/GPU1 的物理空闲不构成提前启动下一 delivery method 的授权。

## 10. 失败、重试与停止规则

- B1–B4 任一门失败：停止该方法正式矩阵，保留失败证据；
- OOM：只允许按冻结规则减小 batch size 或使用等价 gradient accumulation；
  改 tokenization、窗口、channel support 或模型宽度必须升协议版本；
- NaN、常量特征、类别塌缩、mask/coverage 不完整：不得进入 protected unlock；
- job 中断：只用同 config、fold、seed 和 checkpoint 重试，不更换更有利的 seed；
- protected 结果低于 baseline：按数字准入规则保留真实值并诊断，不回到同一协议
  调参；
- 任一正式单元缺失且无事前 unsupported 声明：整格 aggregate 为
  `INVALID_VALUE`，不得缩小分母。

## 11. 完成清单

当前代码进度如下：

1. 共享 public split/input/metric contract、adapter-alignment v2 contract 与
   preflight tests 已完成；
2. BIOT、CBraMod、REVE-base 各完成 `6 tasks × 5 folds × 3 seeds = 90`
   个 public jobs 和 A0-A8；三者的 REFED v1 均事前标为 unsupported；
3. BrainFusion 完成五个支持任务的 75 个 public jobs 和 A0-A8，DSR 与 REFED
   为事前 unsupported；NormWear 完成六任务 90 个 public jobs 和 A0-A8，
   REFED 为事前 unsupported；
4. 上述 435 个 public jobs 的汇总均明确为 `table_admissible=false`；随后只在联合
   candidate、lane 和双签授权通过后开放一次 protected evaluation；
5. EFRM LODO v2 的 4/4 selection、4/4 final refit、七任务 full-public replay、
   `7 tasks × 5 folds × 3 seeds = 105` 矩阵与 A0-A8 已全部完成并独立审计；
6. 42-cell、36 supported/6 unsupported 的 release candidate、单 GPU lane、双签
   authorization 和 ORR `GO` 已冻结；
7. 540/540 protected jobs sealed complete，零失败、零无效、零缺失和零技术失败；
8. 双签 unblind 与 aggregate 已完成，得到 22 ready-with-note、12 rejected、
   2 overlap-only 和 6 unsupported 终态。REVE MI/MA overlap track 与 STA-Net
   method-native context track 仍保持单独标签。

完整数值和证据哈希见
[`docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md`](../docs/comparisons/PROTECTED_CAMPAIGN_RESULTS_20260814.md)。
本轮完成不授权再次执行；任何新 formal/protected run 必须使用新 candidate 和新双签。
