# 项目架构与研究演进图谱

_审计快照：`main@55cee3f`，覆盖 2025-12-02 至 2026-07-23 的全部 303 个可达 Git 提交，并区分已提交事实、当前工作区草案与未来计划_

---

## 📋 阅读边界

这张图不是当前模型的一张静态结构图，而是项目的“研究程序地图”。它同时回答四个问题：

1. 数据处理与统一化、对比方法复现、理论讨论、自有模型架构与实验各自经历了哪些版本；
2. 哪些节点是历史版本、当前事实、阻塞状态或未来计划；
3. 哪一条证据改变了另一条主线；
4. 下一步为什么必须按当前顺序执行。

版本节点按“改变研究问题、数据契约、模型职责或证据门禁的语义版本”聚合，而不是把 303 个提交逐个画成节点。每个聚合节点都在下方台账中给出代表提交和权威文档。仓库没有 Git tag；`D*`、`C*`、`T*`、`M*` 是本图的稳定索引，不是新增的软件发布号。

| 视觉状态 | 含义 |
| --- | --- |
| 蓝色 | 已完成或已合并的历史版本 |
| 绿色 | 当前仍有效的契约或已经通过的当前门禁 |
| 黄色 | 正在进行、结果未闭合或仅有开发证据 |
| 红色 | 被证据明确阻塞 |
| 紫色虚线 | 已写入权威计划、尚未执行的未来阶段 |

## 🗺️ 单图全景

[![PID-MCM 项目四线架构与研究演进全图](figures/project_evolution_map.svg)](figures/project_evolution_map.svg)

_图 1｜原生 SVG 全图。实线是同一主线的版本继承，`F1`–`F19` 虚线是跨主线影响；点击图可打开完整尺寸。SVG 内保留可编辑文本、节点 ID、状态、无障碍标题/描述和完整因果索引。_

> 📌 **当前关键路径：** `M12 → C5`。统一 EEG/QC 契约、adaptive teacher 重建与符号校准已经完成；完整 E0 已通过，SSM 生理信息完全可接受。后续信息保持、语义、foundation/certificate 与冻结对比测试仍按各自门禁推进。

## 📚 四条主线的版本台账

### 数据集处理与统一化

| 节点 | 版本变化 | 代表 Git 记录 | 当前含义 |
| --- | --- | --- | --- |
| `D0` | 从合成数据进入 BBCI EEG–fNIRS，建立过滤、EOG、HbO/HbR 与实验标准化入口 | `7e899b7`, `9205797`, `3ed5977` | 历史单数据集入口 |
| `D1` | 引入 registry/factory、同步数据适配器、session/event 对齐与 adaptive lag | `aaa5cb8..aba7bae` | 后续统一 loader 的源头 |
| `D2` | Croce 求解、event-relative cache、fNIRS 类型/单位、full-channel target 与存储布局统一 | `1cc60a1..ef05064` | 物理 teacher 的历史数据基础 |
| `D3` | 增加跨数据集 fNIRS 规范、HOMER2 cache、统一 event index 与训练 readiness 审计 | `434248e..19ee827` | 揭示数据分支间不可直接互换 |
| `D4` | 四原始数据集统一进入 `UnifiedPhysiologyWindowDataset`，改为 measurement-first | `3308257`, `220c04d` | 当前强制 measured-data 入口 |
| `D5` | Single-Trial EEG artifact clean v3 经 controlled-artifact 与 sham 验证接纳 | `67fb36c`, `97179ed` | 已提交默认清洗基线 |
| `D6` | 恢复/修正 Visual、REFED、DSR，统一通道、mask、geometry 与 task contract | `814c0e3`, `f9d8e26` | 比较与自有模型共享的数据边界 |
| `D7` | clean v4 加入 50 Hz 处理并取消数据集特有坏道动作 | [未提交的 v4 决策与全数据验证](project_changelog/2026-07-23_single_trial_line_noise_no_bad_mask_v4.md) | 全数据 gate 已通过，但提交前仍是工作区草案 |
| `D8` | 用最终 QC/channel contract 重建 teacher sidecar，并冻结 benchmark/splits | [E2 计划](physiology_semantic_tokenizer/analysis/20260722_E2_IMPLEMENTATION_AND_EXPERIMENT_PLAN.md)；[比较工作流](physiology_semantic_tokenizer/11_COMPARATIVE_METHOD_EXPERIMENT_WORKFLOW.md) | 当前最近的共同前置任务 |

### 对比方法复现

| 节点 | 版本变化 | 代表 Git 记录 | 当前含义 |
| --- | --- | --- | --- |
| `C0` | UMAP 从 baseline、可视化、subject-dependent/DE 到四轮总结 | `2906b63..7919437` | 当时最佳为 cross-subject DE/no-pretrain；暴露 subject shift |
| `C1` | 固定四数据集、七任务、paired supervised 与 pretrained-transfer 分轨，并定义 C0–C6 | `61f2c95` | 当前比较方法权威协议 |
| `C2` | STA-Net 独立 PyTorch FGSA/EGTA 实现，支持分类、DSR context 与 REFED regression | `f8615b5`, `ef2ba4b` | 七任务 smoke 已通，属于项目适配版 |
| `C3` | 增加多保真 tuning、best-checkpoint 选择、轨迹审计和双评估族 split registry | `d0219f1..f561104` | 开发阶段完成；尚非正式 protected 结果 |
| `C4` | EFRM 同步四数据集 ViT-base、可恢复训练、CLIP/检索审计与 detached launcher | `168880c..55cee3f` | epoch 8 显示 alignment warning；必须续跑至 epoch 20 决策门 |
| `C5` | 源协议 fidelity、C0–C5 冻结、within/cross-subject 双矩阵、C6 一次性 protected test | [比较工作流](physiology_semantic_tokenizer/11_COMPARATIVE_METHOD_EXPERIMENT_WORKFLOW.md) | 正式结果的唯一允许路径 |

### 自有模型理论讨论

| 节点 | 理论版本 | 代表 Git 记录/文档 | 被保留或否定的核心观点 |
| --- | --- | --- | --- |
| `T0` | PID/MCM 与 synthetic XOR | `544689d`, `dd3262a`, `66557ac` | 先在可控系统验证部分信息分解 |
| `T1` | tokenizer-first | `3e60bd1`; [连续数据 tokenizer](notes/continuous_data_tokenizer.md) | token 表示成为主任务，不再先做完整 PID 系统 |
| `T2` | 模态异质性、共享码本与时延对齐 | `e362123`, `c352da4`, `b3a5582` | EEG/fNIRS 不应按同时同尺度直接对齐 |
| `T3` | shared/private 因子化与残差 | `b61df3`, `c9eb679`, `633286b` | 尝试把共享信息和模态私有信息分开 |
| `T4` | 显式语义、source/observation 与 Croce 物理目标 | `d6d154a..5b4b77f` | 从“共享码本即语义”转向外部可解释 target |
| `T5` | 信息阶梯、局部耦合与 whole-brain 审计 | `1204f0f..5cf74fa`; [旧设计 postmortem](physiology_semantic_tokenizer/01_LEGACY_DESIGN_POSTMORTEM.md) | hard ID 丢信息；全局 coupling 不能替代任务局部证据；X3 会污染检验 |
| `T6` | physiology-semantic、measurement-first、physical teacher 与 entry routing | `b81c31b..d7255d9`; [理论基础](physiology_semantic_tokenizer/03_THEORETICAL_FOUNDATIONS.md) | sign-calibrated SSM teacher 已接纳；推理仍须模态独立 |
| `T7` | foundation discovery + fresh certificate | [目标架构](physiology_semantic_tokenizer/02_TARGET_ARCHITECTURE.md); [实现计划](physiology_semantic_tokenizer/04_IMPLEMENTATION_VALIDATION_PLAN.md) | 最终主张限定为：在 history/marginal/null 控制后，EEG 是否提供未来 fNIRS 的增量信息 |

### 自有模型架构与实验

| 节点 | 架构/实验版本 | 代表 Git 记录 | 门禁结果 |
| --- | --- | --- | --- |
| `M0` | synthetic PID/MCM 与 XOR time-series | `544689d..66557ac` | 早期可控性验证 |
| `M1` | 时域、频域、NeuroRVQ、VQNSP tokenizer 与 probe | `c6a5615..faf8a3c` | 多方案存在，但真实数据多次接近 chance |
| `M2` | shared-codebook contrastive alignment、warmup、lag-aware validation | `0fdcbc4..a2444a2` | 建立跨模态训练线，但未形成稳健语义 |
| `M3` | factorized shared/private、residual 与 multimodal auxiliary | `b61df3..cd48991` | 对齐更强，但尚未形成稳定语义 |
| `M4` | source/observation 全迁移 | `84299b4..ab7d7e0` | 旧 shared/private 语义从主线删除 |
| `M5` | Gate1 稳定化、dual decoder、Croce target/local cache、coupling suites | `3cc2724..5cf74fa` | 工程可运行；语义和局部 coupling 仍不足 |
| `M6` | physiology-semantic P1–P5、独立分支、修正 EMA VQ、trainer 与 export | `b81c31b..43bdef1` | 软件迁移完成，科学门未自动通过 |
| `M7` | E0/E0-v2 teacher validation | `2b4f3b3..0a38a7c` | 符号校准前历史诊断；旧 fNIRS 负标记不代表当前 E0 状态 |
| `M8` | adaptive shared-neural SSM、task parameter audit、gauge/sign correction、完整接纳 | `612e8c3..d7255d9` | 完整 E0 通过；physical teacher 与全部 SSM 生理信息（含 fNIRS）可接受 |
| `M9` | E1 quantizer v2–v23，最终 fixed K=128 diverse-farthest/T2-T2 | `0d00f28`, `7f1149c` | G1 仅在 occupancy/retention 意义上通过；跨模态 hard-token coupling 未证实 |
| `M10` | E2 T0/T1/T2、sidecar、entry masks、梯度审计与冻结 probe | `b4ffc82`; [E2 计划](physiology_semantic_tokenizer/analysis/20260722_E2_IMPLEMENTATION_AND_EXPERIMENT_PLAN.md) | 历史 channel/QC 冲突已由 v4 重建解决 |
| `M11` | channel-aware E0 重建、训练被试权重校准、9 个 E2 development jobs | [E2 计划](physiology_semantic_tokenizer/analysis/20260722_E2_IMPLEMENTATION_AND_EXPERIMENT_PLAN.md) | 已完成；完整 E0 通过，E2 保留 T0 |
| `M12` | E6/G2 信息保留、G3 语义、E7 coupling preservation、E8 foundation、E9 certificate | [实验设计](physiology_semantic_tokenizer/05_EXPERIMENT_DESIGN.md) | 只有前一门通过才进入后一门 |

## 🔗 跨主线影响证据

图中的实线只表示同一主线的版本继承；`F1`–`F19` 虚线表示跨主线影响。下表区分文档直接记录的因果和基于提交/实验顺序的审计综合。

| 影响 | 证据类型 | 证据到决策 |
| --- | --- | --- |
| `F1–F2` | 直接记录 | 项目在真实数据 tokenizer 多次接近 chance 后，从完整 PID 计划转为 tokenizer-first，并扩展 probe/可视化 |
| `F3` | 审计综合 | UMAP 表示按被试而非任务聚类、fusion 接近均匀，强化了模态异质性与 subject-invariance 问题，而不是证明 UMAP 架构无效 |
| `F4–F5` | 直接记录 | registry、session alignment 和 adaptive lag 使共享码本实验可执行；模态异质性分析又推动 shared/private 因子化 |
| `F6` | 直接记录 | 更多 alignment loss 没有解决码本语义与泛化，2026-04-08 后明确转向 codebook quality，随后改为显式 source/observation 语义 |
| `F7` | 直接记录 | Croce solver、event-relative cache 和单位统一把物理 target 从理论假设变成可训练 sidecar |
| `F8–F9` | 直接实验 | hard/quantized 表示丢失 LOSO 信息、全局 coupling 在任务局部失效、X3 直接交换污染检验，直接产生 2026-07-01 redesign |
| `F10` | 直接记录 | 四数据集原始测量审计否定“物理分解是必需输入”，架构改为 measurement-first、teacher optional |
| `F11` | 直接实验 | 旧 fNIRS 诊断促成 adaptive SSM；gauge/sign 修正消除坐标歧义并形成完整 E0 接纳 |
| `F12` | 直接审计 | 对比准备暴露 DSR 标签、Visual event/geometry、REFED continuous target 和 mask 消费缺口，反向修改统一 loader |
| `F13–F15` | 直接实验 | artifact/bad-channel 契约进入 measured-data 后，旧 E0 channel selection 只剩 93/230 target 可用，因此正式 E2 必须先回到数据/teacher 重建 |
| `F14` | 直接继承 | E1 的 fixed K=128 candidate 是 E2 的固定量化基础；E1 只通过占用，不把 hard-token 共现当作 coupling |
| `F16` | 计划约束 | 自有模型与 STA-Net/EFRM 必须共享 ordered sample IDs、split hashes、mask 与 target contract，才能进行同轨比较 |
| `F17` | 当前开发证据 | EFRM epoch-8 的同窗 CLIP 检索接近 chance，只能警告 exact-window identity objective 未激活；它不否定慢时延生理耦合，反而要求 history/lag 控制 |
| `F18–F19` | 理论到实验 | 因果主张边界决定 E7–E9 的顺序；只有独立冻结证书完成后，项目模型才可与比较方法报告匹配的下游效用 |

## 🧭 当前状态与未来计划

### 现在已经可以依赖

- 四数据集统一 measured-data 入口、20 秒默认上下文和独立 EEG/fNIRS mask
- physiology-semantic P1–P5 软件接口与 teacher-free runtime
- E1 fixed `K=128` quantizer 的三 seed occupancy/retention 结果
- STA-Net 七任务开发 smoke/tuning 工具链和 EFRM 同步数据训练/分析工具链
- protected subjects/tests 仍关闭这一证据边界

### 当前不能宣称

- T1/T2 比 T0 更有语义，或 G2/G3 已通过
- E1 hard tokens 已发现 EEG–fNIRS coupling
- STA-Net/EFRM 已完成源协议数值复现或正式 protected 比较
- E0 通过本身证明 SSM 参数唯一可辨识、因果方向成立，或某个 EEG token 导致某个 fNIRS token

### 推荐执行顺序

1. 以已通过完整 E0 的 sign-calibrated adaptive SSM 作为 physical teacher
2. 按 E2 结果处理语义目标与信息保持问题，不回退 E0 接纳
3. 通过 E6/G2 与 G3 后，依次执行 E7 preservation、E8 foundation 和 E9 fresh certificate
4. 并行完成 comparison 源协议复现与 C0–C5 冻结；只在自有模型和比较方法都冻结后执行 C6

## 🧾 Git 分支与文档一致性审计

### 实际 Git 分支

`main` 含 294 个提交；`--all` 共 303 个唯一提交。五个未合并远端尖端都停留在 2026-03 的 alignment/factorization 阶段，现已落后 `main` 178–205 个提交。它们应作为历史实验分支读取，而不应与图中的四条研究主线等同。

| 远端分支 | 相对共同祖先新增 | 相对 `main` 落后 | 处理解释 |
| --- | ---: | ---: | --- |
| `copilot/add-first-execution-stage-eeg-fnirs` | 1 | 199 | 早期自动执行计划，后续主线已重写 |
| `copilot/implement-eeg-fnirs-alignment` | 3 | 199 | warm-start/lag-aware 原型；概念已进入后续主线 |
| `copilot/monitor-alignment-experiment` | 1 | 198 | 监控计划分支，无当前权威性 |
| `copilot/monitor-labram-tokenizer-tests` | 2 | 205 | shared-codebook 原型；已被后续模型谱系取代 |
| `feature/private-factorization-tokenizer` | 2 | 178 | factorized tokenizer 原型；主线存在后续等价实现 |

### 已发现的文档漂移

- 根 [`README.md`](../README.md) 仍写着“target code 尚未实现”，但 2026-07-02 以后 P1–P5、trainer、E1 和 E2 软件已经落地
- [`architecture_changelog/INDEX.md`](architecture_changelog/INDEX.md) 同时使用 `Merged`、`In Progress` 和 `Complete — G1 passed`，但约定区只声明了前三阶段状态中的一部分
- 个别旧 architecture record 自身仍写 `Planned`/`Active`，而索引把对应谱系概括为已合并；阅读时应以更新的实验日志、当前实现和本图台账交叉判断
- 当前工作区的 clean v4 尚未提交，因此任何新 run 都必须记录 dirty-worktree flag，不能把 v4 结果归到 `main@55cee3f`

这些漂移说明旧 changelog 仍适合记录单次 before/after，但不足以承担跨数据、理论、模型和比较方法的全局导航。本图应在每次出现新的 `D/C/T/M` 语义版本或 `F` 跨线影响时更新。

## 🔍 审计来源

- 全部 Git 历史：`git log --all`，303 个唯一提交，时间范围 2025-12-02 至 2026-07-23
- [Architecture changelog](architecture_changelog/INDEX.md) 与 [project operations changelog](project_changelog/INDEX.md)
- [Physiology-semantic 文档入口](physiology_semantic_tokenizer/README.md)、[实验日志](physiology_semantic_tokenizer/06_EXPERIMENT_LOG.md) 与 `analysis/`
- [数据质量/HOMER2 审计](physiology_semantic_tokenizer/09_DATA_QUALITY_HOMER2_ALIGNMENT_AUDIT.md)
- [对比方法工作流](physiology_semantic_tokenizer/11_COMPARATIVE_METHOD_EXPERIMENT_WORKFLOW.md)
- [STA-Net PyTorch](../comparative_methods/STA-Net-PyTorch/README.md)、[EFRM PyTorch](../comparative_methods/EFRM-PyTorch/README.md) 与 [UMAP 总结](../comparative_methods/UMAP/EXPERIMENT_SUMMARY.md)

_最后审计：2026-07-23_
