# E1 阶段 tokenizer 优化、码本健康度、性能与生理耦合痕迹报告

_日期：2026-07-22 · 证据级别：训练/验证集回顾性审计 · 保护测试集：未打开 · 固定码本：EEG K=128，fNIRS K=128_

![图形摘要：24 个 E1 运行经过抗塌缩修复、健康度审计与跨模态痕迹分析。](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/00_graphical_abstract.png)

**图形摘要。** 本报告汇总 24 个主线 E1 运行，对全部可用 checkpoint 检查码本占用和内部几何，并在 8 个关键谱系运行上重放验证集 hard token。G1 的占用与停止复活后的保留规则通过，但最终码本仍有显著谱集中和近重复原型；最终三种子的 EEG–fNIRS token 关联不能区别于被试内且标签内的置换零分布，因此不构成生理耦合证书。[SVG](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/00_graphical_abstract.svg) · [PDF](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/00_graphical_abstract.pdf)

## 摘要

E1 的原始问题是固定 K=128 的 EEG 和 fNIRS 码本在短程 teacher-free 重建训练中严重塌缩。项目随后依次恢复 hard/annealed-hard 重建、K-means 初始化、余弦分配、潜变量 L2 归一化、可达梯度的码本平衡、EMA 老化、有限死码复活、模态温度和复活停止规则，并以三种子保留门完成 G1 判定。本报告补齐此前缺失的可视化，并把“占用了多少码字”“码字是否均匀使用”“原型是否真正分离”“优化是否依赖持续复活”“EEG 与 fNIRS token 是否存在可泛化关联”分开检验。

最终 diverse-farthest/T2-T2 三种子的验证末轮平均活跃码字为 EEG 86.33±2.08、fNIRS 110.00±1.73，有效码字为 EEG 65.85±1.66、fNIRS 39.99±1.38，最佳验证损失为 1.5826±0.0087。复活在 step 200 后保持冻结，三种子均满足既定有效码字和活跃比例门槛，因此历史 G1 占用/保留结论仍成立。新增几何审计同时发现：三种子码本的代数矩阵秩均为 64，但中心化奇异谱 participation rank 只有 EEG 1.92±0.12 和 fNIRS 2.45±0.18；PC1 单独解释 EEG 67.8%–73.2%、fNIRS 52.2%–58.8% 的码本方差，每个码本仍含 40–90 对余弦相似度不低于 0.99 的近重复原型。因此，G1 应准确表述为“固定 K 下的占用与复活后保留通过”，不应扩展为“64 维原型空间已经各向同性或充分解耦”。

跨模态审计使用每个最终种子 300 个验证窗口、5 名验证被试和 1,777 对同时有效的 lag-0 token，比较原始归一化互信息（normalized mutual information, NMI）、被试留一条件预测和 200 次被试内且标签内 EEG token 置换。最终三种子的 lag-0 NMI 为 0.2405–0.2598，但相对零分布的超额仅为 0.0011–0.0027，经验置换 p 值为 0.378、0.254 和 0.189；留一被试条件预测相对 fNIRS 边际基线的准确率增益全部为负（−0.0231、−0.0174、−0.0129）。在 −8 至 +8 秒的九个预设滞后上，三种子最小的未校正 p 值仍为 0.070、0.095 和 0.189，且没有正的留一被试增益。E1 因而没有提供可复现的 EEG–fNIRS 生理耦合证据；可见的原始共现主要随码本占用丰富度上升，并未转化为跨被试预测信息。

## 1. 审计范围与问题定义

本报告以 [E1 占用率汇总 v6](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_occupancy_comparison_v6/summary.json) 为主索引，覆盖从 v2 塌缩基线到 v23 三种子确认的 24 个主线运行。早期 2026-07-03 软件烟雾和恢复运行不进入横向科学比较，因为它们只验证训练循环和 checkpoint 恢复。所有健康度与性能数据来自训练/验证产物；跨模态分析只重放 `val` loader，未迭代 `test` loader，也没有改变既有 gate 决策文件。

这里的“性能”限于 E1 自身能够回答的重建/VQ 验证目标、码本占用、稳定性和几何质量，不等同于 E2 语义、E6 信息保留、下游任务性能或 E9 独立耦合证书。这里的“耦合痕迹”指配对窗口内 EEG token 与 fNIRS token 的统计关联；只有在控制被试、标签、边际占用和历史后仍能在新被试上改善预测，才可能升级为后续生理耦合证据。

## 2. 项目此前如何可视化码本

历史 source/observation tokenizer 已经具备较完整的诊断体系，但这些工具在 2026-07-01 架构迁移时被隔离到 compatibility 命名空间，当前 E1 trainer 只输出 JSON/JSONL 数值。历史工具主要由 [`source_observation_analysis.py`](../../../src/compatibility/pre_physiology_semantic_20260701/visualization/source_observation_analysis.py) 和 [`source_observation_token_sequence.py`](../../../src/compatibility/pre_physiology_semantic_20260701/visualization/source_observation_token_sequence.py) 提供，归档运行中仍保留实际 PNG 产物。

| 历史可视化 | 观察对象 | 能回答的问题 | E1 原有缺口 |
|---|---|---|---|
| 码字使用四联图 | 排序计数、累积质量/Gini、活跃码计数分布、统计摘要 | 是否死码、头部集中、少数码字垄断 | E1 只有 active/effective 标量，没有每码字图 |
| token 热图与时序模式 | 样本×时间 hard ID、原信号与 token 序列 | token 是否只跟随样本/任务，是否存在持续或跳变模式 | E1 没有验证 token 导出和时序图 |
| 转移矩阵 | 同模态 token_t→token_t+1 | 时序语法是否退化为自环或少数转移 | E1 未检查 |
| token histogram PCA | 按任务、被试着色的序列直方图 | 表示是否主要编码任务或被试来源 | E1 未检查来源混杂 |
| 跨模态共现热图 | lag 0/1/2 的 EEG→fNIRS token 联合矩阵 | 是否存在表面配对结构 | E1 未输出 |
| 滞后平衡经验耦合图 | MI、被试留一预测、被试内置换、滞后切片 | 关联是否超过边际/被试构成并可泛化 | E1 未输出 |
| 条件、任务、来源分层图 | source/task/context 分层 coupling | 结构是否由数据源或任务频率驱动 | E1 未输出 |
| 耦合 tensor/结构图 | 行熵、top-1 质量、lag 张量与条件诊断 | 学得的耦合参数是否有结构、是否可辨识 | 当前 E1 架构没有耦合参数，不能直接复用 |
| 重建时域/频域图与 gate dashboard | 原始/重建、频谱、损失与 gate | 码本健康改善是否牺牲信号保真 | E1 只有损失曲线和汇总数值 |

历史体系的优点是把占用、序列、重建和跨模态关系放在同一分析套件中；其不足是早期共现图常直接显示原始联合频率，容易把稀疏占用、被试构成或任务标签造成的结构误读为生理耦合。本次 E1 补图保留前者的覆盖范围，同时把被试留一和分层置换作为跨模态图的必要对照。

## 3. E1 抗塌缩优化链

E1 的变化不是某一个超参数独立解决了塌缩，而是实现缺陷修复、量化几何、训练路径和运行时复活共同形成的组合。详细代码变更见 [E1 码本占用契约恢复记录](../../architecture_changelog/2026-07-20_e1_codebook_occupancy_contract_restoration.md)，完整运行登记见 [实验日志](../06_EXPERIMENT_LOG.md)。

| 阶段 | 关键运行 | EEG active/effective | fNIRS active/effective | 最佳验证损失 | 结论 |
|---|---|---:|---:|---:|---|
| 塌缩基线 | v2 expected/no balance | 3 / 2.16 | 12 / 4.50 | 1.6960 | 两模态均塌缩 |
| hard 路径 | v4 | 2 / 2.00 | 16 / 6.00 | 1.4134 | 重建损失降低，但占用未恢复 |
| 余弦+K-means | v8 | 3 / 2.41 | 120 / 80.95 | 2.0269 | fNIRS 恢复而 EEG 仍塌缩，强烈模态不对称 |
| 平衡梯度修复 | v12 | 58 / 6.81 | 11 / 4.17 | 1.8397 | EEG 覆盖变宽但质量高度偏斜，fNIRS 未恢复 |
| 完整复活 bundle | v14 | 71 / 55.32 | 121 / 42.89 | 1.8326 | 两模态首次同时恢复广覆盖，但依赖 160/82 次复活 |
| T2/T2 平衡 | v17 | 90 / 66.54 | 116 / 45.65 | 1.8217 | 八轮候选中联合健康最好，仍有 160/100 次复活 |
| top-error 保留 | v20–v21 | 均满足 EEG 门 | 一种子最小 effective=21.18 | 1.5726–1.5898 | step 200 后不再复活，但三种子门失败 |
| diverse-farthest 确认 | v22–v23 | 86.33±2.08 / 65.85±1.66 | 110.00±1.73 / 39.99±1.38 | 1.5826±0.0087 | 三种子占用/保留门通过 |

![E1 24 个主线运行的覆盖、有效使用、均匀度与复活依赖。](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/02_e1_health_landscape.png)

**图 1｜E1 消融全景。** active code 只要求某码字在验证轮出现一次，effective code 为赋值熵的指数，二者之比反映活跃码内部均匀度。v8/v9 的 fNIRS 与 EEG 分化说明单纯几何修改不能保证两模态共同健康；v14 之后的阶跃来自复活 bundle，而非平滑的无复活优化。[SVG](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/02_e1_health_landscape.svg) · [数据](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/tables/e1_run_metrics.csv)

图 1 显示，hard reconstruction 曾把最佳验证损失降至 1.3956，但 EEG 只剩约 2 个有效码字。因此，仅以重建损失选模型会偏好离散瓶颈失效的解。反过来，v8 的 fNIRS 有效码字达到 80.95，却伴随 2.0269 的较差验证损失和 EEG 塌缩。占用健康和重建性能是两个不同目标，必须联合报告。

![有效码字与验证损失的权衡，点面积表示复活次数。](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/03_health_performance_tradeoff.png)

**图 2｜健康度—性能权衡。** 八轮运行之间不存在“有效码字越多、重建损失越低”的单调关系；14 轮保留运行的更低损失也不能与八轮消融做严格同预算比较。点面积随累计复活增大。[PDF](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/03_health_performance_tradeoff.pdf)

## 4. 复活停止后的保留

真正区分“临时铺满码本”和“训练后稳定使用”的证据来自 step 200 后的保留窗口。top-error 与 diverse-farthest 运行在 step 200 后的累计复活数均保持常数；EEG 有效码字继续上升，说明后半程的占用并非由持续注入新原型维持。top-error 的 seed 20260721 fNIRS 在第一次停止后验证降到 21.18，低于预注册下限 24，随后恢复到 32.08。把复活候选改为 diverse-farthest 后，同一种子的最低值提高到 24.80，另外两个确认种子的最低值分别为 29.64 和 29.63，最终三种子全部通过既定门。

![六个保留运行的 EEG 和 fNIRS effective-code 轨迹。](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/04_post_revival_retention.png)

**图 3｜复活后保留。** 虚线表示 top-error，实线表示 diverse-farthest；黑色竖线为复活停止 step 200，水平线为模态门槛。最终结论依赖“停止后至少七个验证轮保持复活常数”而不是最终单点占用。[SVG](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/04_post_revival_retention.svg) · [轨迹数据](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/tables/e1_health_trajectories.csv)

## 5. 新增内部码字几何审计

现有 gate 的 `effective_rank` 名称容易造成误解。实现实际调用 `torch.linalg.matrix_rank(codebook)`，它回答 128×64 矩阵是否在数值阈值下满列秩，却不回答方差是否均匀分布到 64 个方向。现有 `nearest_neighbor_cosine` 也取 128 个码字各自最近邻相似度的均值，而不是最大值。最终三种子的平均最近邻余弦确实低于既定 0.99 上限，但 95 分位和最大值均高于 0.99。

新增 checkpoint 审计显示，最终 EEG 码本有 43–90 对、fNIRS 码本有 40–84 对余弦不低于 0.99 的近重复原型。中心化码本的 PC1 已解释 EEG 67.8%–73.2% 和 fNIRS 52.2%–58.8% 的方差，前两个主成分合计解释 81.4%–82.0% 与 85.7%–90.1%。对应 participation rank 仅为 EEG 1.82–2.06、fNIRS 2.25–2.58；奇异谱熵秩也只有 EEG 3.34–3.55、fNIRS 3.23–3.85。换言之，码字 ID 的使用已经不塌缩，但原型云仍主要沿少数轴展开。

![最终三种子码本 PCA，点大小和颜色表示 EMA 占用。](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/05_final_codebook_pca.png)

**图 4｜最终码本原型分布。** EEG 原型形成弧形低维流形，fNIRS 原型形成多个高密度弧段；高占用原型并未均匀填充 64 维空间。PCA 仅用于描述单个模态内部几何，不把 EEG 与 fNIRS 的坐标轴直接对齐。[PDF](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/05_final_codebook_pca.pdf)

![最终三种子的最近邻余弦、近重复对、代数秩与谱集中。](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/06_codebook_geometry_diagnostics.png)

**图 5｜占用门没有覆盖的几何风险。** 面板 A 的柱为平均最近邻余弦，菱形和叉号分别为 95 分位和最大值；面板 C 对比满代数秩与谱 participation rank。结果不推翻历史占用 gate，但要求将其声明限定为 occupancy/retention pass。[SVG](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/06_codebook_geometry_diagnostics.svg) · [几何数据](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/tables/e1_checkpoint_geometry.csv)

EEG 与 fNIRS 两个分支虽都输出 64 维向量，但各自有独立 encoder、semantic head、EMA codebook 和 decoder，E1 又没有跨模态对齐损失。两个坐标系因此可以独立旋转或改变基底而不改变各自重建；直接画 EEG codebook 与 fNIRS codebook 的逐 ID 余弦或把两个 PCA 云叠加，没有可识别的生理含义。本报告只比较各模态内部几何，并把跨模态问题放到配对 token 的统计预测中。

## 6. 输入归一化与模态不对称

当前 canonical 输入是全记录、逐通道 median/MAD 稳健标准化，不是归档模型的逐 crop、逐成分 mean/std 标准化。训练集审计包含 1,080 个窗口和 18 名被试。EEG channel SD 的 q05/q50/q95 为 0.680/0.962/1.220，仅 0.494% 的通道窗口落在 [0.5,2] 外；fNIRS 对应值为 0.314/0.665/1.306，超范围比例为 27.315%。fNIRS 的窗口内 SD 比 q50/q95 达到 1.391/2.935，高于 EEG 的 1.131/1.409。

![EEG 与 fNIRS 训练窗口残余尺度异质性。](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/07_input_normalization_audit.png)

**图 6｜输入归一化审计。** 当前标准化确实限制了全记录尺度，但 fNIRS 仍保留显著更大的窗口级均值和方差异质性。这可以解释部分模态温度和复活敏感性，但不能单凭此图授权逐窗口标准化，因为后者可能移除具有生理意义的幅度信息。[数据来源](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_training_input_normalization_audit_v1/summary.json)

## 7. EEG–fNIRS 码本是否存在耦合

跨模态分析选择 v2、v12、v14、v17、top-error 失败种子及 diverse-farthest 三个最终种子，覆盖塌缩、梯度修复、复活、温度和最终确认谱系。每个运行从 `last.pt` 重放同一验证集，只保存 hard ID、有效 mask、被试和标签。正滞后定义为 EEG 领先 fNIRS；滞后范围为 −4 至 +4 token，即 −8 至 +8 秒。对每个滞后计算联合计数、NMI、同数据条件预测、留一被试条件预测，以及 200 次在“被试×标签”组内打乱 EEG token 的零分布。组内置换保留每个被试和标签的 EEG/fNIRS 边际占用，因此比原始共现热图更能排除 prevalence 假结构。

![八个关键运行的 lag-0 NMI、留一被试增益和零分布超额。](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/08_cross_modal_coupling_summary.png)

**图 7｜lag-0 跨模态痕迹。** v14 以后原始 NMI 大幅增高，主要因为两个码本同时从塌缩变为丰富占用；其置换零分布也同步升高。最终三个 diverse-farthest 种子的超额 NMI 很小，且留一被试条件预测均劣于只使用训练被试 fNIRS 边际的基线。[SVG](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/08_cross_modal_coupling_summary.svg) · [完整滞后数据](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/tables/e1_coupling_lag_metrics.csv)

v12 在 lag 0 上出现了 NMI 超额 0.0055、经验 p=0.030 和正的留一被试增益 0.0304，但这个信号没有随更健康的码本稳定保留。v17 和 top-error seed 20260721 虽有未校正 p≈0.03–0.04 的 NMI 超额，留一被试增益已为负；最终三种子 lag 0 的经验 p 均不显著。该谱系不支持“健康度越高，跨模态生理耦合越强”的单调叙事。

![八个关键运行在 −8 至 +8 秒的 NMI 超额和留一被试预测增益。](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/09_cross_modal_lag_profiles.png)

**图 8｜滞后曲线。** 最终种子在某些正滞后上出现小的 NMI 峰，但没有跨种子共同峰，也没有正的留一被试增益。九个滞后尚未做 family-wise 校正；即使使用未校正经验 p，最终种子的最小值也只有 0.070、0.095 和 0.189。[PDF](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/09_cross_modal_lag_profiles.pdf)

![最终三种子 top-64 码字的 lag-0 联合富集图。](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/figures/10_final_gate_joint_enrichment.png)

**图 9｜最终码字对富集。** 颜色表示平滑后的 `log2(observed/independent)`，每个种子都按自身验证占用排序，不能把相同坐标解释为跨种子同一生理状态。热图存在视觉结构，但图 7–8 的分层零分布和留一被试结果表明，该结构尚未形成可泛化耦合证据。[稀疏联合计数](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/tables/joint_counts/)

## 8. 综合解释与结论边界

第一，E1 的抗塌缩工程是有效的。相比 v2，最终候选在不缩小 K 的前提下把 EEG/fNIRS 有效码字从 2.16/4.50 提升到三种子均值 65.85/39.99，并在停止复活后保持至少八个验证轮。diverse-farthest 的价值不是让末轮平均数最大化，而是修复 top-error 在失败种子上的 fNIRS 保留下界，使三种子均满足预先冻结的门。

第二，当前 G1 的名称和解释应进一步精确化。它证明的是 assignment occupancy、复活冻结和种子级保留，不是原型几何充分展开。满 `matrix_rank=64` 与低 participation rank 可以同时成立；平均最近邻余弦低于 0.99 也可与大量局部近重复码字同时成立。未来若要把“码本健康”用于表示容量声明，应在新版本 metric registry 中增加谱熵秩/participation rank、最近邻分位数、余弦≥0.99 的重复对计数，以及 top-k 占用质量，并预注册阈值后重新校准，而不能追溯性修改已完成的 G1。

第三，E1 没有证实 EEG 与 fNIRS 码本耦合。最终码本的原始 NMI 较高只是丰富离散变量的预期现象；在保留被试和标签边际的置换对照下，超额接近零，在留一被试上更没有预测改善。由于 E1 两分支独立训练且不含跨模态目标，这一负结果在机制上并不意外。它也不等于“神经血管耦合不存在”，只说明当前 teacher-free E1 hard token、当前验证样本和这组简单条件计数没有保留可复现的跨被试增量痕迹。

第四，图 7–9 不能替代 E7–E9。E7 需要在冻结 tokenizer 的前提下测试耦合信息是否被保留；E8 需要匹配的 `q0/q1` 时序模型；E9 需要新评估器、被试级不确定性、fNIRS 历史、来源/任务/边际控制、时间移位与完整 null family，并在跨种子原型签名匹配后检验耦合图稳定性。本报告的共现分析最多是开发阶段的负向筛查。

## 9. 局限性

跨模态审计是事后新增的诊断，不是 E1 原始预注册 endpoint；p 值仅为 200 次 Monte Carlo 置换的经验分辨率，未对运行和滞后多重性校正。每个最终运行只有 5 名验证被试，1,777 对 lag-0 有效 token 不能被当作 1,777 个独立生物学重复。条件计数模型没有显式使用 fNIRS 历史、窗口位置或连续生理量，因此其负结果不能回答完整的 E9 问题。

checkpoint 几何使用 `last.pt` 以匹配最终保留轨迹，而“best validation”来自训练期间最佳 checkpoint；二者服务不同问题，不能混作同一时间点。EMA count 反映训练历史加权占用，验证 effective code 反映当前验证 hard assignment；二者在图中明确分开。PCA 和谱秩描述码本本身，不证明码字具有可解释的生理语义。

## 10. 最终结论

E1 可以继续保持“G1 占用/保留通过”的项目决策：固定 K=128 的 diverse-farthest/T2-T2 候选在三种子上停止复活后仍维持既定有效使用和活跃比例，且保护测试集未打开。不过，完整可视化揭示出两个此前汇总表无法看到的限制。其一，最终原型空间仍高度低维和局部冗余，因此“码本内部几何全面健康”尚未成立；其二，最终 token 的 EEG–fNIRS 共现没有超过分层零分布，也不能改善留一被试预测，因此 E1 不支持生理耦合声明。

最稳妥的后续顺序是：保留当前 tokenizer 作为 G2/G3/E6 的固定占用候选；把几何指标作为新诊断而非追溯性 gate；在 E7 先验证连续/soft/hard 表示是否保留可预测的跨模态信息；只有 E7 通过后再进入 E8/E9 的独立证书。若后续语义或信息保留失败，应优先处理原型谱集中和 fNIRS 尺度异质性，而不是仅继续提高 active-code 数。

## 11. 可复现产物

分析入口为 [`analyze_e1_tokenizer_health_and_coupling.py`](../../../experiments/analyze_e1_tokenizer_health_and_coupling.py)，统计定义测试为 [`test_e1_tokenizer_health_and_coupling.py`](../../../tests/test_e1_tokenizer_health_and_coupling.py)。完整机器可读摘要见 [`summary.json`](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/summary.json)，产物和源哈希见 [`manifest.json`](../../../experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1/manifest.json)。所有图均提供 SVG、PDF 和 300-DPI PNG；表格目录包含运行级健康度、逐轮轨迹、checkpoint 几何、逐滞后耦合指标和非零联合计数。

```bash
.venv/bin/python experiments/analyze_e1_tokenizer_health_and_coupling.py \
  --summary experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_occupancy_comparison_v6/summary.json \
  --run-root experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness \
  --input-normalization-audit experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260720_e1_training_input_normalization_audit_v1/summary.json \
  --output-dir experiments/runs/physiology_semantic_tokenizer/e1_quantizer_correctness/20260722_e1_health_coupling_visual_report_v1 \
  --device cuda:0 --permutations 200
```

_本报告不引用外部文献；所有结论均来自本仓库的版本化实现、训练/验证产物和本次可复算审计。_
