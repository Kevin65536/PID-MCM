# Neuro-Tokenization Implementation Plan

> Rewritten: 2026-04-30
> Status: Active mainline execution guide
> Detailed design rationale: [docs/PHYSIOLOGICAL_COUPLING_PLAN.md](docs/PHYSIOLOGICAL_COUPLING_PLAN.md)
> Archived reset foundation: [docs/archive/plans/NEXT_STAGE_ALIGNMENT_PLAN.md](docs/archive/plans/NEXT_STAGE_ALIGNMENT_PLAN.md)
> Evaluation scorecard: [docs/SEMANTIC_TOKEN_SCORECARD.md](docs/SEMANTIC_TOKEN_SCORECARD.md)
> Experiment log: [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)

---

## 1. Role of This Document

本文件是当前仓库的实现主文档。从现在开始：

1. **IMPLEMENTATION_PLAN.md** 负责回答“接下来先做什么、做到什么程度算通过、哪些内容明确延后”。
2. **docs/PHYSIOLOGICAL_COUPLING_PLAN.md** 负责回答“为什么这样做、机制的数学形式是什么、预期生理含义是什么”。
3. **docs/SEMANTIC_TOKEN_SCORECARD.md** 负责回答“如何评价 tokenizer 是否真的更好”。
4. **docs/EXPERIMENT_LOG.md** 负责记录正式实验结论。

如果多个文档之间出现实现顺序冲突，以本文件为准；如果是机制定义或数学细节冲突，以生理耦合计划为准。

---

## 2. Current Mainline Status

### 2.1 Current recommended baseline

当前默认主线是 **V6 codebook-focused factorized tokenizer**，参考实现为：

- [src/tokenizers/codebook_focus_factorized_labram_vqnsp.py](src/tokenizers/codebook_focus_factorized_labram_vqnsp.py)
- [src/tokenizers/factorized_labram_vqnsp.py](src/tokenizers/factorized_labram_vqnsp.py)
- [src/losses/multimodal_tokenizer.py](src/losses/multimodal_tokenizer.py)

该主线已经明确保留以下结构：

1. EEG / fNIRS shared-private factorization
2. lag-aware coupling for shared branch
3. temporal-smoothed common target for shared branch as the current working proxy
4. temporal residual target for private branches as the current working proxy
5. codebook health regularization and branch responsibility separation

### 2.2 What has changed in the innovation story

上一阶段的主要问题不是“模型不能工作”，而是“创新表达不够主动”。

此前的叙事是：

- tokenizer 学出 EEG 与 fNIRS token 之间的条件概率；
- 我们在事后分析这些条件概率，并据此提供可解释性。

这个叙事的问题在于，它更像分析结果，而不是机制设计。

新的主线改为：

- 在 coupling 参数本身加入生理指导的结构先验；
- 让“神经血管耦合原理”直接体现在离散表示学习机制里；
- 把创新从 post-hoc interpretability 转为 explicit structured alignment design。

### 2.3 What is no longer the main target

以下目标不再作为 tokenizer 主线的出发点：

1. shared token identity overlap 最大化
2. alignment-first 叙事
3. 通过让 shared branch 承担更强 raw reconstruction 来“逼出”跨模态共性
4. 只依赖分析指标来支撑创新陈述

---

## 3. Non-Negotiable Design Rules

任何后续实现都必须满足以下规则：

1. **生理先验只加在 coupling 参数上**
   - 不直接改 encoder / quantizer 的主梯度路径
   - 不把 reconstruction 训练变成先验对抗问题

2. **先验必须是软约束**
   - 小系数 regularizer
   - 允许数据覆盖先验
   - 禁止硬编码不可违背的波形或拓扑规则

3. **当前阶段不组合机制 A 与机制 C**
   - 先分别与 V6 baseline 比较
   - 只有单机制独立通过后，才讨论组合实验

4. **Layer A 不退化是硬门槛**
   - reconstruction / codebook health 退化，则该机制不能进入默认主线

5. **在新增 coupling 结构先验之前，必须先通过 branch semantics gate**
   - 如果 shared/private 的职责定义仍然模糊，不应继续给 shared branch 叠加新的生理先验
   - A/C 机制只应加在已经通过职责审查的 shared branch 上

---

## 4. Active Document Layout

当前文档分工固定如下：

| 类型 | 位置 | 用途 |
|------|------|------|
| 主实现计划 | `IMPLEMENTATION_PLAN.md` | 当前开发顺序、优先级、准入标准 |
| 活跃机制设计 | `docs/PHYSIOLOGICAL_COUPLING_PLAN.md` | 生理耦合约束的动机、公式、实验设计 |
| 活跃理论背景 | `docs/THEORY.md` | 总体理论框架与长期背景 |
| 活跃评价标准 | `docs/SEMANTIC_TOKEN_SCORECARD.md` | Layer A-D 评价框架 |
| 活跃实验记录 | `docs/EXPERIMENT_LOG.md` | 项目级实验结论 |
| 归档计划 | `docs/archive/plans/NEXT_STAGE_ALIGNMENT_PLAN.md` | V6 reset 的设计基础 |
| 归档实验日志 | `docs/archive/logs/ARCHIVED_PRE_EXPERIMENTS.md` | 第一轮预实验历史记录 |

顶层 `docs/` 只保留当前阅读时需要频繁访问的活跃文档；历史材料统一进入 `docs/archive/`。

---

## 5. Current Development Goal

当前阶段的核心目标分为两个顺序化步骤：

1. **先重新审阅 shared/private branch 是否必要，以及每个分支的语义是否足够明确**；
2. **只有在 branch semantics 明确后，才把 EEG-fNIRS coupling 从一个事后可分析的统计量，升级为一个带有明确生理结构先验的离散表示机制。**

基于 [docs/references/TokenFlow_analysis.md](docs/references/TokenFlow_analysis.md) 和当前实现状态，当前的暂定判断是：

1. **不应直接清除 shared/private factorization**；
2. **也不应把当前 shared-common / private-residual 代理目标当作已经定型的最终语义定义**；
3. **在 Mechanism A / C 之前，必须先完成一个 branch semantics decision gate**。

### 5.1 Branch semantics review: provisional conclusion

当前更合理的结论不是“回到单瓶颈”，而是：

1. 保留 factorization 这个大方向；
2. 重新定义 shared/private 的职责边界；
3. 把当前 `shared = smoothed common`、`private = temporal residual` 看作工作代理，而不是最终理论语义；
4. 把“single shared quantizer + equal token count per window”看作当前实现假设，而不是不可动摇的结构真理。

### 5.2 Decision options that must be compared before A/C

在接入新的生理 coupling 先验之前，当前主线先比较以下三类结构：

1. **S0: remove factorization control**
   - 单瓶颈或 shared-only 结构
   - 目的不是回归主线，而是验证 factorization 是否确实必要

2. **S1: current V6 factorized baseline**
   - 保留单 shared quantizer
   - 保留当前 common/residual 代理目标
   - 作为当前主线对照

3. **S2: explicit-semantics factorized variant**
   - 保留 shared/private 结构
   - 弱化“强共享 quantizer”假设，优先考虑 lightly tied shared codebooks 或显式 state mapping
   - 把 shared branch 更明确地定义为 cross-modal state branch，而不是简单的低频通道
   - 把 private branch 更明确地定义为 modality-specific reconstruction debt，而不是简单的平滑残差桶

### 5.3 Branch semantics exit gate

只有满足下面条件之一，才允许继续进入 A/C 机制实验：

1. S1 已证明 shared/private 职责边界清晰，且 shared branch 不是低频重建捷径；
2. S2 相比 S1 在 Layer B 或 branch responsibility gap 上更清晰，因此成为新的默认主线。

如果 S0 的结果与 S1/S2 相近，说明 factorization 的必要性仍未建立，此时不应直接继续做 coupling priors。

---

## 6. Workstream 0: Freeze and Audit the V6 Baseline

在实现任何新机制之前，先把 V6 baseline 作为固定对照面。

### 6.1 Baseline definition

当前 baseline 必须满足：

1. 使用 codebook-focused factorized tokenizer
2. 保留 lag-aware shared coupling
3. 不启用 coupling smoothness
4. 不启用 asymmetric coupling
5. 使用当前 canonical evaluation pipeline 输出 Layer A-D 指标
6. 保留 shared-only / private-only ablation 诊断

### 6.2 Required baseline artifacts

每次新机制实验都必须对齐以下 baseline 工件：

1. reconstruction metrics
2. codebook health metrics
3. shared-branch structure metrics
4. subject leakage / task signal diagnostics
5. coupling matrix visualization
6. shared-only vs private-only reconstruction diagnostics
7. branch responsibility gap summary

如果 baseline 工件不完整，不进入 A/C 机制比较。

### 6.3 Audit questions that must be answered first

在进入 A/C 之前，先回答下面四个问题：

1. 当前 shared branch 学到的是跨模态共性状态，还是仅仅是更容易重建的低频成分？
2. 当前 private branch 学到的是 modality-specific 信息，还是只是 shared 之外的剩余误差桶？
3. `single shared quantizer` 是否是必要结构，还是当前 shared semantics 模糊的来源之一？
4. `equal token count per window` 是否是科学假设，还是暂时的工程便利？

---

## 7. Workstream A: Coupling Smoothness

本 workstream 只有在 Section 5 的 branch semantics exit gate 通过后才进入实现。

### 7.1 Objective

为 forward coupling 矩阵加入局部平滑先验：相近的 shared EEG token 应映射到相近的 fNIRS token 分布。

### 7.2 Implementation scope

需要修改的主文件：

| 文件 | 变更 |
|------|------|
| `src/losses/multimodal_tokenizer.py` | 新增 `coupling_smoothness_loss()` |
| `src/losses/multimodal_tokenizer.py` | 在 `compute_factorized_shared_alignment_losses` 中返回 `smoothness_loss` |
| `src/tokenizers/factorized_labram_vqnsp.py` | 新增 smoothness 参数并接入 loss 汇总 |
| `src/tokenizers/codebook_focus_factorized_labram_vqnsp.py` | 透传新参数，默认关闭 |
| `experiments/configs/**` | 新增 A1 / A2 配置 |

### 7.3 Parameter contract

最小参数面如下：

```yaml
loss:
  alignment:
    coupling_smoothness_weight: 0.0
    coupling_smoothness_neighbors: 5
```

warm-start 调度属于训练策略，不是 tokenizer 本体结构。调度参数应放在训练配置或 trainer 逻辑中，而不是硬编码进模型架构：

```yaml
loss:
  alignment:
    coupling_smoothness_warmup_epochs: 30
    coupling_smoothness_final_weight: 0.01
```

### 7.4 Required diagnostics

机制 A 至少记录以下诊断：

1. `smoothness_loss`
2. 邻居 token vs 随机 token 的 coupling JS 散度差距
3. coupling row variance，防止所有行塌成同一分布
4. Layer A reconstruction/codebook health 是否稳定

### 7.5 Experiment queue

1. **Exp A1**: `coupling_smoothness_weight` sweep = 0.005 / 0.01 / 0.02
2. **Exp A2**: 在 V6 稳定后启用 warm-start smoothness

### 7.6 Pass / fail gate

- ✅ Pass: Layer A 不退化，且 coupling 结构更平滑，至少一项 Layer C 指标改善
- ❌ Fail: reconstruction / codebook health 退化，或 coupling 结构改善不可辨认

---

## 8. Workstream C: Causal Direction Asymmetry

本 workstream 只有在 Section 5 的 branch semantics exit gate 通过后才进入实现。

### 8.1 Objective

把“前向 EEG→fNIRS”和“反向 fNIRS→EEG”从共享一组参数改为独立参数化，让生理因果方向的不对称性可以通过参数化自由度体现出来。

### 8.2 Implementation scope

需要修改的主文件：

| 文件 | 变更 |
|------|------|
| `src/tokenizers/factorized_labram_vqnsp.py` | 新增 `coupling_asymmetric` 开关与 `coupling_logits_fwd/rev` |
| `src/losses/multimodal_tokenizer.py` | `compute_factorized_shared_alignment_losses` 支持 fwd/rev 独立 logits |
| `src/tokenizers/codebook_focus_factorized_labram_vqnsp.py` | 透传 asymmetric 参数，默认关闭 |
| `experiments/configs/**` | 新增 C1 配置 |

### 8.3 Current-stage constraint

当前阶段的 C 机制只做两件事：

1. 前向与反向 coupling 参数独立化
2. 记录不对称诊断指标

**不在 C1 中叠加 smoothness 正则。**

也就是说，虽然机制设计上允许“前向带结构先验、反向自由参数化”，但当前执行顺序仍然要求：

- A 单独验证
- C 单独验证
- A + C 组合延后

### 8.4 Required diagnostics

机制 C 至少记录以下诊断：

1. `asymmetry_ratio`
2. forward / reverse per-row entropy
3. 双向 coupling loss
4. forward vs reverse 的下游比较（若当前 probe 已具备）

### 8.5 Experiment queue

1. **Exp C1**: `coupling_asymmetric = true`, `coupling_bidirectional = true`

### 8.6 Pass / fail gate

- ✅ Pass: `asymmetry_ratio` 稳定大于 1，且 Layer A 不退化
- ⚠️ Inconclusive: `asymmetry_ratio` 接近 1，但 Layer A/C 没有倒退
- ❌ Fail: Layer A 退化，或反向路径明显失稳

---

## 9. Validation Standard

所有 A/C 实验统一按 Layer A-D 评价，不允许只看单一漂亮指标。

| Layer | 必看内容 | 当前要求 |
|------|----------|----------|
| Layer A | reconstruction, perplexity, utilization, dead-code behavior | 不退化 |
| Layer B | semantic consistency, branch responsibility, prototype separation | 不退化，最好改善 |
| Layer C | best-lag MI, conditional KL gain, coupling structure diagnostics | 至少一项明确改善 |
| Layer D | subject leakage, task signal, downstream sanity | 不明显倒退 |

### Promotion rule

任何机制要进入默认 mainline，必须同时满足：

1. Layer A 不退化
2. Layer B 不退化
3. Layer C 有明确增益
4. Layer D 不出现明显倒退
5. 能通过 ablation 解释，不把 shared branch 重新变成第二条全能重建捷径

---

## 10. Implementation Order

当前严格执行以下顺序：

1. 维护 V6 baseline 作为固定对照
2. 完成 shared/private branch semantics audit
3. 比较 S0 / S1 / S2 三类结构，决定是否保留当前 factorization 以及如何定义 shared branch
4. 选定通过 gate 的 branch semantics baseline
5. 在该 baseline 上实现并验证 Mechanism A
6. 记录 A 的 scorecard 与实验结论
7. 回到同一 baseline，独立实现并验证 Mechanism C
8. 记录 C 的 scorecard 与实验结论
9. 只有当 A 或 C 中至少一个独立通过后，才讨论 A + C 组合
10. tokenizer 证据充分后，才考虑 foundation model 层面的目标替换

任何跳步都意味着解释链断裂，不能作为主线证据。

---

## 11. Deliverables Required for Every Mainline Change

任何进入主线候选的改动都必须同时交付：

1. 默认关闭的向后兼容实现
2. 对应实验配置
3. 至少一份正式实验记录
4. Layer A-D scorecard 摘要
5. 必要的可视化或诊断图
6. 本文件中的状态更新

没有文档和评价闭环的改动，不视为主线推进。

---

## 12. Explicitly Deferred Work

以下内容当前明确延后，不进入这轮实现主线：

1. A 与 C 同时启用的联合实验
2. HRF-shaped lag weighting
3. 显式熵 margin 式 asymmetry loss
4. 重新引入 identity-style alignment losses
5. foundation model 预训练目标的大改
6. 把 shared branch 改回 full raw reconstruction 主通路

这些方向不是永久否定，而是必须等到 A/C 的单机制证据成立后再决定是否继续。

---

## 13. Bottom Line

当前项目的 tokenizer 主线已经从“证明 token 条件概率可分析”切换到“设计一个带有生理结构先验的离散表示机制”。

但在继续给 coupling 施加新的生理先验之前，当前更紧迫的问题是：shared/private 是否真的有清晰、可辩护的语义分工。

因此，接下来的工作重点不是直接进入 A/C，而是：

1. 以 V6 codebook-focused factorized tokenizer 为固定基线；
2. 先完成 shared/private branch semantics decision；
3. 只在通过该 gate 后，按顺序实现 coupling smoothness 与 causal asymmetry；
4. 只在 Layer A-D 评价闭环成立时推进默认主线。

本文件即为当前实现顺序与准入标准的唯一主文档。