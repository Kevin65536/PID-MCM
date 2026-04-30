# Neuro-Tokenization Implementation Plan

> Rewritten: 2026-04-30 | Last revised: 2026-04-30
> Status: Active mainline execution guide — branch semantics gate PASSED, entering Phase 1 implementation
> Detailed design rationale: [docs/PHYSIOLOGICAL_COUPLING_PLAN.md](docs/PHYSIOLOGICAL_COUPLING_PLAN.md) — Section 2 contains the complete Source/Observation redesign
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
3. temporal-smoothed common target for shared branch as the current working proxy **(待替换为 HRF model target)**
4. temporal residual target for private branches as the current working proxy **(待替换为 observation branch 隐式定义)**
5. codebook health regularization and branch responsibility separation

**⚠️ 分支语义 redesign 已完成**：shared/private 将被重命名为 source/observation，单一 shared quantizer 将被替换为双 source codebook + constrained coupling，HRF 卷积模型将替代 smooth_signal proxy。详见 [PHYSIOLOGICAL_COUPLING_PLAN.md Section 2](docs/PHYSIOLOGICAL_COUPLING_PLAN.md)。V6 仍然是当前运行的基线，但已被指定为 S1 对照，不再是最终目标架构。

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

当前阶段的核心目标：

1. ~~**先重新审阅 shared/private branch 是否必要，以及每个分支的语义是否足够明确**~~ ✅ **已完成** — S2 design selected
2. **实现 source/observation branch semantics redesign**（Phase 1-3），然后在明确的 branch semantics 上做 A/C 机制实验

### 5.1 Branch semantics review: conclusion ✅ DECIDED

经过对 V6 baseline、TokenFlow 论文、以及神经血管耦合生理模型的分析，结论如下：

1. ✅ 保留 factorization 这个大方向；
2. ✅ 重新定义 shared/private 的职责边界 —— 现在命名为 **source/observation**：
   - **source branch** = HRF-modeled neurovascular coupling state
   - **observation branch** = modality-specific reconstruction debt
3. ✅ 当前 `shared = smoothed common`、`private = temporal residual` 已被正式弃用；
4. ✅ 单一 shared quantizer 被替换为双 source codebook + constrained coupling matrix；
5. ⚠️ equal token count per window 当前保留为工作假设，延后至 Phase 5 审计。

### 5.2 Decision: S2 selected as new architecture target

在 S0 / S1 / S2 的比较中，**S2 (explicit-semantics factorized variant)** 被选定为新的架构目标：

| 结构 | 决策 | 理由 |
|------|------|------|
| S0: remove factorization | ❌ 不采用 | 不做 control experiment；factorization 必要性由 TokenFlow 分析和生理直觉支持 |
| S1: V6 factorized baseline | ⚠️ 保留为对照 | 作为 Phase 1-3 实验的 comparison baseline |
| **S2: explicit-semantics factorized** | ✅ **选定** | 完整设计规范见 PHYSIOLOGICAL_COUPLING_PLAN.md Section 2 |

S2 的核心变更：
1. shared/private → **source/observation**（命名 + 语义）
2. 单一 shared quantizer → **双独立 source codebook**（eeg_source + fnirs_source）
3. smooth_signal proxy → **HRF convolution model** 作为 source target
4. 自由 coupling → **concentration-constrained coupling**

### 5.3 Branch semantics exit gate ✅ PASSED

以下条件已满足，允许进入 A/C 机制实验（在 Phase 3 concentration baseline 完成后）：

1. ✅ S2 的设计规范明确（PHYSIOLOGICAL_COUPLING_PLAN.md Section 2）
2. ✅ source branch 语义 = HRF-modeled neurovascular coupling state（生理可解释，非低频重建捷径）
3. ✅ observation branch 语义 = modality-specific reconstruction debt（由 ablation gap 定义，非平滑残差桶）
4. ✅ 耦合结构从”事后统计量”升级为”concentration-constrained physiological mapping”
5. ✅ 损失函数精简（V6: 12 terms → S2: 9 terms）

**不再需要比较 S0 作为 gate condition**——S2 的设计已经充分论证了 factorization 的必要性和每个分支的独立生理语义。

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

### 6.3 Audit questions — all answered ✅

在进入 A/C 之前需要回答的四个问题：

1. ~~当前 shared branch 学到的是跨模态共性状态，还是仅仅是更容易重建的低频成分？~~ → **V6 shared branch 确实退化为低频重建捷径（smooth_signal proxy）。S2 redesign 用 HRF 物理模型替代。**
2. ~~当前 private branch 学到的是 modality-specific 信息，还是只是 shared 之外的剩余误差桶？~~ → **V6 private branch 是残差桶（raw - smoothed）。S2 redesign 用 reconstruction necessity 定义 observation branch。**
3. ~~`single shared quantizer` 是否是必要结构，还是当前 shared semantics 模糊的来源之一？~~ → **单一 shared quantizer 是 shared semantics 模糊的来源之一。S2 采用双 source codebook + constrained coupling。**
4. ~~`equal token count per window` 是否是科学假设，还是暂时的工程便利？~~ → **是工程便利。当前保留为工作假设，延后至 Phase 5 审计。**

所有四个 audit 问题已在 S2 redesign 中得到回答。

---

## 7. Workstream A: Coupling Smoothness

本 workstream 现在有两个前置条件：
1. ~~Section 5 的 branch semantics exit gate 通过~~ ✅
2. **Phase 3 (concentration baseline) 完成并通过 gate** ← 当前阻塞
3. 机制 A 不与机制 C 同时启用（文档要求）

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

本 workstream 现在有两个前置条件：
1. ~~Section 5 的 branch semantics exit gate 通过~~ ✅
2. **Phase 3 (concentration baseline) 完成并通过 gate** ← 当前阻塞
3. 机制 C 不与机制 A 同时启用（文档要求）

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

1. ~~维护 V6 baseline 作为固定对照~~ ✅ V6 baseline 维护中
2. ~~完成 shared/private branch semantics audit~~ ✅ **已完成** — 结论见 Section 5
3. ~~比较 S0 / S1 / S2 三类结构，决定是否保留当前 factorization 以及如何定义 shared branch~~ ✅ **已完成** — S2 selected
4. ~~选定通过 gate 的 branch semantics baseline~~ ✅ **已完成** — S2 design spec in PHYSIOLOGICAL_COUPLING_PLAN.md Section 2
5. **← 当前步骤**：实现 Phase 1: Structural Migration（拆分 quantizer、重命名、删除废弃 loss）
6. 实现 Phase 2: Source Target Introduction（HRF convolution model）
7. 实现 Phase 3: Concentration Prior（coupling row entropy）
8. 在 Phase 3 baseline 上实现并验证 Mechanism A (coupling smoothness)
9. 记录 A 的 scorecard 与实验结论
10. 回到 Phase 3 baseline，独立实现并验证 Mechanism C (causal asymmetry)
11. 记录 C 的 scorecard 与实验结论
12. 只有当 A 或 C 中至少一个独立通过后，才讨论 A + C 组合
13. tokenizer 证据充分后，才考虑 foundation model 层面的目标替换

**Phase 1-3 的具体实现步骤见 PHYSIOLOGICAL_COUPLING_PLAN.md Section 2.9。**

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

当前项目的 tokenizer 主线已经从”证明 token 条件概率可分析”切换到”设计一个带有生理结构先验的离散表示机制”。

**Branch semantics gate 已通过**：shared/private 已被重新定义为 **source/observation**。Source branch 编码 HRF-modeled neurovascular coupling state；observation branch 编码 modality-specific reconstruction debt。单一 shared quantizer 被双 source codebook + constrained coupling 替代。

**接下来的工作重点是**：

1. 将 V6 baseline 作为 S1 固定对照，保留不动；
2. 按 PHYSIOLOGICAL_COUPLING_PLAN.md Section 2.9 的 Phase 1-3 顺序实现 S2 architecture；
3. 在 Phase 3 concentration baseline 通过 gate 后，按顺序独立验证 Mechanism A 和 Mechanism C；
4. 只在 Layer A-D 评价闭环成立时推进默认主线。

**当前步骤**：Phase 1 — Structural Migration（拆分 dual source quantizer、重命名 shared→source/private→observation、删除废弃 loss terms）。

本文件即为当前实现顺序与准入标准的唯一主文档。