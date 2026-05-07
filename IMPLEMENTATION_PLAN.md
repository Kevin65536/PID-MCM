# Neuro-Tokenization Implementation Plan

> Rewritten: 2026-04-30 | Last revised: 2026-05-06
> Status: Active mainline execution guide — direct migration to the Source/Observation architecture
> Detailed design rationale: [docs/PHYSIOLOGICAL_COUPLING_PLAN.md](docs/PHYSIOLOGICAL_COUPLING_PLAN.md) — Section 2 contains the complete Source/Observation redesign
> Archived reset foundation: [docs/archive/plans/NEXT_STAGE_ALIGNMENT_PLAN.md](docs/archive/plans/NEXT_STAGE_ALIGNMENT_PLAN.md)
> Evaluation scorecard: [docs/SEMANTIC_TOKEN_SCORECARD.md](docs/SEMANTIC_TOKEN_SCORECARD.md) — simplified to 4 evaluation gates (Health / Semantics / Structure / Utility)
> Experiment log: [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)

---

## 1. Role of This Document

本文件是当前仓库唯一的 tokenizer 主实现计划文档。从现在开始，它同时负责回答四类问题：

1. 接下来按什么顺序修改主线代码；
2. 哪些旧结构要直接删除，哪些机制要继续保留；
3. 分析工具需要怎么精简；
4. 历史实验如何归档，以及新实验结果如何标准化保存。

文档分工固定如下：

1. **IMPLEMENTATION_PLAN.md**：实现顺序、代码改造范围、分析与实验产物规范。
2. **docs/PHYSIOLOGICAL_COUPLING_PLAN.md**：机制动机、数学形式、生理解释。
3. **docs/SEMANTIC_TOKEN_SCORECARD.md**：4-gate 评价框架（Health / Semantics / Structure / Utility）。
4. **docs/EXPERIMENT_LOG.md**：正式实验结论。

如果多个文档之间出现冲突，以本文件的实现顺序为准；如果是机制定义或数学细节冲突，以生理耦合计划为准。

---

## 2. Current Mainline Status

### 2.1 Current starting surface

当前仓库的主实现面仍然落在以下文件上：

- [src/tokenizers/codebook_focus_factorized_labram_vqnsp.py](src/tokenizers/codebook_focus_factorized_labram_vqnsp.py)
- [src/tokenizers/factorized_labram_vqnsp.py](src/tokenizers/factorized_labram_vqnsp.py)
- [src/losses/multimodal_tokenizer.py](src/losses/multimodal_tokenizer.py)
- [src/visualization/factorized_alignment_analysis.py](src/visualization/factorized_alignment_analysis.py)

这些文件当前仍带有 shared/private 语义和相应的 smooth common / residual 训练代理。它们只是**当前需要被重构的实现表面**，不再被视为需要长期保留的对照架构。

### 2.2 Mainline decision after redesign

当前决策已经明确：

1. 主线架构从 shared/private 直接切换到 **source/observation**；
2. 单一 shared quantizer 直接切换到 **双 source codebook + constrained coupling**；
3. `smooth_signal` 代理直接退出主训练路径，由 **HRF convolution target** 接管；
4. private residual 监督直接退出主训练路径，observation branch 通过 reconstruction debt 隐式定义；
5. 旧 shared/private 代码不再作为仓库内长期对照面保留。

这意味着：**Source/Observation 不是旁路试验实现，而是对当前主线的直接升级。**

### 2.3 Control policy

本轮主线推进不再依赖“把旧 shared/private 代码继续留在仓库里”来形成对照。对照面改为以下两类：

1. **外部研究方法对照**：来自 [comparative_methods](comparative_methods) 和 [reference_repository](reference_repository) 的方法实现或复现结果；
2. **历史主线参考**：通过 git 历史、归档 run、归档实验记录回看 shared/private 阶段结果，而不是让旧代码继续留在活跃主线上。

### 2.4 What is no longer the main target

以下目标不再作为 tokenizer 主线的出发点：

1. shared token identity overlap 最大化；
2. alignment-first 叙事；
3. 通过让 shared branch 承担更强 raw reconstruction 来“逼出”跨模态共性；
4. 通过保留旧架构代码来维持所谓控制面；
5. 只依赖事后分析指标支撑创新陈述。

---

## 3. Non-Negotiable Design Rules

任何后续实现都必须满足以下规则：

1. **生理先验只加在 coupling 参数上**
    - 不直接改 encoder / quantizer 的主梯度路径；
    - 不把 reconstruction 训练变成先验对抗问题。

2. **先验必须是软约束**
    - 小系数 regularizer；
    - 允许数据覆盖先验；
    - 禁止硬编码不可违背的波形或拓扑规则。

3. **A 与 C 仍然分开验证**
    - 先有 Phase 3 concentration baseline；
    - 再分别验证 Mechanism A 与 Mechanism C；
    - 当前阶段不做 A + C 联合实验。

4. **Gate 1 (Health) 不退化是硬门槛**
    - reconstruction / codebook health 退化，则该机制不能进入默认主线。

5. **Branch semantics gate 已通过，但旧语义不应残留在活跃主线里**
    - shared/private 命名、旧 loss 名称、common/residual 代理指标都不应继续作为活跃实现的一部分；
    - 如果某个文件或接口继续用 shared/private 命名表达主语义，它就还没有完成迁移。

6. **历史可追溯性交给 git 与 archive，而不是活跃代码兼容层**
    - 不为了保留旧架构而维持并行类、并行 loss 路径或默认关闭的 legacy 分支。

---

## 4. Active Document Layout

当前活跃文档固定如下：

| 类型 | 位置 | 用途 |
|------|------|------|
| 主实现计划 | `IMPLEMENTATION_PLAN.md` | 开发顺序、文件改造、分析精简、归档与结果格式 |
| 活跃机制设计 | `docs/PHYSIOLOGICAL_COUPLING_PLAN.md` | 生理耦合约束的动机、公式、实验设计 |
| 活跃理论背景 | `docs/THEORY.md` | 总体理论框架与长期背景 |
| 活跃评价标准 | `docs/SEMANTIC_TOKEN_SCORECARD.md` | 4-gate 评价框架（Health / Semantics / Structure / Utility） |
| 活跃实验记录 | `docs/EXPERIMENT_LOG.md` | 项目级实验结论 |
| 归档计划 | `docs/archive/plans/NEXT_STAGE_ALIGNMENT_PLAN.md` | reset 阶段设计基础 |
| 归档实验日志 | `docs/archive/logs/ARCHIVED_PRE_EXPERIMENTS.md` | 第一轮预实验历史记录 |

顶层 [docs](docs) 只保留当前主线需要反复阅读的活跃文档；历史材料统一进入 [docs/archive](docs/archive)。

---

## 5. Current Development Goal

当前阶段的核心目标只有一个：

**直接把当前主线 tokenizer 从 shared/private 改造成 source/observation，并在同一条主线上依次完成 Structural Migration、HRF Source Target、Concentration Prior。**

### 5.1 Architecture decision

当前已经明确的架构结论：

1. 保留 factorization 大方向；
2. shared/private 直接改名并改义为 **source/observation**；
3. 单一 shared quantizer 直接替换为 **双独立 source codebook**；
4. smooth common proxy 直接替换为 **HRF convolution target**；
5. observation branch 不再接显式 residual target，而由 reconstruction necessity 定义；
6. coupling 从自由参数化升级为 **concentration-constrained physiological mapping**。

### 5.2 Mainline replacement policy

本轮实现采用**直接替换**而不是**并排新增**：

1. 主线 tokenizer、loss、config、analysis surface 都将直接改成 source/observation 语义；
2. 不保留 shared/private 活跃类、活跃 loss 汇总、活跃 config schema 作为兼容层；
3. 如果某个旧实现需要回看，依赖 git 与 archive，而不是仓库工作树中的 legacy 代码。

### 5.3 External comparison policy

正式实验比较不再以“旧 shared/private tokenizer 继续可运行”作为前提，而改为：

1. 采用 [comparative_methods](comparative_methods) 下的独立方法结果作为控制面；
2. 采用 [reference_repository](reference_repository) 中参考方法的复现/迁移结果作为控制面；
3. 所有对照方法统一输出到同一套 scorecard 与 summary schema 中，避免比较口径不一致。

---

## 6. Direct Mainline Replacement Plan

### 6.1 File-level migration scope

本轮不是在旧代码旁边加一套新实现，而是直接改造主线文件。计划如下：

| 文件/目录 | 处理方式 |
|------|------|
| `src/tokenizers/factorized_labram_vqnsp.py` | 直接替换为 source/observation 主实现，shared/private 逻辑退出主类 |
| `src/tokenizers/codebook_focus_factorized_labram_vqnsp.py` | 直接替换为 codebook-focused source/observation 主实现 |
| `src/losses/multimodal_tokenizer.py` | 删除 shared/private 专属 loss 汇总，只保留 source/observation 主线所需逻辑 |
| `src/tokenizers/registry.py` | 删除旧 shared/private config 解析与旧模型类型注册，切到 source/observation schema |
| `src/tokenizers/__init__.py` | 删除旧类导出，切换到新主类导出 |
| `src/visualization/factorized_alignment_analysis.py` | 改造或替换为 source/observation 对齐分析入口 |
| `src/visualization/semantic_space_analysis.py` | 移除 common/residual 代理指标，保留 Gate 1-4 所需主线指标 |
| `src/visualization/tokenizer_analysis_suite.py` | 作为唯一标准化分析入口继续保留 |
| `experiments/scripts/train_shared_tokenizer.py` | 切换到 source/observation 主线参数与产物协议 |
| `experiments/scripts/probe/*` | 清理 shared/private 旧语义依赖，只保留标准 rerun 入口 |
| `experiments/configs/**` | 清掉 shared/private 活跃配置面，建立新的 source/observation 配置簇 |

### 6.2 Phase 1: Structural Migration

目标：完成架构语义切换，但此阶段不引入 HRF target 和 concentration prior。

**需要落地的变更**：

1. `shared` 全面替换为 `source`；
2. `private` 全面替换为 `observation`；
3. 单一 shared quantizer 替换为 `eeg_source_quantizer` 与 `fnirs_source_quantizer`；
4. `eeg_private_quantizer` / `fnirs_private_quantizer` 替换为 `eeg_observation_quantizer` / `fnirs_observation_quantizer`；
5. 输出字典、分析键名、logger 指标键名、配置字段名同步改为 source/observation；
6. 删除以下旧 loss 项：
    - `latent_align_loss`
    - `assignment_align_loss`
    - `hard_assignment_align_loss`
    - `shared_entropy_loss`
    - `private_entropy_loss`
    - `shared_eeg_common_loss`
    - `shared_fnirs_common_loss`
    - `eeg_private_residual_loss`
    - `fnirs_private_residual_loss`
    - `shared_eeg_recon_loss`
    - `shared_fnirs_recon_loss`

**Phase 1 输出要求**：

1. active code 中不再出现 shared/private 作为主语义分支名称；
2. 主 tokenizer forward 只暴露 source/observation 结构；
3. reconstruction、codebook utilization、dead-code behavior 保持稳定；
4. observation branch 先只通过 full reconstruction 贡献与 orthogonality 约束定义，不重新引入显式 residual 目标。

### 6.3 Phase 2: Source Target Introduction

目标：用 HRF 卷积 target 替代 smooth proxy。

**需要落地的变更**：

1. 实现 double-gamma HRF kernel；
2. 从 EEG 侧生成 fNIRS source target；
3. 为 EEG source branch 保留弱辅助 target，防止 source collapse；
4. 在 trainer 中加入 source target warmup 调度；
5. 在 analysis suite 中加入 source target reconstruction 与 source codebook 健康诊断。

**Phase 2 输出要求**：

1. `source_target_loss` 可稳定下降；
2. fNIRS source codebook 不 collapse；
3. source-only 结果不再以低频平滑代理解释，而是以 HRF-modeled coupling target 解释。

### 6.4 Phase 3: Concentration Prior

目标：形成第一版 physiology-aware source/observation baseline。

**需要落地的变更**：

1. 实现 `concentration_loss`；
2. 记录 row entropy、concentration ratio、best lag；
3. 完成小系数 sweep：`0.001 / 0.005 / 0.01`；
4. 把 concentration 结果接入 scorecard 与 final summary。

**Phase 3 输出要求**：

1. coupling row entropy 明显低于 `log(K)` 基线；
2. Gate 1 (Health) 不退化；
3. concentration ratio 稳定大于 1.5。

### 6.5 Workstream A: Coupling Smoothness

前置条件：Phase 3 完成并通过 gate。

**Objective**：为 forward source coupling 加局部平滑先验，让相近 EEG source token 映射到相近 fNIRS token 分布。

**Implementation scope**：

| 文件 | 变更 |
|------|------|
| `src/losses/multimodal_tokenizer.py` | 新增 `coupling_smoothness_loss()` |
| `src/tokenizers/factorized_labram_vqnsp.py` | 接入 smoothness 参数与 loss 汇总 |
| `experiments/configs/source_observation/**` | 新增 A1 / A2 配置 |

**Parameter contract**：

```yaml
loss:
   coupling:
      smoothness_weight: 0.0
      smoothness_neighbors: 5
      smoothness_warmup_epochs: 30
```

**Required diagnostics**：

1. `smoothness_loss`
2. 邻居 token vs 随机 token 的 coupling JS 散度差距
3. coupling row variance
4. Gate 1 (Health) reconstruction / codebook health 是否稳定

**Pass / fail gate**：

- ✅ Pass：Gate 1 (Health) 不退化，coupling 结构更平滑，Gate 3 (Structure) 至少一项指标改善
- ❌ Fail：reconstruction / codebook health 退化，或 coupling 结构改善不可辨认

### 6.6 Workstream C: Causal Direction Asymmetry

前置条件：Phase 3 完成并通过 gate。

**Objective**：把 EEG→fNIRS 与 fNIRS→EEG 从共享一组参数改为独立参数化。

**Implementation scope**：

| 文件 | 变更 |
|------|------|
| `src/tokenizers/factorized_labram_vqnsp.py` | 新增 `coupling_logits_fwd/rev` 与 `coupling_asymmetric` |
| `src/losses/multimodal_tokenizer.py` | 支持 fwd/rev 独立 logits |
| `experiments/configs/source_observation/**` | 新增 C1 配置 |

**Current-stage constraint**：

1. C1 只做参数独立化与不对称诊断；
2. 不在 C1 中叠加 smoothness；
3. 反向路径保持自由参数化，不追加 margin 式显式不对称损失。

**Required diagnostics**：

1. `asymmetry_ratio`
2. forward / reverse per-row entropy
3. 双向 coupling loss
4. forward vs reverse 的下游比较

**Pass / fail gate**：

- ✅ Pass：`asymmetry_ratio` 稳定大于 1，且 Gate 1 (Health) 不退化
- ⚠️ Inconclusive：`asymmetry_ratio` 接近 1，但 Gate 1/3 没有倒退
- ❌ Fail：Gate 1 (Health) 退化，或反向路径明显失稳

---

## 7. Config, Analysis, and Tool Simplification

### 7.1 Config surface simplification

新的活跃配置面不再沿用 shared/private schema，而统一切到 source/observation schema：

```yaml
model:
   type: source_observation_labram_vqnsp
   source:
      codebook_size: 128
      eeg_codebook_dim: 48
      fnirs_codebook_dim: 48
   eeg_observation:
      codebook_size: 256
      codebook_dim: 64
   fnirs_observation:
      codebook_size: 128
      codebook_dim: 48

loss:
   reconstruction:
      eeg_amplitude_weight: 1.0
      eeg_phase_weight: 1.0
      eeg_time_weight: 0.9
      fnirs_amplitude_weight: 1.0
      fnirs_phase_weight: 0.2
      fnirs_time_weight: 1.0
   source_target:
      weight: 0.15
      eeg_aux_weight: 0.075
      warmup_epochs: 30
   coupling:
      weight: 0.07
      concentration_weight: 0.0
      bidirectional: true
      lag_candidates: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
   branch:
      orthogonality_weight: 0.01
   codebook:
      balance_weight: 0.02
```

活跃配置目录调整为：

```text
experiments/configs/source_observation/
   phase1/
   phase2/
   phase3/
   mechanism_a/
   mechanism_c/
```

旧 shared/private 配置全部迁出活跃目录。

### 7.2 Analysis surface simplification

S2 活跃分析面只保留：

1. `tokenizer_analysis_suite.py` 作为唯一标准化入口；
2. source/observation alignment analysis；
3. Gate 1-4 scorecard 汇总；
4. 手动 rerun 入口：
    - `experiments/scripts/probe/analyze_alignment.py`
    - `experiments/scripts/probe/analyze_semantic_token_space.py`
    - `experiments/scripts/probe/generate_tokenizer_analysis_suite.py`

以下 shared/private 旧语义内容退出活跃报告：

1. common / residual target MSE
2. `shared_*_common_loss_objective`
3. `*_private_residual_loss_objective`
4. 任何把 smooth proxy 当作分支定义依据的图表
5. 以 shared/private 为主命名的主报告标题

### 7.3 Probe cleanup policy

`experiments/scripts/probe/` 只保留对标准化分析有价值的脚本。凡是依赖旧 shared/private 代理定义的 exploratory probe，都应迁入 archive，而不是继续作为主线分析入口。

---

## 8. Historical Archive and New Artifact Standard

### 8.1 Historical archive policy

进入 source/observation 主线前，需要先把 shared/private 阶段实验产物从活跃目录中移开，避免与新结果混淆。

新增归档位置：

```text
experiments/runs/archive/source_observation_reset_20260506/
experiments/probe_results/archive/source_observation_reset_20260506/
experiments/configs/archive/source_observation_reset_20260506/
docs/archive/logs/ARCHIVED_SHARED_PRIVATE_MAINLINE.md
```

归档对象包括：

1. 所有 shared/private 阶段 tokenizer runs；
2. 所有依赖 common/residual 语义的 probe results；
3. 所有 shared/private 活跃配置；
4. 与这些结果直接绑定的日志摘要。

历史信息保留在 archive 和 git 中，不再通过活跃主线代码表达。

### 8.2 New run naming convention

新的 source/observation run 统一命名：

```text
s2_<phase>_<variant>_<timestamp>
```

示例：

1. `s2_p1_structural_20260506_101500`
2. `s2_p2_hrf_20260506_143000`
3. `s2_p3_concentration_0005_20260507_090000`

### 8.3 Required run artifacts

每个正式 run 目录必须包含：

```text
experiments/runs/<run_name>/
   config.yaml
   metrics.json
   run_manifest.json
   final_summary.json
   checkpoints/
   figures/
   analysis/
      tokenizer_report/
         manifest.json
         scorecard/
            gate_summary.json
            gate_summary.md
```

### 8.4 Manifest and summary schema

`run_manifest.json` 至少包含：

1. `schema_version`
2. `run_name`
3. `model_type`
4. `semantics_version = s2_source_observation_v1`
5. `phase`
6. `config_hash`
7. `git_commit`
8. `dataset`
9. `analysis_type`
10. `control_group`

`final_summary.json` 只记录最终结论：

1. Gate 1-4 核心指标（Health / Semantics / Structure / Utility）
2. gate pass/fail verdict
3. best checkpoint
4. best lag
5. 简短结论

### 8.5 External control standardization

所有外部方法对照也必须导出到同一套 summary schema。换句话说，控制面不再靠保留旧主线代码，而靠**统一的实验结果协议**。

新增聚合索引：

`experiments/results/source_observation_index.json`

每条记录至少包含：

1. run name
2. phase
3. method family
4. config path
5. Gate 1-4 核心指标
6. promotion verdict

---

## 9. Validation Gates

所有正式实验统一按 4-gate 体系评价，不允许只看单一漂亮指标。Gate 定义详见 [SEMANTIC_TOKEN_SCORECARD.md](docs/SEMANTIC_TOKEN_SCORECARD.md)。

| Gate | 回答的问题 | 当前要求 |
|------|-----------|----------|
| **Gate 1: Health** | codebook 是否健康？reconstruction 是否收敛？ | 4 个 quantizer 均满足健康阈值；full recon 收敛 |
| **Gate 2: Semantics** | source/observation 是否在做各自该做的事？ | HRF target 收敛；obs gap > 0；cross-modal predictability > chance |
| **Gate 3: Structure** | coupling matrix 是否表现出生理合理的集中结构？ | row entropy < log(K)/2；concentration ratio > 1.5 |
| **Gate 4: Utility** | 表示空间是否有 downstream value？ | source SSR > 1.0；subject leakage 集中在 observation |

### Gate dependency

```
Gate 1 ──→ Gate 2 ──→ Gate 3 ──→ Gate 4
(Phase 1)  (Phase 2)  (Phase 3)  (Phase 4+)
```

每个 Phase 只验证一个 Gate。不通过则阻塞，不回退到更早的 Gate。

### Promotion rule

任何机制要进入默认 mainline，必须同时满足：

1. Gate 1 (Health) 不退化；
2. Gate 2 (Semantics) 不退化；
3. Gate 3 (Structure) 有明确增益；
4. Gate 4 (Utility) 不出现明显倒退；
5. 能通过 ablation 解释，不把 source branch 重新变成另一条全能重建捷径；
6. 与至少一类外部研究方法对照相比，能够给出清晰的结构性增益说明。

---

## 10. Implementation Order

当前严格执行以下顺序：

1. ~~完成 shared/private branch semantics audit~~ ✅ 已完成
2. ~~确定 source/observation redesign 机制定义~~ ✅ 已完成
3. **归档 shared/private 阶段实验产物，清理活跃配置与分析入口**
4. **直接改造主线 tokenizer / loss / registry / config surface，完成 Phase 1 Structural Migration**
5. 实现 Phase 2 Source Target Introduction（HRF convolution model）
6. 实现 Phase 3 Concentration Prior（coupling row entropy）
7. 在 Phase 3 baseline 上独立实现并验证 Mechanism A（coupling smoothness）
8. 在 Phase 3 baseline 上独立实现并验证 Mechanism C（causal asymmetry）
9. 统一导出主线与外部方法对照结果到同一 summary schema
10. 更新 scorecard 与 experiment log
11. tokenizer 证据充分后，再考虑 foundation model 层面的目标替换

任何跳步都会导致解释链断裂，不能作为主线证据。

---

## 11. Deliverables Required for Every Mainline Change

任何进入主线候选的改动都必须同时交付：

1. 直接替换后的主线实现，而不是默认关闭的兼容层；
2. 对应 source/observation 配置；
3. archive manifest（如果清理了旧结果或旧配置）；
4. 至少一份正式实验记录；
5. Gate 1-4 scorecard 摘要；
6. 必要的可视化或诊断图；
7. `run_manifest.json` 与 `final_summary.json`；
8. 本文件中的状态更新。

没有文档和评价闭环的改动，不视为主线推进。

---

## 12. Explicitly Deferred Work

以下内容当前明确延后，不进入这轮实现主线：

1. A 与 C 同时启用的联合实验；
2. HRF-shaped lag weighting；
3. 显式熵 margin 式 asymmetry loss；
4. 重新引入 identity-style alignment losses；
5. foundation model 预训练目标的大改；
6. equal token count per window 的结构审计。

这些方向不是永久否定，而是必须等到 Phase 1-3 与 A/C 单机制证据成立后再决定是否继续。

---

## 13. Bottom Line

当前项目的 tokenizer 主线已经从“证明 token 条件概率可分析”切换到“设计一个带有生理结构先验的离散表示机制”。

**当前主线决策非常明确**：

1. shared/private 将被 source/observation 直接取代；
2. 旧架构代码不会继续作为活跃对照面保留；
3. 历史可追溯性由 git 和 archive 提供；
4. 对照实验改由外部研究方法承担；
5. 分析、归档与结果格式规范全部以本文件为准。

**当前步骤**：先完成 shared/private 阶段实验与配置归档，然后直接改造主线实现面，进入 Phase 1 Structural Migration。

本文件即为当前实现顺序与准入标准的唯一主文档。