# Neuro-Tokenization Implementation Plan

> Last revised: 2026-05-14
> Status: Architecture stabilized — Phase 2B (Croce 2017 Physical Model + Coupling Structure Priors) implemented
> Detailed design rationale: [docs/PHYSIOLOGICAL_COUPLING_PLAN.md](docs/PHYSIOLOGICAL_COUPLING_PLAN.md)
> Evaluation scorecard: [docs/SEMANTIC_TOKEN_SCORECARD.md](docs/SEMANTIC_TOKEN_SCORECARD.md) — 4 evaluation gates (Health / Semantics / Structure / Utility)
> Experiment log: [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)
> Current architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 1. Role of This Document

本文件是当前仓库唯一的 tokenizer 主实现计划文档。

文档分工固定如下：

1. **IMPLEMENTATION_PLAN.md**：实现顺序、代码改造范围、分析与实验产物规范。
2. **docs/PHYSIOLOGICAL_COUPLING_PLAN.md**：机制动机、数学形式、生理解释。
3. **docs/SEMANTIC_TOKEN_SCORECARD.md**：4-gate 评价框架（Health / Semantics / Structure / Utility）。
4. **docs/EXPERIMENT_LOG.md**：正式实验结论。

如果多个文档之间出现冲突，以本文件的实现顺序为准；如果是机制定义或数学细节冲突，以生理耦合计划为准。

---

## 2. Current Architecture Status

### 2.1 Architecture is stabilized

经过 Phase 1 → Phase 2A → Phase 2B 的连续架构演进，当前主线 tokenizer 架构已基本稳定。核心架构不再进行大范围探索性改动，后续工作聚焦于：(1) 在当前架构上验证生理假设，(2) 调参和诊断，(3) 下游任务评估。

当前主实现面落在以下文件上：

- [src/tokenizers/factorized_labram_vqnsp.py](src/tokenizers/factorized_labram_vqnsp.py) — 主 tokenizer：`SourceObservationLaBraMVQNSP`
- [src/losses/multimodal_tokenizer.py](src/losses/multimodal_tokenizer.py) — coupling 结构先验（lag_focus + joint_smoothness）与分支正交损失
- [src/inference/neurovascular_smc.py](src/inference/neurovascular_smc.py) — Croce 2017 SMC 滤波器模块
- [src/data/channel_adjacency.py](src/data/channel_adjacency.py) — 导联邻接与空间加权 source target
- [src/visualization/source_observation_analysis.py](src/visualization/source_observation_analysis.py) — source/observation 对齐分析与 Gate scorecard
- [src/visualization/tokenizer_analysis_suite.py](src/visualization/tokenizer_analysis_suite.py) — 标准化分析入口

### 2.2 Architecture decisions made and executed

以下决策已在当前主线代码中落地：

1. 主线架构：**source/observation**（shared/private 已完全移除）；
2. 双 source codebook（K=32） + constrained coupling（lag focus + joint smoothness）；
3. HRF convolution target 接管 source branch 目标，smooth_signal 代理已退出；
4. observation branch 通过 reconstruction debt 与显式 orthogonality 约束定义；
5. 双 decoder 架构：source/observation 各有独立 decoder，full = source_recon + obs_recon（加法组合）；
6. Source target 采用 Croce et al. 2017 物理模型：shared latent neural state s(t) → 同步驱动 EEG（signed RMS carrier）和 fNIRS（HRF-convolved）；
7. Coupling 结构先验：lag_focus_loss（delay marginal entropy）+ joint_smoothness_loss（neighbor JS divergence）。

### 2.3 Control policy

对照面由以下两类提供：

1. **外部研究方法对照**：来自 [comparative_methods](comparative_methods) 和 [reference_repository](reference_repository) 的方法实现或复现结果；
2. **历史主线参考**：通过 git 历史、归档 run、归档实验记录回看 shared/private 阶段结果。

### 2.4 What is no longer the main target

以下目标不再作为 tokenizer 主线的出发点：

1. shared token identity overlap 最大化；
2. alignment-first 叙事；
3. 通过让 shared branch 承担更强 raw reconstruction 来"逼出"跨模态共性；
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

3. **当前主线先验是 lag focus + joint smoothness**
   - delay 结构和 EEG 邻居平滑构成 coupling 结构先验；
   - 不做 smoothness + asymmetry 的联合实验（Mechanism C 已废弃，见 §12）。

4. **Gate 1 (Health) 不退化是硬门槛**
    - reconstruction / codebook health 退化，则该机制不能进入默认主线。

5. **Branch semantics 已通过，旧语义不应残留在活跃主线里**
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
| 活跃架构文档 | `docs/ARCHITECTURE.md` | 当前架构的 Mermaid 图、组件目录、数据流（始终反映主线最新状态） |
| 架构修改日志 | `docs/architecture_changelog/INDEX.md` | 每次架构变更的独立记录（含 before/after 图、组件变更表、设计决策） |
| 归档实验日志 | `docs/archive/logs/PHASE1_GATE1_STABILIZATION_20260511.md` | Phase 1 Gate1 稳定化记录 |

顶层 [docs](docs) 只保留当前主线需要反复阅读的活跃文档；历史材料统一进入 [docs/archive](docs/archive)。历史预实验记录和 shared/private 阶段设计文档通过 git 历史查阅，不再维护独立的归档文档文件。

架构修改记录体系：每次架构变更必须先在 [docs/architecture_changelog/](docs/architecture_changelog/) 中建立独立记录（模板见 [template.md](docs/architecture_changelog/template.md)），然后更新 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 到新的当前状态。

---

## 5. Completed Architecture Evolution

以下架构演进阶段已完成，此处仅保留摘要记录。详细设计决策见各 Phase 的 changelog。

### 5.1 Phase 1: Structural Migration ✅

shared/private → source/observation 语义切换，双独立 source codebook 替换单一 shared quantizer，删除所有旧 loss 项。详见 [2026-05-06 changelog](docs/architecture_changelog/2026-05-06_source_observation_migration.md)。

**成果**：Gate 1 (Health) 稳定通过；baseline locked at `gate1_best_current.yaml`。

### 5.2 Phase 2: HRF Source Target Introduction ✅

实现 double-gamma HRF kernel，用 HRF convolution target 替代 smooth proxy。详见 [2026-05-11 changelog](docs/architecture_changelog/2026-05-11_phase2a_branch_target_redesign_dual_decoder.md)。

**成果**：source_target_loss 可稳定下降，但 Gate 2-4 均 fail（需要 Phase 2A 修复结构和训练协议）。

### 5.3 Phase 2A: Branch Target Redesign + Dual Decoder Architecture ✅

修复 Phase 2 的四个结构性问题：(1) 单 decoder → 4 独立 decoder（source/observation per modality）；(2) 三模式显式训练（full / source-only / observation-only）；(3) 加法组合 reconstruction；(4) observation target 显式定义。

**成果**：Gate 2 (Semantics) pass；离线梯度诊断揭示了 source branch 扁平化问题，由 Phase 2A-spatial 和 Phase 2B Croce 物理模型修复。

### 5.4 Phase 2B: Croce 2017 Physical Model Targets ✅

采纳 Croce et al. 2017 联合 EEG-fNIRS 状态空间模型：shared latent neural state s(t) 通过 AR(1) 平滑驱动两个模态。EEG source target 从 power envelope (μV²) 改为 signed RMS carrier (μV)，恢复加法分解的物理意义。fNIRS source target 改为 HRF(s(t))。详见 [2026-05-13 changelog](docs/architecture_changelog/2026-05-13_phase2b_croce2017_physical_model_targets.md)。

新增组件：`_compute_shared_neural_state`（AR-smoothed neural driver）、`_compute_eeg_source_target`（signed RMS carrier 模式）、`src/inference/neurovascular_smc.py`（SMC 滤波器验证模块）。

### 5.5 Coupling Structure Priors (Lag Focus + Joint Smoothness) ✅

在 `src/losses/multimodal_tokenizer.py` 中实现 `coupling_lag_focus_loss()` 和 `coupling_eeg_neighbor_smoothness_loss()`，直接约束 coupling 矩阵形状：

| 约束 | 作用维度 | 数学形式 |
|------|---------|---------|
| `lag_focus_loss` | EEG token 的 delay 边际 | $H(p_i(\tau))$, 其中 $p_i(\tau)=\sum_j Q_i(\tau,j)$ |
| `joint_smoothness_loss` | EEG 邻居之间 | JS($Q_i(\tau,j)$ \|\| $Q_{i'}(\tau,j)$) |

两个先验已在 `forward()` 中接入 total_loss。耦合诊断：lag_focus_loss, joint_smoothness_loss, delay entropy histogram, neighbor JS gap。

---

## 6. Current Config, Analysis, and Tool Standards

### 6.1 Config surface

活跃配置面统一使用 source/observation schema：

```yaml
model:
   type: source_observation_labram_vqnsp
   source:
      codebook_size: 32
      eeg_codebook_dim: 48
      fnirs_codebook_dim: 48
   eeg_observation:
      codebook_size: 64
      codebook_dim: 64
   fnirs_observation:
      codebook_size: 64
      codebook_dim: 48

loss:
   reconstruction:
      eeg_amplitude_weight: 1.0
      eeg_phase_weight: 0.0
      eeg_time_weight: 0.9
      fnirs_amplitude_weight: 1.0
      fnirs_phase_weight: 0.0
      fnirs_time_weight: 1.0
   source_target:
      weight: 0.3
      eeg_source_aux_weight: 0.3
      warmup_epochs: 30
   observation_target:
      weight: 0.15
      warmup_epochs: 30
   coupling:
      weight: 0.01
      lag_focus_weight: 1.0
      smoothness_weight: 0.2
      smoothness_neighbors: 5
      bidirectional: true
      lag_candidates: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
   branch:
      orthogonality_weight: 0.05
   codebook:
      balance_weight: 0.08
      source_balance_scale: 1.0
      observation_balance_scale: 0.5
```

活跃配置目录：

```text
experiments/configs/source_observation/
   phase1/           # Phase 1 Gate1 baseline（锁定）
   phase2/           # Phase 2 HRF target（历史参考）
   phase2a/          # Phase 2A dual decoder + spatial source targets（活跃）
```

旧 shared/private 配置已全部迁出活跃目录。

### 6.2 Analysis surface

活跃分析面：

1. `tokenizer_analysis_suite.py` 作为唯一标准化入口；
2. `source_observation_analysis.py` — source/observation 对齐分析与 Gate 1-4 scorecard 汇总；
3. 手动 rerun 入口：
    - `experiments/scripts/probe/analyze_alignment.py`
    - `experiments/scripts/probe/analyze_semantic_token_space.py`
    - `experiments/scripts/probe/generate_tokenizer_analysis_suite.py`

以下旧语义内容退出活跃报告：

1. common / residual target MSE
2. `shared_*_common_loss_objective`
3. `*_private_residual_loss_objective`
4. 任何把 smooth proxy 当作分支定义依据的图表
5. 以 shared/private 为主命名的主报告标题

### 6.3 Spatial source targets

`src/data/channel_adjacency.py` 提供 10-10 EEG 邻居表、fNIRS 通道名解析、mnt.mat 3D 坐标校验、邻接矩阵构建和可视化。空间 source target 模式通过 `_compute_eeg_source_target`（per-channel RMS）和 `_compute_fnirs_source_target`（空间加权 HRF）实现。所有新参数可空，默认回退到全局均值路径，保证旧 checkpoint 可正常加载。

---

## 7. Archive and Artifact Standards

### 7.1 Archive policy

已完成归档：

- `experiments/runs/archive/source_observation_phase1_gate1_stabilization_20260511/` — Phase 1 Gate1 稳定化 runs
- `experiments/configs/archive/source_observation_phase1_gate1_stabilization_20260511/` — Phase 1 Gate1 调参配置
- `docs/archive/logs/PHASE1_GATE1_STABILIZATION_20260511.md` — Phase 1 Gate1 记录

历史 shared/private 阶段实验产物通过 git 历史追踪，不再维护额外的独立归档目录。

### 7.2 Run naming convention

```text
s2_<phase>_<variant>_<timestamp>
```

示例：

1. `s2_p1_structural_20260506_101500`
2. `s2_p2_hrf_20260506_143000`
3. `s2_p2a_dual_decoder_20260511_120000`

### 7.3 Required run artifacts

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

### 7.4 Manifest and summary schema

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

### 7.5 External control standardization

所有外部方法对照统一导出到同一套 summary schema。聚合索引：

`experiments/results/source_observation_index.json`

---

## 8. Validation Gates

所有正式实验统一按 4-gate 体系评价。Gate 定义详见 [SEMANTIC_TOKEN_SCORECARD.md](docs/SEMANTIC_TOKEN_SCORECARD.md)。

| Gate | 回答的问题 | 当前要求 |
|------|-----------|----------|
| **Gate 1: Health** | codebook 是否健康？reconstruction 是否收敛？ | 4 个 quantizer 均满足健康阈值；full recon 收敛 |
| **Gate 2: Semantics** | source/observation 是否在做各自该做的事？ | HRF target 收敛；obs gap > 0；cross-modal predictability > chance |
| **Gate 3: Structure** | coupling tensor 是否表现出生理合理的 delay-aware 结构？ | delay entropy 低于均匀基线；neighbor JS gap > 0；可视化出现平滑 ridge |
| **Gate 4: Utility** | 表示空间是否有 downstream value？ | source SSR > 1.0；subject leakage 集中在 observation |

Gate dependency（当前阶段）：

```
Gate 1 ──→ Gate 2 ──→ Gate 3 ──→ Gate 4
(Phase 1)  (Phase 2A) (Phase 2B) (Phase 4+)
```

Gate 1 和 Gate 2 已通过。Gate 3 (Structure) 是当前验证目标。Gate 4 (Utility) 是长期指标。

### Promotion rule

任何机制要进入默认 mainline，必须同时满足：

1. Gate 1 (Health) 不退化；
2. Gate 2 (Semantics) 不退化；
3. Gate 3 (Structure) 有明确增益；
4. Gate 4 (Utility) 不出现明显倒退；
5. 能通过 ablation 解释，不把 source branch 重新变成另一条全能重建捷径；
6. 与至少一类外部研究方法对照相比，能够给出清晰的结构性增益说明。

---

## 9. Current and Near-Term Work

当前严格按照以下顺序推进：

1. ~~完成 shared/private branch semantics audit~~ ✅
2. ~~确定 source/observation redesign 机制定义~~ ✅
3. ~~归档 shared/private 阶段实验产物，清理活跃配置与分析入口~~ ✅
4. ~~Phase 1: Structural Migration（shared/private → source/observation）~~ ✅
5. ~~锁定 Phase 1 Gate1 baseline，归档调参结果~~ ✅
6. ~~Phase 2: Source Target Introduction（HRF convolution model）~~ ✅
7. ~~Phase 2A: Branch Target Redesign + Dual Decoder Architecture~~ ✅
8. ~~Phase 2A-spatial: Spatially-Informed Source Targets~~ ✅
9. ~~Phase 2B: Croce 2017 Physical Model Targets~~ ✅
10. ~~Phase 2B: Coupling Structure Priors（lag focus + joint smoothness）~~ ✅
11. **验证 Gate 3 (Structure) — 当前焦点**
12. 统一导出主线与外部方法对照结果到同一 summary schema
13. 更新 scorecard 与 experiment log
14. tokenizer 证据充分后，再考虑 foundation model 层面的目标替换

---

## 10. Deliverables Required for Every Mainline Change

任何进入主线候选的改动都必须同时交付：

1. 直接替换后的主线实现，而不是默认关闭的兼容层；
2. 对应 source/observation 配置；
3. archive manifest（如果清理了旧结果或旧配置）；
4. 至少一份正式实验记录；
5. Gate 1-4 scorecard 摘要；
6. 必要的可视化或诊断图；
7. `run_manifest.json` 与 `final_summary.json`；
8. 本文件中的状态更新；
9. [架构修改日志](docs/architecture_changelog/INDEX.md) 条目（含 before/after 架构图、组件变更表、设计决策）；
10. 更新后的 [ARCHITECTURE.md](docs/ARCHITECTURE.md)（反映变更后的当前架构状态）。

---

## 11. Explicitly Deferred and Abandoned Work

### Deferred（延后至远期，当前不进入实现主线）

1. fNIRS→EEG 跨模态预测器（原 Phase 2C core）：为 EEG source target 引入 fNIRS 侧信息约束
2. Coupling-aware quantization（原 Phase 2C core）：在量化步骤中消费 coupling 结构先验
3. HRF-shaped lag weighting
4. Foundation model 预训练目标的大改
5. Equal token count per window 的结构审计
6. HRF 频带分解（delta/theta/alpha/beta/gamma）
7. 重新引入 identity-style alignment losses

### Abandoned（已废弃，不再计划实现）

1. **Mechanism C: Causal Direction Asymmetry** — 将 EEG→fNIRS 与 fNIRS→EEG coupling 从共享一组参数改为独立 fwd/rev 参数化。废弃理由：当前架构已通过 Croce 物理模型提供清晰的方向性（EEG → shared state → HRF → fNIRS），独立不对称参数化不再提供额外的解释力增益；且架构已稳定，不再进行此类机制层面的探索性改动。
2. A 与 C 同时启用的联合实验（随 C 废弃而取消）
3. 显式熵 margin 式 asymmetry loss

---

## 12. Bottom Line

当前项目的 tokenizer 主线已经完成从"被动分析条件概率"到"主动设计生理结构先验"的转变。架构核心决策已全部落地：

1. shared/private → source/observation ✅
2. 单一 shared quantizer → 双 source codebook + coupling 结构先验 ✅
3. smooth proxy → HRF convolution target (Croce 2017 physical model) ✅
4. 单 decoder → 双 decoder 加法架构 + 三模式显式训练 ✅
5. 自由参数化 coupling → lag focus + joint smoothness 结构先验 ✅

**架构已稳定。** 当前工作焦点是验证 Gate 3 (Structure)，即 coupling tensor 是否确实被生理先验塑造成 delay-aware 的结构形态。后续工作聚焦于实验验证、诊断细化和下游评估，不再进行大范围架构探索。

本文件即为当前实现顺序与准入标准的唯一主文档。
