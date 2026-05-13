# Neuro-Tokenization Implementation Plan

> Rewritten: 2026-04-30 | Last revised: 2026-05-13
> Status: Active mainline execution guide — Phase 2B: Croce 2017 Physical Model Targets
> Detailed design rationale: [docs/PHYSIOLOGICAL_COUPLING_PLAN.md](docs/PHYSIOLOGICAL_COUPLING_PLAN.md)
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

3. **当前主线先验是 lag focus + joint smoothness，C 仍单独验证**
   - delay 结构和 EEG 邻居平滑现在一起构成 Phase 2B baseline；
   - Mechanism C（causal asymmetry）仍在 baseline 稳定后独立验证；
   - 当前阶段不做 smoothness + asymmetry 的联合实验。

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
| 活跃架构文档 | `docs/ARCHITECTURE.md` | 当前架构的 Mermaid 图、组件目录、数据流（始终反映主线最新状态） |
| 架构修改日志 | `docs/architecture_changelog/INDEX.md` | 每次架构变更的独立记录（含 before/after 图、组件变更表、设计决策） |
| 归档计划 | `docs/archive/plans/NEXT_STAGE_ALIGNMENT_PLAN.md` | reset 阶段设计基础 |
| 归档实验日志 | `docs/archive/logs/ARCHIVED_PRE_EXPERIMENTS.md` | 第一轮预实验历史记录 |

顶层 [docs](docs) 只保留当前主线需要反复阅读的活跃文档；历史材料统一进入 [docs/archive](docs/archive)。

**架构修改记录体系**：每次架构变更（新组件引入、组件语义变更、数据流重构）必须先在 [docs/architecture_changelog/](docs/architecture_changelog/) 中建立独立记录（模板见 [template.md](docs/architecture_changelog/template.md)），然后更新 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 到新的当前状态。这两个文档分别回答"改了什么"和"现在是什么"——前者是增量记录，后者是当前真相。要回看历史架构状态，查 changelog INDEX；要了解当前主线结构，只读 ARCHITECTURE.md。

---

## 5. Current Development Goal

当前阶段的核心目标只有一个：

**直接把当前主线 tokenizer 从 shared/private 改造成 source/observation，并在同一条主线上依次完成 Structural Migration、HRF Source Target、Lag-Focused Coupling Prior。**

### 5.1 Architecture decision

当前已经明确的架构结论：

1. 保留 factorization 大方向；
2. shared/private 直接改名并改义为 **source/observation**；
3. 单一 shared quantizer 直接替换为 **双独立 source codebook**；
4. smooth common proxy 直接替换为 **HRF convolution target**；
5. observation branch 不再接显式 residual target，而由 reconstruction necessity 定义；
6. coupling 从自由参数化升级为 **lag-focused + EEG-neighbor-smoothed physiological mapping**。

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

### 6.4 Phase 2A: Branch Target Redesign + Dual Decoder Architecture

**动机**：

当前 Phase 2 的实现存在四个结构性问题，需要在此阶段一次性修复：

1. **Decoder 从未被训练来处理分支 ablations**：`source_only_reconstructions` 和 `observation_only_reconstructions` 通过将另一分支的 latent 置零后 decode 得到，但 decoder 训练时从未见过 `[source_q, 0]` 或 `[0, obs_q]` 这种输入。评测时 decoder 面对的是 OOD 输入。

2. **EEG source target 与 fNIRS source target 不是同一个东西**：EEG source 重建 coarse downsampled signal（纯低通滤波），fNIRS source 重建 HRF(EEG_power_envelope)。耦合矩阵要在这两个不同概念的离散化之间建立映射，在概念上就是矛盾的——这是当前 coupling 坍缩为均匀分布的根本原因之一。

3. **Observation branch 零显式监督**：没有显式 observation target，导致 source/observation 分解不可辨识。模型的最优策略是把几乎所有信息放入 observation（通过 full reconstruction 梯度），source 只学到刚好满足弱 auxiliary loss 的最小信息。

4. **单 decoder + concat(latents) 架构**：source 和 observation 信息在 decoder 内部自由混合，没有结构性压力迫使它们分离。

**设计决策（来自 TokenFlow 和多视图学习的启示）**：

TokenFlow 的核心模式：每个分支有自己独立的 decoder 和独立的 target。多视图学习（VCCA, MVAE, FactorCL）的生成假设：信号 = 共享成分 + 独有成分。

本 Phase 将这两个原则结合：
- **双 decoder 架构**：source 和 observation 各有一个独立 decoder
- **加法组合**：full_recon = source_decoder(source_q) + observation_decoder(obs_q)
- **三个模式全部显式训练**：source-only → source_target, observation-only → observation_target, full → original
- **统一的 source target 定义**：两侧 source target 来自同一个 neural driver D(t)

**架构变更**：

```
旧架构 (Phase 1/2):
  eeg_source_q ─┐
                 ├→ concat → eeg_decoder → eeg_recon
  eeg_obs_q ────┘

新架构 (Phase 2A):
  eeg_source_q → eeg_source_decoder → eeg_source_recon ─┐
                                                          ├→ sum → eeg_recon
  eeg_obs_q    → eeg_obs_decoder    → eeg_obs_recon ─────┘
```

fNIRS 侧同理。每个 modality 从 1 个共享 decoder 变为 2 个独立 decoder。

**需要落地的变更**：

1. 新增 4 个独立 decoder（替换现有的 2 个共享 decoder）：
   - `eeg_source_decoder` / `eeg_observation_decoder`
   - `fnirs_source_decoder` / `fnirs_observation_decoder`

2. 重定义 source target，统一为同一个 neural driver D(t) = EEG 宽带功率包络：
   - **EEG source target**：D(t) 在 EEG 原生采样率上计算（2000 timepoints），扩展至所有 EEG 通道。**不降采样到 fNIRS 频率。**
   - **fNIRS source target**：D(t) 降采样到 fNIRS 分辨率后经 HRF 卷积，再按 fNIRS 通道做均值/方差缩放
   - 两侧 target 都是 D(t) 的函数，语义一致

3. 新增 observation target：`obs_target = original - source_target`（两个模态各自计算）

4. 三个 decoder 模式全部显式训练：
   - `decode(source_q, 0) → source_target` (source_target_loss)
   - `decode(0, obs_q) → obs_target` (observation_loss, **新增**)
   - `decode(source_q, obs_q) → original` (full reconstruction loss, 加法组合)

5. Codebook 容量调整：source 保持 32，observation 扩容至 64

6. Loss 权重重新平衡

**Implementation scope**：

| 文件 | 变更 |
|------|------|
| `src/tokenizers/factorized_labram_vqnsp.py` | 新增 4 个独立 decoder；重定义 source/obs target 构造方法；三个 decoder 模式前向；新 loss 汇总 |
| `experiments/configs/source_observation/phase2a/` | 新增 Phase 2A 配置 |

**参数合约**：

```yaml
model:
  source:
    codebook_size: 32           # 保持 Phase 1 成果
  eeg_observation:
    codebook_size: 64           # 扩容（32 → 64）
  fnirs_observation:
    codebook_size: 64           # 扩容（32 → 64）

loss:
  source_target:
    weight: 0.3                 # 0.15 → 0.3
    eeg_source_aux_weight: 1.0  # 0.5 → 1.0（target 现在有意义了）
    warmup_epochs: 30
  observation_target:
    weight: 0.15               # 新增
    warmup_epochs: 30
  coupling:
      weight: 0.0
    bidirectional: true
  branch:
    orthogonality_weight: 0.05  # 0.01 → 0.05
  codebook:
    balance_weight: 0.08
    source_balance_scale: 1.0
    observation_balance_scale: 0.5  # 从 0.0 恢复
```

**诊断指标**：

1. `source_target_loss` (fNIRS) 和 `eeg_source_aux_loss` (EEG) 是否稳定下降
2. `observation_loss` 是否收敛
3. Source-only vs observation-only MSE gap 是否增大
4. Gate 1 (Health) reconstruction 是否稳定（不应显著退化）
5. Coupling row entropy 是否仍为 log(K)（预期仍高，因为 concentration prior 在 Phase 2B 才引入）

**门控标准**：

- ✅ Pass：三个 decoder 模式均收敛；source target 两侧均优于 random baseline；Gate 1 (Health) 不退化
- ❌ Fail：任意 decoder 模式不收敛，或 Gate 1 (Health) 显著退化

**设计理念总结**：

TokenFlow 的"双 decoder + 各自独立 target"模式提供了最清晰的分支语义定义。多视图学习（FactorCL, CoMM, VCCA）的"共享信息 = 跨模态可预测"原则为 source target 提供了一致的理论框架：source 编码的是两个模态共同可预测的神经驱动状态，observation 编码的是各自模态特有的残差。

### 6.5 Phase 2B: Croce 2017 Physical Model Targets (2026-05-13)

**动机**：

Phase 2A 的 EEG source target 使用 power envelope（μV²，非负），破坏了加法分解的物理意义。此外，数据验证显示 raw EEG broadband power → fNIRS 的 per-window 相关性极弱（~0.01），不适合作为 fNIRS source target 的唯一信息源。

采纳 Croce et al. 2017 的联合 EEG-fNIRS 状态空间模型：一个共享的隐神经状态 s(t) 同时驱动两个模态。

**三处关键修改**：

| 修改 | 旧设计 | 新设计 |
|------|--------|--------|
| Shared neural state | 无（各自独立计算 driver） | $s_k = \alpha s_{k-1} + (1-\alpha)x_k$，$\alpha=0.90$ |
| fNIRS source target | HRF(raw EEG power) | HRF(s(t)) — 时序平滑后的驱动 |
| EEG source target | power envelope (μV²) 或 RMS envelope | signed_rms_carrier (μV, 有符号) — 同量纲 |

**新增参数**：

- `shared_state_alpha` (default 0.90): AR(1) 平滑系数。α=0.90 在 10 Hz 下半衰期 ~0.66s，保留 α/β 节律的功率包络变化
- `eeg_target_mode` 默认改为 `signed_rms_carrier`：有符号 μV 输出，加法分解物理有意义

**验证结果**（SMC 分析）：

- 预处理对 alignment（phase1 default.yaml）✓
- 重建信号与原始信号时间同步（lag-0）✓
- HRF 卷积正确吸收神经血管延迟 ✓
- α=0.998 时状态太慢（34.6s 半衰期），任务区分度不显著（p=0.66）→ α=0.90 预期改善

**涉及文件**：

| 文件 | 变更 |
|------|------|
| `src/tokenizers/factorized_labram_vqnsp.py` | `_compute_shared_neural_state`，`_compute_fnirs_source_target`，`_compute_eeg_source_target` 重写 |
| `src/inference/neurovascular_smc.py` | 新增：SMC 滤波器模块 |
| `experiments/configs/source_observation/phase2a/` | 新增 `shared_state_alpha`，`eeg_target_mode` |
| `docs/ARCHITECTURE.md` | Section 7 重写为 Croce 物理模型 |
| `docs/architecture_changelog/2026-05-13_phase2b_*.md` | 新增 changelog |

### 6.6 Phase 2C: Coupling Structure Priors (Lag Focus + Joint Smoothness)

**动机**：

当前 EEG-fNIRS 匹配项不再包含 KL 数据项。Phase 2B 仅保留 coupling 结构先验设计，目标是直接约束 coupling 矩阵形状，避免继续依赖数据锚点。

**两个互补的先验**：

| 约束 | 作用维度 | 数学形式 | 优先级 |
|------|---------|---------|--------|
| `lag_focus_loss` | EEG token 的 delay 边际 | $H(p_i(\tau))$, 其中 $p_i(\tau)=\sum_j Q_i(\tau,j)$ | **P0** |
| `joint_smoothness_loss` | EEG 邻居之间 | JS($Q_i(\tau,j)$ \|\| $Q_{i'}(\tau,j)$) | **P1** |

其中

$$
Q_i(\tau, j) = P(\tau, z_{fnirs}=j \mid z_{eeg}=i)
$$

是对 lag 和 fNIRS token 做联合 softmax 后得到的 EEG 条件分布。新的 lag focus 不再要求整个 token-lag 空间只保留少数几个点，而是只要求每个 EEG token 偏好少数几个 delay；对于固定 lag，允许对应多个 fNIRS token。joint smoothness 则要求 EEG codebook 中相近的状态拥有相近的联合 delay-response 分布。

**⚠️ 耦合两项先验的潜在冲突与监控要求**：

这两个先验作用在同一 `coupling_logits` 矩阵上，理论上存在张力：

1. `lag_focus_loss` 将每个 EEG token 的 delay 边际推向低熵（极端情况：所有 token 都只偏好单一 lag）
2. `joint_smoothness_loss` 将邻居 token 推向相似的联合分布（极端情况：所有邻居共享同一 delay-response 模板）

如果 lag focus 过强而 joint smoothness 适当 → 不同 EEG token 可能都只压向单一 lag，delay 结构变窄
如果 joint smoothness 过强而 lag focus 适当 → 所有邻居共享近乎相同的 delay-response 分布，失去区分度
如果两者都过强 → coupling 退化为少数 delay 模板，且 lag 内 token 分布也缺少差异

**必须在训练和分析中持续监控以下指标**：
- `lag_focus_loss` 值的时间序列（应下降后稳定，不应持续下降至零）
- `joint_smoothness_loss` 值的时间序列（应下降后稳定）
- Delay marginal entropy 直方图（应明显低于均匀基线，但不应坍缩到接近零）
- Neighbor JS gap：邻居对的平均 JS vs 随机对的平均 JS（前者应显著低于后者）
- Coupling tensor 可视化（应出现平滑的 delay-aware ridge，而不是整片均匀灰）

训练日志中应同时输出这两个先验的原始值。

**需要落地的变更**：

1. 在 `src/losses/multimodal_tokenizer.py` 实现 `coupling_lag_focus_loss()` 和 `coupling_eeg_neighbor_smoothness_loss()`
2. 在 `forward()` 中接入两个新 loss 项并汇总到 total_loss
3. 新增耦合诊断指标：lag_focus_loss, joint_smoothness_loss, delay entropy histogram, neighbor JS gap
4. smoothness 子项默认保持小系数，仅作为 lag focus 的邻域正则

**参数合约**：

```yaml
loss:
  coupling:
      weight: 0.01                  # overall source_coupling_loss scale
      lag_focus_weight: 1.0         # delay marginal entropy term
      smoothness_weight: 0.2        # internal multiplier on joint smoothness
    smoothness_neighbors: 5
    bidirectional: true
```

**Phase 2B 输出要求**：

1. delay marginal entropy 明显低于均匀基线，但不接近零
2. Gate 1 (Health) 不退化
3. 主导 lag 不是完全均匀平摊到所有候选 delay
4. 邻居 token 的 joint coupling JS 散度显著低于随机 token 对

### 6.6 Phase 2C (延后): Cross-Modal EEG Source Target + Coupling-Aware Quantization

前置条件：Phase 2B 完成并通过 gate。

**Objective**：
1. 为 EEG source target 引入 fNIRS 侧信息约束（fNIRS→EEG 预测器），实现真正的"source = 跨模态可预测"
2. 实现 coupling-aware quantization（原 Phase 2A），在量化步骤中消费 coupling 结构的先验

**Implementation scope**：在 Phase 2B baseline 稳定后细化为独立计划文档。

**门控标准**：Gate 2A (Quantization-Coupling Consistency) 通过 + Gate 1 (Health) 不退化。

### 6.7 Workstream C: Causal Direction Asymmetry

前置条件：Phase 2B 完成并通过 gate。

**Objective**：把 EEG→fNIRS 与 fNIRS→EEG 从共享一组参数改为独立参数化。

**Implementation scope**：

| 文件 | 变更 |
|------|------|
| `src/tokenizers/factorized_labram_vqnsp.py` | 新增 `coupling_logits_fwd/rev` 与 `coupling_asymmetric` |
| `src/losses/multimodal_tokenizer.py` | 支持 fwd/rev 独立 logits |
| `experiments/configs/source_observation/mechanism_c/` | 新增 C1 配置 |

**Current-stage constraint**：

1. C1 只做参数独立化与不对称诊断；
2. 不在 C1 中叠加 smoothness（smoothness 已在 2B 中引入，此处不额外叠加）；
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
      codebook_size: 32         # Phase 1 成果保持，不小于 observation
      eeg_codebook_dim: 48
      fnirs_codebook_dim: 48
   eeg_observation:
      codebook_size: 64         # 扩容承载模态特异细节
      codebook_dim: 64
   fnirs_observation:
      codebook_size: 64         # 扩容承载模态特异细节
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
      eeg_source_aux_weight: 1.0
      warmup_epochs: 30
   observation_target:
      weight: 0.15
      warmup_epochs: 30
   coupling:
      weight: 0.0
      concentration_weight: 0.01
      smoothness_weight: 0.002
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

活跃配置目录调整为：

```text
experiments/configs/source_observation/
   phase1/
   phase2/
   phase2a/
   phase2b/
   phase2c/
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

### 7.4 Phase 2A-spatial: Spatially-Informed Source Targets (NEW)

> Status: Design approved 2026-05-12, implementation pending
> Detailed design: [docs/PHYSIOLOGICAL_COUPLING_PLAN.md §2.4](docs/PHYSIOLOGICAL_COUPLING_PLAN.md#24-source-branch-target-spatially-informed-hrf-convolution-model)
> Replaces: Section 12 item 8 "导联空间关系建模" (removed from deferred, now active)

#### 7.4.1 Motivation

Phase 2A 的离线梯度诊断揭示了当前 source target 的根本问题：EEG source target 是跨通道均值功率包络（所有通道相同），fNIRS source target 是全局 HRF-convolved 驱动（所有通道共享）。这导致 branch 相关 loss 占据 59-67% 梯度份额，且多个目标一起把 source branch 推向低方差、共享模板的退化解。

修复方向：在 source target 中引入**空间结构**——每个 EEG 通道有自己的 RMS 包络作为 source target，每个 fNIRS 通道的 source target 由其**空间邻近**的 EEG 通道加权驱动。

#### 7.4.2 Key Design Decisions

1. **输入保持全通道** (30 EEG + 36 fNIRS)：EEG 容积传导需要全局上下文进行空间去混叠；耦合矩阵需要跨区域视野才能建立有意义的全局离散状态空间；配对输入会造成通道重复编码。
2. **EEG source target = per-channel RMS 包络**：`sqrt(eeg_ch²)` → 电压单位，`observation = original - rms_envelope` 维度一致。
3. **fNIRS source target = 空间加权 HRF 预测**：`Σ_{nearby_EEG} w * power → HRF → per_fNIRS_ch_target`。
4. **空间权重基于导联实际位置校验**：优先使用 `mnt.mat` 中的 3D 坐标，回退到 10-10 标准邻居表；仅考虑 1 步邻居。
5. **新增可视化**：导联位置散点图、邻接矩阵热力图、跨模态通道相关矩阵，集成到标准分析 pipeline。

#### 7.4.3 Files

| 文件 | 变更 |
|------|------|
| `src/data/channel_adjacency.py` | **新建** — 10-10 邻居表、fNIRS 通道名解析、mnt.mat 加载与校验、邻接矩阵构建、可视化 |
| `src/tokenizers/factorized_labram_vqnsp.py` | `_compute_eeg_source_target` 重写为 per-channel RMS；`_compute_fnirs_source_target` 重写为空间加权 HRF；`__init__` 新增可选参数与 spatial buffer |
| `experiments/scripts/train_source_observation_tokenizer.py` | 模型创建前注入通道名称到 config |
| `src/visualization/source_observation_analysis.py` | 集成空间邻接可视化 |
| `experiments/configs/source_observation/phase2a/` | 新增 `gate2_phase2a_spatial_target.yaml` |

#### 7.4.4 Backward Compatibility

所有新参数可空，默认回退到旧行为。旧 checkpoint 可正常加载（spatial buffer 为空时使用旧全局均值路径）。

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

Phase 1 Gate1 稳定化工作现已追加归档到：

```text
experiments/runs/archive/source_observation_phase1_gate1_stabilization_20260511/
experiments/configs/archive/source_observation_phase1_gate1_stabilization_20260511/
experiments/results/source_observation_index.json
docs/archive/logs/PHASE1_GATE1_STABILIZATION_20260511.md
```

这批 Phase 1 run 采用“in-place archive”方式封存：保留原始 run 目录路径不变，通过 manifest 和索引标记阶段闭环，避免破坏既有分析产物和 comparison reports 中写死的 provenance 路径。

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
| **Gate 2A: Quantization-Coupling Consistency** | coupling 先验是否能有效引导量化决策？ | token agreement > 1/K；Gate 1 不退化；fNIRS source utilization 稳定 |
| **Gate 3: Structure** | coupling tensor 是否表现出生理合理的 delay-aware 结构？ | delay entropy 低于均匀基线；neighbor JS gap > 0；可视化出现平滑 ridge |
| **Gate 4: Utility** | 表示空间是否有 downstream value？ | source SSR > 1.0；subject leakage 集中在 observation |

### Gate dependency

```
Gate 1 ──→ Gate 2 ──→ Gate 3 ──→ Gate 2A ──→ Gate 4
(Phase 1)  (Phase 2A) (Phase 2B) (Phase 2C)  (Phase 4+)
```

Phase 2A (Branch Target Redesign) 阻塞 Gate 2 (Semantics)。Phase 2B (Coupling Structure Priors) 阻塞 Gate 3 (Structure)。Phase 2C (Coupling-Aware Quantization) 阻塞 Gate 2A。

每个 Phase 只验证一个 Gate。不通过则阻塞，不回退到更早的 Gate。

### Promotion rule

任何机制要进入默认 mainline，必须同时满足：

1. Gate 1 (Health) 不退化；
2. Gate 2 (Semantics) 不退化（当前阻塞目标）；
3. Gate 3 (Structure) 有明确增益；
4. Gate 2A (Quantization-Coupling Consistency) 通过（Phase 2C 验证）；
5. Gate 4 (Utility) 不出现明显倒退；
6. 能通过 ablation 解释，不把 source branch 重新变成另一条全能重建捷径；
7. 与至少一类外部研究方法对照相比，能够给出清晰的结构性增益说明。

---

## 10. Implementation Order

当前严格执行以下顺序：

1. ~~完成 shared/private branch semantics audit~~ ✅ 已完成
2. ~~确定 source/observation redesign 机制定义~~ ✅ 已完成
3. ~~归档 shared/private 阶段实验产物，清理活跃配置与分析入口~~ ✅ 已完成
4. ~~直接改造主线 tokenizer / loss / registry / config surface，完成 Phase 1 Structural Migration~~ ✅ 已完成
5. ~~锁定 no-phase Gate1 baseline，并归档 Phase 1 Gate1 调参结果~~ ✅ 已完成
6. ~~实现 Phase 2 Source Target Introduction（HRF convolution model）~~ ✅ 已完成（Gate 2-4 均 fail，需要 Phase 2A 修复）
7. **实现 Phase 2A Branch Target Redesign + Dual Decoder Architecture（**当前阻塞目标 **）**
   7a. **实现 Phase 2A-spatial: Spatially-Informed Source Targets**（per-channel RMS envelope + 空间加权 fNIRS source target + 导联邻接模块，详见 [PHYSIOLOGICAL_COUPLING_PLAN.md §2.4](docs/PHYSIOLOGICAL_COUPLING_PLAN.md#24-source-branch-target-spatially-informed-hrf-convolution-model)）
8. **实现 Phase 2B Coupling Structure Priors（lag focus + joint smoothness）**
9. 实现 Phase 2C Cross-Modal Source Target + Coupling-Aware Quantization（延后）
10. 在 Phase 2B baseline 上独立实现并验证 Mechanism C（causal asymmetry）
11. 统一导出主线与外部方法对照结果到同一 summary schema
12. 更新 scorecard 与 experiment log
13. tokenizer 证据充分后，再考虑 foundation model 层面的目标替换

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
8. 本文件中的状态更新；
9. [架构修改日志](docs/architecture_changelog/INDEX.md) 条目（含 before/after 架构图、组件变更表、设计决策）；
10. 更新后的 [ARCHITECTURE.md](docs/ARCHITECTURE.md)（反映变更后的当前架构状态）。

没有文档和评价闭环的改动，不视为主线推进。

---

## 12. Explicitly Deferred Work

以下内容当前明确延后，不进入这轮实现主线：

1. A 与 C 同时启用的联合实验；
2. HRF-shaped lag weighting；
3. 显式熵 margin 式 asymmetry loss；
4. 重新引入 identity-style alignment losses；
5. foundation model 预训练目标的大改；
6. equal token count per window 的结构审计；
7. HRF 频带分解（delta/theta/alpha/beta/gamma）；
8. fNIRS→EEG 跨模态预测器（Phase 2C）；
9. Coupling-aware quantization（原 Phase 2A，现 Phase 2C）。

这些方向不是永久否定，而是必须等到 Phase 1-3 与 2A/A/C 单机制证据成立后再决定是否继续。

---

## 13. Bottom Line

当前项目的 tokenizer 主线已经从“证明 token 条件概率可分析”切换到“设计一个带有生理结构先验的离散表示机制”。

**当前主线决策非常明确**：

1. shared/private 将被 source/observation 直接取代；
2. 旧架构代码不会继续作为活跃对照面保留；
3. 历史可追溯性由 git 和 archive 提供；
4. 对照实验改由外部研究方法承担；
5. 分析、归档与结果格式规范全部以本文件为准。

**当前步骤**：Gate 1 已在 no-phase baseline 上稳定通过。Phase 2 (HRF Source Target) 已实现但 Gate 2-4 均 fail。Phase 2A (Branch Target Redesign + Dual Decoder) 已完成并实现 Gate 2 pass，但离线梯度诊断揭示了 source branch 扁平化的根本问题：跨通道均值 source target 导致 branch losses 主导梯度（59-67%）且多个目标共同推向低方差解。当前进入 **Phase 2A-spatial: Spatially-Informed Source Targets**，通过 per-channel RMS envelope + 空间加权 fNIRS neural driver 修复 branch 扁平化。

本文件即为当前实现顺序与准入标准的唯一主文档。