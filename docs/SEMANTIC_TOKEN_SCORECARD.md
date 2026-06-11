# Tokenizer Evaluation Gates

> Last Updated: 2026-06-04
> Status: Active — covers Croce local source/observation tokenizer training with highWL-only fNIRS input
> Architecture: [PHYSIOLOGICAL_COUPLING_PLAN.md](PHYSIOLOGICAL_COUPLING_PLAN.md) — source/observation branch semantics
> Supersedes: pre-S2 five-layer scorecard (Layer A-E) — archived by this rewrite

---

## 1. Why This Document Exists

S2 redesign 已将 tokenizer 主线从"事后分析条件概率"转为"设计带有生理结构先验的离散表示机制"。旧 scorecard 的五个 layer（A-E）中有大量指标是为 V6 的 shared/private + smooth_signal proxy 架构设计的，在 S2 中不再适用。

**旧 scorecard 的问题**：
- Layer B (ITSC, PSR, TPG, AC) 是 word2vec 时代"几何语义"的遗产——衡量 token 空间的内部结构是否优美，但从未建立与下游性能的因果联系
- Layer C (LMIG, CKG, SUB) 计算的是耦合的事后统计量，但 S2 的 concentration prior 已将耦合结构注入训练——这些指标变成了训练目标的回响，不是独立验证
- Layer E (gradient diagnostics) 为 V6 的 12-term loss 设计，S2 的 9-term loss 更简洁
- BRG 依赖 smooth_signal 产生的 common/residual target（已在 S2 中删除）

**新 scorecard 的设计原则**：
1. 每个 gate 必须对应一个明确的决策——"通过/不通过"意味着"进入下一阶段/回退修复"
2. 指标应尽可能从模型参数或训练损失中直接读取，不需要 heavy post-hoc 计算
3. 评估范式对齐当前领域共识：功能验证（tokenizer 是否做对事）优先于几何验证（token 空间是否优美）

---

## 2. S2 Semantic Target

S2 架构中，token 的语义目标分为两类：

| Token 类型 | 语义定义 | 验证方式 |
|-----------|----------|----------|
| **source token** (EEG side) | EEG 侧的神经血管耦合状态——可通过 coupling 预测 fNIRS token，可通过 HRF 模型重建 fNIRS | Semantic Gate |
| **source token** (fNIRS side) | fNIRS 侧的神经血管耦合状态——与 EEG source token 通过 coupling matrix 建立对应 | Semantic Gate |
| **observation token** (EEG) | EEG 的模态特异重建债务——不能被 fNIRS source 预测，对 subject 敏感 | Semantic Gate + Utility Gate |
| **observation token** (fNIRS) | fNIRS 的模态特异重建债务——不能被 EEG source 预测，对 subject 敏感 | Semantic Gate + Utility Gate |

判断 tokenization 成功与否的标准：**source tokens 编码了可跨模态预测的神经血管耦合状态，observation tokens 编码了不能跨模态预测的模态特异信息，耦合矩阵具有生理合理的集中结构。**

---

## 3. Evaluation Gates

每个 gate 是进入下一实现阶段的准入条件。Gate 0 不通过 → 不解释任何后续语义指标；Gate 1 不通过 → 不进入 Gate 2，以此类推。

### Gate 0: Cache/Input Contract

**回答的问题**：当前 run 是否真的使用了约定的 highWL-only Croce local cache 输入？

这是当前 highWL-only 训练范式的硬门槛。它不是训练效果指标，而是防止把 fNIRS 两个波长误读成两个物理通道、或把 optical measurement-space 信号误写成 HbO/HbR concentration。

| 指标 | 健康阈值 | 来源 |
|------|----------|------|
| **Selected fNIRS component** | `highWL` | `dataset.get_gate0_metadata()` |
| **Ignored fNIRS component** | includes `lowWL` | `dataset.get_gate0_metadata()` |
| **Cache pair mode** | `wavelength` | `cache_manifest.json` via dataset metadata |
| **Cache pair labels** | `["highWL", "lowWL"]` | `cache_manifest.json` via dataset metadata |
| **fNIRS layout** | `1 spatial anchor × 1 optical component = 1 channel` | config + dataset metadata |

**通过条件**：上述所有合同检查均通过。

**失败处理**：停止解释 Gate1-Gate4；先修复 cache source、component selector 或 config 的 `model.fnirs.*` 字段。

### Gate 1: Architecture Health

**回答的问题**：tokenizer 的基本组件是否健康？

这是所有语义讨论之前的硬性进入门槛。不通过此 gate 的 run 不应被解释其"语义"。

| 指标 | 计算 | 健康阈值 | 来源 |
|------|------|----------|------|
| **Codebook perplexity** (每个 quantizer) | $\exp(-\sum_k p_k \log p_k)$ | $\geq 0.3 \times K$ | 直接从 quantizer 统计 |
| **Active code ratio** | active codes / total codes | $\geq 0.5$ | 直接从 quantizer 统计 |
| **Dead code count** | $\sum_k \mathbf{1}[p_k = 0]$ | $\leq 0.3 \times K$ | 直接从 quantizer 统计 |
| **Top-5 coverage** | 使用最多的 5 个 code 的频率之和 | $\leq 0.5$ | 直接从 quantizer 统计 |
| **EEG full reconstruction** | MSE(raw, recon_full) | 随训练收敛 | 从 forward 输出读取 |
| **fNIRS full reconstruction** | MSE(raw, recon_full) | 随训练收敛 | 从 forward 输出读取 |

**通过条件**：所有 4 个 quantizer（eeg_source, fnirs_source, eeg_obs, fnirs_obs）均满足健康阈值，且 full reconstruction 收敛。

**失败处理**：检查 VQ 超参数（beta, decay, codebook size）、学习率、encoder 容量。

### Gate 2: Branch Semantics

**回答的问题**：source 和 observation 分支是否在做各自该做的事？

| 指标 | 计算 | 健康阈值 | 来源 |
|------|------|----------|------|
| **fNIRS source target MSE** | MSE(fnirs_source_recon, highWL source target) | 随训练下降，且显著低于 random baseline | 训练损失直接读取 |
| **EEG source target MSE** | MSE(eeg_source_recon, local EEG source target) | 随训练下降，且显著低于 random baseline | 训练损失直接读取 |
| **Observation contribution gap** | MSE(source_only_recon) - MSE(source+obs_recon) | $> 0$（observation 有正贡献） | 从 forward 输出计算 |
| **Source codebook independence** | eeg_source_quantizer 和 fnirs_source_quantizer 各自的利用率偏差 | $|u_{eeg} - u_{fnirs}| < 0.3$ | 从两个 quantizer 的 marginal 统计 |

**通过条件**：source target reconstruction 显著优于 random；observation gap > 0；两个 source codebook 都健康且没有明显利用率偏斜。

**失败处理**：
- source target 不收敛 → 检查 Croce cache 质量、source_target_weight、warmup schedule
- Observation gap ≈ 0 → observation branch 可能 collapse；检查 orthogonality weight
- EEG/fNIRS source codebook 利用率严重偏斜 → 检查 source quantizer balance、branch normalization、source branch 容量

### Gate 3: Coupling Structure

**回答的问题**：coupling matrix 是否表现出生理合理的信息结构？

| 指标 | 计算 | 健康阈值 | 来源 |
|------|------|----------|------|
| **Coupling row entropy** | $H_{row} = -\frac{1}{K}\sum_{i,j} T_{ij}\log T_{ij}$ | $< \log(K)/2$（显著非均匀） | 直接从 coupling_logits 计算 |
| **Concentration ratio** | $\frac{\max_j T_{ij}}{\text{mean}_j T_{ij}}$，按行平均 | $> 1.5$（行有明确峰值） | 直接从 coupling_logits 计算 |
| **Row entropy variance** | $\text{Var}(H(T_{i,:}))$ across rows | $> 0$（不同 source state 有不同的确定性） | 直接从 coupling_logits 计算 |
| **Cross-modal token predictability** | 给定 EEG source token 和 coupling tensor，预测 fNIRS source token 的 top-1 accuracy | $> 1/K_{src}$（random baseline） | source token 序列 + coupling tensor |
| **Coupling matrix visualization** | 按行熵排序的热力图 | 可见块状或带状结构 | 单次 matplotlib 可视化 |

**通过条件**：row entropy < log(K)/2；concentration ratio > 1.5；row entropy 的方差 > 0（不是所有行相同）；cross-modal token predictability 高于 chance。

**失败处理**：
- Row entropy ≈ log(K)（接近均匀）→ concentration_weight 太小，增大或检查 coupling_kl_loss 是否正常
- Concentration ratio < 1.5 → concentration prior 未生效，sweep weight
- 所有行熵相同 → coupling 可能 collapsed 到 trivial solution
- Cross-modal predictability ≈ random → coupling tensor 的结构没有对应到实际 EEG/fNIRS source token 序列

### Gate 4: Representation Utility

**回答的问题**：离散表示空间是否有 downstream value？

| 指标 | 计算 | 健康阈值 | 来源 |
|------|------|----------|------|
| **Subject leakage** (source branch) | 冻结 tokenizer，用 nearest-centroid probe 预测 subject ID 的 accuracy | 显著低于 observation branch 的 subject leakage | 独立 probe 脚本 |
| **Subject leakage** (observation branch) | 同上，但用 observation tokens | 应 > source branch | 独立 probe 脚本 |
| **Task signal** (source branch) | 冻结 tokenizer，用 nearest-centroid probe 预测 task/condition | $>$ chance | 独立 probe 脚本 |
| **Semantic selectivity ratio** | $\text{SSR} = \frac{\text{TCS}}{\text{SLS} + \epsilon}$ (for source branch) | $> 1.0$（task 信息 > subject 信息） | 由上述两项计算 |

**通过条件**：
- source branch 的 subject leakage < observation branch 的 subject leakage（subject identity 集中在 observation）
- source branch 的 SSR > 1.0（source 更关注 task，而非 subject）
- task signal > chance

**失败处理**：
- 两个分支的 subject leakage 相似 → orthogonality 或 branch semantics 可能失败
- SSR < 1.0 → source branch 可能记忆了 subject identity 而非 task-relevant coupling state

---

## 4. Gate Decision Protocol

每个实现 phase 只关注一个 gate。不通过则阻塞，不回退到更早的 gate。

```
Phase 0: Cache/Input Contract
  └── Gate 0: highWL-only Croce local cache contract
      ├── ✅ PASS → interpret architecture and semantic gates
      └── ❌ FAIL → fix cache/config semantics; do NOT interpret tokenizer metrics

Phase 1: Structural Migration
  └── Gate 1: Architecture Health
      ├── ✅ PASS → proceed to Phase 2
      └── ❌ FAIL → fix VQ/encoder; do NOT proceed

Phase 2A: Branch Target Redesign + Dual Decoder
  └── Gate 2: Branch Semantics
      ├── ✅ PASS → proceed to Phase 2B
      └── ❌ FAIL → fix HRF target or branch training; do NOT proceed

Phase 2B: Croce 2017 Physical Model + Coupling Structure Priors
  └── Gate 3: Coupling Structure
      ├── ✅ PASS → architecture validated; proceed to downstream evaluation (Gate 4)
      └── ❌ FAIL → sweep coupling prior weights; inspect lag focus + joint smoothness balance
```

### Cross-phase comparison rule

当比较 S1 (V6) 和 S2 时，**不要求 S2 在 Gate 4 上立刻超越 S1**。S2 的核心创新是 source/observation 语义的清晰性和 coupling 的生理结构化——这些由 Gate 2 和 Gate 3 验证。Gate 4 (utility) 是长期指标，受限于当前 probe 数据规模和下游任务设计。

比较 S1 vs S2 的有效口径：
1. Gate 1: 两者都应通过
2. Gate 2: S2 应有**定性更清晰**的 branch semantics（explicit source/observation target vs. smooth_signal proxy）
3. Gate 3: S2 应有**定量更结构化**的 coupling（concentration prior vs. free parameter）
4. Gate 4: 期望 S2 不倒退

---

## 5. S1 vs S2 Comparison Framework

| 维度 | S1 (V6) | S2 (Source/Observation) | 期望差异 |
|------|---------|------------------------|----------|
| **Branch naming** | shared / private | source / observation | — |
| **Source semantics** | smooth_signal(low-freq) proxy | HRF convolution model target | S2 语义更清晰 |
| **Observation semantics** | raw - smooth_signal residual | modality-specific reconstruction debt | S2 语义更明确 |
| **Source codebook** | 单一 shared quantizer (K=128) | 双独立 source codebook (K=128 each) | S2 不强制同一空间 |
| **Coupling** | 自由参数 `[K,K]` | concentration-constrained `[K,K]` | S2 结构化 |
| **Loss count** | 12 terms | 9 terms | S2 更简洁 |
| **Gate 1** | 通过 | 应通过 | 不应退化 |
| **Gate 2** | BRG (common/residual) | explicit source target + obs gap | 定性差异 |
| **Gate 3** | post-hoc CKG/LMIG | concentration metrics from logits | S2 应更结构化 |
| **Gate 4** | SLS/TCS on shared/private | SLS/TCS on source/obs | S2 不倒退 |

---

## 6. Report Contract

每个正式 mainline 实验至少输出以下内容：

```
Gate 0: Cache/Input Contract
  - selected_fnirs_component = highWL
  - ignored_fnirs_components includes lowWL
  - pair_mode = wavelength; pair_labels = [highWL, lowWL]
  - fNIRS layout = 1 spatial anchor × 1 highWL optical component

Gate 1: Architecture Health
  - 4 个 codebook 的 perplexity, utilization, dead codes, top-5 coverage
  - EEG/fNIRS full reconstruction MSE

Gate 2: Branch Semantics
  - Source target MSE（含 random baseline 对比）
  - Observation contribution gap (source_only vs source+obs)
  - Source codebook independence

Gate 3: Coupling Structure
  - Coupling row entropy（与 log(K) 的比值）
  - Concentration ratio
  - Cross-modal token predictability (from source tokens + coupling tensor)
  - Coupling matrix visualization（按行熵排序的热力图）

Gate 4: Representation Utility
  - Subject leakage (source vs observation branches)
  - Task signal (source branch)
  - Semantic selectivity ratio (source branch)

Comparison (if applicable)
  - S1 vs S2 gate-by-gate summary
  - Narrative: 是否 S2 在 branch semantics (Gate 2) 和 coupling structure (Gate 3) 上
    提供了比 V6 更清晰、更有生理依据的离散表示
```

---

## 7. What Is Deliberately Removed

以下内容从旧 scorecard 中明确删除：

| 删除项 | 删除理由 |
|--------|----------|
| ITSC (intra-token state consistency) | word2vec 几何语义遗产；无明确决策阈值；依赖 feature extractor 选择 |
| PSR (prototype separation ratio) | 同上；token 原型分离度对下游 performance 无已知因果关系 |
| TPG (transition predictability gain) | 过渡结构重要但应由 downstream probing 验证，非 tokenizer 层指标 |
| AC (augmentation consistency) | 对 S2 意义不大；HRF target 已提供更强的一致性约束 |
| BRG (branch responsibility gap) | 依赖 V6 的 smooth_signal common/residual target；在 S2 中无对应概念 |
| LMIG (lagged MI gain) | 被 coupling concentration metrics (Gate 3) 替代——从 coupling_logits 直接读取 |
| CKG (conditional KL gain) | 同上；CKG 本质上 = coupling_kl_loss 的 post-hoc 计算，是训练目标回响 |
| SUB (shared usage balance) | Dual source codebook 后不再有"共享垄断"问题 |
| Overlap / token match rate | 旧共识已否定；TokenFlow analysis 确认不应追求 token identity overlap |
| Gradient diagnostics (Layer E) | 为 V6 的 12-term loss 设计；S2 的 9-term loss 更简洁，不需要此层 |
| Session/device stability (SDS) | P2 级别，从未实现；延后至 foundation model 层面评估 |

---

## 8. Bottom Line

S2 的评估逻辑与 V6 有根本性差异：

- **V6**：tokenizer 产生离散 token → 事后分析 token 统计量 → 从分析中推断"语义可能存在"
- **S2**：tokenizer 设计时注入生理先验（HRF target, concentration prior）→ 验证这些先验是否确实塑造了表示 → Gate 1-3 验证训练是否按预期工作，Gate 4 验证表示是否有 downstream value

这意味着 S2 的"语义"不是从事后分析中发现的——它是在训练目标中构建的。Scorecard 的角色从"发现语义"变为"验证设计是否生效"。
