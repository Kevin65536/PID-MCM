# Physiological Coupling Constraints for EEG-fNIRS Tokenizer

> Created: 2026-04-30
> Status: Active development plan — next-stage tokenizer innovation
> Supersedes: [NEXT_STAGE_ALIGNMENT_PLAN.md](NEXT_STAGE_ALIGNMENT_PLAN.md) (archived as historical reference)
> Reference implementation: [src/tokenizers/codebook_focus_factorized_labram_vqnsp.py](../src/tokenizers/codebook_focus_factorized_labram_vqnsp.py)

---

## 1. Motivation

### 1.1 Problem statement

当前 V6 mainline（codebook-focused factorized tokenizer）已经建立了健康的 shared/private factorization，并通过 lag-aware coupling 矩阵提供 EEG 与 fNIRS token 分布之间的条件概率。但 coupling 矩阵本身是**完全自由的参数化**——一个 `[n_lags, K, K]` 的 `nn.Parameter`，初始化为零，没有任何生理先验指导它应该呈现什么结构。

这导致了创新声明上的薄弱："通过离散化提供 EEG 与 fNIRS 生理模式之间的条件概率"本质上只是一个**被动观察**——我们发现了一个统计量，而不是我们设计了一个机制。在神经信号处理领域，仅仅提供"分析能力"不足以作为核心创新点。

### 1.2 Core idea

在 coupling 机制中引入**生理指导的结构约束**，将创新从"我们发现条件概率存在"转变为**"我们设计了一个体现神经血管耦合原理的离散表示机制"**。

关键设计原则（继承自 V6 reset 的结论）：

1. 约束施加在 coupling 参数上，**不直接触碰 encoder/quantizer 的梯度路径**——不与 reconstruction 形成对抗
2. 约束是**软先验**（小系数正则项），不是硬约束——允许数据覆盖先验
3. 约束表达的是**已知的生理结构**，而不是统计优化的目标

### 1.3 Innovation narrative

| 旧声明 | 新声明 |
|--------|--------|
| "离散化揭示了 EEG-fNIRS 之间的条件概率" | "我们在离散化过程中引入了神经血管耦合的结构先验" |
| 被动观察 | 主动机制设计 |
| 一个统计分析工具 | 一个体现生理原理的表示学习框架 |

---

## 2. Mechanism A: Token-Space Coupling Smoothness

### 2.1 Physiological basis

神经血管耦合的基本性质：**相近的神经活动状态引起相近的血流动力学响应**。

如果两个 EEG token 在 shared codebook 空间中代表相似的神经状态，它们经由 coupling 矩阵映射到的 fNIRS token 分布也应该是相似的。当前自由参数化的 coupling 矩阵不保证这一性质——两个 codebook 向量几乎相同的 token 可能学到完全不同的耦合分布。

### 2.2 Mathematical formulation

设 shared codebook 的归一化权重为 $C \in \mathbb{R}^{K \times D}$。对每个 token $i$，找到其在 codebook 空间中的 $M$ 个最近邻 $\mathcal{N}(i)$（基于余弦相似度）。

对于给定的 lag，coupling 矩阵 $T = \text{softmax}(\text{coupling\_logits}[lag]) \in \mathbb{R}^{K \times K}$（行随机矩阵），定义平滑性损失：

$$\mathcal{L}_{smooth}(T) = \frac{1}{K \cdot M} \sum_{i=1}^{K} \sum_{j \in \mathcal{N}(i)} D_{JS}\big(T_{i,:} \,\|\, T_{j,:}\big)$$

其中 $D_{JS}(P \| Q) = \frac{1}{2} D_{KL}(P \| M) + \frac{1}{2} D_{KL}(Q \| M)$，$M = (P + Q) / 2$。

**选择 JS 散度而非 L2 的理由**：
- $T_{i,:}$ 是概率分布，JS 散度有界且对称
- JS 散度对低概率区域的微小波动不敏感，避免被噪声 token 主导

**选择局部邻居而非全局平滑的理由**：
- 不假设 codebook 的全局拓扑结构已经学到生理上有意义的组织
- 局部约束更稳健：只要求最相似的 token 有相似耦合行为
- 计算量可控（M=5 时每个 token 只计算 5 个 pair）

### 2.3 Implementation

新文件或修改位置：

**`src/losses/multimodal_tokenizer.py`** — 新增函数：

```python
def coupling_smoothness_loss(
    coupling_logits: torch.Tensor,       # [n_lags, K, K]
    codebook_weight: torch.Tensor,       # [K, D]
    n_neighbors: int = 5,
) -> torch.Tensor:
    """Encourage tokens with similar codebook vectors to have similar coupling profiles."""
    n_lags, K, _ = coupling_logits.shape
    
    # Find local neighbors in codebook space
    normed = F.normalize(codebook_weight, dim=-1)
    sim = normed @ normed.t()
    _, neighbors = sim.topk(n_neighbors + 1, dim=-1)
    neighbors = neighbors[:, 1:]  # [K, n_neighbors], exclude self
    
    total_loss = normed.new_tensor(0.0)
    for lag_idx in range(n_lags):
        T = F.softmax(coupling_logits[lag_idx], dim=-1)        # [K, K]
        T_neighbors = T[neighbors]                              # [K, M, K]
        T_i = T.unsqueeze(1)                                    # [K, 1, K]
        M = 0.5 * (T_i + T_neighbors)                           # [K, M, K]
        js = 0.5 * (
            F.kl_div((T_i + 1e-8).log(), M, reduction='none').sum(dim=-1) +
            F.kl_div((T_neighbors + 1e-8).log(), M, reduction='none').sum(dim=-1)
        )
        total_loss = total_loss + js.mean()
    
    return total_loss / n_lags
```

**`src/tokenizers/factorized_labram_vqnsp.py`** — 新增参数：

```python
# In __init__:
coupling_smoothness_weight: float = 0.0,
coupling_smoothness_neighbors: int = 5,
```

**`compute_factorized_shared_alignment_losses`** — 新增返回：

```python
'smoothness_loss': coupling_smoothness_loss(...) if enabled else zero_tensor
```

### 2.4 Expected behavioral signatures

| 指标 | 预期变化 | 验证方式 |
|------|----------|----------|
| $\mathcal{L}_{smooth}$ | 随训练下降 | 直接监控 |
| Coupling 矩阵可视化 | 按 codebook 相似度排序后呈现平滑结构 | TensorBoard image |
| Token neighborhood coupling consistency | 邻居 token 的耦合分布 JS 散度低于随机基线 | 定量比较 |
| Reconstruction | 无显著变化 | MSE / STFT |
| Codebook health | 无显著退化（可能轻微改善，因为耦合结构更清晰） | Perplexity / utilization |

### 2.5 Failure modes

1. **Codebook 未收敛时无意义**：如果 codebook 本身还在剧烈变化，邻居关系不稳定，平滑性约束会引入噪声。缓解：在 reconstruction 稳定后再 warm-start 此约束。
2. **系数过大导致所有 token 耦合相同**：如果 $\lambda_{smooth}$ 过大，所有行收敛到相同分布。缓解：从小系数（0.005）开始，监控行间方差。
3. **Codebook collapse 时退化为无操作**：如果所有 codebook 向量都相似，邻居没有意义。缓解：此约束假设 Layer A health gates 已通过。

---

## 3. Mechanism C: Causal Direction Asymmetry

### 3.1 Physiological basis

神经血管耦合的因果方向在试次时间尺度（~10s）上是明确的：

- **EEG → fNIRS**：电活动 → 代谢需求 → 血管扩张 → HRF（延迟 2-8s），这是主要的因果通路
- **fNIRS → EEG**：反向因果在此时窗内很弱。血管状态对神经兴奋性的调节（通过 CO₂、pH）发生在更慢的时间尺度上

当前 V6 的 bidirectional coupling 实现中，反向耦合使用前向耦合矩阵的转置（[factorized_labram_vqnsp.py:251](src/tokenizers/factorized_labram_vqnsp.py#L251)）：

```python
reverse_transition = F.softmax(coupling_logits[lag_index].transpose(0, 1), dim=-1)
```

这意味着 $P(\text{fNIRS}_j \mid \text{EEG}_i) \propto P(\text{EEG}_i \mid \text{fNIRS}_j)$——两个方向的耦合共享同一组参数。这在生理上是不合理的：EEG→fNIRS 的预测结构应该比 fNIRS→EEG 更集中、更有组织性。

### 3.2 Design principle: asymmetric prior, not asymmetric loss

不引入显式的"前向必须比反向更集中"的损失项。而是：

1. 为两个方向使用**独立参数矩阵**（`coupling_logits_fwd` 和 `coupling_logits_rev`）
2. 仅在**前向**（EEG→fNIRS）施加结构约束（如机制 A 的平滑性）
3. 反向保持自由参数化，让数据决定其结构
4. 通过诊断指标（asymmetry ratio）观察两个方向的差异

这种"不对等先验"方案比显式不对称损失更干净：它不给优化器增加对抗性约束，而是通过**不对等的参数化自由度和正则化水平**让生理结构自然浮现。

### 3.3 Mathematical formulation

**参数独立化**：

前向（EEG → fNIRS）：
$$T^{fwd}_l = \text{softmax}(W^{fwd}_l), \quad W^{fwd}_l \in \mathbb{R}^{K \times K}$$

反向（fNIRS → EEG）：
$$T^{rev}_l = \text{softmax}(W^{rev}_l), \quad W^{rev}_l \in \mathbb{R}^{K \times K}$$

其中 $W^{fwd}_l$ 和 $W^{rev}_l$ 是独立参数，初始化为零。

**不对等处理**：

前向耦合损失（可加机制 A 平滑约束）：
$$\mathcal{L}_{fwd} = \mathcal{L}_{coupling}(T^{fwd}) + \lambda_{smooth} \cdot \mathcal{L}_{smooth}(T^{fwd})$$

反向耦合损失（自由参数化，无结构约束）：
$$\mathcal{L}_{rev} = \mathcal{L}_{coupling}(T^{rev})$$

总耦合损失：
$$\mathcal{L}_{coupling}^{total} = 0.5 \cdot (\mathcal{L}_{fwd} + \mathcal{L}_{rev})$$

**诊断指标**：

$$\text{asymmetry\_ratio} = \frac{\mathbb{E}_k[H(T^{rev}_{k,:})]}{\mathbb{E}_k[H(T^{fwd}_{k,:})]}$$

其中 $H(\cdot)$ 是行分布的熵。期望 asymmetry_ratio > 1.0（反向比前向更分散）。

### 3.4 Implementation

**`src/tokenizers/factorized_labram_vqnsp.py`** — 参数变更：

```python
# In __init__:
coupling_asymmetric: bool = False,  # toggle for mechanism C

# Replace single coupling_logits:
self.coupling_logits = nn.Parameter(...)  # kept for backward compat when asymmetric=False

# If asymmetric:
self.coupling_logits_fwd = nn.Parameter(
    torch.zeros(len(self.alignment_lag_candidates), shared_codebook_size, shared_codebook_size)
)
self.coupling_logits_rev = nn.Parameter(
    torch.zeros(len(self.alignment_lag_candidates), shared_codebook_size, shared_codebook_size)
)
```

**`src/losses/multimodal_tokenizer.py`** — 修改 `compute_factorized_shared_alignment_losses`：

```python
def compute_factorized_shared_alignment_losses(
    ...,
    coupling_logits: torch.Tensor | None = None,       # legacy shared param
    coupling_logits_fwd: torch.Tensor | None = None,    # mechanism C: forward
    coupling_logits_rev: torch.Tensor | None = None,    # mechanism C: reverse
    coupling_asymmetric: bool = False,
    ...
):
    for lag_index, lag in enumerate(alignment_lag_candidates):
        ...
        # Forward coupling
        if coupling_asymmetric:
            transition_fwd = F.softmax(coupling_logits_fwd[lag_index], dim=-1)
        else:
            transition_fwd = F.softmax(coupling_logits[lag_index], dim=-1)
        pred_fnirs_probs = torch.einsum('bnk,kl->bnl', aligned_eeg_probs, transition_fwd)
        coupling_loss = coupling_kl_loss(pred_fnirs_probs, aligned_fnirs_probs)
        
        # Reverse coupling
        if coupling_bidirectional:
            if coupling_asymmetric:
                transition_rev = F.softmax(coupling_logits_rev[lag_index], dim=-1)
            else:
                transition_rev = F.softmax(coupling_logits[lag_index].transpose(0, 1), dim=-1)
            pred_eeg_probs = torch.einsum('bnk,kl->bnl', aligned_fnirs_probs, transition_rev)
            coupling_loss = 0.5 * (coupling_loss + coupling_kl_loss(pred_eeg_probs, aligned_eeg_probs))
        ...
```

**诊断指标**（在 tokenizer forward 中新增）：

```python
if self.coupling_asymmetric:
    with torch.no_grad():
        T_fwd = F.softmax(self.coupling_logits_fwd[selected_lag_idx], dim=-1)
        T_rev = F.softmax(self.coupling_logits_rev[selected_lag_idx], dim=-1)
        h_fwd = -(T_fwd * (T_fwd + 1e-8).log()).sum(dim=-1).mean()
        h_rev = -(T_rev * (T_rev + 1e-8).log()).sum(dim=-1).mean()
        asymmetry_ratio = h_rev / (h_fwd + 1e-8)
```

### 3.5 Expected behavioral signatures

| 指标 | 预期变化 | 验证方式 |
|------|----------|----------|
| asymmetry_ratio | 稳定 > 1.0 | 直接监控 |
| EEG→fNIRS coupling per-row entropy | 低于反向 | 分布直方图 |
| CMTP EEG→fNIRS vs fNIRS→EEG | 前向优于反向 | 下游预测任务 |
| Reconstruction | 无显著变化 | MSE / STFT |
| Codebook health | 无显著退化 | Perplexity / utilization |

### 3.6 Failure modes

1. **asymmetry_ratio ≈ 1.0**：数据中两个方向的信息流确实对称，或反向参数学到的结构与前向类似。这不是严格意义上的"失败"——它说明数据不支持神经血管耦合的不对称假设。这仍然是有价值的发现。
2. **反向耦合退化**：如果 fNIRS→EEG 的耦合损失变得很大（远大于前向），可能是因为反向参数未被充分优化。缓解：确保两个方向的 coupling loss权重相同，不对反向施加额外的压制。

---

## 4. Experimental Design

### 4.1 Independent experiments (not combined)

在当前研究阶段，机制 A 和机制 C **分别进行实验**，不组合使用。每个机制的实验独立于 V6 baseline 进行比较。

### 4.2 Experiment ladder

```
                        ┌── V6 Baseline (current mainline)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   Exp A1           Exp C1          (future)
   coupling_        coupling_       A + C
   smoothness_      asymmetric=     combined
   weight sweep     True
        │               │
   Exp A2           Exp C2
   A + warm-        C + A
   start schedule   smoothness
```

### 4.3 Exp A: Coupling smoothness

**Config changes** (relative to V6 baseline):

```yaml
loss:
  alignment:
    coupling_smoothness_weight: [0.005, 0.01, 0.02]  # sweep
    coupling_smoothness_neighbors: 5
    coupling_asymmetric: false
```

**Warm-start schedule**: 在 reconstruction 稳定后（通常 epoch 20-30）才启用 smoothness 约束：

```yaml
loss:
  alignment:
    coupling_smoothness_warmup_epochs: 30
    coupling_smoothness_final_weight: 0.01
```

**Comparison metrics vs. V6 baseline**:

1. Layer A: reconstruction (full/common/residual), codebook health
2. Layer B: intra-token consistency, prototype separation ratio
3. Layer C: best-lag MI, conditional KL gain, coupling matrix structure
4. Layer D: subject leakage, task signal

**Decision gate**:

- ✅ Pass: Layer A 不退化 + coupling 矩阵呈现可辨识的平滑结构 + ≥1 项 Layer C 指标改善
- ❌ Fail: Layer A 退化，或 coupling 矩阵无明显结构改善，或无任何 Layer B/C 指标改善

### 4.4 Exp C: Causal asymmetry

**Config changes** (relative to V6 baseline):

```yaml
loss:
  alignment:
    coupling_asymmetric: true
    coupling_bidirectional: true  # keep bidirectional, but with separate params
```

**Comparison metrics vs. V6 baseline**:

1. Layer A: reconstruction, codebook health
2. Layer C: asymmetry_ratio, forward vs. reverse coupling entropy, CMTP direction comparison
3. Layer D: subject leakage, task signal

**Decision gate**:

- ✅ Pass: asymmetry_ratio 稳定 > 1.0 + Layer A 不退化 + 前向 CMTP 优于反向
- ⚠️ Inconclusive: asymmetry_ratio ≈ 1.0 但 Layer A/C 不退化（说明数据不支持不对称先验）
- ❌ Fail: Layer A 退化

### 4.5 What NOT to do

- ❌ 同时启用机制 A 和机制 C（当前阶段）
- ❌ 在 shared codebook baseline 上测试这些机制（它们依赖 factorization）
- ❌ 在没有 warm-start 的情况下直接启用以 reconstruction 为主的 run
- ❌ 把 coupling 结构改善当作唯一的成功指标——Layer A（reconstruction/codebook health）是前提

---

## 5. Integration Plan

### 5.1 Code changes summary

| 文件 | 变更 | 机制 |
|------|------|------|
| `src/losses/multimodal_tokenizer.py` | 新增 `coupling_smoothness_loss()` | A |
| `src/losses/multimodal_tokenizer.py` | `compute_factorized_shared_alignment_losses` 支持独立 fwd/rev coupling logits | C |
| `src/tokenizers/factorized_labram_vqnsp.py` | 新增 `coupling_smoothness_weight`, `coupling_smoothness_neighbors` | A |
| `src/tokenizers/factorized_labram_vqnsp.py` | 新增 `coupling_asymmetric`, `coupling_logits_fwd`, `coupling_logits_rev` | C |
| `src/tokenizers/factorized_labram_vqnsp.py` | Forward 中新增 asymmetry_ratio 诊断 | C |
| `src/tokenizers/codebook_focus_factorized_labram_vqnsp.py` | 透传新参数（默认 smoothness=0, asymmetric=False） | A+C |
| Config YAML | 新增 `coupling_smoothness_*` 和 `coupling_asymmetric` 字段 | A+C |

### 5.2 Backward compatibility

- `coupling_smoothness_weight=0.0` → 行为与 V6 完全相同（机制 A 默认关闭）
- `coupling_asymmetric=False` → 行为与 V6 完全相同（机制 C 默认关闭）
- 现有 config 无需修改即可运行

---

## 6. Success Criteria & Promotion Rule

任何机制要进入默认 mainline，必须满足（继承自 V6 reset 的 promotion rule）：

1. Layer A codebook health 不退化
2. Layer B semantic state quality 不退化，最好有明确提升
3. Layer C shared-branch structured value 有明确提升
4. Layer D invariance / downstream sanity 不出现明显倒退
5. 可以通过 ablation 解释，且不把 shared branch 变回第二条全能重建捷径

---

## 7. Relationship to Foundation Model

本计划聚焦于 tokenizer 层面的 coupling 约束。与 foundation model pretraining 的关系：

- Tokenizer 的 coupling 约束提供 token 级的生理结构化先验
- Foundation model 的跨模态目标（当前为 InfoNCE）可随后调整为利用 coupling 先验的 Cross-modal Masked Token Prediction
- Tokenizer 层的约束和 pretraining 层的调整是**正交的**：当前实验先验证 tokenizer 层约束的效果，再决定 pretraining 层是否需要调整

详见后续文档（待 A/C 实验结果后撰写）。

---

## Appendix: Rejected Approaches

### HRF-shaped lag weighting

考虑过使用 SPM  HRF 形状的先验权重替代当前的 `alignment_selection='min'`，使不同时间偏移的耦合损失按 HRF 幅度加权。**放弃理由**：

- 类似物理约束神经网络的经验表明，硬编码的波形形状先验在实际数据上往往不匹配（个体间 HRF 变异性大，被试-被试、试次-试次差异显著）
- 10s 窗口仅 5 patches (lag 0-4)，离散化后的 HRF 先验信息量有限
- 当前 `alignment_selection='min'` 已经允许模型在每个 batch 选择最优 lag，不需要额外的 temporal 先验
- 如果将来需要，可作为扩展方向重新评估

### Explicit asymmetry loss

考虑过显式约束 `H(T_fwd) < H(T_rev)` 的 margin-based loss。**放弃理由**：

- 额外的对抗性约束增加了优化复杂度
- "不对等先验"方案（独立参数 + 仅对前向加结构约束）在概念上更干净
- 如果不对等先验已能产生 asymmetry_ratio > 1.0，显式损失是冗余的
