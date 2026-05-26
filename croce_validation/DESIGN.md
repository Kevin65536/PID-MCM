# Croce-Style SSM: Observation-Constrained r(t), No Endogenous Dynamics

> 最终方案：r(t) 无内生动力学。EEG 提议，fNIRS 选择。
> 状态空间保持 5D，粒子滤波结构根本性修改。

---

## 1. 设计原理

### 1.1 r(t) 的三条规则

1. **初始化**：r⁰(t) = L⁺ y_eeg(t)（EEG 确定性投影）
2. **无内生动力学**：r(t) 不受 OU、随机游走或任何转移密度的约束
3. **双观测约束**：r(t) 仅被两个外部力塑形——EEG 观测和 fNIRS 观测

### 1.2 "EEG 提议，fNIRS 选择" 机制

在 PF 的每一步，r(t) 的粒子不从前一步的 r(t-1) 转移而来，而是**从 EEG 观测重新提议**：

$$r^{(i)}(t) \sim q(r \mid y_{eeg}(t)) = \mathcal{N}\left(L^+ y_{eeg}(t),\ \sigma_{prop}^2\right)$$

然后，fNIRS 似然决定哪些粒子存活：

$$w^{(i)} \propto p\left(y_{nirs}(t) \mid \mathbf{x}^{(i)}_{hemo}(t)\right)$$

**σ_prop 不是噪声参数**——它是搜索带宽。它控制 r(t) 可以在多大范围内偏离 EEG 解以探索更好的 fNIRS 拟合。这是纯粹的优化超参数，不需要概率解释。

### 1.3 为什么 fNIRS 可以影响 r(t)

不同粒子有不同的**血流动力学状态历史** [s, Δf, ΔHbO, ΔHb]——这些状态是过去所有 r 值的累积积分。即使当前步所有粒子的 r 值都在 L⁺ y_eeg 附近，它们的血流动力学状态不同，导致不同的 fNIRS 预测。

fNIRS 权重选择那些血流动力学状态与观测最匹配的粒子。因为血流动力学状态编码了 r 的历史，选择血流动力学状态就是在选择 **r 的轨迹**。

**综合机制**：EEG 确定 r 应该在何处（提议中心）。fNIRS 确定哪些 r 轨迹是生理上合理的（粒子权重）。后验 r̂(t) 是 EEG 提议与 fNIRS 选择的加权折中。

---

## 2. 状态空间与动力学

### 2.1 状态向量

$$\mathbf{x}(t) = [s(t),\ \Delta f(t),\ \Delta HbO(t),\ \Delta Hb(t),\ r(t)]^T \quad (5D)$$

### 2.2 血流动力学方程（与 Croce 2017 相同）

$$\frac{ds}{dt} = \epsilon \cdot r(t) - k_{as} \cdot s(t) - k_{af} \cdot (f(t) - 1)$$

$$\frac{d(\Delta f)}{dt} = s(t)$$

$$\frac{d(\Delta HbO)}{dt} = \frac{f(t) - HbO(t)^{1/\alpha}}{\tau_0}$$

$$\frac{d(\Delta Hb)}{dt} = \frac{f(t) \cdot E(f(t), E_0)/E_0 - HbO(t)^{1/\alpha - 1} \cdot Hb(t)}{\tau_0}$$

参数：ε=1.0, k_as=0.41, k_af=0.65, τ₀=2.0, α=0.32, E₀=0.34

### 2.3 r(t) 方程

**无。**

r(t) 没有微分方程。在 drift 函数中，dr/dt = 0（r 在确定性积分步骤中不变）。r 的变化完全来自 PF 的提议步骤。

### 2.4 过程噪声

$$\omega_t \sim \mathcal{N}\left(0,\ \text{diag}([q_s, q_f, q_{hbo}, q_{hb}, 0]) \cdot dt\right)$$

**r 维度无过程噪声。** 仅血流动力学状态有过程噪声（q_s=(0.02)², q_f=q_hbo=q_hb=(0.015)²）。

---

## 3. 前向模型（确定性，无观测噪声注入）

### 3.1 源位置

所有神经源位于 fNIRS 通道位置。每个源独立估计一个 r_k(t)。

### 3.2 EEG 前向模型

$$\hat{y}_{eeg}(t) = L \cdot r(t)$$

**L** 的构造：
- Case A（局部）：仅空间邻近（R<60mm）的 EEG 通道参与
- Case B（全脑）：所有 EEG 通道参与，距离加权
- 距离衰减：$w_i = \exp(-||p_i - p_{anchor}||^2 / (2\sigma_{spatial}^2))$
- 符号：协方差模式
- 归一化：$L_{nearest} = 1$（anchor 归一化）

### 3.3 fNIRS 前向模型

$$\hat{y}_{nirs, \lambda}(t) = \kappa \cdot \left(\epsilon_\lambda^{HbO} \cdot \Delta HbO(t) + \epsilon_\lambda^{Hb} \cdot \Delta Hb(t)\right)$$

- 源与 fNIRS optode 同位置，无需 J 空间权重
- κ：无量纲 → 测量单位映射

---

## 4. 粒子滤波

### 4.1 核心修改

| 组件 | 原 Croce PF | 新设计 |
|------|------------|--------|
| r 的转移 | r_t = φ·r_{t-1} + ω_r（OU） | r ~ N(L⁺ y_eeg, σ²_prop)（EEG提议） |
| r 的过程噪声 | q_r = σ² | 0 |
| EEG 似然 | \|\|y_eeg - L·r\|\|²/σ²_eeg | **无**（EEG 通过提议进入） |
| fNIRS 似然 | 有 | 有（权重计算的核心） |
| σ_eeg 参数 | 有 | **无** |

### 4.2 算法

```
输入: y_eeg[t] at 200 Hz, y_nirs[t_nirs] at fNIRS 速率
       L (lead field)

超参数:
    σ_prop: r 提议带宽 (μV), 典型值 1-3
    σ_nirs: fNIRS 观测噪声, 数据集相关
    N: 粒子数, 典型值 300-500

初始化:
    r_eeg[t] = L⁺ · y_eeg[t]  对所有 t（EEG 确定性投影）
    x^(i) ~ 先验分布 (仅血流动力学状态随机, r=0)

For t_nirs = 0 .. T_nirs-1:

    # === 1. 20 个子步积分 (dt=0.005s) ===
    For substep = 1 .. 20:
        t_eeg = t_nirs * 20 + substep

        For each particle i:
            # ═══ 核心：从 EEG 重提议 r ═══
            r^(i) ~ N(L⁺ · y_eeg[t_eeg], σ²_prop)

            # 血流动力学积分（确定性 drift + 过程噪声）
            r_input = r^(i)
            x^(i)[0:4] = local_linearized_step_4d(x^(i)[0:4], dt=0.005, r=r_input)
            x^(i)[0:4] += ω^(i)[0:4] ~ N(0, Q_hemo · 0.005)
            Clip Δf, ΔHbO, ΔHb to [-0.95, +∞)

    # === 2. fNIRS 似然（仅此一项决定权重）===
    For each particle i:
        ŷ_high = κ · (1.00 · ΔHbO^(i) + 0.25 · ΔHb^(i))
        ŷ_low  = κ · (0.35 · ΔHbO^(i) + 1.00 · ΔHb^(i))
        log_w^(i) += -||y_high[t_nirs] - ŷ_high||² / (2·σ²_nirs)
                    -||y_low[t_nirs]  - ŷ_low||²  / (2·σ²_nirs)

    # === 3. 归一化与重采样 ===
    w = softmax(log_w)
    ESS = 1 / Σ (w^(i))²
    If ESS < 0.5·N: systematic_resample; w = 1/N

    # === 4. 后验估计 ===
    x̂[t_nirs] = Σ w^(i) · x^(i)

# --- 后处理：输出 r̂(t) 的完整时间序列 ---
对于 PF 在每个 fNIRS 步输出的 r̂，在 fNIRS 步之间线性插值，
得到 200 Hz 的 r̂(t)（用于与 EEG 对齐的 source target 计算）
```

### 4.3 关键细节

**r 只在每个 200Hz 子步被提议，不在 fNIRS 似然步被提议。** r 在 0.005s 时间尺度上跟踪 EEG，血流动力学在 0.1s 间隔累积 r 的影响。

**σ_prop 的校准**：
- σ_prop 过小：r 几乎等于 L⁺ y_eeg，fNIRS 无法影响 r
- σ_prop 过大：r 变得噪声化，失去与 EEG 的相似性
- 推荐初始值：σ_prop = 2 μV（约等于 EEG 通道噪声水平，提供适度的探索范围）

**r 提议在子步内是独立的**——每次提议都从 N(L⁺ y_eeg, σ²_prop) 新采样，不与上一步的 r 相关。粒子间的差异来自：
1. 每次提议的随机性（σ_prop）
2. 血流动力学状态历史的差异（被 fNIRS 选择）

### 4.4 为什么这个设计是合理的

**r(t) 的自发行为**：由于每次提议中心在 L⁺ y_eeg(t)，r̂(t) 的期望轨迹自然紧密追随 EEG。σ_prop 提供了有限的探索空间。

**fNIRS 的约束途径**：考虑时刻 t，粒子 i 和 j 有不同的血流动力学历史。fNIRS 似然选择血流动力学状态更匹配的粒子。由于血流动力学状态是**过去所有 r 提议的积分**，选择血流动力学状态就是在隐式地选择"更好的 r 历史"。存活到时刻 t 的粒子，其 r 轨迹在低频成分上与 fNIRS 一致。

**频率分离是涌现的**：
- 高频：r 紧密追随 L⁺ y_eeg（σ_prop 相对于 EEG 变化幅度小，且 fNIRS 对高频 r 变化不敏感）
- 低频：粒子间血流动力学状态差异主要来自 r 的低频历史差异；fNIRS 选择产生系统的低频选择压力

---

## 5. Source / Observation Target

### 5.1 Source Target（无噪声生理信号）

EEG：$\hat{y}_{eeg}^{source}(t) = L \cdot \hat{r}(t)$

fNIRS：$\hat{y}_{nirs,\lambda}^{source}(t) = \kappa \cdot (\epsilon_\lambda^{HbO} \cdot \Delta\widehat{HbO}(t) + \epsilon_\lambda^{Hb} \cdot \Delta\widehat{Hb}(t))$

### 5.2 Observation Target

$y_{eeg}^{obs}(t) = y_{eeg}^{raw}(t) - \hat{y}_{eeg}^{source}(t)$

$y_{nirs}^{obs}(t) = y_{nirs}^{raw}(t) - \hat{y}_{nirs}^{source}(t)$

---

## 6. 对 `run_local_neighborhood_solver_audit.py` 的修改

### 6.1 `state_drift()` — r 维度零漂移

```python
def state_drift(x, params):
    s, delta_f, delta_hbo, delta_hb, r = x
    # ... (血流动力学部分不变) ...
    dr = 0.0              # ← 仅此一行改变
    return np.asarray([ds, d_delta_f, d_delta_hbo, d_delta_hb, dr])
```

### 6.2 `state_jacobian()` — Jacobian 第5行全零

第5行（r 对全部状态的偏导）全为零。r 不参与确定性动力学，不影响 Jacobian 的其他部分。

### 6.3 `run_particle_filter()` — 核心重构

原代码中 r 通过 `local_linearized_step` + `process_noise` 传播。修改为：

```python
# 在 PF 循环中的每个子步：
for idx in range(num_particles):
    # --- r 从 EEG 观测提议 ---
    r_proposal = r_eeg_obs[t_eeg] + sigma_prop * rng.normal()
    particles[idx, 4] = r_proposal

    # --- 血流动力学状态正常积分 ---
    particles[idx, 0:4] = local_linearized_step_4d(
        particles[idx, 0:4], integration_dt, params, r=r_proposal
    )
    # 血流动力学过程噪声
    particles[idx, 0:4] += hemo_noise_std * np.sqrt(integration_dt) * rng.normal(size=4)
    particles[idx, 1:4] = np.clip(particles[idx, 1:4], -0.95, None)
```

### 6.4 权重计算 — 仅 fNIRS

```python
# 移除 EEG 似然项
log_weights = np.log(np.clip(weights, 1e-300, None))

# 仅 fNIRS 似然
log_weights += -0.5 * np.sum(
    np.square(bundle.fnirs_primary_obs[step] - pred_primary), axis=1
) / (sigma_nirs ** 2)
log_weights += -0.5 * np.sum(
    np.square(bundle.fnirs_secondary_obs[step] - pred_secondary), axis=1
) / (sigma_nirs ** 2)
```

### 6.5 删除的参数

- `--prior-std` 中 r 的项（设为零）
- `--state-noise-std` 中 r 的项（设为零）
- σ_eeg（不再需要）
- λ_r（不再需要）

### 6.6 新增的参数

- `--sigma-prop`：r 提议带宽（默认 2.0，单位 μV）
- `--sigma-nirs`：fNIRS 观测噪声（数据集相关）

---

## 7. 实验方案

### Phase 1 — 合成验证

验证"EEG提议 + fNIRS选择"机制在已知 ground truth 下工作。

**生成**：
- r_true(t)：10Hz 振荡 + 慢变漂移
- EEG 观测：L·r_true + 小量测量噪声
- fNIRS 观测：r_true → Croce 血流动力学 → fNIRS 通道 + 噪声
- r_eeg(t) = L⁺ y_eeg(t)（用于 PF 提议中心）

**验证指标**：
- RMSE(r̂, r_true) / std(r_true) < 0.3
- r̂(t) 与 r_true(t) 的高频（>1Hz）相关性 > 0.9
- r̂(t) 与 r_true(t) 的低频（<0.3Hz）相关性 > 0.7
- r̂(t) 在 0-0.3Hz 频段与 r_eeg(t) 的差异 > 0（确认 fNIRS 修改了 r 的低频成分）
- EEG source target 重建相关性 > 0.85
- fNIRS source target 重建相关性 > 0.7

### Phase 2 — 真实数据（EEG+NIRS Single-Trial, Subject 1）

3 个 fNIRS 源 × 2 个 EEG case × 3 个随机种子：

**指标**：
- EEG source target 重建相关性（per channel mean）
- fNIRS source target 重建相关性
- r̂(t) 的 PSD：alpha 波段（8-13Hz）功率 vs. delta 波段（0.5-4Hz）
- r̂(t) 与 r_eeg(t) = L⁺ y_eeg 的差异（全频段和分频段）
- 差异的频谱：是否集中在 <0.5Hz（应如此，因为 fNIRS 仅约束低频）
- 种子复现性
- ESS ratio

**通过标准**：
- EEG 重建相关性 > 0.5
- r̂(t) 的 alpha 功率 > r_eeg 的 alpha 功率的 80%（高频成分保留）
- r̂(t) 与 r_eeg 的低频（<0.3Hz）差异 > 0（fNIRS 确实修改了 r）
- 种子复现性 > 0.6
- ESS > 0.3

---

## 8. 参数总表

| 参数 | 默认值 | 类型 | 含义 |
|------|--------|------|------|
| σ_prop | 2.0 μV | 搜索超参数 | r 提议带宽 |
| σ_nirs | 数据集相关 | 测量参数 | fNIRS 观测噪声 |
| κ | 数据集相关 | 物理参数 | fNIRS 单位映射 |
| q_s | (0.02)² | 过程噪声 | s 状态噪声方差 |
| q_f, q_hbo, q_hb | (0.015)² | 过程噪声 | 血流动力学状态噪声方差 |
| N | 500 | 计算参数 | 粒子数 |
| resample_fraction | 0.5 | 计算参数 | ESS 阈值 |
| integration_dt | 0.005 s | 数值参数 | 积分步长 (200 Hz) |
| Croce 参数 | 见表 | 固定物理参数 | ε, k_as, k_af, τ₀, α, E₀ |
