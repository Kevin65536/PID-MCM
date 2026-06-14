# Physiological Coupling Constraints for EEG-fNIRS Tokenizer

> Created: 2026-04-30 | Last revised: 2026-06-04
> Status: Active design document — Croce local highWL-only source/observation training contract under evaluation
> Reference implementation surface: [src/tokenizers/factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py), [src/inference/neurovascular_smc.py](../src/inference/neurovascular_smc.py)

---

## 1. Motivation

当前 source/observation 主线已经不再卡在“要不要 factorization”上，而是卡在一个更根本的问题上：

> source target 究竟应该是什么，才能同时服务生理解释和 tokenizer 训练？

我们现在需要的不是一个只在 latent 中自洽的共享状态，也不是单向构造的跨模态代理，而是一种可以直接服务 tokenizer 的物理分解：

1. 从 EEG 与 fNIRS 的联合观测中得到一个共享生理源；
2. 这个共享生理源能够对称地产生 clean EEG 与 clean fNIRS；
3. 两个 clean 信号都仍然足够贴近原始时间轴，使 observation target 保持线性残差定义。

因此，本文件定义的是当前 branch-target 合同，而不是为某个既有 proxy 实现背书。

---

## 2. Current Branch-Target Contract

### 2.0 Current Training Input Scope

As of 2026-06-04, the active tokenizer experiment uses the generated Croce local cache rather than raw whole-brain windows:

1. EEG input is one local six-channel neighbourhood per fNIRS spatial anchor: `eeg [B, 6, 4000]`.
2. fNIRS input is one spatial anchor and one optical component: `fnirs [B, 1, 200]`.
3. The selected fNIRS component is `highWL`, read from `source_fnirs_optical_channel_0` and `obs_fnirs_optical_channel_0`.
4. `lowWL` remains recorded in the cache as the second wavelength (`pair_labels=["highWL", "lowWL"]`) but is ignored for this tokenizer training phase.
5. This is still optical measurement-space high-wavelength signal, not HbO concentration. It is only treated as an HbO-sensitive response proxy.

### 2.1 Naming

| 名称 | 当前语义 |
|------|----------|
| source branch | 编码并重建两个模态各自的干净生理观测成分 |
| observation branch | 编码并重建无法归入共享生理源的观测污染、被试差异和剩余模态特异成分 |

### 2.2 Source Branch Semantics

当前 source branch 的监督对象应是如下输出对：

$$
(\hat y^{src}_{EEG}(t), \hat y^{src}_{fNIRS}(t))
$$

它们必须同时满足：

1. 联合约束：两者必须由 EEG 与 fNIRS 联合决定，不能由单一模态单向构造。
2. 双边对称：clean EEG 与 clean fNIRS 具有同等物理地位。
3. 测量空间可监督：source decoder 的目标应直接落在各自模态的测量空间。
4. 线性残差可用：必须保留

$$
 y^{obs}_{m}(t) = y^{raw}_{m}(t) - y^{src}_{m}(t), \qquad m \in \{EEG, fNIRS\}
$$

5. 空间局域性：source model 只应依赖局部相邻通道的共享生理结构，而不是 whole-head 的单标量代理。

### 2.3 Observation Branch Semantics

observation branch 当前应被理解为：

1. 导联接触、source-detector 几何误差、仪器漂移、被试差异等 nuisance factor 的承载者；
2. 在 source 已解释掉干净生理成分之后，保留原始信号重建所需的剩余观测成分；
3. 不通过“平滑补集”定义，而通过 raw - clean source 明确定义。

### 2.4 What Is Explicitly Rejected

以下内容不再属于当前主线 branch-target 语义：

1. smooth_signal(raw) 式 common/residual proxy；
2. 任何单向构造另一模态 clean target 的方法；
3. 把单模态幅值代理本身当作 clean source 的定义；
4. 通过目标模态自身统计量“制造” clean source；
5. 只凭 coupling 的跨模态可预测性来替代 clean source target 的显式定义。

---

## 3. Candidate Physical Model Families

### 3.1 Croce-Style Joint State-Space Model

Croce 2017 仍然重要，因为它提供了“共享生理状态 + 双模态观测重建”的生成式骨架。但要成为当前主线候选，它必须满足比原论文和现有 proxy 实现更强的条件：

1. posterior 必须同时由 EEG 与 fNIRS 更新，而不是先由单一模态构造共享状态再生成另一模态；
2. 输出必须是 clean EEG / clean fNIRS 观测成分，而不是只给出一个 latent driver；
3. 局部 electrical source 不应再是单个标量，而应允许少量局部多源状态；
4. 必须显式建模 nuisance states 或观测污染，而不是把它们都推给高斯噪声。

### 3.2 Nuisance-Augmented Local Croce+

这是当前最贴合 tokenizer 需求的候选方向：

$$
 x_t = (q_t, h_t, n_t)
$$

其中：

1. $q_t$ 表示局部 electrical source states；
2. $h_t$ 表示 hemodynamic states；
3. $n_t$ 表示 contact / device / subject drift 等 nuisance factors。

模型输出直接给出：

$$
\hat y^{src}_{EEG}(t), \quad \hat y^{src}_{fNIRS}(t), \quad \hat y^{obs}_{EEG}(t), \quad \hat y^{obs}_{fNIRS}(t)
$$

### 3.3 Simpler Dynamic-Factor Baseline

如果一个较弱的动态因子模型能够更稳定地输出“对称 clean source + 线性 residual”，它也应被接受为对照面。当前主线不应因为“生理模型更优雅”就拒绝更适合 tokenizer 分解的 baseline。

---

## 4. Coupling Priors: What They Are and Are Not

lag focus 与 joint smoothness 仍是有效的结构先验，但它们的职责已经被限定：

1. 它们只约束 source codebook 之间的 correspondence 结构；
2. 它们不能替代 clean source target 的定义；
3. 它们不应该重新偷渡任何单向主从叙事回 branch semantics；
4. 它们应服务于“共享生理源在两个模态上如何对应”，而不是“哪一边是主因、哪一边是从属 proxy”。

当前实现进一步把经验配对监督拆为两个互不混淆的目标：

1. **lag-balanced pair likelihood**：每个有效 lag 独立平均条件负对数似然，再对 lag 等权平均，避免短 lag 因有效配对数量更多而天然占优；
2. **lag evidence**：只用相对该 lag 的 fNIRS 边缘分布所获得的条件对数似然增益来监督 lag marginal，避免 lag focus 集中到一个没有 EEG 条件信息的频繁 lag。

这些目标学习的是 $P(z_{fnirs}\mid z_{eeg},\tau)$，不要求 EEG 与 fNIRS codebook 的相同数值 indice 表示相同状态，也不加入对角映射约束。

---

## 5. Validation Criteria For Any Candidate Model

任何进入当前主线评审的物理模型都必须回答以下问题：

1. 双模态 clean-source fidelity：能否同时给出可信的 clean EEG 和 clean fNIRS？
2. 线性 residual usability：raw - source_target 是否真的形成可训练的 observation target？
3. 时间同步：clean source 是否保留与原始观测足够接近的时间结构？
4. 联合贡献：相较于只用 EEG 或只用 fNIRS，联合推断是否改善了两个模态的 clean source 估计？
5. 空间特异性：spatial null 是否显著破坏结果？若不会，则该模型没有真正利用局部生理结构。
6. nuisance concentration：被试/设备/接触差异是否主要落在 observation branch？

---

## 6. Repo Contract

从 2026-05-22 起，以下规则适用于仓库文档解释：

1. 本文件、[IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md)、[ARCHITECTURE.md](ARCHITECTURE.md) 和 [SEMANTIC_TOKEN_SCORECARD.md](SEMANTIC_TOKEN_SCORECARD.md) 共同定义当前 branch-target 合同。
2. 旧 proxy 文档只保留为历史记录或候选 baseline 说明。
3. 代码中现存的 legacy target-construction 路径不能反向定义当前文档语义。
4. 如果历史 changelog 与当前活动文档冲突，以当前活动文档为准。
5. 当前 highWL-only 输入选择是训练阶段的临时约束，不等同于删除 lowWL 缓存，也不等同于完成 HbO/HbR 浓度转换。
