# Physiological Token Semantic Scorecard

> Last Updated: 2026-04-13
> Status: Active tokenizer evaluation design for the codebook-first mainline

## 1. Why This Scorecard Exists

当前仓库已经完成一次重要方向切换：tokenizer 的主线目标不再是更高的 token identity overlap，而是更健康、更可解释、且更能承载 shared/private 生理语义分工的离散状态空间。

因此，旧的评估口径已经不够了。只看 reconstruction、perplexity 和 overlap，会漏掉三个更关键的问题：

1. token 是否真的对应稳定的局部生理状态；
2. token 序列是否保留了状态转移和跨模态滞后耦合；
3. shared branch 是否真的在表达跨模态共性，而不是被训练成另一条 reconstruction 通路。

这个 scorecard 的目标，就是把“token 的语义是否成立”变成一套固定、可复盘、可比较的报告结构。

## 2. Semantic Target

当前 mainline 里，token 的语义目标应明确区分为三类。

1. shared token：跨模态共通的局部生理状态，尤其是允许时滞的 EEG-fNIRS 共同因子；
2. EEG private token：快速电生理残差状态，例如频段活动、瞬时波形和局部相位结构；
3. fNIRS private token：慢速血流动力学残差状态，例如 HRF 形状、缓慢上升/回落和局部血氧轨迹。

判断一个 tokenization 是否成功，不是看它是否让两模态在同一时间输出相同 index，而是看它是否让这些状态在离散空间中变得可压缩、可预测、可迁移、可解释。

## 3. Canonical Metric Layers

### 3.1 Layer A: codebook health guardrails

这是所有语义讨论之前的进入门槛。

| 指标 | 记号 / 计算 | 解释 |
|------|-------------|------|
| Shared/private perplexity | $\exp(-\sum_k p_k \log p_k)$ | 检查 shared 或 private codebook 是否 collapse |
| Active-code coverage | active codes / total codes | 检查实际使用范围 |
| Dead-code count | $\sum_k \mathbf{1}[p_k = 0]$ | 检查是否存在大面积无效 code |
| Gini / top-k coverage | usage distribution statistics | 检查是否被极少数 token 垄断 |
| Reconstruction guardrails | full/common/residual reconstruction metrics | 保证语义讨论不是建立在严重失真上 |
| Branch ablation gaps | branch-only decoding degradation | 检查 shared/private 是否在承担不同职责 |

没有通过 Layer A 的 run，不应继续解释它的“语义”。

### 3.2 Layer B: semantic state quality

这一层回答：token 是否真的对应稳定的局部生理状态。

#### 3.2.1 Intra-token state consistency (ITSC)

设 $\phi(x_t)$ 是窗口的状态摘要，可以取 shared-common target、branch latent summary，或固定的生理特征摘要（EEG bandpower/topography, fNIRS HRF summary）。定义：

$$
ITSC = \sum_k p_k \mathbb{E}[\|\phi(x_t) - \mu_k\|_2^2 \mid z_t = k]
$$

其中 $\mu_k$ 是 token $k$ 对应样本的状态原型。ITSC 越低，说明同一 token 内部越稳定。实践中应报告归一化版本：

$$
ITSC_{norm} = \frac{ITSC}{\mathrm{Var}(\phi(x_t))}
$$

#### 3.2.2 Prototype separation ratio (PSR)

$$
PSR = \frac{\sum_{i \neq j} \|\mu_i - \mu_j\|_2^2}{\sum_k p_k \mathbb{E}[\|\phi(x_t) - \mu_k\|_2^2 \mid z_t = k] + \epsilon}
$$

PSR 越高，说明 token 原型之间的区分度越强。

#### 3.2.3 Transition predictability gain (TPG)

$$
TPG(\Delta) = H(Z_{t+\Delta}) - H(Z_{t+\Delta} \mid Z_t)
$$

TPG 越高，说明 token 序列保留了状态转移结构，而不是接近独立采样。

#### 3.2.4 Augmentation consistency (AC)

对不改变生理语义的扰动 $a(\cdot)$，例如轻度加噪、幅值缩放、时间轻微平移，定义：

$$
AC = P\big(z(x) = z(a(x))\big)
$$

也可以放宽为近邻一致率。AC 越高，说明 token 对 nuisance factor 更稳。

#### 3.2.5 Branch responsibility gap (BRG)

BRG 不是单一数字，而是一组对照差分：

1. shared ablation 对 common target reconstruction 的损害，应明显大于对 residual target 的损害；
2. private ablation 对 residual target reconstruction 的损害，应明显大于对 common target 的损害。

如果这个模式不成立，就说明 branch semantics 仍然是模糊的。

### 3.3 Layer C: structured cross-modal semantics

这一层回答：token 是否真的表达了 EEG-fNIRS 之间有时滞的结构关系。

#### 3.3.1 Lagged mutual information gain (LMIG)

$$
LMIG = I(Z^{eeg}_{t-\tau}; Z^{fnirs}_t) - I(Z^{eeg}_t; Z^{fnirs}_t)
$$

如果 LMIG 稳定为正，说明 shared tokens 更接近真实生理耦合，而不是被迫做同步匹配。

#### 3.3.2 Conditional KL gain (CKG)

$$
CKG = \mathbb{E}_{e \sim P(Z^{eeg})}\left[D_{KL}\big(P(Z^{fnirs} \mid Z^{eeg}=e) \| P(Z^{fnirs})\big)\right]
$$

CKG 越高，说明 EEG token 对 fNIRS token 分布提供了真实条件约束。

#### 3.3.3 Cross-modal masked token prediction gain (CMG)

冻结 tokenizer，仅训练轻量 probe，比较：

1. EEG + partial fNIRS context；
2. only partial fNIRS context；
3. shuffled EEG + partial fNIRS context。

定义 CMG 为版本 1 相对版本 2 和版本 3 的预测收益。CMG 稳定为正，说明 shared states 具备跨模态可计算价值。

#### 3.3.4 Shared usage balance (SUB)

shared branch 不应被单一模态长期垄断。可报告：

$$
SUB = 1 - \frac{|u_{shared}^{eeg} - u_{shared}^{fnirs}|}{u_{shared}^{eeg} + u_{shared}^{fnirs} + \epsilon}
$$

它不是越高越好到无条件，而是用于检查 shared branch 是否已经退化成某一模态的专用通道。

#### 3.3.5 Overlap is supplementary only

token identity overlap 和 exact match rate 仍可保留，但只能作为补充诊断。它们不能再作为 shared branch 是否成功的主要结论依据。

### 3.4 Layer D: invariance and utility

这一层回答：语义空间是否保留了真正有用的信息，同时压低 nuisance factor。

#### 3.4.1 Subject leakage score (SLS)

冻结表示，用轻量 probe 预测 subject ID。SLS 越低越好，特别是在 shared branch 上。

#### 3.4.2 Task / condition signal (TCS)

冻结表示，用同量级 probe 预测 task / condition。TCS 不要求一开始很高，但不能长期接近 chance。

#### 3.4.3 Semantic selectivity ratio (SSR)

$$
SSR = \frac{TCS}{SLS + \epsilon}
$$

如果 SSR 很低，说明 shared 表示更多记住了被试特征，而不是生理任务状态。

#### 3.4.4 Session / device stability (SDS)

可使用 session-split retrieval agreement、MMD 或 distribution shift diagnostics。目标不是完全不变，而是在同任务/同状态条件下，shared token 统计不应随采集条件大幅漂移。

### 3.5 Layer E: optimization support diagnostics

这一层不直接定义语义本身，但用来解释为什么语义没有形成。最近的 gradient diagnostics 已经说明这一层必须纳入 mainline report。

| 指标 | 解释 |
|------|------|
| Semantic gradient share | shared-common / coupling / semantic losses 是否长期几乎拿不到梯度预算 |
| Reconstruction dominance ratio | reconstruction losses 是否长期压制其它目标 |
| Conflict rate | 多目标训练的冲突是否持续存在 |
| Mean / min pairwise cosine | 是否存在长期结构性对抗 |

如果 Layer B-D 失败，而 Layer E 同时显示 semantic gradient share 长期接近 0，那么问题更可能在训练动力学，而不是表征容量本身。

## 4. Canonical Report Contract

从现在开始，每一个 mainline tokenizer run 至少应输出以下内容。

1. Reconstruction guardrails：EEG full、fNIRS full、shared-common、EEG residual、fNIRS residual；
2. Codebook health：shared/private perplexity、active codes、usage coverage、gini、top-k coverage；
3. Branch semantics：branch ablation gaps 与 BRG；
4. State semantics：ITSC、PSR、TPG、AC；
5. Cross-modal semantics：LMIG、CKG、CMG、shared usage balance；
6. Invariance：SLS、TCS、SSR、SDS；
7. Optimization support：gradient semantic-share、conflict dashboard milestone snapshots；
8. Narrative conclusion：明确说明 shared branch 当前表达的是共性状态、重建捷径，还是某一模态偏置通道。

## 5. Implementation Priority

为了让这套 scorecard 能落地，建议分三步实现，而不是一次性把所有分析脚本都铺开。

### P0: immediately mandatory

这些指标已经最接近当前主线，可以优先纳入标准报告：

1. reconstruction guardrails；
2. codebook health；
3. branch ablation gaps；
4. lagged MI gain；
5. shared usage balance；
6. gradient conflict / semantic gradient share。

### P1: next evaluation pass

这些指标需要基于 token dump、common/residual targets 或 lightweight feature summaries 实现：

1. ITSC；
2. PSR；
3. TPG；
4. CKG。

### P2: probe-based semantic validation

这些指标最能说明“语义空间是否真的有 foundation value”，但需要轻量 probe：

1. CMG；
2. SLS；
3. TCS；
4. SSR；
5. SDS；
6. AC。

## 6. Bottom Line

对 EEG-fNIRS foundation model 来说，tokenization 的目标不是造出一个更漂亮的离散压缩器，也不是追求更高的同步 overlap。它的目标是构建一个离散生理语义空间，使 shared/private 状态、时序转移和滞后耦合都能够被稳定地计算、比较、预测和解释。

因此，今后任何 tokenizer 改动都应该回答同一个问题：它到底是在改善生理语义空间，还是只是在改变 reconstruction 曲线和 token 频率统计。