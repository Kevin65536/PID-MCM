# Theory v2: Neuro-Tokenization & Semantic Alignment (EEG/fNIRS)

> **Version**: 3.0  
> **Last Updated**: 2026-01-05  
> **Roadmap**: See `IMPLEMENTATION_PLAN.md`

---

## 1. Why Tokenization-First?

EEG 与 fNIRS 都是高噪声、强个体差异、且多时间尺度的生理信号。直接在连续空间做“信息分解”会遇到：

- 表示不稳定：模型先学到预处理差异/噪声模式，导致分解结论漂移
- 时标冲突：EEG 毫秒级、fNIRS 秒级（且 HRF 有延迟），共享表示容易被采样率/频带差异主导

因此我们先学习一个离散化的 codebook（Neuro-Tokens），让表示具备：

1) **压缩性**：把“重复出现的模式”离散成 token
2) **鲁棒性**：对轻微增强/噪声不敏感
3) **可解释结构**：能被生理先验（节律、HRF/ERP、空间拓扑）约束

之后再在 token 空间中进行 PID 分析与结构化改造。

---

## 2. Neuro-Tokenization (Continuous → Discrete)

### 2.1 基本形式

对每个模态 $m \in \{eeg, fnirs\}$：

- 编码器：$z_m = E_m(x_m)$
- 量化器：$q_m = Q_m(z_m) \in \{1,\dots,K_m\}$（离散 index），以及对应 embedding $e_{q_m}$
- 解码器：$\hat{x}_m = D_m(e_{q_m})$

训练目标首先以重构为主：

$$\mathcal{L}_{rec} = \|x_m - \hat{x}_m\| + \lambda_f\,\mathcal{L}_{freq}(x_m,\hat{x}_m)$$

其中 $\mathcal{L}_{freq}$ 可取 multi-scale STFT loss 或频带能量误差（对 EEG 更重要）。

### 2.2 Codebook 健康度（避免“坍塌”）

离散 codebook 的常见失败模式是只用很少的 code（collapse），这会使“语义空间”失去区分能力。

因此必须跟踪：

- usage / perplexity（token 使用丰富度）
- top-k 覆盖度（是否只有少数 code 被反复使用）
- embedding 方差 / 有效秩（是否退化到低维）

### 2.3 为什么推荐 Separate Codebooks + Shared Projector

EEG 与 fNIRS 的统计特性差异很大。

- 共享 codebook 往往把“采样率差异/噪声差异”编码成 token
- 分离 codebook 可以让每个模态先学“本模态可重构的基本单元”
- 再用共享 projector 把 token embedding 映射到共同语义空间，专门处理“跨模态共享的语义”

---

## 3. Semantic Alignment: What Are We Aligning?

### 3.1 共同语义空间 $S$

我们定义共享 projector：

$$s_m = P(e_{q_m}) \in S$$

其中 $e_{q_m}$ 是 token embedding，$P$ 将不同模态的 token 映射到同一个语义空间。

最小可行对齐目标：同一时间 window 内的 EEG 与 fNIRS（或其聚合）在 $S$ 中应更接近。

例如用 InfoNCE：

$$\mathcal{L}_{align} = -\log \frac{\exp(\text{sim}(s_{eeg}, s_{fnirs})/\tau)}{\sum_{k}\exp(\text{sim}(s_{eeg}, s_{fnirs}^{(k)})/\tau)}$$

### 3.2 Alignment Targets (四类目标的可操作化)

#### A) True Brain Activity / Brain State

若有标签（任务段、睡眠分期、事件标注），可用 probe 衡量 $S$ 的线性可分性；不一定要强监督训练 tokenizer。

若无标签，优先用自监督一致性（跨增强、跨模态）让 $S$ 稳定。

#### B) Physiological Rhythm（需要区分“去除”与“编码”）

生理节律（心跳/呼吸/血压/Mayer 波）对两模态影响不同：

- 对 fNIRS 常是强干扰源（系统性波动）
- 对 EEG 可能既是噪声也可能携带状态信息

因此有两条理论路径：

1) **Nuisance removal**：对抗/约束让该成分不进入 brain-semantic token
2) **Explicit branch**：单独设置 physiological head/token，避免污染 brain state

#### C) Pseudo-Trace（模板先验，先弱后强）

- fNIRS：canonical HRF 或 HRF family 可作为弱正则（shape correlation / smoothness）
- EEG：ERP 模板非常依赖范式与预处理，建议仅作为可选弱正则

关键思想：Pseudo-Trace 是“引导结构”，不是“真值标签”。

#### D) Spatial Pattern（拓扑与局部一致性）

空间先验本质是图结构约束：

- EEG：10-20 邻接图
- fNIRS：探头布局邻接图

最小实现是对邻接通道的表示做平滑/一致性正则，或在 ROI 级别做 token 聚合。

---

## 4. PID Comes Back: PID on Tokens (Analysis & Refinement)

当 tokenization 与 alignment 稳定后，我们在离散变量上定义 PID，避免连续估计不稳定。

### 4.1 定义随机变量

- $C_{eeg}$：EEG token 序列或其统计（如 bag-of-codes / transition counts）
- $C_{fnirs}$：fNIRS token 序列或其统计
- $Y$：目标（优先任务/状态标签；备选 pseudo-trace 参数或自定义伪标签）

### 4.2 PID 分解（两源）

$$I(C_{eeg}, C_{fnirs}; Y) = R + U_{eeg} + U_{fnirs} + S$$

我们关心的不是一个标量结果，而是：

- 哪些 code / code-pair 在 $R/U/S$ 中贡献最大
- 去除 physiological nuisance 或加入 pseudo-trace prior 后，$R/S$ 是否更符合预期

### 4.3 为什么把 PID 从“训练主损失”降级为“分析工具”

PID 的估计对表示分布非常敏感；在表示尚未稳定时，把 PID 当作强约束会放大噪声与错误先验。

更稳健的做法是：

1) 先得到稳健 token
2) 用 PID 找到冗余/协同模式
3) 再回到模型结构/对齐策略中做闭环改造（合并/拆分 codebook、分支建模、调整 projector）

---

## 5. Practical Implications (落地检查表)

- 任何 alignment/先验加入后，都必须不破坏 codebook health（usage/perplexity）
- Pseudo-Trace 必须从弱正则开始，避免硬拟合导致错误语义
- Physiological 先验必须先决定“去除 vs 编码”，否则训练信号会互相冲突

---

## 6. Notes

旧版 ELP/PID-first 理论已备份于 `docs/THEORY_v1_ELP.md`，以便后续回顾对比。
