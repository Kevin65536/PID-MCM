# Shared-Driver Semantic Tokenization 的理论基础

_理论合同：2026-07-25；详细历史反思见 [架构回归与方法启示](12_ARCHITECTURE_RETURN_AND_METHOD_LESSONS.md)_

---

## 📋 研究问题

项目要检验的不是“能否把两种信号压缩成 token”，而是：

> 独立从 EEG 与 fNIRS 原始观测产生的离散状态，是否在合理时延、被试外泛化以及 history/nuisance/null 控制后，仍表现出可重复的条件对应？

这个问题包含四个不能由同一指标替代的层次：

1. **压缩**：连续信号能否稳定离散化；
2. **语义锚定**：prototype 是否保留预先指定的 teacher-defined proxy coordinate；
3. **耦合发现**：双向整窗 token 是否具有离线时延条件关联；在满足时间截断时，是否还能增加窗外未来 fNIRS 的可预测信息；
4. **科学确认**：相应 estimand 是否在冻结、被试外和 matched null 下成立。

SD-SVQ 让模型只承担前两层，把后两层交给模型外的冻结评估。

## 🧠 生成假设

设潜在神经驱动为 \(R_t\)，血流状态为 \(H_t\)，模态私有观测因素为 \(U_t^E,U_t^F\)：

\[
X_t^E=g_E(R_t,U_t^E)+\epsilon_t^E,
\]

\[
H_{t+1}=f(H_t,R_t;\theta)+\omega_t,
\]

\[
X_t^F=g_F(H_t,U_t^F)+\epsilon_t^F.
\]

EEG 与 fNIRS 的对应不是同一时刻、同一尺度、同一观测函数下的相等关系。fNIRS 对神经驱动的反映延迟且被血流自历史强烈解释。因此，理论上需要：

- 独立的 modality-specific encoder；
- 量化前的长时上下文；
- 显式 neural-time alignment；
- 超过 fNIRS history 的增量 estimand；
- 对 task phase、subject 和边际分布的控制。

这些是观测过程要求的复杂性，不是为了优化而新增的装饰。

本文的 `raw-only` 不是“未经任何处理的电压/光密度”。它严格指：**冻结预处理合同下的 measured EEG 或 measured HbO/HbR，加 boundary/finite 计算 mask**。teacher state、task/condition、subject、phase、motion/nuisance feature 和另一模态观测均不属于 tokenizer 输入。

## 🎯 共享驱动作为训练坐标

E0 joint smoother 给出：

\[
R^J=T(X^E,X^F;\hat\theta_{\mathrm{train}}).
\]

\(R^J\) 的角色是 privileged training coordinate。当前 E0 target 的实际 provenance 是 **subject-specific leave-one-trial**：空间 anchor、EEG proxy 投影和 SSM 参数可由同一被试的其他 trial 拟合，再应用到被留出的 trial；这不等于 population-frozen 的被试外 teacher。R1-D 可保留这种 development-crossfit target 做探索，R1-P 则必须只在 development-training subjects 上拟合全部参数，并对 held-out/protected subject 纯 apply。因为 estimand 已改变，R1-P 不能继承旧 E0 admission，必须重新通过 population-frozen teacher panel。两个 student 分别只看本模态：

\[
K^E=Q_E(E_E(X^E)),\qquad
K^F=Q_F(E_F(X^F)),
\]

\[
\mathcal L_{\mathrm{driver}}
=d(D_r(K^E),R^J)+d(D_r(K^F),R^J).
\]

该设计的目标是形成“从各自模态可实现、且能保留同一 joint-driver 坐标的离散区域”；只有 R2/R3 通过后，才可称这种语义在相应总体中可实现。它不把 \(R^J\) 升格为 ground truth。teacher 仍依赖模型族、先验、gauge、训练折参数和 fixed-interval smoothing。

E0 的合理解释是：该 teacher 被授权作为受限监督坐标；不是其隐变量或生理参数已经唯一识别。

## 🔗 Teacher-grounded alignment 不等于 coupling

若距离 \(d\) 满足三角不等式，则：

\[
d(D_r(K^E),D_r(K^F))
\le d(D_r(K^E),R^J)+d(R^J,D_r(K^F)).
\]

因此，在两侧重建误差都小的样本上，解码状态接近是训练目标在数学上的预期后果；它并不自动证明这种接近跨被试、跨 seed 或 held-out 稳定，更不能反过来当作独立耦合证据。

证据必须分三级：

1. **grounded alignment**：held-out 检验证明两侧 token 保留同一 \(R^J\) 坐标；
2. **emergent symbolic correspondence**：在没有 same-ID 或 token-pair loss 时出现稳定的滞后 token-pair 分布；
3. **offline delayed conditional association**：双向整窗 token 冻结并移除 teacher 后，在 history/phase/null 控制下仍有时延条件关联；
4. **prospective incremental prediction**：只有 token 的 receptive-field end 早于 endpoint start 时，EEG history 对窗外未来原始 fNIRS endpoint 提供增量信息。

第 3、4 层是不同 temporal estimand，可按主张独立预注册；R6A 不是 R6B 的科学前门。

以下未来预测 estimand 只适用于第 4 层：

\[
\Delta\ell =
\log p(Y^F_{t+\tau}\mid K^E_{\le t},H_t^F,C_t)
-
\log p(Y^F_{t+\tau}\mid H_t^F,C_t),
\]

其中 \(H_t^F\) 是 cutoff 前的 fNIRS 自历史，\(C_t\) 是仅在评估阶段使用的 subject/task/phase 等控制；还必须断言所有 \(K^E\) 的 receptive-field end 早于 \(Y^F\) 的起点。双向 token 若与 endpoint 共享同一 20 秒窗口，只能进入第 3 层，不能代入这个式子并称作未来预测。

## 🗣️ 生理 token 的三层语义

token 语义不等于 ID：

| 层面 | 定义 | 证据 |
| --- | --- | --- |
| 指称语义 | prototype 能稳定解码出什么 driver trajectory | decoded signature、held-out reconstruction |
| 分布语义 | 它与哪些前后状态、异模态 token 和时延共同出现 | lagged co-occurrence、matched null |
| 操作语义 | 它是否改善独立 endpoint 的被试外预测 | fresh frozen incremental evaluator |

只有三层汇合，才可称为稳定的生理 token 语义。

两个独立 codebook 分别具有 permutation symmetry。`EEG token 17` 与 `fNIRS token 17` 没有天然同义性，跨 seed 也不能按 ID 对齐。有效比较对象是 decoded driver signature 或其他外部行为。

## 📐 可识别性边界

重建目标缩小了解空间，但没有消除：

- token ID 的置换不确定性；
- encoder/codebook/decoder 的等价变换；
- teacher 模型的 gauge 和参数非唯一性；
- 原始测量中 subject、设备、运动和 task phase 的混杂；
- joint teacher 将两模态信息共同写入训练坐标的 circularity。

因此必须区分：

- **predictive sufficiency**：某模态是否足以预测 \(R^J\)；
- **latent identifiability**：\(R^J\) 是否为唯一真实神经驱动；
- **offline incremental association**：双向整窗 EEG 表示是否增加同一离线记录中的时延条件信息；
- **prospective prediction**：在严格 cutoff 后是否增加窗外未来 fNIRS 信息；
- **causality**：EEG 表示的状态是否导致血流变化。

R 系列可以检验 predictive sufficiency、offline association 和 prospective prediction；它不能由这些指标建立 latent identifiability。没有干预或额外识别假设时，也不声明 causality。

## 🔭 为什么必须先做 continuous observability

数据处理不等式意味着 token 不能创造输入中不存在的信息。若 fNIRS raw window 连续表示都无法在 held-out subject 上恢复 \(R^J\)，更强 VQ、更多 code 或额外 nuisance token 不会使 shared-driver 语义变得可信。

E2 还揭示了 target-space quantizability 与 raw-to-token realizability 的差别：一个目标可以由 128 个 prototype 很好近似，却不一定能从单模态原始信号推断。因此 R2 必须在 VQ 前检验两侧 continuous student。

## ⏱️ 时间与因果边界

E0 使用整窗 fixed-interval RTS smoother；fNIRS 的晚期观测可修正窗口早期的 driver。与其对齐的 full-window tokenizer 是 offline contextual representation。

它可用于：

- 离线状态发现；
- 整窗条件关联；
- 表示与 teacher 的一致性分析。

它不可直接用于：

- 在线预测；
- “过去 EEG 预测未来 fNIRS”的因果措辞；
- 不标注 receptive field 的 Granger-like 解释。

需要未来预测证书时，首版可只汇总已经完整结束、且通过语义门的双向 token 窗口；也可另建只看过去的 causal token，但新版本必须重过语义门。两者都要按原始 record 的绝对时间证明整个 receptive field（含预处理支持和 embargo）早于 endpoint，并确保 target construction 也不把未来 endpoint 泄漏回 token。

## 🧪 反事实 teacher 控制

真实配对的 candidate joint teacher 必须与至少三类同复杂度目标比较：

1. EEG-only \(R^E\)：检验 joint correction 是否真正有作用；
2. 在构造 teacher 前破坏 fNIRS 配对/时延的 \(R^{J,\pi}\)：检验共同 task phase 是否足以解释结果；
3. 频谱、平滑度和条件均值匹配的 pseudo-target：检验模型是否只是编码低频模板。

\(R^E\) 不是独立重拟合的另一个模型：它必须与 \(R^J\) 共用 R1-P 参数、EEG projection、anchor、gauge、sample 和时间坐标，只移除 fNIRS observation update；\(\delta^F=R^J-R^E\) 使用两者 pointwise support 交集。否则所谓 joint correction 会混入参数与坐标差异。

若 registered candidate teacher 不优于这些控制，只能声明模型重建了平滑或任务锁定坐标，不能声明物理目标特异性。

## 💾 private representation 的理论角色

continuous private branch 是 teacher misspecification 的可选保险，而不是被命名即成立的生理本体。它只在 semantic-only 模型已通过、但存在明确的信息保留缺口时引入。

private branch 的存在不证明其中是“噪声”“个体差异”或“模态特异生理”。这些解释必须由外部 probe 建立。若它没有改善预注册 retention，或侵蚀 semantic endpoint，则按简洁性原则删除。

## 🧊 K=128 的解释

`K=128` 是此前预实验和 E1 工程健康审计后冻结的容量预算。E1 证明在该容量下量化器可训练、占用可保持；它没有证明：

- 128 是最优 K；
- 存在 128 个真实生理状态；
- 所有码必须均匀使用；
- 占用率高意味着 semantic 或 coupling 成立。

codebook health、prototype semantics 和 coupling evidence 是三个正交 gate。

## 🧩 复杂性分类

| 类型 | 例子 | 处理 |
| --- | --- | --- |
| 物理必要复杂性 | 双模态 encoder、采样率、HbO/HbR、时延、整窗上下文 | 保留 |
| 优化补偿复杂性 | same-ID、orthogonality、多 balance loss、多离散 branch | 默认删除 |
| 证据必要复杂性 | cross-fit、subject holdout、null、development nested evaluation、protected boundary | 加强 |

一个新组件只有在回答已命名失败、产生外部可观察预测、具有独立 ablation、带 stop rule 且不进入最终自证 endpoint 时，才获得进入主线的资格。

## 🔐 主张阶梯

| 若该层级通过 | 通过后允许表述 |
| --- | --- |
| R2 | 本模态 raw signal 对 joint-driver proxy 具有 held-out 可实现性 |
| R3 | 固定 K128 hard token 保留该 proxy 的信息 |
| R5 | 独立 codebook 形成可重复的 driver-grounded prototype |
| R6A | 双向整窗 token 在 development 数据上具有受控的离线时延条件关联 |
| R6B | receptive field 截止于 endpoint 之前的 EEG 表示在 development 数据上增加窗外未来 fNIRS 信息 |
| R7 | 预先冻结且满足相应时间合同的结论在一次性 protected cohort 上确认 |

即使 R7 通过，也根据实际 temporal scope 写“offline delayed conditional correspondence”或“out-of-window incremental predictive information”，不写“发现因果神经血管机制”。

## 📚 方法论命题

1. 压缩不创造信息。
2. 重建误差不自动定义语义。
3. branch 名称不构成 latent identification。
4. token ID 是名义地址。
5. 共同 teacher 产生 grounded alignment，不自动产生 coupling evidence。
6. 共现必须来自独立推理，不能由 same-ID、pair loss 或 feature exchange直接写入。
7. coupling 是超过 fNIRS history、marginal、task/phase 和 null 的增量命题。
8. teacher 不能认证自身。
9. codebook health、semantic validity 和 coupling evidence 不可替代。
10. 数据、mask、channel 与 target coverage 共同定义 estimand。
11. 负结果应缩小假设空间，而不是默认增加 branch。
12. 模型极简，证据严格。

## 🔗 相关文档

- [目标架构](02_TARGET_ARCHITECTURE.md)
- [R 系列实验设计](05_EXPERIMENT_DESIGN.md)
- [架构回归与方法启示](12_ARCHITECTURE_RETURN_AND_METHOD_LESSONS.md)
