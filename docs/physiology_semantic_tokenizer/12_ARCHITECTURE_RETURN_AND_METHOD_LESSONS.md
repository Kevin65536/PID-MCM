# 架构回归：方法论审计与理论启示

_决策日期：2026-07-25；状态：当前研究纲领；适用于 R0–R7，E0–E9 保留为历史谱系_

---

## 📋 结论

这次回归不是把项目恢复到某个旧提交，而是恢复最初设计中最有价值的研究纪律：

1. EEG 与 fNIRS 分别从本模态原始测量推断 token；
2. 不预先规定两个模态必须产生相同 token ID；
3. 先证明表示保留了预先指定的生理对象，再冻结表示检验跨模态依赖；
4. 弱耦合或无耦合是允许的科学结果。

在此基础上，本轮只保留一项新增的强归纳偏置：两个独立的 `K=128, D=64` semantic codebook 都以 E0 联合状态空间模型得到的**完整联合共享驱动代理轨迹** \(r^J\) 为主要重建目标。tokenizer 的输入仍然只有各自模态的原始生理信号和计算有效性 mask。任务、被试、阶段、nuisance、teacher 状态和另一模态观测都不得作为 tokenizer 输入。

因此，目标架构的最短表述是：

> 两个仅看本模态原始测量的时序量化器，分别把 EEG 和 fNIRS 映射到独立的 128 项词表；训练时用同一个联合共享驱动代理轨迹规定“应保留什么”，训练后冻结 token。按主张选择 R6A 离线时延条件关联和/或独立的 R6B 窗外预测；后者要求表示的 receptive field 完全截止于 endpoint 之前。

这比“多类 effect token + 多入口 teacher + coupling shaper + foundation model + certificate”的链条更短，也更容易被证伪。

## 🧭 从原始设计到回归设计

```mermaid
flowchart LR
    accTitle: 项目方法从原始原则到复杂化再回归
    accDescr: 项目从独立模态量化与冻结后耦合检验出发，经历共享码本、共享私有分解、物理多目标和多阶段耦合塑形，最终回到以共享驱动为单一语义目标的独立量化器。

    original["原始原则<br/>本模态输入 · 独立 token<br/>冻结后检验 coupling"]
    shared_ids["共享码本与 ID 对齐<br/>InfoNCE / index matching"]
    factorized["shared/private 因子化<br/>更多分支与损失"]
    routed_teacher["物理 teacher 多入口<br/>local/prototype/context"]
    staged["preserve–discover–certify<br/>shaper + foundation + evaluator"]
    e2_failure["E2：teacher 无指标增益<br/>支持稀疏且目标冲突"]
    returned["回归后的最小核心<br/>raw-only 双流 + 独立 K=128<br/>完整 rJ 代理轨迹主监督"]
    evaluator["冻结后外部评估<br/>R6A 离线关联 · R6B cutoff 预测"]

    original --> shared_ids --> factorized --> routed_teacher --> staged
    staged --> e2_failure --> returned --> evaluator
    original -. "恢复研究纪律" .-> returned

    classDef history fill:#e5e7eb,stroke:#6b7280,stroke-width:2px,color:#1f2937
    classDef evidence fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#78350f
    classDef active fill:#dbeafe,stroke:#2563eb,stroke-width:3px,color:#1e3a5f
    classDef evaluation fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class original,shared_ids,factorized,routed_teacher,staged history
    class e2_failure evidence
    class returned active
    class evaluator evaluation
```

回归保留了探索中已经付费获得的工程资产：统一数据入口、train-only 标准化、固定 `K=128` 量化器、目标 sidecar 的版本化和 hash、protected split 管理、表示导出和冻结评估器。被删除的是没有被独立证据支持的结构性承诺，而不是历史经验。

| 阶段 | 当时引入复杂性的合理动机 | 同时加入的未证假设 |
| --- | --- | --- |
| 原始 raw VQ | 先验证连续生理信号能否离散化 | raw reconstruction 会自然产生跨模态语义 |
| shared codebook | 直接得到 token correspondence | 生理对应等价于 same/near ID |
| shared/private | 缓解重建与对齐竞争 | branch 名称足以识别 shared/private 内容 |
| source/observation | 用物理 proxy 替代抽象 shared | proxy 分解等同于真实神经源与观测残差 |
| coupling/exchange | 增强很弱的跨模态信号 | 训练写入的对应仍可当作发现证据 |
| physiology-semantic E2 | 把 teacher、tokenizer、certificate 分开 | 多个低权重局部目标能共同组织 hard-token geometry |
| 当前回归 | 只保留一个可直接否定的 semantic estimand | \(r^J\) 是有用坐标；该假设由 R2/R3/controls 检验 |

这些复杂化并非任意拼接：每一步都回应了真实失败。问题是新模块通常同时改变
多个 estimand，使下一轮无法判断收益来自物理假设、优化补偿、信息泄漏还是评估
口径变化。当前回归的目的正是恢复单次实验的归因能力。

## 🔍 E2 到底否定了什么

E2 的可靠结论是：在其准确实现的训练合同下，弱权重的 routed teacher 没有改善预注册的 hard-token semantic endpoint。T1 相对 T0 的三组差值为 `-0.0271/-0.0413/+0.0065`，T2 为 `-0.0343/-0.0560/-0.0324`；配对被试 bootstrap 均值分别为 `-0.0326` 和 `-0.0575`。因此，E2 不支持继续沿用“teacher 作为多个局部辅助头之一”的方案。

E2 没有检验以下命题：

- 完整 \(r^J\) 轨迹作为 semantic branch 的主要目标；
- fNIRS full-window encoder 对联合驱动是否可观测；
- 两套独立 token 在冻结后是否具有受控的离线时延条件关联；
- 在严格 cutoff 后，它们是否对窗外未来原始 fNIRS innovation 有增量预测力；
- 真实配对的 candidate joint teacher 是否优于 EEG-only teacher、相位匹配伪目标或破坏时延关系的 teacher；
- `K=128` 与更小词表的容量优劣。

还必须更正一个历史口径。E2 validation 有 300 个窗口、3,000 个 2 秒 patch；E0 sidecar 只覆盖 `session_01/MA` 的 50 个 validation 窗口，因此 teacher target 只有 500 个 patch，即全部 validation patch 的 `16.67%`。历史报告中的 EEG `178/500` 是旧 artifact mask 下 frozen probe 的有效支持数，不是 teacher semantic loss 实际使用的支持数；当时 loss 没有与该 signal mask 相交，teacher loss 使用了 500 个 target patch。当前代码已取消 EEG artifact mask 的有效性权威，因此旧 E2 结果只能按旧数据合同重放，不能与新 R 系列直接拼接。

这意味着 E2 是一个精确但窄的负结果：它否定旧入口和旧目标组合，不是否定“共享驱动塑造离散语义”这一更强、尚未被执行的假设。

## 🧠 理论对象：token 不是什么

生理 token 不是语言中的自然词元。语言 token 的重复使用由离散符号和语料统计共同支撑，而连续生理信号没有天然分词边界、天然同义关系或唯一离散状态。生理 token 的含义由三件事共同定义：

\[
z_t^{(m)}
= Q_m\!\left(E_m\!\left(x_{1:T}^{(m)}\right)_t\right),
\qquad m\in\{\mathrm{EEG},\mathrm{fNIRS}\},
\]

\[
\hat r_{t}^{J,(m)}=D_r\!\left(e_{z_t^{(m)}}^{(m)}\right).
\]

其中 \(x^{(m)}\) 是本模态原始测量，\(E_m\) 是只看本模态 20 秒窗口的时序编码器，\(Q_m\) 是独立 `K=128` 量化器，\(D_r\) 是共享的 driver decoder，\(r^J\) 是训练期的联合共享驱动代理坐标。token 的操作性语义不是 ID 数字，而是该原型能够稳定重建的 \(r^J\) 轨迹区域。

因此：

- EEG 的 token 17 与 fNIRS 的 token 17 没有预设同一性；
- token ID 只是在单个 codebook 内的名义标签，跨 seed 需按原型/解码轨迹匹配；
- codebook 高占用不等于语义丰富；
- 离散化后的语义必须通过重建、稳定性和外部预测共同建立，不能由聚类图或 token 名称宣布。

## 🧪 为什么 tokenizer 只接收原始生理测量

把 task、subject、trial phase、motion、teacher uncertainty 等变量量化成额外 token，会引入三个困难。

第一，tokenizer 可以沿最容易的捷径编码实验设计，而不是生理状态。第二，不同离散变量的度量结构不相容；把任务类别、连续噪声水平和信号片段放入同一词表，没有自然距离。第三，发现阶段会难以区分“生理共现”和“共同读取了同一个 nuisance token”。

因此，raw-only input contract 是合理而重要的简化。这里的 `raw-only` 明确定义为：**冻结预处理合同下的 measured EEG 或 measured HbO/HbR，加 boundary/finite 计算 mask**；不包括 teacher state、task/condition、subject、phase、motion/nuisance feature 或另一模态。它只保证输入边界纯净，并不保证所得表示天然具有生理语义，因为原始信号本身仍包含被试、设备、运动、任务阶段和预处理残留。nuisance 变量必须保留，但只用于：

- 分层采样与 split；
- 统计协变量；
- 条件置换和 matched null；
- 结果分层与敏感性分析。

它们不进入 encoder 或 codebook。由此得到的理论边界是：

> representation purity 是防止显式捷径的设计约束；semantic validity 仍然是必须通过反事实目标和 held-out 检验获得的经验结论。

## 🎯 为什么选择完整联合共享驱动代理轨迹

E0 的 \(r^J\) 是 EEG proxy 与 HbO/HbR 在一个固定区间状态空间模型中的联合平滑中间态。当前可用 target 采用 subject-specific leave-one-trial provenance：同一被试其他 trial 可参与参数、空间 anchor 与 EEG proxy 投影的拟合；因此它还不是 population-frozen 的被试外 teacher。它在构造上比任务标签更直接针对所假设的神经血管桥梁，也比 patch 均值/斜率保留更多动态形状；这不是其构念有效性已经成立的证据。用完整轨迹作为单一目标，可以避免 E2 的 `local/prototype/context` 多头分别定义语义；R1-P 仍必须另建只在 development-training subjects 拟合、对 held-out/protected subject 纯 apply 的 teacher，并重新通过 population-frozen teacher panel，不能继承旧 E0 admission。

对每个 2 秒位置，主损失为：

\[
\mathcal L_{\mathrm{driver}}
=
\frac{
\sum_{b,m,t,u}m^{\mathrm{loss}}_{b,t,u}
\rho\!\left(D_r(q_{b,t}^{(m)})_u,r^J_{b,t,u}\right)}
{\sum_{b,m,t,u}m^{\mathrm{loss}}_{b,t,u}},
\]

其中 \(m^{\mathrm{loss}}_{b,t,u}\) 是 boundary/finite patch mask、teacher support 与 target point support 的逐点交集；训练、验证和 common probe 使用同一规则，不能退化为整 patch 的标量权重。

\[
\mathcal L_{\mathrm{semantic}}
=\mathcal L_{\mathrm{driver}}
+\lambda_{\mathrm{commit}}\mathcal L_{\mathrm{VQ}}.
\]

semantic branch 不承担原始波形重建。若后续证明需要保持模态私有信息，可增加连续 private branch 重建本模态原始信号，但 raw-reconstruction 梯度不得进入任何 semantic encoder 或 codebook 参数；必须使用独立 private encoder/optimizer，或在分支点显式 detach，并用参数级 gradient allowlist 验证。仅使用独立 decoder 不足以保证隔离。

完整轨迹目标也解释了为何 patch-local MLP 不够。E0 teacher 使用整窗 fixed-interval RTS smoother，而 fNIRS 观测相对神经活动有延迟。新 encoder 必须在量化前具有本模态整窗上下文，随后输出 10 个神经时间位置的 latent。这是离线、非因果 tokenizer；同窗 endpoint 只能支持离线关联。窗外预测可只使用 cutoff 前已完整结束的窗口，并按绝对时间核验其完整 receptive field；如另建只看过去的 causal tokenizer，则必须重新通过语义门。两者都不授权因果机制措辞。

## ⚠️ 共享驱动的证据边界

\(r^J\) 不是生理真值。它是一个依赖模型族、先验、标度规约、训练折参数和整窗观测的 privileged consensus coordinate。它可能：

- 由 EEG proxy 主导；
- 把任务锁定的平滑趋势误作共享驱动；
- 受 hemodynamic model misspecification 影响；
- 通过双模态联合平滑把 fNIRS 信息反向写入监督目标；
- 在训练目标中制造后续看似跨模态的一致性。

[E0-D7 诊断](analysis/E0_D7_ADAPTIVE_SHARED_NEURAL_SSM.md)显示 \(r^J\) 与
EEG-only \(r^E\) 高相关，约为 `0.9267`，而 joint correction 的中心化标准差约为
EEG-only driver \(r^E\) 标准差的 `0.357`。这说明 teacher 既不是纯 EEG，也不能被描述为均衡可辨识的
双模态真值。它目前最合适的称谓是“联合共享驱动代理”。

尤其需要区分两个命题：

1. **表示充分性**：本模态输入能否预测这个代理；
2. **潜变量可辨识性**：这个代理是否等于唯一真实的共享神经驱动。

R 系列只试图先建立第一个命题。没有干预、额外传感器或更强生成假设时，第二个命题不能由重建损失解决。

这里还存在一个必须公开保留的治理—证据边界。[2026-07-24 的正式决定](analysis/20260724_E0_SIGN_CALIBRATED_PHYSICAL_TEACHER_ACCEPTANCE.md)把
complete E0 标为 `PASS`、授权 teacher supervision，并明确退役旧 machine/fNIRS
failure labels 作为 E0 gate criteria。与之对应的
[历史 immutable gauge-corrected run artifact](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_teacher_e0_v3_gauge_corrected_validation_v1/summary.md)
仍记录 `Physical observation reconstruction=False`、
`Synthetic posterior calibration=False`、machine conjunction 为 false，以及 fNIRS
physical gain `-0.0845`。这些诊断不改变正式 admission status，但限制从“获准作为限定监督坐标”外推到
“teacher 是真实且已校准生理状态”的解释。R1-P、R2、`E-control` 和 `Null-J` 必须重新承担
被试外可实现性与目标特异性检验；后续文档既不能否认正式 PASS，也不能用准入标签抹去构念风险。

## 🔗 共现是如何成为证据的

若两个 codebook 都接受同一个 teacher，token co-occurrence 可能只是共同监督的设计产物。因此不能用“跨模态 token 共现增加”作为最终发现指标。主张必须逐级升级：

| 层级 | 可回答的问题 | 必须通过的对照 |
| --- | --- | --- |
| C1 连续可观测 | 单个模态能否从原始测量重建 \(r^J\)？ | 常数、相位/条件、train-mean、简单连续 baseline |
| C2 离散保持 | `K=128` hard token 是否保留连续模型中的 \(r^J\) 信息？ | continuous student、soft posterior、codebook vector |
| C3 目标特异 | 提升是否依赖真实配对的 registered joint candidate？ | EEG-only \(r^E\)、fNIRS-shuffled \(r^{J,\pi}\)、平滑伪目标 |
| C4A 离线冻结依赖 | 双向整窗 EEG token 是否保留离线时延条件关联？ | fNIRS history、任务/阶段、被试、边际和时移 null；不得称未来预测 |
| C4B 窗外预测 | receptive field 截止于 endpoint 前时，EEG 表示是否增加未来原始 fNIRS innovation 的预测？ | 显式 cutoff、无重叠 endpoint、fNIRS history 与 matched null |
| C5 稳健性 | 结果是否跨被试、seed 和合理 teacher 变体稳定？ | subject bootstrap、跨 seed prototype matching、敏感性分析 |

只有 C1–C3、C4A 与 C5 连续成立，才可写“共享驱动监督产生了可重复、具有目标特异性、并具有离线跨模态时延外部效度的离散状态”。还通过 C4B 才可写“具有窗外增量预测信息”。即使如此，也不等于因果发现；因果措辞仍需干预或可识别因果设计。

## 🧩 失败路径揭示的研究债务

这段演进暴露了四种可复用的失败模式。

### 目标漂移

每个负结果后新增一个 branch、loss、teacher entry 或评估阶段，会让下一个实验回答不同问题。模型越来越强，核心假设却越来越难被直接否定。以后每个新增组件都必须对应一个已经观测到且可单独复现的失败模式，并同时登记一个能否定该组件必要性的 ablation。

### 设计对齐冒充发现

共享 codebook、index matching、跨模态 attention 或共同 teacher 都能提高表面一致性，但这些属于 imposed alignment。真正的发现必须来自冻结表示之后、面对未参与塑形的原始 endpoint 和 null controls 的增量证据。

### 工程门禁冒充科学门禁

量化器占用稳定、梯度可达、mask 正确和训练收敛只是 measurement validity。它们保证测量工具没有明显坏掉，却不保证 construct validity。E1 的 `K=128` 结论应表述为“预先指定容量下量化器可训练并保持健康”，而不是“128 是已证明最优的生理状态数”。

### teacher 接纳冒充 teacher 真实性

符号校准和 E0 admission 解决了坐标一致性与治理上的可用性，不自动证明 posterior calibration、物理重建增益或潜变量唯一性。后续文档必须同时保留原始 run summary 与治理决定，不能用“完全接纳”覆盖构念不确定性。

## 🪶 简洁性原则

今后用以下预算约束架构：

1. 两个 raw-only encoder；
2. 两个独立 `K=128, D=64` codebook；
3. 一个共享 driver decoder；
4. 一个主 semantic loss 加 VQ 正则；
5. 一个训练后冻结的外部 evaluator：主线先做离线条件关联；需要未来预测时增加严格 cutoff 模式；
6. private continuous branch 只有在信息保持失败时才进入，不作为默认核心。

不进入首轮主线的组件包括：共享 codebook、token ID 匹配、跨模态 attention、离散 nuisance/effect token、多 teacher entry、context head、coupling shaper、foundation model、联合训练的 coupling loss。它们并非永远禁止，而是必须由 R 系列的具体失败重新获得引入资格。

## 📐 最重要的可行性闸门

最危险的不是 VQ，而是 fNIRS 对 \(r^J\) 的独立可观测性。E0 teacher 使用了双模态整窗观测，且现有线性同 patch 探针在 fNIRS 侧接近零或为负。非线性 full-window encoder 也许能利用延迟结构，但这必须先由无量化的 R2 实验回答。

所以在训练任何新 codebook 之前：

1. 仅用 EEG 原始窗口训练 continuous student 重建 \(r^J\)；
2. 仅用 fNIRS 原始窗口训练同容量 continuous student；
3. 与 \(r^E\)、\(r^{J,\pi}\) 和平滑相位目标比较；
4. 在 held-out subjects 上报告每个被试的完整轨迹指标；
5. 若 fNIRS student 不超过预注册 baseline，则停止 shared-driver token 假设，不用更强 VQ 或额外 token 掩盖不可观测性。

这个 stop rule 是本次回归最关键的简化：先检查目标是否能从各自原始模态实现，再讨论离散化和共现。

## 🔐 允许的当前表述

当前可以陈述：

- E2 的弱辅助 teacher 入口没有带来预注册 semantic endpoint 增益；
- `K=128` 是已冻结且软件健康的容量预算，不在本轮缩减；
- E0 联合共享驱动是一个值得以严格对照检验的 privileged proxy；
- shared-driver primary supervision、full-window pre-VQ encoder，以及冻结后的 R6A 离线关联/R6B cutoff 预测，是尚待验证的新架构与评估合同；
- 无显著 coupling 是预先允许的结果。

当前不可陈述：

- E0 已证明真实共享神经驱动；
- 两个模态重建同一 teacher 即发现了生理耦合；
- 相同 token ID 具有跨模态同义性；
- codebook 占用或视觉聚类证明生理语义；
- foundation model 是检验耦合所必需；
- E2 已否定 shared-driver semantic tokenization。

## 🔗 权威后续

- [目标架构](02_TARGET_ARCHITECTURE.md)
- [理论基础](03_THEORETICAL_FOUNDATIONS.md)
- [实现与验证计划](04_IMPLEMENTATION_VALIDATION_PLAN.md)
- [R 系列实验设计](05_EXPERIMENT_DESIGN.md)
- [实验日志与 E2 口径更正](06_EXPERIMENT_LOG.md)
- [项目演进图](../PROJECT_EVOLUTION_MAP.md)
