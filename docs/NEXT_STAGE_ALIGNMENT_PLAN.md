# EEG-fNIRS Alignment Next-Stage Plan

> Last Updated: 2026-03-19
> Status: Planning document for the first explicit alignment stage after single-modality tokenizer validation

## 1. Goal

下一阶段的核心目标不再是证明 tokenization 可行，而是显式推动 EEG 与 fNIRS token 空间对齐，使得具有相同或稳定对应生理意义的信号片段被编码为：

1. 同一个 token
2. 或一对稳定对应的 token

只要第二种情况足够稳定，同样具备生理解释价值。因为我们最终关心的是：怎样的原始 EEG 片段会系统性地对应怎样的 fNIRS 片段，以及这种对应是否符合已知神经血流耦合规律。

## 2. Current Validated Facts

### 2.1 Single-modality tokenizers are ready enough to support alignment experiments

当前已经验证过并适合进入对齐阶段的单模态编码器包括：

- P0plus_eeg_patch_vqvae_10s_30ch_recon_20260317_220959
- P0plus_fnirs_patch_vqvae_10s_36ch_20260317_192610
- eeg_labram_vqnsp_10s_1s_20260319_141551
- fnirs_labram_vqnsp_10s_2s_20260318_192520

其中，LaBraM 组合在当前阶段最适合作为 alignment 主线：

- EEG 10s/1s 编码器的 codebook 健康度显著优于 EEG 6s/1s
- fNIRS 10s/2s 编码器的重建、频谱保持和 codebook 使用都非常稳定

### 2.2 Cross-modal mapping already exists, but it is mainly lag-aware rather than synchronous

基于独立训练好的 EEG 10s/1s 与 fNIRS 10s/2s tokenizer 的 probe 结果显示：

- 零时滞 mutual information 较弱
- 但允许时滞后，最佳 lag 很稳定
- 当前最强信号出现在 lag = 4 tokens 附近

这说明接下来的 alignment 设计不能只押注“严格同步同位点完全一致”，必须把 delayed coupling 作为一等公民。

### 2.3 The interrupted shared-codebook run is not negative evidence against the design itself

此前效果不佳的 shared-codebook LaBraM 运行是意外暂停的训练过程，不能据此否定共享码本路线本身。它最多只能说明：

- 当前这一版训练流程还不够稳健
- 现有 loss 设计主要针对同步一一对应
- 当前监控指标还不足以支撑长期 alignment 研发

因此，结论不是放弃 shared codebook，而是把它重新定位为需要系统打磨的 baseline。

## 3. Review of Existing Shared-Codebook Implementation

本节对应的当前实现主要包括：

- experiments/scripts/train_shared_tokenizer.py
- src/tokenizers/shared_labram_vqnsp.py
- experiments/configs/phase0plus/shared_labram_vqnsp_eeg_fnirs_10s_2s.yaml

### 3.1 What the current implementation already does well

当前 shared LaBraM 方案已经具备一个完整 baseline 所需的关键要素：

1. 双模态各自 encoder 和 decoder，允许 EEG 与 fNIRS 保持各自的观测空间特性。
2. 一个共享 VQ codebook，允许两模态落在同一离散索引空间。
3. 显式 alignment loss：
   - latent alignment: 对齐连续 latent
   - assignment alignment: 对齐 code assignment 分布
4. 配套训练入口、checkpoint、验证和日志记录已经齐备。

这意味着下一阶段并不是从零开始，而是在一个已经能跑通的 shared-codebook baseline 上继续扩展。

### 3.2 What assumptions are hard-coded in the current implementation

当前 shared LaBraM 默认内含以下强假设：

1. EEG 与 fNIRS 在每个 window 内必须产生相同数量的 token。
2. 第 t 个 EEG token 应与第 t 个 fNIRS token 直接对齐。
3. 对齐主要发生在同位 token 上，而不是一个 lag window 内。
4. 目标更接近“共享 token identity”，而不是“稳定 paired-token correspondence”。

这些假设对于一个起始 baseline 是合理的，但对于真实 EEG-fNIRS 生理过程而言偏强。

### 3.3 What is missing for the current research objective

如果研究目标升级为：

- 相同生理意义的片段尽量编码为同一个 token
- 或编码为严格稳定对应的一对 token

那么当前实现还缺少三类关键能力：

1. Lag-aware alignment
   - 现实现仅比较同位 token，没有显式支持时滞窗口或可学习时滞。

2. Paired-token alignment
   - 现实现默认最好是同一个 token，但没有显式建模“EEG token 101 总是对应 fNIRS token 300”这一类稳定映射。

3. Curriculum / warm-start support
   - 现实现从头训练 shared model，没有直接利用已验证过的单模态 encoder/decoder 作为初始化锚点。

## 4. Decision: Do We Need New Schemes?

需要，但不是替换，而是并行增加。

### 4.1 Keep the current shared-codebook design as the baseline

原因：

- 它最直接对应“同一个 token 表示跨模态共同生理意义”这一理想目标。
- 结构简单，便于解释和比较。
- 已经有训练脚本、配置和日志体系。

因此，shared LaBraM baseline 应继续保留，并作为后续所有新增 alignment 方案的对照组。

### 4.2 Add at least two new alignment schemes

新增方案不是因为 shared codebook 思路错误，而是因为现有 baseline 只覆盖了最强版本的同步同位对齐假设，没有覆盖更符合当前数据证据的两种情况：

1. 延迟对应
2. 配对但不相同的 token 对应

因此，至少应新增：

- 方案 A: Lag-aware shared alignment
- 方案 B: Paired-code alignment without forced identity

### 4.3 Add one training-strategy upgrade even for the baseline

即使不改模型结构，也建议补一项训练流程增强：

- 方案 C: Warm-start shared training from validated single-modality checkpoints

这是低风险高收益项，应优先于大改模型。

## 5. Proposed Alignment Schemes

## 5.1 Scheme S0: Shared-Codebook LaBraM Baseline

### Purpose

验证“共享 token identity”是否能够在现有 paired EEG-fNIRS 数据上自然出现。

### Core mechanism

- 一个 shared codebook
- EEG/fNIRS 双 encoder 双 decoder
- 同位 latent alignment
- 同位 assignment alignment

### Role in the roadmap

- 作为所有新增方案的统一对照组
- 验证 shared token space 的上限和训练稳定性

### Immediate improvements needed

1. 支持从单模态 tokenizer 初始化 shared encoder / decoder。
2. 支持 alignment loss warmup，而不是从 epoch 1 就全量施压。
3. 增加 lag-aware offline validation 指标，避免只看同步 token match。
4. 更稳妥地 resume scheduler / optimizer / early-stopping 状态。

## 5.2 Scheme S1: Warm-Started Shared Codebook

### Purpose

减少从头训练 shared model 的难度，把单模态已经学到的重建能力先保住，再逐步注入共享语义。

### Core idea

1. 从已完成的单模态 checkpoint 中加载 EEG encoder/decoder 到 shared model 的 EEG 分支。
2. 同样加载 fNIRS encoder/decoder 到 fNIRS 分支。
3. shared quantizer 可以采用：
   - 随机初始化 + 短 warmup
   - 或从其中一个模态 codebook 初始化
   - 或拼接后聚类再初始化
4. 训练前期弱化 alignment loss，优先稳住 reconstruction。

### Why this matters

当前我们已经知道单模态编码器是有效的，因此没有必要每次 alignment 实验都重新学习全部底层表征。

## 5.3 Scheme S2: Lag-Aware Shared Alignment

### Purpose

把神经活动和血流响应之间的时滞直接纳入 loss，而不是指望模型自己从同步约束里“猜”出来。

### Core idea

对每个 EEG token 位置 t，不只和 fNIRS 的 t 做对齐，还和一个小范围 lag 集合内的位置比较，例如：

$$
\tau \in \{0, 1, 2, 3, 4\}
$$

然后对 latent loss 和 assignment loss 采用以下之一：

1. hard best lag
2. softmin over lags
3. learnable lag attention

一种可行的起始形式是：

$$
L_{align}^{lag} = \text{softmin}_{\tau \in \mathcal{T}}\left[
\lambda_z \lVert z^{eeg}_t - z^{fnirs}_{t+\tau} \rVert_2^2 +
\lambda_q \operatorname{KL}(p^{eeg}_t \Vert p^{fnirs}_{t+\tau})
\right]
$$

### Why this matters

现有 probe 已经显示 best lag 远强于 sync lag，因此这一方案是下一阶段最有必要新增的主方案。

## 5.4 Scheme S3: Paired-Code Alignment

### Purpose

直接建模“稳定一对多或一对一 token 对照关系”，而不是强迫 EEG 与 fNIRS 使用同一个 code index。

### Core idea

保留两种模态各自的 codebook，但学习一个稳定映射：

$$
M \in \mathbb{R}^{K_{eeg} \times K_{fnirs}}
$$

其中：

- 第 i 行表示 EEG token i 对应到哪些 fNIRS token 的概率分布
- 训练目标鼓励 paired windows 的 token co-occurrence 被 M 解释
- 当 M 某些行高度尖锐时，就得到“EEG token 101 对应 fNIRS token 300”这类可解释对照

### Possible implementations

1. 简单映射矩阵 + 交叉熵 / KL
2. Prototype pairing head
3. Sinkhorn / optimal transport 约束的双向匹配矩阵

### Why this matters

这条路线最贴近你提出的“严格对照两个 token”的需求，而且不必强迫两模态共享同一个 codebook 尺寸和使用习惯。

## 5.5 Scheme S4: Shared + Private Codebooks (Optional later)

### Purpose

如果发现完全共享的 token 空间总是牺牲重建质量，或者完全独立的 token 空间又缺乏跨模态语义，那么 shared + private 结构会是更自然的折中。

### Core idea

- shared codebook 负责跨模态共性
- private codebook 负责模态特有成分
- decoder 使用 shared + private token 共同重建

### Priority

这条路线有价值，但不应该先于 S1 和 S2。因为它会显著提高结构复杂度。

## 6. Recommended Development Order

建议按以下顺序推进，而不是同时发散到过多大改方案：

### Step 1. Solidify the existing shared-codebook baseline

目标：让 S0 成为可信 baseline。

工程任务：

1. 校正并保留现有 shared LaBraM 配置。
2. 确认 interrupted run 的 resume 流程稳定。
3. 为 shared training 增加更细的日志与验证输出。
4. 增加 offline lag-aware evaluation，而不只记录 sync token_match。

### Step 2. Add warm-start shared training

目标：提高 shared baseline 的收敛稳定性。

工程任务：

1. 在 train_shared_tokenizer.py 中增加 optional checkpoint initialization。
2. 支持单独加载 EEG / fNIRS 分支权重。
3. 增加 alignment warmup 配置。

### Step 3. Add lag-aware alignment as the main new model variant

目标：显式适配神经血流时滞。

工程任务：

1. 在 shared_labram_vqnsp.py 中增加 lag-aware alignment mode。
2. 在 config 中增加 lag_set、aggregation mode、lag regularization。
3. 记录 best lag、lag-wise MI、lag-wise token agreement。

### Step 4. Add paired-code mapping as the main alternative route

目标：验证“paired token”是否比“same token”更自然、更稳定。

工程任务：

1. 新增 pair-mapping module 或 prototype mapping head。
2. 基于独立 tokenizer 或 shared encoder 输出训练 mapping。
3. 统计映射矩阵的尖锐度、稳定性和任务依赖性。

## 7. File-Level Engineering Plan

## 7.1 train_shared_tokenizer.py

建议新增：

1. 分支 warm-start 参数
   - --init-eeg-checkpoint
   - --init-fnirs-checkpoint

2. alignment schedule
   - 前若干 epoch 只训重建或极弱 alignment
   - 之后逐步拉高 latent / assignment 权重

3. 更完整的 resume
   - scheduler state
   - best monitor
   - epochs_without_improvement

4. 额外验证指标
   - lag-aware validation summary
   - paired-token concentration metrics

## 7.2 shared_labram_vqnsp.py

建议新增：

1. lag-aware alignment mode
2. optional paired-code head
3. optional shared/private hybrid extension hook
4. alignment metrics that do not assume identity only

## 7.3 experiments/configs/phase0plus/

建议新增三组配置：

1. shared_labram_warmstart_eeg_fnirs_10s_2s.yaml
2. shared_labram_lag_align_eeg_fnirs_10s_2s.yaml
3. paired_code_alignment_eeg_fnirs_10s.yaml

## 8. Experiment Matrix

| ID | Route | Main hypothesis | Priority |
|----|-------|-----------------|----------|
| A0 | S0 Shared baseline | 同位共享 token 可自然形成 | High |
| A1 | S1 Warm-start shared | 单模态初始化可显著稳住 shared training | Highest |
| A2 | S2 Lag-aware shared | 引入时滞后 shared alignment 明显增强 | Highest |
| A3 | S3 Paired-code mapping | paired token 比 identical token 更自然 | High |
| A4 | S4 Shared + private | 共性与模态特性能更好解耦 | Medium |

## 9. Success Criteria

下一阶段不应再只用 reconstruction 判断成败，而应同时满足以下三类标准：

### 9.1 Single-modality quality is preserved

- EEG 与 fNIRS 的 reconstruction 不应比当前最好单模态模型显著退化。
- codebook utilization 与 perplexity 不能明显塌缩。

### 9.2 Cross-modal alignment becomes stronger in the intended sense

- 若目标是 same-token alignment：同步或 lag-aware token agreement 提升。
- 若目标是 paired-token alignment：映射矩阵变得尖锐、稳定、跨被试可复现。

### 9.3 The result becomes more interpretable rather than merely more coupled

- paired token 应能回溯到相对稳定的 EEG/fNIRS 原始波形模式。
- 这些模式应尽量符合已知生理规律，而不是纯统计偶合。

## 10. Risks and Mitigations

### Risk 1. Shared training collapses codebook usage

Mitigation:

- warm-start
- alignment warmup
- 分阶段增大 alignment 权重

### Risk 2. The model overfits synchronous correspondence and misses physiological lag

Mitigation:

- 把 lag-aware alignment 设为主方案之一
- offline evaluation 强制输出 lag curve

### Risk 3. Same-token objective is too strict

Mitigation:

- 并行推进 paired-code mapping
- 不把 token identity 作为唯一成功标准

### Risk 4. Better alignment hurts reconstruction too much

Mitigation:

- 始终保留 reconstruction floor
- 所有实验都与单模态 best model 对比

## 11. Immediate Recommendation

最推荐的下一阶段起步顺序是：

1. 保留并继续 shared LaBraM baseline，不因意外暂停的 run 否定该路线。
2. 优先补 warm-start shared training，这是最低风险、最高回报的增强项。
3. 将 lag-aware shared alignment 作为下一阶段的主新增方案。
4. 将 paired-code mapping 作为与 shared identity 平行的第二主方案。

换言之，下一阶段不是在“shared codebook”与“new scheme”之间二选一，而是：

- 继续 shared codebook baseline
- 同时新增对时滞和 paired-token 友好的 alignment 方案

这才与当前实验事实一致，也最符合后续生理解释分析的目标。