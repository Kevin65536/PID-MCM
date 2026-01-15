
## 1. 你的想法在当前文献图景中的位置

你的设想是：

> 用**一个统一的 codebook** 同时离散化 EEG 与 fNIRS，使得“某个 EEG token 周围的 fNIRS 上下文服从特定的概率分布”，从而在 token 级别实现两个模态的对齐与建模。

从已经检索到的工作来看：

- 在 **EEG foundation model** 方向，基于 codebook / VQ 的离散化已经是主流路线之一（LaBraM、LaBraM++、NeuroRVQ、NeuroLM、CSBrain 等）。
- 在 **EEG–fNIRS 多模态** 方向，目前主流是：
  - 共享 latent space + 对比学习（如 EFRM）[1]
  - 跨模态注意力融合（如 MBC-ATT）[2]
  - EEG→fNIRS 生成（如 SCDM）[3]
- 但：**尚没有**工作针对 EEG+fNIRS 设计类似“统一 codebook / 共享 token 空间”的 foundation model。

同时，在其他脑成像和多模态领域有几个强烈的类比信号：

- **Brain Harmony**：在结构 MRI + 功能 fMRI 之间通过“shared brain hub tokens”实现多模态统一 1D token 空间[4]。
- **多模态 vision–language 模型** 中，已经有比较成熟的“统一 codebook / Cascaded codebooks”框架（如 UniCode²、TokenFlow 等）用于图像、视频等模态共享离散空间[5]。

综合来看：  
**你的设想目前在 EEG–fNIRS 场景下是“空白 + 自然扩展方向”**，在思想上与 Brain Harmony 和多模态 LLM 的统一 token 化高度一致，因此**是合理且有研究价值的方向**。

---

## 2. 已有 EEG codebook 方法的关键点（为多模态扩展打基础）

### 2.1 LaBraM / LaBraM++：频谱 + VQ 代价本

- 将多通道 EEG 分割成 patch，经过神经频谱预测器输出幅值 A 与相位 φ 的频域表示；
- 使用**向量量化**（VQ）把 patch 表示映射到一个大小约为 8192×64 的 codebook，获得**离散神经 token**；
- LaBraM++ 进一步提出用 `sin(φ), cos(φ)` 替代表相位直接回归，解决相位在 ±π 处的不连续梯度问题[6]；
- 训练目标是：**重构频谱（幅值 + 相位） + VQ 量化损失**，从而学到语义丰富的离散 token。

### 2.2 NeuroRVQ：多尺度残差 VQ codebook

- 提出 **Hierarchical Residual VQ (RVQ)**：同一 patch 经过 N 级 codebook 逐级量化残差，每级 K=8192, D=128，得到高分辨率 token 组合[7]；
- 同样是频域重构、并引入相位/幅度感知的损失；
- 明确宣称该 tokenizer “为通用脑电 codebook 奠定先验，利于日后多模态生物信号集成”[7]——虽然论文中还没做到 fNIRS，但**方向与“通用多模态 codebook”完全对齐**。

### 2.3 NeuroLM、CSBrain 等

- **NeuroLM**：先训练 VQ neural tokenizer，将 EEG → 离散 neural tokens，再把这些 token 当作“外语”输入 LLM，实现多任务 EEG 解码[8]；
- **CSBrain**：提出 Cross-scale Spatiotemporal Tokenization (CST)，在时间窗口与脑区内做多尺度卷积，将 EEG 转为“跨尺度时空 token”供结构化稀疏注意力使用[9]；
- 这些都在强化一个事实：**“EEG→token→Transformer/LLM” 已是成熟范式**。

对你而言有两点重要启示：

1. **频域 + VQ 的 EEG tokenizer 已经被反复验证有效**，可以直接拿来作为多模态中的“EEG 分支”；
2. NeuroRVQ 明确把“通用脑波 codebook + 多模态集成”写在愿景中，说明从 EEG VQ 向多模态 VQ 扩展的思路与主流路线一致。

---

## 3. EEG–fNIRS 多模态：现在怎么做，而没做到哪里？

### 3.1 代表性 EEG–fNIRS 模型

1. **MBC-ATT**（跨模态注意力融合）[2]  
   - EEG 和 fNIRS 分别用 CNN 提取特征；
   - 映射到同一 hidden space，再各自生成 Q/K/V；
   - 做双向 cross-attention（EEG→fNIRS / fNIRS→EEG），再分类。  
   → 特点：**连续向量空间 + 注意力融合，没有离散 token 或 codebook。**

2. **EFRM（EEG–fNIRS Representation Model）**[1]  
   - 预训练阶段：
     - 对每个模态使用 Masked Autoencoder（MAE），学习模态特定表示；
     - 对 paired EEG–fNIRS 使用对比学习，拉近同一试次的 EEG / fNIRS embedding；
   - 下游可支持 EEG-only、fNIRS-only 以及 paired EEG–fNIRS。  
   → 特点：**“模态专属 encoder + 共享 latent space”的典型设计，仍无离散 token。**

3. **SCDM（EEG→fNIRS Cross-Modal Diffusion）**[3]  
   - 通过 spatio-temporal controlled diffusion model，从 EEG 生成 fNIRS；
   - 引入 Multi-scale Temporal Representation (MTR) 和 Spatial Cross-Modal Generation (SCG) 模块；
   - 实验表明合成 fNIRS 在分类上能与真实 fNIRS 相当。  
   → 特点：建模 `P(fNIRS | EEG)`，但在**连续空间**下做，不引入 codebook。

4. 多篇 EEG–fNIRS BCI 研究（EF-Net、ECA-FusionNet 等）普遍采用：
   - 早期特征拼接（feature-level fusion）
   - 共享 MLP/CNN backbone
   - 或 MI-based feature selection[10]  
   → 核心仍在**特征融合，而不是 token 化/离散表示。**

### 3.2 关键缺口

从所有这类工作中可以明确说：

- **没人用共享 codebook 来同时量化 EEG 与 fNIRS**；
- 很多工作在做“共享 latent space”，但都是**连续隐空间**；
- 即使是最接近你想法的 NeuroRVQ，也只是说“促进多模态生物信号集成”，没有落到 EEG–fNIRS 实验。

因此，你设想的“一个统一 codebook + token 级跨模态概率建模”，在 EEG/fNIRS 领域可以说**尚属首提**，至少在公开文献中还没有直接实现。

---

## 4. 理论上这是不是合理？

我认为：**从神经生理和建模两个层面看，这是合理且有潜在优势的，但不能一刀切要求“完全共享一个 codebook”，更现实的是“共享 + 模态专属的混合 codebook”**。

### 4.1 生理层面：为何“一个 EEG token 对应一簇 fNIRS 分布”是有道理的？

1. **神经电–血流动力学耦合（Neurovascular Coupling）**  
   - EEG 反映 ms 级同步电活动；  
   - fNIRS 反映 s 级血氧变化（HbO/HbR/HbT）；  
   - 经典观点：**“功能随结构，血流随电活动”**（function follows structure），Brain Harmony 也正是基于这一原则统一 fMRI/MRI token[4]。  
   → 对于特定任务范式和脑区，一个 EEG 频谱–时空模式，确实会在几秒后诱发特定形态的 hemodynamic response。  
   这就支持你的假设：**给定某一 EEG token，其时间邻域内的 fNIRS token 分布是有条件结构的，而非任意的。**

2. **已有 EEG–fNIRS 工作的间接证据**  
   - EEG/fNIRS 同步实验普遍发现，特定 ERD/ERS 模式与 HbO/HbR 变化存在稳定关联（如 motor imagery, workload, emotion 任务的多篇工作与综述[2][10]）。  
   - SCDM 的 EEG→fNIRS 生成实验表明，从 EEG continuous latent 中能预测/重构出相当逼真的 fNIRS 时空模式[3]。  
   → 如果连续 latent 能做到这一点，那么在足够表达力的 codebook + transformer 下，**离散 token 空间同样可以逼近这种条件分布**。

### 4.2 表示学习层面：统一 codebook 的利弊

**潜在优势：**

- **统一表示空间，便于跨模态迁移与共享先验**  
  类似 Brain Harmony 用“hub tokens”统一 MRI+fMRI，图像–文本用统一 codebook 的 MIO, UniCode² 等[5]。
- **可以直接做 token-level 跨模态建模**，例如：
  - `P(z_fNIRS | z_EEG)`，做 EEG→fNIRS token 预测；
  - `P(z_EEG_next | z_EEG_prev, z_fNIRS_context)`，用 fNIRS 提供慢变量上下文。
- **工程上方便复用现有 EEG tokenizer**：比如先用 LaBraM / NeuroRVQ 训练的 EEG codebook，再扩展支持 fNIRS。

**主要风险与挑战：**

1. **模态统计差异极大**  
   - EEG 是高频、多通道、噪声强、频谱结构明显；
   - fNIRS 是低频、少通道（或中等密度）、信号缓慢、SNR 也不高；
   - 如果强行完全共享一个 codebook，可能导致：
     - codebook 更贴近 EEG，fNIRS 被迫“挤进”不适合的簇；
     - 或者反之，使 codebook 变得过于“折中”，两个模态都学不好。

2. **时间尺度错位**  
   - EEG token 可用 200 ms 窗长；  
   - fNIRS 更合理是 1–3 s 甚至更长；  
   - 直接一个 codebook 去量化“不同时间粒度下的 patch”，需要在 encoder 部分做足够的归一和跨尺度建模（CSBrain 的 CST 思路其实很值得借鉴[9]）。

3. **codebook collapse（退化到少量使用）**  
   - 多模态 codebook 学习中很常见：强模态主导，弱模态只用到少量 code；
   - 最近 vision–language 的多模态 codebook 论文（如 Discrete Tokenization for Multimodal LLMs）也专门分析了 collapse 问题[5]。

**更合理的折中方案**（也是我更推荐你尝试的）：

> 用**“共享 + 模态专属”混合 codebook，而不是硬性使用一个完全统一 codebook。**

例如：

- 每个 patch 的 latent 经过 encoder 后，被拆分为：
  - 前一部分与 Shared Codebook 对齐（跨模态共享语义）；
  - 后一部分与 Modality-Specific Codebook 对齐（保留模态私有信息）。

这种设计在 vision–text 多模态 codebook 里已有类似思路，在脑成像方向上 Brain Harmony 也采用了类似“共享 hub + 模态分支”的范式[4]。

---

## 5. 如何把你的想法落到可做的研究方案？

结合现有文献，我会建议你按“循序渐进 + 明确验证假设”的方式来设计：

### 5.1 第一步：保守版 —— 各自 codebook + 共同 latent space

- 对 EEG：使用 LaBraM++ 或 NeuroRVQ 的设计，训练一个高质量 EEG tokenizer（频谱重构 + VQ/RVQ）。
- 对 fNIRS：设计一个**类 LaBraM 的 fNIRS tokenizer**：
  - 用多尺度时间卷积 + 简单的空间卷积（对应光源–探测器分布）；
  - 同样预测 fNIRS 的频域（低频）谱或时域 patch，训练一个单独 fNIRS codebook。
- 在 token 级别，通过 transformer + 对比学习拉近“同时刻的 EEG token 序列”与“延迟后的一段 fNIRS token 序列”的 embedding。  
  → 这一步相当于多模态的 EFRM，但在 token level 工作，而不是 continuous latent。

**目的**：证明“EEG token 附近的 fNIRS token 分布确实存在结构可学习”，为共享 codebook 奠定经验基础。

### 5.2 第二步：中间版 —— 共享 + 私有混合 codebook

在上一步的 tokenizer 基础上，改造为：

- `z = [z_shared, z_EEG_private]` 用于 EEG；
- `z' = [z_shared, z_fNIRS_private]` 用于 fNIRS；
- 共享部分使用 **同一套 codebook**，训练时加入以下损失：

  - **跨模态一致性损失**：  
    对于同一时间窗口内的 EEG/fNIRS，对 `z_shared` 施加对比或 KL 正则，使它们靠近；
  - **模态平衡损失**：  
    约束共享 codebook 的使用频率在两个模态之间不要过于失衡（参考多模态 VQ–VAE 的 anti-collapse 技术）。

这一步中，你就已经在做你最初设想的事情，只不过是放在共享子空间上，而不是整个表示空间。

### 5.3 第三步：激进版 —— 完全统一 codebook

在有了前两步的经验后，如果你想做更“干净”的理论验证，可以尝试：

- 只保留一个统一的 codebook；
- 升级 encoder 结构，使 EEG 与 fNIRS 在 encoder 后已经处于形态相近的 latent space 中（可以借鉴 CSBrain 的多尺度时空 tokenization 以及 Brain Harmony 的几何预对齐思想）。

这一步风险比较大（尤其是对 fNIRS），但一旦做通会非常有话题性。

---

## 6. 回到你的核心问题：**“一个 codebook 合并两个模态合理吗？”**

综合所有已知事实，我的结论是：

1. **从理论与宏观趋势看，这是合理且重要的方向**  
   - 与 Brain Harmony 在 MRI+fMRI 上做的事情概念上完全一致，只是你面对的是 EEG+fNIRS；
   - 与 LaBraM/NeuroRVQ/NeuroLM 这类 EEG codebook 工作的“向多模态扩展”愿景高度契合；
   - 现有 EEG–fNIRS 模型只停留在连续 latent 的共享，还没有人进入“token / codebook 级统一”。

2. **但从具体实现看，“单一完全共享 codebook”未必是最优的工程策略**  
   - 模态差异（频率、噪声、通道拓扑）和血流延迟会让统一 codebook 面临 collapse 或折中表征的问题；
   - 更实用的方案往往是**“共享 + 私有 codebook”混合设计**，通过额外损失约束共享部分真正学到跨模态语义。

3. **你的“EEG token 周围 fNIRS 概率分布”这个建模视角非常值得抓紧做**  
   - 文献中尚无人在 EEG–fNIRS 上做 “token-level conditional modeling”；
   - 你可以直接把研究问题形式化为：
     \[
     P(z^{\text{fNIRS}}_{t:t+\Delta} \mid z^{\text{EEG}}_{t-\tau:t})
     \]
     再通过 codebook + transformer + masked modeling 来估计；
   - 这可以成为一个明确、可评估的新 benchmark 任务。

---

## 7. 建议你可以立刻做的两件事（可行且有创新）

1. **用 REFED 数据集做一个“小规模原型”**  
   - 只用 LaBraM++ 的 EEG tokenizer（冻结 codebook），  
   - 为 fNIRS 训练一个 encoder，将 fNIRS patch 直接量化到这个 EEG codebook 上，观察：
     - fNIRS 在共享 codebook 中的使用谱（哪些 token 被频繁用到）；
     - 同一时间窗口 EEG / fNIRS 所选 token 的联合分布是否显著偏离独立假设。
   → 这是最低成本的“统一 codebook 可行性”实证检验。

2. **设计一个跨模态 masked-token 任务**  
   - 任务：遮蔽一段 fNIRS token，只给对应时窗及之前的 EEG token，让模型去预测被遮蔽 fNIRS token；
   - baseline：只用 fNIRS 上下文预测自己 vs 加入 EEG token；  
   - 如果统一 codebook 或共享子 codebook 能显著提升预测性能，那就是对你假设的直接支持。

---

## 总结一句话：

- **在现有文献中，还没有人用统一 codebook 来对齐 EEG 与 fNIRS，但从神经生理和多模态 foundation model 的发展趋势看，你的想法是合理的，也是一个非常自然、前沿且空白的研究方向。**
- 工程上，更建议从“共享 + 私有 codebook”做起，逐步向更强的统一过渡，并通过“EEG→fNIRS token 条件分布”这样的具体任务来验证你的核心假设。

---

### References

[1] EFRM: A Multimodal EEG–fNIRS Representation-learning Model. https://www.sciencedirect.com/science/article/abs/pii/S0010482525016464  
[2] Multimodal MBC-ATT: cross-modality attentional fusion of EEG and fNIRS. https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2025.1660532/full  
[3] SCDM: Unified Representation Learning for EEG-to-fNIRS Cross-Modal Generation in MI-BCIs. https://pubmed.ncbi.nlm.nih.gov/40031192/  
[4] Brain Harmony (BrainHarmonix): A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens. https://arxiv.org/abs/2509.24693  
[5] Discrete Tokenization for Multimodal LLMs & UniCode² 等综述/方法（多模态 codebook 学习）。https://arxiv.org/pdf/2507.22920, https://arxiv.org/html/2506.20214v1  
[6] LaBraM++: Advancing Brainwave Modeling with a Codebook-Based Tokenizer. https://arxiv.org/pdf/2505.16724.pdf  
[7] NeuroRVQ: Multi-Scale EEG Tokenization for Generative Large Brainwave Models. https://arxiv.org/abs/2510.13068  
[8] NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals. https://arxiv.org/abs/2409.00101  
[9] CSBrain: A Cross-scale Spatiotemporal Brain Foundation Model for EEG Decoding. https://arxiv.org/pdf/2506.23075.pdf  
[10] Multimodal fNIRS–EEG sensor fusion: Review of data-driven methods. https://pmc.ncbi.nlm.nih.gov/articles/PMC12592382/

---

## 一、形式类比：从 Word2vec 到 EEG–fNIRS 条件概率

### 1.1 Word2vec 的条件概率是怎样的？

以 Skip-gram 为例，word2vec 实际上在最大化：
\[
\sum_{w_t \in D}\sum_{c \in \mathcal{C}(w_t)}
\log P(w_c \mid w_t)
\]

其中  
\[
P(w_c \mid w_t)=\frac{\exp(\mathbf{v}_{w_c}^\top \mathbf{v}_{w_t})}{\sum_{w\in V}\exp(\mathbf{v}_{w}^\top \mathbf{v}_{w_t})}
\]

- \(w_t\)：中心词  
- \(w_c\)：上下文词  
- \(\mathbf{v}_{w}\)：词向量  
- 分母在实际中用负采样近似（SGNS）

在**概率论视角**下，它做的是：

> 学一个参数化分布 \(P_\theta(\cdot \mid w_t)\)，使它在大语料中“尽量接近真实的上下文经验分布”。

### 1.2 把这个形式迁移到 EEG–fNIRS

你关心的是：
\[
P\big(z^{\text{fNIRS}}_{t:t+\Delta} \mid z^{\text{EEG}}_{t-\tau:t}\big)
\]

把它拆开看：

- 条件：一段 EEG token 序列 \(z^{\text{EEG}}_{t-\tau:t}\)，相当于 word2vec 里的“中心词及其局部历史”
- 输出：一段 fNIRS token 序列 \(z^{\text{fNIRS}}_{t:t+\Delta}\)，相当于“上下文词序列”
- 只是这里“上下文”跨了模态，同时存在**时间延迟 \(\tau\)** 和**时间尺度差异 \(\Delta\)**。

完全平行地，可以写：
\[
P\big(z^{\text{fNIRS}}_{t:t+\Delta} \mid z^{\text{EEG}}_{t-\tau:t}\big)
=\frac{\exp\big(\phi(z^{\text{EEG}}_{t-\tau:t})^\top \psi(z^{\text{fNIRS}}_{t:t+\Delta})\big)}
{\sum_{k \in \mathcal{Z}_{\text{fNIRS}}}\exp\big(\phi(z^{\text{EEG}}_{t-\tau:t})^\top \psi(k)\big)}
\]

- \(\phi(\cdot)\)：把 EEG token 序列编码成一个“查询”向量  
- \(\psi(\cdot)\)：把 fNIRS token 或 token 片段编码成“候选上下文”向量  
- \(\mathcal{Z}_{\text{fNIRS}}\)：fNIRS codebook 中所有可能 token / 片段的集合  

**形式上**，这与 word2vec 的  
\(
P(w_c \mid w_t)\propto \exp(\mathbf{v}_{w_c}^\top\mathbf{v}_{w_t})
\)  
是几乎一模一样的。

---

## 二、关键异同：为什么“看起来一样，内在却不一样”？

### 2.1 相同点：都是“条件分布 + 向量相似度”

- 都在学习一个**条件概率分布**：  
  - word2vec：\(P(\text{上下文词} \mid \text{中心词})\)  
  - 你这里：\(P(\text{fNIRS 片段} \mid \text{EEG 片段})\)

- 都把条件分布写成“向量相似度 + Softmax / NCE”的形式：
  - 用内积或相似度函数 \(\text{sim}(q,k)\) 来度量“这个上下文是否合理”
  - 通过最大化对数似然或 InfoNCE，让**真实上下文的得分 > 虚假上下文的得分**

- 都隐含一个“分布式语义 / 分布式神经状态”的假设：
  - 相似的中心词有相似的上下文分布  
  - 相似的 EEG token 有相似的 fNIRS 条件分布

### 2.2 本质差异 1：生成机制——统计共现 vs 生理耦合

**Word2vec：**  
- 分布来自**语料统计**：人类语言共现模式  
- 并没有一个物理生成过程，只是高维经验分布

**EEG–fNIRS：**  
- 分布来自**神经电活动 → 血流动力学反应**的**物理机制**（neurovascular coupling）  
- EEG 在 ms 级捕捉电活动，fNIRS 在 s 级捕捉慢血流响应  
- 即：
  \[
  z^{\text{EEG}}_{t-\tau:t} \xrightarrow{\text{neural}} \text{局部代谢需求} \xrightarrow{\text{血流调节}} z^{\text{fNIRS}}_{t:t+\Delta}
  \]
- 因此  
  \(P(z^{\text{fNIRS}} \mid z^{\text{EEG}})\)  
  在理想极限下更接近一个**有物理约束的“条件密度”**，而不是纯经验分布。

这直接导致：**只要你的 token 设计贴合频段/脑区结构，这个条件分布本身就具有物理可解释性。**

### 2.3 本质差异 2：时间与尺度

- word2vec 的“上下文”是一个离散窗口（±k 个词），没有**系统性时间延迟**
- EEG–fNIRS 必须显式考虑：
  - 延迟 \(\tau\)（几秒）
  - 响应窗长 \(\Delta\)（5–10 秒）
  - EEG 高采样 / 高频，fNIRS 低采样 / 低频

因此，你的条件分布更像是：
\[
P\big(z^{\text{fNIRS}}_{t:t+\Delta} \mid z^{\text{EEG}}_{t-\tau:t}\big)
\approx P\big(\text{慢变量} \mid \text{快变量的时间卷积}\big)
\]

### 2.4 本质差异 3：空间拓扑与结构

- word2vec 的词表是**无拓扑的索引集合**，邻接关系是“统计语义”  
- fNIRS token 背后是**光源–探测器对 + 皮层区域**，有明确的**解剖拓扑**：
  - 相邻通道 ≈ 相邻皮层区域
  - EEG 通道布局 + fNIRS 通道布局在同一头模型上对齐时，更有结构性

这意味着：
- 条件分布不仅是“哪个 token 概率高”，还包含**在皮层空间上的分布模式**  
- 这对可解释性极其重要（可以画“条件概率脑图”）。

---

## 三、从这个类比推导可用的具体方法

下面只说“你现在就可以用”的技术路线，并点明和 word2vec 对应的部分。

### 3.1 路线一：跨模态 Skip-gram / 对比学习（最直接的类比）

**目标：**  
用 EEG 片段作为“中心”，用同时段/延迟对齐的 fNIRS 片段作为“上下文”，学习：
\[
P_\theta\big(z^{\text{fNIRS}}_{t:t+\Delta} \mid z^{\text{EEG}}_{t-\tau:t}\big)
\]

**做法：**

1. **token 化与 codebook（前一步你已经有思路）**
   - EEG：用类 LaBraM / NeuroRVQ 的 VQ tokenizer，把窗口后的 EEG → \(z^{\text{EEG}}\) token 序列
   - fNIRS：设计类似 token 化（可以是频域低频谱、或者时域 patch + VQ）

2. **构造训练样本**
   - 对每个时间点 \(t\)：取 EEG token 序列 \(z^{\text{EEG}}_{t-\tau:t}\)  
   - 取相应的 fNIRS token 序列 \(z^{\text{fNIRS}}_{t:t+\Delta}\) 作为“正上下文”
   - 取其他试次/其他时间窗的 fNIRS token 作为“负上下文”

3. **定义打分函数（对应 word2vec 的内积）**
   \[
   s^+ = \phi(z^{\text{EEG}}_{t-\tau:t})^\top \psi(z^{\text{fNIRS}}_{t:t+\Delta})
   \]
   \[
   s^-_k = \phi(z^{\text{EEG}}_{t-\tau:t})^\top \psi(z^{\text{fNIRS}}_{k}) \quad (k\in\text{负样本})
   \]

4. **使用 InfoNCE / SGNS 类似的目标**
   \[
   \mathcal{L} = -\log \frac{\exp(s^+/T)}{\exp(s^+/T)+\sum_{k}\exp(s^-_k/T)}
   \]
   - 这就是“跨模态 Skip-gram”：给定 EEG，“把真正的 fNIRS 上下文拉近，把随机 fNIRS 拉远”。

**得到什么？**  
- EEG encoder \(\phi\) 和 fNIRS encoder \(\psi\)：把两个模态嵌入到一个**对齐的语义空间**  
- 一个隐式的条件分布估计器：  
  \(P(z^{\text{fNIRS}}\mid z^{\text{EEG}})\propto\exp(\phi^\top\psi)\)

---

### 3.2 路线二：跨模态 Masked Token Prediction（BERT 式）

这里的类比不是 Skip-gram，而是 BERT MLM，只不过 MLM 的“词”换成了 fNIRS token，条件一部分来自 EEG。

**任务定义：**

- 给定：
  - 一段 EEG token 序列 \(z^{\text{EEG}}_{t-\tau:t}\)
  - 一段 fNIRS token 序列 \(z^{\text{fNIRS}}_{t:t+\Delta}\)
- 随机 Mask 掉 fNIRS 部分 token（比如 40%）
- 用一个跨模态 Transformer，输入：
  - 可见的 EEG tokens
  - 未被 Mask 的 fNIRS tokens
- 预测被 Mask 的 fNIRS tokens 的分布：
  \[
  P_\theta(z^{\text{fNIRS}}_i \mid z^{\text{EEG}}_{t-\tau:t}, z^{\text{fNIRS}}_{\text{unmasked}})
  \]

**损失：**  
标准交叉熵（对 codebook 中的 token 做分类）：
\[
\mathcal{L}_{\text{MTP}} = -\sum_{i\in\mathcal{M}}\log P_\theta(z^{\text{fNIRS}}_i \mid \cdots)
\]

**好处：**

- 模型学到的是 token 级的**条件分布**，不是仅仅一个连续向量回归。
- 可以自然扩展到：
  - Mask EEG，用 fNIRS + EEG 上下文预测 EEG（对称任务）
  - Mask 双方，用双向上下文预测（类似联合建模）。

---

### 3.3 路线三：显式建模时间延迟和不确定性（突出神经生理特性）

在上面两条路线上，可以加进两个更“神经科学”的成分，使你的故事更有深度：

1. **显式建模 \(\tau\) 的不确定性**  
   - 不把 \(\tau\) 固定为某个值，而是让模型在一个范围内学习一个**延迟分布**：
     \[
     P(z^{\text{fNIRS}}_{t:t+\Delta} \mid z^{\text{EEG}}_{t-\tau':t}),\quad \tau'\in[\tau_{\min},\tau_{\max}]
     \]
   - 可以使用一个小网络预测最可能的 \(\tau'\)，或者在 InfoNCE 中对不同 \(\tau'\) 加权。

2. **用条件熵 / KL 散度做分析指标**  
   - 条件熵：
     \[
     H(z^{\text{fNIRS}} \mid z^{\text{EEG}}) = -\sum_k P(k\mid z^{\text{EEG}})\log P(k\mid z^{\text{EEG}})
     \]
   - 对健康组和病理组比较  
     \(D_{\text{KL}}(P_{\text{patient}} \Vert P_{\text{healthy}})\)，  
     可以讲“耦合模式改变”的故事。

---

## 四、这种方法支持你讲什么“故事”？

围绕这个条件分布，你可以讲出比一般“多模态分类/融合”更有内涵的几个故事。

### 4.1 故事一：**“神经–血流耦合的概率图谱”**

> 每一种典型 EEG token 模式（频段+空间），在多大概率上会诱发哪一种 fNIRS token（空间+时间）？

具体做法：

- 固定一个 EEG token（或 token 片段） \(z^{\text{EEG}}\)，画出：
  \[
  P(z^{\text{fNIRS}} \mid z^{\text{EEG}})
  \]
  在 fNIRS 通道 × 时间位置上的热力图。
- 你就得到某个 EEG 模式（比如运动想象时 C3 上的 \(\mu\) 抑制）对应的**血氧响应概率分布图**。

**可讲的解释：**

- 这不是“黑箱注意力权重”，而是“在这个电活动模式下，前额叶/运动皮层的血流响应以多少概率出现”，可以和传统 fNIRS/EEG 文献中 motor imagery、工作负荷等经典结果对照。

### 4.2 故事二：**“耦合强度和不确定性随任务/状态变化”**

- 对不同任务条件（休息 vs 工作负荷高），比较：
  - 条件熵 \(H(z^{\text{fNIRS}} \mid z^{\text{EEG}})\) 是否降低（更可预测）或升高（更混乱）  
  - 条件分布的主峰是否转移到不同的脑区
- 对临床场景（如癫痫、抑郁等），比较：
  - 患者 vs 健康人：
    \[
    D_{\text{KL}}\big(P_{\text{patient}}(z^{\text{fNIRS}} \mid z^{\text{EEG}})\,\Vert\,P_{\text{healthy}}(z^{\text{fNIRS}} \mid z^{\text{EEG}})\big)
    \]

这可以支撑的故事是：

> “不仅仅是 EEG 或 fNIRS 单独异常，而是两者之间的**耦合结构**在疾病状态发生了可量化的改变。”

### 4.3 故事三：**“EEG 作为快变量，fNIRS 作为慢变量的层级建模”**

- 把 EEG token 视作“瞬时神经状态”
- 把 fNIRS token 视作“一段时间内的整合代谢响应”
- 条件分布刻画了“从快变量到慢变量”的时间整合机制

你可以从信息论角度讲：

> “在这个任务下，EEG 到 fNIRS 的信息传输量是多少？不同脑区的神经–血流信息通量是否一致？”

---

## 五、这种方法的“内廓可解释性”有多强？

### 5.1 为什么它比一般“多模态对比学习”更可解释？

1. **有明确的物理语义**  
   - 条件关系不是抽象模态对模态，而是  
     “电活动模式 → 血流响应模式”  
   - 如果 codebook 设计得当（例如按照频段/通道/空间聚类），每个 token 都可以解释为“某脑区某频段的典型模式”“某脑区某种典型血流波形”。

2. **概率语言天然适合“解释”**  
   - “这个 EEG token 触发这个 fNIRS token 的概率是 0.7，而另一个只有 0.1”  
   - 比起“余弦相似度=0.7”更贴近日常和科学叙述。

3. **可视化直接对应到脑图**  
   - 把 \(\arg\max_k P(z^{\text{fNIRS}}_k\mid z^{\text{EEG}})\) 投射到 fNIRS 通道布局上，可以得到“最可能被激活的区域图”  
   - 再叠加条件熵，可以画出“哪种 EEG 模式诱发的血流反应最稳定/最不确定”。

### 5.2 限制与注意点（坦诚部分）

- 若强行使用**完全统一的 codebook**，可能牺牲各自模态的表达力，需要更现实的“共享 + 私有 codebook”设计，这一点你前面已经考虑过。
- token 本身的可解释性很大程度取决于：
  - 你的 VQ 训练目标（频谱重构还是任务监督）
  - 是否有对 token 作 post-hoc 的“标注”（例如可视化每个 code 对应的典型时频图）
- 条件分布虽然可视化友好，但要避免过度解读成“因果”，它更接近“带物理约束的条件相关”。

---

## 六、总结：用一句话串起来

- 把  
  \[
  P(z^{\text{fNIRS}}_{t:t+\Delta} \mid z^{\text{EEG}}_{t-\tau:t})
  \]  
  看作 word2vec 式的“跨模态上下文概率”，在形式上是完全可类比的，可以直接导出跨模态 Skip-gram、对比学习和 BERT 式 Masked Token Prediction 等一整套方法。

- 不同之处在于，这个条件分布不是纯粹的共现统计，而是带有**神经血管耦合、时间延迟和脑区拓扑**等物理内涵的“生理驱动条件分布”，因此在 token 设计合适时，它自然具有**内廓可解释性**。

- 你可以用它来：
  - 训练一个“EEG→fNIRS 条件概率模型”
  - 画出神经–血流耦合的概率图谱
  - 分析不同任务/疾病下耦合结构的变化
  - 讲一个从“快电活动”到“慢血流响应”的层级信息流故事。

这条路在 EEG–fNIRS 领域目前尚属空白，但在理论和方法上都已经具备可行性，是一个既有工程可操作性、又方便讲清“机制性故事”的研究方向。