下面的总结**只基于我们已经获取到的文献信息**整理，不再引入新来源。重点放在：  
1）和“EEG+fNIRS + 预训练/表征学习”强相关；  
2）来自 IEEE Transactions 或高水平期刊/会议；  
3）可直接对标您要做的“多模态生理信号预训练模型”。

---

# 一、总体建议：该选哪些工作来对标？

结合您的目标（EEG+fNIRS 多模态预训练 / 表征学习），建议把对标对象分三层：

1. **直接同类：多模态 EEG‑fNIRS 预训练 / 表征学习**
   - **EFRM: A Multimodal EEG‑fNIRS Representation‑learning Model for few‑shot brain‑signal classification**（Computers in Biology and Medicine, 2025）[1]  
   → 您工作的**首要对标对象**（真正意义上的 EEG‑fNIRS 预训练 / foundation‑like 模型）。

2. **紧密相关：多模态 EEG‑fNIRS 深度融合 + 高水平 IEEE Trans / 顶刊**
   - **Integrated EEG–fNIRS for Characterizing Cortical Responses and Neurovascular Coupling in Automated and Discrete Gait Tasks**（IEEE T‑NSRE, 2025）[2]  
   - **Fusion Analysis of EEG‑fNIRS Multimodal Brain Signals**（IEEE T‑IM, 2025）[3]  

   → 用于对标**实验设计、数据采集规范、多模态特征构造与融合策略**。

3. **方法论延展：多模态自监督/对比学习 & 跨模态生成/融合**
   - **Multimodal Physiological Signals Representation Learning via Multiscale Contrasting for Depression Recognition (MRLMC)**（ACM 顶会 ICMI 2024 / arXiv 2406.16968）[4]  
   - **SCDM: Unified Representation Learning for EEG‑to‑fNIRS Cross‑Modal Generation in MI‑BCIs**（arXiv 2407.04736）[5]  
   - **TSMMF / Bidirectional Cross‑Modal Transformer for EEG‑fNIRS Multimodal Affective BCI**（Expert Systems with Applications, 2025）[6]  

   → 用来对标**自监督目标设计、对比学习结构、跨模态生成与 Transformer‑式融合**。

下面按论文逐篇提炼“**核心创新** + **实验设计** + **可直接借鉴的点**”。

---

## 二、核心对标：EFRM – 多模态 EEG‑fNIRS 预训练模型 [1]

### 1. 论文定位与核心思想

- **目标痛点**：  
  多数 EEG 预训练/迁移学习工作只覆盖 EEG，对 fNIRS 或 EEG‑fNIRS 共享表征几乎没有系统探索。
- **核心思想**：  
  构建一个**多模态表示学习模型**，在**大规模无标注 EEG+fNIRS 数据**上预训练，学习：
  - 每种模态的**模态特定特征**；
  - EEG 与 fNIRS 之间的**共享域特征**（shared domain representations）。
- **使用方式**：  
  先进行**自监督预训练**，然后在各类少样本下游任务（EEG‑only / fNIRS‑only / paired EEG‑fNIRS）上进行**few‑shot 迁移**。

这与您要做的“EEG+fNIRS 生理信号预训练模型”高度对齐，可以认为是同类问题的**当前代表作**。

### 2. 核心创新点

1. **双阶段框架（pre‑train + transfer）**
   - **预训练阶段**：
     - 使用九个公开数据集，约 **1250 小时、918 名被试**的脑电/近红外数据；
     - 利用自监督目标学习：
       - EEG‑only 表示；
       - fNIRS‑only 表示；
       - EEG‑fNIRS 共享表征。
   - **迁移阶段**：
     - 在少量标注数据（few‑shot）上微调，适配具体 BCI 任务。

2. **同时支持三种输入形态**
   - **EEG‑only**、**fNIRS‑only**、**paired EEG‑fNIRS** 三类数据都能输入同一个预训练模型；
   - 解决以往多模态方法“训练时必须有配对数据”的刚性要求，大幅提高真实应用中的使用率。

3. **共享域表示（跨模态语义空间）**
   - 通过自监督学习显式构建 EEG 与 fNIRS 的**共同潜在空间**；
   - 实证表明：**共享域信息越多，下游性能越好**，尤其是对 fNIRS 分类提升显著。

4. **与单模态预训练/监督 SOTA 的系统对比**
   - 与多种**全监督模型**和**其他预训练模型**比较：
     - 在极少标注（few‑shot）场景下仍能达到与全监督接近或更优的效果；
     - 特别在 fNIRS 分类上，**显著超过**已有预训练方法。

### 3. 实验设计要点（可直接借鉴）

- **数据规模与构成**  
  - 9 个公开 EEG/fNIRS 数据集；  
  - 合计约 **1250 hours**，**918 subjects**；  
  - EEG 重采样到 128 Hz（统一预处理管线）。  
  → 对标您的工作：建议也构建“跨数据集的大规模预训练池”，并在论文中清晰罗列。

- **任务设置**
  - 下游任务为**少样本分类（few‑shot）**，分别在：
    - EEG‑only；
    - fNIRS‑only；
    - paired EEG‑fNIRS 三种配置下评估；
  - 与多种 SOTA / 预训练模型做系统对标，验证“预训练 + 共享域”的价值。  
  → 对标您的工作：  
  - 不仅做“paired EEG‑fNIRS”，也最好设计 EEG‑only / fNIRS‑only 的 few‑shot 场景，突出“预训练 + 多模态兼容性”。

- **评价指标与对比**
  - 与**全监督方法**对比：展示在使用更少标注时性能不降甚至更优；
  - 与**已有预训练方法**对比：突出共享域学习带来的增益；  
  - 强调在 **fNIRS 分类** 上的显著优势。

### 4. 可直接启发您设计的点

1. **框架结构**：  
   - 保持“**预训练（自监督） + 下游微调（few‑shot）**”两阶段形式；
   - 在预训练损失设计中，显式区分：
     - 模态内建模（EEG‑only / fNIRS‑only）；
     - 模态间建模（EEG↔fNIRS 共享空间）。

2. **研发叙事**：  
   - 以“**缓解标注缺乏** + **跨模态泛化**”为主线；
   - 通过把 EEG‑only 与 fNIRS‑only 也纳入预训练与下游评估，强调“一个预训练模型，适配多种实际采集条件”。

---

## 三、IEEE Trans 代表性工作：多模态融合与实验设计对标

### 1. Integrated EEG–fNIRS for Gait Tasks (T‑NSRE 2025) [2]

**核心创新**

- **TRCA（Task‑Related Component Analysis）驱动的多模态表示**：
  - 分别对 EEG 与 fNIRS 做 TRCA，得到“任务相关分量”和空间滤波器；
  - TRCA 后的特征在类别内更一致、类别间更可区分。

- **神经血管耦合（NVC）显式定量**：
  - 将 α/β 波段 ERSP 与 HbO 通过 canonical HRF 卷积，统一到 hemodynamic 时标；
  - 通过带时延（±6 s）交叉相关度量 EEG‑fNIRS NVC 强度。

- **结合 TRCA 的多模态融合**：
  - 比较 EEG‑Avg、EEG‑TRCA、fNIRS‑Avg、fNIRS‑TRCA 及其融合；
  - TRCA‑加权 EEG+fNIRS 融合在 3 类步态任务上 LOSO 准确率达 **74.51 %**，显著优于单模态。

**实验设计亮点**

- **被试与任务**
  - 18 名健康被试，三类任务：
    - Continuous Walking (CW)；
    - Isolated Gait Phase Task (IGPT)；
    - Single‑Limb Stance (SS)。
  - 每任务 6 个 trial，每 trial 含 baseline、提示、任务、休息四个阶段。

- **模态与采集**
  - EEG：10 通道（运动和感觉‑运动相关区），1000 Hz；  
  - fNIRS：12 源+12 探测器，38 通道覆盖 PMC/M1/S1，10.2 Hz。

- **预处理与特征**
  - EEG：EEGLAB，0.5–40 Hz，ICA+ICLabel 去伪迹，ERSP（θ, α, β）；  
  - fNIRS：HbO/HbR 提取，0.05–0.2 Hz，TDDR 去运动伪迹；  
  - 通过 HRF 卷积+重采样实现 EEG‑fNIRS 时标对齐。

- **评估**
  - Leave‑One‑Subject‑Out (LOSO) 交叉验证；
  - 使用 XGBoost 分类三任务，统计显著性（ANOVA + Bonferroni）。

**对标价值**

- **实验设计**：  
  - 标准而复杂的临床/步态范式：多任务、多 trial、精确定义基线与任务窗口；
  - 可作为您设计**多任务、多范式 EEG‑fNIRS 数据集**的模板（哪怕您更关注 MI/情绪）。

- **特征层面**：  
  - NVC 分析 + HRF 卷积 + 时延相关，是跨模态对齐的一个可借鉴“先验层对齐”方案；
  - 即便您做 end‑to‑end 预训练，也可以在**分析实验结果时使用类似的 NVC 指标**，增强神经科学说服力。

---

### 2. Fusion Analysis of EEG‑fNIRS Multimodal Brain Signals (T‑IM 2025) [3]

**核心创新**

- **卷积+双注意力的端到端多模态融合**
  - 各模态独立：时空卷积提取局部特征；
  - 双注意模块：
    - 通道或空间注意：强调重要通道/区域；
    - 时间注意：捕捉关键时间片；
  - 最终融合成统一特征后分类。

- **针对非运动任务的高精度轻量级模型**
  - 任务：MI、Mental Arithmetic、Word Generation 均为想象/认知任务；
  - 兼顾计算开销与准确率，适合实时 BCI。

**实验设计亮点**

- **数据**：两个公开 EEG‑fNIRS BCI 数据集；
- **任务**：
  - Motor Imagery (MI)；
  - Mental Arithmetic；
  - Word Generation（词生成）。
- **结果**：
  - MI: 92.2 %；
  - Mental arithmetic: 98.6 %；
  - WG: 95.2 %；
  - 所有任务结果均超过现有方法。

**对标价值**

- 如果您的预训练模型需要一个**强监督基线**来对标，“CNN+Attention 融合网络”是现实可行且 SOTA 级别的选项；
- 您可以在论文中设置：
  - “预训练 + 线性探测 / 轻微微调” vs  
  - “端到端 CNN‑Attention 监督训练”，  
  以突出预训练在低标注场景下带来的优势。

---

## 四、多模态自监督/对比学习与跨模态生成：方法层面对标

### 1. MRLMC – 多尺度对比表征学习 (EEG+fNIRS) [4]

**核心创新**

- **Siamese 架构 + 多尺度时空卷积**
  - EEG 与 fNIRS 经时域增强变为“不同但相关”的输入；
  - 两支共享权重的 CNN 进行多尺度时空卷积；
  - 在 Siamese 空间中，通过对比损失学习模态间共同结构。

- **语义一致性对比模块**
  - 在任务标签层级（如抑郁程度），最大化 EEG 与 fNIRS 语义表示相似度；
  - 从“同一刺激/同一被试条件”的多模态信号里抽取共享情绪/病理信息。

**实验设计**

- **任务**：抑郁识别；  
- **数据**：公开 + 自采多模态生理信号（包含 EEG 和 fNIRS）；  
- **结果**：在多个数据集上优于现有方法，并证明 learned representation 可迁移到其他多模态时间序列任务。

**对标价值**

- 在您设计**预训练目标（loss）**时，可以参考：
  - 一部分是“模态内”对比（类似 SimCLR/BYOL）；
  - 一部分是“模态间语义一致性”对比（align EEG 与 fNIRS）。

---

### 2. SCDM – EEG→fNIRS 跨模态生成 (MI‑BCI) [5]

**核心创新**

- **基于扩散模型的 EEG→fNIRS 生成**
  - 提出 SCDM（Spatio‑temporal Controlled Diffusion Model），作为 EEG→fNIRS 跨模态生成框架；
  - 核心模块：
    - SCG（Spatial Cross‑Modal Generation）：基于 EEG 的空间拓扑，预测 fNIRS 分布；
    - MTR（Multi‑scale Temporal Representation）：学习多尺度时间特征。

- **训练目标**
  - 让合成的 fNIRS 在统计与时空特征上逼近真实 fNIRS；
  - 使“EEG + 生成 fNIRS”的联合分类性能**不低于甚至优于**“EEG + 真实 fNIRS”。

**实验结果**

- 合成 fNIRS 与真实 fNIRS 高度相似（统计指标）；  
- EEG + synthetic fNIRS 的联合分类性能 ≥ EEG + real fNIRS；  
- 合成信号保留了与 EEG 的空间关系。

**对标价值**

- 如果您未来考虑“**预训练 + 生成式（如 diffusion）**”，SCDM 给出了一个“**EEG 作为条件、fNIRS 作为目标**”的范式；
- 在您的模型中可以：
  - 把 shared latent space 既用于下游分类，也用于跨模态生成，以增强表征的可解释性。

---

### 3. TSMMF / Bidirectional Cross‑Modal Transformer (ESWA 2025) [6]

**核心创新（基于项目代码与二手信息）**

- **BCMT（Bidirectional Cross‑Modal Transformer）**
  - 模式：EEG Stream + fNIRS Stream；
  - 第一阶段：各模态 CNN + Self‑Attention 提取模态内时空特征，缓解分布差异；
  - 第二阶段：BCMT 进行双向跨模态注意力，建模模态间时空对齐；
  - 第三阶段：modality‑specific 分支 + 融合分支，共同输出情感分类结果。

- **显式情绪解码与空间解释性**
  - 模型可学习各模态中与情绪相关的脑区权重；
  - 发现融合分支更倾向于关注跨模态互补的脑区，而非各自的“常规强响应区”。

**实验表现（在 REFED 数据集作为 baseline 时）**

- 以 EEG、fNIRS、EEG+fNIRS 分别为输入，在 Valence/Arousal 分类上提供 baseline 准确率和 F1；
- 虽然在 REFED 报告中 TSMMF 结果相对一般，但它作为“复杂 transformer 融合模型”的代表，仍然具有架构参考价值。

**对标价值**

- 如果您打算在预训练后使用 **Transformer‑式融合头**，TSMMF 提供了一个完整的 blueprint：
  - 先做模态内 CNN+Self‑Attention；
  - 再做跨模态 Transformer 融合；
  - 最后加上 modality‑specific 与 shared 分支。

---

## 五、如何基于这些工作构建您的对标体系？

结合上述工作，建议您的论文在**方法与实验两个层面**设置对标：

### 1. 方法层面对标

- **主线对标**：EFRM [1]  
  - 采用类似的“两阶段 + 三模态兼容 + 共享域表示”框架；
  - 但在：
    - 自监督任务设计（如 Masked Reconstruction + Contrastive）、
    - 模型架构（如是否引入 Transformer/图网络，或 fNIRS 专用 encoder）  
    上做出您的创新。

- **辅助对标**：
  - 与 MRLMC [4] 对比：  
    - 您的 shared domain 是否通过更丰富/更稳定的对比目标实现；
  - 与 SCDM [5] 对比：  
    - 您是否能在预训练表征的基础上进一步实现跨模态生成或补全；
  - 与 TSMMF [6] 对比：  
    - 您的融合结构是否在参数量、性能与可解释性上有更优折中。

### 2. 实验设计与数据层面对标

- **数据规模 & 多任务设置**
  - 向 EFRM [1] 看齐：尽可能整合**多数据集、大规模被试、多任务范式**，统一到同一预训练框架中；
  - 参考 T‑NSRE gait 研究 [2] 与 T‑IM 融合研究 [3] 的**任务多样性**设计（MI、认知任务、步态、情绪等）。

- **评估设置**
  - Few‑shot：对标 EFRM [1]，构建 1/5/10‑shot 等多个少样本场景；
  - Cross‑subject（LOSO）：对标 T‑NSRE [2] 和 T‑IM [3] 的跨被试评估；
  - Single‑modality vs multimodal：显式报告 EEG‑only，fNIRS‑only，EEG+fNIRS 三种情形，突出预训练的泛化。

- **分析指标**
  - 模型性能：Accuracy/F1/MAE；  
  - 神经科学解释：借鉴 T‑NSRE [2] 和 SCDM [5] 的 NVC 与时空特征对比；
  - 预训练收益：与
    - 从零训练的监督模型（如 Fusion Analysis [3] 的 CNN‑Attention）、
    - 单模态预训练模型（如纯 EEG foundation models）  
    做 ablation。

---

如果您愿意，我可以在下一步帮您“反向工程”一份基于这些对标工作的**实验方案草图**（包含预训练数据构建、损失设计、对比实验矩阵），以便直接落地到您的项目中。

---

### References

[1] EFRM: A Multimodal EEG-fNIRS Representation-learning Model for few-shot brain-signal classification. <https://pubmed.ncbi.nlm.nih.gov/41223650/>  

[2] Integrated EEG–fNIRS for Characterizing Cortical Responses and Neurovascular Coupling in Automated and Discrete Gait Tasks. <https://ieeexplore.ieee.org/document/11165470/>  

[3] Fusion Analysis of EEG-fNIRS Multimodal Brain Signals. <https://ieeexplore.ieee.org/document/10886982/>  

[4] Multimodal Physiological Signals Representation Learning via Multiscale Contrasting for Depression Recognition. <https://arxiv.org/abs/2406.16968>  

[5] SCDM: Unified Representation Learning for EEG-to-fNIRS Cross-Modal Generation in MI-BCIs. <https://arxiv.org/abs/2407.04736>  

[6] A bidirectional cross-modal transformer representation learning model for EEG-fNIRS multimodal affective BCI (TSMMF/ESWA 2025). <https://openreview.net/pdf/139fd58f83940d08487db57c58b448b92db0f8c0.pdf>