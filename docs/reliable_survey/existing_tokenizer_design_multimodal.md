下面是基于已收集文献，对“生理信号 foundation model 领域中，将**多个模态都转化为离散 token**的方法”的系统整理与总结。重点只涵盖目前已经检索到、**明确对多模态生理信号做离散 token 化**的典型做法。

---

# 1. 问题定义与筛选范围

你关心的是：  
在生理信号基础模型（foundation model）中，**有哪些工作把多个模态（EEG/ECG/PPG/EMG/IMU…）统一转换成离散 token**，再用 Transformer 等模型建模？

这里我们只纳入：

- 至少同时处理 **两个以上生理模态**；  
- 对所有这些模态都有**统一的“离散 token 化”步骤**（不局限于单一模态或只对其中一模态离散化）；  
- 针对 foundation model / 大规模预训练场景。

符合上述条件、并且信息较完整的代表性方法主要集中在三类：

1. **基于时–通道 patch 的序列 token 化**（LSM 系列）  
2. **基于时–频图 + patch 的“图像式 token 化”**（NormWear）  
3. **基于向量量化 codebook 的多模态 tokenizer（共享+私有码本）**（PhysioOmni）

其它如 NeuroRVQ、SleepFM 等，要么是单模态为主，要么输出连续嵌入而非离散 codebook token，因此这里只会作为对比背景简要提及，不算“所有模态都转为离散 token”的范畴。

---

# 2. 按方法类型整理：如何把多模态生理信号变成离散 token

## 2.1 时–通道 Patch Token 化：LSM / LSM‑2

**核心思想**：  
把多模态可穿戴信号（HR、HRV、ACC、EDA、皮温等）视为一个二维矩阵（时间 × 传感器维度），再切分成规则 patch，每个 patch 视为一个 token，输入 Transformer 进行预训练与下游任务。[1]

### 2.1.1 Token 化流程

- 输入：多模态时间序列，例如：
  - HR / HRV（1D）
  - 三轴加速度（3D）
  - 皮温、EDA 等（若干通道）
- 步骤：
  1. **对齐/重采样**：所有模态按统一时间步对齐，堆叠成 `T × C`（时间 × 通道）。
  2. **Patch 切分**：在时间轴上（有时也在通道轴上）用固定窗口长度划分：
     - 比如以 1 min / 30 s / 10 s 为单位形成 patch，每个 patch 覆盖所有可用通道；
     - 每个 patch 是一个 `L × C` 的小矩阵（L 为该时间片的采样点数）。
  3. **Patch 嵌入**：用线性层/小 MLP 将 `L×C` 展平成向量并投影到 D 维：
     - 得到每个 patch 的 token 向量 `e_i ∈ R^D`。
  4. **Token 序列建模**：序列 `{e_1, e_2, …, e_N}` 输入 Transformer / ViT‑like 架构。

- 所有模态都经过同样的切分和投影流程，**天然处于同一 token 空间**，只是在 patch 中包含的通道组合不同。

### 2.1.2 自监督训练与多模态利用

- **Masked Signal Modeling**：在时–通道 patch 上随机 mask 一部分 patch 或通道，让模型预测被 mask 的 patch。
- **多模态融合**：
  - 不再为每个模态单独建模型，而是在同一 Transformer 里让 token 自行学到跨模态依赖关系；
  - LSM‑2 引入 Adaptive & Inherited Masking（AIM），显式建模“继承的缺失 + 人为 mask”的缺失机制，对真实可穿戴数据里的缺失通道很鲁棒。

### 2.1.3 评价

- **优点**：
  - 非常直接，易于扩展到 N 个模态；
  - token 完全统一（patch token 即多模态 token），模型结构简单；
  - 对大规模预训练（40M 小时）极其友好。
- **局限**：
  - Patch 粒度较粗，难以做细粒度 codebook 解释；
  - 没有显式区分“模态共享 vs 模态特异”信息。

---

## 2.2 时–频图 + Patch 的图像式 Token 化：NormWear

NormWear 专门针对**多变量可穿戴生理信号**提出了一套统一 tokenization 方案，既支持 ECG、PPG、EEG、GSR、IMU 等异构模态，又对通道数量变化和模态组合变化保持鲁棒。[2][3]

### 2.2.1 单通道到“RGB 图像”

对每一个**通道/模态**，NormWear执行以下步骤：

1. **时间导数增强**：
   - 计算原始信号的一阶导数、一阶导数的二阶导数；
   - 三个序列：`x(t), x'(t), x''(t)`。
2. **连续小波变换 (CWT)**：
   - 对三个序列分别做 CWT，得到三张 time–frequency scalogram；
   - 使用 Mexican Hat 小波，尺度范围 1–64。
3. **RGB-like 合成**：
   - 把三张 scalogram 堆叠成 3 通道“图像”，相当于一个 `H×W×3` 的输入。
   - 对任意模态，这一过程完全相同，因此 **是模态无关的统一表示**。

### 2.2.2 Patch Token 化与通道融合

1. **Patch 切分与嵌入**：
   - 使用 Conv2D 作为 patchify 层，例如 kernel=(9,5), stride=(9,5)，把 scalogram 切成小 patch；
   - 每个 patch 通过该 Conv2D 投影到 768 维，得到一串 token：`{t_1^c, t_2^c,…}`，c 表示某个通道。
2. **每个通道一串 token**：
   - 每个通道独立生成一条 token 序列；
   - 即便不同样本拥有不同数量的通道，这一过程也适用。
3. **Channel-aware Fusion**：
   - 为每个通道序列加一个 [CLS]_c token；
   - 经过若干 Transformer 层后，在“每隔一层”用 **[CLS]-Attention Fusion** 融合各通道的 [CLS]_c：
     - 将各通道 [CLS]_c 堆叠成序列；
     - 对这些 [CLS] 做 self-attention，得到“跨通道融合后”的 [CLS]'_c；
     - 再把更新后的 [CLS]'_c 放回各自通道，继续后续层的时序建模。

### 2.2.3 评价

- **满足你关心的要求**：
  - 所有模态（ECG / PPG / EEG / GSR / IMU…）都映射到同一种形式的**时频“图像 patch token”**；
  - token 是离散的 patch embedding 序列，统一输入同一个 Transformer 主干。
- **优点**：
  - 频域+时域信息兼顾；
  - 通道数可变、模态组合可变，仍可统一处理；
  - 通过 [CLS] Fusion 显式建模跨模态关系。
- **局限**：
  - 计算成本高于纯时域 patch；
  - token 粒度仍然是 patch 级别，不像 codebook 那样可复用为“生理词表”。

---

## 2.3 向量量化 Codebook Tokenizer：PhysioOmni（Decoupled Multimodal Tokenizer）

PhysioOmni 的核心贡献之一，就是设计了一个**解耦式多模态 tokenizer**，用**共享 + 私有 codebook**把 EEG、ECG、EOG、EMG 等多模态统一编码成离散 token，并显式分离“模态不变信息”和“模态特定信息”[4]。

### 2.3.1 编码器与双路表示

对于每个模态 j（EEG/ECG/EOG/EMG）：

- 使用一个模态专属编码器 `E_j` 输出两类嵌入：
  - 私有嵌入：`z_j^p`（模态特定）
  - 共享嵌入：`z_j^s`（模态不变）

### 2.3.2 共享 + 私有码本设计

- **共享码本**：`V_s ∈ R^{K×D}`  
- **私有码本**：每个模态一个，如
  - `V_e, V_c, V_o, V_m ∈ R^{K×D}`  
- 典型超参：`K = 8192`, `D = 64`。

**量化规则**：

- **私有量化**（每模态单独）：
  \[
  \hat{z}^p_{j,i} = V_j[\arg\min_k||\ell_2(z^p_{j,i}) - \ell_2(v_{jk})||^2]
  \]
- **共享量化**（所有模态共享同一 codebook）：
  - 先通过 **Temporal Alignment(TA)** 将各模态的 `z_j^s` 对齐到 EEG 的时间尺度：
    \[
    TA(z^s_{j,i}) = CrossAttention(q, W^K_j z^s_{j,i}, W^V_j z^s_{j,i})
    \]
  - 再对齐后做 nearest-neighbor 量化到 `V_s`：
    \[
    \hat{z}^s_{j,i} = V_s[\arg\min_k||\ell_2(TA(z^s_{j,i})) - \ell_2(v_{sk})||^2]
    \]

**结果**：  
每个时间 patch 上，每个模态都会得到：

- 一个来自**私有码本**的离散 token（私有 code index）；  
- 一个来自**共享码本**的离散 token（共享 code index）。

这两者拼接后就是该模态在该 patch 的完整离散 token 表示。

### 2.3.3 重构与跨模态对齐

- **每模态都有一个 decoder**：`D_e, D_c, D_o, D_m`
  - 输入：`[ \hat{z}^p_{j,i} ∥ \hat{z}^s_{j,i} ]`
  - 输出：重构信号 `o_{j,i}`：
    - EEG/EMG：重构 Fourier amplitude；
    - ECG/EOG：重构原始时域波形。
- **Cross-Modal Alignment (CMA)**：
  - 通过 EEG 的共享 token `\hat{z}^s_{e,i}` 作为 anchor，用 cross-attention 预测其他模态的共享部分，并辅以交叉重构损失：
    \[
    o^{cross}_{c,i} = D_c(\hat{z}^p_{c,i} ∥ CMA(\hat{z}^s_{e,i}))
    \]
  - 鼓励 `V_s` 捕捉跨模态共性模式。

### 2.3.4 训练目标与阶段

1. **Stage 1：Tokenizer 训练**：
   - 损失函数包括：
     - `L_CB`：各模态重构损失 + VQ 损失 + commitment 损失；
     - `L_CR`：跨模态重构损失；
     - `L_D`：私有与共享嵌入的“去相关”损失（cosine 相似度惩罚）；
   - 总损失：`L_T = L_CB + α1 L_CR + α2 L_D`。
2. **Stage 2：Masked Signal Modeling**：
   - 在 patch 层面对每个模态随机 mask；
   - 用 encoder 预测对应 patch 的共享+私有 code index；
   - 目标是 code 级别的 masked token prediction（类似离散版 MAE/BERT）。
3. **Stage 3：Resilient Fine-tuning**（对缺失模态鲁棒）：
   - 通过 Homogeneous Representation Mapping + prototype alignment，将各模态的高层表示对齐到统一 prototype 空间，支持任意模态子集的推理。

### 2.3.5 评价

- **非常符合你的筛选条件**：
  - EEG/ECG/EOG/EMG 四个模态全部通过 shared/private codebook 转为**离散 code index 序列**；
  - codebook 层面是明确离散 token 空间，可直接视为“生理信号的离散词表”。
- **优势**：
  - 显式解耦 + codebook，可解释性和重用性好；
  - 自然支持 masked token 预训练和缺失模态推断。
- **局限**：
  - 训练复杂度、实现门槛较高；
  - 目前实验主要集中在 EEG/ECG/EOG/EMG 四类信号。

---

# 3. 其它相关工作与“非完全离散化”方法

为完整起见，简单说明几篇相关但**不满足“所有模态都变成离散 token”**的代表性工作——理解它们有助于你界定“离散 token”的边界。

## 3.1 NeuroRVQ：多尺度 RVQ Tokenizer（单模态为主）

- **对象**：主要是 EEG，大体上也展示了对 ECG、EMG 的可迁移性。[5]
- **方法**：
  - 多尺度时域卷积 + Transformer 生成 patch 表示；
  - 每个尺度使用多级 RVQ codebook（例如 S=4, N=4，共 16 个 code per patch）；
  - 目标是重构 Fourier amplitude+phase，损失中加入 unit-circle phase loss。
- **多模态性**：
  - EEG、ECG、EMG 是分别用类似结构训练的 tokenizer；
  - 论文**没有**展示一个单一 tokenizer 同时对多模态进行统一 token 化与跨模态训练。
- **结论**：方法非常适合作为**单模态 codebook tokenizer**的借鉴，但尚未形成完整的“多模态统一 codebook”。

## 3.2 SleepFM：5 秒窗口连续 token

- **对象**：多导 PSG（EEG/BAS、ECG、EMG、呼吸等）。[6]
- **token 定义**：
  - 所有信号统一重采样到 128 Hz；
  - 以 5 秒为一个 token（640 个采样点），用 1D CNN + 池化 + Linear 映射为 128 维向量；
  - 模态内部用 attention-based channel pooling 聚合多通道，再用 Transformer 做时序建模；
  - 使用 leave-one-out 对比学习（LOO-CL）对齐模态。
- **关键区别**：
  - 这些 token 是**连续实值嵌入**，而不是 codebook 索引；
  - 不属于“将所有模态都映射到统一离散 code 空间”的范畴。

## 3.3 QualityFM 等频谱重构模型

- QualityFM（ECG+PPG）使用 Transformer 对连续特征做编码，重构频谱幅度与相位，没有明确离散化步骤。[7]

---

# 4. 综合对比与实践建议

下表总结上面三类“真正把多模态都变成离散 token”的代表路径：

| 方法/工作 | 主要模态 | token 类型 | 是否统一作用于所有模态 | 是否有共享 token 空间 | 典型用途 |
|----------|----------|------------|------------------------|-----------------------|----------|
| LSM / LSM‑2 | HR, HRV, ACC, EDA, 皮温 等可穿戴信号 | 时–通道 patch embedding（实值，但 patch 视作离散 token 单元） | 是：所有模态统一切 patch、统一投影 | 是：同一 Transformer embedding 空间 | 大规模可穿戴 foundation model，自监督 MAE |
| NormWear | ECG, PPG, EEG, GSR, IMU 等 | CWT 时频图 patch token（768 维） | 是：每个通道/模态同样的 CWT→RGB→patch 流程 | 是：所有 patch token 进入同一 backbone，CLS 融合 | 多模态 wearable foundation model，异构传感器统一处理 |
| PhysioOmni | EEG, ECG, EOG, EMG | codebook index（共享+私有 codebook） | 是：每个模态都输出 shared+private code index token | 是：共享码本 V_s 提供真正的“跨模态离散词表” | 多模态生理 foundation model，支持缺失模态、下游迁移 |

> 注：严格意义上的“离散 token”，只有 PhysioOmni（和单模态的 NeuroRVQ）使用了 codebook / VQ；LSM、NormWear 的 patch token 在 Transformer 视角上是“离散位置的 embedding”，在序列层面等价于 token，但不是 codebook 索引。

---

# 5. 如果你要“照着做”，该怎么选路线？

结合以上整理，可以给出一些可操作的建议：

1. **需要真正的“离散 codebook + 生理词表”**  
   - **优先参考 PhysioOmni**：
     - 采用 shared + private codebook 的解耦结构；
     - 必须实现 TA（Temporal Alignment）和 CMA（Cross-Modal Alignment），解决 EEG/ECG/EMG 时间尺度不同问题；
     - 适合希望在 code 级别做生成、压缩、离线索引的场景。

2. **需要“统一 token 序列 + 大规模预训练”，对 codebook 没强需求**  
   - 可采 **LSM/LSM‑2** 风格的 **时–通道 patch tokenization**：
     - 统一时间步长、拼成 `T × C`；
     - 选择合适的 patch 长度（如 1–10 s），线性/Conv 投影为 token；
     - 自监督任务可用 MAE/Forecasting/Interpolation 等。

3. **模态类型极多且采样率差异大，且你更关注频域模式**  
   - 可沿着 **NormWear** 路线：
     - 对所有通道统一做 CWT（或 STFT），拼成“RGB”时频图；
     - 用 ViT-style patch 作为 token；
     - 再在 [CLS] 级别做 channel-aware fusion。

4. **目标是单模态（例如 EEG）的大模型，再逐步扩展到多模态**  
   - 可先实现类似 **NeuroRVQ** 的多尺度 RVQ tokenizer，之后借鉴 PhysioOmni 的 shared codebook 思路，把其它模态逐步纳入统一 code 空间。

---

# 6. 小结：当前“多模态生理信号离散 token 化”的版图

- **真正实现“多模态全部离散化到统一 code 空间”的**，目前最系统的是：
  - **PhysioOmni**：shared+private codebook，多模态 EEG/ECG/EOG/EMG，分三阶段训练（tokenizer、masked signal modeling、resilient fine-tuning）。[4]
- **在序列层面将多模态统一为“离散 patch token 序列”的**：
  - **LSM/LSM‑2**：时–通道 patch 化，把所有可穿戴模态统一为 patch token。[1]
  - **NormWear**：CWT 时频图 + 图像式 patch token，结合 channel-aware CLS 融合，各模态在 token 空间统一。[2][3]
- 其它如 NeuroRVQ、SleepFM、QualityFM 等，为多模态离散 token 提供了重要构件和灵感，但要么尚未做到多模态统一 codebook，要么 token 是连续嵌入。

如果你后续希望针对某一个方法深挖（例如：照着 PhysioOmni 自己实现一个多模态 tokenizer，或在 LSM 的 patch token 之上叠加 codebook），我可以帮你按工程步骤分解具体实现路线。  

---

### References

[1] Foundation Models for Physiological Signals. <https://openreview.net/pdf?id=u3nat9mOIo>  
[2] Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals (NormWear). <https://arxiv.org/html/2412.09758v2>  
[3] Foundation Model Representations – Multimodal physiological time-series (NormWear 部分). <https://www.emergentmind.com/topics/foundation-model-representations>  
[4] Towards Robust Multimodal Physiological Foundation Model (PhysioOmni). <https://arxiv.org/pdf/2504.19596>  
[5] NeuroRVQ: Multi-Scale EEG Tokenization for Generative Brainwave Models. <https://arxiv.org/pdf/2510.13068>  
[6] A multimodal sleep foundation model for disease prediction (SleepFM). <https://www.nature.com/articles/s41591-025-04133-4>  
[7] QualityFM: a Multimodal Physiological Signal Foundation Model. <https://arxiv.org/pdf/2509.06516>