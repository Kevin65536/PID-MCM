# 多模态大模型中的Tokenization范式：从word2vec到跨模态离散化

## 核心判断：三个问题的条件性答案

多模态tokenization研究在2023–2026年间已形成相对清晰的格局，但三个核心问题的答案均具有条件性：

- **收益如何被声明**：取决于模态与任务类型——视觉领域倾向于消融实验与重建质量指标，生理信号领域则更依赖生理事件的语义对应性论证；专项基准（AudioCodecBench、SEAM）的出现标志着评估体系正在从分散走向系统，但这一进程截至2025年中仍不完整。
- **语义空间信条是否延续**：部分延续，但已发生实质性转移——社区从追求连续语义平滑性转向追求"语义等价性+LLM兼容性"的双重目标；离散VQ方案的主导地位意味着工程效率已与语义质量并列为设计约束。
- **技术演进轨迹**：各模态均呈现从固定粒度→自适应粒度、从单模态→跨模态统一、从重建导向→生成+理解双导向的共同趋势，但生理信号领域的技术成熟度显著落后于视觉与音频领域。

**最关键的单一洞察**：评估体系的滞后是整个领域最大的隐患——SEAM benchmark直到2025年8月才作为首个跨模态语义等价性系统评估框架出现[1]，这意味着此前大量工作的tokenization收益声明缺乏统一验证基础，研究者在解读已发表结论时需保持审慎。

---

## Section 1：Tokenization收益的声明方式——研究者如何"证明"这一步有价值？

**收益声明并非统一范式，而是一套因模态、任务和研究传统而异的论证策略组合。** 理解这些策略的逻辑链，是批判性阅读相关论文的前提。

### 1.1 消融实验：最直接但最难控制的证明方式

消融实验是视觉和音频领域最常见的收益声明手段，其逻辑链通常为：对比有/无特定tokenizer设计时的下游任务指标，以增量差异归因于tokenization步骤。Emu3（arXiv 2409.18869）的论证尤为激进——该工作声称仅凭next-token prediction在统一离散token序列上即可达到多模态SOTA，其逻辑是：如果一个统一的tokenization框架能够让单一next-token prediction目标覆盖图像、视频、文本，则tokenization步骤的价值体现在它使得这种架构简化成为可能[2]。这种"架构简化即收益"的论证方式将tokenizer的价值从局部特征质量提升到了系统设计层面。

TexTok（CVPR 2025）的语言引导图像tokenization则采用更传统的消融路径：通过对比不同token数量下的重建质量（FID指标），以及有/无语言引导时的视觉细节保留程度，来论证语言引导机制对tokenizer的贡献[3]。

### 1.2 压缩率与重建质量：代理指标的有效性存疑

LongCat-Next（arXiv 2603.27538）和Wave-Particle CDD-VT均以压缩效率和重建保真度作为tokenizer质量的代理指标[4][5]。这类指标的吸引力在于可量化、可比较，但其与下游任务性能之间的关联并未被严格验证——这是一个系统性的方法论缺陷。高重建质量的tokenizer不一定在下游理解任务上表现更好，因为理解任务需要的是语义抽象而非像素级保真。离散tokenization综述（arXiv 2507.22920）明确指出，VQ方法的核心优势在于计算效率和LLM兼容性，而非重建质量本身[6]——这一表述隐含了对"重建质量即tokenizer质量"这一代理假设的质疑。

### 1.3 跨模态对齐能力：离散化的系统级收益

跨模态对齐是离散tokenization最具说服力的收益声明维度。通过将连续模态（图像、音频、视频）转化为统一离散token空间，不同模态的表示在形式上变得可比较，从而降低了跨模态对齐的难度[6]。这一论证的机制是：连续嵌入空间的模态间分布差异（distribution shift）是跨模态对齐的主要障碍，而离散化通过共享码本（codebook）构建了一个"公共语言"，使得模态间的对齐问题部分转化为码本学习问题。

AToken声称是首个在图像、视频、3D资产上同时实现高保真重建和语义理解的统一视觉tokenizer[7]，其收益声明同时覆盖了重建质量和跨模态统一两个维度，代表了视觉领域收益声明的"全覆盖"策略。

### 1.4 生理信号领域：语义对应性优先于压缩效率

生理信号领域的收益声明逻辑与视觉领域存在根本性差异。HeartLang（ICML 2025）通过检测QRS复合波将ECG录音切分为生理事件并映射为离散token[8]，其收益声明的核心不是压缩率或重建质量，而是**生理事件与token的一一对应性**——这种对应性使得LLM能够以"理解文本"的方式"理解"心电信号，从而将NLP领域的预训练知识迁移到心脏病学诊断。这与视觉领域的"像素→token"逻辑不同，更接近于"生理事件→词汇"的类比。

多模态生理信号基础模型（arXiv 2504.19596）则从另一角度声明收益：统一架构对EEG/ECG/EOG/EMG四种信号的泛化能力，即tokenization使得原本需要四套专用架构的问题可以用单一框架处理[9]。这是"架构经济性"维度的收益声明，与Emu3的逻辑类似，但应用于医疗领域。

### 1.5 基准测试：评估体系的迟来规范化

AudioCodecBench（OpenReview JeIDPXc9XG）建立了针对神经音频编解码器作为tokenizer的系统评估框架[10]，标志着音频领域开始脱离"每篇论文自定义评估指标"的混乱状态。更重要的是SEAM benchmark（arXiv 2508.18179，2025年8月），作为首个跨模态语义等价性系统评估基准[1]，其出现本身就是一个信号：在此之前，多模态tokenization领域长期缺乏统一的质量评估标准，各工作的收益声明在很大程度上是自说自话。

**表1：代表性工作的tokenization收益声明方式对比**

| 工作 | 模态 | Tokenizer设计 | 收益声明方式 | 主要论点 |
|------|------|--------------|------------|---------|
| Emu3 [2] | 图像/视频/文本 | 统一离散VQ序列 | 消融+基准 | next-token prediction统一框架即可达SOTA |
| TexTok [3] | 图像 | 语言引导VQ | 消融（FID/重建质量） | 语言引导提升视觉细节编码 |
| SeTok [11] | 图像 | 动态语义聚类 | 消融（语义等价性） | 语义完整token优于固定patch |
| AToken [7] | 图像/视频/3D | 统一视觉VQ | 重建质量+语义理解双指标 | 首个统一重建与理解的tokenizer |
| HeartLang [8] | ECG | QRS事件级分割 | 生理事件对应性+诊断准确率 | 生理事件→token直接映射 |
| 生理信号基础模型 [9] | EEG/ECG/EOG/EMG | 统一架构tokenizer | 泛化能力（多信号统一） | 架构经济性+跨信号迁移 |
| FocalCodec [12] | 语音 | 超低比特率编解码 | 压缩率+LLM集成效果 | 超低比特率下保持语义质量 |
| AudioCodecBench [10] | 音频 | 基准评估框架 | 系统化基准测试 | 建立统一评估标准 |
| MTVCraft [13] | 4D动作 | 4D运动捕捉直接tokenize | 与2D渲染对比消融 | 原始信号优于中间表示 |

生理信号与视觉领域在收益声明逻辑上的核心差异在于：视觉领域的"好token"通常以重建质量或下游视觉任务精度衡量，而生理信号领域的"好token"首先需要回答"这个token对应什么生理事件"——语义可解释性是前置条件，而非可选属性。

---

## Section 2：语义空间的信条是否延续？——从word2vec到离散化时代的哲学传承与分歧

**word2vec的核心遗产是一套认识论：好的表示应当在几何空间中反映语义关系。** 这一信条在多模态tokenization中既有直接继承，也经历了深刻的工程性改造，部分工作甚至构成了对这一信条的根本性替代。

### 2.1 延续的一面：语义等价性作为新版本的语义平滑性

word2vec的"king - man + woman ≈ queen"揭示的是嵌入空间的线性语义结构——相似语义的词在空间中相近，语义关系可以用向量运算表达。这一信条在多模态领域的直接继承者是**语义等价性**（semantic equivalence）概念。

ICLR 2025论文《Towards Semantic Equivalence of Tokenization in Multimodal LLM》明确批评现有方法"激进地切割视觉输入，破坏视觉语义"，并提出将低层视觉特征转化为可变数量的语义完整概念token[11]。这一批评的逻辑与word2vec的精神完全一致：切割边界应当对应语义边界，而非任意的空间网格。SeTok通过动态聚类将视觉特征组织为语义单元[11]，其机制与word2vec将共现词汇组织为语义邻居的逻辑存在结构性相似性。

ATM（Adaptive Time Series Tokenization）对时序生理信号的自适应分割[14]，本质上是在寻找具有生理意义的切割点——这与word2vec"词边界即语义边界"的逻辑一脉相承，只是将"词"替换为"生理事件"。3MToken（NeurIPS 2025）将音频、语义标签、艺术家信息融合为统一token[15]，追求的是一个能够同时编码声学特征和文化语境的多维语义空间，这是对word2vec语义空间概念的扩展而非否定。

### 2.2 分歧的一面：工程效率对语义平滑性的替代

离散VQ tokenization（VQVAE、VQ-GAN系列，以及2025年后的各种变体）对word2vec信条的背离是结构性的：连续语义空间被有限大小的码本（codebook）所取代，语义平滑性被码本的紧凑性和LLM兼容性所取代[6]。这一转向的驱动力并非对语义平滑性价值的否定，而是工程约束的胜利——LLM的自回归架构天然要求离散输入，连续嵌入需要额外的适配机制（如Q-Former、cross-attention），而离散token可以直接进入词表扩展框架。

这一权衡的代价是真实存在的：离散化不可避免地引入量化误差，码本大小有限意味着语义空间的分辨率受限。MoToRec明确将连续特征对齐中的问题称为"语义雾"（semantic fog），并试图通过稀疏正则化tokenization来构建更鲁棒的语义表示——这一表述本身就承认了当前离散化方案在语义质量上的不足。

Emu3代表了这一转向最彻底的版本：完全依赖next-token prediction的涌现能力，将语义结构的形成完全委托给Transformer本身，而非tokenizer设计[2]。这是否是对word2vec信条的根本性替代？从机制上看，是的——word2vec显式地在训练目标中构建语义空间，而Emu3的tokenizer只负责离散化，语义结构由预训练数据规模和Transformer的上下文学习能力隐式涌现。但从结果上看，如果Transformer确实能够从离散token序列中重建出具有语义结构的内部表示，则这只是实现路径的不同，而非语义空间信条的放弃。

Wave-Particle CDD-VT（连续-离散双重机制）[5]试图在两种哲学之间架桥：连续分支保留语义平滑性，离散分支保证LLM兼容性。这类折中方案的实际效果取决于两个分支的融合机制设计——如果融合层引入的信息损失超过了连续分支的语义收益，则这种折中只是形式上的。当前证据尚不足以对此做出定论。

### 2.3 新信条的雏形：四维评估框架

综合SEAM benchmark[1]和离散tokenization综述[6]的论述，当前社区对"好的多模态token"正在形成一个新的四维共识，可与word2vec时代的单一维度进行对比：

| 评估维度 | word2vec时代 | 多模态离散化时代 |
|---------|------------|---------------|
| 核心质量指标 | 语义平滑性（线性结构） | 语义等价性（信息完整性） |
| 跨单元关系 | 向量空间距离 | 跨模态对齐性 |
| 系统约束 | 无（独立嵌入层） | LLM兼容性（离散输入） |
| 效率指标 | 词表大小 | 压缩效率（序列长度） |

这一转变的深层含义是：word2vec的信条是**表示论**的（好的表示应当具有良好的几何性质），而多模态离散化时代的新信条是**功能论**的（好的token应当使下游LLM能够有效处理）。这不是简单的传承或背离，而是在不同约束条件下对"好的表示"的重新定义。

---

## Section 3：各模态Tokenization技术的演进轨迹（2023–2026）

**各模态tokenization技术的演进共享一条底层逻辑线：从通用压缩工具向LLM专用语义接口的转型。** 但不同模态的起点、速度和当前位置存在显著差异。

### 3.1 视觉（图像与视频）：演进最快，竞争最激烈

视觉tokenization是整个领域的技术前沿，代际更替最为密集。

**第一代（2020–2022）**：ViT式固定patch分割确立了将图像切分为固定大小patch并线性投影为嵌入向量的基本范式。VQVAE和VQ-GAN将连续视觉特征离散化，为后续工作奠定了码本学习的技术基础，但这一阶段的tokenizer主要服务于图像生成而非多模态理解。

**第二代（2023–2024）**：TokenFlow（CVPR 2025）明确以"弥合多模态理解与生成之间的长期鸿沟"为目标[16]，代表了视觉tokenizer从单一生成导向转向理解+生成双导向的关键转折。同期，语言引导方向出现：TexTok（CVPR 2025）通过引入语言监督信号引导tokenizer关注语义相关的视觉细节[3]；SeTok通过动态语义聚类实现可变数量的语义完整token[11]。这一阶段的核心矛盾是：固定粒度patch切割与视觉内容的语义粒度不匹配。

**第三代（2025–2026）**：三条技术路线并行演进。第一条是**自适应分辨率**：LongCat-Next（arXiv 2603.27538）支持任意分辨率的分层离散token，彻底打破了固定patch的空间约束[4]。第二条是**连续-离散融合**：Wave-Particle CDD-VT通过波粒二象性机制在连续和离散tokenization之间自适应切换[5]。第三条是**统一序列**：Emu3将图像、视频、文本统一为单一离散token序列，在Nature 2026发表[2]，代表了next-token prediction范式在多模态领域的最彻底实践。此外，AToken声称是首个在图像、视频、3D资产上统一实现高保真重建和语义理解的tokenizer[7]，而CVPR 2026的"A More Word-like Image Tokenization for MLLMs"则直接回应了word2vec的类比，试图让图像token具备类似词汇token的语义结构。

医学图像专用tokenizer（MedITok）的出现，以及Slot-MLLM基于Slot Attention的对象中心视觉tokenizer，则代表了视觉tokenization向特定应用场景深度定制的分化趋势。

### 3.2 音频与语音：从通用压缩转向LLM语义接口

音频tokenization的演进路径清晰：从通用音频压缩（EnCodec等神经编解码器）→面向LLM语义理解的专用tokenization。

VALL-E使用EnCodec提取离散声学token[17]，确立了"音频→离散token→LLM自回归生成"的基本范式。这一框架的局限在于EnCodec设计目标是音频压缩而非语义提取，导致生成的token序列语义密度较低、序列过长。

FocalCodec（Mila）针对这一问题，专门面向多模态LLM设计了超低比特率语音tokenization方案[12]，核心贡献是在极低比特率下保持语义质量，直接减少了输入LLM的token数量。3MToken（NeurIPS 2025）将音频、语义标签、艺术家信息融合为统一token[15]，代表了音频tokenization从单一声学维度向多维语义空间的扩展。

AudioCodecBench的出现[10]标志着音频tokenization评估体系的规范化——在此之前，不同工作使用不同的评估协议，横向比较困难。离散语音tokenization综述（arXiv 2502.06490）系统梳理了这一子领域的技术演进[17]，为后续研究提供了分类框架。

### 3.3 生理信号（EEG、ECG、EOG、EMG）：语义化是核心挑战

生理信号tokenization在技术成熟度上显著落后于视觉和音频领域，但正在经历快速发展。

核心挑战在于：生理信号的"语义边界"定义远比图像patch或音频帧更困难——心电信号的一个QRS复合波是一个有意义的生理事件，但EEG的语义边界在哪里？这一问题目前没有公认答案。

HeartLang（ICML 2025）通过QRS复合波检测解决了ECG的语义边界问题[8]，将生理事件到离散token的映射建立在明确的生理学知识基础上。这一方法的优势是可解释性强，但局限是只适用于有明确事件结构的信号（如ECG），对EEG等连续性更强的信号难以直接推广。

ATM（Adaptive Time Series Tokenization）引入时序自适应分割模块[14]，试图从数据中学习语义边界而非依赖先验生理知识，代表了更通用但解释性较弱的路线。多模态生理信号基础模型（arXiv 2504.19596）则尝试用统一架构处理EEG/ECG/EOG/EMG四种信号[9]，核心贡献是证明了跨生理信号模态的统一tokenization在原则上是可行的。

### 3.4 动作与4D信号：原始信号直接tokenization的论证

MTVCraft（arXiv 2505.10238）提供了一个重要的方法论案例：直接tokenize 4D运动捕捉数据，声称比传统的2D渲染姿态图像保留更多忠实信息[13]。这一工作的核心论点是"原始信号直接tokenization优于中间表示"——将3D/4D数据先渲染为2D图像再tokenize，会在渲染步骤引入不可逆的信息损失。这一论点如果成立，对整个多模态tokenization领域有重要启示：中间表示的选择本身是一个需要被质疑的设计决策。

### 3.5 跨模态统一：Token压缩成为并行约束

随着各模态tokenization粒度的细化，序列长度爆炸成为制约多模态LLM实际部署的核心瓶颈。Token压缩综述（arXiv 2507.20198）记录了这一趋势：token压缩技术正在与tokenization技术并行演进，成为多模态LLM效率优化的另一主战场[18]。这一矛盾的本质是：更细粒度的tokenization带来更丰富的语义信息，但也带来更长的序列，而LLM的计算复杂度与序列长度的平方成正比（标准注意力机制下）。

ICLR 2026 Workshop on Multimodal Intelligence: Next Token Prediction & Beyond[19]的召开，预示着社区正在探索超越next-token prediction的新范式——这可能是下一代tokenization技术的方向性信号。

**表2：多模态Tokenization技术演进时间线（2023–2026）**

| 时间 | 视觉（图像/视频） | 音频/语音 | 生理信号 | 4D/动作 | 跨模态 |
|------|----------------|---------|---------|---------|-------|
| 2023前 | ViT patch / VQVAE / VQ-GAN | EnCodec / SoundStream | 专用架构（无统一tokenizer） | 2D渲染姿态 | — |
| 2023–2024 | TokenFlow [16] / TexTok [3] / SeTok [11] | VALL-E EnCodec | HeartLang ECG [8] | — | Emu3统一序列 [2] |
| 2025 | AToken [7] / WeTok / MedITok / Slot-MLLM | FocalCodec [12] / 3MToken [15] / AudioCodecBench [10] | ATM自适应分割 [14] / 多模态生理基础模型 [9] | MTVCraft 4D [13] | SEAM benchmark [1] |
| 2026 | LongCat-Next [4] / Wave-Particle CDD-VT [5] / Word-like tokenization | 离散语音tokenization综述 [17] | （持续发展中） | — | ICLR 2026 Workshop [19] / Token压缩综述 [18] |

---

## Section 4：Tokenizer设计中的深层张力——四个未解问题

### 4.1 离散 vs. 连续：不可避免的语义损失与工程效率的根本性权衡

离散VQ tokenization的LLM兼容性优势是明确的，但量化误差引起的语义损失同样是真实的——这不是一个可以通过更好的工程实现消除的问题，而是离散化的数学本质决定的。Wave-Particle CDD-VT等折中方案[5]的代价在于：连续分支需要额外的融合机制，增加了模型复杂度；如果融合机制设计不当，可能同时失去连续分支的语义平滑性和离散分支的LLM兼容性。当前社区尚未形成关于"何时使用连续、何时使用离散"的设计准则。

### 4.2 Tokenizer与Backbone的耦合问题：可复现性的隐患

当前研究中，tokenizer往往与特定预训练框架深度绑定——Emu3的tokenizer是为其特定的next-token prediction框架设计的[2]，HeartLang的QRS分割策略依赖于特定的信号处理流程[8]。这种耦合使得tokenizer难以在不同backbone间迁移，降低了研究的可复现性和方法的通用性。Harmonizer（MDPI 2026）试图提供一个通用信号tokenization框架，但其在实际部署中的灵活性尚待验证。

### 4.3 评估体系的系统性滞后：已发表结论的可信度问题

SEAM benchmark在2025年8月才作为首个跨模态语义等价性系统评估基准出现[1]，这一时间节点本身具有重要意义：它意味着在此之前发表的大量工作，其tokenization收益声明都是在缺乏统一评估标准的条件下做出的。AudioCodecBench[10]为音频领域提供了类似的规范化，但视觉、生理信号等领域的系统性评估框架仍在建设中。对研究者而言，这意味着阅读2025年前的tokenization论文时，应当对其收益声明持更审慎的态度，尤其是当声明仅基于自定义指标时。

### 4.4 Token数量爆炸：正在成为部署瓶颈的核心矛盾

更细粒度的tokenization与序列长度爆炸之间的矛盾，是当前多模态tokenization发展的核心结构性张力。Token压缩综述（arXiv 2507.20198）记录的大量压缩技术的涌现[18]，本质上是对这一矛盾的工程响应。然而，token压缩与tokenization是相互制约的：如果在tokenizer层面就进行过度压缩，可能损失语义信息；如果在backbone层面进行压缩，则增加了系统复杂度。这一矛盾目前没有优雅的解决方案，而是被分散到tokenizer设计、模型架构、推理优化等多个层面分别处理。

---

## Section 5：值得深入探索的三个方向

**方向一：生理信号tokenization的系统综述**

生理信号tokenization目前处于"百花齐放但缺乏统一"的阶段：ECG有基于QRS的事件级方案，EEG的语义边界定义仍无共识，EOG和EMG的tokenization研究更为稀少。与视觉tokenization相比，生理信号面临的独特挑战在于：语义边界的定义需要领域知识（生理学、神经科学）的介入，而不能仅依赖数据驱动的方法。此外，临床应用的监管要求对可解释性的要求远高于计算机视觉任务，这对tokenizer设计提出了额外约束。系统梳理这一子领域的技术路线、评估标准和临床转化前景，将是一个高价值的研究综述方向。

**方向二：VQ码本设计的演进轨迹**

从固定码本（早期VQVAE）→可学习码本（VQ-GAN）→层次码本（RVQ，残差向量量化）→跨模态共享码本（当前部分工作尝试），码本设计的演进直接决定了多模态语义空间的结构。特别是跨模态共享码本的设计，涉及一个根本性问题：不同模态的语义是否可以在同一码本中被统一表示，还是必然需要模态特定的码本？这一技术线索的系统梳理，将为理解多模态语义空间的本质提供重要视角。

**方向三：基于SEAM和AudioCodecBench的横向比较研究**

SEAM benchmark[1]和AudioCodecBench[10]的出现，使得对现有主流tokenizer进行严格横向比较成为可能。一个系统性的比较研究，覆盖视觉、音频、跨模态三个维度，在统一评估协议下测试不同tokenizer设计选择（固定vs.自适应粒度、连续vs.离散、单模态vs.统一架构）对语义等价性的影响，将为社区提供目前缺失的设计选择指南。

---

## 给研究者的建议

**立即可行**：在阅读tokenization论文时，区分三类收益声明——消融实验（最可信，但依赖对照组设计）、代理指标（压缩率/重建质量，与下游任务相关性需单独验证）、理论论证（逻辑合理但缺乏实验支撑）。对于2025年8月前发表的工作，额外关注其评估协议是否与SEAM或AudioCodecBench等新兴标准兼容。

**短期（3–6个月）**：如果研究方向涉及视觉tokenization，Wave-Particle CDD-VT[5]和LongCat-Next[4]代表了当前技术前沿，值得重点跟踪；如果涉及生理信号，ATM[14]和多模态生理基础模型[9]的统一架构路线是最有前景的方向，但需警惕评估标准缺失带来的方法论风险。

**长期**：ICLR 2026 Workshop on Multimodal Intelligence: Next Token Prediction & Beyond[19]预示的方向值得持续关注——超越next-token prediction的新范式一旦成熟，可能对整个tokenization技术栈产生颠覆性影响。与此同时，token压缩与tokenization的协同设计（而非分别优化）将是提升多模态LLM实用性的关键工程方向[18]。

---

## 参考文献

[1] SEAM: Semantic Equivalence Across Modalities Benchmark. https://arxiv.org/html/2508.18179v1

[2] Emu3: Next-Token Prediction is All You Need. https://arxiv.org/abs/2409.18869

[3] Language-Guided Image Tokenization for Generation (TexTok, CVPR 2025). https://openaccess.thecvf.com/content/CVPR2025/papers/Zha_Language-Guided_Image_Tokenization_for_Generation_CVPR_2025_paper.pdf

[4] LongCat-Next: Tokenization at Arbitrary Resolutions. https://arxiv.org/abs/2603.27538

[5] Wave-Particle Continuous–Discrete Dualistic Visual Tokenization (CDD-VT). https://openreview.net/forum?id=VK3p5dXYL6

[6] Discrete Tokenization for Multimodal LLMs: A Comprehensive Survey (arXiv 2507.22920). https://arxiv.org/abs/2507.22920

[7] AToken: A Unified Visual Tokenizer. https://openreview.net/forum?id=a4fSF5pGJq

[8] HeartLang: ECG Tokenization via QRS Complex Detection (ICML 2025). https://icml.cc/virtual/2025/poster/44523

[9] Multimodal Physiological Foundation Model (EEG/ECG/EOG/EMG, arXiv 2504.19596). https://arxiv.org/pdf/2504.19596

[10] AudioCodecBench: Benchmark for Neural Audio Codecs as Tokenizers. https://openreview.net/forum?id=JeIDPXc9XG

[11] Towards Semantic Equivalence of Tokenization in Multimodal LLM (SeTok, ICLR 2025). https://iclr.cc/virtual/2025/poster/28428

[12] FocalCodec: Ultra-Low Bitrate Speech Tokenization for Multimodal LLMs. https://mila.quebec/en/article/focalcodec-giving-llms-ears-and-a-voice-at-ultra-low-bitrates

[13] MTVCraft: Tokenizing 4D Motion for Arbitrary Character Animation. https://arxiv.org/html/2505.10238v5

[14] ATM: Adaptive Time Series Tokenization with Semantic. https://openreview.net/pdf/ac0f9e8cfd9e2be9e064b43d6c2fe14e3990f750.pdf

[15] 3MToken: Multimodal Music Tokenizer (NeurIPS 2025). https://neurips.cc/virtual/2025/123762

[16] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation (CVPR 2025). https://openaccess.thecvf.com/content/CVPR2025/papers/Qu_TokenFlow_Unified_Image_Tokenizer_for_Multimodal_Understanding_and_Generation_CVPR_2025_paper.pdf

[17] Discrete Speech Tokenization Survey (arXiv 2502.06490). https://arxiv.org/html/2502.06490v2

[18] A Survey of Token Compression for Efficient Multimodal Large Language Models (arXiv 2507.20198). https://arxiv.org/html/2507.20198v5

[19] ICLR 2026 Workshop on Multimodal Intelligence: Next Token Prediction & Beyond. https://openreview.net/group?id=ICLR.cc/2026/Workshop/MM_Intelligence