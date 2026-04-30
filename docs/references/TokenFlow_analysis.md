# TokenFlow 对当前 EEG-fNIRS Tokenizer 路线的启示

> Paper: [docs/paper_pdf/Qu 等 - 2025 - TokenFlow Unified Image Tokenizer for Multimodal Understanding and Generation.pdf](../paper_pdf/Qu%20%E7%AD%89%20-%202025%20-%20TokenFlow%20Unified%20Image%20Tokenizer%20for%20Multimodal%20Understanding%20and%20Generation.pdf)
> Scope: 基于原文阅读后的路线反思，不是二手综述摘要

## 1. 先澄清：TokenFlow 实际做的是什么

TokenFlow 不是“把两个异构模态直接压到同一个 codebook 并要求输出同一个 token”。

它的实际结构是：

1. 同一个图像输入经过两条流：
   - semantic encoder，学习高层语义特征；
   - pixel encoder，学习低层像素细节。
2. 两条流各自拥有独立 codebook：
   - semantic codebook；
   - pixel codebook。
3. 两个 codebook 通过 shared mapping 绑定在同一组 index 上。
4. 对每个 patch，不是分别选两个 index，而是通过

$$
i^* = \arg\min_i \left(d_{sem,i} + w_{dis} d_{pix,i}\right)
$$

选出一个共享 index，再同时取出该 index 对应的 semantic embedding 和 pixel embedding。

因此，TokenFlow 的“映射到同一 token”本质上是：

- 同一图像 patch 的两种特征粒度，被绑定到同一个 index；
- 不是不同模态在物理上或时间上被强制输出同一个 index；
- 也不是一个单一 codebook 同时承担全部语义和细节。

这点非常重要。它支持的是“解耦后再用 index 相关性统一”，而不是“直接把所有东西塞进一个共享离散瓶颈”。

## 2. 它和我们当前问题真正相似的地方

TokenFlow 与当前 EEG-fNIRS tokenizer 路线真正相似的地方，不在“同 token 对齐”，而在下面三个结构判断。

### 2.1 单一瓶颈很难同时满足冲突目标

TokenFlow 的出发点是：

- understanding 需要高层语义；
- generation 需要低层细节；
- 单一 reconstruction-oriented VQ tokenizer 无法同时把两件事做好。

这和我们当前已经得到的结论高度同构：

- EEG/fNIRS shared branch 要表达跨模态共性；
- private branches 要表达模态特异残差；
- 一个 shared bottleneck 同时承担共性、细节、时滞耦合和重建，目标冲突过大。

因此，TokenFlow 强化了一个方向判断：

> 当任务要求的“信息粒度”本身冲突时，先做结构解耦，再谈统一，比继续强化一个单瓶颈更合理。

### 2.2 统一不一定等于单码本塌缩

TokenFlow 没有选择“一个 codebook 统一语义和像素”，而是选择：

- dual codebooks；
- shared index mapping；
- dual decoders；
- 联合量化目标。

这与我们在 [docs/experiement_reports/Shared_private_factorization_design.md](../experiement_reports/Shared_private_factorization_design.md) 中提出的 shared/private factorization 非常接近。两者共同说明：

> “统一”更合理的含义，是建立可计算的对应结构，而不是把所有表示责任压缩进同一个离散空间。

### 2.3 语义先验可以显式注入 tokenizer，而不是完全依赖下游模型涌现

TokenFlow 使用 CLIP-style teacher 初始化 semantic encoder，并显式加入 semantic decoder 去对齐 teacher feature。这说明作者并不相信“只靠 reconstruction 或 next-token prediction 就能自然长出足够好的语义 tokenizer”。

这对我们是有价值的提醒：

- 如果 shared branch 的目标定义过弱，它就会退化成重建捷径；
- 如果我们希望 shared states 更接近生理共性状态，就需要更明确的 shared target，而不是只看 raw reconstruction 或 overlap。

## 3. 它和我们问题不相似、不能直接照搬的地方

### 3.1 TokenFlow 的 shared index 建立在“同一 patch 的双视角”上

TokenFlow 的 semantic flow 和 pixel flow 虽然信息粒度不同，但它们来自同一张图像的同一 patch。

因此：

- 两条流对齐的是同一视觉对象；
- shared index 对应的是同一个视觉局部同时满足“语义相近”和“像素相近”的组合；
- 强行要求同 index 是有物理基础的。

EEG 与 fNIRS 不是这种关系。

我们的问题里：

- 两个模态来自不同生理机制；
- 时间尺度不同；
- fNIRS 还存在血流延迟；
- 同一时刻窗口不一定对应同一个状态边界。

所以 TokenFlow 不能被当作“EEG/fNIRS 也应该同 index”的证据。

### 3.2 它解决的是“同模态双粒度统一”，不是“异构模态语义耦合”

TokenFlow 的核心矛盾是：

- 一个图像 tokenizer 如何同时服务理解与生成。

我们的核心矛盾是：

- 两个异构生理模态如何划分 shared/private 语义职责；
- 如何用允许时滞的结构关系表达跨模态耦合；
- 如何避免 subject leakage 和 reconstruction shortcut。

因此，TokenFlow 更像是“结构设计启发”，不是“问题同构的直接答案”。

### 3.3 它的评估口径仍然主要是任务表现和重建质量

TokenFlow 的主要证据包括：

- reconstruction FID；
- multimodal understanding benchmark；
- generation benchmark；
- codebook utilization。

这些证据足以证明它对视觉任务有效，但不足以证明“token 本身已经具备稳定语义边界”。

对我们的问题，这一点更关键，因为生理 token 是否成立，不能只靠下游 accuracy 和 reconstruction 曲线判断。这也是 [docs/SEMANTIC_TOKEN_SCORECARD.md](../SEMANTIC_TOKEN_SCORECARD.md) 仍然必要的原因。

## 4. 对当前主线的直接影响

### 4.1 当前 codebook-first factorized 主线没有被削弱，反而被间接强化

TokenFlow 给出的最强支持不是 overlap，而是：

- 结构解耦优于单瓶颈；
- 不同表示职责应该由不同 codebook/branch 承担；
- 统一机制应建立在明确的结构关系上。

因此，它支持继续坚持：

- [docs/archive/plans/NEXT_STAGE_ALIGNMENT_PLAN.md](../archive/plans/NEXT_STAGE_ALIGNMENT_PLAN.md)
- [src/tokenizers/codebook_focus_factorized_labram_vqnsp.py](../../src/tokenizers/codebook_focus_factorized_labram_vqnsp.py)

而不是回到 shared-codebook overlap-first 路线。

### 4.2 它不支持恢复 identity-style alignment loss

TokenFlow 的 same-index 机制成立，是因为两条流描述的是同一 patch 的两种粒度，而不是两个异构模态之间存在天然同 token 对应。

所以它不能被用来支持下面这些旧想法：

1. EEG 与 fNIRS 应该尽量落到同一个 shared index；
2. token identity overlap 越高越好；
3. latent/assignment alignment 越强，tokenizer 越好。

这些判断在我们自己的实验复盘里已经被否定，TokenFlow 不构成反例。

### 4.3 它提示我们一个更值得尝试的下一步：耦合 codebook，而不是强共享 codebook

TokenFlow 最值得借鉴的，不是“共享一个 codebook”，而是：

- 分开学两套 embedding；
- 用 index 级机制维持相关性；
- 让统一发生在映射层，而不是发生在表示责任塌缩上。

这与 [docs/experiement_reports/Shared_private_factorization_design.md](../experiement_reports/Shared_private_factorization_design.md) 中提到的

- shared quantizer，或
- a pair of lightly tied quantizers

非常一致。

对 EEG-fNIRS 来说，这比“继续让 shared branch 进入同一个 quantizer”更值得认真考虑。

更具体地说，若当前单 shared quantizer 仍不稳定，下一步更合理的变体不是恢复 alignment auxiliary，而是：

1. EEG shared 与 fNIRS shared 各自使用独立但轻度耦合的 codebook；
2. 通过 learned mapping / tied state IDs / lag-conditioned transition coupling 建立对应关系；
3. 只在 common-target 层面对齐，不在 raw modality 层面追求同 index。

### 4.4 当前实现里仍有一个需要继续反思的隐含强假设

当前代码实现中：

- [src/tokenizers/shared_labram_vqnsp.py](../../src/tokenizers/shared_labram_vqnsp.py)
- [src/tokenizers/factorized_labram_vqnsp.py](../../src/tokenizers/factorized_labram_vqnsp.py)

仍要求 EEG 和 fNIRS 在每个窗口内产出相同 token 数。

这意味着我们虽然已经放弃“同 index overlap 是主目标”，但还没有完全放弃“同时间网格离散化”的结构假设。

TokenFlow 的成功提醒我们：

- 当 shared mapping 有坚实对象对应关系时，同 index 是合理的；
- 当没有这种对应关系时，更应先反思 token 边界和映射结构本身。

对 EEG-fNIRS 而言，这提示我们未来可能需要进一步松动：

- 相同 token 数；
- 同步 patch 边界；
- 单 shared quantizer。

## 5. 现阶段更稳妥的结论

基于 TokenFlow 原文，当前最稳妥的结论是：

1. 它支持“先 factorize，再 unify”；
2. 它不支持“异构模态直接追求同 token identity”；
3. 它支持“统一应建立在明确映射机制上，而不是单瓶颈塌缩”；
4. 它进一步说明我们当前最缺的不是新 alignment loss，而是更清晰的 shared target 与更完整的语义评估闭环。

换句话说，TokenFlow 对我们的真正启发不是“去追求同一 token”，而是：

> 如果确实需要统一，应该统一的是 state correspondence mechanism，而不是强迫 EEG 和 fNIRS 在原始离散空间里长得一样。

## 6. 对下一阶段实验的具体建议

### 必须继续坚持的

1. 保持 codebook-focused factorized mainline；
2. 保持 shared-common / private-residual 的职责划分；
3. 继续把 overlap 当补充诊断，而不是 success criterion；
4. 把 scorecard 自动化补齐，而不是只看 reconstruction 与 perplexity。

### 最值得新增的一条结构实验

做一个 TokenFlow-inspired，但更适合生理信号的变体：

1. 不再让 EEG shared 与 fNIRS shared 进入完全相同的 quantizer；
2. 改为两套 lightly tied shared codebooks；
3. 用 lag-aware mapping 或 transition coupling 建立 shared state correspondence；
4. 仍然用 [docs/SEMANTIC_TOKEN_SCORECARD.md](../SEMANTIC_TOKEN_SCORECARD.md) 做评估，而不是只看 overlap。

### 当前不该做的

1. 因为 TokenFlow 使用 shared indices，就恢复 latent/assignment identity alignment；
2. 把 TokenFlow 误解为“单一共享 codebook 比 factorization 更优”；
3. 用下游分类收益直接替代生理 token 语义验证。

## Bottom line

TokenFlow 不是 shared-codebook overlap-first 的证据，而是 factorized-unification 的证据。

它最有价值的启发是：

- 冲突的信息粒度应先结构解耦；
- 统一机制应落在映射层；
- same-index 只有在对象对应关系足够强时才成立。

对 EEG-fNIRS 来说，这意味着我们当前路线的大方向是对的，但“shared branch 如何统一”这件事还可以继续从“强共享 quantizer”推进到“耦合 quantizer / 共享状态映射”这一层。