# EEG-fNIRS Tokenizer Reset Plan

> Last Updated: 2026-04-08
> Status: Active tokenizer mainline reset document
> This document replaces the previous alignment-first interpretation of the next stage.

## 1. Reset Statement

我们现在需要停止在旧的 EEG-fNIRS shared-codebook 设计上继续叠加约束，并把 tokenizer 主线重新定义为：

1. 先做出健康、可解释、可复现的 codebook。
2. 再用跨模态指标验证 shared branch 是否真的承载了生理共性。
3. 只有当一个新增机制能够改善 codebook 表现或 shared/private 语义边界时，才保留它。

旧的推进方式有一个根本问题：它默认“alignment stronger = tokenizer better”。现有实验已经不支持这个判断。当前更可靠的结论是：

- token identity overlap 不是 EEG-fNIRS 生理对应的正确主目标；
- reconstruction 与 codebook usage 的改善，并不会自动破坏跨模态关系；
- 真正需要被约束的是 shared branch 应该重建什么，而不是继续给它加更多 identity-style alignment loss。

因此，从现在开始，alignment 不再是 tokenizer 主线的出发点，而是 codebook-first 设计完成后的验证面。

## 2. Evidence Base

当前重整不是凭印象做方向切换，而是基于仓库里已经形成的三类证据：

1. 单模态与早期对齐计划
   - [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md)
   - 此前的 alignment-first 叙事已经被本文件替换，不再适合作为当前主线依据

2. shared-codebook 与 factorized 设计复盘
   - [docs/experiement_reports/Shared_codebook_structure_report.md](experiement_reports/Shared_codebook_structure_report.md)
   - [docs/experiement_reports/Shared_private_factorization_design.md](experiement_reports/Shared_private_factorization_design.md)

3. 当前代码实现状态
   - [src/tokenizers/shared_labram_vqnsp.py](../src/tokenizers/shared_labram_vqnsp.py)
   - [src/tokenizers/factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py)
   - [src/tokenizers/codebook_focus_factorized_labram_vqnsp.py](../src/tokenizers/codebook_focus_factorized_labram_vqnsp.py)

## 3. What Has Actually Been Tried

下面按“尝试族”而不是按零散 run 来重排现有工作。重点不是穷举所有配置，而是识别哪些设计假设已经被证实、被否定，或只能保留为对照。

| 阶段 | 代表实现 | 主要假设 | 结果 | 当前判断 |
| --- | --- | --- | --- | --- |
| 单模态验证 | P0 / P0plus 单模态 VQ-VAE 与 LaBraM tokenizer | 先验证 EEG 与 fNIRS 各自能否学出稳定 codebook | 单模态重建与 codebook 健康度已足够支撑进入多模态阶段 | 不再是主瓶颈 |
| Shared-codebook baseline | [src/tokenizers/shared_labram_vqnsp.py](../src/tokenizers/shared_labram_vqnsp.py) | 两模态共享一个 codebook，并用 latent/assignment alignment 推动 token identity | 架构可运行，但把“共享 token index”当成主要目标，假设过强 | 保留为 control baseline |
| Early factorized runs | [src/tokenizers/factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py) | shared/private 分解，允许共性与模态特异信息分离 | 结构方向正确，开始摆脱一个 shared bottleneck 同时承担所有任务的问题 | 结构上应保留 |
| V4 long-run to V5 | factorized 方案后期长跑与 shared-only full reconstruction 版本 | 共享分支可以通过更强重建压力学到更好的跨模态共性 | V5 的 full reconstruction、shared perplexity、best-lag MI 都改善，但 token identity 仍接近 0，且 shared branch 开始像第二条通用重建通路 | 证明 shared-only full raw reconstruction 方向错误 |
| V6 / codebook-focused factorized | [src/tokenizers/codebook_focus_factorized_labram_vqnsp.py](../src/tokenizers/codebook_focus_factorized_labram_vqnsp.py) | 保留 factorization、lag-aware shared coupling、common/residual 目标，移除遗留 alignment auxiliaries | 与最新复盘结论一致，是当前最接近可收敛主线的实现 | 作为当前推荐参考实现 |

## 4. Stable Conclusions From Existing Experiments

### 4.1 单模态 tokenizer 已经不是当前关键矛盾

P0 与 P0plus 阶段已经说明，EEG 和 fNIRS 的单模态 tokenizer 至少满足两个条件：

1. 可以稳定重建；
2. 可以提供足够健康的 codebook 使用统计，支持多模态阶段继续推进。

所以现在继续纠结“tokenization 是否可行”没有意义。真正的问题是多模态 tokenizer 应该如何分配 shared 与 private 的表示职责。

### 4.2 shared token identity 不是主要目标

shared-codebook baseline 把下面这件事当作默认成功形态：

1. EEG 与 fNIRS 尽量落到同一个 code index；
2. 同位 token 最好直接对齐；
3. token overlap 越高越好。

但已有证据表明，这些指标至多是诊断量，不是主目标。EEG 与 fNIRS 的真实关系更接近“允许时滞的结构化预测”，而不是“同一时刻必须产出同一个离散 index”。

### 4.3 reconstruction 与 codebook usage 的改善不会自动破坏跨模态关系

V4 long-run 与 V5 的比较给出了一个关键反例：

1. V5 的 full-signal reconstruction 更好；
2. shared codebook perplexity 更高；
3. best-lag mutual information 也更好；
4. 但 token identity match 仍接近 0。

这说明“只要重建更强，就一定把 shared token overlap 压坏”这个叙事不成立。

### 4.4 真正的失败模式是 shared branch 的目标定义错了

V5 暴露的核心问题不是 overlap 太低，而是 shared branch 被训练去单独重建完整原始信号后，会退化成第二条通用重建路径。这样做虽然能提高 raw reconstruction，但会带来两个直接后果：

1. shared/private 语义边界变得模糊；
2. shared branch 不再是“跨模态共性瓶颈”，而是“又一条高容量捷径”。

因此，下一阶段不应该继续加大 alignment 力度，而应该限制 shared branch 只去建模 common component，并把 private branch 明确地绑定到 residual reconstruction。

### 4.5 factorization 是结构性修正，不是又一个 patch

shared/private factorization 的价值不在于“再多加几个 loss”，而在于它终于允许模型表达：

- shared branch 负责跨模态共性；
- EEG private branch 负责快速电生理细节；
- fNIRS private branch 负责缓慢血流细节。

这比单一 shared codebook 同时编码共性、模态特异、延迟耦合与重建细节更符合问题本身。

### 4.6 downstream 诊断也支持这次重置

[docs/experiement_reports/factor_probe_experiment_based_on_early_stage_downstream_implementation.md](experiement_reports/factor_probe_experiment_based_on_early_stage_downstream_implementation.md) 表明，当前表示栈仍然存在两个明显问题：

1. task signal 偏弱；
2. multimodal representation 中 subject leakage 依然明显。

这进一步说明，继续在旧对齐叙事上加约束不会自然把 tokenizer 变成更好的 foundation 表示。必须先把 codebook 与 shared/private 结构本身做干净。

## 5. Current Implementation Status In Repo

当前仓库里的 tokenizer 实现已经形成了清晰的三层关系：

### 5.1 Control baseline

- [src/tokenizers/shared_labram_vqnsp.py](../src/tokenizers/shared_labram_vqnsp.py)

作用：

1. 作为 shared-codebook 家族的可运行基线；
2. 保留给对照实验使用；
3. 不再作为默认开发主线。

### 5.2 Legacy factorized research surface

- [src/tokenizers/factorized_labram_vqnsp.py](../src/tokenizers/factorized_labram_vqnsp.py)

作用：

1. 表达 shared/private factorization 的完整研究空间；
2. 保留各种辅助损失与研究接口；
3. 不应继续作为“默认配置里什么都开一点”的主线。

### 5.3 Current recommended mainline

- [src/tokenizers/codebook_focus_factorized_labram_vqnsp.py](../src/tokenizers/codebook_focus_factorized_labram_vqnsp.py)

这个实现已经把主线原则写得很明确：

1. 保留 shared/private factorization；
2. 保留 lag-aware shared coupling；
3. 使用 shared common target 与 private residual target；
4. 保留 orthogonality 与 codebook balance；
5. 从默认优化路径里移除 legacy experimental auxiliaries。

这与实验复盘结论一致，因此它应该被正式提升为当前 tokenizer 研发的参考实现，而不是继续把 shared-codebook 计划文档当作默认路线。

## 6. Mainline Decision Matrix

接下来所有 tokenizer 研发都按下面这个决策矩阵推进。

### 6.1 Keep in default mainline

这些组件属于当前默认保留项：

1. shared/private factorization；
2. lag-aware shared coupling；
3. shared-common / private-residual target decomposition；
4. shared/private orthogonality or decoupling constraint；
5. codebook-balance regularization；
6. shared/private branch ablation diagnostics；
7. reconstruction-first then coupling-warm-start training schedule。

### 6.2 Keep as baselines or stress tests only

这些设计仍有价值，但只能作为对照，不应继续被包装成主线：

1. single shared-codebook LaBraM；
2. legacy generic factorized runs with many auxiliaries enabled；
3. overfit/high-capacity factorized variants；
4. exact token overlap 追求型实验。

### 6.3 Remove from default optimization path

这些项不应再出现在默认 mainline 配置里，除非作为明确 ablation：

1. latent alignment loss；
2. assignment alignment loss；
3. hard assignment alignment；
4. shared entropy / private entropy regularization；
5. shared-only full raw waveform reconstruction；
6. 把 token identity match 当作主要 success criterion。

## 7. Physiological Token Semantics And Success Criteria

以后我们不再把 token 当作“原始波形片段的离散名字”，也不再把 shared token 是否出现更多同位 index match 当作主要目标。当前主线里，一个好的生理 token 更接近“可复用的局部生理状态标识”，它应该让后续 foundation model 能够在离散状态空间里做预测、转移建模和跨模态推断。

### 7.1 What a physiological token should mean

当前主线里，token 的语义归纳应满足下面五条。

1. state semantics：同一个 token 应对应相近的局部生理状态，而不是仅仅对应形状相似的原始波形片段；
2. transition semantics：token 序列应保留状态转移结构，使未来状态比边际分布更可预测；
3. lag-aware cross-modal semantics：EEG 与 fNIRS 的 shared token 应表达允许时滞的共性状态，而不是追求同步同 index；
4. nuisance-invariant semantics：token 应尽量对被试、设备、噪声和预处理细节不敏感，只保留对生理机制有用的信息；
5. branch semantics：shared token 应代表跨模态共性状态，EEG private token 应保留快速电生理残差，fNIRS private token 应保留慢速血流残差。

### 7.2 Layer A: codebook health gates

这是最优先的通过门槛。

1. reconstruction 不能明显退化；
2. shared 与 private codebook 都不能出现系统性 collapse；
3. perplexity、active-code count、usage coverage 必须稳定；
4. branch ablation 必须显示 shared 与 private 都在承担不同职责，而不是某一支完全失效或单独包办全部重建。

这是语义空间的最低保真门槛，但还不是“有意义的生理 token”本身。

### 7.3 Layer B: semantic state quality

只有 Layer A 通过，才值得谈 token 是否真的承载了状态语义。这里需要新增一组专门针对语义空间的指标：

1. intra-token state consistency：同一 token 覆盖的 shared-common target 或生理特征摘要，其类内离散度应显著低于全局离散度；
2. prototype separation ratio：不同 token 原型之间的距离，应显著高于各自类内方差；
3. transition predictability gain：$H(Z_{t+\Delta}) - H(Z_{t+\Delta} \mid Z_t)$ 应显著大于 0，说明 token 序列确实保留了状态转移结构；
4. augmentation consistency：在不改变生理语义的扰动下，相同窗口应落到同一 token 或近邻 token；
5. branch responsibility gap：shared ablation 应主要损害 common target，private ablation 应主要损害 residual target，而不是两边职责混在一起。

### 7.4 Layer C: structured cross-modal value

只有通过 Layer A，才谈跨模态价值。

1. best-lag mutual information 应稳定优于 lag 0；
2. conditional KL gain，即 $D_{KL}(P(Z^{fnirs} \mid Z^{eeg}) \| P(Z^{fnirs}))$，应明显大于 0；
3. shared EEG 到 delayed fNIRS 的预测质量或 masked token prediction gain 应提升；
4. bidirectional coupling 如果保留，至少一侧要有稳定收益；
5. overlap 与 token identity 仅作为补充诊断，而不是主要门槛。

### 7.5 Layer D: invariance and downstream sanity

tokenizer 不是直接为了分类器优化，但如果主线真的更健康，下游不应继续强化错误信号。

1. task signal 不应持续停留在近 chance；
2. subject leakage 不能继续随 multimodal 表示增强而变得更强；
3. session / device stability 不应在 shared states 上明显恶化；
4. 如果 Layer A-C 都改善，但 Layer D 没有任何正向变化，需要重新检查 shared target 是否仍不对。

## 8. What Current Development Should Focus On

下面是从 codebook performance 出发的当前开发主线，不是“再想一个 alignment scheme”。

### 8.1 Freeze one canonical mainline family

默认主线应固定为 codebook-focused factorized family，而不是继续在 shared_labram_vqnsp 与 generic factorized 之间来回摇摆。

建议的 canonical family：

1. one control baseline: shared_labram_vqnsp；
2. one structural baseline: factorized_labram_vqnsp；
3. one mainline reference: codebook_focus_factorized_labram_vqnsp；
4. one stress test only: overfit_factorized_labram_vqnsp。

### 8.2 Standardize a mandatory tokenizer scorecard

每一次 mainline run 都必须输出同一套 scorecard，而不能只汇报最显眼的几项曲线。

最低要求应包括，详细定义见 [SEMANTIC_TOKEN_SCORECARD.md](SEMANTIC_TOKEN_SCORECARD.md)：

1. EEG full reconstruction；
2. fNIRS full reconstruction；
3. shared-common reconstruction；
4. EEG-private residual reconstruction；
5. fNIRS-private residual reconstruction；
6. shared/private perplexity、active-code counts、usage coverage、gini、top-k coverage；
7. branch-only decoding gaps 与 branch responsibility gap；
8. intra-token consistency、prototype separation ratio、transition predictability gain；
9. augmentation consistency 或其轻量替代指标；
10. best-lag MI、lag-0 MI、conditional KL gain 与 masked token prediction gain；
11. shared usage by modality 与 private usage by modality；
12. subject leakage、task signal、session/device stability；
13. gradient semantic-share 与 gradient conflict dashboard 的关键节点快照。

### 8.3 Make ablation the default workflow

以后不再接受“把多个新约束一起加上再看总结果”的实验方式。默认流程应该是：

1. reconstruction-first run；
2. coupling warm-start run；
3. branch ablation run；
4. single-auxiliary ablation；
5. only then compare against baselines。

任何新机制如果无法在这条 ladder 上解释清楚自己的贡献，就不应该进主线。

### 8.4 Optimize for branch semantics, not more constraints

当前最该开发的不是新的 identity-style alignment，而是让 shared/private 分工更稳定、更可测：

1. common target 的构造是否足够稳定；
2. residual target 是否真的把模态特异细节留给 private；
3. shared codebook size 与 private codebook size 的比例是否合理；
4. branch dropout / masking 是否只是在防 bypass，而不是制造假共享；
5. coupling head 是否真正利用了 shared states，而不是仅仅当作另一个附加 loss。

### 8.5 Clean the experiment surface instead of widening it

实验面已经太大了。当前开发应该做减法：

1. 把默认 loss 集固定住；
2. 把默认 config 家族固定住；
3. 把日志与可视化指标固定住；
4. 把 shared-codebook 家族正式降级成 baseline；
5. 只有在 canonical scorecard 稳定后才开放新的结构探索。

## 9. Immediate Development Backlog

这是现在最值得做、也最能减少后续噪声的工作序列。

### Priority 1. Promote the codebook-focused factorized tokenizer to the documented mainline

要明确写死：

1. 当前推荐参考实现是 [src/tokenizers/codebook_focus_factorized_labram_vqnsp.py](../src/tokenizers/codebook_focus_factorized_labram_vqnsp.py)；
2. shared_labram_vqnsp 只保留为 control baseline；
3. generic factorized 只保留为研究表面，不再作为默认配置承载所有实验变量。

### Priority 2. Standardize canonical configs and reports

需要建立一个最小且稳定的 canonical run contract：

1. 固定主线配置；
2. 固定 baseline 配置；
3. 固定 ablation 配置；
4. 固定可视化与日志字段；
5. 固定复盘模板。

### Priority 3. Re-run a small, clean comparison matrix

下一轮不应继续大范围扫参数，而应只跑少量、可解释的对照：

1. shared-codebook baseline；
2. generic factorized baseline；
3. codebook-focused mainline；
4. one ablation removing common/residual targets；
5. one ablation restoring a legacy alignment auxiliary。

目标不是比谁短期分数最高，而是回答：哪些机制真的改善 codebook health，哪些只是制造更复杂的训练过程。

### Priority 4. Push all future proposals through one promotion rule

任何新的 tokenizer 机制，只有满足以下条件才允许进入默认 mainline：

1. Layer A codebook health 不退化；
2. Layer B semantic state quality 不退化，最好有明确提升；
3. Layer C shared-branch structured value 有明确提升；
4. Layer D invariance / downstream sanity 不出现明显倒退；
5. 可以通过 ablation 解释，且不把 shared branch 再次变回第二条全能重建捷径。

## 10. What We Should Explicitly Stop Doing

从现在开始，以下做法应视为偏离主线：

1. 在 shared-codebook baseline 上继续叠加更多 alignment loss；
2. 用 token identity match 或 overlap 当作主要成功指标；
3. 在没有 branch ablation 的情况下解释 shared branch 的意义；
4. 在没有 canonical scorecard 的情况下宣布某个 run “更好”；
5. 继续把“更多约束”当作“更接近跨模态生理对应”的默认方向。

## 11. Bottom Line

当前 EEG-fNIRS tokenizer 的主问题，已经不是“怎么让两个模态更像”，而是“怎样构造一个 shared/private 语义边界清晰、codebook 健康、并且能承载 lagged physiological structure 的离散表示系统”。

因此，当前最合理的开发方向是：

1. 正式把 tokenizer 主线切换到 codebook-focused factorized family；
2. 把 shared-codebook 叙事降级为 baseline/control；
3. 把 common/residual target、lag-aware shared coupling、orthogonality、codebook balance 作为默认保留结构；
4. 把 latent alignment、assignment alignment、hard assignment、entropy regularization、shared-only raw reconstruction 移出默认 mainline；
5. 所有后续研发都先回答 codebook performance 有没有变好，再谈 alignment 有没有更漂亮。

如果这个重置不做，后面所有 tokenizer 开发都会继续停留在“给旧设计加 patch”的状态里。现在已有实验已经足够说明，这条路不应该再走下去了。