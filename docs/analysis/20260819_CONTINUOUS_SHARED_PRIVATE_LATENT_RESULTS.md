# 无 VQ 连续 shared/private latent 完整实验结果

_状态：开发范围内的探索性实验已完成；2026-08-19_

## 结论

在不引入 VQ、codebook、离散赋值或显式 latent 对齐损失的条件下，当前
shared/private 架构**没有构建出具有可验证匹配关系的 EEG–fNIRS shared
模式**。预登记的严格判据要求 16 个主检验单元的 simultaneous 95% 区间下界
全部大于零；实际仅 `2/16` 通过，因此总体结论为“不支持”。

两个通过单元都是 EEG shared 对 SSM target 的任务内预测：motor imagery
与 word generation。四个 fNIRS target 单元和八个跨模态 swap 单元均未通过。
尤其是 matched shared-latent substitution 几乎不优于同受试者、同条件、同
token time 下的 trial-deranged substitution，说明两种模态的 continuous shared
latent 没有呈现所要求的样本级可互换匹配关系。

这是对当前架构、当前开发 split 和当前 SSM proxy 的负结果。它不证明原始测量
中不存在生理耦合，也不支持恢复 VQ；相反，它把下一步问题定位在 shared target
的有效性、跨模态可识别性与训练构念本身。

## 实验范围与完整性

- 四个 task-specific 模型：mental arithmetic、motor imagery（LMI+RMI）、
  word generation、n-back（0/2/3-back）；
- 三个固定随机种子：`20260819`、`20260820`、`20260821`，共 `12/12`
  task/seed 训练完成、零失败；
- 共 `2,001` 个 canonical EEG–fNIRS window；fit subjects 01–18，development
  subjects 19–23；
- 每种模态各自具有 independent shared/private encoder；两个 shared encoder
  通过同一个、无 modality ID 的 trajectory decoder 预测完整 SSM shared driver；
- raw decoder 只接收 `[stop_gradient(shared), private]`，raw reconstruction
  gradient 不回传 shared encoder；
- 模型、loss、输出、配置与 manifest 均不包含 quantizer/codebook/VQ；
- Single-Trial subjects 24–29 保持关闭；manifest 中 `protected_open=false`，
  protected loader 未被调用。

每个 seed 先得到受试者级统计量，再在受试者内平均三个 seed。最终区间来自
10,000 次 subject-block bootstrap，并用 studentized max-stat 同时校正 16 个
主检验单元；数据集内的 task pairing 在重采样中保持不变。

## 预登记 16 单元主结果

| 任务 | 主检验端点 | mean ΔR² | simultaneous 95% interval | 正值受试者 | 通过 |
| --- | --- | ---: | ---: | ---: | --- |
| Mental arithmetic | EEG → SSM target | 0.0377 | [-0.0185, 0.0938] | 4/5 | 否 |
| Mental arithmetic | fNIRS → SSM target | -0.0206 | [-0.0386, -0.0026] | 1/5 | 否 |
| Mental arithmetic | fNIRS shared → EEG swap | 0.0000 | [-0.0014, 0.0014] | 2/5 | 否 |
| Mental arithmetic | EEG shared → fNIRS swap | 0.0051 | [-0.0301, 0.0403] | 2/5 | 否 |
| Motor imagery | EEG → SSM target | 0.0710 | [0.0176, 0.1244] | 5/5 | 是 |
| Motor imagery | fNIRS → SSM target | -0.0126 | [-0.0264, 0.0012] | 0/5 | 否 |
| Motor imagery | fNIRS shared → EEG swap | -0.0005 | [-0.0020, 0.0010] | 2/5 | 否 |
| Motor imagery | EEG shared → fNIRS swap | -0.0048 | [-0.0133, 0.0038] | 1/5 | 否 |
| Word generation | EEG → SSM target | 0.1087 | [0.0272, 0.1901] | 5/5 | 是 |
| Word generation | fNIRS → SSM target | -0.0139 | [-0.0335, 0.0057] | 1/5 | 否 |
| Word generation | fNIRS shared → EEG swap | 0.0000 | [-0.0002, 0.0002] | 2/5 | 否 |
| Word generation | EEG shared → fNIRS swap | -0.0112 | [-0.0274, 0.0050] | 1/5 | 否 |
| N-back | EEG → SSM target | 0.0916 | [-0.0017, 0.1848] | 5/5 | 否 |
| N-back | fNIRS → SSM target | -0.0280 | [-0.0784, 0.0224] | 1/5 | 否 |
| N-back | fNIRS shared → EEG swap | 0.0001 | [-0.0009, 0.0011] | 3/5 | 否 |
| N-back | EEG shared → fNIRS swap | 0.0073 | [-0.0025, 0.0171] | 4/5 | 否 |

EEG target 的 mean ΔR² 为 `0.0377–0.1087`，表明 EEG shared encoder 在部分
任务上能预测当前 SSM proxy。fNIRS target 的四个均值均为负
（`-0.0280–-0.0126`），不支持 bilateral proxy observability。所有 swap
效应均非常接近零（`-0.0112–0.0073`），且没有一个 simultaneous 下界大于零。

![16-cell primary sharedness result](../../experiments/runs/physiology_semantic_tokenizer/continuous_shared_private/20260819_continuous_shared_private_analysis_v1/figures/primary_16_cell_sharedness.svg)

## Raw reconstruction 消融

下表为 seed-averaged subject estimates 的等受试者均值；R² 基线是只由训练折
估计的 condition-by-relative-time mean。

| 任务 | 模态 | self | matched cross-modal | deranged cross-modal | private-only | shared-only |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mental arithmetic | EEG | 0.192 | 0.129 | 0.129 | 0.173 | -0.038 |
| Mental arithmetic | fNIRS | 0.745 | 0.395 | 0.392 | 0.616 | -0.061 |
| Motor imagery | EEG | 0.281 | 0.181 | 0.181 | 0.247 | -0.054 |
| Motor imagery | fNIRS | 0.831 | 0.424 | 0.427 | 0.661 | -0.061 |
| Word generation | EEG | 0.253 | 0.164 | 0.164 | 0.222 | -0.050 |
| Word generation | fNIRS | 0.886 | 0.448 | 0.454 | 0.693 | -0.088 |
| N-back | EEG | 0.192 | 0.134 | 0.134 | 0.169 | -0.048 |
| N-back | fNIRS | 0.847 | 0.462 | 0.458 | 0.683 | -0.061 |

同模态 self reconstruction 始终最好，private-only 保留了大部分重建能力；
shared-only 在八个 task/modality 单元中全部低于基线。matched 与 deranged 的
均值几乎相同。因此较高的 raw reconstruction 主要来自 modality-specific private
capacity，不能作为 shared pattern 匹配证据。

![Raw reconstruction ablation](../../experiments/runs/physiology_semantic_tokenizer/continuous_shared_private/20260819_continuous_shared_private_analysis_v1/figures/raw_reconstruction_ablation.svg)

## Latent 分离与几何诊断

32-component capacity-matched ridge probe 没有显示稳定的 shared/private
分离。EEG shared probe 的 task mean ΔR² 为 `-0.0139`、`0.0376`、`0.0637`
和 `0.0293`，EEG private 为 `0.0115`、`0.0093`、`0.0246` 和 `0.0054`；
fNIRS shared/private probe 在所有任务均为负。private probe 未被证明与 shared
等价，但这些结果也不支持 clean disentanglement。

matched 与 deranged EEG/fNIRS shared latent 的 linear CKA 都很低，任务均值
分别约为 `0.0070–0.0258` 与 `0.0076–0.0283`，不存在一致的 matched advantage。
effective rank、dimension variance 与 CKA 均只作描述性诊断，不替代主判据。

## 跨任务解释与局限

- Motor imagery 和 word generation 的 EEG target 单元为局部正证据，但它们
  不能补救 fNIRS observability 与 swap interchangeability 的失败；
- n-back 的五名受试者 EEG target point estimate 均为正，但 simultaneous
  下界仍略低于零，按预登记规则保持失败；
- SSM reliability 前置实验已显示 EEG-only fNIRS 恢复较弱。本实验继续发现
  fNIRS shared encoder 无法预测同一 proxy，因此结果可能同时反映 target
  偏向 EEG 与架构无法识别共享因素；
- 受试者数为每任务五名 development subjects，区间忠实反映这一限制；没有把
  window 或 seed 当作生物学重复；
- 没有打开 protected split，也没有把探索结果升级为总体人群或生理机制主张。

## 证据、图形与可复现性

完整运行：

- [`manifest.json`](../../experiments/runs/physiology_semantic_tokenizer/continuous_shared_private/20260819_continuous_shared_private_full_v1/manifest.json)，SHA-256
  `c780c1517bb86923729c70e9b2b59136a2e93e005c140ab22bff461e3343425f`；
- 12 个 task/seed checkpoint、validation prediction、latent export、loss curve、
  exact derangement registry 与 source/split/target hashes 均由该 manifest 绑定；
- smoke v2 已验证 serialization、checkpoint reload、derangement 与 analysis
  路径；首次 full staging 在样本数 preflight 处 fail closed，未启动训练并按证据
  保留。

后处理分析：

- [`manifest.json`](../../experiments/runs/physiology_semantic_tokenizer/continuous_shared_private/20260819_continuous_shared_private_analysis_v1/manifest.json)，SHA-256
  `3a322c81c03f860bc5cc934d70b2435517b4c351b8ed58a7bc941e0a11d15a08`；
- [`analysis_summary.md`](../../experiments/runs/physiology_semantic_tokenizer/continuous_shared_private/20260819_continuous_shared_private_analysis_v1/analysis_summary.md)，SHA-256
  `1685741b594e6f28c1b138ac5e5f137397f3d4a5e0038b96eb4e19c018c1a286`；
- 主结果、raw 消融、ridge probe、latent geometry、训练曲线和四类任务的代表性
  EEG/fNIRS pattern 均提供 SVG、PNG、alt text、精确 source table 与 figure
  provenance manifest。

## 对方法主线的含义

移除 VQ 后，信息瓶颈不再能解释 sharedness 失败；当前失败发生在 continuous
shared/private 架构及其 SSM target 本身。现有结果不支持继续进入 VQ/codebook
阶段。若继续该主线，应先提出能够区分“共同任务结构”“模态特异结构”和“真实
跨模态共享状态”的新构念，并在 development 范围内建立新的 identifiability
与负对照设计，再考虑任何离散化。
