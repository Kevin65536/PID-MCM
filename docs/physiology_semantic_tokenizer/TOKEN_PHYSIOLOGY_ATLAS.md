# Token Physiology Atlas：离散生理 token 的标准分析契约

_状态：development-only 的描述性与探索性分析工具；不是 token 生理状态命名器，也不是 protected-test 结果。_

Core 层级 12 张 summary figures 的逐图编码、单位、跨 split 比较方法与
常见误读见
[Token Physiology Atlas 可视化阅读指南](TOKEN_PHYSIOLOGY_ATLAS_VISUALIZATION_GUIDE.md)。

## 1. 目标与声明边界

Token Physiology Atlas 回答的是“哪些测量、分配与序列特征和某个 codebook 项共同出现”，而不是“该 token 就是什么生理状态”。所有 token ID 都是 **nominal category**：

- ID 的数值大小和相邻关系没有生理意义；
- codebook 重新初始化、排列或重新训练后，同一个数字 ID 不保证保留原语义；
- `token 17` 可以被描述为“在当前 development 数据、当前 checkpoint 与 support gate 下 alpha 相对功率较高”，但不能直接命名为“放松”“注意”或其它潜在状态；
- token 与已定义状态的关联仍是条件分布和不确定估计，不是身份等同或因果证据；
- EEG token 与 fNIRS token 即使 ID 相同，也不表示跨模态同义。

因此，Atlas 的基本叙事是 **token-conditioned measurement phenotype**。若未来要给 token 使用状态名称，必须另行定义构念、测量有效性、跨被试复现、跨 seed 稳定性、混杂排除和外部效标门；本工具不会自动跨过这些门。

## 2. 标准分析项目

每个模态分别生成以下分析，再在明确的对齐与 support 条件下生成跨模态结果。

| 项目 | 主要输出 | 解释边界 |
| --- | --- | --- |
| 支持度 | patch count、被试覆盖数、被试支配度、effective subjects、rare/unsupported 标记 | 先判断“是否有足够数据解释”，再展示 profile |
| 分配质量 | posterior entropy、top-1/top-2 margin、latent-to-code distance、hard/soft 差异 | 衡量量化分配是否明确，不等于生理可解释性 |
| 原始 patch 表型 | EEG 频谱与时域特征；fNIRS 局部形态特征；中位数、IQR、被试 bootstrap CI、标准化 enrichment | 描述 canonical 输入中的共同变化，不恢复物理电压或浓度单位 |
| 重建诊断 | expected、hard、semantic-only、residual-only 等 patch MSE | 模型依赖诊断，不能混作生理特征 |
| 条件关联 | 双向条件概率、lift、NMI、support | 关联不构成状态身份或因果解释 |
| nuisance/context audit | subject、dataset、task、label、anchor、token position 等条件关联 | 高关联可能表示身份、任务或采样位置泄漏 |
| 信息账本 | train-subject CV 后冻结到 validation，比较 continuous semantic、hard one-hot、posterior、hard/expected embedding、residual 与 semantic+residual 的生理特征 R² | 表示中可线性恢复的信息，不等于因果机制；validation 不参与超参数选择 |
| 稳定性 | train/val 或 seed 间的 support-gated phenotype signature matching | token 按 phenotype 匹配，不按数字 ID 匹配 |
| 序列 | occupancy、within-window transition、run length、dwell、order-0/1 validation log loss | 不跨 trial/window 拼接，不把重叠窗口当连续记录 |
| 跨模态时延 | 各 lag 的计数、MI/NMI、条件熵与 circular-shift null | `K×K` 稀疏单元必须连同 pair support 报告 |
| exemplars | 指向代表性 sample/patch 的引用和分配置信度 | 默认不复制原始 patch，避免重复数据与大体积输出 |

“状态”字段只接受研究者已明确定义的 operational category。仅为了让 token
看起来可命名而把同一连续特征切成 low/mid/high，会造成循环解释；连续测量
应优先留在 phenotype profile 中。默认每个 state 或 metadata 字段最多允许
64 个 category；超过上限的字段会在 manifest 中明确标记为 skipped，避免把
连续变量误当作类别后产生失控长表。

checkpoint 一键入口会读取 `associations.state_fields`，并要求这些字段在
每个 dataloader batch 中是 sample-aligned 或 token-aligned，再写入 assignment
export。若先单独运行底层 export CLI，需为相同字段重复传入
`--extra-field <name>`；分析已有 export 时，字段必须已经存在于 NPZ。

## 3. Patch 特征契约

### 3.1 共同输入

标准输入为 tokenizer 实际看到的 `canonical_robust_sd` 信号。幅值相关特征的单位因此是 canonical robust-SD，而不是 EEG 电压或血红蛋白浓度。特征保持 channel identity；如果 EEG channel 是按样本选择的局部通道，跨样本汇总不得把局部 channel index 误写成固定头皮位置。

无效或不可计算的特征用 `NaN + false validity mask` 表示，不能补零。特征 specification、采样率、patch 长度、channel 名和 feature hash 都必须进入 manifest。

### 3.2 EEG：200 Hz、2 s patch

默认 EEG patch 为 400 samples。频谱使用去均值、symmetric Hann window 的单边 periodogram；默认参考频带为 `[1, 45) Hz`，频带采用半开区间：

- delta `[1, 4) Hz`
- theta `[4, 8) Hz`
- alpha `[8, 13) Hz`
- beta `[13, 30) Hz`
- low-gamma `[30, 45) Hz`

每个频带同时报告 log absolute power 和相对参考频带 power，并补充 spectral entropy、peak frequency、mean、standard deviation、RMS、slope、endpoint delta、line length 和 Hjorth descriptors。图表与表格必须注明 absolute/relative、参考频带、变换和单位。

2 s 窗口只提供局部频谱描述；它不自动支持稳态、疾病或认知构念解释。工频、肌电、运动伪迹和 channel selection 仍需通过 nuisance/quality audit 解释。

### 3.3 fNIRS：10 Hz、2 s patch

默认 fNIRS patch 只有 20 samples。Atlas **禁止对该 2 s 局部 patch 报告 fNIRS band power**，因为该窗口不能分辨常见的慢血流动力学振荡。标准局部特征限于 mean、median、standard deviation、RMS、slope、endpoint delta、AUC 和 derivative spike，并分别保留 HbO/HbR channel identity。

对慢波、响应延迟、峰值时间或完整 HRF 形状的主张，必须使用单独的长时窗、明确的滤波与边界处理，不能由 2 s token patch 的局部形态外推。

## 4. 估计、support 与条件概率

### 4.1 Hard 与 soft profile

- **Hard profile** 使用 argmax code ID，回答实际离散序列中被分到该 token 的 patch 有什么表型。
- **Soft profile** 使用完整 posterior 权重，回答量化边界不确定性被保留时的加权表型。

二者均须输出，差异本身是诊断信号。若 posterior 缺失或无效，soft 结果应明确缺失，不能复制 hard 结果。

### 4.2 Subject-equal 汇总

统计单位是被试：先在每个 `subject × token` 内汇总，再让每个被试等权进入总体估计。默认 bootstrap 也以被试为重采样单位，报告 95% CI；patch 不能被当作彼此独立的 bootstrap 单位。

默认 support gate 为：

- `min_count = 30`
- `min_subjects = 5`

低于任一阈值的 token 保留在完整结果中，但标为 unsupported；图中用灰色/斜线而不是零值或空白表示。阈值是解释门，不是“显著性”门。

### 4.3 双向条件概率

对每一个经过定义的 category，至少同时报告：

\[
P(\text{category}\mid\text{token}), \qquad
P(\text{token}\mid\text{category})
\]

并报告边缘概率、lift、joint support 和 normalized mutual information。只报一个方向容易混淆“token 纯度”和“状态覆盖率”：某 token 内某 category 占比高，不代表该 category 主要由这个 token 表示。
每个方向同时保留 patch 加权估计与“先在被试内计算、再让被试等权”的
估计，并写出各方向实际贡献的被试数。前者可由 joint count 直接复算，
后者用于检查结果是否由 patch 较多的被试支配；两者都是描述性关联。

## 5. 序列与跨模态 null

transition、run 和 dwell 只在同一个记录窗口内计算；mask gap 会终止 run，样本边界不会产生 transition。order-1 是否增加信息，使用 train 拟合、validation 比较的 smoothed Markov log loss，而不是训练内拟合优度。

默认跨模态 lag 为 `[-2, -1, 0, 1, 2]` 个 token，即在 2 s patch 下为 `[-4, -2, 0, 2, 4] s`。lag 符号必须随输出 manifest 一起定义，不能只靠文件名推断。

默认 null 在每个 trial/window 内对整条 token 序列做非零 circular shift。该策略保留单模态局部自相关和每窗 occupancy；逐 token 独立 shuffle 会破坏这些结构，不属于 Atlas 的标准 null。`K=128` 时原始 pair table 有 16,384 个单元，任何 pair-level 解释都必须经过 support gate，不能从稀疏热图挑选故事。

## 6. 缓存、输出与可追溯性

Atlas 将数据分为两层缓存：

1. **measurement-feature cache v2**：cache key 直接哈希 cache schema、
   sample IDs、原始 patch 内容、`token_valid_mask`、采样率、
   feature-spec hash 和 channel identity，与 checkpoint 无关；只有这些
   内容完全相同的测量视图才可跨 checkpoint 复用。NPZ 明确保存
   `token_valid_mask`、token summary、channel-resolved features、feature/
   channel validity masks、sample IDs 和 EEG channel identity，不保存 raw
   patches。cache hit 会再次比较 assignment 与 cache 的 sample IDs、token
   grid、`token_valid_mask` 和 EEG channel identity。
2. **checkpoint-assignment cache**：目录层级依次包含 checkpoint SHA-256、
   canonical model-config SHA-256、Atlas analysis-view contract SHA-256
   （cache schema v2 + `input` + `features` + requested state fields）各自的
   前 24 个十六进制字符，
   再按 `full_split` 或 `first_<max_batches>_batches` replay scope 隔离。
   manifest 保存完整 checkpoint/config/analysis-view contract hash、
   `max_batches`、replay scope 和 assignment NPZ SHA-256。NPZ 保存 hard
   IDs、完整 posterior、semantic/quantized/expected embeddings、residual、分配/重建诊断和
   export metadata。checkpoint 一键流程先生成含 patch 的 v3 export，
   验证两侧 measurement cache 后再原位 compact；正常完成的 compact
   assignment NPZ 不含 `eeg_patches` 或 `fnirs_patches`。

compact assignment manifest 保存每个 measurement NPZ 的绝对路径、
measurement-cache key、NPZ SHA-256、feature-spec hash 和 source sample-order
hash；读取时逐项验证。因此 compact assignment **不是独立可搬运 artifact**：
归档、迁移或删除 cache 前必须连同被引用的 measurement NPZ 与 sidecar
处理，并保持 manifest 路径可解析。

当前 checkpoint cache reuse gate 会验证 checkpoint/config 完整 hash、
`max_batches`、split、assignment NPZ SHA-256、export 内部 sample-order
hash、必需数组和 analysis-view contract。对 compact cache，
它还会重新验证两侧 measurement references。config 内容、Atlas
input/features contract 或 smoke/full replay scope 改变时会进入不同目录，
不会再复用旧 assignment。

compact 前采用 fail-closed 验证：两侧 cache 必须是 v2，reference、sidecar
和实际 NPZ hash 必须一致，feature-spec 与 sample-order hash 必须一致；
cache 中的 sample IDs、token grid、`token_valid_mask` 以及逐样本 EEG
channel identity 必须与仍含 raw patch 的 assignment export 精确对齐。
任一检查失败时，不删除 raw patches，也不发布 compact artifact。

剩余边界是 cache reuse 不会重读 live dataset。若 config 内容不变，但同一
config 所引用的数据文件、预处理 cache 或 dataset 内容被原地替换，
checkpoint/config/contract/replay-scope identity 仍可能相同；此时必须使用
`--force` 重新 replay，或改用新的 `--measurement-cache-dir`，不能依赖旧
assignment 自动感知 live data drift。

已有 v3 export 模式是只读的：Atlas 会生成/复用 measurement cache，但不会
compact 用户提供的源 export，因此源文件原来若含 raw patches，仍会保留。
无论输入模式如何，Atlas output 自身不复制 raw patches；exemplar 只保存
`sample_id + patch_position + channel identity` 引用。YAML 的
`cache.store_raw_patches: false` 描述成功完成后的 steady-state cache，
不表示 checkpoint replay 的中间 export 从未把 patch 写入磁盘。

标准持久化格式为：

- CSV：长表结果，适合审计和下游统计；
- JSON：summary、manifest、schema、阈值、缺失/跳过原因和 provenance；
- Atlas output NPZ：within-window transition 与跨模态 lag count matrices；
- measurement-cache NPZ：summary/channel features、masks 和 identity；
- assignment-cache/export NPZ：posterior、latent、embeddings、residual、
  diagnostics 与 metadata；compact 版本不含 raw patches；
- PNG/PDF/SVG：紧凑 summary figures。

`token_analysis_manifests.json` 按 split 与 modality 保存有效/排除数量、
feature names、support/bootstrap 契约、state/metadata 字段以及因 category
过多而 skipped 的字段；这些跳过信息不能只留在运行日志中。

其中 `tables/token_feature_distributions.csv` 保存 hard-token 对应 patch
特征的 mean、SD、5/25/50/75/95% 分位数；这些是 patch 层面的描述性分布。
`tables/token_channel_feature_distributions.csv` 使用真实的逐样本 EEG
channel name（或固定 HbO/HbR role）保存 channel-resolved 分布，不能把
“local channel 0”误写成固定头皮位置。
`tables/token_profiles.csv` 则保存被试等权估计与被试 bootstrap，两者不得
混用为同一种不确定性。`tables/hard_soft_profile_differences.csv` 单独量化
soft 与 hard profile 的差异，作为 assignment-boundary sensitivity，而不是
生理状态“分歧”。

每张图同时生成 manifest sidecar 和 alt-text sidecar。manifest 至少记录输入 artifact/hash、分析 split、support threshold、单位/变换、生成时间、软件版本和输出 hash。缺失与 insufficient support 必须采用不同图形编码，profile heatmap 使用以零为中心的对称色标。写出采用 atomic publish，默认拒绝覆盖已有 artifact。
heatmap sidecar 还保存实际入图 token ID、排序规则和色标上下限；codebook
sidecar 保存 embedding shape、PCA 方法、着色字段和色标，PCA 坐标轴直接
标出各轴解释的 codebook variance。

## 7. Split 与 protected-test policy

默认且可自动运行的 split 只有 `train` 和 `val`。protected test 不会因 YAML 中出现名称而自动打开；必须在 CLI 上给出独立、显式的 `--allow-test` 授权，并在顶层 manifest 留下 `protected_test_opened: true`。

训练完成后的常规分析不得把 test 加入 profile、阈值选择、token matching、figure 排序或 narrative。若未来获得正式授权，test 只执行已冻结的分析契约。

## 8. 自动化成本分层

以下预算以当前 formal 规模约 13,800 patches/模态、EEG `6×400`
samples、fNIRS `2×20` samples、`K=128`、embedding `D=64`，以及
EEG/fNIRS residual 分别为 64/32 维为量级假设。数值是按 dtype 与 shape
计算的未压缩 payload 预算，不是实测磁盘大小；NPZ 压缩率、metadata、
optional teacher arrays、batch size 和硬件都会改变实际空间与时间，manifest
应记录实测 wall time 和 artifact bytes。

| 层级 | 内容 | 主要复杂度与存储 | 建议 |
| --- | --- | --- | --- |
| Core / cache hit | support、hard/soft profile、patch/channel 分布、assignment/nuisance audit、基本 within-window sequence、紧凑图 | 以 `O(NK + NF)` 表格归约为主；channel-resolved CSV 依真实 channel identity 约可增加 10–50 MiB，无 model forward | 每次 tokenizer 训练成功后自动运行；存储紧张时定期归档长表 |
| Core / cache miss | 每个 requested split 各一次 deterministic checkpoint replay，加 EEG rFFT 与 fNIRS morphology，随后 compact assignment | EEG 特征约 `O(NC P log P)`；两模态 measurement cache 的 feature/summary/mask 数值约 13.0 MiB，NPZ 压缩后依数据而定 | 每个精确 measurement view 首次自动运行；随后按 content hash 复用 |
| Statistical | grouped-ridge 信息账本、1,000 次 subject bootstrap、train/val 或 seed signature matching | ridge 成本随表示维度增长；bootstrap 随被试数、token 数、feature 数和迭代数线性增长，通常为秒到分钟级 CPU 工作 | final checkpoint 由外部 scheduler 异步运行；smoke 可降到 200 |
| Null / coupling | 5 个 lag × 200 次 whole-window circular-shift null | 约 1,000 次 `O(NT)` pair recount；常成为后处理中的主要 CPU 成本 | 由外部 scheduler nightly 运行，或对候选 checkpoint 显式按需运行 |
| Compact assignment（所有 tier） | 完整 posterior、semantic/quantized/expected embeddings、residual、hard IDs 与 diagnostics | posterior + semantic latent 约 20.2 MiB；连同其余必需数值数组约 39.8 MiB，再加 metadata/optional teacher arrays；这些不是 `full` tier 才保存 | 每个 checkpoint/split 必需；依赖共享 measurement cache |
| Replay 临时峰值 | compact 前的 EEG/fNIRS raw patches | raw patch 未压缩约 128.4 MiB；checkpoint 流程成功后删除，异常中断时可能残留含 patch export | 为自动任务预留峰值空间，并审计未完成 cache |
| Multi-seed suite | 所有 checkpoint 的 assignment、matching、bootstrap/null | checkpoint 数近似线性扩展；measurement cache 可共享 | 模型选择完成后批量运行 |

因此，适合“每次训练结束自动进行”的是 Core。CLI 本身同步执行：
`--tier statistical` 和 `--tier full` 不会创建后台任务；1,000 次 bootstrap
与 200 次 coupling null 若要异步/nightly 运行，必须由训练器或外部 job
scheduler 另行调度。YAML 中 `bootstrap_mode`、`coupling_null_mode` 与
`full_vector_export_mode` 是 orchestration policy，其中
`full_vector_export_mode` 目前是预留项；当前 compact assignment 的完整
posterior/latent 表示集合在所有 tier 都固定保存，没有单独的 vector-export
开关。单次 CLI run 只自动生成该 run 的 train/val stability matching；
multi-seed 比较需要对各 checkpoint 分别生成 Atlas，再由外部 suite 调用
signature-matching API 汇总。

## 9. 默认配置与运行入口

版本化默认配置位于：

```text
experiments/configs/physiology_semantic_tokenizer/token_physiology_atlas.yaml
```

分析已有 v3 export、不重放 checkpoint：

```bash
.venv/bin/python experiments/scripts/analyze_token_physiology_atlas.py \
  --atlas-config experiments/configs/physiology_semantic_tokenizer/token_physiology_atlas.yaml \
  --export train=<exports>/train.npz \
  --export val=<exports>/val.npz \
  --output-dir <run>/analysis/token_physiology_atlas
```

从 checkpoint 一键 deterministic replay 与分析：

```bash
.venv/bin/python experiments/scripts/analyze_token_physiology_atlas.py \
  --atlas-config experiments/configs/physiology_semantic_tokenizer/token_physiology_atlas.yaml \
  --checkpoint <run>/checkpoints/best.pt \
  --model-config <run>/config.yaml \
  --splits train,val \
  --measurement-cache-dir <shared-cache>/token_physiology_measurements \
  --output-dir <run>/analysis/token_physiology_atlas
```

CLI 默认 `--tier core`，会自动把 bootstrap 和 coupling null 迭代数设为
0、跳过 grouped-ridge 信息账本并只输出 PNG，适合每次训练结束后同步运行。
`--tier statistical` 启用 YAML 中的 subject bootstrap、信息账本和完整图格式，
`--tier full` 再启用 circular-shift null；Core 可用 `--information-ledger`
单独启用账本，任一层级都可用 `--bootstrap-iterations` 与
`--coupling-permutations` 显式覆盖。

`--model-config` 只在 checkpoint 没有嵌入完整训练 config 时需要。`--export SPLIT=PATH` 可重复提供，且与 `--checkpoint` 二选一。smoke 可用 `--max-batches`、`--bootstrap-iterations` 和 `--coupling-permutations` 降低成本；图格式可由 `--formats` 覆盖，`--no-plots` 可跳过绘图。CLI 的 `--help` 是参数名称的运行时权威；YAML 是统计与输出默认值的版本化契约。重复运行应选择新的 output directory，除非显式使用 `--force`；protected test 仍另需 `--allow-test`。

Python API 的标准入口是：

- `src.analysis.physiological_patch_features`
- `src.analysis.token_physiology`
- `src.analysis.token_information_ledger`
- `src.analysis.token_physiology_atlas`
- `src.analysis.token_sequence`
- `src.visualization.token_physiology_plots`

## 10. 旧工具映射

旧文件为历史复现保留，不在 import 时发出 warning，也不改变旧 run 的计算。

| 旧入口 | 保留用途 | 新标准入口 |
| --- | --- | --- |
| `coupling_identifiability.patch_features` / `patch_features_torch` | 复现旧 coupling suite 的最小、扁平特征 | `extract_eeg_patch_features` / `extract_fnirs_patch_features`：versioned、mask-aware、channel-resolved |
| `coupling_identifiability.build_lag_pair_table` 与旧 suite shuffle | 精确重放历史非负 lag 结果 | `token_sequence.analyze_cross_modal_lags`：双向 lag、within-window boundary、whole-window circular-shift null |
| `coupling_identifiability.conditional_probabilities_from_counts` | 重放旧 residual-coupling 统计 | `token_physiology.analyze_token_physiology`：token/category 双向概率、lift、NMI、support |
| `tokenizer_plots.TokenizerVisualizer` | 旧训练曲线、重建、usage、embedding dashboard | `token_physiology_plots`：support/missing-aware Atlas figures 与 atomic sidecars |
| 旧 monolithic/sharded token export | 历史 artifact 消费 | v3 export + measurement/assignment 双层 cache；新分析不复制 raw patch |
| 数字 ID 对齐 | 仅同一冻结 artifact 内查看 | `match_token_signatures`：经过 support gate 的 phenotype matching |

迁移原则是“历史结果不改写，新结果只走标准入口”。旧 visualizer 仍可用于训练过程监控，但不能代替 Atlas 的被试等权估计、support gate、缺失显示和 provenance sidecar。

## 11. 最终人工检查

自动生成不等于科学验证。纳入报告前至少确认：

- sample/split/hash 与预期一致，protected test 未被意外打开；
- channel identity、采样率、单位、mask 和 feature specification 正确；
- unsupported、missing、hard/soft 分歧和负结果均未被隐藏；
- conditional probability 的两个方向、分母和 support 都可复算；
- 图的色标、单位、CI 定义、alt text 与源表一致；
- 任何文字标签都保持描述性，没有把关联升级为生理状态身份或因果结论。
