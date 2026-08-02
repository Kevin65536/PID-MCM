# Adapter 对齐门控 v2

_生效日期：2026-07-31。适用于尚未冻结的新对比单元；不回开已经完成的
protected evaluation，也不改变正在执行的冻结协议。机器合同见
[`adapter_alignment_gate_contract_v2.yaml`](adapter_alignment_gate_contract_v2.yaml)。_

## 结论

对比方法不需要在 adapter 输出端变成同一种 tensor；需要严格对齐的是 adapter
输入端所代表的**科学信息预算**。主对比表采用 `support_matched_direct`：同一
sample、split、target、时间锚点、每模态观测区间、真实模态、真实通道集合、记录
支持 mask 和 canonical signal branch。进入 adapter 后，方法可以保留原生的
patch/token、通道顺序、几何编码、池化和源方法声明的固定变换，但必须可追溯、
不可按目标任务分数选择，并服从训练分区拟合边界。

这一定义解决两个相反风险：只对齐 shape 会把不同长度上下文或伪造通道藏起来；
把所有上游预处理和 tokenization 强制做成相同又会破坏 named method 的复现边界。

## 新方法严格串行落地

从本版本开始只允许一个 active delivery method。串行范围包括 adapter 实现与审核、
public preflight/development、协议冻结和该新方法的正式执行；当前方法未完成前，不
提前实现或试跑队列中的下一种方法。

协议生效前已经运行的 EFRM LODO v2 冻结训练可以继续，但它是不可修改的后台协议，
不阻塞新方法代码落地。STA-Net 同样只保留已完成结果。新的实现顺序为 BIOT、
CBraMod、REVE、BrainFusion、NormWear。BIOT、CBraMod、REVE 和 BrainFusion 均已
完成各自 public delivery，protected 仍分别锁定；当前 active method 是 NormWear。
若当前方法 blocked，则暂停新方法队列处理 blocker，不能静默跳过。任何延后的
protected 执行都不得与 active delivery method 并发。

## 数据集特征对门控的约束

| 数据面 | 当前事实 | 对齐决定 |
| --- | --- | --- |
| 采样与数值坐标 | loader 输出 EEG 200 Hz、fNIRS 10 Hz；EEG 为 1–45 Hz 后的 record-wise robust coordinate；fNIRS 原生物理语义不同 | direct profile 从同一 canonical array 和 branch hash 开始；native unit/transform provenance 仍必须保留 |
| Single-Trial | 29 人、30 EEG；36 个双波长位置先经 OD/MBLL 得到 HbO/HbR | 禁止把 wavelength 直接重命名成 HbO/HbR；主窗口为 registry 起点后的 8 s |
| Simultaneous | 28 个 admitted scalp EEG；DSR 为 25 人/8,980 个 EEG-native Go/No-go event，fNIRS 只有 block anchor | WG/n-back 主窗口 8 s；DSR 为 EEG 2 s + 同步 fNIRS 2 s context，不能声称 event-level fNIRS response |
| Visual | 16 人；3 s stimulus + 9 s decision；几何为 graphical template | 主窗口保持现有 8 s offline estimand；模板坐标不能表述成个体 digitization |
| REFED | 64 EEG、51 fNIRS locations；20 s `[2,T]` 动态目标；terminal signal/target support 可部分缺失 | `valid_mask` 与 `target_valid_mask` 必须分别传播；无 truthful time-mask 能力的方法将整个 cell 事前标为 unsupported，不允许按方法删样本 |
| 通道容量 | BIOT 主 checkpoint 固定 16 channels，而其他 EEG encoder 可接收更多 | EEG foundation 主表冻结每任务同一组 16 个真实通道；全通道结果只能进入 `native_capacity_secondary` |
| 几何语义 | REVE 使用官方 name-to-position bank；STA-Net 使用 source grid 或 unified template；其余方法主要由顺序表达通道 | direct profile 对齐通道**身份集合**，不强制相同数值坐标表达；每个方法的 delivered order 和 geometry provenance 分别哈希 |

上述“主窗口”延续当前 method-neutral registry 的 offline estimand，而不是宣称它是
唯一生理学最优窗口。若要比较 HRF-lag/NVC 方法，应新建一个所有参比方法都使用
相同 fNIRS relative interval 的 `hemodynamic_context_matched` 协议版本；不能把每个
方法各自最有利的时间上下文放进同一直接排名。

## 到底对齐到什么程度

### 必须完全相同：否则不是直接比较

- dataset/task/sample identity，以及 subject、record、event/window 的依赖关系；
- public/protected split fingerprint、target schema、target mask、endpoint 和聚合规则；
- observation anchor 以及 EEG、fNIRS 各自的 relative interval；
- 模态身份、真实 measured channel identity set、recorded support mask；
- canonical signal branch、rate、component role 和生成该 array 的 transform identity；
- 每个 method × dataset cell 的预训练语料暴露类别。

任一项不同，结果必须进入另一个 alignment profile，不能继续做 paired difference
或合并成同一 ranking。

### 可以方法原生：属于算法而不是信息预算

- patch 长度、overlap、token grid 和时频变换；
- method-native delivered channel order，但共享 channel set 与 delivered order 都要
  单独保存 hash；
- REVE position bank、STA-Net grid projection 等几何编码；
- token pooling、attention pooling、head interface；
- 上游明确要求的、无标签且不可调的 per-window/stateless transform；
- 只在 outer-training 拟合并保存 state hash 的 feature/target scaler。

允许这些差异不等于允许任意适配。它们必须来自固定 source-fidelity 决策，而不是
看过目标分数后选择；输出层取错、test-wide normalization、未记录的插值或删样本
都会阻断门控。

## 三类结果口径

| Profile | 可以直接排名 | 用途 |
| --- | --- | --- |
| `support_matched_direct` | 是 | 主表；estimand 与 observation/support information budget 完全相同 |
| `native_capacity_secondary` | 否 | 允许方法消费不同数量的真实通道，回答部署容量而非纯表征对比 |
| `method_native_context_reference` | 否 | 保存已有/源方法时间上下文结果，只作背景或方法内结果 |

训练范式仍需分表。即使都通过 `support_matched_direct`，frozen linear probe、
supervised end-to-end、target-overlap 和 independent reimplementation 也不能合成一个
foundation-model 排名。

## A0–A8 门控

门控的判定单位是 `method × task × track × alignment_profile`，不是整个方法。

| 门 | 问题 | 最低证据与通过标准 |
| --- | --- | --- |
| A0 cell registration | 比较单元是否在看分数前定义 | 登记 track/profile/exposure；每个计划单元预先是 supported 或有冻结的 unsupported 理由 |
| A1 estimand identity | 是否回答同一个统计问题 | inventory、split、target、metric、aggregation fingerprints 与 profile 一致；禁止 method-specific deletion |
| A2 observation budget | 是否消费同一时空上下文 | 明确每模态 anchor/interval 和 offline/causal 语义；direct profile 必须完全相同 |
| A3 measurement coordinate | 是否从同一测量分支出发 | branch/rate/unit/component/transform hashes 完整；同一 canonical arrays 到达 direct adapters |
| A4 measured support & masks | 是否保持真实信息支持 | 全量 public unique inventory；模态/通道集合与 profile 一致；四类 mask 不混写；无复制、生成或镜像 |
| A5 adapter semantics | 模型实际接收和输出什么 | order、shape、output layer、tokenization、geometry、pooling、fit scope、trainable boundary 和 replay identity 全部显式 |
| A6 fidelity & exposure | 还是不是所命名的方法/轨道 | checkpoint、上游代码、trusted code、position bank 等 hash 完整；deviation 和 per-cell exposure 明确 |
| A7 production replay | production path 是否覆盖真实 public 数据 | 全部 unique public samples 无截断通过；feature/input finite、非异常常量；cache key 覆盖所有语义输入 |
| A8 freeze & unlock | 是否可以进入正式执行 | 所有计划 cell 有终态；job/retry/failure rules 冻结；protected 仍不可解引用；信息预算改变必须升版本 |

状态不能只用 pass/pending。`unsupported` 是在看分数前确认无法忠实支持某单元的
合法终态；`blocked` 表示证据或实现仍缺失；`fail` 表示已观察到合同违规。synthetic
或 public mini 只能证明连通性，不能提升 A4/A7 的 full-coverage 结论。

## 当前 adapter 的迁移判定

这是基于仓库代码和现有 public 证据的审计结论，不是新的正式结果。

| 方法 | v2 判定 | 需要先处理的事项 |
| --- | --- | --- |
| BIOT | 六个分类 cell 以 `public_complete` 通过 A0–A8；REFED v1 unsupported；protected locked | 22,442 个唯一公开样本和 90 个串行 public selection/refit jobs 已全部审核通过，失败/重试均为 0；public delivery 已完成并晋级 CBraMod，BIOT protected 仍需独立授权 |
| CBraMod | blocked | 当前 adapter 直接执行完整 encoder 后 mean pool；上游 quick example 和 downstream modules 先将 `proj_out` 替换为 `Identity`。必须先固定实际 representation layer，再做全量覆盖 |
| REVE | blocked；Single-Trial 两任务为 overlap track；REFED v1 unsupported | cache/identity 需包含 position bank、trusted code 与实际模型代码 hashes；完成全 public name-to-position 覆盖 |
| NormWear | 六个分类 cell 以 adapted 名称和 `public_complete` 通过 A0–A8；REFED unsupported；protected locked | 22,442 个公开输入的 production replay 与 90 个串行 public selection/refit jobs 全部审核通过，失败/重试均为 0；新方法交付队列已完成 |
| EFRM | 现有冻结协议继续；新 direct table 为 pending | 当前 observation budget 是分类 8/8 s、DSR 2/2 s、REFED 20/20 s；补齐 v2 evidence 后可作为 synchronous profile 的基准 |
| BrainFusion | blocked | 先冻结 NVC/HRF observation interval；NVC/CSP/selection/stacking 必须全部 fold-local，不能用 source-native context 自动进入 direct 表 |
| STA-Net | 完成结果保留，但当前归 `method_native_context_reference` | 默认分类实际为 EEG 3 s + fNIRS 13 s，DSR 为 2 s + 13 s，与 synchronous profile 不同；不回写既有 protected 结果。如需 direct profile，必须新版本、重新冻结并独立授权 |

因此，旧 manifest 中的 `B1_input_contract: pass...` 只能保留为 legacy software
状态，不能自动等价为 v2 A0–A8 通过。当前 `public_preflight_stage1.json` 的
`status=partial_started`、`max_records_per_task=5` 也不能支持 full-coverage 声明。

## 每个 cell 必须保留的证据包

```text
cell declaration + alignment profile + exposure class
sample/split/target/observation/channel-set fingerprints
canonical branch + transform + mask summaries
adapter source/output-layer/delivered-order/geometry identities
checkpoint + upstream/trusted-code/auxiliary-asset identities
full-public coverage report + retained failures
feature/input cache identity + replay result
freeze/unlock manifest
```

只有 A0–A8 在相应 evidence scope 下满足，才能开始新的 protected evaluation。
已有冻结协议按自己的边界继续运行；v2 不以重新查看结果为代价“修复”旧比较。

合同和后续 cell evidence 可用只读审计器检查：

```bash
.venv/bin/python comparative_methods/audit_adapter_alignment.py \
  path/to/cell_a.yaml path/to/cell_b.yaml
```

不传 cell 文件时只校验 v2 合同本身。对 `support_matched_direct`，审计器还会按
`comparison_group_id` 检查所有 `exact_equal_fields`；public-mini evidence 不能把
A4/A7/A8 标为 pass，protected evidence 也不能反向修补这些门。
