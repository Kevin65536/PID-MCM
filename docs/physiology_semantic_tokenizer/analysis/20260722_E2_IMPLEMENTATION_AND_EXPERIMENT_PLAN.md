# E2 语义 token 实现、实验与决策

_2026-07-22 至 2026-07-23 · development-only · protected subjects 24–29 closed_

## 结论

E1 入选的 K=128 `diverse_farthest + T2/T2 balance` 量化器配置已经完整
继承到 E2。Single-Trial EEG v4 工频处理与全零 bad-channel mask 契约生效
后，adaptive E0 教师在同源数据上重跑，local target observability、gauge
与 K=128 transmissibility 通过；sidecar 从历史的 `93/230` 恢复为
`230/230` target trial 接纳。保护被试 24–29 全程关闭。

训练梯度校准最终冻结 state/prototype 权重 `0.005`。三行三 seed 共 9 个
正式 development run 均完成 462 steps，梯度 allowlist、E1 码本健康、
复活停止和 checkpoint 哈希契约全部通过。冻结评估不支持 semantic row：
T1 相对 seed-matched T0 的两模态 required hard-token 主终点变化为
`-0.0271/-0.0413/+0.0065`，T2 为
`-0.0343/-0.0560/-0.0324`；EEG hard-token R² 在所有 run 中均为负且
未超过 shuffled-target q95。正式决策是
`no_semantic_row_admitted_retain_T0`，不接纳 T1/T2，也不产生 G3、
E6/G2 或耦合声明。

## 已实现代码闭环

| 环节 | 实现 | 固定约束 |
| --- | --- | --- |
| Sidecar schema/join | `src/data/physiology_semantic_targets.py`、`UnifiedPhysiologyLocalViewDataset` | 以 anchor-independent measured sample key 连接；sidecar 不提供测量信号 |
| Target 构建 | `experiments/build_adaptive_teacher_sidecar.py` | 仅 01–23 号开发被试；train-subject-only 标准化；不使用 posterior uncertainty 加权 |
| 开发门禁 | `physiology_semantic_target_family_gate_v1` | 哈希绑定 split、measured cache、sidecar manifest；禁止 promotion 和 protected test |
| T0/T1/T2 | `e2_semantic_objective_suite.yaml`、`launch_e2_semantic_objective_suite.py` | 数据、K、latent、优化器、训练步数、E1 quantizer 设置完全匹配 |
| 梯度审计 | trainer 的 `gradient_entry_audit.json` | EEG objective 禁入 fNIRS branch，反之亦然；记录 norm 与 cosine conflict |
| 冻结探针 | `evaluate_e2_semantic_tokens.py` | train-subject grouped ridge；连续 latent、hard ID、posterior、saved codebook vector 分开评估 |
| 原型证据 | `semantic_token_evaluation.py` | train-only signature，Hungarian 跨 seed 匹配，不比较原始 token 编号 |
| 导出 | `export_physiology_semantic_tokens.py` v2 | 统一 sample ID、token mask、target/mask、continuous/hard/posterior/codebook/residual |

正式 suite 的 9 个 run 均保存 best/last checkpoint、implementation
snapshot、训练与验证轨迹、量化器健康以及前四步 objective 梯度审计。
冻结评估生成 `state_decoding.json`、`prototype_signatures.csv`、
`prototype_stability.json`、`objective_ablation.csv` 和
`gradient_entry_audit.json`；配对决策工具另生成 `decision.json`、
`paired_subject_deltas.csv` 与 `coordinate_deltas.csv`。

## 当前 sidecar 审计

| 范围 | source trial | 当前通道契约接纳 | 排除 |
| --- | ---: | ---: | ---: |
| Train subjects 01–18 | 180 | 180 | 0 |
| Validation subjects 19–23 | 50 | 50 | 0 |
| 合计 | 230 | 230 | 0 |

接纳率为 `100%`，覆盖全部 23 名开发被试。sidecar 仍保留严格的通道审计
与回退/拒绝实现；当前 v4 measured contract 下没有 target 因 bad channel
被排除。180 个 train target trial 只用于训练集 target 标准化。

## T0/T1/T2 固定设计

公共设置：Single-Trial mental-arithmetic namespace；18/5/6 subject split；20 s window（事件前 5 s 至后 15 s）；2 s patch；K=128、D=64；E1 入选的 cosine assignment、normalized latents、first-batch K-means、bounded diverse-farthest revival、balance weight `0.16`、temperature `2.0`、annealed-hard semantic-only reconstruction；三 seeds `20260719/20260720/20260721`。

| Row | Local/prototype objective | EEG coordinates | fNIRS coordinates |
| --- | --- | --- | --- |
| T0 | 无 teacher loss；仍加载同一 sidecar 以保持局部 view 完全匹配 | — | — |
| T1 | state `0.005` + prototype `0.005` | `r_mean`, `r_slope` | HbO/HbR mean 与 slope |
| T2 | state `0.005` + prototype `0.005` | T1 + `s_mean`, `s_slope` | 与 T1 相同 |

`0.25` 是最初的 integration candidate，不是冻结科学超参数。重建
channel-aware E0 后先在训练被试上运行初始候选
`{0.1, 0.25, 0.5}`；它们均因梯度数量级失衡失败，随后按下方、在查看
validation target decoding 之前登记的修订网格完成校准。最终正式权重
冻结为 `0.005`；查看 validation target endpoint 后不得再改权重。

## 执行阶段

### A. 已完成：软件闭环

1. 从旧 E0-v3 轨迹构建严格过滤 sidecar。
2. T0/T1/T2 CPU dry-run。
3. T1 4-step CUDA optimizer/gradient smoke。
4. 单 run 冻结评估 smoke。

此阶段只接受 tensor、mask、hash、梯度和 artifact correctness，不报告 semantic performance。

### B. 已完成：v4 同源 E0 重验证

1. adaptive SSM 的局部 EEG/fNIRS 选择必须直接消费当前 unified sample 的 bad-channel mask。
2. 每个 target 的 selected channel names、sample key、QC mask 和 gauge provenance 必须在生成时冻结。
3. 重新运行 required `r`、HbO/HbR 与 optional `s` 的 train/validation observability、K=128 transmissibility 和 target coverage。
4. 要么获得预注册开发样本的完整覆盖，要么在看 E2 validation 前冻结一个有科学理由的排除规则并重新校准 E0；不得沿用当前 93-sample 子集直接做正式 E2。

结果为 `230/230` 完整覆盖；required `r`、HbO/HbR 与 optional `s` 的
local target 均通过既定可观测性，EEG/fNIRS K=128 target vocabulary
transmissibility 均超过 random reference。经 observation-aligned
符号校准后，adaptive SSM physical teacher 完整通过 E0，SSM 获取的生理
信息（包括 fNIRS 内容）完全可接受。旧的 fNIRS physical reconstruction
与 HbR posterior calibration 负标记属于符号校准前诊断，不再构成 E0
失败。E2 只使用预登记 local/prototype family 是本轮实验范围选择，不是
E0 对 physical teacher 的限制。

### C. 训练被试权重校准

在一个 seed 上只用 subjects 01–18 比较 semantic weight `{0.1,0.25,0.5}`。固定一个权重后重新生成 suite protocol；该阶段不使用 subjects 19–23 的 target decoding 选择权重。

**2026-07-23、查看任何 E2 validation target decoding 之前登记的校准修订。**
原候选的四步训练梯度审计均通过 entry allowlist，但最小权重 `0.1`
下，EEG state 对共享编码干路的 batch-mean 梯度范数仍约为
reconstruction 的 `189` 倍；`0.25/0.5` 近似按权重线性放大。因此原
候选集没有满足“不发生数量级失衡”的候选，不能直接启动正式 suite。
追加纯训练梯度尺度候选 `{0.0025,0.005,0.01}`；用同一 seed、同一四个
训练 batch，计算 state/prototype 相对同模态 reconstruction 的共享干路
梯度范数比。候选要求四类 objective 的 batch-median 比值全部落在
`[0.1,10]`；若多个候选通过，选择最大绝对 `log10` 比值最小者，再以较小
权重打破平局。校准脚本不得读取 validation target decoding，保护测试仍
保持关闭。

校准结果：`0.0025` 与 `0.005` 通过，`0.01` 因 EEG state 的
batch-median 比值为 `18.91` 而失败。`0.005` 的 EEG state、EEG
prototype、fNIRS state、fNIRS prototype 中位比值依次为
`9.49/1.87/0.80/0.20`，最坏绝对 `log10` 比值小于 `0.0025`，因此按
预登记 tie-break 冻结 `0.005`。

### D. 三 seed E2 development suite

按 seed-matched T0/T1/T2 运行 9 个 full-development jobs。每个 run 必须通过：E1 codebook health 继续保持、target coverage 与 frozen sidecar 一致、梯度 allowlist 无违反、训练后不发生未注册 mass revival、checkpoint 和 implementation hash 完整。

9 个 run 均完成上述契约。T0/T1/T2 的最终 EEG 有效码字范围分别为
`68.80–70.40`、`88.06–90.98`、`80.88–88.19`；fNIRS 分别为
`45.34–55.60`、`44.03–52.00`、`45.20–52.63`。所有 run 在 retention
window 内 total revival 保持常数，末轮 revival 为 0。

### E. 冻结评估与决策

主终点始终是 required signature family 在 hard token one-hot probe 上的 subject-held-out mean R²；T2 不通过增加 `s` 坐标改变主终点。`s_mean/s_slope` 是单独的 optional endpoint，用于决定 T2 是否优于 T1。连续 latent、posterior 和 codebook vector 是解释性并列表示。

E2 development 支持 T1/T2 的必要条件：

1. 相对 seed-matched T0，required hard-token endpoint 在 validation subjects 上方向一致改善，并给出 subject bootstrap uncertainty；
2. hard-token observed endpoint 高于训练 target permutation null q95；
3. improvement 不是单一 seed 或单一 coordinate 驱动；
4. signature-matched prototype stability 不下降，且 codebook health 仍在 E1 calibration 范围内；
5. T2 只有在 optional `s` endpoint 相对 T1 有稳定增益且 required endpoint 不下降时才入选；否则选择 T1；
6. E6/G2 information-retention gate 尚未通过时，只能记录“E2 development semantic evidence”，不能宣布 G3 promotion，更不能进入 E7 coupling-preservation。

正式结果不满足第 1、2、4、5 条：

- T1 的 seed-matched 两模态平均主终点变化为
  `-0.0271/-0.0413/+0.0065`；配对受试者固定-seed均值为 `-0.0326`，
  95% bootstrap CI `[-0.0770, 0.0042]`。
- T2 的对应变化为 `-0.0343/-0.0560/-0.0324`；配对受试者均值为
  `-0.0575`，CI `[-0.1107, -0.0147]`。
- fNIRS hard-token endpoint 在各行均高于 null，但 EEG 在 9 个 run 中
  均未高于 null，说明 fNIRS 的可解码性不是 teacher row 新增的证据。
- T2 相对 T1 的 optional `s` pooled 变化为
  `+0.0128/-0.0070/-0.0041`，且 required endpoint 三 seed 均下降。
- EEG signature-matched stability 从 T0 的 `0.9184` 提高至
  T1/T2 的 `0.9348/0.9399`，但 fNIRS 从 `0.9318` 降至
  `0.9226/0.9247`；量化器健康本身在全部 run 中通过。

## 运行命令

先重建 sidecar（当前命令会显式保留 channel-audit 排除记录）：

```bash
PYTHONPATH=. .venv/bin/python experiments/build_adaptive_teacher_sidecar.py \
  --source-run experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260723_adaptive_teacher_e0_v3_line_clean_v4_revalidation_v1 \
  --e2-config experiments/configs/physiology_semantic_tokenizer/e2_semantic_objective_suite.yaml \
  --output-dir data/cache/physiology_semantic_targets_v1/adaptive_ssm_e2_development
```

生成三行、三 seed 的冻结配置和 suite protocol：

```bash
PYTHONPATH=. .venv/bin/python experiments/launch_e2_semantic_objective_suite.py
```

完成前置修复和权重冻结后才执行训练：

```bash
PYTHONPATH=. .venv/bin/python experiments/launch_e2_semantic_objective_suite.py \
  --execute --device cuda:0
```

训练完成后，把 9 个 run directory 逐个作为 `--run` 传入：

```bash
PYTHONPATH=. .venv/bin/python experiments/evaluate_e2_semantic_tokens.py \
  --run <T0-seed1-run> --run <T1-seed1-run> --run <T2-seed1-run> \
  --run <remaining-six-runs> \
  --output-dir experiments/runs/physiology_semantic_tokenizer/e2_semantic_objectives/<suite>/evaluation \
  --device cuda:0
```

当前环境没有 `pyarrow`，因此 evaluation manifest 明确记录
`prototype_parquet_written=false`。主结果、逐 run 探针、配对 bootstrap
和决策均保存在 JSON/CSV；没有把缺失的 Parquet 伪装成已生成产物。

## Claim boundary

允许的结论是：v4 同源 E0 target family 和 E2 软件/训练契约已闭合；在
本次 18/5 subject development split、三 seed、K=128 和冻结 `0.005`
权重下，T1/T2 没有改善 required hard-token endpoint，因此不接纳 semantic
row 并保留 T0。仍不允许声称 G3 通过、E6/G2 信息保持通过、存在生理耦合
证据，或把本结果外推到保护测试被试。
