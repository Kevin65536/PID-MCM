# E2 语义 token 实现与实验计划

_2026-07-22 · development-only · protected subjects 24–29 closed_

## 结论

E1 入选的 K=128 `diverse_farthest + T2/T2 balance` 量化器配置已经完整继承到 E2。T0/T1/T2 的训练、开发门禁、统一 measured-data loader、版本化 adaptive-teacher sidecar、entry-specific mask、逐 objective 梯度审计、四种冻结表示探针和跨 seed 原型匹配均已实现并通过软件测试。

正式 E2 科学实验暂不启动。把 E0-v3 轨迹接回当前 measured-data contract 时发现，原 E0 局部通道选择未消费当前 bad-channel mask：230 个开发 trial 中仅 93 个 target 的六 EEG/两 fNIRS teacher view 仍被当前测量入口接纳，137 个被严格排除。当前产物足以验证代码闭环，但不足以把 E0 的全量可观测性结论直接迁移成正式 E2 证据。下一步必须先以当前 channel/QC contract 重建 adaptive teacher，并重新通过 E0 开发验证。

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

软件证据：29 项聚焦测试通过；T1 CUDA smoke 完成 4 个 optimizer steps，4 次梯度审计均通过，其中 3 个 batch 有 semantic target support，1 个稀疏 batch 被明确记录为 `zero_support`。冻结评估 smoke 能生成 `state_decoding.json`、`prototype_signatures.csv`、`prototype_stability.json`、`objective_ablation.csv` 和 `gradient_entry_audit.json`。这些数值只证明工具连通，不进入科学结果表。

## 当前 sidecar 审计

| 范围 | source trial | 当前通道契约接纳 | 排除 |
| --- | ---: | ---: | ---: |
| Train subjects 01–18 | 180 | 54 | 126 |
| Validation subjects 19–23 | 50 | 39 | 11 |
| 合计 | 230 | 93 | 137 |

接纳率为 `40.43%`。接纳 target 覆盖 16 名开发被试；7 名训练被试没有任何可接纳 target。排除原因与具体通道已写入 sidecar manifest。loader 对冲突 target 的行为是：在非 required 模式下把该 target 置为无效并回退到 measured-data 的非坏通道局部视图；在 required 模式下直接报错。任何情况下都不会使用被 bad-channel contract 拒绝的 teacher view。

## T0/T1/T2 固定设计

公共设置：Single-Trial mental-arithmetic namespace；18/5/6 subject split；20 s window（事件前 5 s 至后 15 s）；2 s patch；K=128、D=64；E1 入选的 cosine assignment、normalized latents、first-batch K-means、bounded diverse-farthest revival、balance weight `0.16`、temperature `2.0`、annealed-hard semantic-only reconstruction；三 seeds `20260719/20260720/20260721`。

| Row | Local/prototype objective | EEG coordinates | fNIRS coordinates |
| --- | --- | --- | --- |
| T0 | 无 teacher loss；仍加载同一 sidecar 以保持局部 view 完全匹配 | — | — |
| T1 | state `0.25` + prototype `0.25` | `r_mean`, `r_slope` | HbO/HbR mean 与 slope |
| T2 | state `0.25` + prototype `0.25` | T1 + `s_mean`, `s_slope` | 与 T1 相同 |

`0.25` 是 integration candidate，不是冻结科学超参数。重建 channel-aware E0 后，只能在训练被试上用 `{0.1, 0.25, 0.5}` 做一次梯度尺度校准；选择规则是 local/prototype 梯度可达且与 reconstruction 的 norm 比例不发生数量级失衡。查看 validation target endpoint 后不得再改权重。

## 执行阶段

### A. 已完成：软件闭环

1. 从旧 E0-v3 轨迹构建严格过滤 sidecar。
2. T0/T1/T2 CPU dry-run。
3. T1 4-step CUDA optimizer/gradient smoke。
4. 单 run 冻结评估 smoke。

此阶段只接受 tensor、mask、hash、梯度和 artifact correctness，不报告 semantic performance。

### B. 阻塞前置：channel-aware E0 重建

1. adaptive SSM 的局部 EEG/fNIRS 选择必须直接消费当前 unified sample 的 bad-channel mask。
2. 每个 target 的 selected channel names、sample key、QC mask 和 gauge provenance 必须在生成时冻结。
3. 重新运行 required `r`、HbO/HbR 与 optional `s` 的 train/validation observability、K=128 transmissibility 和 target coverage。
4. 要么获得预注册开发样本的完整覆盖，要么在看 E2 validation 前冻结一个有科学理由的排除规则并重新校准 E0；不得沿用当前 93-sample 子集直接做正式 E2。

### C. 训练被试权重校准

在一个 seed 上只用 subjects 01–18 比较 semantic weight `{0.1,0.25,0.5}`。固定一个权重后重新生成 suite protocol；该阶段不使用 subjects 19–23 的 target decoding 选择权重。

### D. 三 seed E2 development suite

按 seed-matched T0/T1/T2 运行 9 个 full-development jobs。每个 run 必须通过：E1 codebook health 继续保持、target coverage 与 frozen sidecar 一致、梯度 allowlist 无违反、训练后不发生未注册 mass revival、checkpoint 和 implementation hash 完整。

### E. 冻结评估与决策

主终点始终是 required signature family 在 hard token one-hot probe 上的 subject-held-out mean R²；T2 不通过增加 `s` 坐标改变主终点。`s_mean/s_slope` 是单独的 optional endpoint，用于决定 T2 是否优于 T1。连续 latent、posterior 和 codebook vector 是解释性并列表示。

E2 development 支持 T1/T2 的必要条件：

1. 相对 seed-matched T0，required hard-token endpoint 在 validation subjects 上方向一致改善，并给出 subject bootstrap uncertainty；
2. hard-token observed endpoint 高于训练 target permutation null q95；
3. improvement 不是单一 seed 或单一 coordinate 驱动；
4. signature-matched prototype stability 不下降，且 codebook health 仍在 E1 calibration 范围内；
5. T2 只有在 optional `s` endpoint 相对 T1 有稳定增益且 required endpoint 不下降时才入选；否则选择 T1；
6. E6/G2 information-retention gate 尚未通过时，只能记录“E2 development semantic evidence”，不能宣布 G3 promotion，更不能进入 E7 coupling-preservation。

## 运行命令

先重建 sidecar（当前命令会显式保留 channel-audit 排除记录）：

```bash
PYTHONPATH=. .venv/bin/python experiments/build_adaptive_teacher_sidecar.py \
  --source-run experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260716_adaptive_teacher_e0_v3_gauge_corrected_validation_v1 \
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

正式 artifact 要求 `prototype_signatures.parquet` 时，评估环境需提供 `pyarrow` 并添加 `--require-parquet`；当前基础环境无 Arrow，因此软件 smoke 同时写出的 CSV 仅用于连通性检查。

## Claim boundary

当前允许的结论只有：E2 软件闭环已实现；旧 E0 target 接入暴露出 channel/QC contract 不一致；严格过滤行为正确。当前不允许声称 adaptive teacher 改善了 token semantics、T1/T2 优于 T0、G3 通过、信息保持通过或存在耦合证据。
