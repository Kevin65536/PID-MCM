# 2026-08-12 对比方法 campaign 本轮产物清单

> **HISTORICAL SNAPSHOT — 不是当前状态或 authorization。** 本文保留当日
> `NO-GO` 语义；后续 campaign 的终态只看统一项目状态和带日期的最终结果报告。

## 冻结结论

本轮完成了六方法联合 protected campaign 的控制面、540-job release candidate、两次
全量 public CPU shadow 和独立最终审查。正式运行保持 **NO-GO**，因为只有一张健康
空闲 GPU、lane manifest 尚未生成且 authorization template 尚未双签。

没有读取 protected manifest/output；`protected_test_opened=false`，正式 job 数为 0。

## 核心身份

| 工件 | SHA-256 |
| --- | --- |
| `evidence/protected_campaign/joint_release_candidate_v1.json` | `4de176558d1ae502e2cfdc0a127bfa9f9c71caf5ccfdcfdc7f3cf9f47ac016a9` |
| `evidence/protected_campaign/authorization_template_v1.json` | `3c038c34dd6e552059107d93f844d8e028c51fb51b1034f5b8c93bc1153516c4` |
| `evidence/protected_campaign/orr_preflight_v1.json` | `78a3050671afd4a0b4437b002c54c70cdc37251551f6ec068a31afc89c5edb6e` |

候选包含 42 cells、36 supported、6 unsupported、34 direct、2 overlap、540 unique
jobs；folds 为 0–4，seeds 为 17/42/73。方法 job 数为 BIOT 90、CBraMod 90、REVE
90、EFRM 105、NormWear 90、BrainFusion 75。

## 本轮受控实现

- `build_joint_protected_release_candidate.py`
- `protected_campaign_common.py`
- `protected_campaign_worker.py`
- `benchmark_protected_campaign_shadow.py`
- `protected_campaign_controller.py`
- `prepare_protected_authorization.py`
- `aggregate_protected_campaign.py`
- `build_joint_protected_unlock_candidate_v2.py`

实现覆盖 candidate/source/config/environment/data/artifact hash、split fingerprint、严格
job/cell 路由、随机性和 GPU UUID 冻结、原子提交、quarantine、同 UUID 技术恢复、日志
脱敏、sealed aggregation 与双签 unblind。

## Public shadow 证据

两个目录各包含 BIOT、CBraMod、REVE、EFRM、NormWear、BrainFusion 的完整 MI outer0
seed17 public validation 推理，每方法 480 样本：

- `evidence/protected_campaign/shadow_cpu_pass_v1/`
- `evidence/protected_campaign/shadow_cpu_pass_v1_repeat/`

最新 verifier 对所有数组逐字段比较 name、dtype、shape 和 `np.array_equal`；六方法均
bitwise identical。两次运行的 runtime JSON 均通过 forbidden-token redaction scan。

双 GPU benchmark 已实际尝试并 fail-closed：GPU0 健康空闲，GPU1 当时约 88°C、98%
利用率，因此 `healthy_idle_gpu_count=1`，未生成 lane manifest。

## 验证结果

- 方法级测试：159 passed
  - BIOT 14
  - CBraMod 17
  - REVE 21
  - EFRM 34
  - NormWear 35
  - BrainFusion 38
- Campaign tests：16 passed
- 联合候选与 adapter gate 必需测试：9 passed
- release candidate `--check`：pass
- unlock candidate `--check`：pass
- 子代理最终只读复核：候选、authorization、ORR SHA 对齐；无未声明 high blocker

## 当前 ORR NO-GO 原因

- candidate state 为 `DRAFT`
- `orr_decision=NO_GO_PENDING_SHADOW_LANE`
- `lane_manifest_v1.json` 不存在
- 健康空闲 GPU 少于 2 张
- authorization template 为 false/pending，未双签

下一次继续时，应先等待第二张 GPU 健康空闲，运行双 GPU shadow 并冻结 lane；随后必须
重建 release candidate，重新核对 SHA，再由协议负责人和运行负责人在单独文件中双签
GO。不得复用本轮非授权模板作为授权。

## 同轮配套产物

本次 Git 整理同时纳入与当前状态直接相关的：

- 六方法/EFRM public-development v2 对齐和完成证据
- adapter gate 与 joint unlock candidate
- `single_modal_eeg/` public-only runner
- `docs/comparisons/STATUS.md` 和 adapter progress 审查
- `deliverables/comparative_methods_progress_review/` 的报告、数据、PPTD/PPTX、图片和
  可复现构建脚本

完整阅读顺序见 [`README.md`](README.md)。
