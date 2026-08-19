# SSM 重建可信度完整实验结果

_状态：开发范围内的探索性实验已完成；2026-08-19_

## 结论

当前 SSM 可以形成较好的 EEG–fNIRS 联合后验重建，但不能可靠地仅由 EEG
恢复 fNIRS 轨迹。因此，本实验支持把 joint smoother 用作多模态拟合诊断，
不支持把当前 shared driver 当作已经验证的、可由 EEG 独立观测的共享生理状态。

核心 7 个任务的开发验证受试者结果如下：

- joint HbO 轨迹 NRMSE 为 `0.810–1.037`，经验 95% coverage 为
  `0.935–0.953`；
- EEG-only HbO 轨迹 NRMSE 为 `1.906–2.286`，coverage 为
  `0.749–0.867`；
- 7/7 个任务的 paired `log(joint NRMSE / EEG-only NRMSE)` 均小于零，
  10,000 次 subject-bootstrap 区间均未跨零；
- EEG-only 对其直接观测的 EEG log-power PCA proxy 重建较好，NRMSE 为
  `0.289–0.368`，但 HbO/HbR 误差仍约为观测轨迹时间标准差的两倍。

最后一项是本实验最重要的可信度判断：保留 EEG proxy 并不等价于可靠恢复
血氧动力学轨迹。

## 实验范围与完整性

完整运行覆盖 13 个 task cell、279 个 subject/task 单元，全部完成、无失败：

- 核心：MA、LMI、RMI、WG、0-back、2-back、3-back；
- 描述性附录：Visual RR/RF/FR/FF、REFED video、DSR block context；
- 每个单元采用依赖组完整的 5 折 within-subject cross-fit；
- channel selection、EEG PCA、SSM 参数和噪声平衡只在训练折拟合；
- 生成 29,504 条 window metric 和 4,665,600 条逐时刻轨迹记录；
- Single-Trial 24–29 与 Simultaneous VP024–VP026 未打开，manifest 中
  `protected_open=false`。

核心主表只解释 subjects 19–23 的 development-validation profile；subjects
01–18 保留为 fit-cohort 稳定性检查。两组都使用 subject 内重新拟合，因此该对比
不是外部泛化试验。

## 核心任务结果

下表为 local spatial path 的 equal-subject mean。joint 路径使用 held-out EEG、
HbO 与 HbR 参与 smoothing；EEG-only 才是跨模态恢复检查。

| 任务 | 路径 | HbO NRMSE | HbR NRMSE | HbO 重建/观测时间 SD | HbO 标准化残差 RMS | HbO coverage | EEG-proxy NRMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MA | joint | 0.834 | 0.786 | 0.845 | 0.763 | 0.953 | 0.484 |
| MA | EEG-only | 2.005 | 1.839 | 1.145 | 1.185 | 0.867 | 0.289 |
| LMI | joint | 0.810 | 0.752 | 0.785 | 0.795 | 0.953 | 0.592 |
| LMI | EEG-only | 2.286 | 2.466 | 1.313 | 1.629 | 0.749 | 0.320 |
| RMI | joint | 0.875 | 0.774 | 0.887 | 0.773 | 0.952 | 0.548 |
| RMI | EEG-only | 2.179 | 1.964 | 1.306 | 1.433 | 0.794 | 0.307 |
| WG | joint | 0.976 | 0.981 | 0.756 | 0.899 | 0.935 | 0.548 |
| WG | EEG-only | 1.906 | 2.057 | 1.090 | 1.379 | 0.822 | 0.368 |
| 0-back | joint | 1.037 | 0.782 | 0.946 | 0.844 | 0.944 | 0.508 |
| 0-back | EEG-only | 2.210 | 1.893 | 1.320 | 1.238 | 0.857 | 0.307 |
| 2-back | joint | 0.913 | 0.990 | 0.841 | 0.867 | 0.938 | 0.525 |
| 2-back | EEG-only | 2.154 | 2.527 | 1.149 | 1.526 | 0.784 | 0.306 |
| 3-back | joint | 0.868 | 0.971 | 0.745 | 0.878 | 0.942 | 0.534 |
| 3-back | EEG-only | 1.983 | 2.169 | 1.151 | 1.321 | 0.825 | 0.341 |

joint 的时间 SD ratio 多数低于 1，标准化残差 RMS 也低于 1，说明其后验带相对
偏宽或重建被平滑；EEG-only 的 HbO 时间 SD ratio 与标准化残差 RMS 多数高于
1，同时 coverage 低于名义值，表现为过度变化且不确定度不足。重建偏离与后验
不确定度因此不能合并成单一“置信度”。

## 描述性任务

REFED 与 DSR 的 EEG-only HbO NRMSE 分别为 `1.896` 和 `1.963`；joint 为
`0.708` 和 `0.960`。这些是 video/block-context 的任务特异估计，不与核心任务
合并成总分，也不形成事件级 DSR fNIRS 主张。

Visual RF 的统计量稳定，但 RR、FR、FF 的 primary mean 被 S01 少数近零观测
时间 SD 窗口放大。最极端的 Visual FR EEG-only window NRMSE 为
`6.41e6`，对应观测时间 SD `1.18e-7`；该值高于协议规定的 `1e-8` undefined
阈值，所以在主分析中仍是有效值。注册主结果不被替换。额外的 posthoc
`observed temporal SD >= 0.01` 敏感性分析分别标记 RR 6、RF 0、FR 3、FF 2
个窗口，并同时报告 subject-window median，以说明结论对分母的依赖。

## 额外可靠性检查

- local 与 global 核心空间消融的 28 个 task/model/modality 区间中，仅 7 个
  未跨零，且方向不一致；没有证据支持 all-scalp EEG 一致优于六通道 local 路径。
- fit 与 development cohort 的最大均值变化为 LMI EEG-only HbO NRMSE
  `+0.218`；该比较只用于稳定性描述。
- Visual 异常说明 MSE、观测时间 SD 与 NRMSE 必须共同保留，不能只展示标准化
  指标。

## 证据与图形

完整运行：

- [`manifest.json`](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260819_ssm_reconstruction_reliability_full_v1/manifest.json)，SHA-256
  `4899c4bbc9a8596243a4cbf892a60070458be39f67c74ac0571eff56d4d449d5`；
- [`task_summary.csv`](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260819_ssm_reconstruction_reliability_full_v1/task_summary.csv)，SHA-256
  `7c5f6d9e11f303e6781ce0c03d16470877f78a9e5fe0792edba00550b41a1824`；
- [`summary.md`](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260819_ssm_reconstruction_reliability_full_v1/summary.md)。

后处理分析：

- [`manifest.json`](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260819_ssm_reconstruction_reliability_analysis_v1/manifest.json)，SHA-256
  `b7d42852040575aa417789134daf736b897bd8d229850dd214931662427050d6`；
- [`analysis_summary.md`](../../experiments/runs/physiology_semantic_tokenizer/e0_teacher_validity/20260819_ssm_reconstruction_reliability_analysis_v1/analysis_summary.md)，SHA-256
  `5e6425981a17942c79a67d920bdd016b1ba30f26a5ed3ffee8d3031a6991c69f`；
- 核心 profile、描述性对数 profile、paired model contrast、空间消融和 Visual
  denominator sensitivity 均提供 SVG、PNG、alt text、精确 source table 与
  figure provenance manifest；完整运行另保留五组 task-specific time-course 图。

## 对方法主线的含义

本实验没有验证唯一共享状态，反而显示当前 SSM 的跨模态可恢复性不足。因此下
一阶段不应恢复 VQ，而应按既定计划直接测试无信息瓶颈的 continuous
shared/private latent，并加入 matched swap、derangement、private-only 与
shared-only 对照。该实验需要把“联合拟合良好”和“跨模态共享存在”继续视为两个
不同命题。
