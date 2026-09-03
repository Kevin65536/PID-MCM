# T3c 实验第四步：层次综合参数准入详细报告

_实验日期：2026-09-03；报告状态：准入实验完整，measured hierarchical arm 未启动_

## 结论

第四步已经启动并完成了计划要求的首个 fail-closed 准入检查。正式 v3 运行的机器判定为
`BLOCKED_PREREQUISITE`：8 项必需条件中 2 项满足、6 项不满足。因此本轮只完成了
综合参数坐标和 Normal–Normal 收缩的软件自检，**没有启动 measured hierarchical
partial pooling**，也没有产生任何层次模型的预测、参数或生理结论。

阻塞不是计算错误。第二步的 `beta/kappa/tau` practical-identifiability 主终点在全部案例中
均未通过；第三步虽然观察到参数稳定性 screen 失败，但跨 session 坐标不是共同 gauge，
不能把该失败直接当成同一物理参数的跨 session 不稳定。第三步三折还使用了 3 个不同
gauge fingerprint 和 2 个不同 fNIRS endpoint，且既有合成检查明确不是 SBC、profile/
multistart 或 practical-margin 实验。按照 `T3c-hierarchical` “只池化 T3a 已可辨识参数”
的条件，直接运行 measured 层次收缩会把不可辨识性或 observation gauge 差异伪装成稳定
个体效应，因此在读取新 measured metadata/array 之前停止。

正式证据由 v3 [manifest](../../experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/manifest.json)、
[summary](../../experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/summary.json)、
[准入检查表](../../experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/admission_checks.csv)
和 [composite contract](../../experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/composite_contract.json)
共同支持；manifest 是本运行状态、边界、source hash 和 artifact hash 的唯一 owner。

## 1. 运行身份与边界

| 字段 | 冻结值 |
| --- | --- |
| 正式 suite/run | `t3c_hierarchical_composite_admission/20260903_step4_admission_v3` |
| schema | `t3c_hierarchical_composite_admission_v1` |
| 范围 | `preflight_only_no_measured_arrays` |
| 状态 | `status=admission_check_complete`; `run_state=complete`; `completion_status=complete` |
| 准入判定 | `BLOCKED_PREREQUISITE`; `required_met=false` |
| measured hierarchy | `not_started` |
| UTC 时间 | 2026-09-03 03:19:14.952 至 03:19:14.959 |
| Asia/Shanghai 时间 | 2026-09-03 11:19:14.952 至 11:19:14.959 |
| 墙钟时长 | 0.006908 秒 |
| 新 measured metadata / array 读取 | 0 / 0 |
| validation/protected array 读取 | 0 / 0 |
| 资格边界 | `qualification_eligible=false`; `decision_eligibility=false` |

本次运行只读取第二、三步已冻结的 JSON 证据和第三步 fold calibration artifact，不调用
canonical measured loader，不枚举新的 trial metadata，也不打开任何 EEG/fNIRS 数组。
`subject_19–23` validation 和 `subject_24–29` protected 保持关闭。以上字段见
[v3 manifest](../../experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/manifest.json)。

v1/v2 是准入实现收紧过程中的本地预备运行；本报告不使用它们作科学证据。v3 记录了最终
runner SHA `846b349847ac93dc6522ecc4262f39742366ec28316bae9e47e7be0f0db5ac5a`
与 config SHA `6ce049571bd268f5e68e7f3f1b1e43a66f875b06e3b056b28e36c7ec77bec0b1`。

## 2. 为什么第四步先做准入而不是直接拟合

项目计划把 `T3c-hierarchical` 定义为条件扩展：只有 `T-P2` 已支持的参数才能进入 partial
pooling，并且还需要预先冻结、可比较坐标中的跨被试稳定性失败。计划附件也要求只放开一至
两个可辨识综合方向，优先 gain 和 time scale，固定 `alpha`、`E0` 与光学 gauge。证据见
[实验计划](../EXPERIMENT_PLAN.md)和本次
[resolved config](../../experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/resolved_config.yaml)。

层次收缩可以减少方差，但不能创造 likelihood 中不存在的信息。若 local estimate 由边界、
prior 或不同 observation gauge 主导，收缩后的数值会更集中，却不能因此被解释为更可靠的
生理 trait。因此准入要求先证明方向可辨识、gauge 与 endpoint 可比较、并在 measured 评分
前冻结 practical margin；任何一项缺失都停止 measured arm。

## 3. 冻结的综合参数坐标

本次只冻结解析坐标与候选阶梯，不拟合参数：

\[
G_f=\frac{\beta}{\gamma},\qquad
T_f=\frac{1}{\sqrt{\gamma}},\qquad
\zeta_f=\frac{\kappa}{2\sqrt{\gamma}},\qquad
T_v=\tau\alpha,
\]

\[
\phi=(\log G_f,\log T_f,\operatorname{logit}\zeta_f,\log T_v).
\]

固定 `alpha` gauge 后的逆映射为：

\[
\gamma=T_f^{-2},\quad
\beta=G_fT_f^{-2},\quad
\kappa=2\zeta_f/T_f,\quad
\tau=T_v/\alpha.
\]

参考 primitive 为 `beta=1.0, kappa=0.64, gamma=0.32, tau=2.0,
alpha=0.32, E0=0.32`，对应 `G_f=3.125`、`T_f=1.767767 s`、
`zeta_f=0.565685`、`T_v=0.64 s`。候选顺序固定为 `C0_fixed → C1_G →
C1_T → C2_GT`；`zeta/Tv/alpha/E0` 均 deferred，完整协方差禁用，最多两个独立
Normal random effects。证据见 [composite contract](../../experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/composite_contract.json)。

`logit(zeta)` 只在 `0 < zeta < 1` 定义；当前 software smoke 使用内部参考点，未证明整个
primitive 外包络都处于该子域。若将来释放 `zeta`，必须另行冻结其联合可行域，不能对
`beta/kappa/gamma` 分别 clipping。本轮不释放 `zeta`，所以这一点是后续合同约束而非本次
数值失败。

## 4. 软件自检结果

解析 round-trip 的最大绝对误差为 `1.1102230246251565e-16`。缺少显式
`alpha_gauge` 或逆映射诱导出的 primitive 越界会 fail closed，不进行静默 clipping。

Normal–Normal smoke 使用 local log-coordinate `[-0.4, 0, 0.4]`、measurement
variance `[0.01, 0.04, 0.16]`、population mean `0` 和 population variance
`0.09`。得到的 local-information weight 为 `[0.9, 0.6923077, 0.36]`，posterior
mean 为 `[-0.36, 0, 0.144]`，posterior variance 为
`[0.009, 0.0276923, 0.0576]`。因此不确定性较高的第三个 local estimate 被更强地拉回
population mean，闭式公式按预期工作。证据见
[software preflight](../../experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/software_preflight.json)。

这个 smoke **不等于** Balloon known-truth recovery：它没有生成合成 EEG/fNIRS，没有拟合
hierarchical hyperparameter，也没有执行 SBC、profile likelihood、multistart、LOSO、null
或 held-out NLL。因此它只验证代码坐标和收缩代数，不能用来支持“gain/time 可辨识”或
“partial pooling 有效”。

## 5. 八项准入检查

| 必需检查 | 结果 | 冻结证据 |
| --- | :---: | --- |
| 既有 Step 2/3 运行完整 | PASS | 两个 manifest 的 `completion_status=complete` |
| `T-P2` identifiability 得到支持 | FAIL | Step 2 `supported_in_all_cases=false` |
| 在共同 gauge 中已有跨被试稳定性失败 | FAIL | 参数 screen 失败，但 `cross_session_gauge_invariant=false` |
| 共同 observation/driver gauge 已冻结 | FAIL | 无独立 Step 4 registry；三折有 3 个 gauge fingerprint |
| 固定 local fNIRS endpoint 已前瞻冻结 | FAIL | 无 Step 4 endpoint registry；三折出现 2 个 post-hoc endpoint |
| composite synthetic SBC/profile/multistart 已通过 | FAIL | Step 3 synthetic 只声明 deterministic software preflight |
| measured 评分前已冻结 practical margin | FAIL | 无独立 composite technical-repeat margin |
| validation/protected 数组保持关闭 | PASS | Step 2/3 manifest 均为 closed；本运行数组读取 0 |

完整逐项字符串和布尔值见 [admission checks](../../experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/admission_checks.csv)。

## 6. 阻塞证据详解

### 6.1 可辨识性没有通过

第二步主终点要求每个案例的 `beta/kappa/tau` profile 网格完整，且 likelihood support
有限、连续、不接触注册边界。合成案例和 3 个 measured 代表案例全部为 `False`；合成案例
还被解释为 `parameters_nonidentifiable_but_state_stable`。因此第二步不能给第四步提供一个
已获准的 primitive random-effect 方向。证据见 [Step 2 summary](../../experiments/runs/physiology_semantic_tokenizer/t3_identifiability/20260902_step2_identifiability_v3/summary.json)
和 [Step 2 详细报告](20260902_T3_IDENTIFIABILITY_STEP2_REPORT.md)。

这也不等于新的 `G_f/T_f` 一定不可辨识；正确表述是：它们尚未经过自己的 known-truth
SBC/profile/multistart/SVD 合同，因此目前既不能通过，也不能宣称失败。

### 6.2 第三步稳定性失败不在共同坐标中

第三步 effective-κ screen 只有 6/18 被试达到注册稳定阈值，7/18 被试至少一次命中边界；
但跨 session driver 明确标记 `gauge_invariant=false`。预测上，候选相对固定 M0 的
subject-equal `candidate − M0` ΔNLL 为 `+5.288570`，95% subject-block CI
`[+1.319282,+9.417574]`，方向不利于候选。这支持停止把现 κ 当 trait，却不能证明
同一共同坐标下存在可由 hierarchy 修复的跨被试方差。证据见
[Step 3 summary](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/summary.json)
和 [Step 3 详细报告](20260902_T3_MULTISESSION_LOSO_STEP3_REPORT.md)。

三折 observation operator 的差异是实质性的：

| 留出 session | fNIRS endpoint | `P0` | `Q0` | EEG `pc_scale` |
| --- | --- | ---: | ---: | ---: |
| `session_01` | `FC3FC1_HbO/HbR` | 22.821893 | 7.987662 | 4.491670 |
| `session_03` | `FC3FC5_HbO/HbR` | 28.942496 | 10.129874 | 4.488301 |
| `session_05` | `FC3FC1_HbO/HbR` | 21.561968 | 7.546689 | 4.486479 |

对应 canonical JSON 形成 3 个不同 SHA-256 fingerprint。fNIRS endpoint 也不是同一 pair。
虽然每折内部 calibration 只用训练 session、没有 held-out fit leakage，但这些折内坐标不能
直接合并成一个跨折 biological trait。证据见 [Step 3 fold calibration](../../experiments/runs/physiology_semantic_tokenizer/t3_multisession_loso/20260902_step3_multisession_loso_v2/fold_calibration.json)
和 [v3 observed endpoints](../../experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/summary.json)。

### 6.3 不能事后选择 endpoint 或 margin

现在依据第三步“2/3 折选择 FC3FC1”来把 FC3FC1 登记成正式 endpoint，会使用已经看过的
fold outcome；同理，直接把历史 FC3FC5 或全部三个 session 拟合出的共同 gauge 当作新的
LOSO 独立坐标，也会让每个将来 held-out session 参与设计。正确解锁方式必须是信号无关的
前瞻规则、对所有评分 fold 独立的 calibration source，或明确降级成 fold-local exploratory
prediction。后两者不能再声称 strict cross-fold trait。

本次也没有从当前结果反推 practical margin。先看结果再设 margin 会使后续
`candidate − M0` 判定循环；必须先用独立 technical repeats 冻结。

## 7. 本轮可以和不可以得出的结论

可以得出：

- 第四步准入流程与 composite/partial-pooling 软件原语可以运行，且解析自检通过。
- 当前冻结证据不足以准入 measured hierarchical partial pooling。
- 本次停止发生在任何新 measured metadata/array 访问之前，validation/protected 保持关闭。

不可以得出：

- “partial pooling 没有效果”或“gain/time 一定不可辨识”；本轮没有拟合这些模型。
- 任何 `G_f/T_f` 被试差异、血管健康 trait、绝对生理 rate、OEF/CMRO2 结论。
- teacher qualification、physical-token label 或 tokenizer promotion。
- population generalization；`subject_19–23` 没有被打开或重标，`subject_24–29` 仍受保护。

## 8. 解锁 measured 第四步所需的最小顺序

1. 前瞻冻结一个 signal-independent local endpoint、EEG sign/scale、`P0/Q0`、observation
   scale 和完整 gauge hash；不得按第三步结果多数票选择。
2. 单独运行 composite known-truth panel，对 `C1-G` 和 `C1-T` 各自执行 SBC、profile、
   multistart、白化 sensitivity/SVD、边界和 null 检查；truth 不进入 fitter，所有 trial
   independent reset。`C2-GT` 只在两个一维方向分别通过后进入。
3. 用独立 technical repeats 在看 measured score 前冻结 practical NLL/recovery margin 和
   invalid/inconclusive 规则。
4. 只有以上三项通过，才可在 `subject_01–18 × session_01/03/05 × MA` 上运行三折
   whole-session LOSO partial pooling；`19–23` 和 `24–29` 继续关闭。主比较应只保留一个
   预注册 contrast，并用 subject-block 推断。

这四项是后续工作建议，不是本次 v3 已执行或已获准的实验。

## 9. 实现与验证

本次新增独立 [config](../../experiments/configs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission_v1.yaml)、
[runner](../../experiments/evaluate_t3c_hierarchical_composite_admission.py) 和
[tests](../../tests/test_t3c_hierarchical_admission.py)，没有修改第二、三步的冻结 runner/config
或结果目录。runner 校验五个 source artifact 的预注册 SHA，拒绝缺 `alpha_gauge`、非有限
输入和 induced primitive 越界，并把状态统一写入 `manifest.json`。

目标回归加第二、三步联合测试共 **25 passed**：覆盖 composite round-trip、alpha gauge、
联合边界拒绝、Normal–Normal 收缩极限、空证据 fail-closed，以及当前冻结证据恰好产生
2 PASS / 6 FAIL。`py_compile` 与 `git diff --check` 同时通过。

v3 的 6 个必需 artifact 均存在并由 manifest 记录大小与 SHA；`admission_checks.csv` 恰有
8 个数据行。artifact 完整性见 [manifest artifacts](../../experiments/runs/physiology_semantic_tokenizer/t3c_hierarchical_composite_admission/20260903_step4_admission_v3/manifest.json)。
