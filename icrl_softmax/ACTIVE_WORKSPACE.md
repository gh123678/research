# 当前研究状态

更新日期：2026-09-12。本页是当前研究的阅读入口；历史任务中的“当前”“下一步”等表述属于该任务创建时的语境。

## 研究问题

在明确限定的环境与结构条件下，固定权重的 softmax 注意力网络能否从经验数据估计动作价值，并支持有误差保证的策略改进？

当前实现链条是：

**经验数据 → 固定策略下的动作价值估计 → 误差保证 → 策略更新判断 → 再次评估与更新。**

注意力网络负责产生估计值 `Qhat`；误差证书、是否更新的判断与环境审计由相应程序执行。网络构造、策略更新规则和整个实验系统各自的范围，需按任务协议区分。

## 最近任务与状态

最近任务：[FP-ITER6-001](docs/research_tasks/FP-ITER6-001.md)；上一任务 [FP-ITER5-001](docs/research_tasks/FP-ITER5-001.md)。

- 已有第五步的两条计算实现运行记录：数组公式实现（numpy）与注意力网络实现。
- 已有 [结果报告](docs/research_branches/FP-ITER5-001/claude/first_result.md) 和 [同执行者检查记录](docs/research_branches/FP-ITER5-001/claude/verification_same_actor.md)。
- 任务单元数据仍标记 `ACTIVE`。本次仅清理材料，不修改任务定义、验收状态或升级验证等级。
- 本次清理没有发起新的实验任务。

## 最近记录显示什么

下表摘自第五步结果报告，表示每一步通过更新判断的路线记录数。两个计算实现的记录数一致；收益列使用报告中的 numpy 路线数据。

| 更新步数 | 发出更新的记录数 | 发出更新记录的平均总价值增量 |
|---|---:|---:|
| 1 | 22 | 2.717707 |
| 2 | 20 | 2.277997 |
| 3 | 15 | 1.286906 |
| 4 | 12 | 0.807500 |
| 5 | 12 | 0.361474 |

原始协议为 24 个环境记录，每个环境使用两种估计路线；每种计算实现因此有 48 条起始路线记录。上述记录数不是对任意环境的成功率保证。

第五步的直接观察：

- 第四步和第五步继续更新的集合都是同样的 12 条记录。
- 预先提出的“第五步记录数继续减少”预测被否定。
- 第五步平均收益低于第四步，报告中的最小总价值增量为 0.082395。
- 报告未记录两个计算实现之间的更新决策分歧，或在所检查更新中的证书/价值退化违规。
- 第五步网络与 numpy 的最大估计差约为 `4.567e-06`，仍在该实验的冻结容差内。

这些是受限小环境、指定数据和协议下的**初步结果**。最近这一系列主要由同一执行者实现和检查；两种计算实现的一致性不能排除共有的概念错误，也不能说明任意网络、任意环境或任意多步都会改善。

## 第六步与资格普查（2026-09-12 新增）

本次把两项工作合并执行：资格普查 [FP-CENSUS-001](docs/research_tasks/FP-CENSUS-001.md)
与第六个认证步 [FP-ITER6-001](docs/research_tasks/FP-ITER6-001.md)。记录见
[结果报告](docs/research_branches/FP-CENSUS-001/claude/first_result.md)、
[普查同执行者检查](docs/research_branches/FP-CENSUS-001/claude/verification_same_actor.md)、
[第六步同执行者检查](docs/research_branches/FP-CENSUS-001/claude/verification_same_actor_iter6.md)。

第六步（本轮仅 numpy 路线）：

| 更新步数 | 发出更新的记录数 | 平均总价值增量 | 最小总价值增量 |
|---|---:|---:|---:|
| 1 | 22 | 2.717707 | 0.104197 |
| 2 | 20 | 2.277997 | 0.779902 |
| 3 | 15 | 1.286906 | 0.378197 |
| 4 | 12 | 0.807500 | 0.184902 |
| 5 | 12 | 0.361474 | 0.082395 |
| 6 | 9 | 0.167057 | 0.019347 |

- 第四、五步“都是同样的 12 条”只是一个区间，不是不动点：第六步降到 9 条，失去 3 条，没有新增。
- 第六步最小总价值增量 `0.019347` 低于第四、五步通过的非平凡下限 `0.05`；该项预先登记的预测被否定。
- 第六步全部 9 条更新分量不退化且总值严格上升；六步之内没有证书违规。
- 第六步 horizon 变更已验证为惰性：第五步及以前共 `117` 条记录逐步完全一致。
- 网络路线本轮未运行，因此比 numpy 路线落后一步。

资格普查（48 条路线记录 × 至多 5 步；先复现第五步的判定，`48/48` 完全一致）：

- `sigma_min / E_Q` 在第一步区分度很高，但**不是分类器**：最优阈值仍有 `1/48` 误分类，两组区间重叠。
- 区分主要来自状态内的动作价值跨度（两组相差 `2.6` 倍），而不是认证误差（`1.2` 倍）；单用 `E_Q` 预测弃权会误分类 `16/48`。
- 普查中每一次弃权的原因都是同一条 `improvement_lcb_nonpositive`：证书从未失败，失败的是改进预算。
- 该判据能描述前五步，却预测不了第六步：停下的 3 条中有 2 条的第五步比值高于仍在继续的最低记录，没有任何阈值能把它们分开。
- 第六步预测在运行前冻结（`2026-09-12T11:23:59Z`，sha256 `70b79f03…`）；两条登记规则给出了完全相同的预测，故该比较无法区分优劣，这一点作为设计缺陷记录在报告中。
- 普查批次与第五步所用批次一致，已由 `verify_census_batch_frozen.py` 逐对计数核对。

以上同样是受限小环境与冻结协议下的**初步结果**，且仍为同一执行者自检。

## 研究走到这里的关键环节

历史材料保留用于追溯证据，以下链接不代表新任务授权。

| 环节 | 已有记录及范围 | 入口 |
|---|---|---|
| 固定策略迭代构造 | 检查明确条件下的网络/公式对应关系 | [FP-ITER-001](docs/research_tasks/FP-ITER-001.md)、[FP-EXPL-001](docs/research_tasks/FP-EXPL-001.md) |
| 误差保证与安全更新 | 早期冻结协议出现零更新，推动对误差界尺度的诊断 | [动作差值报告](docs/research_branches/action_gap_certificate_report.md)、[固定策略报告](docs/research_branches/fixed_policy_expected_sarsa_report.md) |
| 改善误差保证尺度 | FP-SCALE-001 留下诊断；FP-SCALE-002 在其新协议下记录首次非零更新 | [尺度诊断](docs/research_branches/FP-SCALE-001/claude/first_result.md)、[FP-SCALE-002](docs/research_branches/FP-SCALE-002/claude/first_result.md) |
| 接入实际注意力网络 | 比较网络产生的估计与数组公式实现 | [FP-ATTN-001](docs/research_branches/FP-ATTN-001/claude/first_result.md) |
| 连续策略改进 | 从两步逐步检查到五步，最近结果见上表 | [FP-ITER5-001](docs/research_branches/FP-ITER5-001/claude/first_result.md) |
| 跨状态借用数据的另一条探索 | 两个冻结任务未得到支持其规定方法的证据，不能据此否定一切泛化方法 | [FP-KERN-001](docs/research_tasks/FP-KERN-001.md)、[FP-KERN-002 综合报告](docs/research_branches/FP-KERN-002/codex/final_synthesis.md) |

早期“尚无网络实现”“所有协议都无法发出更新”“FP-SCALE-001 是当前待启动任务”等总览表述已移除。具体旧实验的零结果仍保存在原报告中，不因后续实验进展而被改写。

## 当前需要分清的科学问题

- 能发出若干步更新，是否足以支持更一般的策略改进结论？当前小环境实验不能独自回答。
- 停止更新时，是可获得的改进变小，还是现有误差保证不足以确认改进？需按具体记录区分。
- 继续增加更新步数，会排除什么解释或检验什么新假设？不能仅凭已跑到第五步就确定下一个研究任务。

以上是理解现有证据的阅读问题，不是本次新建的正式研究计划。

## 代码与证据位置

最近路线的主要代码：

- 网络构造：`model.py`。
- 固定策略估计及更新规则：`fixed_policy_expected_sarsa.py`、`fixed_policy_expected_sarsa_scaled.py`。
- 误差保证：`fixed_policy_variance_certificate.py`。
- 最近迭代评估：`evaluate_fp_iter2_001.py`、`evaluate_fp_attn_iter_001.py`。
- 第五步分析与同执行者检查：`analyze_fp_iter5_001.py`、`verify_fp_iter5_001_same_actor.py`。
- 资格普查与第六步：`evaluate_fp_census_001.py`、`analyze_fp_census_001.py`、`analyze_fp_iter6_001.py`、`verify_census_batch_frozen.py`、`verify_fp_census_001_same_actor.py`、`verify_fp_iter6_001_same_actor.py`。

第五步输出：

- `results/FP-ITER5-001/claude/numpy/`
- `results/FP-ITER5-001/claude/network/`

第六步与普查输出：

- `results/FP-ITER6-001/claude/numpy/`
- `results/FP-CENSUS-001/claude/smoke/`
- `results/FP-CENSUS-001/claude/formal/`（含冻结的 `prediction_step6.json`）

各次实验的准确版本、环境、命令及失败记录，以对应 [任务单](docs/research_tasks/) 和 [结果报告](docs/research_branches/) 为准。旧实验依赖的程序与结果目录保持原位置。

### FP-KERN 历史结果的实际位置

这两次任务的结果位于其独立工作区内部；不要误用宿主目录下不存在的简写位置：

- `results/FP-KERN-001/codex_worktree/icrl_softmax/results/FP-KERN-001/codex/`
- `results/FP-KERN-001/claude_worktree/icrl_softmax/results/FP-KERN-001/claude/`
- `results/FP-KERN-002/input/`：共同冻结输入。
- `results/FP-KERN-002/codex_worktree/icrl_softmax/results/FP-KERN-002/codex/`
- `results/FP-KERN-002/claude_worktree/icrl_softmax/results/FP-KERN-002/claude/`

## 历史材料与协作规则

正式任务、设计、推导、原始实验与验证记录继续保留；它们是研究依据，不能按文件日期判断是否失效。旧实验代码与结果另有历史归档：`../_archive/icrl_softmax_history_20260831/`。

当前协作规则仅以仓库根目录 [AGENTS.md](../AGENTS.md) 为完整依据。本页不修改双方的任务分工、异议流程或合并要求。
