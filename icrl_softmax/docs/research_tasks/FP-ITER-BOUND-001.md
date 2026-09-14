# FP-ITER-BOUND-001：把收紧后的证书推回认证迭代

- 任务编号：FP-ITER-BOUND-001
- 状态：ACTIVE（治理例外同本线：用户 2026-09-13 直接指令"都去做"）
- 日期：2026-09-13
- 基线提交：`e1bdc63`（`claude/FP-CENSUS-001`）
- 上游依据：[FP-BOUND-001](../research_branches/FP-BOUND-001/claude/first_result.md)（`L12`/`L123` 有效）与 [FP-BOUND-002](../research_branches/FP-BOUND-002/claude/first_result.md)（`L12M` 无模型地拿回核版绝大部分收益）；以及 [FP-SHORT-001](../../research_tasks/FP-SHORT-001.md) 的实测——停步很窄，`E_Q/h` 中位 `1.12`、最紧 `1.00`。

## 研究问题

把 `E_Q` 压下去之后，**认证迭代到底能走多远**？它是否真的把"停步"从"界不够紧"这一侧推到了"改进真的用完"那一侧？

## 设计

在**同一批认证数据**上并列跑四条轨迹（每步新批次、种子按步索引，四臂共享该步批次，故比较是配对的）：

| 臂 | 需要核？ | 依据 |
|---|---|---|
| `frozen` | 否 | 现行 MP 证书（对照组） |
| `L12` | 否 | 已知 `Qhat` 值域 + 花满风险预算 |
| `L12M` | **否** | `L12` + 无模型传播（本任务的主臂） |
| `L123` | 是 | `L12` + 精确传播（上界参考，标注读核） |

总体：`task_index 12..23` × 两种 mixing（`FP-EARLYSTOP-001` 打开的 24 个环境），两条路线，共 `48` 条记录。Horizon `K = 16` 预先冻结；每步 `16384` 条独立链 × `64`；`δ_k = 0.05/K` 按定理 2 分配；判决规则、`η` 网格一字不改。某臂某步未发出即该臂轨迹在该步停止（其余臂继续）。

## 可证伪假设（登记于运行前）

- **H1（强制，两项）**：① 每个臂的每一个发出步都**逐分量不退化**（`v_new ≥ v_prev − 1e-12`，0 例外）；② 每个臂每一步都满足 `E_Q ≥` 实测 `‖Q̂−Q^π‖∞`（0 覆盖违规）。任一不成立即整任务 FAIL。
- **H2（长度）**：平均轨迹长度 `L12M > frozen`，且 `L123 ≥ L12M`。
- **H3（发出总量）**：`L12M` 的总发出步数 ≥ `frozen` 的 `1.10` 倍。
- **H4（价值）**：`L12M` 每条记录的平均总价值增益 ≥ `frozen` 的 `1.20` 倍。
- **H5（不早停）**：至少 `5` 条记录在 `L12M` 下**走到 horizon `K=16` 仍在发出**。
- **H6（救回第一步停步者）**：在 `frozen` 于**第 1 步**即停的记录中，`L12M` 至少救回 `50%`。
- **H7（机制）**：全体步上 `E_Q/h` 的中位数，`L12M` 比 `frozen` 低至少 `20%`。

## 允许方法

- 新建 `evaluate_fp_iter_bound_001.py`（复用 `fp_certfix_first_n`、`fp_sample_vectorised_batch`、`fixed_policy_tight_certificate`、`fs.improvement_for`）。
- 新建 `analyze_fp_iter_bound_001.py` 与 `verify_fp_iter_bound_001.py`。

## 禁止事项

- 不改判决规则、`η` 网格、`δ_total`、horizon（运行前冻结）。
- 不用 `L123` 的收益支持"只用行为数据"的结论。
- 不延长 horizon 来"多看几步"；`K=16` 是登记值。
- 不把"零退化"当作有效性证明（FP-TIGHT-001 已示不成立的臂也能 0 违规）；有效性由 FP-BOUND-001/002 的推导与验证承担。

## 产物与位置

- D1 评估/封存：`icrl_softmax/evaluate_fp_iter_bound_001.py`，`results/FP-ITER-BOUND-001/claude/fresh_K16`
- D2 分析/验证：`icrl_softmax/analyze_fp_iter_bound_001.py`、`icrl_softmax/verify_fp_iter_bound_001.py`
- D3 报告：`docs/research_branches/FP-ITER-BOUND-001/claude/first_result.md`

## 失败判据与停止条件

- 若 H1① 出现任一退化，停止并定位到具体记录与步，不得丢弃该记录。
- 若 `L12M` 与 `frozen` 的轨迹长度无差异，作为真实负结果报告：说明停步不是由界的保守度造成的。

## 记录区

- 2026-09-13：任务创建（用户直接指令"都去做"）。假设与 `K=16` 登记于运行前。

## 记录区（续）：执行完成

- 2026-09-13：执行完成，报告 `docs/research_branches/FP-ITER-BOUND-001/claude/first_result.md`，封存 `results/FP-ITER-BOUND-001/claude/`。
- 登记假设判定与限制见报告 §4/§3；所有数字经独立重算脚本验证（见报告"Verification"节）。
- 仍为**初步结果**（单执行者）。
