# FP-RERUN-001：用修复后的 `L12S` 重跑迭代与网络，并补成本与逐步审计

- 任务编号：FP-RERUN-001
- 状态：ACTIVE（用户 2026-09-13 直接指令"好去做"，执行独立审计给出的第 3 项顺序）
- 日期：2026-09-13
- 基线提交：`b2c39a0`（`claude/FP-CENSUS-001`）
- 上游：
  - 审计与撤回：[FP-BOUND-002-l12m-withdrawal-and-split-repair.md](../../derivations/FP-BOUND-002-l12m-withdrawal-and-split-repair.md)（`L12M` 浓度步不成立 → 撤回；切半修复 `L12S` 在 `c64k` 步 1 实测 `−31.2%`、发出 `44/48`）
  - 被撤回的记录：[FP-ITER-BOUND-001](../research_branches/FP-ITER-BOUND-001/claude/first_result.md)、[FP-ITER-BOUND-002](../research_branches/FP-ITER-BOUND-002/claude/first_result.md)、[FP-NET-BOUND-001](../research_branches/FP-NET-BOUND-001/claude/first_result.md)

## 研究问题

1. 把**有效的**模型无关臂 `L12S` 接回认证迭代，能发出多少步、走多远、拿到多少价值？与 `frozen`、`L12`、以及需要核的 `L123` 各差多少？
2. 网络生产者与 numpy 生产者在**同一有效证书**下是否仍逐格一致？
3. 每个臂的**实际认证数据成本**是多少？在**同等总数据**下比较会得到什么结论？

## 设计

同一总体（`task_index 12..23` × 两种 mixing × 两条路线 = `48` 条记录），`K = 16` 预先冻结，每步 `16384` 链 × `64`，`δ_k = 0.05/K`。**八个格子共享每一步的认证批次**，各自独立演化：

```text
producer ∈ {numpy（封存数组路线）, network（model.py 字面注意力网络，LAYERS=160，Q_0=0）}
lever    ∈ {frozen, L12, L12S(f=0.9, p=0.05), L123（需核，oracle 参考）}
```

`L12S` 用 `split_fraction = 0.9`、`delta_prop_fraction = 0.05`（修复记录中的最优格，**在步 1 上选定**；本任务在迭代上使用它，属于把已登记的选择外推到新场景）。

**新增审计字段**：每步记录 `items_drawn`（若该格单独运行所需数据）、累计；每条记录记录墙钟；分析器输出逐步审计表。

## 可证伪假设（登记于运行前）

- **H1（强制，两项）**：① 每个格子的每个发出步**逐分量不退化**；② 每一步 `E_Q ≥` **该格自己生产者**的实测 `‖Q̂−Q^π‖∞`（0 覆盖违规）。
- **H2（修复后的迭代收益）**：`numpy|L12S` 的总发出步数 `≥ 1.30 ×` `numpy|frozen`；平均总价值增益 `≥ 1.30 ×`。
- **H3（`L12S` 相对 `L12`）**：`numpy|L12S` 的总发出步数 `≥` `numpy|L12` 的总发出步数（步 1 上 `L12S` 优于 `L12` 约 `20%`，迭代上应当仍占优；若否，则传播杠杆在迭代里不成立，作为负结果报告）。
- **H4（生产者一致）**：同一证书下 `network` 与 `numpy` 的总发出步数相对差 `≤ 20%`；第一步判决一致率 `≥ 80%`。
- **H5（成本）**：报告八个格子的实际认证数据量（若单独运行）与墙钟；并给出**同总数据截断**比较——把 `L12S` 轨迹按累计数据截到 `frozen` 的总量后重算发出数与价值。
- **H6（与作废数字的关系）**：`numpy|L12S` 的总发出步数 `≤` 作废的 `numpy|L12M` 的 `99`（修复必然更弱；若反而更高，说明两次运行的批次不一致，需查）。

## 允许方法

- 新建 `evaluate_fp_rerun_001.py`、`analyze_fp_rerun_001.py`、`verify_fp_rerun_001.py`。
- 复用 `fixed_policy_tight_certificate`（`L12S` 已实现）、`evaluate_fp_certfix_001.make_producer`、`fp_certfix_first_n`、`fp_sample_vectorised_batch`、`fs.improvement_for`。

## 禁止事项

- 不改判决规则、`η` 网格、`δ_total`、`K`、`split_fraction`/`delta_prop_fraction`（视为登记值）。
- 不改任何既有的 `evaluate_fp_*.py`（其哈希记录在既有封存的 config 中）。
- 不重新引用任何 `L12M` 数字；本任务的数字只能配 `L12S` 标签。
- 不因为 `H3` 或 `H4` 不成立而删格或改口径。

## 产物与位置

- D1 评估/封存：`icrl_softmax/evaluate_fp_rerun_001.py`，`results/FP-RERUN-001/claude/fresh_K16`
- D2 分析/验证：`icrl_softmax/analyze_fp_rerun_001.py`、`icrl_softmax/verify_fp_rerun_001.py`
- D3 报告：`docs/research_branches/FP-RERUN-001/claude/first_result.md`

## 记录区

- 2026-09-13：任务创建，假设与设计写于运行前。

## 记录区（续）：执行完成

- 2026-09-13：执行完成，报告 `docs/research_branches/FP-RERUN-001/claude/first_result.md`，封存 `results/FP-RERUN-001/claude/fresh_K16`。
- **判定**：`H1` PASS（八格 0 退化、0 覆盖违规）；`H2` PASS（`36 → 60`，`1.67×`；价值 `1.30×`）；**`H3` FALSIFIED**（`L12S 60 < L12 66`）；`H4` PASS（生产者逐格一致）；`H5` PASS；`H6` PASS。
- **结论**：修复后的传播臂被免费的 `L12` 全面压过 → **传播杠杆结案为负结果**。可用的是 `L12`（迭代 `36 → 66` 步、价值 `1.734 → 2.377`）。
- 成本与墙钟已记录（八格总 `650 s`）。验证 `928` 个重放步零差异。
- 仍为**初步结果**（单执行者）。

## 记录区（续）：第三方审阅 FAIL 后的口径处置（2026-09-14）

- **审阅判定**：本任务结果报告 `FAIL`；`L12S` 因此**不是有效界**，"sound/licensed"定性不成立。**测量保留**（发出数、价值、覆盖与退化观测、成本、墙钟）。见 [`FP-L12S-REVIEW-001-withdrawal.md`](../../derivations/FP-L12S-REVIEW-001-withdrawal.md)。
- **`H5` 判据漂移（接受）**：任务单登记的是**产物要求**（"报告每格每步预算、总数据量与同总成本截断"），实现里被改成**收益阈值**（`L12S` 收益 ≥ `frozen`）。**这不是原预登记的 PASS**，记为判据漂移。原登记要求已在分析报告中以表格形式履行（成本表 + 截断表），但不得据此声称 `H5` PASS。
- **`H6` 划掉（接受）**：`H6` 把"`L12S` ≤ 作废的 `L12M` 的 `99`"作为批次一致性证据，而本任务单自己禁止引用作废数字——冲突由**删除**消解，不由解释消解。其**意图**（批次是否一致）改由既有的更强检查承担：`FP-COST-001` 的 `16k` 档与**独立封存**的 `FP-XRULE-002` 做**逐位**核对（`max |Δvalue| = 0.000e+00`，已过）。该检查不引用任何作废数字。
- **`H3`/`H4` 的适用范围收窄（接受审阅第 3 条）**：保留的只是"**在本配置、本总体、本预算协议下** `L12S` 发出 `60 < L12` 的 `66`"；不得推广为"所有有效无模型传播方法均不成立"，也不得声称成本/收益全面支配（数据量并不相同）。
- **口径更正**：截断后 `L12S/L12/frozen` 的数据量为 `85,983,232 / 87,031,808 / 88,080,384`（GPT 独立重算），准确说法是**相同逐记录步数上限**，不是严格等量。
- **风险公式更正**：推导文字里 `2·n_iter·d·δ_k + d·δ_f` 多了一个系数 `2`；正确为 `n_iter·d·δ_k + d·δ_f = δ_prop`，且最终 estimate 复用 `log_k` 而非 `log_f`（更保守）。**代码本身是对的**，错在文字。
