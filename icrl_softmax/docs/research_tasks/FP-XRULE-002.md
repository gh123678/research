# FP-XRULE-002：按状态循环在更长的 horizon 下停在哪里

- 任务编号：FP-XRULE-002
- 状态：ACTIVE（用户 2026-09-13 直接指令"继续"）
- 日期：2026-09-13
- 基线提交：`583bcaa`（`claude/FP-CENSUS-001`）
- 上游：[FP-XRULE-001](../research_branches/FP-XRULE-001/claude/first_result.md) 在 `K=4` 上把两个机制都搬到了新族，但**按状态格在两族上都走到了 horizon**（均长 `4.000`），因此 `K=4` 无法区分"循环本来能走更远"与"到第 4 步为止"；审计第 ④ 条的条件（证明闭合）已由 `L12m` 撤回与 `L12S` 修复满足。

## 研究问题

1. 在更长的 horizon 下，**按状态保守更新的循环停在第几步**？它是自终止的，还是仅仅被我们选的 `K` 截断？
2. 证书（`frozen` vs `L12S`）在长 horizon 下还有没有边际价值？
3. 按状态规则的优势是否来自**部分更新**（每步只更新一部分状态）？

## 设计

与 FP-XRULE-001 **完全相同的族、环境、种子、路线、四格与判决规则**，只把 `K` 从 `4` 改到 **`12`**（`δ_k = 0.05/12`），并把 `states_updated` 的逐步分布纳入输出。

- 族：F1（6×4，`task_index 200..223`，mixing {0.08,0.5}）、F2（4×3，`GAP_BONUS=1.0`，`task_index 300..323`，mixing {0.2,0.7}）。
- 四格：`{conj, perstate} × {frozen, L12S(f=0.9, p=0.05)}`，共享每步认证批次，各自独立演化。
- 每族 96 条路线记录，`16384` 链 × `64`/步，`min_visits = 2000`。

## 可证伪假设（登记于运行前）

- **H1（强制，两项）**：① 每族每格每个发出步**逐分量不退化**；② 每步 `E_Q ≥` 该步实测 `‖Q̂−Q^π‖∞`（0 覆盖违规）。
- **H2（是否自终止）**：两族中**至少各有一条** `perstate` 轨迹走到 `K=12`。若两族都无轨迹走到 horizon，说明按状态循环在 12 步内自终止——这是比 `K=4` 饱和更强的结论，原样报告。
- **H3（停在哪里）**：`perstate` 格的**平均模拟步数**在两族均 `≥ 8`（若否，说明循环在 8 步内普遍停止）。
- **H4（证书在长 horizon 的边际价值）**：`perstate|L12S` 的平均总价值增益 `≥` `perstate|frozen`（方向为正；幅度不设下限——FP-XRULE-001 已表明二者是替代品）。
- **H5（成本）**：报告每格单独运行的数据量与墙钟；并给出**同总成本截断**到 `conj|frozen` 的实际步数后的发出数与价值。
- **H6（机制：部分更新）**：`perstate` 的**发出步中**至少 `50%` 满足 `states_updated < S`（即优势来自选择性更新，而非"每步都全量更新"）。

## 允许方法

- 新建 `evaluate_fp_xrule_002.py`：**复用** `evaluate_fp_xrule_001.run_family`，通过设置其**模块级 `HORIZON`** 参数化，从而不改动该文件（其哈希仍与既有封存 config 一致）。
- 新建 `analyze_fp_xrule_002.py`、`verify_fp_xrule_002.py`（后者复用 `verify_fp_xrule_001.py` 的重放逻辑，参数化 horizon）。

## 禁止事项

- 不改族定义、环境、种子、路线、规则、证书参数。
- 不改任何既有 `evaluate_*.py` / `verify_*.py` 的字节。
- 不因 `H2`/`H3` 落空而改口径；被否定即原样记录。
- 不重新引用任何 `L12M` 数字。

## 产物与位置

- D1 评估/封存：`icrl_softmax/evaluate_fp_xrule_002.py`，`results/FP-XRULE-002/claude/f1f2_K12`
- D2 分析/验证：`icrl_softmax/analyze_fp_xrule_002.py`、`icrl_softmax/verify_fp_xrule_002.py`
- D3 报告：`docs/research_branches/FP-XRULE-002/claude/first_result.md`

## 失败判据与停止条件

- 若 `H1`① 出现任一退化，停止并定位。
- 若 `H1`② 出现覆盖违规，说明长 horizon 下构造失效——最重要的可能负结果，原样报告。

## 记录区

- 2026-09-13：任务创建，假设与设计写于运行前。

## 记录区（续）：执行完成

- 2026-09-13：执行完成，报告 `docs/research_branches/FP-XRULE-002/claude/first_result.md`，封存 `results/FP-XRULE-002/claude/f1f2_K12`。墙钟 `26.0 min`。
- **H1–H6 六条全部 PASS**；八格 0 退化、0 覆盖违规。
- **核心结论**：按状态循环在 12 步内不自终止（F1 `60/96`、F2 `90/96` 走到 horizon），长度由 horizon 决定而非证书或规则；合取臂仍由门槛杀死；按状态优势来自部分更新（`95%/89%`、`61%/60%`）；替代品结论在长 horizon 下复现。
- 验证 `5592` 个重放步零差异（`2209` 次批次抽样，批次已改为每步抽一次、四格复用）。仍为**初步结果**（单执行者）。
