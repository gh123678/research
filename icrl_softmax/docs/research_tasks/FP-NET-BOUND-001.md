# FP-NET-BOUND-001：把收紧的证书接到注意力网络那条路上

- 任务编号：FP-NET-BOUND-001
- 状态：ACTIVE（治理例外同本线：用户 2026-09-13 直接指令"好"）
- 日期：2026-09-13
- 基线提交：`4ff222e`（`claude/FP-CENSUS-001`）
- 上游依据：证书线收口于 [FP-ITER-BOUND-002](../research_branches/FP-ITER-BOUND-002/claude/first_result.md)（无模型 `E_Q` `−39.6%`、认证迭代 `36 → 99` 步）；网络线的既有结论是 [FP-ATTN-8X-001](../research_tasks/FP-ATTN-8X-001.md)（网络与 numpy 逐位一致到 12 步）与 [FP-CERTCHECK-001](../research_tasks/FP-CERTCHECK-001.md)（同协议、旧证书下的零分歧）。**两条线此前从未在收紧后的证书下合流。**

## 研究问题

项目的主问题：**固定权重的 softmax 注意力网络能否把上下文经验变成可认证的、不退化的策略改进？** 现在证书比之前紧 `40%`，该问句可以第一次在"不因为界太松而假装成功"的条件下回答：

1. 证书是否仍然覆盖**网络自己**的误差（而不是 numpy 的）？
2. 网络在收紧后的证书下能发出多少步、走多远、拿到多少价值？
3. **网络的认证判决是否与精确数组公式一致**——即网络是否在决策层面复现了精确计算？

## 设计

同一总体（`task_index 12..23` × 两种 mixing × 两条路线 = `48` 条记录），`K = 16` 预先冻结，每步 `16384` 链 × `64`，`δ_k = 0.05/K`。**八个格子共享每一步的认证批次**，各自独立演化：

| 维度 | 取值 |
|---|---|
| 生产者 | `numpy`（封存数组公式路线）/ `network`（`model.py` 的字面注意力网络，`network_qhat` 从 `Q_0 = 0` 跑 `LAYERS` 层） |
| 路线 | `expected_exact`（masked 网络）/ `expected_finite`（finite 网络） |
| 证书 | `frozen`（对照）/ `L12M(0.05)`（本线的最终臂，无模型） |

`δ_prop = 0.05·δ_step`。判决规则、`η` 网格、`δ_total` 一字不改。

## 可证伪假设（登记于运行前）

- **H1（强制，两项）**：① 每个格子的每个发出步**逐分量不退化**；② 每个格子的每一步 `E_Q ≥` 实测**网络自己**的 `‖Q̂−Q^π‖∞`（0 覆盖违规）。任一例外即 FAIL。
- **H2（收紧是否转移）**：`network × L12M(0.05)` 的总发出步数 `≥ 1.10 ×` `network × frozen`。
- **H3（价值是否转移）**：`network × L12M(0.05)` 的平均总价值增益 `≥ 1.20 ×` `network × frozen`。
- **H4（生产者对比）**：同一证书下，`network` 与 `numpy` 的总发出步数之差在 `±20%` 以内。
- **H5（决策层一致）**：第一步的发出/弃权判决，`network` 与 `numpy` 在 `≥ 80%` 的记录-路线格上一致（两侧均为 `48` 个格）。
- **H6（网络精度）**：每一步 `network` 的实测 `‖Q̂−Q^π‖∞ ≤ 3 ×` 同格 `numpy` 的实测值。

## 允许方法

- 新建 `evaluate_fp_net_bound_001.py`（复用 `evaluate_fp_certfix_001.make_producer` 与 `fixed_policy_tight_certificate`；不修改任何既有文件）。
- 新建 `analyze_fp_net_bound_001.py`、`verify_fp_net_bound_001.py`。
- 复用 `fp_sample_vectorised_batch`、`fp_certfix_first_n`、`fs.improvement_for`。

## 禁止事项

- 不改判决规则、`η` 网格、`δ_total`、`K`，不改任何封存文件，不训练网络（权重是结构给定的）。
- 不把 `numpy` 格的结果当作网络的证据，反之亦然：每个格子只证明自己。
- 不因为 `H4`/`H5` 不成立而把网络结果降级为"实现细节"——网络与精确计算的决策是否一致，本身就是主问题的一部分。
- 不对外发布；不合并 main。

## 产物与位置

- D1 评估/封存：`icrl_softmax/evaluate_fp_net_bound_001.py`，`results/FP-NET-BOUND-001/claude/fresh_K16`
- D2 分析/验证：`icrl_softmax/analyze_fp_net_bound_001.py`、`icrl_softmax/verify_fp_net_bound_001.py`
- D3 报告：`docs/research_branches/FP-NET-BOUND-001/claude/first_result.md`

## 失败判据与停止条件

- 若 `H1`① 出现任一退化，停止并定位到具体记录、格与步，不得丢弃该记录。
- 若 `H1`② 出现覆盖违规，即证书对网络**不成立**——这是本任务最重要的可能负结果，必须原样报告，不得改用 numpy 的误差重述。

## 记录区

- 2026-09-13：任务创建（用户直接指令"好"）。假设与 `K=16` 登记于确认性运行之前。

## 记录区（续）：执行完成

- 2026-09-13：执行完成，报告 `docs/research_branches/FP-NET-BOUND-001/claude/first_result.md`，封存 `results/FP-NET-BOUND-001/claude/fresh_K16`。
- **H1–H6 六条全部 PASS**：四个格子 0 退化、0 覆盖违规（审计对的是各格自己生产者的实测误差）；网络 `36 → 99` 步（2.75×）、价值 `1.673×`；生产者差 `0.0%`；第一步判决一致 `48/48 = 100%`；网络/numpy 实测误差比 `1.000`。
- 独立重放验证 `462` 步零差异，其中重算的网络输出与封存 `q_hat` **逐位一致**（`|Δ| = 0.0`）。
- 仍为**初步结果**（单执行者）；`0` 条轨迹走到 `K=16`，该限制来自总体而非生产者。
