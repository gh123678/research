# FP-NETX-002：把注意力网络搬到新族

- 任务编号：FP-NETX-002
- 状态：ACTIVE（用户 2026-09-13 直接指令"你先做"）
- 日期：2026-09-13
- 基线提交：`0453a57`（`claude/FP-CENSUS-001`）
- 上游：[FP-NET-BOUND-001](../research_branches/FP-NET-BOUND-001/claude/first_result.md) 在原族上确认"网络与 numpy 在同一证书下逐格一致"；[FP-XRULE-001](../research_branches/FP-XRULE-001/claude/first_result.md)/[FP-XRULE-002](../research_branches/FP-XRULE-002/claude/first_result.md) 把两个机制搬到了 F1/F2 但**只用数组路线**——网络从未跨族测过，这是本任务要补的洞。

## 研究问题

1. 固定权重的 softmax 注意力网络在**新族**（F1 6×4 = 24 个对；F2 4×3 但 `R*=2.0`）上，是否仍然与数组路线**逐格一致**？
2. `float32` 网络的数值差是否随**对数**增长（12 对 → 24 对）？
3. 网络的认证与不退化结论是否跨族保持？

## 设计

与 FP-XRULE-001 **完全相同的族、环境、种子、路线、`K=4`、每步预算与批次计划**，改动只有两处：

- 规则固定为 **`perstate`**（按状态保守更新——这条线现在由它承载）；因此格子为 `producer × certificate` 而非 `rule × certificate`；
- 增加 `network` 生产者：`model.py` 的 `EndToEndMaskedSoftmaxExpectedSARSA` / `EndToEndFiniteSoftmaxExpectedSARSA`，从 `Q_0 = 0` 跑 `LAYERS` 层（维度通用版 `network_qhat`，`fs.N_STATES/N_ACTIONS` 硬编码已去除）。

四格：`{numpy, network} × {frozen, L12S(f=0.9, p=0.05)}`，共享每步批次。每族 96 条路线记录。

**内建交叉核对**：由于族、环境、种子、批次计划与 `K` 都与 FP-XRULE-001 相同，`numpy|perstate|*` 两格应当**逐位复现** FP-XRULE-001 的 `perstate|frozen` / `perstate|L12S`——若不成立，说明管线被改动，本任务结果作废。

## 可证伪假设（登记于运行前）

- **H1（强制，两项）**：① 每族每格每个发出步**逐分量不退化**；② 每步 `E_Q ≥` **该格自己生产者**的实测 `‖Q̂−Q^π‖∞`（0 覆盖违规）。
- **H2（跨族生产者一致）**：每个证书下，`network` 与 `numpy` 的总发出步数相对差 `≤ 20%`；两族的第一步判决一致率均 `≥ 80%`。
- **H3（网络精度）**：每一步 `network` 的实测 `‖Q̂−Q^π‖∞ ≤ 3 ×` 同格 `numpy` 的值。
- **H4（内建交叉核对）**：`numpy|perstate|frozen` 与 `numpy|perstate|L12S` 的每记录总价值增益与发出步数**逐位等于** FP-XRULE-001 的同名格。
- **H5（成本）**：报告每格单独运行的数据量、网络前向调用次数与墙钟。
- **H6（float32 差是否随对数增长）**：报告 F1（24 对）与 F2（12 对）上 `max|q̂_net − q̂_numpy|` 的最大值；登记为 `≤ 1e-4`，并如实给出两者之比。

## 允许方法

- 新建 `evaluate_fp_netxfam_001.py`（复用 `evaluate_fp_xfam_001` 的族定义、采样器、数组路线、`training_batch`、按状态规则；自行实现维度通用的 `network_qhat`）。
- 新建 `analyze_fp_netxfam_001.py`、`verify_fp_netxfam_001.py`。

## 禁止事项

- 不改族定义、环境、种子、路线、`K`、批次计划、判决规则语义、证书参数。
- 不改任何既有 `evaluate_*.py`（其哈希记录在既有封存 config 中）。
- 不把网络格的结论外推到"网络自身完成认证"：证书与判决仍由外部程序完成，网络只产生 `Q̂`。
- 不因 `H2`/`H4` 不成立而改口径。

## 产物与位置

- D1 评估/封存：`icrl_softmax/evaluate_fp_netxfam_001.py`，`results/FP-NETX-002/claude/f1f2_K4`
- D2 分析/验证：`icrl_softmax/analyze_fp_netxfam_001.py`、`icrl_softmax/verify_fp_netxfam_001.py`
- D3 报告：`docs/research_branches/FP-NETX-002/claude/first_result.md`

## 失败判据与停止条件

- 若 `H1`① 出现任一退化，停止并定位到族、记录、格与步。
- 若 `H1`② 出现覆盖违规，说明证书对网络在新族上不成立——最重要的可能负结果，原样报告。
- 若 `H4` 不成立，本任务结果作废并重跑。

## 记录区

- 2026-09-13：任务创建，假设写于运行前；已确认 `model.py` 的两个 `EndToEnd*ExpectedSARSA` 对 `6×4` 输入通用（输出形状正确）。

## 记录区（续）：执行完成

- 2026-09-13：执行完成，报告 `docs/research_branches/FP-NETX-002/claude/first_result.md`，封存 `results/FP-NETX-002/claude/f1f2_K4`。墙钟 `43.6 min`，`1536` 次网络前向。
- **H1–H6 六条全部 PASS**：八格 0 退化、0 覆盖违规；network 与 numpy 在每个族×证书下发出数差 `0.0%`、第一步判决 `96/96`；`H4` 内建交叉核对 `max |Δvalue| = 0.000e+00`（numpy 格逐位复现 FP-XRULE-001）；float32 差 `1.286e-05`（24 对）vs `1.718e-05`（12 对），**不随对数增长**。
- 验证 `3072` 个重放步零差异，两个生产者的 `q̂` 均从零重算后与封存逐位一致。仍为**初步结果**（单执行者）。
