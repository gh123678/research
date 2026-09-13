# FP-CERTFIX-001：证书重推——有效保证的最小参考协议

- 任务编号：FP-CERTFIX-001
- 状态：ACTIVE（治理例外：本任务由用户 2026-09-13 直接授权 Claude 起草并执行，代替"GPT 起草 + Claude 预审"流程；授权记录见会话：用户选择"阶段1剩余+阶段2+阶段3+阶段4"全部四项。本任务仅覆盖阶段 2。）
- 日期：2026-09-13
- 基线提交：`a2df7f8`（2026-09-13 勘误提交之后，`claude/FP-CENSUS-001`）
- 执行分支：`claude/FP-CERTFIX-001`
- 上游依据：GPT《科研工作审阅与规划（2026-09-13）》第二节（最高优先级：认证的数学依据）；Claude 核查存档 `docs/research_branches/2026-09-13-review-audit-claude.md`（12 项裁定全部属实）。

## 研究问题

在 4 状态 / 3 动作、行为策略 `pi_min=0.15` 的冻结 MDP 族上，能否构造一个**前提明确、可独立审读**的残差证书与多步策略更新协议，使得：

1. 单步：对固定目标策略 `pi` 与固定估计 `Qhat`，证书 `E_Q` 以至少 `1−δ` 的概率覆盖 `‖Qhat − Q^pi‖_∞`；
2. 多步：对整条长度不超过 K 的轨迹，以至少 `1−δ_total` 的概率，**每一步**的证书同时有效（覆盖声明是单次的还是整轨迹的，必须写清）；
3. 实现与推导逐行对应。

## 要修复的三个确定缺陷（证据位置见核查存档）

1. **frozen 证书均值步**：把估计尺度 `s_x` 代入需要真实值域的 Hoeffding（`fixed_policy_variance_certificate.py:175`）。
2. **Bernstein 臂二阶矩余量少 `√2`**：`fixed_policy_bernstein_certificate.py:263` 的余量是正确 Hoeffding 余量的 `1/√2`（方向：过紧，保证比宣称更弱）。
3. **随机计数与多步适应性**：每对样本量由访问计数随机决定后切半（`fixed_policy_bernstein_certificate.py:111-124`）；同一认证集被 12 步复用且下一步策略由它筛选（`evaluate_fp_iter8x_001.py:120,144,191`）。Maurer–Pontil Thm 4 要求 iid、固定 n；随机计数与适应性都不在其前提内。

## 可证伪假设

- **H1（桥梁引理）**：行为链对 `(s,a)` 的第 j 次访问的后继状态，是在访问时刻从 `P(·|s,a)` 新鲜抽取、与访问计数过程满足停时论证所需条件独立性的样本。因此"每对保留**前 n 次**访问（n 预先固定）、不足 n 则弃权"的协议，在"所有对都达到 n"的条件下事件下，给出每对 iid、固定 n 的样本。裁定方式：推导完整性审读 + 实现与推导一致性核对。
- **H2（覆盖，经验必要非充分）**：在修正协议下重跑 48 条记录的第一步与短轨迹时，oracle 审计 `‖Qhat−Q^π‖∞ ≤ E_Q` 在每步成立（0 违规）。注意：0 违规不证明界成立（FP-TIGHT 已示不成立的臂也能 0 违规），它只作为失败检测器。
- **H3（有效性的代价）**：在同数据规模（每对相同 n）下，MP 证书的 `E_Q` 与 empB 臂不同；报告第一步发出数与 empB@8x 封存值（44/48）的差异。若修正后第一步发出数大幅下降甚至为 0，**作为真实负结果保留**。
- **H4（风险账务）**：单轨迹、K 步、每步 d 对、每对 2 个方向（或 3 个事件）的风险分配总和 ≤ `δ_total = 0.05`，且由代码常量显式可加。

## 输入

- 冻结 MDP 族与种子计划：`fixed_policy_expected_sarsa_scaled.build_task`（24 个环境 × 2 条路线 = 48 条记录）。
- 采样器：`fp_sample_vectorised_batch.vectorised_batch`（已通过 FP-SAMPLE-001 门禁）。
- 参考封存结果：FP-ITER8X-001 / FP-HORIZON-001 / FP-SHORT-001 的封存 JSON（只读）。
- 文献：Maurer & Pontil 2009（empirical Bernstein，Thm 4）；标准 Bellman 收缩论证。

## 协议设计（推导文档中完整给出）

1. **固定计数**：每对预先固定 `n_per_pair`（在任务冻结时选定，例如 65536）；每对保留行为链中**前 n 次**访问的残差；任一对不足 n 则整条记录该步弃权（`heldout_pair_support_missing`）。禁止按实际访问数切半。
2. **证书（默认臂 MP）**：对每对的固定 n 个 iid 残差直接应用 Maurer–Pontil Thm 4（双侧，每对每方向 `δ_step/(2d)`），范围常数 `Y ∈ [−E, E]`，`E = 2·R*/(1−γ) = 10` 为已知量（`divergence_guard` 保证 `‖Qhat‖∞ ≤ R*/(1−γ)`）。不切半、不估二阶矩尺度。
3. **对照臂（保守修复，可选）**：保留切半结构但修正为——预先固定每半 m，前 2m 次访问；二阶矩步用正确 Hoeffding（含 `√2`）；均值步用 Bernstein（以 half-A 上界 `v_x` 与真实值域）。用于量化"最小修复"与 MP 的差距。
4. **多步（适应性修复）**：每步 k 在 `π_{k−1}` 与 `Qhat_k` 确定后抽取**全新独立**认证批 `B_k`（种子计划预先固定且按步索引），使 `B_k ⊥ (train, B_1..B_{k−1})`；`δ_k = δ_total / K`，K 预先冻结。轨迹级保证：`P(∩_{k≤K} Ω_k) ≥ 1 − δ_total`。
5. **Bellman 残差→价值误差**：`‖Qhat − Q^π‖∞ ≤ ‖ρ‖∞/(1−γ)`（收缩论证，推导文档写明）。
6. **改进判决不变**：`LB_s = Î_s − E_Q‖Δπ_s‖₁`，合取门槛 `min_s LB_s > 0`；在 `Ω_k` 上对所有候选 η 同时成立，故网格选择不引入额外适应性。

## 允许方法

- 新建 `fixed_policy_mp_certificate.py`（附加模块，不改封存文件）。
- 新建采样辅助：从向量化批次提取每对前 n 次访问（按链顺序）；按步生成新批次的种子计划。
- 新评估脚本 `evaluate_fp_certfix_001.py`（第一步 48 条记录 + 短 horizon 多步演示）与对应分析/验证脚本。
- 推导文档用中文，公式完整，逐步可核。

## 禁止事项

- 不修改任何封存文件（`fixed_policy_variance_certificate.py`、`fixed_policy_bernstein_certificate.py`、`fixed_policy_expected_sarsa*.py` 等）。
- 不把旧证书的零违规当作新证书有效性的证据。
- 不延长 horizon 来"看多几步"；本任务 horizon 只用于演示协议运转（≤12 步）。
- 不把单次固定对象证明当成整轨迹证明；不在同一数据上既筛选策略又声称覆盖由该数据选出的对象。
- 不对外发布；不合并 main。

## 产物与位置

- D1 推导文档：`icrl_softmax/docs/derivations/FP-CERTFIX-001-certificate-rederivation.md`
- D2 证书模块：`icrl_softmax/fixed_policy_mp_certificate.py`
- D3 采样辅助 + 评估脚本：`icrl_softmax/fp_certfix_first_n.py`、`icrl_softmax/evaluate_fp_certfix_001.py`
- D4 结果：`icrl_softmax/results/FP-CERTFIX-001/claude/{smoke,formal}/`
- D5 报告：`icrl_softmax/docs/research_branches/FP-CERTFIX-001/claude/first_result.md`
- D6 验证脚本：`icrl_softmax/verify_fp_certfix_001.py`（从封存输出独立重算头条数字）

## 验收标准

1. D1 中每个不等式标注名称、前提（iid/固定 n/有界）、风险花费；桥梁引理有完整证明；多步保证声明明确区分"单次"与"整轨迹"。
2. D2 与 D1 逐行对应；`delta_each` 等风险分配量可从输出直接加总验证（H4）。
3. smoke 与 formal 运行完成；H2 零违规；所有失败/弃权记录保留。
4. D6 能从 D4 封存输出重算每个头条数字。
5. 若 H3 显示修正后更新数大幅下降或为 0，报告保留该负结果，不用旧错误界撑结论。

## 失败判据与停止条件

- 若桥梁引理在当前行为采样下无法证明（例如停时论证不成立），停止实现，回写推导缺口，转入"预先固定每对 iid 直接采样（oracle 转移核）"的替代协议评估——该替代改变数据访问声明，需在报告中明示。
- 若 MP 证书在 n_per_pair = 65536 下第一步 0 发出，不继续加数据，先报告负结果。

## 预计耗时与资源

- 推导 + 审读自检：1–2 天等效。
- 实现 + smoke：0.5–1 天。
- formal（48 记录 × 第一步 + ≤12 步演示）：向量化采样约小时级。

## 双方范围与验证

- 本次为用户授权的治理例外：Claude 同时承担起草与执行。独立验证缺口照旧：**本任务全部结果保持"初步结果"**，直到 GPT 或用户指定的独立方完成推导审读与结果重跑。验证报告要求照旧（PASS/FAIL/OBJECTION，指向可检查证据）。
- 交接记录、异议与用户裁决写回本文件下方。

## 记录区

- 2026-09-13：任务创建（用户直接授权）。阶段 1（封存证据备份）已于同日完成：`docs/evidence/2026-09-13-results-backup.md`，2209/2209 文件哈希验证通过。
