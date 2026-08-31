# Direct-Q 与 V-first 双路线均衡探索执行计划

> 日期：2026-08-29  
> 依据：`docs/superpowers/specs/2026-08-29-direct-q-v-first-balanced-exploration-design.md`  
> 状态：已获用户授权执行

## 执行原则

- 两条路线共享 MDP、策略、轨迹、随机种子和总样本预算。
- 先完成 fixed-policy estimation，再进入 blockwise control。
- 原始论文中的结论、本项目新推导和经验观察分别标注。
- 新工作写入独立研究笔记、验证脚本和结果目录；在比较结论稳定前不重写现有论文主文。
- 任一路线失败时记录可复核的条件或反例，不用调参掩盖。

## Task 1：建立来源与假设对照表

**新建文件**

- `docs/research_branches/source_assumption_matrix.md`

**读取材料**

- `papers/Xie_2026_beyond_linear_attention.pdf`
- `论文_草稿/preliminaries.md`
- `论文_草稿/端到端_softmax_SARSA_构造性证明.md`
- `论文_草稿/method_experiments.md`
- 通过学术搜索找到的原始论文和正式预印本

**工作内容**

1. 逐项记录 Xie 型 fixed-policy theorem 的状态空间、奖励、轨迹、遍历性、coverage、kernel、步长和概率保证。
2. 标出哪些假设可直接用于状态链，哪些迁移到 pair-chain 时必须重新证明。
3. 补充 Markov-chain concentration、policy evaluation 与 plug-in Q recovery 的直接相关原始来源。
4. 为每个公式标注 `source theorem`、`derived here` 或 `empirical target`。

**验收**

- 表中不存在无来源的“已有定理”表述；
- Branch A 与 Branch B 的假设可以逐行比较；
- 列出尚不能证明的具体缺口。

## Task 2：完成 Branch A 理论草案

**新建文件**

- `docs/research_branches/branch_a_direct_q_theory.md`

**工作内容**

1. 定义 pair-chain \(P_X^\pi\)、奖励 \(r_X\)、占用率 \(\mu_X^\pi\) 与 population kernel \(M_X^\pi\)。
2. 证明 \(Q^\pi\) 是 pair-chain Bellman operator 的唯一不动点。
3. 逐步迁移 population contraction；重新推导含一般步长 \(\alpha\) 的收缩常数。
4. 分解 kernel leakage、有限轨迹、随机奖励和初始化误差。
5. 对 one-hot pair kernel 推出满足 diagonality 所需的 \(\eta\) 与 \(\mu_{X,\min}^\pi\) 关系。
6. 给出稀有动作下的 coverage barrier 或反例。

**验收**

- 每个等号和不等式均能追溯到已列假设；
- 不把 fixed-policy 结果扩张为 fully online control；
- 明确标出已闭合证明与仍为证明草图的部分。

## Task 3：完成 Branch B 理论草案

**新建文件**

- `docs/research_branches/branch_b_v_first_theory.md`

**工作内容**

1. 写出 weighted-softmax \(V^\pi\) evaluation bound 及其适用条件。
2. 证明 population recovery 引理：
   \[
   \|\bar Q_{\widehat V}-Q^\pi\|_\infty
   \le\gamma\|\widehat V-V^\pi\|_\infty.
   \]
3. 为有限样本 recovery 分离 reward、transition、kernel 与 action-conditioned concentration error。
4. 给出样本切分版本的高概率误差结构。
5. 推导 error 对每个 \((s,a)\) 计数、\(\pi_{\min}\) 和动作数的依赖。
6. 分析 oracle-V 时仍然存在的 recovery 下界或 coverage barrier。

**验收**

- 总体误差同时包含 V estimation 与 Q recovery；
- 不用 state coverage 替代 state-action recovery coverage；
- 主理论版本清楚说明数据依赖如何处理。

## Task 4：实现共享公式级验证器

**新建文件**

- `verify_fixed_policy_q_routes.py`

**尽量复用**

- `mdps.py`
- `model.py`
- `verify_end_to_end_sarsa.py`

**工作内容**

1. 构造可精确求解 \(V^\pi,Q^\pi\) 的小型确定性和随机 MDP fixture。
2. 实现 Direct-Q exact pair update 与 finite-softmax pair kernel update。
3. 实现 V-first population recovery、sample-split recovery 与 oracle-V recovery。
4. 逐项计算所有误差分量。
5. 加入未访问 pair、低动作概率、正负 TD residual 和零 action gap 边界例。

**运行**

```powershell
python verify_fixed_policy_q_routes.py
```

**验收**

- population \(V\to Q\) 误差不超过 \(\gamma\|\widehat V-V^\pi\|_\infty\) 加数值容差；
- exact Direct-Q 与直接 pair-chain reference 一致；
- finite-kernel 偏差分项覆盖实际误差；
- 所有边界 fixture 有显式断言并以 `PASS` 结束。

## Task 5：建立同预算 fixed-policy 实验

**新建文件**

- `evaluate_fixed_policy_q_routes.py`

**新建结果目录**

- `results/fixed_policy_q_routes/`

**输出**

- `config.json`
- `task_results.json`
- `summary.json`
- coverage、误差和 kernel 图表

**工作内容**

1. 在相同任务和共享轨迹上运行两个分支。
2. 扫描轨迹长度、动作数、\(\pi_{\min}\)、kernel sharpness、mixing 和 action gap。
3. Branch B 同时运行 sample-split、no-split 与 oracle-V 消融。
4. 两个分支同时运行 exact matching 与有限 softmax kernel。
5. 报告 \(Q\) 无穷范数误差、平均 pair error、greedy-action 识别率、effective counts、diagonal mass 和 bound slack。

**快速运行**

```powershell
python evaluate_fixed_policy_q_routes.py --tasks 5 --trajectory-lengths 256 1024 --quick
```

**正式运行**

```powershell
python evaluate_fixed_policy_q_routes.py --tasks 30 --trajectory-lengths 256 1024 4096 16384
```

**验收**

- 任一比较点使用相同总轨迹预算；
- 配置与随机种子完整保存；
- 每个汇总量可从 `task_results.json` 重算；
- 结果同时显示平均表现和最差 coverage 位置。

## Task 6：实现 blockwise control 对照

**新建文件**

- `evaluate_blockwise_q_routes.py`

**新建结果目录**

- `results/blockwise_q_routes/`

**工作内容**

1. 每个 block 内冻结 \(\pi_k\)，使用同预算分别估计 \(Q^{\pi_k}\)。
2. 用相同 tie-breaking 与策略改进规则得到 \(\pi_{k+1}\)。
3. 每个 block 重新计算 occupancy、coverage、kernel diagonal 和 action gap。
4. 记录真实回报、近似改进下界、错误动作切换和累计误差。

**运行**

```powershell
python evaluate_blockwise_q_routes.py --tasks 20 --blocks 10
```

**验收**

- 不复用上一策略块的 fixed-policy 假设数据；
- 两条路线每个 block 的轨迹预算相同；
- 明确报告非单调 block，不只报告最终均值。

## Task 7：交叉审查与比较报告

**新建文件**

- `docs/research_branches/direct_q_vs_v_first_report.md`

**工作内容**

1. 对照理论预测与实验曲线。
2. 分别列出两条路线最强结论、必要假设和反例。
3. 判断 V-first 是否真正降低总样本复杂度，或只是重分配误差。
4. 给出 Direct-Q、V-first 或无支配关系三种判定之一。
5. 说明是否值得进入 fully online control，以及还缺什么。

**验收**

- 所有数字链接到机器可读结果；
- 所有外部结论链接到原始来源；
- “已证明”“证明草图”“经验观察”标注一致；
- 不根据单一指标宣布胜者。

## Task 8：决定论文整合范围

只有 Task 1–7 完成后，才决定是否修改：

- `论文_草稿/preliminaries.md`
- `论文_草稿/method_experiments.md`
- `论文_草稿/conclusion.md`

若结果尚不稳定，则研究分支材料保持独立，不改现有论文主线。

