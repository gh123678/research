# Direct-Q 与 V-first 双路线交叉审查报告

> 日期：2026-08-29  
> 范围：fixed-policy estimation 与 blockwise control  
> 判定：理论上无统一支配；无 gap cross-fit 已在正式同预算实验中追平 Direct-Q，但 Direct-Q 仍是当前证明主线

## 1. 结论先行

在当前设定下，Direct-Q 应作为证明主路线继续推进；V-first 已从“备选消融”升级为实证上有竞争力的并行路线，但仍不应宣称其中一条在所有条件下支配另一条。

理由分三层：

1. **已证明的结构**：Direct-Q 把 \(Q^\pi\) 直接视作 state-action pair chain 的价值函数；V-first 则有稳定的 population recovery 界
   \[
   \|\bar Q_{\widehat V}-Q^\pi\|_\infty
   \le \gamma\|\widehat V-V^\pi\|_\infty.
   \]
   两者都正确，但 V-first 的 recovery 阶段仍需要 state-action coverage。
2. **第二阶段正式 fixed-policy 实验**：480 个匹配比较中，Direct-Q exact 的平均误差为 0.5852；无 gap V-first cross-fit 为 0.5873，配对差 \(+0.0022\)，95% CI 为 \([-0.0079,0.0122]\)。这支持“经验追平”，不支持任一路线统一占优。
3. **blockwise control 实验**：20 个任务、每任务 10 块、每块 1024 个新 transition 下，Direct-Q exact 的平均最终回报为 1.5793，V-first exact 为 1.5117；但两者都出现约三成非单调块，因此 fixed-policy 结果不能直接升级为单调控制定理。

V-first 的 no-split 版本在第二阶段仍然最好：总体平均误差 0.5671，较 Direct-Q 的配对差为 \(-0.0180\)，95% CI 为 \([-0.0237,-0.0124]\)。不过它重复使用同一批数据做 \(V\) 估计和 \(Q\) recovery，当前高概率论证不能覆盖它。无 gap cross-fit 保留全部 recovery transition 并消除了大部分切分损失；带 gap cross-fit 则因删除短轨迹中的 15%–22% recovery 数据而退化到 0.6494。

## 2. 理论结构对照

| 维度 | Branch A：Direct-Q | Branch B：V-first |
|---|---|---|
| 估计对象 | pair chain 上的 \(Q^\pi(s,a)\) | 先估计 state chain 上的 \(V^\pi(s)\)，再恢复 \(Q^\pi(s,a)\) |
| population 固定点 | \(Q^\pi=r_X+\gamma P_X^\pi Q^\pi\) | \(Q^\pi(s,a)=\mathbb E[R+\gamma V^\pi(S')\mid s,a]\) |
| softmax kernel 主条件 | pair-kernel 对角质量足够大 | V 阶段需要 state-kernel；recovery 仍需要 pair-kernel 或条件均值 |
| 一般步长收缩 | 若 \(M(x,x)\ge(1+\gamma)/2+C_A\)，则条件收缩因子为 \(1-2\alpha C_A\) | 继承 state-value 评估收缩；recovery 本身是一次 \(\gamma\)-Lipschitz 映射 |
| coverage | 全程依赖 \((s,a)\) 访问 | 长时域迭代只依赖 state coverage，但最后一步不能消除 pair coverage |
| 当前主要风险 | 稀有动作使 pair occupancy 与 kernel 对角条件恶化 | 样本切分损失一半 recovery 数据；no-split 数据依赖尚未闭合 |
| 适合场景 | 目标就是 \(Q^\pi\)，且动作覆盖可控 | 已有高质量 \(V\) 估计器、预训练 value prior，或多个动作共享 state 表征 |

Direct-Q 的完整推导见 [branch_a_direct_q_theory.md](branch_a_direct_q_theory.md)，V-first 的推导见 [branch_b_v_first_theory.md](branch_b_v_first_theory.md)。来源与假设逐项对照见 [source_assumption_matrix.md](source_assumption_matrix.md)。

这里对 Xie 等人的结果只作 fixed-policy state-value 基线使用。其正式有限样本分析采用遍历有限状态 MRP、状态访问与特定 kernel 对角条件；把它迁移到 pair chain、随机 transition reward 和控制循环都需要新证明。[Xie et al. (2026)](https://arxiv.org/abs/2605.07333)

Markov 轨迹的浓缩工具可从一般状态空间 Markov-chain Hoeffding 不等式继续推进，但尚未在本项目里代入所有 mixing 与 occupancy 常数。[Fan, Jiang & Sun (2021)](https://www.jmlr.org/papers/v22/19-479.html) 同时，同步生成模型下的 policy-evaluation 上界不能直接当成单轨迹结果。[Pananjady & Wainwright (2020)](https://arxiv.org/abs/1909.08749) 单轨迹 policy iteration 的已有正结果也依赖额外算法结构，说明从 fixed-policy 到 control 并非自动成立。[Winnicki & Srikant (2023)](https://proceedings.mlr.press/v206/winnicki23a.html)

## 3. 公式级契约验证

[verify_fixed_policy_q_routes.py](../../verify_fixed_policy_q_routes.py) 对以下性质给出确定性断言：

- pair-chain 平稳分布公式与 \(Q^\pi\) Bellman 方程；
- Direct-Q population 固定点与一般步长的对角占优算子界；
- finite pair-kernel 的单步 leakage 上界；
- population \(V\to Q\) 的 \(\gamma\)-Lipschitz 界；
- sample-split ratio 分解与总误差合成界；
- 稀有动作未访问、确定性转移、正负 TD residual、零 action-gap 边界。

当前运行的关键数值为：

| 契约 | 实际值 | 上界或容差 |
|---|---:|---:|
| pair stationary formula error | (9.714\times10^{-17}) | (10^{-10}) |
| Direct-Q 实际算子无穷范数 | 0.805000 | 0.914046 |
| finite write-back 最大误差 | (6.049\times10^{-4}) | (1.441\times10^{-3}) |
| V-first population recovery error | 0.04526 | 0.1257 |
| sample-split 总 \(Q\) 误差 | 0.008997 | 0.01160 |
| 确定性 fixture Bellman error | 0 | (10^{-12}) |
| zero-gap fixture 最大动作差 | 0 | (10^{-12}) |

这些是公式与实现一致性的数值证据，不替代有限样本概率定理。

## 4. Fixed-policy 同预算实验

### 4.1 协议

完整配置与逐任务结果：

- [config.json](../../results/fixed_policy_q_routes/config.json)
- [task_results.json](../../results/fixed_policy_q_routes/task_results.json)
- [summary.json](../../results/fixed_policy_q_routes/summary.json)

快速扫描包含：

- \(n_S=6\)，\(n_A\in\{2,4\}\)；
- \(\pi_{\min}=0.05\)；
- mixing \(\in\{0.08,0.50\}\)；
- action-0 reward bonus \(\in\{0,0.5\}\)；
- softmax sharpness \(\beta\in\{6,10\}\)；
- 每个配置 5 个任务，轨迹长度 \(N\in\{256,1024\}\)。

相同任务中的路线共享 MDP、策略、起始分布与轨迹。Direct-Q 使用全部 \(N\) 个 transition；V-first split 使用前半段估计 \(V\)，后半段恢复 \(Q\)。no-split 与 oracle-V 是诊断消融，不混入主公平比较。

### 4.2 主结果

| \(N\) | 路线 | 平均 \(Q\) sup error | greedy action accuracy | 平均缺失 pair |
|---:|---|---:|---:|---:|
| 256 | Direct-Q exact | 1.2282 | 0.775 | 2.35 |
| 256 | V-first split exact | 1.5017 | 0.700 | 5.83 |
| 256 | V-first no-split | 1.2004 | 0.787 | 2.35 |
| 256 | V-first oracle-V | 1.4652 | 0.667 | 5.83 |
| 1024 | Direct-Q exact | 0.6144 | 0.858 | 0.23 |
| 1024 | V-first split exact | 0.9540 | 0.804 | 1.15 |
| 1024 | V-first no-split | 0.5650 | 0.879 | 0.23 |
| 1024 | V-first oracle-V | 0.9405 | 0.804 | 1.15 |

在 \(N=256\) 时，仅 1/40 个独立任务配置同时覆盖 Direct-Q 全段与 V-first recovery 半段的全部 pair；在 \(N=1024\) 时为 16/40。结果文件中 exact 路线按两个 \(\beta\) 标签重复保存，所以机器记录数分别显示为 2/80 与 32/80。

oracle-V 与 split V-first 接近，说明当前误差的主要来源不是 \(V\) 的长时域估计，而是后半段 action-conditioned recovery。另一方面，no-split 利用全部样本同时做两阶段估计，经验上消除了大部分样本切分损失。

softmax 结果也提供了一个细节：Direct-Q finite kernel 在迭代收敛后与 exact matching 达到相同经验固定点；V-first 的一次 recovery 不会迭代消除 kernel 混合偏差，但 \(\beta=6\) 的平滑在此扫描中平均降低了约 0.0383 的 sup error，表现为有限样本下的偏差—方差折中，而不是普遍改进定理。

![Fixed-policy error](../../results/fixed_policy_q_routes/q_sup_error.png)

![Coverage versus error](../../results/fixed_policy_q_routes/coverage_vs_error.png)

## 5. 第二阶段：cross-fit 与 Markov certificate

### 5.1 新增对象与证据边界

第二阶段实现并验证了四条新路线：

- 无 gap 的 two-fold V-first cross-fit，exact 与 finite-softmax recovery；
- 由 state/pair/edge chain 的正反向 total-variation mixing 诊断选择 gap 的 blocked cross-fit，exact 与 finite-softmax recovery。

每个 recovery fold 都使用另一个 fold 得到的 value estimate；无 gap 版本保留全部 \(N\) 个 recovery transition，带 gap 版本在折边界两侧各删除 \(g\) 个 transition。公式、Markov 常数与诚实边界见 [crossfit_markov_certificate_theory.md](crossfit_markov_certificate_theory.md)，实现契约由 [verify_crossfit_markov_certificate.py](../../verify_crossfit_markov_certificate.py) 检查。

certificate 同时计算：

- state、pair 与 edge chain 的右谱 Hoeffding inflation；
- pair occupancy 的 stationary coverage radius；
- population/empirical one-hot kernel diagonal slack；
- 正反向最坏 total-variation dependency at gap；
- gap 是否因 \(N/8\) 预算上限而截断。

这里必须区分两种说法：pair-count Hoeffding 条件和 kernel diagonal 可以称作 primitive certificate；gap 处的 total variation 目前只是依赖诊断。它尚未被组合成 blocked cross-fit 的完整有限样本定理。

### 5.2 正式协议

完整产物：

- [config.json](../../results/fixed_policy_q_routes_crossfit/config.json)
- [task_results.json](../../results/fixed_policy_q_routes_crossfit/task_results.json)
- [summary.json](../../results/fixed_policy_q_routes_crossfit/summary.json)

扫描包含 30 个独立随机任务，\(n_S=6\)、\(n_A=4\)、\(\pi_{\min}=0.05\)、\(\beta=8\)，并交叉：

- \(N\in\{256,1024,4096,16384\}\)；
- mixing \(\in\{0.08,0.5\}\)；
- reward gap bonus \(\in\{0,0.5\}\)。

总计 480 个匹配比较。相同任务、样本长度、mixing 与 gap bonus 下的全部路线共享 MDP、策略和轨迹。报告的置信区间是任务级配对差的双侧 95% Student-\(t\) 区间。

### 5.3 主结果

| \(N\) | Direct-Q | V-first split | V-first no-split | V-first cross-fit | V-first gap cross-fit |
|---:|---:|---:|---:|---:|---:|
| 256 | 1.3230 | 1.5742 | 1.3077 | 1.3476 | 1.4633 |
| 1024 | 0.6402 | 1.0606 | 0.6147 | 0.6498 | 0.7701 |
| 4096 | 0.2570 | 0.3502 | 0.2326 | 0.2387 | 0.2491 |
| 16384 | 0.1205 | 0.1433 | 0.1135 | 0.1132 | 0.1152 |
| **总体** | **0.5852** | **0.7821** | **0.5671** | **0.5873** | **0.6494** |

总体 95% CI 分别为：

- Direct-Q：\([0.5330,0.6373]\)；
- no-split：\([0.5150,0.6193]\)；
- no-gap cross-fit：\([0.5340,0.6406]\)；
- gap cross-fit：\([0.5915,0.7073]\)。

相对 Direct-Q 的任务级配对结论更有判别力：

| 路线减 Direct-Q | 平均差 | 95% CI | 胜 / 平 / 负 |
|---|---:|---:|---:|
| V-first split | +0.1969 | [0.1582, 0.2355] | 19.0% / 5.6% / 75.4% |
| V-first no-split | -0.0180 | [-0.0237, -0.0124] | 54.6% / 16.3% / 29.2% |
| V-first cross-fit | +0.0022 | [-0.0079, 0.0122] | 50.6% / 13.3% / 36.0% |
| V-first gap cross-fit | +0.0642 | [0.0425, 0.0860] | 44.8% / 9.2% / 46.0% |

“胜”表示该 V-first 路线误差更低。无 gap cross-fit 与 Direct-Q 的配对区间跨过 0，现有扫描不能区分两者；no-split 的小幅优势稳定，但仍属于同样本 plug-in 诊断，不能借用 cross-fit 论证。

finite-softmax 没有形成统一收益：no-gap cross-fit 从 0.5873 变为 0.5941，gap cross-fit 从 0.6494 变为 0.6529。当前 \(\beta=8\) 下，平滑偏差与有限样本方差的净效应依场景变化。

![Cross-fit error](../../results/fixed_policy_q_routes_crossfit/crossfit_error.png)

### 5.4 coverage 与 gap 诊断

| \(N\) | 经验全 pair 覆盖 | kernel certified | gap 被截断 | gap recovery 保留率 |
|---:|---:|---:|---:|---:|
| 256 | 2.5% | 2.5% | 52.5% | 78.1% |
| 1024 | 85.0% | 61.7% | 50.0% | 84.8% |
| 4096 | 100% | 95.8% | 0% | 94.0% |
| 16384 | 100% | 100% | 0% | 98.3% |

最保守的 stationary pair-count Hoeffding coverage certificate 在 480 个配置上均未通过，因为
\(\varepsilon_B\le \mu_{X,\min}/2\) 的 worst-case radius 仍大于最小 pair occupancy。这不表示长轨迹没有覆盖：\(N\ge4096\) 时经验全覆盖率已经是 100%。它表示当前 union-bound/sup-norm certificate 太保守，不能把经验覆盖升级为高概率保证。

带 gap cross-fit 的短轨迹退化与有效样本直接对应：\(N=256\) 和 \(1024\) 时平均只保留 78.1% 和 84.8% 的 recovery transition，且约一半任务的目标 gap 被 \(N/8\) 上限截断。到 \(N=16384\) 时保留率升至 98.3%，gap cross-fit 的平均误差 0.1152 已接近 no-gap cross-fit 的 0.1132。

![Recovery budget](../../results/fixed_policy_q_routes_crossfit/recovery_budget.png)

![Certificate versus error](../../results/fixed_policy_q_routes_crossfit/certificate_vs_error.png)

### 5.5 第二阶段判定

1. **cross-fit 已解决主要的样本切分损失。** 它把 recovery 预算从 \(N/2\) 恢复到 \(N\)，总体上追平 Direct-Q。
2. **显式 gap 暂不适合作为默认估计器。** 它提供更清晰的依赖诊断，但短轨迹删除样本的代价显著；应作为理论消融保留。
3. **no-split 仍是经验最优基线。** 下一理论问题不是再做更大扫描，而是用 stability、leave-neighbor-out 或直接 data-dependent operator perturbation 分析解释它。
4. **primitive certificate 有用但未闭合。** edge-chain 的加入修正了把 TD numerator 错当 pair-state function 的问题；coverage radius 的保守性则明确指出下一步需要 occupancy-adaptive/Bernstein 型界，而不是扩大经验结论。

## 6. Blockwise control 对照

### 6.1 协议

完整结果：

- [config.json](../../results/blockwise_q_routes/config.json)
- [task_results.json](../../results/blockwise_q_routes/task_results.json)
- [summary.json](../../results/blockwise_q_routes/summary.json)

20 个随机任务均运行 10 个策略块。每块冻结当前路线自己的策略，重新采样 1024 个 transition，从零开始评估当前 \(Q^\pi\)，随后使用相同 \(\pi_{\min}=0.05\) exploratory-greedy 规则改进。不同路线的策略会在首块之后分叉，所以实验只保证相同 MDP、每块预算与耦合随机数，不声称分叉后仍共享同一条轨迹。

### 6.2 结果

| 路线 | 平均最终回报 | 平均回报增益 | 平均 \(Q\) sup error | 错误贪心动作比例 | 平均缺失 pair | 非单调块比例 |
|---|---:|---:|---:|---:|---:|---:|
| Direct-Q exact | 1.5793 | +1.6171 | 0.5582 | 0.121 | 0.10 | 0.305 |
| Direct-Q softmax | 1.5793 | +1.6171 | 0.5582 | 0.121 | 0.10 | 0.305 |
| V-first exact | 1.5117 | +1.5495 | 1.0193 | 0.160 | 0.81 | 0.340 |
| V-first softmax | 1.5360 | +1.5738 | 0.9964 | 0.153 | 0.83 | 0.345 |

配对任务中，Direct-Q exact 的最终回报高于 V-first exact 的比例为 55%，持平为 30%，平均差为 +0.0676。finite-softmax 配对中 Direct-Q 的胜率为 35%、持平率为 45%，但少数较大优势使平均差仍为 +0.0433。这不足以宣布统计意义上的全面支配。

全部 800 个 route-block 更新都满足本实验使用的保守 exploratory-greedy 下界；但实际回报仍在约 30%–35% 的块下降。该现象与现有理论一致：无穷范数 \(Q\) 误差给出的是允许下降的近似改进下界，不是无 action-margin 条件的单调性。

![Return by block](../../results/blockwise_q_routes/return_by_block.png)

![Q error by block](../../results/blockwise_q_routes/q_error_by_block.png)

## 7. 证据等级与剩余缺口

### 已闭合的推导

- \(Q^\pi\) 与 pair-chain value 的等价；
- Direct-Q population 固定点及一般 \(\alpha\) 的条件收缩界；
- one-hot pair kernel 的 sharpness—occupancy 关系；
- V-first population recovery 的 \(\gamma\)-Lipschitz 界；
- sample-split ratio 误差分解；
- exploratory-greedy 的保守单块回报下界。

### 数值契约验证

- 固定点、算子范数、kernel leakage、ratio 分解、组合误差；
- 未访问动作、确定性转移、正负 residual、零 action-gap。
- two-fold block accounting、opposite-fold target isolation 与 full-budget recovery；
- state/pair/edge chain 的谱、正反向 mixing、gap 截断与 coverage/kernel certificate。

### 正式扫描观察

- sample-split V-first 的半预算 recovery 显著落后；
- 无 gap cross-fit 已在配对置信区间意义上追平 Direct-Q；
- V-first no-split 小幅领先，但其同样本依赖未闭合；
- 显式 gap 在短轨迹中产生明显删样本代价；
- finite-softmax 在正式 \(\beta=8\) 扫描中没有统一优势；
- blockwise 平均提升不意味着逐块单调。

### 尚未闭合

1. 把已实现的 pair-count 与 edge-numerator primitive bounds 组合成所有 Direct-Q 迭代层同时成立的 operator perturbation theorem；
2. 把 transition-dependent bounded reward 的 edge-chain 处理写成正式高概率定理，而不只停留在 certificate 组件；
3. 为无 gap cross-fit 建立相邻 Markov 折的 coupling/stability 界，或直接闭合 V-first no-split 的同样本依赖；
4. 优于保守 \(1/\mu_{X,\min}\) ratio 界的 \(1/\sqrt{n\mu_{X,\min}}\) 级结论；
5. 随策略改变的 occupancy、mixing、kernel diagonality 与 action gap 的跨块联合控制；
6. 在未知 transition kernel 下，用可观测量估计或上界 certificate 常数。

## 8. 路线判定与论文整合决定

当前判定是：

- **证明主线：Direct-Q。** 它与控制目标直接对齐，pair/edge-chain primitive certificate 已有明确接口；剩余工作是处理同轨迹迭代算子的 data dependence。
- **并行实证路线：V-first no-gap cross-fit。** 它已消除半样本 recovery 代价并在 480 个匹配比较中追平 Direct-Q，值得保留为论文中的同预算对照和潜在第二定理。
- **诊断上界：V-first no-split。** 它是当前经验最佳路线，但在同样本依赖闭合前，不作为理论主结果。
- **理论消融：gap cross-fit。** 保留其依赖—样本量权衡，不作为默认算法。
- **不进入 fully online control 定理。** blockwise 结果已说明非单调块普遍存在，尚缺跨块联合保证。
- **可以进入论文附录候选，但不改主定理。** 正式长轨迹扫描、配对区间和 Markov primitive certificate 已完成，足以整理为 fixed-policy methodology/ablation 附录；完整高概率 cross-fit 或 Direct-Q 迭代定理完成后再升级正文主张。

因此，这次探索不是简单选出“赢家”，而是把问题收敛为两个清晰的证明任务：Direct-Q 完成 pair/edge-chain 迭代扰动定理；V-first 解释为什么 full-budget cross-fit/no-split 能在不牺牲 recovery coverage 的情况下成立。
