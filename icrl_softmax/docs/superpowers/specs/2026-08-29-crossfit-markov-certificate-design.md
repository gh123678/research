# V-first Cross-fit 与 Direct-Q Markov Certificate 设计

> 日期：2026-08-29  
> 状态：用户已批准三节口头设计，等待书面规格复核  
> 上游结论：docs/research_branches/direct_q_vs_v_first_report.md

## 1. 目标与范围

本阶段继续均衡推进两条 fixed-policy 路线：

- **Direct-Q**：把现有 pair-chain 有限样本证明缺口收紧为可计算的 Markov、coverage 与 kernel certificate。
- **V-first**：实现无间隙与带间隙的两折 blocked cross-fit，检验能否保留 no-split 的样本效率，同时显式暴露跨块 Markov 依赖。

本阶段只研究 fixed-policy estimation。只有这些结果稳定后，才决定是否把 cross-fit 加入 blockwise control；不在本阶段声称 fully online control 定理，也不修改论文主文。

## 2. 设计选择

采用独立模块，而不是继续把所有逻辑堆入现有实验入口：

1. **crossfit_vfirst.py**
   - 负责构造两折数据块；
   - 分别估计两套 \(V\)；
   - 用另一块的 \(V\) 构造每个 recovery target；
   - 按 pair 实际计数合并两侧估计；
   - 返回估计值、样本使用量和误差分解所需诊断量。
2. **markov_coverage_certificate.py**
   - 计算 state chain、pair chain 与 transition-edge chain 的平稳分布；
   - 计算多步总变差/Dobrushin 型收缩指标；
   - 给出经验 pair counts、kernel 对角质量、所需 sharpness 和阈值余量；
   - 为带间隙 cross-fit 建议 gap，并报告是否被预算上限截断。
3. **evaluate_fixed_policy_q_routes.py**
   - 保持实验入口职责；
   - 调用独立 cross-fit 与 certificate 模块；
   - 统一保存路线指标、配置、汇总和图表。

不把 certificate 写成“定理证明器”。如果原始浓缩定理的假设尚未全部满足，输出必须标为 diagnostic 或 conditional certificate。

## 3. Cross-fit 数据流

长度为 \(N\) 的轨迹被划分为：

\[
A=[0,L),\qquad
G=[L,L+2g),\qquad
B=[L+2g,N),
\]

其中

\[
L=\lfloor(N-2g)/2\rfloor,
\]

\(G\) 是总长度 \(2g\) 的隔离区；当两侧长度相差 1 时，多出的 transition 归入 B。实现上把中点左右各丢弃 \(g\) 个 transition，因此：

- 无间隙：\(g=0\)，两折 recovery 合计使用 \(N\) 个 transition；
- 带间隙：\(g>0\)，两折 recovery 合计使用 \(N-2g\) 个 transition。

两条 recovery 流为：

\[
A\longrightarrow\widehat V_A
\longrightarrow B\text{ 上恢复 }Q,
\]

\[
B\longrightarrow\widehat V_B
\longrightarrow A\text{ 上恢复 }Q.
\]

因此，A 中每个 recovery target 必须使用 \(\widehat V_B\)，B 中每个 recovery target 必须使用 \(\widehat V_A\)。契约测试将通过人为设置两套明显不同的 \(V\) 来检测同块泄漏。

每个 pair \(x=(s,a)\) 的最终估计按实际计数合并：

\[
\widehat Q_{\mathrm{cf}}(x)
=
\frac{
N_{A,x}\widehat Q_{A\leftarrow B}(x)
+
N_{B,x}\widehat Q_{B\leftarrow A}(x)
}{
N_{A,x}+N_{B,x}
}.
\]

若某一折缺失 \(x\)，仅使用另一折；若两折都缺失，则保持显式 missing 状态，并按当前实验约定以零初始化计算误差。

## 4. 误差分解

定义两折对应的 population recovery：

\[
\bar Q_A(s,a)
=
\mathbb E[R+\gamma\widehat V_B(S')\mid s,a],
\]

\[
\bar Q_B(s,a)
=
\mathbb E[R+\gamma\widehat V_A(S')\mid s,a].
\]

按 pair 计数权重构造参考混合量 \(\bar Q_{\mathrm{cf}}\)。由每个 population recovery 的 \(\gamma\)-Lipschitz 性质，

\[
\|\bar Q_{\mathrm{cf}}-Q^\pi\|_\infty
\le
\gamma\max\{
\|\widehat V_A-V^\pi\|_\infty,
\|\widehat V_B-V^\pi\|_\infty
\}.
\]

有限样本数值分解为：

\[
\|\widehat Q_{\mathrm{cf}}-Q^\pi\|_\infty
\le
\gamma\max(\varepsilon_{V,A},\varepsilon_{V,B})
+
\varepsilon_{\mathrm{rec}},
\]

其中

\[
\varepsilon_{\mathrm{rec}}
=
\|\widehat Q_{\mathrm{cf}}-\bar Q_{\mathrm{cf}}\|_\infty.
\]

对高概率结论，额外记录

\[
\varepsilon_{\mathrm{dep}}(g),
\]

表示两个轨迹块之间的 Markov 依赖。带间隙版本将尝试用原始 Markov 浓缩定理和 forward/reverse mixing certificate 控制该项。无间隙版本令 \(g=0\)，但只保留经验分解，不宣称已有闭合的高概率保证。

## 5. Markov 与 coverage certificate

合成 MDP 的转移矩阵已知，因此 certificate 使用真实 \(P^\pi\) 与 \(P_X^\pi\)，而不是从同一短轨迹反推 mixing。

对 state chain、pair chain 和 transition-edge chain 分别计算：

1. 平稳分布及 \(\mu_{\min}\)；
2. 一步 Dobrushin 系数；
3. 最小的多步 \(k\)，使 \(P^k\) 的 Dobrushin 系数严格小于 1；
4. forward 与 stationary time-reversal chain 的总变差 mixing 时间；
5. 经验最小 count、缺失 pair 数和 occupancy 偏差；
6. one-hot softmax kernel 的最小对角质量；
7. 满足
   \[
   M(x,x)>(1+\gamma)/2
   \]
   的 sharpness 要求及实际阈值余量。

transition-edge chain 定义为 \(E_t=(X_t,X_{t+1})\)，用于承载 reward、TD residual 与 recovery numerator 这类依赖相邻 transition 的函数。固定置信水平 \(\delta=0.05\)。令 \(d_{\max}(k)\) 为 state/pair/edge、forward/reverse 六条总变差 mixing 曲线的最大值，原始建议间隙定义为

\[
g_{\mathrm{raw}}
=
\min\{k:d_{\max}(k)\le\delta/(4N)\}.
\]

若搜索上限内不存在这样的 \(k\)，则 \(g_{\mathrm{raw}}=\infty\)。实际实验间隙为

\[
g
=
\min\{\lfloor N/8\rfloor,g_{\mathrm{raw}}\}.
\]

若 \(g_{\mathrm{raw}}>N/8\) 或为无穷，保存 **gap_capped=true**，并禁止把该样本点描述为已充分去相关。该 gap 规则本身是可计算的依赖诊断；只有与核对后的原始浓缩定理假设完全对齐时，才升级为 conditional high-probability certificate。

## 6. 公平比较与路线

所有路线共享 MDP、固定策略、起始分布、原始轨迹和总采样预算：

- direct_exact
- direct_softmax
- vfirst_split_exact
- vfirst_split_softmax
- vfirst_nosplit_exact
- vfirst_crossfit_exact
- vfirst_crossfit_softmax
- vfirst_crossfit_gap_exact
- vfirst_crossfit_gap_softmax
- vfirst_oracle_exact

报告必须同时显示：

- 总 trajectory transitions；
- 实际用于 recovery 的 transitions；
- 被 gap 丢弃的 transitions；
- 每个 pair 的计数和缺失数；
- \(Q\) sup error、平均 pair error、greedy-action accuracy；
- 两折 \(V\) sup error；
- recovery-only error 与组合界余量；
- state/pair/edge mixing、kernel diagonal 与 contraction threshold slack。

no-split、无间隙 cross-fit 和带间隙 cross-fit 的理论等级必须分别标注，不能只按最终误差排序。

## 7. 实验矩阵

### 烟雾验证

- 2–3 个任务；
- 短轨迹；
- exact 与 finite-softmax；
- 人工边界 fixture。

### 聚焦正式扫描

- \(n_S=6\)；
- \(n_A=4\)；
- \(\pi_{\min}=0.05\)；
- mixing \(\in\{0.08,0.50\}\)；
- action-0 reward bonus \(\in\{0,0.5\}\)；
- \(N\in\{256,1024,4096,16384\}\)；
- 每格 30 个任务；
- exact matching 与 finite-softmax 同时运行。

正式扫描使用一个预先固定的 finite-softmax sharpness，避免再次把预算消耗在已经完成的 beta 探索上。具体值沿用上一轮表现稳定的 \(\beta=8\)。

## 8. 测试与验收

必须通过以下契约：

1. \(g=0\) 时 recovery 使用量严格为 \(N\)；
2. \(g>0\) 时 recovery 使用量严格为 \(N-2g\)；
3. A targets 只使用 \(\widehat V_B\)，B targets 只使用 \(\widehat V_A\)；
4. pair 计数合并与直接拼接 targets 后分组求均值一致；
5. cross-fit 数值总误差不超过 population-mixture error 与 recovery error 的三角和；
6. oracle \(V_A=V_B=V^\pi\) 时，population mixture 等于 \(Q^\pi\)；
7. 某一折缺 pair、两折都缺 pair、稀有动作、确定性转移、零 action-gap 均有显式断言；
8. certificate 对随机、sticky 和对称 chain 均返回有限、可解释的结果；
9. 新脚本通过静态检查和编译；
10. 现有 fixed-policy、blockwise 和 end-to-end SARSA 回归不退化。

正式结果必须报告：

- 均值与中位数；
- 以独立 MDP 任务为单位的 95% Student-\(t\) 均值置信区间；
- 配对胜率与平局率；
- 最差 coverage 位置；
- gap_capped 比例；
- 理论证书满足率。

## 9. 交付文件

新增：

- crossfit_vfirst.py
- markov_coverage_certificate.py
- verify_crossfit_markov_certificate.py
- docs/research_branches/crossfit_markov_certificate_theory.md
- results/fixed_policy_q_routes_crossfit/

修改：

- evaluate_fixed_policy_q_routes.py
- docs/research_branches/direct_q_vs_v_first_report.md

本阶段不修改：

- 论文_草稿/ 下的论文主文；
- fully online control 代码；
- 已有实验结果目录。

## 10. 明确的非目标

- 不声称无间隙 cross-fit 已解决 Markov 数据依赖；
- 不把 exact transition matrix 可计算的 certificate 冒充未知环境算法；
- 不因平均回报或平均误差较好而忽略 coverage failure；
- 不把 fixed-policy 结果直接升级为控制收敛定理；
- 不在本轮引入新的神经网络训练流程。
