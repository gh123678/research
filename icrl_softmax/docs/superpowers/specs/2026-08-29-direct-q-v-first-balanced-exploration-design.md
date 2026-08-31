# Direct-Q 与 V-first 双路线均衡探索设计

> 日期：2026-08-29  
> 状态：已确认设计，等待书面规格复核  
> 项目：`icrl_softmax`

## 1. 背景与目标

当前项目已经构造并验证了标准 softmax attention 上的状态—动作价值写回，包括精确 oracle-routed sampled-SARSA 构造、有限 logit 误差界以及闭环控制实验。新的研究问题不是重新证明现有构造，而是比较两条从 fixed-policy evaluation 通向 control 的路线：

1. **Branch A — Direct-Q**：把固定策略下的过程提升为状态—动作马尔可夫链，直接用 state-action softmax kernel 估计 \(Q^\pi\)。
2. **Branch B — V-first**：先用 weighted-softmax TD 估计 \(V^\pi\)，再通过 action-conditioned one-step estimation 恢复 \(Q^\pi\)。

本轮采用“均衡推进”：两条路线都必须形成可审查的理论草图、可运行的最小实现、同预算实验和明确失败条件。研究不预设胜者，目标是识别各自的适用区间和主要瓶颈。

为避免与现有论文中的“Route I: sampled SARSA / Route II: Boltzmann Expected SARSA”混淆，本设计始终使用 **Branch A** 与 **Branch B**。

## 2. 共同研究问题

第一阶段只研究固定策略 \(\pi\) 下的 \(Q^\pi\) estimation：

> 在相同 MDP、策略、轨迹和总样本预算下，Direct-Q 与 V-first 分别需要什么 kernel diagonality、coverage、mixing 和 action-gap 条件，才能得到足以支持可靠策略改进的 \(Q^\pi\) 估计？

固定策略比较完成后，两条路线共同进入 blockwise control：

\[
\pi_k\longrightarrow \widehat Q^{\pi_k}
\longrightarrow \pi_{k+1}.
\]

Fully online control 每一步同时改变策略、占用分布和目标函数，会破坏 fixed-policy 分析所需的稳定算子，因此不作为第一阶段硬目标。

## 3. 共同设定与记号

采用有限折扣 MDP \((\mathcal S,\mathcal A,p,r,\gamma)\)，其中 \(0<\gamma<1\)，奖励有界。固定策略 \(\pi\) 诱导状态链与状态—动作链。主理论版本要求相应链遍历，并显式记录：

- 状态平稳分布 \(\mu^\pi(s)\)；
- 状态—动作占用率
  \[
  \mu_X^\pi(s,a)=\mu^\pi(s)\pi(a\mid s);
  \]
- 最小状态占用率、最小相关动作概率及 mixing 参数；
- 每个状态和状态—动作对的实际访问计数；
- kernel 对角质量、非对角泄漏与上下文长度。

实验使用可精确解线性方程得到 \(V^\pi\) 和 \(Q^\pi\) 的有限表格 MDP，从而把估计误差与回报评估噪声分开。

## 4. Branch A：Direct-Q

### 4.1 算子

令 \(X_t=(S_t,A_t)\)。固定策略诱导的 pair-chain 转移为

\[
P_X^\pi((s,a),(s',a'))
=p(s'\mid s,a)\pi(a'\mid s').
\]

其即时奖励为

\[
r_X(s,a)=\mathbb E[R_{t+1}\mid S_t=s,A_t=a].
\]

因此 \(Q^\pi\) 是该状态—动作 MRP 的 value function。Direct-Q 采用 signed TD residual 与 state-action softmax kernel：

\[
\delta_t^Q
=R_{t+1}+\gamma Q(X_{t+1})-Q(X_t),
\]

\[
Q_{\ell+1}(x)
=Q_\ell(x)+\alpha_\ell\sum_tK_X(X_t,x)\delta_t^Q.
\]

### 4.2 理论目标

把已有 fixed-policy weighted-softmax TD 的 population-kernel 分析迁移到 pair-chain，目标形式为

\[
\|\widehat Q_A-Q^\pi\|_\infty
\le
\rho_A^L\|Q_0-Q^\pi\|_\infty
+\varepsilon_{K,A}
+\varepsilon_{n,A}
+\varepsilon_{r,A}.
\]

其中各项分别表示深度收缩、kernel 偏差、有限轨迹误差和随机奖励误差。证明必须重新核对，而不能只通过符号替换宣称 Xie 型定理自动成立。

### 4.3 核心障碍

Direct-Q 的 coverage 由 \(\mu_X^\pi(s,a)\) 控制。对 one-hot pair kernel，其 population diagonal 具有典型形式

\[
M_X^\pi(x,x)
=\frac{\mu_X^\pi(x)e^\eta}
{\mu_X^\pi(x)e^\eta+1-\mu_X^\pi(x)}.
\]

当 \(\pi(a\mid s)\) 很小时，即使状态覆盖良好，pair 对角质量也可能很差。Branch A 必须把这一点量化为 sharpness、轨迹长度或不可行条件，而不是仅靠提高温度隐藏。

## 5. Branch B：V-first

### 5.1 两阶段算子

第一阶段沿用 fixed-policy weighted-softmax TD：

\[
V_{\ell+1}(s)
=V_\ell(s)+\alpha_\ell\sum_tK_S(S_t,s)\delta_t^V,
\]

\[
\delta_t^V
=R_{t+1}+\gamma V_\ell(S_{t+1})-V_\ell(S_t).
\]

第二阶段定义 population recovery operator：

\[
\bar Q_{\widehat V}(s,a)
:=\mathbb E[R_{t+1}+\gamma\widehat V(S_{t+1})\mid s,a].
\]

有限数据实现以 action-conditioned 样本均值或相应 softmax kernel 估计该条件期望，得到 \(\widehat Q_B\)。

### 5.2 基础误差桥

若恢复阶段使用精确条件期望，则

\[
\|\bar Q_{\widehat V}-Q^\pi\|_\infty
\le
\gamma\|\widehat V-V^\pi\|_\infty.
\]

有限数据下，目标分解为

\[
\|\widehat Q_B-Q^\pi\|_\infty
\le
\gamma\|\widehat V-V^\pi\|_\infty
+\varepsilon_{\mathrm{reward}}
+\varepsilon_{\mathrm{transition}}
+\varepsilon_{K,B}.
\]

若直接估计 one-step target，可把后面三项合并为 action-conditioned concentration error，但必须保留其对每个 \((s,a)\) 计数的依赖。

### 5.3 依赖处理

主证明采用样本切分：同一总轨迹预算的一部分估计 \(V^\pi\)，其余部分恢复 \(Q^\pi\)。这样不额外增加数据，同时使第一版条件集中分析清晰。实验增加不切分版本，判断 sample splitting 的常数损失。

Branch B 的潜在优势是主体 value evaluation 只要求 state coverage；其潜在失败是最终恢复所有动作的 \(Q^\pi(s,a)\) 仍需要 state-action coverage。研究要判断它是真正降低总体样本复杂度，还是只把困难推迟到第二阶段。

## 6. 共同策略改进桥梁

两条路线最终都输出 \(\widehat Q\)。若

\[
\|\widehat Q-Q^\pi\|_\infty\le\xi,
\]

则报告已有的近似策略改进保证，并把误差通过 \(1/(1-\gamma)\) 放大到策略价值层。为了声称可靠地选中更优动作，还需检查 action gap；只有关键动作优势明显大于估计误差时，才报告严格动作识别或单调改进。

Blockwise control 中，每个策略块都要重新测量 occupancy、coverage 和 kernel diagonality。不能假定它们在策略更新后保持不变。

## 7. 统一实验架构

### 7.1 数据流

每个实验实例依次生成：

1. 一个有限随机 MDP；
2. 一个固定策略 \(\pi\)；
3. 精确 \(V^\pi\)、\(Q^\pi\) 与 action gap；
4. 一条指定总长度的共享轨迹；
5. Branch A 的 Direct-Q estimate；
6. Branch B 在同一总预算内的 V estimate 与 Q recovery；
7. 统一指标、误差分项和策略结果。

两个分支共享环境、策略、随机种子和原始数据。各自内部允许不同的数据使用方式，但不得额外获得隐藏轨迹。

### 7.2 四层验证

#### 层 1：公式级核验

- Direct-Q 的核写回逐项匹配目标 state-action TD 算子；
- V-first 的 population recovery 满足 \(\gamma\)-误差传播界；
- 有限样本 recovery 的误差分项与直接计算一致；
- exact matching、有限 softmax kernel 和直接参考实现分别标注。

#### 层 2：fixed-policy estimation

报告：

- \(\|\widehat Q-Q^\pi\|_\infty\)；
- 平均和最差 state-action error；
- greedy-action 识别率；
- action-gap-normalized error；
- kernel 对角质量与 leakage；
- 状态和状态—动作有效样本计数。

#### 层 3：压力扫描

系统改变：

- 轨迹长度；
- 动作数；
- 最小动作概率；
- kernel sharpness；
- 状态链 mixing；
- action gap。

该层专门检验 Direct-Q 是否按 state-action occupancy 退化，以及 V-first 是否真正降低估计难度。

#### 层 4：blockwise control

每个 block 内固定策略、估计 \(Q^{\pi_k}\)、更新策略，再进入下一块。报告真实回报、近似改进界、错误动作切换次数、策略块间 occupancy shift 和累计误差。

### 7.3 公平性与消融

- 两条路线获得相同总轨迹长度；
- V-first 的样本切分发生在其总预算内部；
- oracle-V 消融隔离 Q recovery 难度；
- exact matching 与有限 softmax kernel 对照隔离 kernel 偏差；
- 已知模型或生成式采样器消融区分方法误差与在线动作覆盖不足；
- 不切分 V-first 作为经验效率对照，不替代主理论版本。

## 8. 判定标准

本研究不以单个平均回报决定胜负，而是比较：

- 对 \(\mu^\pi_{\min}\) 与 \(\pi_{\min}\) 的依赖；
- kernel diagonality 所需的 sharpness；
- 达到可靠策略改进精度所需的轨迹长度；
- 对动作数、mixing 和 action gap 的敏感度；
- 证明假设的强弱；
- 模块、内存与提示结构复杂度。

最终结论允许三种形式：

1. Direct-Q 在目标设定中占优；
2. V-first 在明确覆盖区间内占优；
3. 两者不存在支配关系，各有适用区间。

## 9. 失败处理与停止条件

### 9.1 Branch A

若稀有动作使 state-action population diagonal 无法在合理 sharpness 下满足充分条件，则把结果写成 coverage barrier，并给出数值反例或阈值，不继续通过无界增温掩盖。

### 9.2 Branch B

若 V estimation 的优势被 action-conditioned Q recovery 完全抵消，则形成明确负结论。需要说明抵消来自样本计数、模型估计误差还是样本切分，而不能笼统归因于“Q 更难”。

### 9.3 共同情形

若两条路线最终都受同一 state-action coverage 控制，则继续比较误差常数、kernel 条件和构造复杂度。若某个理论界过松，实验必须同时报告界值与实际误差，不用经验成功反向宣称定理已证明。

## 10. 交付物

1. Branch A 的理论笔记：pair-chain lift、population kernel、对角条件、finite-trajectory 项和证明缺口。
2. Branch B 的理论笔记：V evaluation、V-to-Q recovery、误差分解、样本依赖和证明缺口。
3. 公式级验证脚本与测试。
4. 同预算 fixed-policy、压力扫描和 blockwise-control 实验。
5. 机器可读结果、图表和复现说明。
6. 路线比较报告，给出适用区间、失败模式和推荐方向。

## 11. 非目标与主张边界

- 第一阶段不证明 fully online、每步变策略的 Weighted Softmax SARSA 全局收敛；
- 不把 fixed-policy 定理直接套到变化策略；
- 不声称通用预训练 Transformer 会自动学习路由、持久记忆或样本切分；
- 不把 oracle mask、external action sampling 或 exact argmax 描述成标准 attention 内部能力；
- 不在结果出现前重写论文主叙事或预设胜者；
- 外部调研只采用原始论文、正式文档和可核验预印本，并区分已证明结论、证明草图和经验观察。

## 12. 执行顺序

1. 核对原始论文与本地现有定理，建立假设对照表；
2. 分别写出 Branch A 与 Branch B 的 population operator 和首版误差递推；
3. 用最小确定性实例做公式级验证；
4. 建立统一 fixed-policy 实验框架；
5. 运行 coverage、动作数、轨迹长度、sharpness 与 mixing 扫描；
6. 在 fixed-policy 结果通过后实现 blockwise control；
7. 交叉审查两条路线的假设、误差项和实验公平性；
8. 形成比较报告，再决定是否调整论文主线。
