---
title: "标准 Softmax 注意力支持上下文内策略改进"
author: "匿名作者"
lang: zh-CN
CJKmainfont: "Microsoft YaHei"
mainfont: "Times New Roman"
mathfont: "Cambria Math"
---

**关键词：** 上下文内强化学习；softmax 注意力；策略改进；SARSA；Expected SARSA；近似贪心化

# 摘要

标准的归一化 softmax 注意力能否在上下文内支持策略改进？近期的构造性研究已经证明：线性注意力可以实现策略改进，标准 softmax 可以实现固定策略的时序差分（TD）评估；然而，这些工作尚未将归一化 softmax 与动作价值控制联系起来。本文利用状态-动作价值 token 和带符号的 softmax 核写回机制，对这一问题给出肯定答案。

第一种构造从原始 transition token 与持久化 Q-memory token 出发。在一个精确定义的双 block residual attention--FFN 网络中，两个 retrieval head 读取当前 pair 与采样得到的下一 pair 的 Q 值，固定 ReLU 映射形成带符号 sampled-SARSA residual，状态--动作 head 再将其写回 Q-memory。精确恒等式使用外部提供、输入相关的等值路由 mask，以及外部 visited-query/null-token gate，并恢复同步的逐对均值表格 SARSA 更新。移除 equality mask 后得到有限 logit、gate-assisted 的近似，我们分别控制 retrieval leakage 与 write-back leakage。将单例构造与由当前动作价值导出的策略结合，便形成通常的顺序同策略控制闭环。第二种构造进一步加入动作注意力阶段。它在有限温度下的目标，恰好是当前 Boltzmann 策略的条件 Expected-SARSA 目标，同时也近似贪心动作选择。对于动作尖锐度 $\beta$，有

$$
0 \le \max_a Q(s,a)-\sum_a \operatorname{softmax}(\beta Q(s,\cdot))_a Q(s,a) \le \log(|A|)/\beta.
$$

将该界与 Bellman 最优算子的压缩性结合，可为理想表格情形下的近似贪心更新给出显式的邻域保证。Q-retrieval leakage、residual write-back leakage 和有限上下文误差作为彼此分离的扰动项进入分析。这些保证强化了采样 SARSA 的策略改进路径，但并非后者成立所必需。

一个可执行的 literal matrix witness 对完整 prompt、所有投影与 mask、residual 状态和 write-back 进行验证，并同时对照紧凑实现与表格参考。与此分开，受控表格实验用紧凑算子评估策略行为；大规模 sampled-control sweep 直接索引两个 Q 值，并没有执行完整 prompt 网络。在 30 个采样控制任务上，该紧凑闭环使精确平均策略回报提高 0.429，并在 30/30 个任务上取得正的最终 $J_\mu$ 增益，同时紧密跟踪共享轨迹的精确匹配比较器。双阶段算子在 20/20 个期望更新任务上取得正的最终 $J_\mu$ 增益，并随动作注意力变得更尖锐而趋近精确贪心控制。结果建立的是 oracle-routed、固定权重 softmax 注意力对表格动作价值控制的实现；我们不声称模型学会路由、具有逐状态支配性，或每次随机更新都单调改进策略。

# 1. 引言

标准的归一化 softmax 注意力能否在上下文内支持策略改进？固定权重的 Transformer 可以在前向传播中执行学习算法 [1--3]，近期研究也给出了强化学习中的构造性例子。Liang 和 Lai [4] 证明，线性注意力可以实现参数化半梯度 SARSA 与 actor-critic 更新。Xie 等人 [5] 证明，标准 softmax 注意力可以针对固定策略实现加权 TD 策略评估。这两条研究路线合在一起，仍留下一个问题：后一种归一化注意力机制能否不只做策略评估，还能支持动作价值控制？

策略改进并不要求每个 TD 目标中都包含最大值。SARSA 使用当前行为策略实际选出的动作 $a'_k$，构造同策略残差

$$
\delta_k^S=r_k+\gamma Q(s'_k,a'_k)-Q(s_k,a_k).
$$

先更新 $Q$，再由更新后的价值导出下一轮行为策略，就得到一个同策略控制闭环 [6]。贪心 Bellman 备份是另一条更强的路径：它通过 $\max_b Q(s',b)$ 将动作选择置于目标内部。这两条路径回答的是互补问题，不应混为一谈。

我们用一个持久化 Q-memory token 表示每个状态--动作对 $x=(s,a)$，其中存储 $Q(x)$；原始 transition token 则包含当前与下一 pair 标识以及 reward。两个固定的结构化路由 head 检索 sampled SARSA 所需的两个 Q 值，逐位置映射形成带符号 residual，最后的标准 softmax head 执行标量写回。这种 token 化方式无需通过样本相关的 $\delta$ 与 $\phi$ 的乘积来更新全局参数向量。统一的写回算子为

$$
Q^+(x)=Q(x)+\alpha\sum_k K_\eta(x_k,x)\delta_k,
$$

其中，$K_\eta$ 是归一化的状态-动作匹配核，带符号的 TD 残差由 value 向量携带。因此，注意力权重为正并不会消除负的 TD 修正。

第一种构造并不把 sampled SARSA residual 当作输入，而是证明完整 raw-token computation。精确模型是一个采用标准指数 softmax、双向结构化注意力、无 LayerNorm 与 dropout 的双 block residual attention--FFN 网络；它不是 decoder-only 因果 Transformer。外部提供、输入相关的 equality-routing mask 检索 $Q(s,a)$ 与 $Q(s',a')$，固定的两单元 ReLU 映射形成 residual，外部 visited-query/null-token gate 完成同步逐对均值写回。单例版本生成通常的顺序 SARSA iterates。移除 equality mask 后，有限 logit 近似 retrieval 与 write-back，但仍保留外部 gate。

第二种构造增加一个动作注意力阶段。对当前 Boltzmann 策略

$$
p_\beta(b\mid s)=\frac{\exp(\beta Q(s,b))}{\sum_c\exp(\beta Q(s,c))},
$$

该阶段返回

$$
M_\beta Q(s)=\sum_b p_\beta(b\mid s)Q(s,b).
$$

在有限 $\beta$ 下，这恰好是当前 Boltzmann 策略的条件 Expected-SARSA 目标。它同时近似贪心动作选择，并满足一致界

$$
0\le \max_b Q(s,b)-M_\beta Q(s)\le \log(|A|)/\beta.
$$

Expected-SARSA 与近似贪心这两种解释都成立。我们仅用第二种解释推导更强的最优控制结论：在理想的表格、同步设定中，所得更新是松弛 Bellman 最优更新的有界扰动。证明使用 $T^{\star}$ 的压缩性，而不假设尚未证明的平滑算子压缩性。核失配与有限上下文误差作为彼此独立的扰动进入分析。

受控的表格实验通过策略行为检验两条控制路径。为扩大 sweep，采样 SARSA 实验用直接表索引给出两个 Q 值，softmax 只实现 write-back kernel；因此它运行的是紧凑算子，而非 literal prompt 网络。我们将其与共享轨迹的精确匹配写回比较器对照，从而单独考察有限 softmax matching，而不把比较器误称为独立的同策略学习器。另一个 literal matrix witness 单独检验完整 prompt 构造。对于双阶段紧凑算子，我们在改变动作尖锐度和核尖锐度的同时，将其与精确贪心控制比较。

本文的贡献如下：

1. 给出一个端到端、固定权重的标准 softmax 构造，从 Q-memory token 中检索当前动作价值、形成 sampled-SARSA residual 并写回；同时区分 externally routed 的精确恒等式与 equality-mask-free but gate-assisted 的有限 logit 近似。
2. 证明双阶段构造的有限温度目标恰好是 Boltzmann Expected SARSA，同时也是对贪心动作选择的一致有界近似。
3. 针对更强的近似贪心路径，推导动作价值保证与策略性能保证，并显式分离动作尖锐度、核误差和有限上下文误差。
4. 通过配对控制实验和策略回报评估两条路径。

本文结果是构造性的，并且有意限定适用范围。精确 content selection 使用外部提供、输入相关的结构化等值 mask 与外部 visited-query/null-token gate；有限 equality-mask-free logit 仍为 gate-assisted 近似。环境交互、精确 $\varepsilon$-greedy argmax 与随机动作采样均在网络之外。我们不声称每个随机 TD 步骤都会单调改进策略，不声称通用预训练 Transformer 会自动学会这些算子或路由，也不声称主 control sweep 执行完整 token 网络。主要保证针对具有显式覆盖条件的有限表格问题。

# 2. 相关工作

**作为上下文内算法的 Transformer。** 线性注意力和自注意力构造已被用于联系梯度下降、岭回归以及其他可在一次前向传播中执行的学习算法 [1--3]。这种构造性视角促使我们追问：固定注意力机制能够表示哪些强化学习控制操作？

**构造性上下文内强化学习。** Liang 和 Lai [4] 给出了与策略改进最接近的结果。他们的线性注意力构造通过半梯度 SARSA 项更新全局参数向量，其 actor-critic 构造则提供第二条同策略路径。Xie 等人 [5] 使用标准归一化 softmax 注意力，为固定 Markov 奖励过程实现非参数 TD 策略评估。我们的状态-动作 token 构造沿用了 [5] 的带符号 softmax 写回机制，但将其接入动作价值控制。它与 [4] 在注意力类型和表示方式上都不同：归一化 softmax 直接更新 $Q$ token，而线性注意力构造出更新全局权重向量所需的参数化半梯度乘积。

**SARSA、Expected SARSA 与贪心控制。** 当动作价值更新与由当前估计导出的行为策略相结合时，经典 SARSA 是一种同策略控制方法 [6]。Expected SARSA 将采样得到的下一动作价值替换为其在目标策略下的条件期望。对于由当前 $Q$ 值诱导的 Boltzmann 策略，这个条件期望恰好就是我们的双阶段算子所使用的 softmax 加权动作价值。策略改进并不要求使用贪心 Bellman 控制，但后者提供一条直接通向 $Q^{\star}$ 的路径以及策略性能界。因此，我们将采样 SARSA 与近似贪心化视为互补的 softmax 控制构造。

**归一化与动作选择。** 归一化 softmax 注意力头返回其 value 向量的凸组合 [7]。这不会消除 value 中已经携带的带符号坐标，因此 softmax 核可以聚合正、负 TD 残差。当注意力被用于在动作之间选择时，归一化确实会带来可量化的尖锐度权衡。在我们的近似贪心路径中，动作被视为候选项，因此 Velickovic 等人 [8] 关于候选项数量与尖锐度的分析与本文相关。

**范围。** 本文贡献是在算子层面证明标准 softmax 注意力能够实现策略改进。我们既不复现线性注意力的训练动力学，也不声称预训练模型必然会发现该构造。采样 SARSA 路径建立直接的同策略机制；双阶段路径则通过在前向算子内部加入近似贪心化来强化这一结果。

# 3. 端到端构造与保证

### 3.1 精确模型类别与完整 prompt

令 $\mathcal X=\mathcal S\times\mathcal A$，$m=|\mathcal X|$，并固定 $\mathcal X$ 的枚举。状态--动作对 $x$ 的独热标识为 $e_x\in\mathbb R^m$。一次更新接收冻结的表 $Q_l$、固定标量 $\gamma\in[0,1)$ 与 $\alpha>0$，以及 $N$ 条采样转移

$$
(S_k,A_k,R_k,S'_k,A'_k),\qquad k=1,\ldots,N,
$$

其中 $A'_k$ 是当前行为策略实际选择的下一动作。记 $X_k=(S_k,A_k)$、$X'_k=(S'_k,A'_k)$，以及 $n_x=\sum_{k=1}^N\mathbf1\{X_k=x\}$。

我们针对一个精确定义的 residual attention 网络证明结论。它包含两个 attention--FFN block，采用标准指数 softmax，不含 LayerNorm 和 dropout。token 按列排列。网络使用双向结构化注意力，不是 decoder-only 因果 Transformer。投影矩阵在固定 $m,N,\gamma,\alpha$ 后即固定。结构化等值路由掩码由外部提供并且是输入相关的；另一个外部支持门控依据离散 pair 标识提供已访问查询（visited query）/ null token 规则。因此，该定理给出的是 oracle-routed 的精确参考构造，并不证明或声称通用预训练 Transformer 会学习或发现路由。

取 token 宽度 $d=2m+8$，使用直和坐标约定

$$
\mathbb R^d
=\underbrace{\mathbb R^3}_{\text{type}}
\oplus\underbrace{\mathbb R^m}_{\text{current ID}}
\oplus\underbrace{\mathbb R^m}_{\text{next ID}}
\oplus\underbrace{\mathbb R^5}_{(r,q,u,v,\delta)}.
$$

令 $\tau_Q,\tau_T,\tau_Z\in\mathbb R^3$ 为三个 type 基向量。五个标量坐标依次存 reward $r$、持久化 memory 值 $q$、检索到的当前值 $u$、检索到的下一值 $v$ 和 TD residual $\delta$。定义

$$
\begin{aligned}
M_x&=(\tau_Q,e_x,0_m,0,Q_l(x),0,0,0)^\top,\\
T_k&=(\tau_T,e_{X_k},e_{X'_k},R_k,0,0,0,0)^\top,\\
Z&=(\tau_Z,0_m,0_m,0,0,0,0,0)^\top,\\
H^{(0)}&=[M_{x_1},\ldots,M_{x_m},T_1,\ldots,T_N,Z]
\in\mathbb R^{d\times L},\qquad L=m+N+1 .
\end{aligned}
\tag{3.1}\label{eq:prompt}
$$

null token $Z$ 并非零向量，因为它保留 type 坐标；但下文所有作为 attention value 读取的坐标均为零。特别地，原始 transition token 既不包含 $Q_l(X_k)$、$Q_l(X'_k)$，也不包含预计算 residual。

令 $P_c,P_n\in\mathbb R^{m\times d}$ 分别选取 current-ID 和 next-ID block。令 $\mathbf e_r,\mathbf e_q,\mathbf e_u,\mathbf e_v,\mathbf e_\delta\in\mathbb R^d$ 为五个标量坐标的基列。对满足 $W_Q,W_K\in\mathbb R^{d_h\times d}$、$W_V\in\mathbb R^{d_v\times d}$、$W_O\in\mathbb R^{d\times d_v}$ 的 head，采用 query-by-source 权重

$$
A_{ij}
=\frac{\exp\!\left((W_Qh_i)^\top(W_Kh_j)/\sqrt{d_h}+\mathcal M_{ij}\right)}
{\sum_{s=1}^{L}\exp\!\left((W_Qh_i)^\top(W_Kh_s)/\sqrt{d_h}+\mathcal M_{is}\right)} .
$$

矩阵形式的 head 输出为 $W_O(W_VH)A^\top\in\mathbb R^{d\times L}$。以下每个 mask row 都有非空支持；非目标 query 被路由到 $Z$，因此不存在全为 $-\infty$ 的 softmax row。

**定理 3.1（结构化等值路由下的精确 sampled-SARSA）。** 对固定的 $m,N,\gamma,\alpha$，以及下文外部给定的等值路由 mask 和 visited/null 支持规则，两 block residual attention--FFN 网络把 (3.1) 中的原始 prompt 映射成满足下式的 Q-memory：对每个 $x\in\mathcal X$，

$$
Q_{l+1}(x)=
\begin{cases}
Q_l(x)+\dfrac{\alpha}{n_x}\displaystyle\sum_{k:X_k=x}
[R_k+\gamma Q_l(X'_k)-Q_l(X_k)],&n_x>0,\\[3mm]
Q_l(x),&n_x=0.
\end{cases}
$$

### 3.2 定理 3.1 的证明：检索与 residual 构造

**步骤 1：当前 pair 检索。** 第一个 head 取

$$
\begin{gathered}
W_Q^{\rm cur}=P_c,\qquad W_K^{\rm cur}=P_c,\qquad
W_V^{\rm cur}=\mathbf e_q^\top,\qquad W_O^{\rm cur}=\mathbf e_u,\\
\mathcal M^{\rm cur}_{ij}=0
\quad\Longleftrightarrow\quad
\bigl(i=T_k,\ j=M_{X_k}\bigr)
\ \text{or}\ 
\bigl(i\notin\{T_1,\ldots,T_N\},\ j=Z\bigr).
\end{gathered}
\tag{3.2}\label{eq:block1-current}
$$

其余 mask 元素均为 $-\infty$。对 transition query $T_k$，唯一允许的 source 是 $M_{X_k}$，所以无论数值 logit 为何，该 softmax row 都只在这一位置取权重 1，并且

$$
W_O^{\rm cur}\sum_jA^{\rm cur}_{T_kj}W_V^{\rm cur}h_j
=\mathbf e_u\,\mathbf e_q^\top M_{X_k}
=Q_l(X_k)\mathbf e_u .
$$

Q-memory 或 null query 只允许读取 $Z$；由于 $\mathbf e_q^\top Z=0$，其 head 输出为零。

**步骤 2：下一 pair 检索。** 第二个 head 与第一个并行，只在 query 侧读取 next-ID block：

$$
\begin{gathered}
W_Q^{\rm next}=P_n,\qquad W_K^{\rm next}=P_c,\qquad
W_V^{\rm next}=\mathbf e_q^\top,\qquad W_O^{\rm next}=\mathbf e_v,\\
\mathcal M^{\rm next}_{ij}=0
\quad\Longleftrightarrow\quad
\bigl(i=T_k,\ j=M_{X'_k}\bigr)
\ \text{or}\ 
\bigl(i\notin\{T_1,\ldots,T_N\},\ j=Z\bigr),\\
W_O^{\rm next}\sum_jA^{\rm next}_{T_kj}W_V^{\rm next}h_j
=Q_l(X'_k)\mathbf e_v .
\end{gathered}
\tag{3.3}\label{eq:block1-next}
$$

未列出的元素仍为 $-\infty$，每个非 transition query 的 head 输出仍为零。两个 head 都读取同一 $H^{(0)}$，符合并行 multi-head attention；它们的输出投影写入互不相交的坐标。因此 attention residual 后的状态为

$$
\begin{aligned}
H^{(a)}&=H^{(0)}+O^{\rm cur}+O^{\rm next},\\
M_x^{(a)}&=M_x,\qquad Z^{(a)}=Z,\\
T_k^{(a)}&=(\tau_T,e_{X_k},e_{X'_k},R_k,0,
Q_l(X_k),Q_l(X'_k),0)^\top .
\end{aligned}
$$

**步骤 3：精确的逐位置 residual 映射。** 令

$$
\begin{gathered}
g^\top=\mathbf e_r^\top+\gamma\mathbf e_v^\top-\mathbf e_u^\top,\qquad
W_1=\begin{bmatrix}g^\top\\-g^\top\end{bmatrix}\in\mathbb R^{2\times d},\\
W_2=\mathbf e_\delta\begin{bmatrix}1&-1\end{bmatrix}
\in\mathbb R^{d\times2},\qquad
W_2\operatorname{ReLU}(W_1h)=\mathbf e_\delta g^\top h .
\end{gathered}
\tag{3.4}\label{eq:residual-ffn}
$$

最后一个恒等式来自 $\operatorname{ReLU}(z)-\operatorname{ReLU}(-z)=z$。所以第一 block 的 FFN residual 得到

$$
H^{(1)}=H^{(a)}+W_2\operatorname{ReLU}(W_1H^{(a)}),
\qquad
\delta_k^S=R_k+\gamma Q_l(X'_k)-Q_l(X_k).
$$

对 Q-memory 与 null token，$r,u,v$ 坐标均为零，所以 FFN 输出为零。Block I 保留全部标识、reward 和 Q-memory 值，只把两个检索值与带符号的 sampled-SARSA residual 写入各 transition token。

### 3.3 定理 3.1 的证明：写回与组装

**步骤 4：完整 write-back head。** 第二个 block 只有一个 active head：

$$
\begin{gathered}
W_Q^{\rm wr}=P_c,\qquad W_K^{\rm wr}=P_c,\qquad
W_V^{\rm wr}=\mathbf e_\delta^\top,\qquad
W_O^{\rm wr}=\alpha\mathbf e_q,\\
\mathcal M^{\rm wr}_{ij}=0
\quad\Longleftrightarrow\quad
\begin{cases}
i=M_x,\ j=T_k,\ X_k=x,&n_x>0,\\
i=M_x,\ j=Z,&n_x=0,\\
i\in\{T_1,\ldots,T_N,Z\},\ j=Z.&
\end{cases}
\end{gathered}
\tag{3.5}\label{eq:write-back}
$$

所有其余元素都是 $-\infty$。对已访问 memory query $M_x$ 以及任意允许的 source $T_k$，

$$
\frac{(P_cM_x)^\top(P_cT_k)}{\sqrt m}
=\frac{e_x^\top e_{X_k}}{\sqrt m}
=\frac1{\sqrt m}.
$$

因此，$n_x$ 个允许 logit 相等，每个权重恰为 $1/n_x$。value 投影选取带符号标量 $\delta_k^S$，故 $M_x$ 的 head 输出是

$$
\frac{\alpha}{n_x}\sum_{k:X_k=x}\delta_k^S\,\mathbf e_q .
$$

正的 softmax 权重不会消除负 TD correction，因为符号位于 value 坐标而非注意力权重中。若 $n_x=0$，唯一 source 是 $Z$，且 $\mathbf e_\delta^\top Z=0$，所以输出为零。transition 和 null query 同样只读取 $Z$，保持不变。

**步骤 5：组装两个 block。** 将全部未使用的投影 block 置零，并把第二 block 的 FFN 置零。相对于 (3.1) 的初始状态，完整 token-state 变化为

| token | Block I attention 后 | Block I FFN 后 | Block II 后 |
|---|---|---|---|
| $M_x$ | 不变 | 不变 | 置 $q=Q_{l+1}(x)$ |
| $T_k$ | 置 $u=Q_l(X_k)$、$v=Q_l(X'_k)$ | 置 $\delta=\delta_k^S$ | 不变 |
| $Z$ | 不变 | 不变 | 不变 |

表中未列出的每个坐标都与 $H^{(0)}$ 完全相同。

读取 $H^{(2)}$ 的 Q-memory 坐标可得

$$
Q_{l+1}(x)=
\begin{cases}
Q_l(x)+\dfrac{\alpha}{n_x}\displaystyle\sum_{k:X_k=x}
\left[R_k+\gamma Q_l(X'_k)-Q_l(X_k)\right],&n_x>0,\\[3mm]
Q_l(x),&n_x=0.
\end{cases}
\tag{3.6}\label{eq:final-update}
$$

这正是定理陈述。每个展示出的矩阵，对声明的 $m,N,\gamma,\alpha$ 而言都是固定坐标 selector 或固定标量倍数；全部数据依赖只存在于 prompt 与外部给定 mask 中。因此，两次 Q 检索、residual 构造与 Q 写回都包含在这两个 block 的计算内。$\square$

**精确恒等式的范围。** (3.6) 是冻结 $Q_l$ 下的逐对均值（per-pair mean）更新，不是全局 batch convention

$$
Q_l(x)+\frac{\alpha}{N}\sum_{k:X_k=x}\delta_k^S,
$$

也不会在一个冻结 batch 内按顺序处理同一 pair 的重复出现。单例调用满足 $n_x=1$，才恢复通常的顺序表格 SARSA 步。构造需要输入相关的等值路由与外部 visited/null 支持检测；二者都不是 softmax 学出的因果 mask，也不表明通用预训练 Transformer 会发现该布局。精确网络固定于给定的 $m,N,\gamma,\alpha$；可变 Robbins--Monro 步长要求对应的输出投影模块族，或额外的乘法机制。每个控制 round 都重建 prompt，等价地清空并重新初始化 $u,v,\delta$ scratch 坐标。第 3.8 节在有限 logit 下移除 equality mask，但仍保留 visited-query gate。

### 3.4 从更新算子到控制

**推论 3.2（顺序 SARSA 等价）。** 若每个控制 round 只处理一条采样转移，并且只有其已访问 pair 发出非 null write-back query，则以步长 $\alpha_t$ 实例化的构造产生

$$
Q_{t+1}(S_t,A_t)
=Q_t(S_t,A_t)
+\alpha_t[R_{t+1}+\gamma Q_t(S_{t+1},A_{t+1})-Q_t(S_t,A_t)],
$$

其余状态--动作值均不变。因此，固定输出 scale 依次为给定 $\alpha_t$ 的一系列 block，精确生成表格型顺序 sampled-SARSA iterates；单个固定 block 只对应常数步长。

对奖励有界的有限表格 MDP，在每个状态--动作对被无限访问、采用逐 pair Robbins--Monro 步长以及无限探索且极限贪心（GLIE）的策略序列时，顺序构造继承经典的几乎必然收敛结论 [6]。第 5.2 节的冻结 mini-batch 实验采用不同的更新 convention 与固定探索率，因此不对该实验 protocol 声称这一收敛定理。

单个 SARSA residual 并不是策略改进定理。通常的精确策略改进桥梁在逻辑上是分开的。若 $\pi$ 是 $\varepsilon$-soft 策略、$Q^\pi$ 已被精确评估，且 $\pi'$ 关于 $Q^\pi$ 为 $\varepsilon$-贪心策略，则写成 $\pi=\varepsilon/|\mathcal A|+(1-\varepsilon)\widetilde\pi$ 有

$$
\sum_a\pi'(a|s)Q^\pi(s,a)
=\frac{\varepsilon}{|\mathcal A|}\sum_aQ^\pi(s,a)
+(1-\varepsilon)\max_aQ^\pi(s,a)
\ge V^\pi(s).
$$

因此策略改进定理给出 $V^{\pi'}\ge V^\pi$。若估计 $\widehat Q$ 满足 $\|\widehat Q-Q^\pi\|_\infty\le\xi$，最坏情形结论弱化为 $V^{\pi'}\ge V^\pi-2\xi/(1-\gamma)$；单调性需要足够的 advantage margin。定理 3.1 建立表示等价，推论 3.2 给出有条件的在线收敛路径，而冻结 batch 实验提供经验性的 before--after 策略证据；这是三类不同结论。

### 3.5 内部 Boltzmann 动作注意力

对 $\beta>0$，定义当前 Boltzmann 策略及其动作价值期望：

$$
p_\beta(b\mid s;Q)=\frac{\exp(\beta Q(s,b))}{\sum_c\exp(\beta Q(s,c))},
$$

$$
M_\beta Q(s)=\sum_b p_\beta(b\mid s;Q)Q(s,b).
$$

相应的条件 Bellman 目标为

$$
(T_\beta Q)(s,a)=\mathbb{E}\left[R+\gamma M_\beta Q(S')\mid s,a\right].
$$

**命题 3.3（Expected-SARSA 恒等式）。** 若在给定 $S'$ 时，$A'$ 从 $p_\beta(\cdot\mid S';Q)$ 中采样，则

$$
\mathbb{E}\left[R+\gamma Q(S',A')\mid s,a\right]=(T_\beta Q)(s,a).
$$

**证明。** 首先对 $S'$ 条件化。关于 $A'$ 的内层期望为 $\sum_b p_\beta(b\mid S';Q)Q(S',b)=M_\beta Q(S')$。再取剩余的条件期望即可证明该恒等式。

因此，在有限 $\beta$ 下，$T_\beta$ 恰好是与当前 $Q$ 绑定的策略之条件 Expected-SARSA 算子。由于 $\beta$ 增大时 $M_\beta$ 趋近最大值，同一算子也可以作为近似贪心化来分析。这两种解释相互补充；有限 $\beta$ 并不是精确 Q-learning。

### 3.6 Softmax 对贪心动作选择的近似

**引理 3.4（softmax 动作聚合的熵界）。** 对任意 $q\in\mathbb{R}^d$ 与 $\beta>0$，有

$$
0\le\max_a q_a-\sum_a\operatorname{softmax}(\beta q)_a q_a
\le\frac{\log d}{\beta}.
$$

**证明。** 令 $p=\operatorname{softmax}(\beta q)$，并定义

$$
\operatorname{LSE}_\beta(q)=\beta^{-1}\log\sum_a\exp(\beta q_a).
$$

Gibbs 熵恒等式给出 $\operatorname{LSE}_\beta(q)=\sum_ap_aq_a+H(p)/\beta$。又因为 $\max_aq_a\le\operatorname{LSE}_\beta(q)$、$\sum_ap_aq_a\le\max_aq_a$ 且 $0\le H(p)\le\log d$，结论成立。$\square$

**推论 3.5（贪心目标偏差）。** 对任意 $Q$，有

$$
\lVert T_\beta Q-T^{\star}Q\rVert_\infty
\le\gamma\frac{\log(|A|)}{\beta}.
$$

### 3.7 近似贪心路径的更强保证

考虑采用精确状态-动作匹配、具有完全覆盖的同步更新

$$
Q_{\ell+1}=(1-\alpha)Q_\ell+\alpha T_\beta Q_\ell,
$$

其中 $0<\alpha\le1$，并定义 $\rho=1-\alpha(1-\gamma)$。

**定理 3.6（Bellman 最优性的有界扰动）。** 迭代满足

$$
\lVert Q_{\ell+1}-Q^{\star}\rVert_\infty
\le \rho\lVert Q_\ell-Q^{\star}\rVert_\infty
+\alpha\gamma\frac{\log(|A|)}{\beta}.
$$

从而

$$
\limsup_{\ell\to\infty}\lVert Q_\ell-Q^{\star}\rVert_\infty
\le\frac{\gamma\log(|A|)}{\beta(1-\gamma)}.
$$

**证明。** 加上再减去 $T^{\star}Q_\ell$。松弛恒等映射贡献 $1-\alpha$，$T^{\star}$ 的压缩性贡献 $\alpha\gamma$，推论 3.5 则贡献加性目标偏差。迭代所得的标量递推关系，即得上述上极限界。该证明不假设 $T_\beta$ 本身是压缩映射。$\square$

若 $\beta_\ell$ 随深度趋于无穷，则加性项趋于零，同一稳定标量递推关系给出对 $Q^{\star}$ 的收敛。固定 $\beta$ 时，所得结果是一个受控邻域，而非精确收敛。

### 3.8 无等值掩码的有限 logit 偏差

精确定理把 pair equality 委托给 mask。现在移除两个 retrieval mask 与 write-back equality mask，但保留外部 visited-query gate。因此，所得构造是 equality-mask-free 但仍 gate-assisted。

对 retrieval sharpness $\zeta>0$，缩放独热投影，使正确 Q-memory logit 为 $\zeta$、每个错误 logit 为零，并在全部 $m$ 个 Q-memory token 上取 softmax。正确质量、总 off-target 质量以及 induced residual error 满足

$$
\begin{gathered}
p_\zeta=\frac{e^\zeta}{e^\zeta+m-1},\qquad
\lambda_R=1-p_\zeta=\frac{m-1}{e^\zeta+m-1},\\
|\widetilde Q_l(x)-Q_l(x)|
\le\lambda_R\operatorname{span}(Q_l),\\
|\widetilde\delta_k^S-\delta_k^S|
\le(1+\gamma)\lambda_R\operatorname{span}(Q_l)
=:E_R .
\end{gathered}
\tag{3.7}\label{eq:finite-retrieval}
$$

其中 $\operatorname{span}(Q_l)=\max_xQ_l(x)-\min_xQ_l(x)$。第一个 value bound 来自：质量 $\lambda_R$ 从正确表项移到与其相差不超过 table span 的值；residual 包含一次 current retrieval 与一次折扣后的 next retrieval，所以产生因子 $1+\gamma$。

对 write-back sharpness $\eta>0$，使用 logit $\eta\langle e_{X_k},e_x\rangle$。对已访问 query $x$，有限 attention kernel 与精确 group-mean kernel 为

$$
\begin{gathered}
K_\eta(k|x)
=\frac{\exp(\eta\mathbf1\{X_k=x\})}
{n_xe^\eta+N-n_x},\qquad
U(k|x)=\frac{\mathbf1\{X_k=x\}}{n_x},\\
\varepsilon_W(x)
:=\sum_{k=1}^N|K_\eta(k|x)-U(k|x)|
=\frac{2(N-n_x)}{n_xe^\eta+N-n_x}.
\end{gathered}
\tag{3.8}\label{eq:finite-write-back}
$$

最后一个恒等式中，matching token 的总绝对差与 nonmatching token 的总质量都等于 $(N-n_x)/(n_xe^\eta+N-n_x)$。

**命题 3.7（有限 logit 端到端更新误差）。** 假设每条转移都满足 $|\delta_k^S|\le B$。对每个已访问 query，

$$
|Q_{l+1}^{\mathrm{soft}}(x)-Q_{l+1}^{\mathrm{exact}}(x)|
\le\alpha[E_R+B\varepsilon_W(x)].
\tag{3.9}\label{eq:finite-end-to-end-error}
$$

**证明。** 有限更新为 $\alpha\sum_kK_\eta(k|x)\widetilde\delta_k^S$，精确更新为 $\alpha\sum_kU(k|x)\delta_k^S$。加上并减去 $\alpha\sum_kK_\eta(k|x)\delta_k^S$。由于 $K_\eta(\cdot|x)$ 是概率向量，(3.7) 将 residual replacement term 控制在 $\alpha E_R$；由 $|\delta_k^S|\le B$，(3.8) 将 kernel replacement term 控制在 $\alpha B\varepsilon_W(x)$。两项相加得到 (3.9)。$\square$

对冻结的策略和冻结的 $Q$，定义有限上下文残差偏差

$$
\varepsilon_D=\max_{x:n_x>0}
\left|\frac{1}{n_x}\sum_{k:X_k=x}\delta_k^S-\left[(T^\pi Q_l)(x)-Q_l(x)\right]\right|.
$$

实现的更新与相应已访问 population update 的差异至多为 $\alpha(E_R+B\varepsilon_W+\varepsilon_D)$。在 approximate-greedy full-coverage recurrence 中，这些项以加性方式进入：

$$
\lVert Q_{\ell+1}-Q^{\star}\rVert_\infty
\le\rho\lVert Q_\ell-Q^{\star}\rVert_\infty
+\alpha\left[\frac{\gamma\log|\mathcal A|}{\beta}+E_R+B\varepsilon_W+\varepsilon_D\right].
$$

这是一个单步算子偏差结论。它不会把任意有限轨迹变成全局 Bellman recurrence，也不会在没有独立采样、mixing 或 martingale 假设时自动给出 concentration rate。retrieval sharpness、write-back sharpness 与 action sharpness 控制三种不同操作。

### 3.9 策略层面的结论

**推论 3.8。** 若 $\lVert Q-Q^{\star}\rVert_\infty\le\varepsilon$，且 $\pi_Q$ 关于 $Q$ 为贪心策略，则

$$
\lVert V^{\star}-V^{\pi_Q}\rVert_\infty
\le\frac{2\varepsilon}{1-\gamma}.
$$

因此，近似贪心动作价值保证可推出一个有条件的策略性能保证。采样 SARSA 路径则分别从定理 3.1 获得精确算子等价、从推论 3.2 获得有条件的渐近结论，并在独立的冻结 batch protocol 下提供经验性的最终 return 证据。

# 4. 方法

### 4.1 端到端构造与紧凑实验算子

第 3 节在 literal $d\times L$ prompt 上给出完整 token computation。原始 transition token 只包含 pair 标识与 reward；两个 structured-routing retrieval head 从 Q-memory 复制所需的当前 Q 值；显式的两单元 ReLU FFN 形成带符号 TD residual；最终 state--action head 执行 write-back。所有 $W_Q,W_K,W_V,W_O,W_1,W_2$ 都是固定坐标映射，初始 transition token 中不提供 residual 或当前 Q 值。

为提高效率，大规模 control sweep 绕过两个 prompt-level retrieval head：直接 table indexing 给出同样的两个标量，同一固定线性表达式形成 residual，标准 softmax kernel 执行 write-back。因此，该 sweep 检验的是紧凑 write-back/control operator，而不是完整 prompt 网络的执行。独立的 literal matrix witness 则实例化完整 prompt、每个投影、所有 query-by-source routing mask、null-token gate、residual scratch state 与 write-back head，并把结果同时与紧凑实现和表格参考比较。实验使用独热状态--动作标识，因此两条实现都不检验 learned representation 或 function approximation。

对已访问查询 $x$，标准点积 softmax 注意力使用

$$
K_\eta(k\mid x)=
\frac{\exp\!\left(\eta\langle\psi(X_k),\psi(x)\rangle\right)}
{\sum_j\exp\!\left(\eta\langle\psi(X_j),\psi(x)\rangle\right)},
$$

并执行写回

$$
Q_{\ell+1}(x)=Q_\ell(x)+\alpha_\ell\sum_kK_\eta(k\mid x)\delta_k.
$$

注意力权重为正且经过归一化，但 $\delta_k$ 带符号，因此负的 TD 修正仍然可用。上下文中未出现的查询不更新。定理 3.1 的精确构造使用外部提供的 equality-routing mask，并通过外部 gate 将未访问查询路由到零值 null token。移除 equality mask 后，有限 $\eta$ 给出第 3.8 节与 finite-logit Q retrieval 联合分析的显式近似；该近似仍保留 visited-query gate。

### 4.2 路径 I：采样 SARSA 控制

对每个批次，在 $S'_k$ 处从当前行为策略中采样 $A'_k$，并令

$$
\delta_k^S=R_k+\gamma Q_\ell(S'_k,A'_k)-Q_\ell(X_k).
$$

第 3 节的两个 retrieval head 与固定 residual map 构造 $\delta_k^S$，state--action head 执行上述写回。在精确匹配下，这恰好是

$$
Q_{\ell+1}(x)=Q_\ell(x)+\frac{\alpha_\ell}{n_x}
\sum_{k:X_k=x}\delta_k^S,
$$

即在冻结 $Q_\ell$ 下，同步、按状态-动作对取均值的小批量 SARSA 更新。当只查询单个转移时，它就是通常的采样 SARSA 更新。它既不是对重复访问求和的更新，也不是在批次内部不断改变 $Q$ 的顺序更新。

为了闭合控制回路，下一轮行为策略由 $Q_{\ell+1}$ 导出。在采样实验中，我们采用 $\varepsilon$-贪心策略。TD 目标本身不包含最大值：下一动作就是从当前策略实际采样的动作。在定理与 literal witness 中，Q retrieval、residual formation 和 write-back 都发生在构造的 block 内；大规模 control sweep 则用直接 table indexing 取代 retrieval，只运行紧凑 residual/write-back kernel。环境交互、精确 $\varepsilon$-greedy argmax 与随机动作采样均为外部接口。定理 3.1 是 operator identity，推论 3.2 讨论不同的 singleton online protocol，而当前 frozen-batch protocol 只接受经验评估。

### 4.3 路径 II：Boltzmann Expected SARSA 与近似贪心化

更强的路径在每个下一状态的动作 token 上增加一个标准 softmax 注意力头。其得分为 $\beta Q_\ell(S'_k,b)$，value 为 $Q_\ell(S'_k,b)$，分组掩码将注意力限制在共享同一 $S'_k$ 的动作上。输出为

$$
M_\beta Q_\ell(S'_k)=
\sum_b\frac{\exp(\beta Q_\ell(S'_k,b))}
{\sum_c\exp(\beta Q_\ell(S'_k,c))}Q_\ell(S'_k,b).
$$

随后，一个线性投影构造

$$
\delta_k^\beta=R_k+\gamma M_\beta Q_\ell(S'_k)-Q_\ell(X_k),
$$

统一的状态-动作注意力头再将该残差写回 $Q$ token。提出的算子内部没有使用精确最大值。在有限 $\beta$ 下，该目标是当前 Boltzmann 策略的条件 Expected-SARSA 目标。若将其视为贪心控制的近似，则其 Bellman 目标层面的误差由 $\gamma\log(|A|)/\beta$ 控制，从而得到定理 3.6。

### 4.4 构造的适用范围

精确 sampled-SARSA 结论针对一个标准指数归一化、固定坐标投影、双向结构化注意力、无 LayerNorm 与 dropout 的双 block residual attention--FFN 网络；它不是 decoder-only 因果 Transformer。矩阵固定于给定 $m,N,\gamma,\alpha$。构造从原始 transition 与持久化 Q-memory 出发，不假设预计算 TD residual。精确 retrieval 与 write-back 依赖外部提供、输入相关的 equality-routing mask 和外部 visited-query/null-token gate；命题 3.7 移除 equality mask，但保留该 gate。我们不声称通用预训练 Transformer 会发现布局或学会路由，不声称有限 logit 实现精确 content selection，也不把 simulator action execution 与随机采样放入确定性 attention head。可变步长 schedule 需要输出 scale 不同的 block family，或额外乘法机制。

# 5. 实验

### 5.1 问题、环境与策略指标

实验围绕与核心主张直接相关的三个问题展开：

1. 紧凑的单阶段 softmax write-back 算子是否与 sampled-SARSA 参考一致，独立的 literal witness 是否又与该紧凑算子一致？
2. 将该算子与其行为策略闭合后，是否能提高策略回报？
3. 当双阶段算子的内部动作聚合趋近贪心控制时，它是否仍保持策略改进？

每个任务都是一个有限随机 MDP。转移概率的每一行和初始状态分布均从对称 Dirichlet 分布中采样，转移奖励在 $[-1,1]$ 上均匀分布。对于策略 $\pi$ 和初始分布 $\mu$，我们精确计算折扣回报

$$
J_\mu(\pi)=\mu^\top(I-\gamma P_\pi)^{-1}r_\pi,
$$

因此，策略比较不包含 rollout 评估噪声。除非另有说明，文中报告的离散程度均为独立采样任务之间的样本标准差。

### 5.2 采样 SARSA：匹配写回与闭环改进

我们使用 30 个 MDP，每个包含 9 个状态、4 个动作，且 $\gamma=0.5$。标准 softmax 学习器从一个较小的随机 $Q$ 表出发，采用由当前 $Q$ 导出的 $\varepsilon=0.1$ 贪心行为策略。在 200 次控制更新中的每一次，它都生成一条包含 128 个转移的轨迹，冻结该批次的 $Q$ 与策略，并以 $\eta=12$ 和下列步长执行按状态-动作对取均值的采样 SARSA 更新：

$$
\alpha_\ell=\frac{0.35}{\sqrt{1+0.02(\ell-1)}}.
$$

匹配参考实现接收相同的转移和步长，但使用精确匹配核 $U$。它用于分离有限 softmax 匹配的影响，并不是第二个独立采样的控制运行。

**表 1. 采样 SARSA 控制的精确策略回报（30 个 MDP；括号内为样本标准差）。**

| 更新规则 | 初始回报 | 最终回报 | 回报增益 | 正的 $J_\mu$ 增益 |
|---|---:|---:|---:|---:|
| 标准 softmax SARSA | 0.104 (0.211) | 0.533 (0.117) | 0.429 (0.190) | 30/30 |
| 精确匹配写回 | 0.104 (0.211) | 0.532 (0.117) | 0.428 (0.192) | 30/30 |

softmax 学习器平均增益的双侧 Student-$t$ 95% 置信区间为 $[0.358,0.500]$。其贪心策略回报从 0.110 上升到 0.587，而最优回报为 0.621。在最后一次更新时，它的 $Q$ 表与共享轨迹比较器之间的无穷范数差异仅为 $2.19\times10^{-4}$（标准差 $5.27\times10^{-5}$），二者的贪心策略在 0.996 的状态上相同（标准差 0.020）。在相同冻结 $Q$ 下进行的单步检验中，平均差异为 $1.54\times10^{-4}$；所有受检步骤均低于相应的有限核误差界。观测到的最大非匹配注意力质量为 $7.80\times10^{-4}$。这些配对受控结果表明：每个采样任务都获得了正的最终 $J_\mu$ 增益，同时算子具有很高保真度；但它们并不意味着逐状态策略支配，也不意味着每个随机 SARSA 步骤都单调改进。事实上，中间阶段的评估回报有时会下降。

![](figures/softmax-sarsa-policy-improvement.png)

**图 1. 标准 softmax 采样 SARSA 带来正的最终策略回报增益。** 左：控制更新过程中精确回报变化的均值，阴影表示 95% 置信区间。右：每个配对任务的最终回报增益，并叠加共享轨迹的恒等比较器结果。

### 5.3 双阶段路径：策略增益与动作尖锐度

第二个实验从采样噪声中分离内部动作聚合的影响。我们使用另外 20 个 MDP，状态数、动作数与折扣因子均保持不变，并根据精确条件期望更新每个状态-动作对。实验从 $Q_0=0$ 开始，取 $\alpha=0.5$、状态-动作尖锐度 $\eta=20$，执行 200 次同步更新。贪心 $\arg\max$ 出现并列时，包括初始全零时，选择索引最小的动作。精确最大值核算子作为 oracle 参考；提出的算子则取 $\beta\in\{1,2,5,10,20\}$。最终策略关于所得 $Q$ 表为贪心策略，并使用同一并列处理规则。

初始策略回报的均值为 -0.0569（标准差 0.262），而 oracle 达到 0.5209（标准差 0.151），增益为 0.5777（标准差 0.224）。每一种 softmax 温度都在全部 20 个任务上获得正的最终 $J_\mu$ 增益。

**表 2. 经过 200 次期望更新后的双阶段 softmax 控制（20 个 MDP 上的均值）。**

| 算子 | 回报增益 | 最优回报差距 | $Q$ 无穷范数误差 | 贪心一致率 |
|---|---:|---:|---:|---:|
| 精确最大值 oracle | 0.57774 | 0 | $7.48\times10^{-13}$ | 1.000 |
| softmax，$\beta=1$ | 0.57682 | $9.25\times10^{-4}$ | 0.2465 | 0.972 |
| softmax，$\beta=2$ | 0.57766 | $7.51\times10^{-5}$ | 0.1905 | 0.989 |
| softmax，$\beta=5$ | 0.57766 | $7.51\times10^{-5}$ | 0.08758 | 0.989 |
| softmax，$\beta=10$ | 0.57774 | 0 | 0.03441 | 1.000 |
| softmax，$\beta=20$ | 0.57773 | $8.98\times10^{-6}$ | 0.01131 | 0.994 |

动作价值误差随 $\beta$ 增大而降低。观测到的逐状态 softmax 与最大值之差，从 $\beta=1$ 时的 0.3618 降至 $\beta=20$ 时的 0.02007，低于相应的理论界 1.386 和 0.06931。贪心策略是 $Q$ 的分段常值函数，因此，即使动作价值误差尚未消失，策略回报也可能已经与 oracle 相同；表中因此同时报告这两个量，而不把 $Q$ 误差当成策略改进的替代指标。

![](figures/two-stage-temperature.png)

**图 2. 双阶段算子获得正的策略回报增益，并趋近贪心动作选择。** 左：回报增益均值，误差棒为样本标准差；虚线表示精确最大值 oracle。右：观测到的 softmax 与最大值之差始终低于 $\log(|A|)/\beta$。

### 5.4 端到端与有限尖锐度检验

在确定性 fixture 上，literal matrix witness 使用一个 $32\times20$ prompt，实例化文中声明的每个投影与完整 query-by-source mask。所有 mask row 都有非空支持，全部 softmax row 均为有限值且归一化；retrieved value、带符号 residual 坐标与预期 token state 一致，未访问 query 被路由到零值 null token。最终 Q 更新与紧凑实现及显式 frozen-batch 表格参考的最大差异均为 $5.55\times10^{-17}$；连续六次 singleton update 也与顺序表格 SARSA 一致，八个未访问 pair 保持不变。这是完整构造的 implementation witness，与大规模 control sweep 相互独立。故意降低 retrieval 与 write-back sharpness 到 $(\zeta,\eta)=(2.4,2.1)$ 时，观测 retrieval error 为 0.425，低于 0.999 的界；完整 end-to-end update error 为 0.250，低于命题 3.7 给出的 1.170 界。

较早的紧凑直接公式检验显示，精确匹配逐对 write-back 误差为 $1.11\times10^{-16}$。在 control experiment 使用的 $\eta=12$ 下，观测到的合成单步有限核误差为 $8.55\times10^{-5}$，低于 $2.00\times10^{-4}$ 的计算界。在双阶段实验的 20 次更新尖锐度扫描中，将非匹配质量从 $\eta=1$ 时的 0.928 降至 $\eta=10$ 时的 0.00159，使平均 $Q$ 误差从 0.547 降至 0.01158；取 $\eta=20$ 时仍为 0.01158。这些检验支持有限步 retrieval 与 write-back distortion 分析，但不用于声称 leakage 必然形成不可避免的渐近 error floor。

# 6. 讨论

### 6.1 对策略改进问题的直接回答

可以，但结论是本文建立的精确 oracle-routed 构造意义下的“可以”。在路径 I 中，两个 equality-routed retrieval head 从 Q-memory 读取当前与 sampled next-pair value，固定 ReLU map 形成带符号 SARSA residual，final matching head 执行逐对均值 write-back；精确恒等式假设外部 mask 与 visited/null gate。Equality-mask-free 的有限 logit retrieval 与 write-back 在保留 gate 的前提下近似精确版本。闭合外部 operator--policy loop 后，受控任务得到正的最终 $J_\mu$ 增益，其中大规模 sweep 运行的是 compact direct-indexing implementation。在路径 II 中，action-softmax head 提供当前 Boltzmann 策略的期望续接价值，并对贪心动作选择给出受控近似。实证主张建立在 before--after $J_\mu$ 上，而不是 TD error 下降或逐状态支配。

这一答案不依赖 Q-learning。只要从当前动作价值更新行为策略，SARSA 就已经成为控制方法 [6]。保留基于最大值的路径，只是因为它能证明标准 softmax 也可以在内部执行近似贪心化，而且 Bellman 最优性能够提供更强的邻域保证。

### 6.2 与线性注意力 SARSA 的关系

Liang 和 Lai [4] 使用线性注意力更新共享参数向量，这需要构造 TD residual 与 feature vector 的样本相关乘积。我们的构造把 $Q$ 直接存于 state--action memory token；标准 softmax head 先检索两个所需标量，再把带符号 residual 路由回匹配 memory token，而不需要动态 residual--feature product。两种表示方式不同，但都实现 SARSA control update。相对于固定策略 softmax TD 构造 [5]，持久化 state--action memory、显式 Q retrieval 与行为策略闭环是关键变化。

### 6.3 局限性

理论是表格型的，并假设每个状态--动作对有一个持久化 memory token。精确 retrieval 与 write-back 使用外部提供、输入相关的结构化 equality mask 与外部 visited-query/null-token gate；有限 equality-mask-free logit 仍为 gate-assisted 近似。精确网络无 LayerNorm 与 dropout，也不是 decoder-only 因果 Transformer。采样实验在每个 batch 内冻结 $Q$ 与行为策略，直接索引两个 Q 值并使用逐对均值 update；经典 online SARSA convergence 则适用于 GLIE 与逐 pair Robbins--Monro 条件下的 singleton sequential update。固定 block 给出固定步长，因此 variable schedule 需要 block family 或额外 multiplication mechanism。双阶段 expected experiment 具有 full coverage，分离的是 operator bias 而非 finite-trajectory concentration。精确 $\varepsilon$-greedy argmax、随机策略采样与环境 action execution 都在 attention block 外。扩展到 learned representation 还必须处理 approximate identifier、learned 或 mask-free routing、support gating、coverage 与 persistent-memory scaling。

# 7. 结论

标准归一化 softmax 注意力能够支持上下文内策略改进。核心机制是带符号的状态-动作价值写回：注意力权重执行上下文匹配，而正、负 TD 修正仍保留在 value 向量中。

采样 SARSA 构造给出直接的同策略路径。从原始 transition identifier 与 reward 出发，两个 externally equality-routed softmax head 从持久化 Q-memory 检索当前与 sampled next-pair value，显式的两单元 ReLU map 形成带符号 residual，state--action head 再写回已访问的 Q-memory token。外部 visited-query/null-token gate 处理未出现 pair。单例构造生成通常的顺序 SARSA iterates。移除 equality mask 后得到 finite-logit retrieval 与 write-back 近似，但仍保留该 gate。将更新后的价值与下一轮行为策略结合，就闭合了外部 interaction loop，无需在 TD target 中使用最大值。

双阶段构造给出了更强的内部动作选择结果。其有限温度动作聚合是当前 Boltzmann 策略下的期望续接价值；加入奖励与折扣后，即得到条件 Expected-SARSA 目标。随着动作注意力变得尖锐，同一续接价值趋近贪心价值，误差至多为 $\log(|A|)/\beta$。将这一偏差与 Bellman 最优算子的压缩性结合，可在理想表格设定中得到 $Q^{\star}$ 周围的邻域保证，以及相应的贪心策略性能界。

实验通过策略行为评估两条路径。大规模单阶段 sweep 用 direct table indexing 取得两个 Q 值，并检验紧凑 residual/write-back control operator 与共享轨迹 exact-match comparator；它不是完整 prompt 网络的运行。独立的 literal matrix witness 验证 full prompt、projections、routing masks、residual states 与 write-back。双阶段紧凑算子则在改变动作与核 sharpness 时与精确 greedy control 比较。两类结果共同表明：sampled on-policy update 能带来最终 policy-return gain；增加一个 stage 后，approximate internal greedification 也能产生该增益。

精确结果仍是对固定 problem size 与 update parameter 的 oracle-routed、fixed-weight token-level 构造。它的双 block residual attention--FFN 网络无 LayerNorm 与 dropout，采用双向结构化注意力而非 decoder-only 因果 attention，并依赖外部提供、输入相关的 equality-routing mask 与外部 visited-query/null-token gate。它没有把 environment simulation 或随机动作 draw 放入 deterministic attention，没有证明 stochastic sampling 下每次更新都单调改进，没有证明 pretrained Transformer 会自动发现路由，也没有给出超出所述 coverage condition 的有限表格保证。扩展到 learned representation、learned 或 mask-free exact routing 以及 finite-trajectory coverage，是主要开放方向。

# 参考文献

1. J. von Oswald, E. Niklasson, E. Randazzo, J. Sacramento, A. Mordvintsev, A. Zhmoginov, and M. Vladymyrov. Transformers Learn In-Context by Gradient Descent. ICML, 2023. arXiv:2212.07677.
2. E. Akyurek, D. Schuurmans, J. Andreas, T. Ma, and D. Zhou. What Learning Algorithm Is In-Context Learning? Investigations with Linear Models. ICLR, 2023. arXiv:2211.15661.
3. S. Garg, D. Tsipras, P. Liang, and G. Valiant. What Can Transformers Learn In-Context? A Case Study of Simple Function Classes. NeurIPS, 2022. arXiv:2208.01066.
4. H. Liang and L. Lai. Transformers Provably Implement In-Context Reinforcement Learning with Policy Improvement. 2026. arXiv:2605.05755.
5. Z. Xie, X. Liu, C. Chen, S. D. Liu, R. Chandra, and S. Zhang. Beyond Linear Attention: Softmax Transformers Implement In-Context Reinforcement Learning. 2026. arXiv:2605.07333.
6. R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction. Second edition, MIT Press, 2018.
7. O. Richter and R. Wattenhofer. Normalized Attention Without Probability Cage. ICLR, 2021. arXiv:2005.09561.
8. P. Velickovic, C. Perivolaropoulos, F. Barbero, and R. Pascanu. Softmax Is Not Enough (for Sharp Size Generalisation). ICLR, 2025. arXiv:2410.01104.
