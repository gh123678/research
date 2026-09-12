# 端到端 Softmax SARSA 构造性证明

> 历史研究依据：保留原推导或复核记录，供已有代码及旧任务引用。本文不表示当前研究主线；最新状态见 [项目入口](../ACTIVE_WORKSPACE.md)。

> 本文档是主文第 3 节的独立阅读版，便于逐步核对矩阵构造；定理编号、公式编号与结论范围均以双语主文为权威。它不扩大主文的任何主张。

## 1. 精确模型类别与目标

令 $\mathcal X=\mathcal S\times\mathcal A$、$m=|\mathcal X|$，固定 $\mathcal X$ 的枚举，并以 $e_x\in\mathbb R^m$ 表示 pair $x$。一次更新输入冻结的 $Q_l$、固定的 $m,N,\gamma,\alpha$，以及 $N$ 条原始转移

$$
(S_k,A_k,R_k,S'_k,A'_k),\qquad k=1,\ldots,N.
$$

记 $X_k=(S_k,A_k)$、$X'_k=(S'_k,A'_k)$ 与 $n_x=\sum_k\mathbf1\{X_k=x\}$。我们要逐矩阵构造一个双 block residual attention--FFN 网络，使其输出

$$
Q_{l+1}(x)=
\begin{cases}
Q_l(x)+\dfrac{\alpha}{n_x}\displaystyle\sum_{k:X_k=x}
[R_k+\gamma Q_l(X'_k)-Q_l(X_k)],&n_x>0,\\[2mm]
Q_l(x),&n_x=0.
\end{cases}
$$

模型采用标准指数 softmax、双向结构化注意力，无 LayerNorm 与 dropout，不是 decoder-only 因果 Transformer。等值路由 mask 由外部提供且依赖输入；已访问查询 / null token 的支持规则也由外部 gate 提供。因此，这是 oracle-routed 的精确参考构造，不证明或声称通用预训练 Transformer 会学习或发现路由。

## 2. 完整 prompt 与 attention convention

取 $d=2m+8$，并使用

$$
\mathbb R^d
=\mathbb R^3_{\rm type}
\oplus\mathbb R^m_{\rm current-ID}
\oplus\mathbb R^m_{\rm next-ID}
\oplus\mathbb R^5_{(r,q,u,v,\delta)}.
$$

令 $\tau_Q,\tau_T,\tau_Z$ 为 type 基向量，构造 Q-memory、transition 与 null token：

$$
\begin{aligned}
M_x&=(\tau_Q,e_x,0_m,0,Q_l(x),0,0,0)^\top,\\
T_k&=(\tau_T,e_{X_k},e_{X'_k},R_k,0,0,0,0)^\top,\\
Z&=(\tau_Z,0_m,0_m,0,0,0,0,0)^\top,\\
H^{(0)}&=[M_{x_1},\ldots,M_{x_m},T_1,\ldots,T_N,Z]
\in\mathbb R^{d\times L},\quad L=m+N+1.
\end{aligned}
\tag{3.1}\label{eq:prompt}
$$

$Z$ 保留 type 坐标，但所有会被 value projection 读取的坐标均为零。初始 $T_k$ 不包含两个 Q 值或预计算 residual。

token 按列排列。令 $P_c,P_n\in\mathbb R^{m\times d}$ 选取 current-ID 与 next-ID block，$\mathbf e_r,\mathbf e_q,\mathbf e_u,\mathbf e_v,\mathbf e_\delta\in\mathbb R^d$ 选取五个 scalar coordinate。对任意 head，定义 query-by-source 权重

$$
A_{ij}=
\frac{\exp((W_Qh_i)^\top(W_Kh_j)/\sqrt{d_h}+\mathcal M_{ij})}
{\sum_s\exp((W_Qh_i)^\top(W_Kh_s)/\sqrt{d_h}+\mathcal M_{is})},
$$

其矩阵输出是 $W_O(W_VH)A^\top$。下面所有 mask row 均有非空支持；非目标 query 只允许读取 $Z$，从而避免全 $-\infty$ softmax。

## 3. Block I：两个检索 head

### 3.1 当前 pair 的 Q 值

取

$$
\begin{gathered}
W_Q^{\rm cur}=P_c,\quad W_K^{\rm cur}=P_c,\quad
W_V^{\rm cur}=\mathbf e_q^\top,\quad W_O^{\rm cur}=\mathbf e_u,\\
\mathcal M^{\rm cur}_{ij}=0
\Longleftrightarrow
(i=T_k,j=M_{X_k})
\ \text{or}\ 
(i\notin\{T_1,\ldots,T_N\},j=Z).
\end{gathered}
\tag{3.2}\label{eq:block1-current}
$$

其余元素为 $-\infty$。$T_k$ 只允许读取 $M_{X_k}$，故权重严格为 1，输出是 $Q_l(X_k)\mathbf e_u$；其他 query 读取零值 $Z$，输出为零。

### 3.2 下一 pair 的 Q 值

第二个 head 与第一个并行：

$$
\begin{gathered}
W_Q^{\rm next}=P_n,\quad W_K^{\rm next}=P_c,\quad
W_V^{\rm next}=\mathbf e_q^\top,\quad W_O^{\rm next}=\mathbf e_v,\\
\mathcal M^{\rm next}_{ij}=0
\Longleftrightarrow
(i=T_k,j=M_{X'_k})
\ \text{or}\ 
(i\notin\{T_1,\ldots,T_N\},j=Z),\\
W_O^{\rm next}\sum_jA^{\rm next}_{T_kj}W_V^{\rm next}h_j
=Q_l(X'_k)\mathbf e_v.
\end{gathered}
\tag{3.3}\label{eq:block1-next}
$$

两个 head 都读取同一个 $H^{(0)}$，并写入互不相交的 $u,v$ 坐标。attention residual 后，

$$
T_k^{(a)}=(\tau_T,e_{X_k},e_{X'_k},R_k,0,
Q_l(X_k),Q_l(X'_k),0)^\top,
$$

而 $M_x$ 与 $Z$ 不变。

## 4. Block I 的 ReLU FFN：精确形成 residual

定义

$$
\begin{gathered}
g^\top=\mathbf e_r^\top+\gamma\mathbf e_v^\top-\mathbf e_u^\top,\qquad
W_1=\begin{bmatrix}g^\top\\-g^\top\end{bmatrix},\\
W_2=\mathbf e_\delta\begin{bmatrix}1&-1\end{bmatrix},\qquad
W_2\operatorname{ReLU}(W_1h)=\mathbf e_\delta g^\top h.
\end{gathered}
\tag{3.4}\label{eq:residual-ffn}
$$

因为 $\operatorname{ReLU}(z)-\operatorname{ReLU}(-z)=z$，transition token 的 $\delta$ 坐标精确变成

$$
\delta_k^S=R_k+\gamma Q_l(X'_k)-Q_l(X_k).
$$

Q-memory 与 null token 的 $r,u,v$ 坐标为零，因此该 FFN 不改变它们。

## 5. Block II：完整 write-back

第二个 block 只启用一个 head：

$$
\begin{gathered}
W_Q^{\rm wr}=P_c,\quad W_K^{\rm wr}=P_c,\quad
W_V^{\rm wr}=\mathbf e_\delta^\top,\quad
W_O^{\rm wr}=\alpha\mathbf e_q,\\
\mathcal M^{\rm wr}_{ij}=0
\Longleftrightarrow
\begin{cases}
i=M_x,j=T_k,X_k=x,&n_x>0,\\
i=M_x,j=Z,&n_x=0,\\
i\in\{T_1,\ldots,T_N,Z\},j=Z.&
\end{cases}
\end{gathered}
\tag{3.5}\label{eq:write-back}
$$

对于已访问的 $M_x$，所有允许 source 都满足

$$
\frac{(P_cM_x)^\top(P_cT_k)}{\sqrt m}=\frac1{\sqrt m},
$$

所以 $n_x$ 个 logit 相等，每个权重为 $1/n_x$。head 输出

$$
\frac{\alpha}{n_x}\sum_{k:X_k=x}\delta_k^S\mathbf e_q.
$$

注意力权重虽为正，value 中的 $\delta_k^S$ 可正可负，因此负 TD correction 可以通过。若 $n_x=0$，query 只读取 $Z$，写回严格为零。

将第二 block 的未用 projection 与 FFN 置零，可得完整输出

$$
Q_{l+1}(x)=
\begin{cases}
Q_l(x)+\dfrac{\alpha}{n_x}\displaystyle\sum_{k:X_k=x}
[R_k+\gamma Q_l(X'_k)-Q_l(X_k)],&n_x>0,\\[2mm]
Q_l(x),&n_x=0.
\end{cases}
\tag{3.6}\label{eq:final-update}
$$

这就逐矩阵证明了定理 3.1。每个展示矩阵都是固定坐标 selector 或固定 scalar multiple；数据依赖只来自 prompt 与外部 mask。

## 6. 三种 update convention 必须区分

(3.6) 是冻结 $Q_l$ 下的逐对均值更新，系数为 $\alpha/n_x$。它不是全局 batch convention

$$
Q_l(x)+\frac{\alpha}{N}\sum_{k:X_k=x}\delta_k^S,
$$

也不是在 batch 内一条一条改变 $Q$ 的 sequential update。只有 singleton call 的 $n_x=1$ 时，它才恢复通常的顺序表格 SARSA 步。

若每个 round 只处理一条转移，以 $\alpha_t$ 实例化的 block 生成顺序 SARSA iterate；但单个 fixed block 只对应 constant step size。可变 Robbins--Monro 步长需要输出 scale 不同的 block family，或额外 multiplication mechanism。每轮还要重建 prompt，或等价地清空并重新初始化 $u,v,\delta$ scratch coordinate。经典 SARSA 收敛定理只在 infinite visitation、逐 pair Robbins--Monro 与 GLIE 等条件下适用于这一 singleton protocol，不能直接套给论文的 frozen-batch、fixed-exploration experiment。

## 7. 去掉 equality mask 后的有限 logit 界

现在移除两个 retrieval equality mask 与 write-back equality mask，但仍保留外部 visited-query gate。因此以下结果是 equality-mask-free but gate-assisted，而不是完全 mask-free 的 Transformer。

若正确 Q-memory logit 为 $\zeta$、其他 $m-1$ 个 logit 为零，则

$$
\begin{gathered}
p_\zeta=\frac{e^\zeta}{e^\zeta+m-1},\qquad
\lambda_R=\frac{m-1}{e^\zeta+m-1},\\
|\widetilde Q_l(x)-Q_l(x)|\le\lambda_R\operatorname{span}(Q_l),\\
|\widetilde\delta_k^S-\delta_k^S|
\le(1+\gamma)\lambda_R\operatorname{span}(Q_l)=:E_R.
\end{gathered}
\tag{3.7}\label{eq:finite-retrieval}
$$

write-back 使用 $\eta\langle e_{X_k},e_x\rangle$ 时，对已访问 pair 有

$$
\begin{gathered}
K_\eta(k|x)=
\frac{\exp(\eta\mathbf1\{X_k=x\})}{n_xe^\eta+N-n_x},\qquad
U(k|x)=\frac{\mathbf1\{X_k=x\}}{n_x},\\
\varepsilon_W(x):=\sum_k|K_\eta(k|x)-U(k|x)|
=\frac{2(N-n_x)}{n_xe^\eta+N-n_x}.
\end{gathered}
\tag{3.8}\label{eq:finite-write-back}
$$

若 $|\delta_k^S|\le B$，加上并减去 $\alpha\sum_kK_\eta(k|x)\delta_k^S$，利用 $K_\eta$ 的 probability normalisation 与 (3.7)--(3.8)，得到

$$
|Q_{l+1}^{\rm soft}(x)-Q_{l+1}^{\rm exact}(x)|
\le\alpha[E_R+B\varepsilon_W(x)].
\tag{3.9}\label{eq:finite-end-to-end-error}
$$

这个界把“Q 检索错了”与“residual 写错地址了”分开。它只是 one-step operator-deviation statement；不能把任意有限轨迹自动变成全局 Bellman recurrence，也不在缺少独立采样、mixing 或 martingale 假设时给出 concentration rate。

## 8. 代码 witness 与主张边界

`verify_end_to_end_sarsa.py` 构造 literal $d\times L$ prompt、上述全部 projection、完整 query-by-source mask、两 retrieval head、ReLU FFN 与 write-back block，并逐层断言 shape、mask support、softmax normalisation、token preservation、正负 TD、visited/unvisited pair。确定性 fixture 的 prompt shape 为 $32\times20$；literal 与 compact、literal 与 tabular reference 的最大误差均为 $5.55\times10^{-17}$，六次 singleton iterate 也一致。

大规模 control sweep 与这个 literal witness 不同：它直接 table-index 两个 Q 值，只运行同一 residual/write-back kernel。因而，完整 prompt 的证据来自定理与 literal witness，策略回报的证据来自紧凑 control experiment；两者不能互换。

本构造不包括 learned routing、environment simulation、精确 $\varepsilon$-greedy argmax 或随机动作 sampling，也不证明通用 pretrained Transformer 会发现这些操作。可选的 action softmax 是另一阶段，用于 Boltzmann Expected SARSA 或 approximate greedification，不是 sampled-SARSA 写回定理成立所必需。
