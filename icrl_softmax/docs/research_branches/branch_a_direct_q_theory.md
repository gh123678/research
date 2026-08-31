# Branch A：Direct-Q 理论草案

> 日期：2026-08-29  
> 状态：population 与 stationary single-trajectory 显式证书均已闭合；coverage rate 仍保守  
> 主要来源：Xie et al. (2026), Fan et al. (2021)

## 1. 目标与主张级别

本分支研究固定策略 \(\pi\) 下，直接在 \(\mathcal X=\mathcal S\times\mathcal A\) 上运行 weighted-softmax TD。目标是估计 \(Q^\pi\)，而不是 \(Q^\star\)。

本文档区分：

- **Proved here**：本文档中已给出完整代数证明；
- **Conditional theorem**：在明确经验偏差事件上已证明；
- **High-probability theorem**：由共享固定函数事件与路径递推闭合；
- **Empirical question**：交由统一实验验证。

## 2. State-action MRP

令 \(x=(s,a)\)、\(x'=(s',a')\)。固定策略诱导 pair-chain：

\[
P_X^\pi(x,x')
=p(s'\mid s,a)\pi(a'\mid s').
\]

定义 pair reward mean：

\[
r_X(x)=\mathbb E[R_{t+1}\mid X_t=x].
\]

对应 Bellman operator 为

\[
(T_X^\pi Q)(x)
=r_X(x)+\gamma\sum_{x'}P_X^\pi(x,x')Q(x').
\]

### Proposition A.1（pair-chain Bellman fixed point；Proved here）

对于有限 \(\mathcal X\) 和 \(0<\gamma<1\)，\(T_X^\pi\) 是 \(\ell_\infty\) 下的 \(\gamma\)-contraction，并具有唯一不动点 \(Q^\pi\)：

\[
Q^\pi=r_X+\gamma P_X^\pi Q^\pi.
\]

**证明。** 对任意 \(Q,Q'\)，由于 \(P_X^\pi\) 是 row-stochastic，

\[
\|T_X^\pi Q-T_X^\pi Q'\|_\infty
=\gamma\|P_X^\pi(Q-Q')\|_\infty
\le\gamma\|Q-Q'\|_\infty.
\]

Banach fixed-point theorem 给出唯一不动点。展开该不动点的期望定义，正是先执行 \((s,a)\)，随后按 \(\pi\) 行动的 discounted return，即 \(Q^\pi\)。\(\square\)

### Stationary occupancy

若 state chain 的 stationary distribution 为 \(\mu^\pi\)，则 pair-chain 的候选 stationary distribution 是

\[
\mu_X^\pi(s,a)=\mu^\pi(s)\pi(a\mid s).
\]

直接验证：

\[
\begin{aligned}
&\sum_{s,a}\mu^\pi(s)\pi(a\mid s)
p(s'\mid s,a)\pi(a'\mid s')\\
&=\pi(a'\mid s')
\sum_s\mu^\pi(s)P_S^\pi(s,s')\\
&=\mu^\pi(s')\pi(a'\mid s').
\end{aligned}
\]

为避免从 state-chain ergodicity 到 pair-chain ergodicity 的隐含跳步，第一版定理直接假设 \(P_X^\pi\) 在目标支持集

\[
\mathcal X_\pi
=\{(s,a):\mu^\pi(s)\pi(a\mid s)>0\}
\]

上 irreducible 且 aperiodic。之后可再证明更弱的充分条件。

## 3. Population weighted-softmax Q operator

令 \(g_X:\mathcal X\times\mathcal X\to\mathbb R\) 为固定 score。定义 population kernel

\[
M_X^\pi(x,z)
=\frac{\mu_X^\pi(z)e^{g_X(x,z)}}
{\sum_{u\in\mathcal X_\pi}\mu_X^\pi(u)e^{g_X(x,u)}}.
\]

它是 row-stochastic。以矩阵记号定义 relaxed population update：

\[
\mathcal F_{A,\alpha}(Q)
=Q+\alpha M_X^\pi
\bigl(r_X+\gamma P_X^\pi Q-Q\bigr),
\qquad 0<\alpha\le1.
\]

即

\[
\mathcal F_{A,\alpha}(Q)
=\bigl(I-\alpha M_X^\pi
+\alpha\gamma M_X^\pi P_X^\pi\bigr)Q
+\alpha M_X^\pi r_X.
\]

由 Proposition A.1，\(Q^\pi\) 是 \(\mathcal F_{A,\alpha}\) 的不动点。

## 4. 一般步长的对角收缩

### Lemma A.2（row-stochastic diagonal bound；Proved here）

令 \(M,N\) 为同维 row-stochastic 矩阵，\(0<\alpha\le1\)。若

\[
d=\min_iM(i,i),
\]

则

\[
\|I-\alpha M+\alpha\gamma N\|_\infty
\le
1-\alpha[2d-(1+\gamma)].
\]

**证明。** 对第 \(i\) 行，因 \(1-\alpha M(i,i)\ge0\)，

\[
\begin{aligned}
&\sum_j\left|
(I-\alpha M+\alpha\gamma N)(i,j)
\right|\\
&\le
1-\alpha M(i,i)+\alpha\gamma N(i,i)
+\alpha\sum_{j\ne i}M(i,j)
+\alpha\gamma\sum_{j\ne i}N(i,j)\\
&=1+\alpha(1+\gamma)-2\alpha M(i,i)\\
&\le1-\alpha[2d-(1+\gamma)].
\end{aligned}
\]

对所有行取最大值得证。\(\square\)

### Corollary A.3（population contraction；Proved here）

若存在

\[
0<C_A<\frac{1-\gamma}{2}
\]

使

\[
\min_xM_X^\pi(x,x)
\ge\frac{1+\gamma}{2}+C_A,
\]

则

\[
\|\mathcal F_{A,\alpha}(Q)
-\mathcal F_{A,\alpha}(Q')\|_\infty
\le(1-2\alpha C_A)
\|Q-Q'\|_\infty.
\]

由于 \(C_A<(1-\gamma)/2\) 且 \(0<\alpha\le1\)，该 contraction factor 严格位于 \((0,1)\)。Xie 的 finite-context 证明会损失一半 population margin，因此其最终几何因子对应 \(1-\alpha C_A\)。

## 5. One-hot pair kernel 的 sharpness barrier

令 pair feature 为 one-hot，并令

\[
g_X(x,z)=\eta\mathbf1\{x=z\}.
\]

则

\[
M_X^\pi(x,x)
=\frac{\mu_X^\pi(x)e^\eta}
{\mu_X^\pi(x)e^\eta+1-\mu_X^\pi(x)}.
\]

### Lemma A.4（达到指定 diagonal mass 的充要条件；Proved here）

对 \(d\in(0,1)\)，条件 \(M_X^\pi(x,x)\ge d\) 等价于

\[
\eta
\ge
\log\frac{d[1-\mu_X^\pi(x)]}
{(1-d)\mu_X^\pi(x)}.
\]

**证明。** 交叉相乘并整理：

\[
(1-d)\mu_X^\pi(x)e^\eta
\ge d[1-\mu_X^\pi(x)].
\]

取对数得证。\(\square\)

对 contraction threshold

\[
d_A=\frac{1+\gamma}{2}+C_A,
\]

统一 sharpness 至少需要

\[
\eta
\ge
\log\frac{d_A(1-\mu_{X,\min}^\pi)}
{(1-d_A)\mu_{X,\min}^\pi}.
\]

由于

\[
\mu_{X,\min}^\pi
\le\mu_{S,\min}^\pi\pi_{\min},
\]

小动作概率会使所需 \(\eta\) 按 \(\log(1/\pi_{\min})\) 增长。这是 Direct-Q 的明确 kernel barrier。

## 6. Empirical operator

给定长度 \(n\) 的 pair trajectory

\[
(X_0,R_1,X_1,\ldots,R_n,X_n),
\]

令

\[
K_n(k\mid x)
=\frac{e^{g_X(x,X_{k-1})}}
{\sum_{m=1}^ne^{g_X(x,X_{m-1})}}.
\]

定义经验矩阵与奖励：

\[
\widehat M_n(x,z)
=\sum_{k=1}^nK_n(k\mid x)
\mathbf1\{X_{k-1}=z\},
\]

\[
\widehat P_n(x,z)
=\sum_{k=1}^nK_n(k\mid x)
\mathbf1\{X_k=z\},
\]

\[
\widehat r_n(x)
=\sum_{k=1}^nK_n(k\mid x)R_k.
\]

两矩阵都是 row-stochastic。经验 update 为

\[
\widehat{\mathcal F}_{A,\alpha}(Q)
=\bigl(I-\alpha\widehat M_n
+\alpha\gamma\widehat P_n\bigr)Q
+\alpha\widehat r_n.
\]

### Lemma A.5（经验 contraction 事件；Proved here）

若

\[
\min_x\widehat M_n(x,x)
\ge\frac{1+\gamma}{2}+c_A,
\qquad c_A>0,
\]

则

\[
\|\widehat{\mathcal F}_{A,\alpha}(Q)
-\widehat{\mathcal F}_{A,\alpha}(Q')\|_\infty
\le(1-2\alpha c_A)
\|Q-Q'\|_\infty,
\]

采用与 Corollary A.3 相同的非负截断约定。

若 population margin 为 \(C_A\)，并且

\[
\|\widehat M_n-M_X^\pi\|_\infty
\le C_A/2,
\]

则经验 diagonal 至少为

\[
\frac{1+\gamma}{2}+\frac{C_A}{2},
\]

从而得到实际使用的 factor

\[
1-\alpha C_A.
\]

## 7. 条件有限轨迹定理

定义 empirical fixed-point bias

\[
b_{A,n}
=\|\widehat{\mathcal F}_{A,1}(Q^\pi)-Q^\pi\|_\infty.
\]

注意步长为 \(\alpha\) 时，一步 bias 是 \(\alpha b_{A,n}\)。

### Theorem A.6（在经验偏差事件上的误差递推；Conditional theorem）

假设

\[
\|\widehat M_n-M_X^\pi\|_\infty\le C_A/2
\]

且 \(b_{A,n}\le\varepsilon_{A,n}\)。固定同一上下文重复 \(L\) 层，\(0<\alpha\le1\)，则

\[
\|Q_L-Q^\pi\|_\infty
\le
(1-\alpha C_A)^L\|Q_0-Q^\pi\|_\infty
+\frac{1-(1-\alpha C_A)^L}{C_A}
\varepsilon_{A,n}.
\]

特别地，

\[
\limsup_{L\to\infty}
\|Q_L-Q^\pi\|_\infty
\le\frac{\varepsilon_{A,n}}{C_A}.
\]

**证明。** 加上减去 \(\widehat{\mathcal F}_{A,\alpha}(Q^\pi)\)，使用 Lemma A.5：

\[
e_{l+1}
\le(1-\alpha C_A)e_l+\alpha\varepsilon_{A,n}.
\]

展开几何级数即得。\(\square\)

这个结果同时说明：在相同 empirical bias 定义下，减小固定步长只减慢深度收敛，不自动降低无限深度统计地板。

## 8. 共享事件闭合与剩余限制

完整显式版本见 [shared_fixed_policy_finite_sample_theory.md](shared_fixed_policy_finite_sample_theory.md)。关键修正是：不需要对数据依赖的 \(Q_\ell\) 逐层做 concentration。

对每个 pair \(z\)，固定真实 \(Q^\pi\) 并定义

\[
\bar\delta_z^\pi
=\frac1{N_z}\sum_{t:X_t=z}
\left[R_{t+1}+\gamma Q^\pi(X_{t+1})-Q^\pi(z)\right].
\]

在 full coverage 下，冻结经验算子满足

\[
[\widehat{\mathcal F}_{A,1}(Q^\pi)-Q^\pi](x)
=\sum_z\widehat M_n(x,z)\bar\delta_z^\pi.
\]

因此

\[
b_{A,n}\le\max_z|\bar\delta_z^\pi|.
\]

pair-count 与固定 \(Q^\pi\) residual 都是预先固定的 bounded functions。对 state、pair、edge 三条链使用 stationary right-gap Hoeffding，并在 \(M=2m+3d\) 个固定函数上做一次 union bound，即得到所有层共享的事件。事件成立后，Theorem A.6 的递推是纯路径代数，对全部 \(L\ge0\) 同时成立，也覆盖数据依赖的 early stopping 层数。

one-hot softmax 的先验经验对角下界为

\[
m_\beta(u_X)
=\frac{e^\beta u_X}{e^\beta u_X+1-u_X},
\qquad
u_X=\mu_{X,\min}-b_X.
\]

若 \(u_X>0\) 且 \(c_Q=m_\beta(u_X)-(1+\gamma)/2>0\)，则

\[
\rho_Q=1-2\alpha c_Q,
\qquad
\|Q_L-Q^\pi\|_\infty
\le\rho_Q^L\|Q_0-Q^\pi\|_\infty
+\frac{1-\rho_Q^L}{2c_Q}\varepsilon_X.
\]

exact matching 取 \(c_Q=(1-\gamma)/2\)，且完全不依赖 \(\beta\)。

仍未覆盖的是真实扩展问题：

1. 额外随机 reward noise 的 martingale 或 augmented-chain concentration；
2. 非平稳起步的 burn-in/初始分布 prefactor；
3. 缺失 pair 时的全空间一致性不可能性；
4. 策略或 attention score 随层改变；
5. 更尖锐的 visit-indexed/self-normalized rate。

## 9. 与现有构造的关系

当前仓库的 exact masked construction 实现 frozen-Q、per-pair mean sampled-SARSA：

\[
Q^+(x)=Q(x)
+\frac{\alpha}{n_x}
\sum_{k:X_k=x}
[R_k+\gamma Q(X'_k)-Q(X_k)].
\]

它是 **operator identity**。本文件的 population/empirical analysis研究的是同一类 state-action residual 聚合在 fixed policy、重复 layer 下能否收敛到 \(Q^\pi\)。两者不能互相替代：

- exact construction 不提供 single-trajectory concentration；
- 本理论不证明通用 Transformer 会发现 equality routing；
- fixed-policy theorem 不证明策略每步变化时收敛。

## 10. Branch A 当前判定

### 已闭合

- pair-MRP Bellman fixed point；
- population weighted-softmax Q operator；
- 一般 \(\alpha\) 的 diagonal contraction；
- one-hot pair sharpness threshold；
- empirical-event 条件递推。
- stationary right-gap 下的共享固定 residual 事件与显式全层证书。

### 主要风险

- \(\mu_{X,\min}^\pi=\mu_{S,\min}^\pi\pi_{\min}\) 同时恶化 kernel diagonal 与 concentration 常数；
- 高 \(\gamma\) 把 diagonal threshold 推近 1；
- 额外随机 reward noise 需要新增分析；
- visited-only gate 与全查询 population operator 仍需统一。

### 最小可证贡献

即使完整高概率定理最终过于保守，以下结果仍可独立成立：

> fixed-policy \(Q^\pi\) evaluation 等价于 pair-MRP weighted-softmax TD；其 population contraction 由 state-action kernel diagonal 控制，而 one-hot kernel 达到该条件所需的 sharpness 显式依赖最小 state-action occupancy。
