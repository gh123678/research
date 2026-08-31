# Branch B：V-first 理论草案

> 日期：2026-08-29  
> 状态：population、stationary same-trajectory recovery 与显式共享事件均已闭合；更尖锐 rate 仍开放  
> 主要来源：Xie et al. (2026), Fan et al. (2021), Pananjady and Wainwright (2020)

## 1. 目标与路线

固定策略 \(\pi\) 下，Branch B 分成两个阶段：

1. 使用 state kernel 的 weighted-softmax TD 估计 \(V^\pi\)；
2. 通过 action-conditioned one-step target 恢复 \(Q^\pi\)。

路线的潜在优势是：迭代 Bellman evaluation 发生在 \(|\mathcal S|\) 维状态空间，其 kernel diagonality 只依赖 state occupancy。潜在缺点是：完整恢复所有 \(Q^\pi(s,a)\) 仍需要每个目标 pair 的数据。

## 2. 第一阶段：weighted-softmax V evaluation

沿用 Xie et al. 的 fixed-policy MRP：

\[
(T^\pi V)(s)
=r^\pi(s)+\gamma\sum_{s'}P^\pi(s,s')V(s').
\]

weighted-softmax update 为

\[
V_{l+1}(s)
=V_l(s)+\alpha_l
\sum_{k=1}^{n_1}K_S(k\mid s)
[R_k+\gamma V_l(S_k)-V_l(S_{k-1})].
\]

Xie 的正式 convergence theorem 使用 \(\alpha_l=1\)。在其 ergodicity、full-state coverage 和 population diagonal condition 下，以高概率有

\[
\|\widehat V-V^\pi\|_\infty
\le
\rho_V^L\|V_0-V^\pi\|_\infty
+\varepsilon_V(n_1,\delta),
\]

其中

\[
\rho_V=1-C_V,
\qquad
\varepsilon_V
=C'_V\sqrt{\frac{\log(C''_V/\delta)}{n_1}}.
\]

这里的常数依赖 state-chain mixing、\(\mu_{S,\min}^\pi\)、状态数、feature norm、\(\gamma\) 和 \(\|V^\pi\|_\infty\)。Branch B 不改变这部分定理范围。

## 3. Population V-to-Q recovery

对任意有界函数 \(W:\mathcal S\to\mathbb R\)，定义

\[
\bar Q_W(s,a)
:=\mathbb E[R_{t+1}+\gamma W(S_{t+1})\mid S_t=s,A_t=a].
\]

Bellman identity 给出

\[
Q^\pi=\bar Q_{V^\pi}.
\]

### Proposition B.1（population recovery stability；Proved here）

对任意有界 \(W\)，

\[
\|\bar Q_W-Q^\pi\|_\infty
\le
\gamma\|W-V^\pi\|_\infty.
\]

**证明。** 对每个 \((s,a)\)，

\[
\begin{aligned}
|\bar Q_W(s,a)-Q^\pi(s,a)|
&=\gamma\left|
\mathbb E[W(S')-V^\pi(S')\mid s,a]
\right|\\
&\le\gamma\|W-V^\pi\|_\infty.
\end{aligned}
\]

取最大值得证。\(\square\)

这个界的常数恰为 \(\gamma\)，不含 \((1-\gamma)^{-1}\)。该放大因子只在进一步把 Q error 转成长期 policy-performance error 时出现。

## 4. Clipping 与 target 有界性

设奖励几乎处处满足 \(|R_{t+1}|\le R_{\max}\)。则

\[
\|V^\pi\|_\infty
\le V_{\max}:=\frac{R_{\max}}{1-\gamma}.
\]

定义逐状态 clipping：

\[
\widetilde V(s)
=\operatorname{clip}
(\widehat V(s),-V_{\max},V_{\max}).
\]

### Lemma B.2（clipping 不增加误差；Proved here）

\[
\|\widetilde V-V^\pi\|_\infty
\le
\|\widehat V-V^\pi\|_\infty.
\]

**证明。** \(V^\pi(s)\) 位于 clipping 区间内；到闭区间的欧氏投影对区间内任一点非扩张。逐坐标应用即可。\(\square\)

于是 recovery target

\[
Y_t
=R_{t+1}+\gamma\widetilde V(S_{t+1})
\]

满足

\[
|Y_t|
\le
Y_{\max}:=R_{\max}+\gamma V_{\max}
=V_{\max}.
\]

## 5. Sample-split recovery estimator

总轨迹长度为 \(n=n_1+n_2\)。第一段估计 \(\widehat V\)，第二段恢复 Q。对 recovery index set \(\mathcal I_2\)，定义

\[
N_x
=\sum_{t\in\mathcal I_2}\mathbf1\{X_t=x\},
\qquad x=(s,a).
\]

当 \(N_x>0\) 时，

\[
\widehat Q_B(x)
=\frac1{N_x}
\sum_{t\in\mathcal I_2}
\mathbf1\{X_t=x\}Y_t.
\]

连续轨迹切分并不会让两段无条件独立。不过，给定第一段生成的 sigma-field 后，\(\widetilde V\) 固定，第二段仍是从随机边界状态开始的 Markov 链。Fan et al. 的界允许一般初始分布，因此可以条件应用；其 prefactor 会吸收非平稳起点。第一版无需假装两段独立，也无需丢弃边界样本。

## 6. Ratio error 的确定性分解

令 recovery block 长度为 \(m=n_2\)。对 pair \(x\)，定义

\[
A_x
=\frac1m\sum_{t\in\mathcal I_2}
\mathbf1\{X_t=x\}Y_t,
\qquad
B_x=\frac{N_x}{m}.
\]

给定 \(\widetilde V\)，population quantities 为

\[
q_{\widetilde V}(x)
=\mathbb E[Y_t\mid X_t=x],
\]

\[
\mathbb E_\mu[A_x]
=\mu_X^\pi(x)q_{\widetilde V}(x),
\qquad
\mathbb E_\mu[B_x]
=\mu_X^\pi(x).
\]

### Lemma B.3（ratio decomposition；Proved here）

设 \(\mu_x=\mu_X^\pi(x)>0\)。若

\[
|A_x-\mu_xq_{\widetilde V}(x)|\le\varepsilon_{A,x},
\]

\[
|B_x-\mu_x|\le\varepsilon_{B,x}
\le\frac{\mu_x}{2},
\]

则 \(N_x>0\)，且

\[
|\widehat Q_B(x)-q_{\widetilde V}(x)|
\le
\frac{2}{\mu_x}
\left(
\varepsilon_{A,x}
+Y_{\max}\varepsilon_{B,x}
\right).
\]

**证明。** 因 \(B_x\ge\mu_x/2>0\)，

\[
\begin{aligned}
\left|\frac{A_x}{B_x}-q_{\widetilde V}(x)\right|
&=\frac{|A_x-B_xq_{\widetilde V}(x)|}{B_x}\\
&\le\frac{
|A_x-\mu_xq_{\widetilde V}(x)|
+|q_{\widetilde V}(x)||B_x-\mu_x|
}{\mu_x/2}.
\end{aligned}
\]

又因 \(|q_{\widetilde V}(x)|\le Y_{\max}\)，结论成立。\(\square\)

## 7. 两阶段总体误差

### Theorem B.4（条件组合界；Conditional theorem）

假设事件 \(\mathcal E_V\) 上

\[
\|\widetilde V-V^\pi\|_\infty
\le\varepsilon_V,
\]

并且对所有目标 pair，Lemma B.3 的两个 deviation event 同时成立。令

\[
\varepsilon_{\mathrm{rec}}
:=
\max_x
\frac{2}{\mu_X^\pi(x)}
\left(
\varepsilon_{A,x}
+Y_{\max}\varepsilon_{B,x}
\right).
\]

则

\[
\|\widehat Q_B-Q^\pi\|_\infty
\le
\gamma\varepsilon_V
+\varepsilon_{\mathrm{rec}}.
\]

**证明。** 加上减去 \(\bar Q_{\widetilde V}\)，使用 Proposition B.1 与 Lemma B.3，再对 pair 取最大值。\(\square\)

这一定理已经把两个阶段完全分开：

\[
\underbrace{\gamma\varepsilon_V}_{\text{state evaluation}}
+
\underbrace{\varepsilon_{\mathrm{rec}}}_{\text{action-conditioned recovery}}.
\]

## 8. Markov concentration 的第一版闭合

给定第一段历史，第二段的 \(\widetilde V\) 固定。对每个 pair，考虑 bounded functions

\[
f_x(X_t,R_{t+1},X_{t+1})
=\mathbf1\{X_t=x\}Y_t,
\]

\[
h_x(X_t)=\mathbf1\{X_t=x\}.
\]

Fan et al. 的 Hoeffding inequality 可在适当 augmented transition chain 上分别控制 \(A_x\) 与 \(B_x\)。用 union bound 后，可得到保守结构

\[
\varepsilon_{A,x}
\le C_A Y_{\max}
\sqrt{\frac{\log(C'_A|\mathcal X|/\delta)}{m}},
\]

\[
\varepsilon_{B,x}
\le C_B
\sqrt{\frac{\log(C'_B|\mathcal X|/\delta)}{m}},
\]

其中常数依赖 pair-chain 或 augmented-chain 的 spectral parameter 与最小 stationary mass。

代入 Theorem B.4 得到保守的 uniform rate：

\[
\varepsilon_{\mathrm{rec}}
=O\left(
\frac{Y_{\max}}{\mu_{X,\min}^\pi}
\sqrt{\frac{\log(|\mathcal X|/\delta)}{m}}
\right).
\]

该 Hoeffding/ratio 推导可能比真实有效样本率

\[
O\left(
Y_{\max}
\sqrt{\frac{\log(|\mathcal X|/\delta)}
{m\mu_{X,\min}^\pi}}
\right)
\]

更松。后一个形式需要 Bernstein、regenerative 或 count-conditioned 分析，当前标为 **proof gap**，不能先写成已证定理。

## 9. Oracle-V 与 coverage barrier

即使 \(\widetilde V=V^\pi\) 完全精确，Theorem B.4 仍留下

\[
\|\widehat Q_B-Q^\pi\|_\infty
\le\varepsilon_{\mathrm{rec}}.
\]

这说明 V-first 不能从统计上绕开未访问动作。若某个目标 pair 满足

\[
N_x=0,
\]

则在无已知模型、无生成式采样器、无跨 pair 结构假设时，\(Q^\pi(x)\) 不可由该 recovery block 识别。

因此 Branch B 的优势不应表述为“只需 state coverage”，而应表述为：

> 迭代 long-horizon evaluation 只需 state kernel；state-action coverage 被限制在一次 one-step recovery，而不参与每一层 contraction。

这可能改善 kernel 条件和误差常数，但最坏 uniform Q rate 仍可由 \(\mu_{X,\min}^\pi\) 控制。

## 10. 已知模型与生成式采样器消融

若 \(r(s,a)\) 与 \(p(\cdot\mid s,a)\) 已知，则

\[
\widehat Q_B(s,a)
=r(s,a)+\gamma
\sum_{s'}p(s'\mid s,a)\widetilde V(s')
\]

没有 recovery sampling error，只剩

\[
\|\widehat Q_B-Q^\pi\|_\infty
\le\gamma\|\widetilde V-V^\pi\|_\infty.
\]

Pananjady and Wainwright 的 synchronous model 更接近这种“每个状态可独立采样”的信息结构，而不是本项目的单轨迹 setting。实验必须把该版本标为 oracle-model / generative ablation。

## 11. Policy-improvement consequence

设 \(\pi'\) 对 \(\widehat Q_B\) greedy，且

\[
\|\widehat Q_B-Q^\pi\|_\infty\le\xi_B.
\]

则对每个状态，

\[
Q^\pi(s,\pi'(s))
\ge\max_aQ^\pi(s,a)-2\xi_B.
\]

从而得到近似 policy-improvement 下界

\[
V^{\pi'}
\ge V^\pi-\frac{2\xi_B}{1-\gamma}\mathbf1.
\]

若

\[
\max_aQ^\pi(s,a)-V^\pi(s)
\ge2\xi_B
\]

对所有状态成立，则选择动作相对旧策略具有非负 one-step advantage，可恢复单调 policy improvement。若希望精确识别唯一 greedy action，则 top-two action gap 需严格大于 \(2\xi_B\)。

## 11.5 Same-sample no-split 闭合

此前把 value estimate 与 recovery 共用同一条轨迹视为未闭合依赖问题，这是不必要的。完整证明见 [shared_fixed_policy_finite_sample_theory.md](shared_fixed_policy_finite_sample_theory.md)。

对 full-coverage 轨迹定义 exact recovery 算子

\[
\widehat{\mathcal R}_{\mathrm{ex}}(W)(x)
=\frac1{N_x}\sum_{t:X_t=x}
\left[R_{t+1}+\gamma W(S_{t+1})\right].
\]

权重固定且归一化，因此逐轨迹有

\[
\|\widehat{\mathcal R}_{\mathrm{ex}}(W)
-\widehat{\mathcal R}_{\mathrm{ex}}(W')\|_\infty
\le\gamma\|W-W'\|_\infty.
\]

在同一个共享事件上，固定 \(V^\pi\) 的 ghost recovery residual 满足

\[
\|\widehat{\mathcal R}_{\mathrm{ex}}(V^\pi)-Q^\pi\|_\infty
\le\varepsilon_X.
\]

所以即使 \(\widehat V_L\) 完全由同一数据生成，仍有

\[
\|\widehat{\mathcal R}_{\mathrm{ex}}(\widehat V_L)-Q^\pi\|_\infty
\le
\gamma\|\widehat V_L-V^\pi\|_\infty+\varepsilon_X.
\]

该分解不使用阶段独立、sample split、cross-fit 或 gap。finite-softmax recovery 的 plug-in 常数仍为 \(\gamma\)，另加

\[
L_\beta=2B[1-m_\beta(u_X)].
\]

一次 softmax recovery 不需要 pair kernel contraction margin；pair margin 只约束迭代 Direct-Q softmax。固定有限 \(\beta\) 下 \(L_\beta\) 可能不消失，所以 finite bound 与 consistency 必须分开陈述。

## 12. Branch B 当前判定

### 已闭合

- population \(V\to Q\) 的 \(\gamma\)-稳定性；
- clipping 不增加 V error；
- sample-split recovery 的 ratio decomposition；
- 两阶段条件组合界；
- same-sample no-split exact ghost-target 界；
- finite-softmax no-split 的 \(\gamma\)-propagation 加显式 leakage 界；
- oracle-V 仍受 pair coverage 控制的结论。

### 主要风险

- Hoeffding/ratio 分析给出 \(1/\mu_{X,\min}\) 的保守依赖，可能掩盖实际优势；
- 连续轨迹切分若继续使用，仍需条件化处理；但它不是 no-split fixed-policy 定理的必要条件；
- recovery 的 augmented chain 与随机 reward 常数尚未完全展开；
- sample splitting 减少两个阶段各自的有效数据。

### 最小可证贡献

即使更尖锐的 concentration rate 尚未闭合，以下结果仍可独立成立：

> V-first 将 long-horizon state-evaluation error 以因子 \(\gamma\) 传入 Q，并把剩余困难隔离为一次 action-conditioned recovery；该分解澄清了 state coverage 优势与 state-action coverage 下界同时存在。
