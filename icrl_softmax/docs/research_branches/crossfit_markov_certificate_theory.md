# Cross-fit 与 Markov Certificate 理论说明

> 日期：2026-08-29  
> 证据等级：原始定理摘录 + 本项目推导 + 明示未闭合项

## 1. 原始 Markov Hoeffding 定理

Fan、Jiang 与 Sun 的正式 JMLR 论文处理一般状态空间、可非可逆的 Markov chain。[Fan, Jiang & Sun (2021)](https://www.jmlr.org/papers/v22/19-479.html)

### 1.1 Stationary、time-dependent 函数

论文 Theorem 1 假设：

- 链从平稳分布 \(\pi\) 出发；
- Markov operator 在 \(L_2^0(\pi)\) 上的 absolute spectral gap 为 \(1-\lambda>0\)；
- 每个时刻允许使用不同的有界函数 \(f_i\in[a_i,b_i]\)。

其 Hoeffding variance inflation factor 为

\[
\omega_{\mathrm{abs}}
=
\frac{1+\lambda}{1-\lambda}.
\]

这是处理预先固定、随时间变化函数的直接工具，但不能自动覆盖“函数由同一条轨迹自适应估计”的情形。

### 1.2 Stationary、time-independent 函数

论文 Theorem 3 对同一个有界函数 \(f\in[a,b]\) 使用 additive reversiblization

\[
R=(P+P^*)/2
\]

的 right spectral value \(\lambda_r<1\)，得到更紧的 inflation factor

\[
\omega_r
=
\frac{1+\max\{\lambda_r,0\}}
       {1-\max\{\lambda_r,0\}}.
\]

对 sample mean，有双侧形式

\[
\Pr_\pi\left(
\left|
\frac1n\sum_{t=1}^n f(X_t)-\pi(f)
\right|>\varepsilon
\right)
\le
2\exp\left(
-\frac{2n\varepsilon^2}
{\omega_r(b-a)^2}
\right).
\]

### 1.3 非平稳初始分布

论文 Theorem 12 允许初始分布 \(\nu\ne\pi\)，但要求 \(\nu\ll\pi\) 且 \(d\nu/d\pi\) 有有限 \(p\)-moment。概率界会增加依赖初始分布、burn-in \(n_0\)、absolute gap 和 \(p\) 的乘法常数 \(C(\nu,n_0,p)\)。

本项目合成实验直接从精确平稳分布采初态，因此主要 certificate 使用 stationary 版本；这不能外推到未知环境中无法采平稳初态的算法。

## 2. Pair coverage 的直接证书

令 \(X_t=(S_t,A_t)\)，pair chain 的状态数为 \(d=|\mathcal S||\mathcal A|\)，平稳分布为 \(\mu_X\)。对每个 pair \(x\)，取

\[
f_x(X_t)=\mathbf 1\{X_t=x\}.
\]

这是 time-independent、值域为 \([0,1]\) 的函数。对 \(d\) 个 pair 做 union bound，置信水平 \(1-\delta\) 下：

\[
\max_x
\left|
\frac{N_x}{n}-\mu_X(x)
\right|
\le
\varepsilon_B,
\]

\[
\varepsilon_B
=
\sqrt{
\frac{\omega_{r,X}}{2n}
\log\frac{2d}{\delta}
}.
\]

因此，若

\[
\mu_{X,\min}>\varepsilon_B,
\]

则同一事件上所有 pair 均被访问，且

\[
\min_x N_x
\ge
n(\mu_{X,\min}-\varepsilon_B).
\]

这一结论可直接作为 **conditional coverage certificate**，前提是：

- pair chain 从平稳分布出发；
- right spectral gap 为正；
- 使用精确 transition matrix 计算 \(\lambda_{r,X}\)。

## 3. 为什么 numerator 需要 edge chain

Direct-Q residual 和 V-first recovery target 都依赖相邻 pair：

\[
Y_t
=
R_{t+1}+\gamma W(X_{t+1}).
\]

它不是 \(X_t\) 的函数。定义 transition-edge chain

\[
E_t=(X_t,X_{t+1}).
\]

若 pair transition 为 \(P_X\)，则

\[
P_E((x,y),(y,z))=P_X(y,z),
\]

平稳分布为

\[
\mu_E(x,y)=\mu_X(x)P_X(x,y).
\]

对固定的有界 \(W\)，

\[
h_x(E_t)
=
\mathbf 1\{X_t=x\}
\bigl(R_{t+1}+\gamma W(X_{t+1})\bigr)
\]

是 edge chain 上的 time-independent 函数。若 \(|Y_t|\le Y_{\max}\)，对全部 pair union bound 可取

\[
\varepsilon_A
=
Y_{\max}
\sqrt{
\frac{2\omega_{r,E}}{n}
\log\frac{2d}{\delta}
}.
\]

若 \(\varepsilon_B\le\mu_{X,\min}/2\)，ratio lemma 给出

\[
\max_x
\left|
\frac{\widehat A_x}{\widehat B_x}
-\frac{A_x}{B_x}
\right|
\le
\frac{2}{\mu_{X,\min}}
\left(
\varepsilon_A+Y_{\max}\varepsilon_B
\right).
\]

这解释了 certificate 必须同时报告 pair-chain 与 edge-chain spectral factor。

## 4. Direct-Q certificate 已闭合的部分

2026-08-31 的共享事件证明关闭了“\(Q_\ell\) data-dependent，所以必须逐层 concentration”的旧缺口。经验算子在同一条冻结轨迹上固定；概率事件只需控制：

- pair counts；
- 真实 \(Q^\pi\) 处的固定 centered residual；
- 由 count lower bound 推出的经验 one-hot diagonal；
- exact/softmax 路线各自的 contraction margin。

对每个 pair \(z\)，

\[
\bar\delta_z^\pi
=\frac1{N_z}\sum_{t:X_t=z}
\left[R_{t+1}+\gamma Q^\pi(X_{t+1})-Q^\pi(z)\right],
\]

且 full coverage 下

\[
[\widehat{\mathcal F}_{Q,1}(Q^\pi)-Q^\pi](x)
=\sum_z\widehat M(x,z)\bar\delta_z^\pi.
\]

所以经验 fixed-point bias 由 \(\max_z|\bar\delta_z^\pi|\) 控制。共享事件成立后，对所有层的递推是确定性的，无需 layerwise union bound。完整常数见 [shared_fixed_policy_finite_sample_theory.md](shared_fixed_policy_finite_sample_theory.md)。

旧 **coverage_certified**、**kernel_certified** 和 **direct_operator_diagnostic** 字段为历史 primitive 诊断保留。新结果另存：

- **high_probability_certified**；
- **pathwise_bound_verified**；
- route-level margin、\(\rho\)、optimization/statistical term 与 total bound。

仍未覆盖的是非平稳起步、额外随机 reward noise、缺失 pair、随层改变的 score/operator 和 online control，而不是固定轨迹上的 iterate 自适应性。

## 5. Cross-fit gap 的理论等级

Fan 等人的定理控制单个 Markov sample average，并不直接给出“两数据块近似独立”的 cross-fitting theorem。

不过，fixed-target ghost 分解本身不要求两折独立。对每折分别控制固定 \(V^\pi\) recovery residual，并使用 recovery 算子的路径式 \(\gamma\)-Lipschitz 性，再对有限个事件做 union bound 即可。因而 gap 不是 fixed-policy no-split 证明的必要条件；它只在坚持“给定训练折后，把评估折当近似独立样本再 concentration”的替代证明中承担 coupling 作用。

时间序列 sample splitting 在 \(\beta\)-mixing 或其他弱依赖条件下可获得渐近有效性，但这类结果的统计目标与本项目的 plug-in \(Q\) recovery 不同。[Lunde (2019)](https://arxiv.org/abs/1902.07425)

依赖数据中的 neighbor-excluding cross-fitting 为“训练折与评估折之间删除邻近样本”提供了直接设计先例，并可通过 coupling 控制近似独立误差；本项目尚未把其条件和常数迁移到 Markov RL recovery。[Semenova et al. (2023)](https://onlinelibrary.wiley.com/doi/full/10.3982/QE1670)

本项目计算

\[
d_{\max}(g)
=
\max_{\text{state,pair,edge}}
\max_{\text{forward,reverse}}
\sup_x
\|P^g(x,\cdot)-\mu\|_{\mathrm{TV}}.
\]

其用途是：

- 选择 blocked cross-fit 的 gap；
- 量化删去 gap 后仍残留的最坏分布依赖；
- 标记 gap 是否因预算限制而被截断。

在新的 coupling 或 stability 证明完成前，\(d_{\max}(g)\) 只称为 **dependency diagnostic**，不作为 \(\varepsilon_{\mathrm{dep}}(g)\) 的已证明数值上界。

## 6. 与 Xie fixed-policy 分析的关系

Xie 等人的 weighted-softmax TD 分析提供 state-value population contraction 与有限轨迹扰动框架，但其正式设定是 fixed-policy finite-state MRP，并使用状态链和相邻状态对链的浓缩。[Xie et al. (2026)](https://arxiv.org/abs/2605.07333)

本项目的迁移关系为：

| Xie 分析对象 | 本项目对应对象 | 状态 |
|---|---|---|
| state chain \(S_t\) | Direct-Q 的 pair chain \(X_t=(S_t,A_t)\) | 代数迁移已完成 |
| transition-pair chain \((S_t,S_{t+1})\) | edge chain \((X_t,X_{t+1})\) | 本阶段加入 certificate |
| deterministic state reward | transition-dependent bounded reward | 需要 edge-chain 扩展 |
| fixed initial theorem conditions | synthetic stationary start | 实验中满足 |
| fixed-policy evaluation | blockwise/online control | 不可直接外推 |

## 7. 本阶段可诚实报告的结论

完成实现后，可以报告：

1. exact synthetic MDP 下 state/pair/edge chain 的谱与 mixing 指标；
2. pair coverage 的显式 stationary Hoeffding certificate；
3. 固定 bounded target 的 ratio certificate 组成部分；
4. kernel diagonality 的 population 与 empirical slack；
5. cross-fit gap 的 exact total-variation dependency diagnostic；
6. cross-fit 的数值误差分解；
7. frozen-context Direct-Q 的 uniform-in-layer 高概率界；
8. same-sample V-first no-split 的 fixed-ghost 高概率界与路径验证。

仍不能报告：

- blocked cross-fit 已经有限样本独立；
- certificate 可在未知环境中无需 transition model 计算；
- 非平稳起步或额外随机 reward noise 沿用当前常数；
- fixed-policy certificate 自动推出控制收敛。

## 8. 正式扫描后的校准

480 个 matched comparisons 的正式扫描给出三个与理论边界一致的现象：

1. 无 gap cross-fit 使用全部 \(N\) 个 transition 做 recovery，其总体 \(Q\) sup error 为 0.5873；Direct-Q 为 0.5852。任务级配对差 \(+0.0022\) 的 95% CI 为 \([-0.0079,0.0122]\)。
2. gap cross-fit 总体误差为 0.6494。\(N=256,1024\) 时 recovery 保留率分别为 78.1% 和 84.8%，且 gap 在 52.5% 和 50.0% 的任务中被 \(N/8\) 上限截断；这是依赖控制与有效样本量之间的直接有限样本权衡。
3. 最保守的 stationary pair-count coverage certificate 在全部配置中均未通过，但经验全 pair 覆盖率在 \(N=4096,16384\) 时达到 100%，kernel certificate 通过率分别为 95.8% 和 100%。

第三点尤其重要：当前 Hoeffding + union-bound certificate 是充分条件，不是必要条件。它的失败不能解释成 coverage 实际失败；更合理的下一步是推导 occupancy-adaptive/Bernstein 型界，或对 ratio 直接做 self-normalized 控制。

这些结果校准了 primitive certificate 的保守程度和 gap 的经验代价。第 5 节的 gap 仍只是 dependency diagnostic；但这不再阻止 fixed-policy Direct-Q/no-split 定理，因为新证明没有把折间独立作为前提。

## 9. 共享事件正式复现

新目录 results/fixed_policy_finite_sample_certificates/ 用相同 seed 和 480 个配置完成复现。旧 10 条路线的共同字段 mismatch 为 0，最大浮点差为 \(8.88\times10^{-16}\)。

显式共享事件给出：

- state/pair/edge 谱条件通过率 100%；
- 数值 edge-support 截断率 0%；
- conservative pair coverage lower bound 在全部长度上通过率 0%；
- 因而先验高概率证书通过率为 0%，但这不否定定理，只说明 sufficient coverage 条件过于保守；
- 观测 full-support 路径界在 \(N=4096,16384\) 时，Direct-Q exact 与 no-split exact/softmax 的验证率均为 100%。

真正剩余的统计问题是更尖锐的 occupancy-adaptive coverage/residual rate，而不是对每层 iterate 或 no-split 阶段独立性再做额外 concentration。
