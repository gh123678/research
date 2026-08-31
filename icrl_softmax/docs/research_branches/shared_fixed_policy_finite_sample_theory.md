# 共享单事件 fixed-policy 有限样本定理

> 状态：已实现、已做公式契约验证、已完成 480 项同种子正式复现  
> 范围：有限状态动作空间、固定策略、平稳起步、冻结轨迹、确定性 edge reward

## 1. 已闭合的两个问题

本阶段关闭了两个此前被误判为需要额外独立性的缺口：

1. Direct-Q 不需要对数据依赖的每个 \(Q_\ell\) 逐层做 Markov concentration。经验算子由整条冻结轨迹一次确定；同一事件只要控制经验收缩和 \(Q^\pi\) 处的固定 residual，路径递推便同时控制全部层。
2. V-first no-split 不需要 value 与 recovery 阶段独立。同一数据上的 recovery 算子对输入 value 逐轨迹是 \(\gamma\)-Lipschitz；在固定 \(V^\pi\) 处做一次 ghost-target concentration 后即可组合。

因此 cross-fit 与 gap 不是这两个 fixed-policy 证明的必要装置。它们仍可作为样本利用率和依赖诊断的对照路线。

当前定理不覆盖：非平稳初始分布、额外随机 reward noise、层间重采样、随 \(Q_\ell\) 改变的 attention score，以及策略持续变化的 online control。

## 2. 原始来源与项目新增部分

### 2.1 Markov concentration

[Fan、Jiang 与 Sun（2021）](https://www.jmlr.org/papers/v22/19-479.html) 的 stationary、time-independent Hoeffding 定理允许对预先固定的有限个 bounded functions 应用尾界，再做 union bound。对 additive reversiblization 的 right spectral value \(\lambda_r<1\)，本项目使用

\[
\omega(\lambda_r)
=
\frac{1+\max\{\lambda_r,0\}}
{1-\max\{\lambda_r,0\}}.
\]

该来源支持固定函数的浓缩，不直接提供 data-dependent iterate 的逐层定理。本项目恰好不需要后者：全部概率控制对象都固定在真实 \(V^\pi,Q^\pi\) 上。

edge chain 的 absolute gap 在实验中可以为零，因此没有调用 time-dependent absolute-gap 路线。state、pair、edge 三条链都只使用 stationary time-independent right-gap 形式。

### 2.2 冻结经验算子收缩

[Xie et al.（2026）](https://arxiv.org/html/2605.07333v2) 把 weighted-softmax TD 写成冻结经验仿射算子的重复应用，并用经验 matching matrix 的对角质量控制 \(\ell_\infty\) 收缩。其核心行范数计算是

\[
\|I-\widehat M+\gamma\widehat P\|_\infty
\le
2(1-\min_x\widehat M(x,x))+\gamma.
\]

论文的主定理把收缩项与 \(V^\pi\) 处的固定偏差组合成几何衰减加统计地板。

本项目新增：

- 将同一代数迁移到 state-action pair MRP 的 Direct-Q；
- 直接从 count lower bound 推出 one-hot softmax 的经验对角下界；
- 用一个共享事件控制 state、pair 和三族 edge residual；
- 给出同数据 V-first ghost-target 分解；
- 将全部条件实现为 route-level 可计算证书。

这些是本项目推论，不应表述为上述论文已经给出的结果。

## 3. 设定

令

\[
X_t=(S_t,A_t),
\qquad
E_t=(X_t,X_{t+1}).
\]

策略固定为 \(\pi\)，三条链从各自平稳分布起步。state 数为 \(m\)，目标 pair 支持 \(\mathcal X_\pi\) 的大小为 \(d\)。正式实验中策略在全部 state-action pair 上有正 occupancy。

reward 是 edge 的确定函数，并满足

\[
|R_{t+1}|\le R_\star,
\qquad
B=\frac{R_\star}{1-\gamma}.
\]

于是

\[
\|V^\pi\|_\infty\le B,
\qquad
\|Q^\pi\|_\infty\le B.
\]

同一条长度为 \(n\) 的冻结轨迹被所有层重复使用。每层是同步 frozen-value 或 frozen-Q 更新；score 只依赖固定 query/source pair，不依赖 reward、next pair、当前 iterate 或层数。

## 4. 共享高概率事件

同时控制以下预先固定的函数：

1. \(m\) 个 state-count indicators；
2. \(d\) 个 pair-count indicators；
3. \(m\) 个 \(V^\pi\) state Bellman centered residual；
4. \(d\) 个 \(Q^\pi\) pair Bellman centered residual；
5. \(d\) 个固定-\(V^\pi\) recovery centered residual。

总函数数为

\[
M=2m+3d,
\qquad
L_\delta=\log\frac{2M}{\delta}.
\]

记三条链的 right-Hoeffding inflation 为 \(\omega_S,\omega_X,\omega_E\)。定义

\[
b_S=\sqrt{\frac{\omega_S L_\delta}{2n}},
\qquad
b_X=\sqrt{\frac{\omega_X L_\delta}{2n}},
\]

\[
e_E=B\sqrt{\frac{2\omega_E L_\delta}{n}}.
\]

令

\[
u_S=\mu_{S,\min}-b_S,
\qquad
u_X=\mu_{X,\min}-b_X.
\]

在概率至少 \(1-\delta\) 的同一事件上，若相应 lower bound 为正，则

\[
\varepsilon_S=\frac{e_E}{u_S},
\qquad
\varepsilon_X=\frac{e_E}{u_X}.
\]

residual 是以真实 Bellman 条件均值为中心的固定 edge function，因此不需要旧 raw numerator/denominator 分解中的额外 \(B b_S\) 或 \(B b_X\) 项。各函数和各路线不要求独立；一次 union bound 已足够。

若数值谱实现丢弃任何严格正概率 edge，则记录 **numerical_support_truncated** 并拒绝高概率证书。正式运行使用精确正支持，480 项截断率为 0。

## 5. One-hot softmax 对角桥接

对 \(u\in(0,1]\)，定义

\[
m_\beta(u)
=
\frac{e^\beta u}
{e^\beta u+1-u}.
\]

经验 matching mass 满足

\[
\widehat M_\beta(x,x)
=
m_\beta(N_x/n).
\]

由于 \(m_\beta\) 单调，在 count event 上

\[
\min_x\widehat M_\beta(x,x)
\ge
m_\beta(u_X)
\]

或 state 版本的 \(m_\beta(u_S)\)。实现使用稳定 logit 公式，不直接计算可能溢出的 \(e^\beta\)。若 \(u\le0\)，diagonal、margin、\(\rho\) 和 total bound 都写为 JSON null。

exact matching 在 full coverage 后有 \(\widehat M=I\)，完全不依赖 \(\beta\)。

## 6. Direct-Q 所有层定理

对每个 pair \(z\)，定义

\[
\bar\delta_z^\pi
=
\frac1{N_z}
\sum_{t:X_t=z}
\left[
R_{t+1}
+\gamma Q^\pi(X_{t+1})
-Q^\pi(z)
\right].
\]

full coverage 下有精确恒等式

\[
[\widehat{\mathcal F}_{Q,1}(Q^\pi)-Q^\pi](x)
=
\sum_z\widehat M(x,z)\bar\delta_z^\pi.
\]

因为 \(\widehat M\) 行随机，

\[
\|\widehat{\mathcal F}_{Q,1}(Q^\pi)-Q^\pi\|_\infty
\le
\max_z|\bar\delta_z^\pi|
\le
\varepsilon_X.
\]

### 6.1 Finite softmax

令

\[
c_Q
=
m_\beta(u_X)-\frac{1+\gamma}{2}.
\]

若 \(u_X>0,c_Q>0\)，则

\[
\rho_Q=1-2\alpha c_Q\in[0,1),
\]

且在同一事件上对所有 \(L\ge0\) 同时成立

\[
\boxed{
\|Q_L-Q^\pi\|_\infty
\le
\rho_Q^L\|Q_0-Q^\pi\|_\infty
+
\frac{1-\rho_Q^L}{2c_Q}\varepsilon_X
}.
\]

概率事件只控制固定 residual 和固定经验算子的收缩条件。随后对 \(L\) 的递推完全是路径确定性的，所以无需 layerwise union bound；实际 early stopping 的数据依赖层数也被覆盖。

### 6.2 Exact matching

coverage 后

\[
c_Q=\frac{1-\gamma}{2},
\qquad
\rho_Q=1-\alpha(1-\gamma),
\]

从而

\[
\boxed{
\|Q_L-Q^\pi\|_\infty
\le
\rho_Q^L\|Q_0-Q^\pi\|_\infty
+
\frac{1-\rho_Q^L}{1-\gamma}\varepsilon_X
}.
\]

如果 divergence guard 触发，代码不再等于该仿射算子，证书必须拒绝。

## 7. State-value 全层界

将 pair 量替换为 state 量。finite-softmax 情形定义

\[
c_V=m_\beta(u_S)-\frac{1+\gamma}{2},
\qquad
\rho_V=1-2\alpha c_V.
\]

若 \(u_S>0,c_V>0\)，则

\[
\|V_L-V^\pi\|_\infty
\le
\rho_V^L\|V_0-V^\pi\|_\infty
+
\frac{1-\rho_V^L}{2c_V}\varepsilon_S.
\]

exact matching 取 \(c_V=(1-\gamma)/2\) 和 \(\rho_V=1-\alpha(1-\gamma)\)。每层 clip 到 \([-B,B]\) 不增加相对 \(V^\pi\) 的误差，因此不破坏递推。

## 8. V-first no-split 定理

### 8.1 Exact recovery

定义

\[
\widehat{\mathcal R}_{\mathrm{ex}}(W)(x)
=
\frac1{N_x}
\sum_{t:X_t=x}
\left[
R_{t+1}+\gamma W(S_{t+1})
\right].
\]

逐轨迹确定性地有

\[
\|\widehat{\mathcal R}_{\mathrm{ex}}(W)
-\widehat{\mathcal R}_{\mathrm{ex}}(W')\|_\infty
\le
\gamma\|W-W'\|_\infty.
\]

固定-\(V^\pi\) ghost residual 给出

\[
\|\widehat{\mathcal R}_{\mathrm{ex}}(V^\pi)-Q^\pi\|_\infty
\le
\varepsilon_X.
\]

若 state-value total bound 记为 \(\varepsilon_{V,L}\)，同一轨迹上的 estimator 满足

\[
\boxed{
\|\widehat Q_{\mathrm{ns},L}-Q^\pi\|_\infty
\le
\gamma\varepsilon_{V,L}+\varepsilon_X
}.
\]

该结论不使用样本切分、条件独立或 fold gap。

### 8.2 Finite-softmax recovery

softmax recovery 的权重非负、按 query 归一化且不依赖 \(W\)，所以 plug-in Lipschitz 常数仍为 \(\gamma\)，不乘 kernel diagonal。

令 \(\widehat d_{\beta,x}=\widehat M_\beta(x,x)\)。对 full coverage 轨迹，

\[
\|\widehat{\mathcal R}_\beta(V^\pi)
-\widehat{\mathcal R}_{\mathrm{ex}}(V^\pi)\|_\infty
\le
2B\max_x(1-\widehat d_{\beta,x}).
\]

共享 count event 上可用

\[
L_\beta
=
2B[1-m_\beta(u_X)].
\]

因此

\[
\boxed{
\|\widehat Q_{\mathrm{ns},\beta,L}-Q^\pi\|_\infty
\le
\gamma\varepsilon_{V,L}
+\varepsilon_X
+L_\beta
}.
\]

一次 recovery 不要求 pair kernel 收缩，所以 **pair_kernel_margin_nonpositive** 只阻止 Direct-Q softmax，不阻止 V-first softmax recovery 获得有限 leakage 界。固定有限 \(\beta\) 下 \(L_\beta\) 通常不随 \(n\) 消失，不能把 finite bound 写成 consistency。

## 9. 证据等级

实现固定使用四个互斥等级：

- **high_probability_certified**：先验共享事件的 coverage、谱和路线 margin 条件满足；
- **pathwise_bound_verified**：先验 sufficient condition 未通过，但该轨迹上的 full support、fixed ghost residual 和经验对角给出确定性验证；
- **diagnostic_only**：只有不构成上述两类证书的诊断；
- **not_certified**：support、margin、谱或算法契约失败。

主要失败码：

- **state_coverage_failed**；
- **pair_coverage_failed**；
- **state_kernel_margin_nonpositive**；
- **pair_kernel_margin_nonpositive**；
- **spectral_condition_failed**；
- **numerical_support_truncated**；
- **algorithm_mode_mismatch**；
- **divergence_guard_triggered**。

不可用数值和缺失 group mean 都写 JSON null，不用 zero initialization 冒充统计估计。

## 10. 实现与契约

核心文件：

- **fixed_policy_finite_sample_certificate.py**：纯数学证书；
- **markov_coverage_certificate.py**：三条链的 stationary、right-gap 与 support metadata；
- **evaluate_fixed_policy_q_routes.py**：路线估计、共享事件和路径诊断；
- **analyze_fixed_policy_finite_sample_certificates.py**：旧路线回归与正式统计；
- **verify_finite_sample_theorems.py**：公式级契约。

契约覆盖共享 \(\delta\) 分配、Direct-Q residual identity、exact/softmax 全层递推、clipping、same-sample ghost 分解、softmax leakage、主要拒绝边界、exact-\(\beta\) 解耦、实际 stopping depth 和 strict JSON。

## 11. 480 项正式结果

新结果位于 **results/fixed_policy_finite_sample_certificates/**。与旧 formal scan 对齐后：

- 旧 10 条路线共同字段 mismatch 为 0；
- 最大共同数值绝对差为 \(8.88\times10^{-16}\)；
- Direct-Q exact 总体平均 \(Q\) sup error 为 0.5852；
- V-first no-split exact 为 0.5671；
- 新增 no-split softmax 为 0.5741；
- no-gap cross-fit exact 为 0.5873；
- gap cross-fit exact 为 0.6494。

三条链谱条件通过率在所有长度均为 100%，数值 support 截断率为 0%。保守 pair occupancy lower bound 在四个长度上的通过率均为 0%，因此路线的高概率证书率均为 0。这是 sufficient coverage bound 的保守性，不是经验 full coverage 失败。

观测路径界验证率：

| \(n\) | Direct exact | Direct softmax | No-split exact | No-split softmax |
|---:|---:|---:|---:|---:|
| 256 | 2.5% | 2.5% | 2.5% | 2.5% |
| 1024 | 85.0% | 61.7% | 85.0% | 85.0% |
| 4096 | 100% | 95.8% | 100% | 100% |
| 16384 | 100% | 100% | 100% | 100% |

Direct softmax 在 \(n=1024,4096\) 的额外拒绝来自观测 pair kernel margin 非正；V-first softmax recovery 不受该 pair contraction gate 限制。

no-split softmax 减 exact 的平均误差：

| \(n\) | softmax - exact | softmax 更优率 |
|---:|---:|---:|
| 256 | -0.0194 | 37.5% |
| 1024 | -0.0045 | 51.7% |
| 4096 | +0.0190 | 37.5% |
| 16384 | +0.0327 | 19.2% |

有限 softmax 在短轨迹偶有平滑收益，但随样本增长留下可见 leakage；没有形成统一优势。

## 12. 仍然开放的问题

1. 当前 Hoeffding coverage sufficient condition 的 \(\mu_{\min}^{-2}\) 型样本需求很保守；可研究 visit-indexed martingale、multiplicative Markov lower tail 或 self-normalized ratio。
2. visit-indexed residual rate 若要升级为定理，需要完整处理 stopping-time filtration、随机访问次数和 simultaneous-in-count 常数。
3. 额外随机 reward noise 需要 augmented chain 或 martingale 扩展。
4. 非平稳起步需要 burn-in 或初始分布 prefactor。
5. fixed-policy 结果不会自动推出 blockwise 或 online control；策略改变会同时改变 occupancy、mixing、kernel diagonality 与 action gap。
