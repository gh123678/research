# Direct-Q 与 V-first 共享有限样本定理设计

> 日期：2026-08-31  
> 状态：五节口头设计已获用户确认，等待书面规格复核  
> 上游证据：`docs/research_branches/direct_q_vs_v_first_report.md`  
> 活动目标：`ACTIVE_WORKSPACE.md`

## 1. 目标

本阶段在同一条冻结的平稳 fixed-policy 轨迹上闭合两条有限样本结果：

1. **Direct-Q**：用经验收缩与 \(Q^\pi\) 处的固定 Bellman residual，得到对所有迭代层同时成立的高概率界；
2. **V-first no-split**：用固定-\(V^\pi\) ghost recovery target 与经验 recovery 的确定性 \(\gamma\)-Lipschitz 性，证明同一轨迹同时估计 \(V\) 和恢复 \(Q\) 不需要独立性。

两条路线共享 state、pair 和 edge-chain 的 count/residual 事件。交付物必须包括完整定理、可计算证书、契约测试、相同 seeds 的复现实验和更新后的理论报告。

本阶段不进入 fully online control，不建立跨策略块联合定理，不声称未知模型下可直接计算所有谱常数，也不把随机奖励或非平稳初态混入主定理。

## 2. 当前结论与需要修正的旧缺口

现有实现已完成 fixed-policy、cross-fit/Markov certificate 和 blockwise control 实验。正式 fixed-policy 扫描包含 480 个匹配配置；无 gap cross-fit 在配对区间意义上追平 Direct-Q，no-split V-first 的平均误差最低。

本轮数学审查确认两处旧缺口可以关闭。

### 2.1 Direct-Q 不需要逐层浓缩

冻结轨迹后，Direct-Q 使用固定 counts、transition counts、reward sums、score 和 softmax denominator。它是固定经验仿射算子的重复应用。只要在一个事件上同时控制：

- 经验算子的收缩性；
- 经验算子在真实 \(Q^\pi\) 处的偏差；

确定性递推即可覆盖所有 \(L\ge 0\)，包括数据依赖的停止层数。无需对 \(Q_l\) 逐层应用浓缩，也无需对层数做 union bound。

### 2.2 V-first no-split 不需要阶段独立

对经验 recovery 算子 \(\widehat{\mathcal R}\)，逐轨迹有

\[
\|\widehat{\mathcal R}(W)
-\widehat{\mathcal R}(W')\|_\infty
\le
\gamma\|W-W'\|_\infty.
\]

因此

\[
\|\widehat{\mathcal R}(\widehat V)-Q^\pi\|_\infty
\le
\|\widehat{\mathcal R}(V^\pi)-Q^\pi\|_\infty
+\gamma\|\widehat V-V^\pi\|_\infty.
\]

第一项只包含预先固定的 \(V^\pi\)，第二项完全确定。两个概率事件即使由同一轨迹产生，也只需 union bound；union bound 不要求独立。cross-fit 仍保留为算法对照和 ghost-target 推论，但不再是 no-split 定理的必要条件。

## 3. 主定理设定

令 \(\mathcal S\) 和 \(\mathcal A\) 有限，

\[
m=|\mathcal S|,
\qquad
d=|\mathcal S||\mathcal A|.
\]

主定理默认策略在全部 \(\mathcal S\times\mathcal A\) 上具有正 occupancy，因此使用上述 \(d\)。若只在目标支持集 \(\mathcal X_\pi\) 上陈述 corollary，则所有 pair 维度、union bound 和最小 occupancy 均一致替换为 \(|\mathcal X_\pi|\) 及该支持集上的量，不能继续沿用包含零质量 pair 的 \(d\) 和 \(\mu_{X,\min}\)。

固定策略 \(\pi\) 诱导 pair chain

\[
X_t=(S_t,A_t),
\qquad
P_X^\pi((s,a),(s',a'))
=P(s'\mid s,a)\pi(a'\mid s').
\]

主定理采用以下假设。

1. \(0<\gamma<1\)，\(0<\alpha\le1\)。
2. \(P_X^\pi\) 在目标支持集上 irreducible 且 aperiodic。
3. 若结论覆盖全部 \(\mathcal S\times\mathcal A\)，则策略在全部目标 pair 上具有正 stationary occupancy；否则定理只在 \(\mathcal X_\pi\) 上陈述。
4. \(S_0\) 从 state stationary distribution 采样，\(A_0\sim\pi(\cdot\mid S_0)\)。因此 state、pair 与 edge chain 均从各自平稳分布开始。
5. 奖励是有界、确定的 transition reward
   \[
   R_{t+1}=R(S_t,A_t,S_{t+1}),
   \qquad |R|\le R_{\max}.
   \]
6. 轨迹、one-hot score、softmax sharpness、归一化分母与 update rule 在所有层固定。
7. 更新是同步 frozen-value/frozen-Q 更新；attention score 不依赖当前迭代值、奖励、next pair 或层数。

令

\[
B=\frac{R_{\max}}{1-\gamma}.
\]

则

\[
\|V^\pi\|_\infty,
\|Q^\pi\|_\infty
\le B,
\]

且所有固定 Bellman/recovery target 位于 \([-B,B]\)。

## 4. 共享高概率事件

### 4.1 三条链的职责

- **state chain** \(S_t\)：控制 state counts；
- **pair chain** \(X_t=(S_t,A_t)\)：控制 pair counts；
- **edge chain** \(E_t=(X_t,X_{t+1})\)：控制 transition reward、TD numerator 和 recovery numerator。

实验奖励是 \(E_t\) 的确定函数。edge chain 的 right spectral gap 可用于预先固定的 time-independent 函数；其 absolute gap 可以为零，因此本阶段不使用需要 edge absolute gap 的 time-dependent Hoeffding 路线。

### 4.2 被同时控制的固定函数

共享事件包含：

1. \(m\) 个 state-count indicators；
2. \(d\) 个 pair-count indicators；
3. \(m\) 个 \(V^\pi\) state Bellman centered residual；
4. \(d\) 个 \(Q^\pi\) pair Bellman centered residual；
5. \(d\) 个固定-\(V^\pi\) recovery centered residual。

总函数数为

\[
M=2m+3d.
\]

这些 residual 均以真实 \(V^\pi,Q^\pi\) 为中心，不包含数据生成的迭代值。其值域长度为 \(2B\)：虽然 residual 的绝对值可达 \(2B\)，固定条件均值位于 target 区间内，因此 residual 与零的联合取值区间仍只有长度 \(2B\)。

### 4.3 显式半径

令 state、pair、edge chain 的 stationary right-Hoeffding inflation 分别为

\[
\omega_S,\qquad \omega_X,\qquad \omega_E.
\]

对任一链 \(C\in\{S,X,E\}\)，若 additive reversiblization 的 right spectral value 为 \(\lambda_{r,C}<1\)，则规格中的 inflation 定义为

\[
\omega_C
=
\frac{1+\max\{\lambda_{r,C},0\}}
{1-\max\{\lambda_{r,C},0\}}.
\]

谱计算必须使用理论链的精确正支持。若数值实现按阈值丢弃了正概率 edge，必须记录 `numerical_support_truncated` 并拒绝高概率证书，除非先证明该截断没有改变目标链。

统一使用

\[
L_\delta=\log\frac{2M}{\delta}.
\]

定义

\[
b_S=
\sqrt{\frac{\omega_S L_\delta}{2n}},
\qquad
b_X=
\sqrt{\frac{\omega_X L_\delta}{2n}},
\]

\[
e_E=
B\sqrt{\frac{2\omega_E L_\delta}{n}}.
\]

一次 union bound 给出概率至少 \(1-\delta\) 的共享事件。该结论不要求各函数或各路线事件独立。

定义 coverage 下界

\[
u_S=\mu_{S,\min}-b_S,
\qquad
u_X=\mu_{X,\min}-b_X.
\]

若 \(u_S>0,u_X>0\)，则对应 state/pair 全部被访问，并且固定 centered conditional residual 分别满足

\[
\varepsilon_S=\frac{e_E}{u_S},
\qquad
\varepsilon_X=\frac{e_E}{u_X}.
\]

该 centered-residual 形式不需要旧 raw numerator/denominator 界中的额外 \(B b_S\) 或 \(B b_X\) 项。

## 5. One-hot softmax 对角下界

对经验频率 \(u\in(0,1]\)，定义

\[
m_\beta(u)
=
\frac{e^\beta u}
{e^\beta u+1-u}.
\]

实现必须先检查 \(u_S>0\) 或 \(u_X>0\)，再计算对应的 \(m_\beta(u)\)。coverage lower bound 非正时，margin 与依赖它的定理字段写 `null`，不得把定义域外数值继续传递。

one-hot softmax 的经验对角质量为

\[
\widehat M_\beta(x,x)
=m_\beta(N_x/n).
\]

由于 \(m_\beta\) 单调递增，在 count event 上有

\[
\min_x\widehat M_\beta(x,x)
\ge m_\beta(u_X)
\]

或 state 版本的 \(m_\beta(u_S)\)。

需要严格区分：

- `coverage_certified` 只表示 count lower bound 为正；
- `kernel_margin_certified` 还要求 empirical diagonal 的概率下界超过收缩阈值；
- 观测到的 empirical diagonal 是路径式诊断，不能代替先验高概率桥接。

exact matching 在 coverage 后有 \(\widehat M=I\)，不依赖 \(\beta\) certificate。

## 6. Direct-Q 全层定理设计

令 \(\widehat{\mathcal F}_{Q,\alpha}\) 为冻结轨迹产生的 Direct-Q 经验算子。对每个 pair \(z\)，定义真实 \(Q^\pi\) 处的条件经验 residual：

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

在 full-coverage 事件上，

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
\le \varepsilon_X.
\]

### 6.1 Finite-softmax

定义

\[
c_Q
=m_\beta(u_X)-\frac{1+\gamma}{2}.
\]

若

\[
u_X>0,
\qquad c_Q>0,
\]

则经验算子的收缩因子满足

\[
\rho_Q=1-2\alpha c_Q\in[0,1).
\]

在同一共享事件上，对所有 \(L\ge0\) 同时有

\[
\boxed{
\|Q_L-Q^\pi\|_\infty
\le
\rho_Q^L\|Q_0-Q^\pi\|_\infty
+
\frac{1-\rho_Q^L}{2c_Q}\varepsilon_X
}.
\]

该界也适用于数据依赖的实际停止层数，因为事件上路径式递推对全部 \(L\) 同时成立。

### 6.2 Exact matching

coverage 后 \(\widehat M=I\)，因此

\[
c_Q=\frac{1-\gamma}{2},
\qquad
\rho_Q=1-\alpha(1-\gamma).
\]

于是

\[
\boxed{
\|Q_L-Q^\pi\|_\infty
\le
\rho_Q^L\|Q_0-Q^\pi\|_\infty
+
\frac{1-\rho_Q^L}{1-\gamma}\varepsilon_X
}.
\]

## 7. State-value 全层定理设计

state-value 经验算子使用相同证明，但把 pair counts/margin 替换为 state counts/margin。定义

\[
c_V
=m_\beta(u_S)-\frac{1+\gamma}{2},
\qquad
\rho_V=1-2\alpha c_V.
\]

若 \(u_S>0,c_V>0\)，则 finite-softmax state evaluator 满足

\[
\|V_L-V^\pi\|_\infty
\le
\rho_V^L\|V_0-V^\pi\|_\infty
+
\frac{1-\rho_V^L}{2c_V}\varepsilon_S.
\]

exact matching 时取

\[
c_V=\frac{1-\gamma}{2},
\qquad
\rho_V=1-\alpha(1-\gamma).
\]

实现每层把 \(V\) clip 到 \([-B,B]\)。由于 \(V^\pi\) 位于该区间，投影相对 \(V^\pi\) 非扩张；上述递推保持成立。证书使用实际 `iterations_used`。若 Direct-Q 触发 divergence guard，则该样本点不再满足仿射算子定理，必须拒绝证书。

## 8. V-first no-split 定理设计

### 8.1 Exact recovery

对 \(N_x>0\)，定义

\[
\widehat{\mathcal R}_{\mathrm{ex}}(W)(x)
=
\frac1{N_x}
\sum_{t:X_t=x}
\left[
R_{t+1}+\gamma W(S_{t+1})
\right].
\]

逐轨迹有

\[
\|\widehat{\mathcal R}_{\mathrm{ex}}(W)
-\widehat{\mathcal R}_{\mathrm{ex}}(W')\|_\infty
\le
\gamma\|W-W'\|_\infty.
\]

固定-\(V^\pi\) residual 族给出

\[
\|\widehat{\mathcal R}_{\mathrm{ex}}(V^\pi)-Q^\pi\|_\infty
\le \varepsilon_X.
\]

令第 7 节给出的 value bound 为 \(\varepsilon_{V,L}\)。同一轨迹的 no-split estimator 满足

\[
\boxed{
\|\widehat Q_{\mathrm{ns},L}-Q^\pi\|_\infty
\le
\gamma\varepsilon_{V,L}
+\varepsilon_X
}.
\]

该结论不使用样本切分、条件独立或 fold gap。

### 8.2 Finite-softmax recovery

finite-softmax recovery 的权重非负、逐 query 归一化且不依赖 \(W\)，因此 plug-in Lipschitz 常数仍恰为 \(\gamma\)，不乘 kernel diagonal mass。

对 full-coverage 轨迹，令 \(\widehat d_{\beta,x}=\widehat M_\beta(x,x)\)。matching 与 nonmatching target 的凸组合给出

\[
\|\widehat{\mathcal R}_{\beta}(V^\pi)
-\widehat{\mathcal R}_{\mathrm{ex}}(V^\pi)\|_\infty
\le
2B\max_x(1-\widehat d_{\beta,x}).
\]

在 count event 上可进一步使用先验界

\[
L_\beta
=2B[1-m_\beta(u_X)].
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

固定有限 \(\beta\) 时 \(L_\beta\) 通常不会随 \(n\) 消失。证书必须区分“有限误差界成立”和“一致估计成立”；不得把 finite-\(\beta\) leakage 隐藏进统计噪声。

### 8.3 Cross-fit 推论

对两折分别构造固定-\(V^\pi\) ghost targets，并对各折的 value 与 recovery 事件分配失败概率，可得到相同结构的 cross-fit 上界。两折事件只需 union bound，不要求折间独立。gap/coupling 仅在坚持“给定训练折后再浓缩评估折”的替代证明路径中必要。

本阶段只把该结论写成理论推论，不扩张 cross-fit 算法实现。

## 9. 缺失 pair 与证据等级

现有 exact/softmax recovery 对未访问 query 输出零。若 \(N_x=0\)，uniform consistency 不成立，最坏仍可能有

\[
|Q^\pi(x)|\le B.
\]

因此输出必须区分：

- **full-support theorem**：所有目标 state/pair coverage 与 margin 条件满足；
- **visited-support bound**：只在闭合的已访问支持集上陈述；
- **not certified**：缺失 pair、谱条件失败、margin 非正或算法模式不匹配。

不得把 zero initialization 当作缺失 pair 的统计估计。

## 10. Sharper occupancy-adaptive 扩展

核心 Hoeffding 定理给出的 conditional residual rate 为

\[
O\!\left(
\frac{B}{\mu_{\min}}
\sqrt{\frac{\omega_E\log(M/\delta)}{n}}
\right),
\]

其 coverage sufficient condition 也可能非常保守。

第二层目标是 visit-indexed martingale。对固定 group \(g\)，令 \(\tau_{g,k}\) 为第 \(k\) 次访问，定义

\[
Z_{g,k}=Y_{\tau_{g,k}}-\mathbb E[Y_{\tau_{g,k}}\mid g].
\]

在正确的 pre-transition filtration 下，三类固定 target 都可形成 bounded martingale-difference sequence。若 stopping-time 与随机访问次数处理完整，则可获得

\[
|\widehat m_g-m_g|
\lesssim
B\sqrt{\frac{\log(Gn/\delta)}{N_g}}.
\]

该方向不需要 edge absolute gap，并直接产生 occupancy-adaptive rate。但只有在 filtration、访问 stopping time、随机 \(N_g\) 与 simultaneous-in-\(k\) 常数全部写严谨后，才可标为已证明。否则保留为 proof target。

即使该 residual rate 闭合，全 pair 的先验 coverage 仍需要单独的 multiplicative Markov lower-tail，或把结论写成 observed-count implication certificate。

## 11. 代码结构

新增 `fixed_policy_finite_sample_certificate.py`。该模块只负责纯数学证书，不采样、不画图。主要职责：

1. `shared_event_radii`：统一分配 \(\delta\)，计算 \(b_S,b_X,e_E,u_S,u_X\)；
2. `direct_q_uniform_bound`：生成 exact/softmax 的 Direct-Q 全层界；
3. `state_value_uniform_bound`：生成 exact/softmax 的 state-value 全层界；
4. `vfirst_nosplit_bound`：组合 ghost recovery、plug-in 与 leakage；
5. `observed_ghost_residuals`：仅在合成实验中利用真实 \(V^\pi,Q^\pi\) 做路径核验。

`markov_coverage_certificate.py` 继续负责：

- state/pair/edge stationary distributions；
- right spectral gaps 与 Hoeffding inflation；
- forward/reverse mixing 与 gap 诊断；
- 原有 primitive coverage/kernel diagnostics。

它不直接实现路线级定理，避免混合链诊断与算法结论。

`evaluate_fixed_policy_q_routes.py` 将：

- 调用新的 theorem certificate；
- 新增 `vfirst_nosplit_softmax`；
- 保存路线级理论半径、observed ghost residual、margin、证书状态与失败原因；
- 使用新结果目录，不覆盖现有结果。

## 12. 输出 schema

每个任务至少保存：

- `assumptions`：stationary start、fixed context、edge-determined reward、synchronous update、full target support；
- `shared_event`：\(\delta,M,B,\omega_S,\omega_X,\omega_E,b_S,b_X,e_E,u_S,u_X\)；
- `direct_q`：route、margin、\(\rho_Q\)、iterations、optimization term、statistical term、total bound；
- `state_value`：对应的 margin、\(\rho_V\) 与 value bound；
- `vfirst_nosplit`：value propagation、fixed recovery、softmax leakage 与 total bound；
- `observed_diagnostics`：ghost residual、实际 error、pathwise slack；
- `status`：证书等级与失败原因。

证书等级固定为：

- `high_probability_certified`；
- `pathwise_bound_verified`；
- `diagnostic_only`；
- `not_certified`。

失败原因使用稳定枚举，例如：

- `state_coverage_failed`；
- `pair_coverage_failed`；
- `state_kernel_margin_nonpositive`；
- `pair_kernel_margin_nonpositive`；
- `spectral_condition_failed`；
- `numerical_support_truncated`；
- `algorithm_mode_mismatch`；
- `divergence_guard_triggered`。

不可用数值写 `null`，不得写 `NaN` 或无穷值。exact route 不依赖 \(\beta\) certificate。

## 13. 契约测试

新增 `verify_finite_sample_theorems.py`，至少覆盖：

1. Direct-Q 在 \(Q^\pi\) 处的 empirical residual identity；
2. exact 与 finite-softmax 的路径式递推对多个 \(L\) 成立；
3. one-hot empirical diagonal 与 count 下界一致；
4. state clipping 相对 \(V^\pi\) 非扩张；
5. 人为制造完全同数据依赖，no-split ghost-target 分解仍成立；
6. finite-softmax recovery leakage 界；
7. 缺失 state/pair 时拒绝 full-space certificate；
8. margin 非正、谱条件失败、算法模式不匹配时给出正确 reason code；
9. exact route 与 \(\beta\) 完全解耦；
10. early stopping 使用实际 `iterations_used`；
11. 严格 JSON 无非有限数值；
12. 一个均匀、快速混合的小型 fixture 通过完整高概率证书；
13. 现有 fixed-policy、cross-fit、blockwise 与 end-to-end 相关回归不退化。

## 14. 实验协议

新结果写入：

`results/fixed_policy_finite_sample_certificates/`

不覆盖现有：

- `results/fixed_policy_q_routes/`；
- `results/fixed_policy_q_routes_crossfit/`；
- `results/blockwise_q_routes/`。

正式协议复用现有 seeds、MDP、策略、trajectory lengths、mixing、reward gap、\(\beta\) 与迭代数。它不是新的超参数扫描，只是为同一批配置增加：

- `vfirst_nosplit_softmax`；
- 新的共享事件和路线证书；
- observed ghost residual 与理论 slack。

旧路线的数值必须与现有正式结果一致。若保守 Hoeffding coverage 导致正式 480 配置的高概率证书通过率仍为零，该结果应如实报告为“定理闭合但当前充分条件保守”，不得通过修改阈值或扩大扫描制造通过率。

正式汇总至少报告：

- empirical error；
- high-probability total bound；
- pathwise ghost-target bound；
- coverage margin；
- kernel margin；
- certificate pass/fail rate；
- 各 failure reason 的频率；
- bound/error 比率的分布。

## 15. 文档更新

新增：

- `docs/research_branches/shared_fixed_policy_finite_sample_theory.md`。

修改：

- `docs/research_branches/branch_a_direct_q_theory.md`；
- `docs/research_branches/branch_b_v_first_theory.md`；
- `docs/research_branches/crossfit_markov_certificate_theory.md`；
- `docs/research_branches/direct_q_vs_v_first_report.md`；
- `ACTIVE_WORKSPACE.md`。

必须删除或修正以下旧表述：

- Direct-Q 因 \(Q_l\) 数据依赖而必须逐层浓缩；
- no-split 因复用同一数据而原则上无法获得高概率证明；
- cross-fit 的 gap 是 V-first fixed-policy sup-norm 定理所必需。

新的剩余缺口应限定为：

- Hoeffding coverage 的保守性；
- visit-indexed martingale 的严谨 anytime/stopping-time 常数；
- multiplicative Markov coverage lower-tail；
- 非平稳初态与真正随机奖励扩展；
- fixed-policy 到 blockwise/online control 的联合控制。

本阶段不修改论文主文。只有定理、证书代码、契约测试和复现实验完全一致后，才决定是否升级正文主张。

## 16. 验收门槛

### 理论

- 共享事件的函数数、值域、\(\delta\) 分账和常数全部显式；
- Direct-Q 定理对所有层同时成立；
- state-value 定理正确处理 clipping；
- V-first no-split exact/softmax 明确无需独立性；
- full-support、visited-support 和 missing-pair 情形严格分开；
- 未覆盖的奖励与初态模型明确排除。

### 代码

- 新 theorem certificate 契约全部通过；
- 现有验证与回归不退化；
- 一个受控 fixture 真正通过完整高概率证书；
- strict JSON 无非有限值；
- 所有失败均有稳定 reason code。

### 实验

- 使用原参数矩阵与 seeds；
- 旧路线与已有结果一致；
- 新增 no-split softmax 与路线证书；
- 所有 pathwise decomposition 均有非负 slack；
- 不以 certificate pass rate 作为调参目标。

### 文档

- 四类证书等级统一；
- 已关闭的旧缺口被删除；
- sharper rate 未完成时保持为 proof target；
- 不把 fixed-policy 结论外推到 control。

## 17. 停止条件

1. 核心 Hoeffding 定理、证书与复现结果一致即视为本阶段成功；sharper martingale rate 不是硬门槛。
2. sharper rate 只有在 filtration、stopping time、随机访问次数和 simultaneous bound 全部严谨时才升级为已证明。
3. 不为提高保守 certificate 的通过率扩大实验或调整理论阈值。
4. 不进入 fully online control，不重新打开已归档的八月早期实验分支。
5. 若实现与固定仿射算子假设不一致，该路线输出 `algorithm_mode_mismatch`，不得强行套用定理。

## 18. 交付文件

新增：

- `fixed_policy_finite_sample_certificate.py`；
- `verify_finite_sample_theorems.py`；
- `docs/research_branches/shared_fixed_policy_finite_sample_theory.md`；
- `results/fixed_policy_finite_sample_certificates/`。

修改：

- `evaluate_fixed_policy_q_routes.py`；
- `markov_coverage_certificate.py`（仅在需要暴露现有链诊断接口时做最小修改）；
- 四份现有 research-branch 文档；
- `ACTIVE_WORKSPACE.md`。

保留不动：

- 现有三组正式结果目录；
- 论文主文；
- fully online control 实现；
- `C:\Users\Admin\Desktop\research\_archive\icrl_softmax_history_20260831` 中的历史归档。
