# Direct-Q / V-first 来源与假设对照表

> 日期：2026-08-29  
> 目的：区分原始来源中的已证明结论、本项目可直接复用的结论，以及本研究需要重新证明的部分。

## 1. 已核对的原始来源

### S1. Xie et al. (2026)

- Zixuan Xie, Xinyu Liu, Claire Chen, Shuze Daniel Liu, Rohan Chandra, Shangtong Zhang.
- *Beyond Linear Attention: Softmax Transformers Implement In-Context Reinforcement Learning*.
- arXiv:2605.07333v2, 2026-05-17.
- 网页：https://arxiv.org/abs/2605.07333
- 本地 PDF：`papers/Xie_2026_beyond_linear_attention.pdf`

已核对位置：正文 Sections 2.2、3、5、9；Theorem 2；Appendices A、D.1-D.4。

### S2. Fan, Jiang, and Sun (2021)

- Jianqing Fan, Bai Jiang, Qiang Sun.
- *Hoeffding's Inequality for General Markov Chains and Its Applications to Statistical Learning*.
- JMLR 22(139):1-35, 2021.
- 网页：https://www.jmlr.org/papers/v22/19-479.html

Xie Appendix A 的 Lemma 2 是该文 Theorem 12 在零 burn-in、\(p=\infty\) 下的特例。它为一般初始分布下的有界 Markov-chain additive functional 提供 Hoeffding 型集中界，常数依赖平稳最小质量与 additive reversiblization 的谱参数。

### S3. Pananjady and Wainwright (2020)

- Ashwin Pananjady, Martin J. Wainwright.
- *Instance-dependent \(\ell_\infty\)-bounds for policy evaluation in tabular reinforcement learning*.
- arXiv:1909.08749v2; IEEE Transactions on Information Theory manuscript.
- 网页：https://arxiv.org/abs/1909.08749

该文分析有限 MRP 的 plug-in value estimator，并给出 instance-dependent \(\ell_\infty\) 非渐近界。其数据模型是 **synchronous / generative model**，不是本项目的单条 Markov 轨迹。它可作为 oracle-model 消融和 plug-in 分析参考，不能直接提供主实验的 single-trajectory rate。

### S4. Winnicki and Srikant (2023)

- Anna Winnicki, R. Srikant.
- *On The Convergence Of Policy Iteration-Based Reinforcement Learning With Monte Carlo Policy Evaluation*.
- AISTATS 2023, PMLR 206:9852-9878.
- 网页：https://proceedings.mlr.press/v206/winnicki23a.html

该文说明：从单条策略轨迹进行 policy evaluation 后再做 policy iteration 的收敛并非自动成立；其正面结果使用 first-visit Monte Carlo evaluation 和 lookahead improvement。它支持本项目把 blockwise control 单独分析，但不能为本项目的 greedy 或 \(\varepsilon\)-greedy block update 直接背书。

## 2. Xie 定理的准确假设与结论

### 2.1 问题与数据模型

Xie 固定一个策略并直接研究其诱导的有限 MRP：

\[
(\mathcal S,p,r,\gamma,p_0),
\qquad r:\mathcal S\to\mathbb R.
\]

奖励在论文 MRP 定义中是 state reward，轨迹满足

\[
S_0\sim p_0,\qquad
S_{t+1}\sim p(\cdot\mid S_t),\qquad
R_{t+1}=r(S_t).
\]

因此其主证明没有单独的条件 reward noise。把结论用于随机 \(R_{t+1}\mid S_t,S_{t+1}\) 时必须新增噪声项或扩大 Markov state。

论文假设 \(P_\pi\) 在有限状态空间上 ergodic，并有唯一平稳分布 \(\mu_\pi\)。Section 5 还明确假设上下文轨迹访问每个状态：

\[
\{S_0,\ldots,S_{n-1}\}=\mathcal S.
\]

### 2.2 Weighted-softmax TD

对固定上下文轨迹，定义

\[
\delta_k^{(l)}
=R_k+\gamma v_l(S_k)-v_l(S_{k-1}),
\]

\[
K(S_{k-1},S_j)
=\frac{\exp g(S_j,S_{k-1})}
{\sum_{m=1}^n\exp g(S_j,S_{m-1})},
\]

\[
v_{l+1}(S_j)
=v_l(S_j)+\alpha_l\sum_{k=1}^nK(S_{k-1},S_j)\delta_k^{(l)}.
\]

Transformer construction允许通过缩放参数实现一般 \(\alpha_l\)，但 Section 5 的正式收敛分析明确设置 \(\alpha_l=1\)。因此“一般步长的 contraction constant”是本研究的新推导，不是 S1 已证明结论。

### 2.3 Population kernel 与对角条件

\[
M_\pi(s,s')
=\frac{\mu_\pi(s')\exp\langle x(s),x(s')\rangle}
{\sum_{u\in\mathcal S}\mu_\pi(u)\exp\langle x(s),x(u)\rangle}.
\]

Assumption 5.1 要求存在

\[
C_{5.1}\in\left(0,\frac{1-\gamma}{2}\right)
\]

使得

\[
\min_sM_\pi(s,s)
\ge\frac{1+\gamma}{2}+C_{5.1}.
\]

这不是普通的“对角最大”条件，而是随 \(\gamma\) 接近 1 显著变强的定量 diagonal-mass 条件。

### 2.4 经验算子与概率界

经验算子的线性部分为

\[
I-\widehat M_n+\gamma\widehat P_n.
\]

Xie 先用 S2 控制经验状态频率与 transition-pair additive functional，再证明经验 kernel 继承 population diagonal margin。Theorem 2 的结论为：当

\[
n\ge C_{\mathrm{Thm2},1}\log(1/\delta)
\]

时，以至少 \(1-\delta\) 的概率，对所有 \(L\ge0\)，

\[
\|v_L-v_\pi\|_\infty
\le
(1-C_{5.1})^L\|v_0-v_\pi\|_\infty
+C_{\mathrm{Thm2},2}
\sqrt{\frac{\log(C_{\mathrm{Thm2},3}/\delta)}{n}}.
\]

常数吸收了 \(|\mathcal S|\)、\(\mu_{\min}\)、特征范数、谱参数、\(\gamma\) 和 \(\|v_\pi\|_\infty\) 等问题依赖量。该定理不是一个显式 minimax sample-complexity rate。

### 2.5 证明中的 pair-chain

S1 Appendix D.3 已使用 transition-pair chain

\[
(S_{k-1},S_k)
\]

来集中 transition residual。其状态空间是

\[
E=\{(a,b):P_\pi(a,b)>0\},
\]

平稳分布为

\[
\widetilde\mu_\pi(a,b)=\mu_\pi(a)P_\pi(a,b).
\]

这证明 S1 的技术工具能处理相邻状态对，但它不是本项目需要的 state-action chain \((S_t,A_t)\)，也不自动给出 Direct-Q theorem。

## 3. 两条分支的假设矩阵

| 项目 | S1：fixed-policy V | Branch A：Direct-Q | Branch B：V-first |
|---|---|---|---|
| 目标 | \(v^\pi(s)\) | \(Q^\pi(s,a)\) | 先 \(V^\pi(s)\)，再 \(Q^\pi(s,a)\) |
| 基础链 | \(S_t\) | \(X_t=(S_t,A_t)\) | V 阶段用 \(S_t\)；recovery 用 \((S_t,A_t,S_{t+1})\) |
| 平稳质量 | \(\mu^\pi(s)\) | \(\mu_X^\pi(s,a)=\mu^\pi(s)\pi(a\mid s)\) | V 阶段依赖 \(\mu^\pi(s)\)；recovery 仍依赖 \(\mu_X^\pi(s,a)\) |
| 遍历性 | S1 假设 finite ergodic state chain | 必须证明 pair-chain 在支持集上 ergodic，并记录其谱参数 | V 阶段沿用 S1；recovery 需要 ratio/conditional-mean concentration |
| 覆盖 | 上下文访问全部状态 | 上下文必须访问目标 state-action pairs | V 阶段访问全部状态；完整 Q recovery 仍需目标 pairs |
| 奖励 | S1 为确定性 state reward | 项目 MDP 是 action-conditioned、可含随机 reward | recovery target 直接包含 reward noise |
| kernel | state kernel \(M_\pi\) | state-action kernel \(M_X^\pi\) | V 阶段用 state kernel；recovery 可用 exact count 或 pair kernel |
| 对角条件 | \(M_\pi(s,s)\ge(1+\gamma)/2+C\) | 需要 \(M_X^\pi(x,x)\) 的对应条件 | 主体 V 阶段只需 state diagonality；recovery 不一定需要迭代 contraction |
| 步长 | S1 正式收敛证明用 \(\alpha=1\) | 需重推一般 \(\alpha\) 或先固定 \(\alpha=1\) | V 阶段先沿用 \(\alpha=1\)；recovery 是一次 plug-in estimate |
| 数据依赖 | 同一固定上下文反复跨层使用 | 同一 pair context 反复跨层使用 | 主证明用样本切分隔离 \(\widehat V\) 与 recovery targets |
| 有限样本瓶颈 | state frequency + transition-pair concentration | pair frequency + pair-transition concentration + reward noise | V estimation + action-conditioned recovery concentration |
| control | S1 明确不含 control | fixed-policy theorem 后再做 blockwise control | fixed-policy theorem 后再做 blockwise control |

## 4. Branch A 需要新证明的命题

以下均标为 `derived here / proof required`：

1. \((S_t,A_t)\) pair-chain 的状态空间、转移矩阵、平稳分布与遍历性。
2. Direct-Q population operator 以 \(Q^\pi\) 为不动点。
3. state-action empirical kernel 对 population kernel 的一致逼近。
4. pair-transition residual 的 Markov concentration。
5. 随机 action-conditioned reward noise 项。
6. 一般 \(\alpha\) 下的收缩常数与对角阈值。
7. one-hot pair kernel 所需 sharpness 与 \(\mu_{X,\min}^\pi\) 的显式关系。

对 one-hot pair feature，可直接代数得到

\[
M_X^\pi(x,x)
=\frac{\mu_X^\pi(x)e^\eta}
{\mu_X^\pi(x)e^\eta+1-\mu_X^\pi(x)}.
\]

若目标 diagonal threshold 为 \(d\in(0,1)\)，则必要且充分的 sharpness 条件是

\[
\eta
\ge
\log\frac{d[1-\mu_X^\pi(x)]}
{(1-d)\mu_X^\pi(x)}
\]

对所有目标 pair 成立。该式是本项目代数推导，不来自 S1 的显式陈述。

## 5. Branch B 需要新证明的命题

### 5.1 Population recovery

定义

\[
\bar Q_{\widehat V}(s,a)
=\mathbb E[R_{t+1}+\gamma\widehat V(S_{t+1})\mid s,a].
\]

由 Bellman identity 和条件期望的 \(\ell_\infty\) 非扩张性，直接得到

\[
\|\bar Q_{\widehat V}-Q^\pi\|_\infty
\le\gamma\|\widehat V-V^\pi\|_\infty.
\]

该引理标为 `derived here`，不冒充外部论文定理。

### 5.2 Finite recovery

令 recovery 集上某 pair 的计数为 \(N_{s,a}\)，样本 target 为

\[
Y_t=R_{t+1}+\gamma\widehat V(S_{t+1}).
\]

样本切分后，给定 \(\widehat V\)，\(Y_t\) 是有界的 Markov additive functional。需要控制分子

\[
\sum_t\mathbf 1\{(S_t,A_t)=(s,a)\}Y_t
\]

与随机分母 \(N_{s,a}\) 的 ratio error。即使 V 阶段只依赖 state coverage，uniform Q recovery 的最差项仍由最小 pair occupancy 控制。

S2 可提供 additive-functional concentration；把它变成 ratio estimator 的 uniform bound 是本研究需要完成的步骤。

## 6. Blockwise control 的来源边界

S4 表明单轨迹 policy evaluation 与 policy improvement 的组合需要专门分析；其正面结论依赖 first-visit Monte Carlo 和 lookahead，不能直接套到本项目。

因此本研究的 blockwise control 第一版只作：

1. 每块重新验证 fixed-policy 假设；
2. 报告近似 policy-improvement 界与 action gap；
3. 实证记录回报与错误动作切换；
4. 不在没有新证明时声称全局收敛。

## 7. 当前最关键的研究判断

1. **Direct-Q 的技术路线是可行的，但不是零成本 lift。** S1 已提供状态 kernel 与 transition-pair concentration 的模板；新困难集中在 state-action occupancy、pair-chain mixing 与 reward noise。
2. **V-first 的 \(\gamma\)-误差桥非常干净，但不自动消除 action coverage。** 它可能降低迭代评估阶段的 kernel 难度，却仍需要一次可靠的 action-conditioned recovery。
3. **两条路线可能具有相同的最坏 pair-coverage 阶，但常数和迭代结构不同。** 这应由理论分解与 oracle-V 消融共同判断。
4. **最先应验证的是 coverage barrier，而不是完整 control。** 如果两条路线在 fixed-policy 下都无法在目标 \(\pi_{\min}\) 区间达到可靠 action identification，进入 fully online control 没有理论意义。

