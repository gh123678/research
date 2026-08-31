# 文献调研：softmax attention 中的 in-context policy improvement

> 调研日期：2026-08-11 · 服务于 icrl_softmax 研究线
> 目的：确认"softmax attention 能否做构造性的 policy improvement"这一空白的真实性与边界

---

## Abstract（摘要）

本调研回答一个问题：**在 softmax 归一化的 attention 中，是否存在构造性的、类似 Q-learning 的 policy improvement（策略改进）结果？**

结论分三层：

1. **空白成立**。softmax attention 侧的构造性理论目前**只有评估**（唯一一篇：2605.07333）；所有构造性的"改进算子"（半梯度、max bootstrap、GD）都在**线性/ReLU attention** 或 **bandit 自改进** 一侧。
2. **两条硬证据解释空白为何存在**：softmax 输出被限制在输入 token 的凸包内（2005.09561，权重全正、和为 1，带符号更新被禁）；softmax 尖峰随序列增长弥散（2410.01104）。
3. **我们的工作落在格子里**：`δ·softmax(δ/τ)·φ`（score-scaled softmax）这类"把 δ 乘回权重"的实现，检索不到任何直接对应文献——最近邻居是线性侧的 2605.05755 与 softmax 评估侧的 2605.07333，两者之间无人。

---

## 1. 引言（为什么需要这份调研）

本研究（icrl_softmax）尝试把 **in-context policy improvement**（注意力模型在轨迹输入下实现策略改进，而非仅评估）从线性 attention 推进到 softmax attention。写论文前必须回答一个问题：

> **"softmax + 改进"这一格，是不是真的没人站？**

如果别人做过了，我们就是重复劳动；如果没人做，我们需要知道**为什么没人做**——是"没人想到"（低垂果实），还是"有硬障碍"（前人的坑在等我们踩）。这两者决定了论文的定位和难度。

本调研给出结论：**没人做，且有两条结构性的障碍证据**。这既确认了空白（novelty 成立），也预告了难点（为什么我们的方法有边界，见夜间报告 §19 的"评估值自包含"卡点）。

---

## 2. 研究问题（Research Questions）

三个问题，按"支持贡献的空白 → 机制实现 → 边界"三块设计：

- **RQ1**：softmax/归一化 attention 里有哪些**构造性的 policy improvement** 结果？
  （支撑"novelty"：我们的贡献是不是新的）
- **RQ2**：把"改进算子"放进 attention 的**权重或读出**，有哪些实现方式？
  （支撑"mechanism"：半梯度 / max bootstrap / score-scaled 各是谁做的）
- **RQ3**：评估值的**自包含**（bilinear 的 Q=φ·w 由 token 自己重推导出来）在 attention 里有没有被研究过？
  （支撑"边界"：我们卡在评估值自包含，看别人有没有解过）

---

## 3. 分类框架（MECE Taxonomy）

检索到的全部相关文献，按"方法阵营 × 是否改进"划分，保证**不重叠、不遗漏**：

| 阵营 | 定义 | 代表文献 | 数量 |
|---|---|---|---|
| **线性改进** (lin_improve) | 线性/ReLU attention，构造性做改进算子 | 2605.05755, 2603.01335, 2310.08566 | 3 |
| **softmax 评估** (sm_eval) | softmax attention，构造性做评估（Q/V 函数） | 2605.07333 | **1** |
| **softmax 算法模拟** (sm_emulate) | softmax attention 模拟 GD/回归等更新算子 | 2508.17550, 2510.10425, 2605.06609 | 3 |
| **障碍证据** (obstacle) | 证明 softmax 结构限制的理论 | 2005.09561, 2410.01104 | 2 |
| **bilinear 读出 / 自包含** (bilinear) | attention 输出 = 双线性读出；权重自包含 | 2605.10466, 2212.07677, 2211.15661, 2307.03576 | 4 |
| **决策类（旁支）** (decision) | 注意力做决策但不是本研究的构造性改进 | 2106.01345, 2306.14892, 2605.09867 | 3 |

> **术语陷阱（必须排除）**：策略分布 softmax `π(a)∝exp(Q/τ)`（也叫 softmax policy / Boltzmann policy）是**另一个成熟领域**，与"注意力权重 softmax"完全无关。检索时如果把它算进去，会误判空白不存在。本调研一律排除。

---

## 4. 分支比较（各阵营细读）

### 4.1 线性改进（lin_improve）——改进算子的"老家"

- **[2605.05755] Liang & Lai, 2026**《Transformers Provably Implement In-Context RL with Policy Improvement》：线性 self-attention 单块可**显式构造**实现半梯度 SARSA 与 actor-critic，带收敛证明。这是"改进算子放进 attention"的最直接先例——但在**线性**侧（无归一化）。
- **[2603.01335] Yu et al., 2026 (ICPO)**：单层线性 self-attention 在 Fisher 加权 logit-matching 预训练下模仿 linear bandits 的策略优化。注意是"test-time 自我改进"设定，非经典 ICRL 控制。
- **[2310.08566] Lin, Bai, Song, 2023**《Transformers as Decision Makers》：ReLU attention 逼近 LinUCB/TS/UCB-VI 等在线 RL 算法。决策类，非 bootstrap 改进。

**共同点**：全部用**无归一化**的 attention（线性/ReLU）。改进需要的带符号更新，在无归一化下是自由的。

### 4.2 softmax 评估（sm_eval）——唯一的一篇

- **[2605.07333] Xie, Liu, Chen, D. Liu, Chandra, Zhang, 2026**《Beyond Linear Attention: Softmax Transformers Implement In-Context RL》：**唯一真正 softmax attention 的构造性论文**。dual-head 层前向等价于带 softmax 加权的**非参数 TD 更新**（Berthier 式批版），在核空间做评估。**只评估，不改进。**

> 注意：经逐篇核验，曾被归入"softmax 评估阵营"的 2605.07123（CoT）、2405.13861（TD）、2509.18389（Emergence）三篇，理论部分**实际上都用线性 attention 简化证明**。真正的 softmax 构造性评估只有 2605.07333 一篇。→ 空白比预想更薄、更扎实。

### 4.3 softmax 算法模拟（sm_emulate）——softmax 能做更新算子的证据

- **[2508.17550] Hu et al., 2025**：冻结权重 softmax attention 可精确模拟 `f(wᵀx−y)`，涵盖 GD、线性/ridge 回归。**softmax 能做正加权更新算子的证据**。
- **[2510.10425] Dragutinović, Saxe, Singh, 2025**：softmax ≥ linear——softmax transformer 可**按核梯度下降**学习 in-context 分类。softmax 的非线性映射到核空间后就是 GD。
- **[2605.06609] Zhang & Cao, 2026**：softmax transformer 经**归一化 GD** 高效做 in-context logistic 回归。

**共同点**：证明 softmax 能做梯度型更新，但都在**核空间 / 正加权**。这里的关键区分：这些更新的"符号信息"由 `y`（标签）提供，是外生监督；而我们需要的改进是**从 δ 内生出的带符号方向**——后者撞凸包约束（见 4.4）。

### 4.4 障碍证据（obstacle）——为什么空白存在

- **[2005.09561] Richter & Wattenhofer, 2020**《Normalized Attention Without Probability Cage》：attention 权重被限制在概率单纯形 → 输出落在值的**凸包**内。带符号 / 越界更新被禁止。→ 这直接解释了我们框架里"纯凸组合（δ 不乘回）死"的现象：softmax 归一化吃掉了 δ 的符号。
- **[2410.01104] Veličković et al., 2024**《Softmax is not Enough (for Sharp Size Generalisation)》：即使"找最大值"这类任务，softmax 尖峰也随序列长度增长而**弥散**。→ 解释了"尖峰读不出可靠方向"的第二个结构性限制。

### 4.5 bilinear 读出 / 自包含（bilinear）——RQ3 的邻居

- **[2605.10466] Xu & Fang, 2026**《Self-Attention as a Covariance Readout》：softmax 输出**几乎必然收敛**到 `Θ_V · Σ · Θ_Kᵀ · Θ_Q · x`——协方差读出，**bilinear 结构**，单头 = 一步 population GD。**这是最接近我们"评估值自包含 Q=φ·w"的 softmax 侧工作**（但它是正定协方差，读不出带符号的方向）。
- **[2212.07677] von Oswald et al., 2022**：线性 transformer 通过**内隐梯度下降**学习 in-context（线性阵营自包含的源头）。
- **[2211.15661] Akyürek et al., 2023**：证明线性 transformer 可实现 GD 与 **ridge 回归逆算**——权重自包含（线性阵营）。
- **[2307.03576] Mahankali, Hashimoto, Ma, 2023**：单步 GD 是单层线性 self-attention 的最优 in-context learner。（注：此篇**不是** ridge 逆算那篇——ridge 逆算是 Akyürek 的；两者常被混记，已更正。）

### 4.6 决策类（decision）——旁支

- **[2106.01345] Chen et al., 2021 (Decision Transformer)**：attention 做条件行为克隆，非构造性改进。
- **[2306.14892] Lee et al., 2023 (DPT)**：监督预训练可学 in-context RL（行为克隆式）。
- **[2605.09867] Anand et al., 2026**：连续 latent context token 里做 weighted majority + Q-learning，状态存于 latent token（线性组合嵌入），非"注意力权重读出"。（注：最初记忆的标题"softmax 精确实现 tabular Q-learning"与该文不符，已更正——它是隐式 CoT 状态，非 softmax 权重构造。）

---

## 5. 讨论（把证据串起来）

**空白成立，且是"有障碍的空白"**：

- 改进算子的构造理论（RQ1/RQ2 线性侧）全部在无归一化 attention。softmax 侧只有评估（1 篇）。
- 两条结构障碍（凸包 2005.09561、弥散 2410.01104）解释了为什么"softmax + 带符号改进"没人做成：**归一化天生与符号更新冲突**。
- 我们的方法（C3-v3 / att_td_lin 的 `δ·softmax(δ/τ)·φ`，τ→∞ 退化为半梯度）**部分绕过**了凸包——把 δ 乘回权重，符号从"权重符号"恢复；但评估值自包含（RQ3）仍在。这与夜间报告 §19 的边界完全一致：**能做改进，但评估值无法自给**。
- 最接近邻居：2605.05755（线性改进）+ 2605.07333（softmax 评估）。我们的论文是**在两篇之间的格子里填第一笔**。

---

## 6. Open Problems（我们能贡献的开放问题）

1. **softmax 阵营的构造性评估也极薄**（只有 1 篇）——评估侧的空白同样可以写。
2. **带符号更新的凸包绕过**：`δ·softmax(δ/τ)` 的收敛性质、τ 的退火行为，无人研究。
3. **评估值自包含**（RQ3）：bilinear `Q=φ·w` 由 token 自给，softmax 侧无解；2605.10466 的协方差读出是正定的，读不出符号——这是边界，也可能是新方向。

---

## 7. 结论（逐条回答 RQ）

- **RQ1**（softmax 构造性改进）：**空白成立**。softmax 侧仅 2605.07333 一篇做评估；改进算子全部在线性/ReLU/bandit 侧。最近邻居 = 2605.05755（线性改进）+ 2605.07333（softmax 评估）。
- **RQ2**（改进算子放进 attention）：线性侧已有半梯度 δ·φ（2605.05755）；softmax 侧仅评估型"softmax 加权 TD"（2605.07333，softmax 加在 QK 相似度上，**不是**加在 δ 上）。`δ·softmax(δ/τ)·φ` 这类"softmax(δ)"版本**检索不到对应文献**。softmax 能做的梯度更新在核空间/正加权（2508.17550, 2510.10425, 2605.06609），符号来自外生监督 y，非内生 δ。
- **RQ3**（评估值自包含）：**有邻居，无直解**。softmax 侧最近的是协方差读出（2605.10466，bilinear 但正定）；线性阵营有完整自包含（2212.07677, 2211.15661）；softmax 直接"Q=φ·w 由 token 自给"尚无。

---

## 8. References（全部经 arXiv 逐篇核验）

**线性改进**
- [2605.05755] H. Liang, L. Lai. Transformers Provably Implement In-Context Reinforcement Learning with Policy Improvement. 2026.
- [2603.01335] T. Yu et al. Provable and Practical In-Context Policy Optimization for Self-Improvement (ICPO). 2026.
- [2310.08566] L. Lin, Y. Bai, S. Mei. Transformers as Decision Makers: Provable In-Context RL via Supervised Pretraining. 2023.

**softmax 评估**
- [2605.07333] Z. Xie, X. Liu, C. Chen, S. D. Liu, R. Chandra, S. Zhang. Beyond Linear Attention: Softmax Transformers Implement In-Context Reinforcement Learning. 2026.

**softmax 算法模拟**
- [2508.17550] J. Y.-C. Hu, H. Liu, J. Y. Zhang, H. Liu. In-Context Algorithm Emulation in Fixed-Weight Transformers. 2025.
- [2510.10425] S. Dragutinović, A. M. Saxe, A. K. Singh. Softmax ≥ Linear: Transformers may learn to classify in-context by kernel gradient descent. 2025.
- [2605.06609] C. Zhang, Y. Cao. Transformers Efficiently Perform In-Context Logistic Regression via Normalized GD. 2026.

**障碍证据**
- [2005.09561] O. Richter, R. Wattenhofer. Normalized Attention Without Probability Cage. 2020.
- [2410.01104] P. Veličković, C. Perivolaropoulos, F. Barbero, R. Pascanu. Softmax is not Enough (for Sharp Size Generalisation). 2024.

**bilinear / 自包含**
- [2605.10466] H. Xu, G. Fang. Self-Attention as a Covariance Readout: A Unified View of ICL and Repetition. 2026.
- [2212.07677] J. von Oswald et al. Transformers learn in-context by gradient descent. 2022.
- [2211.15661] E. Akyürek et al. What Learning Algorithms is In-Context Learning Using? 2023.
- [2307.03576] A. Mahankali, T. B. Hashimoto, T. Ma. One Step of Gradient Descent is Provably the Optimal In-Context Learner with One Layer of Linear Self-Attention. 2023.

**决策类（旁支）**
- [2106.01345] L. Chen et al. Decision Transformer: Reinforcement Learning via Sequence Modeling. 2021.
- [2306.14892] J. N. Lee et al. Supervised Pretraining Can Learn In-Context Reinforcement Learning. 2023.
- [2605.09867] E. Anand, A. Ateyeh, X. Cao, M. Dabagia. Continuous Latent Contexts Enable Efficient Online Learning in Transformers. 2026.

---

## 附录：调研方法注记（为什么这样做）

- **先宽后窄**：先按"主流学派 / 障碍证据 / 相邻领域 / 决策旁支"多个视角撒网搜，再逐篇收窄。不先宽，会漏掉"别人做过了"；不收窄，会混进策略 softmax 那几万篇。
- **每篇两步验证**（存在性 + 元数据）：用 arXiv API/页面核对 ID、标题、作者。本调研中因此纠正了 3 处记忆错误（2605.07123/2405.13861/2509.18389 实为线性、2307.03576 描述不符）——**不验证就会把错文献写进论文，这是科研硬伤**。
- **MECE 分类**：MECE = 相互独立（Mutually Exclusive）、完全穷尽（Collectively Exhaustive）。分类后每篇只属于一格，且所有相关文献都有归属，这样"空白"的论证才站得住。
- **这一步为什么决定论文生死**：novelty（新不新）在审稿里是 first-order。空白论证只要有一篇漏网之鱼，审稿人一句话就能推翻整篇贡献。所以调研宁慢勿虚。
