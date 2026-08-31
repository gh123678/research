# icrl_softmax

> 当前状态：fixed-policy 有限样本阶段已于 2026-08-31 完成，并完成 480 项同种子正式复现。

本项目研究结构化 softmax attention 与强化学习更新之间的关系。当前主线不是训练通用 Transformer，而是分析：在一条冻结的平稳 Markov 轨迹上，Direct-Q 与 V-first 两条固定策略评估路线何时具有显式、可计算的有限样本保证。

当前工作区状态见 [ACTIVE_WORKSPACE.md](ACTIVE_WORKSPACE.md)。

## 当前研究问题

设 $X_t=(S_t,A_t)$。在有限状态动作空间和固定策略 $\pi$ 下，项目比较两条路线：

- **Direct-Q**：在 state-action pair 空间直接迭代经验算子；
- **V-first**：先估计 $V^\pi$，再用同一条轨迹恢复 $Q^\pi$。

两条路线共享 state、pair 和 edge 三条 Markov 链上的 count/residual 事件。目标是把经验收缩、固定真实目标处的残差和 softmax leakage 组合为全部迭代层同时成立的路径界。

## 已闭合的结论

### Direct-Q

Direct-Q 不需要对数据依赖的每个 $Q_\ell$ 逐层做 concentration。整条轨迹冻结后，经验算子也是固定的；概率事件只需控制经验收缩条件和真实 $Q^\pi$ 处的固定 residual。事件成立后，递推对所有层及数据依赖的 early stopping 层数同时有效。

Exact matching 的收缩条件不依赖 softmax 温度。Finite-softmax 路线还需要正的经验 pair-kernel margin。

### V-first no-split

V-first 使用同一批数据估计 value 和恢复 Q 时，不需要阶段独立、sample split、cross-fit 或人为 gap。Recovery 算子对输入 value 逐轨迹是 $\gamma$-Lipschitz；在固定 $V^\pi$ 处控制一次 ghost residual 后即可组合。

Finite-softmax recovery 保留显式 leakage，因此“有限样本上界成立”和“固定有限温度下一致收敛”必须分开陈述。

### 共享证书

项目已经实现：

- state、pair、edge 三条链上的统一固定函数事件；
- Direct-Q 全层界与 V-first no-split ghost-target 界；
- exact/softmax 路线级 margin、收缩率和误差分解；
- strict JSON 输出、失败原因和路径式 slack；
- 对旧 10 条路线的同种子数值回归。

完整定理与假设见 [共享单事件 fixed-policy 有限样本定理](docs/research_branches/shared_fixed_policy_finite_sample_theory.md)。

## 适用范围与限制

当前正式结论适用于：

- 有限状态动作空间；
- 固定策略；
- 平稳起步；
- 一条冻结轨迹；
- 确定性的 bounded edge reward；
- 定理所需支持上的 full coverage。

当前结论**不覆盖**：

- fully online control 或策略持续变化；
- 非平稳初始分布；
- 额外随机 reward noise；
- 未访问 pair 上的全空间一致保证；
- 随层变化的 attention score 或经验算子；
- 已经尖锐到 visit-indexed/self-normalized 量级的 coverage rate。

Fixed-policy 定理不能自动推出单调 policy improvement。Blockwise control 结果目前只作为诊断证据。

## 环境

当前仓库没有锁定依赖文件。Fixed-policy 验证与扫描使用 Python、NumPy、SciPy 和 Matplotlib；历史端到端 SARSA verifier 还需要 PyTorch。协作环境使用 Windows 11 与 Anaconda Python。

以下命令均从 `icrl_softmax/` 目录运行。

## 快速验证

先运行三个当前 fixed-policy 契约：

```bash
python -B verify_finite_sample_theorems.py
python -B verify_fixed_policy_q_routes.py
python -B verify_crossfit_markov_certificate.py
```

它们分别验证：

- 共享事件、Direct-Q、state-value 和 V-first no-split 的公式契约；
- Direct-Q/V-first 路线的确定性与公式级回归；
- state、pair、edge Markov certificate 与 cross-fit accounting。

这些脚本不会重复正式 480 项扫描。

## 480 项正式复现

正式配置包含：

- 30 个独立任务；
- 轨迹长度 256、1024、4096、16384；
- 2 个 mixing 设置；
- 2 个 reward-gap 设置；
- $n_S=6$、$n_A=4$、$\pi_{\min}=0.05$、$\beta=8$；
- 每个配置 160 次评估迭代。

运行：

```bash
python -B evaluate_fixed_policy_q_routes.py \
  --tasks 30 \
  --trajectory-lengths 256 1024 4096 16384 \
  --n-actions 4 \
  --pi-mins 0.05 \
  --betas 8 \
  --mixing 0.08 0.5 \
  --gap-bonuses 0 0.5 \
  --iterations 160 \
  --seed 20260829 \
  --output-dir results/fixed_policy_finite_sample_certificates
```

分析与旧结果回归：

```bash
python -B analyze_fixed_policy_finite_sample_certificates.py \
  --old-dir results/fixed_policy_q_routes_crossfit \
  --new-dir results/fixed_policy_finite_sample_certificates
```

注意：

- `results/` 被 Git 忽略，新 clone 不包含正式结果；
- 分析脚本要求 `--old-dir` 和 `--new-dir` 中都已有 `task_results.json`；
- 历史 zero-mismatch 审计依赖保留下来的旧 baseline，不能在缺失 baseline 时声称已经复现；
- 正式扫描计算量明显高于三个快速验证脚本。

## 正式结果摘要

480 个 matched comparisons 给出：

- Direct-Q exact 平均 sup-norm error：0.5852；
- V-first no-gap cross-fit：0.5873；相对 Direct-Q 的配对差为 $+0.0022$，95% CI 为 $[-0.0079,0.0122]$；
- V-first no-split exact：0.5671；
- V-first no-split softmax：0.5741；
- 旧 10 条路线共同字段 mismatch 为 0，最大浮点差为 $8.88\times10^{-16}$。

这些结果支持“两条路线没有统一支配关系”。No-split exact 在当前扫描中误差最低，但这不是对所有 MDP、样本预算和超参数的统一优越性结论。

共享证书还显示：

- state、pair、edge 谱条件通过率为 100%；
- 数值 edge-support 截断率为 0%；
- 保守 pair-coverage lower bound 的通过率为 0%，因此先验高概率证书通过率也是 0%；
- 在较长轨迹上，观测路径界的验证率显著提高。

高概率证书通过率为 0% 表示当前 Hoeffding/union-bound 充分条件过于保守，**不表示**经验轨迹没有覆盖、路径界失败或定理错误。详细统计和路线判断见 [Direct-Q 与 V-first 双路线报告](docs/research_branches/direct_q_vs_v_first_report.md)。

## 核心代码

| 文件 | 作用 |
| --- | --- |
| [fixed_policy_finite_sample_certificate.py](fixed_policy_finite_sample_certificate.py) | 纯有限样本事件与路线级确定性界 |
| [evaluate_fixed_policy_q_routes.py](evaluate_fixed_policy_q_routes.py) | 同轨迹、同预算的正式路线扫描 |
| [analyze_fixed_policy_finite_sample_certificates.py](analyze_fixed_policy_finite_sample_certificates.py) | 旧路线回归、证书统计和配对分析 |
| [crossfit_vfirst.py](crossfit_vfirst.py) | V-first split/no-split/cross-fit 估计器 |
| [markov_coverage_certificate.py](markov_coverage_certificate.py) | state、pair、edge 链谱与 coverage 证书 |
| [verify_finite_sample_theorems.py](verify_finite_sample_theorems.py) | 新有限样本定理的契约验证 |
| [verify_fixed_policy_q_routes.py](verify_fixed_policy_q_routes.py) | 路线公式与确定性验证 |
| [verify_crossfit_markov_certificate.py](verify_crossfit_markov_certificate.py) | cross-fit 与 Markov certificate 验证 |
| [evaluate_blockwise_q_routes.py](evaluate_blockwise_q_routes.py) | fixed-policy 之外的 blockwise control 诊断 |

## 理论与研究记录

| 文档 | 内容 |
| --- | --- |
| [共享有限样本定理](docs/research_branches/shared_fixed_policy_finite_sample_theory.md) | 两条 fixed-policy 路线的统一正式陈述 |
| [Direct-Q 理论](docs/research_branches/branch_a_direct_q_theory.md) | pair-space population 与 single-trajectory 分析 |
| [V-first 理论](docs/research_branches/branch_b_v_first_theory.md) | state evaluation、recovery 与 no-split 分解 |
| [Cross-fit/Markov certificate](docs/research_branches/crossfit_markov_certificate_theory.md) | 三条链证书、gap 证据等级与边界 |
| [路线交叉审查报告](docs/research_branches/direct_q_vs_v_first_report.md) | 480 项实验、证书与路线判断 |
| [来源假设矩阵](docs/research_branches/source_assumption_matrix.md) | 外部来源、项目新增推论和假设边界 |

设计与实施记录位于 [docs/superpowers/](docs/superpowers/)。

## 仍开放的问题

下一阶段的主要理论问题是：

1. occupancy-adaptive、multiplicative 或 self-normalized 的更尖锐 coverage/residual rate；
2. 非平稳起步的 burn-in 与初始分布常数；
3. 额外随机 reward noise；
4. 在未知 transition kernel 下估计证书常数；
5. 策略改变时 occupancy、mixing、kernel margin 和 action gap 的跨块联合控制。

当前成果可以作为 fixed-policy 理论附录候选，但尚未自动升级为论文主文中的 online-control 定理。

## 历史成果：sampled-SARSA 构造

项目此前证明：在外部提供 equality-routing mask 与 visited-query/null-token gate、无 layer normalization/dropout、双向结构化注意力和固定步长的条件下，两块 residual attention-FFN 可以对同一 state-action pair 的样本执行精确组内平均 SARSA 更新。

这不是“通用预训练 Transformer 会自行发现 routing”的结论，也不是 decoder-only causal Transformer 的结论。

端到端验证：

```bash
python -B verify_end_to_end_sarsa.py
```

相关材料：

- [端到端 softmax-SARSA 构造性证明](论文_草稿/端到端_softmax_SARSA_构造性证明.md)
- [验证与复现核对指南](论文_草稿/验证_复现核对指南.md)
- [完整构造与双语构建复现](docs/完整构造与双语构建复现.md)

论文全文由分节源文件生成。不要直接编辑 `论文_草稿/full_paper.md`；生成与一致性检查入口为：

```bash
python -B tools/build_fullpaper.py
python -B tools/verify_bilingual_manuscript.py
```

## 本地数据与归档

- `results/`：实验输出，已被 Git 忽略；
- `papers/`：本地参考论文，已被 Git 忽略；
- `figures/`：纳入版本管理的小型正式图；
- 仓库外的 `_archive/icrl_softmax_history_20260831/`：旧诊断、临时构建树和历史运行产物，新 clone 不保证存在。

活跃脚本保持平铺 import 结构；不要在没有同步修改 import 和验证命令的情况下移动它们。
