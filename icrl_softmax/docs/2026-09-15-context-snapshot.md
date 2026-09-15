# 上下文快照（2026-09-15，压缩用）

> 用途：代替完整对话历史。读这份 + `icrl_softmax/ACTIVE_WORKSPACE.md` + 当前任务单即可续接。
> 所有结论均为初步结果，除标注 `VERIFIED` 者外。

## 1. 仓库与提交状态

- 主仓：`C:\Users\Admin\Desktop\research`；Python：`C:\Users\Admin\anaconda3`；远端 `git@github.com:gh123678/research.git`。
- **`main = 9c8f777`（本地=远端）**，已含三条线全部产物；`model.py` 相对 `c1e03dd` 零 diff、未改。
- 已同步远端分支：`claude/FP-COMPOSE-001`(f54b54e 后并入 main)、`codex/FP-COMPOSE-REVIEW-003`(6d8690e)、`codex/FP-SPEC-REVIEW-001`(eb06345)、`codex/FP-L12S-REVIEW-001`(e270a9e)、`codex/FP-SURVIVOR-REVIEW-001`(53408a5)、`claude/FP-SPEC-REVIEW-001`(0446303)。
- 合并过程：我的分支 fast-forward（main 本是祖先）；`310cd41`(REVIEW-003，无冲突)；`ef21677`(SPEC-REVIEW-001，索引冲突 1 处，两块都保留，683→721 行无损失)；`09ba11f` 找回了**从未入库的 FP-COMPOSE-001 任务单**（SHA256(LF)=`fa7b622b…`，与封存清单一致）。
- 我代 GPT 做过三次提交（其账户额度尽）：REVIEW-003 终版、SPEC-REVIEW-001、三处未提交审查成果——**内容一字未改**，提交信息已注明。
- 其他执行者的未跟踪文件（主仓 `LITERATURE_ALERTS.md`、`_audit_claims*.py`、`_backups/`、`_paper*.txt`、`tmp/` 等）**全程未动**。
- `FP-L12S-REVIEW-001` 与 `FP-SURVIVOR-REVIEW-001` 工作树现 clean（成果已代提交推送）。

## 2. 治理关键点（AGENTS.md 摘要）

GPT=主研究员（唯一起草正式任务）；Claude=辅助执行+独立验证；用户=最终裁决。
状态流 `DRAFT→REVIEW→ACTIVE→VERIFYING→VERIFIED`；长任务必须 GPT+Claude 同冻结任务单各自独立执行再双向验证；`VERIFIED` 需双方 PASS 或用户裁决例外。
**例外史**：`FP-COMPOSE-002` 由用户直接指令「你去做1」授权 Claude 起草（首例）；`FP-COMPOSE-002.md` §8 明文「VERIFIED 不适用（无双方验证、无独立预审）」。**不得把 Claude 起草当常规先例。**

## 3. 科学结论现状（一句话一件）

- **主问题**：固定权重 softmax 注意力网络能否把上下文经验变成**可认证、不退化**的策略改进？→ **有限肯定**：在冻结小环境+协议下成立（见下）；措辞受 §5 约束。
- **`FP-COMPOSE-001`（K=4，192 路线记录/1536 生产者步）**：`network×perstate×L12` 组合链——H1 安全（0 覆盖违规、0 退化、最小余量 +0.1147）、H2 非空多步（48/48）、H3 生产者判决 192/192 一致（而 13824 个 Q̂ 无一按位相等）。**状态 `ACTIVE`**（用户裁决**不升级** VERIFIED；独立审查见 §4）。
- **`FP-COMPOSE-002`（K=12，Claude 起草）**：H0 前四步与 001 **逐位一致（1536/1536）**；H1 4534 步 0 违规 0 退化；H4 192/192；后段 6–12 步贡献 **4.309 点**（固定分母 192，非零且不可忽略）；**预登记 P2 被证伪**（余量非线性收窄，−9%/步→实测 −2.9%/步且震荡）；24/384 格停发；**状态 `ACTIVE`**（同上，不升级）。
- **审阅修正**（外部审阅 + 我复算确认）：后段下标错位（4.309/7.839 点）；"192/192 每步正向"实为累计口径（逐步增量：192×5,191,188,186,185,185,180,180）；"小增量≠已耗尽"（反例 task313 仅闭合 62.68%）；扩维"上下界"是未证明断言（已撤回）；唯一认证批实际 **1,136**（非上限 1,152）；`d≈96` 预算结论错（前向成本∝d，约 3.7h>2h 上限）。
- **高价值副产物**：`y_range` 项占 `argmax` 对半径 46%；冻结 `y_range` 公式比证明文档已有的 `2R*+γ·span(V̂)` 松 2.46–3.10×；但换成紧界只降 E_Q **1.238×**（被 |mean| 稀释）——杠杆已计价关闭。维度标度：门限先于支持度关闭（d≈64–96，两点内插，方向性未被建立）。
- **`FP-SPEC-REVIEW-001`（VERIFIED，仅本审查）**：`model.py` 两个 Expected-SARSA 实现 = **本仓库构造的固定策略 Expected-SARSA 注意力算子**（精确路由版/有限-logit 版），**不是** Xie/Liang–Lai 原网络的复现；`model.py:926` "论文 Theorem 3.1" 可定位到本地构造，引用含糊但不等于误读；有限写回核是 Xie 内积 score 特例（φ=√τ·e_x）；有限版不能继承旧误差界。
- **`model.py` 审计**：有效输入 PASS（13,550+721 项；masked/finite 与纯 NumPy 参考一致）；三项 API 边界登记为后续 hardening 任务（policy 契约、sharpness=1000 exp 溢出、device 一致性）；**不改 model.py、不重跑封存实验**（用户裁决）。
- **措辞约束**：可称"本仓库构造的固定策略 Expected-SARSA 注意力算子实现"；**不可**暗示其为外部论文原网络。

## 4. 独立验证分层（哪层被谁验过）

- **算术/实现层**：C1 全量重放（Q̂ 差 0.0）、H0 跨进程逐位一致、80/120 位高精度边界复核、GPT 独立重放（REVIEW-003，87,546 断言全过）——**厚**。
- **定理层**（T1–T5、L12 单侧幅值界、条件性不退化）：GPT 在 REVIEW-003 **独立重证 PASS**。
- **不可消除的结构性残余**：两边都共享 `model.py`、MDP/训练生成器、数组生产者 → 共同概念错误风险仍在；故 `FP-COMPOSE-002` 保持 `ACTIVE`。"独立审查 PASS" ≠ "双方独立执行 PASS"。

## 5. 进行中任务：起草双路线独立复现任务（用户指令「你去起草」）

**核心问题**（用户给定）：在不共享 `model.py`、MDP/训练生成器、数组生产者、判决实现的前提下，能否复现"按状态保守更新缓解早停"的限定结论？用户要求：**先冻结 MDP 规范/种子/输入输出协议/验收标准，GPT 与 Claude 各自独立实现再交叉验证；先起草进 `REVIEW`，不直接跑实验**。

**起草成败线（我自定）**：规格必须**自足**——若附录写"见某某 .py"则"独立实现"从第一天起就是假的。

**复现对象 = FP-EARLYSTOP-001 v2（引理 A' 协议）**：
- 原族：4 状态×3 动作，`GAMMA=0.70, REWARD_BOUND=1.5, GAP_BONUS=0.5, PI_MIN=0.15, ALPHA=0.65, LAYERS=160, ZETA=XI=TAU=8.0`；`SEED=20260911`；mixing∈{0.08,0.5}；`task_index 12..23`（24 环境 × 2 路线 = 48 记录）。
- 环境：`fs.build_task` = `default_rng([SEED, round(100*mixing), task_index])` → `make_mdp`（sample_mdp: P~Dirichlet(1) float32、R~U(-1,1) float32、p0~Dirichlet；然后 P=(1-mixing)·sticky+mixing·P_rand，R[:,0,:]+=GAP_BONUS）→ `make_policy`（pi_min=0.15，每状态一优选动作 1-2·pi_min，`rng.integers`）。
- 训练批：`fs.training_batch`（`rng.choice(4,p=mu_state)` 起点 + `rollout`，`TRAIN_LENGTH=65536`）。
- 认证批（v2）：`vectorised_batch`（fp_sample_vectorised_batch.py L59）：`default_rng(seed_parts)`，`step_seed_parts=[SEED, TASK_SALT=77531, round(100·mixing), task_index, step]`；分块 32768；每块 `rng.choice(4,size,p=mu_state)` 起点；每步 2×`rng.random(size)` 逆 CDF 采样动作/后继；`CHAINS=65536, CHAIN_LENGTH=64`（一记录一步一批，三臂共享；A/C 臂用 `first_visit_batch(..., max_chains=16384)`，B 用全 65536）。
- 首访：`first_visit_batch`（每链每对保留首访，`min_visits=2000`）。
- **证书 = frozen 包络**（`mc.mp_certificate_firstvisit`，**非 L12**）：`value_bound=R*/(1-γ)=5`，`envelope=R*+γ·vb+vb=10`，`y_range=20`；`delta_dir=delta_step/(2·d)`（d=12）；半径 `sqrt(2·var·log(2/δ_dir)/n) + (7/3)·y_range·log(2/δ_dir)/max(n-1,1)`；`var ddof=1`；弃权条件 `N_x<2000`；含 `divergence_guard`（|Q̂|>5）。`delta_step=0.05/12`，K=12。
- 残差：`residuals_for`：`r + γ·(π(ns)·Q̂(ns)) − Q̂(s,a)`。
- 决策：`conjunctive_decision`（η 网格 (1,.5,.2,.1,.05,.02,.01)，首个 `min_s LB_s>0`，LB=⟨Δπ,Q̂⟩−E_Q‖Δπ‖₁）；`perstate_decision`（逐行首个 LB_s>0，不过行不动；行独立组合）。候选=`relative_softmax_candidate`（`π·exp(ηQ)` 行归一化，溢出安全）。
- 生产者：数组路线 `fs.run_route`（expected_exact/expected_finite，es L194/L291）；网络 = `model.py` 两个 EndToEnd*ExpectedSARSA（规格见 FP-SPEC-REVIEW-001 final_synthesis）。
- **v2 封存数字**（复现目标带）：A 合取16k：发出 34 步、首步 13/48、均益 1.702、闭合 **17.9%**；B 合取64k：195、29/48、4.237、**54.9%**；C 按状态16k：**474、46/48、5.054、74.4%**；配对 C>A 46/48、C<A 0、C>B 27/48、A 零发 35→C 救 33。
- **已确认的可行性要点**：`default_rng`（PCG64+SeedSequence）为公开可移植语义；`rng.choice(p)`/`rng.random` 属平台原语——**设计已定为：两边可用 numpy 原语，算法（生成器/批/证书/决策/生产者）各自从规格独立实现；基线模块只作参考 oracle 比对，不进生产路径**。网络用 float32 numpy 按规格实现，对 model.py 只作容差比对（不经 torch）。
- 起草时尚未读完：`fs.run_route` 的 expected_exact/expected_finite 精确公式（es L194/L291）、`optimal_values`、网络算子的行级公式（在 FP-SPEC-REVIEW-001 final_synthesis）。

## 6. 起草前已定调的设计决策

1. **numpy 原语可共享**（视同 IEEE float64）；概念性错误风险只存在于**算法层**与**规格层**——规格层由 REVIEW 覆盖。
2. 规格必须钉死 **RNG 调用序列**（分块大小、调用次序），因为向量化 vs 循环会改变流。
3. 生产者为**各自实现后比对**（判决序列必须逐位一致；数值用冻结容差）。
4. 基线代码（fs/mc/model.py/evaluate_*）**只作参考 oracle**（产生参考哈希/参考输出供比对），**不进生产路径**。
5. 验收草案：G1 两边 spec 实现逐位一致（env/train/batches/reduced 哈希）；G2 spec 输出 vs 基线参考（不一致→查规格，不自动判负）；P1 各自安全检查（全步 E_Q≥实测误差、0 退化）；P2 两边判决逐位一致；P3 数值容差（numpy 路 ~1e-9，网络路 ~1e-4）；R1 限定结论复现（闭合差 ≥40 点、首步 C≥2×A、C>A ≥40/48、救出 ≥25——从封存值留余量）；R2 各自网络 vs 数组判决序列一致。
6. **REVIEW 状态**：我起草但我**不能预审自己的草案**（同 FP-COMPOSE-002 教训），任务单须写明由 GPT 或用户预审后才进 ACTIVE；执行须 GPT 在可用时跑其路线。

## 7. 立即可续的下一步

1. 读完 `fs.run_route`（es L194/L291）与 `optimal_values`、FP-SPEC-REVIEW-001 final_synthesis 中网络算子行级公式。
2. 起草 `docs/research_tasks/FP-INDEP-001.md`（建议名）：例外横幅（Claude 起草、用户指令、非先例）+ 完整规格附录（§5 全部细节）+ 验收/失败/停止 + 双方范围 + REVIEW 状态说明。
3. 提交为 DRAFT→REVIEW，**不跑实验**。
