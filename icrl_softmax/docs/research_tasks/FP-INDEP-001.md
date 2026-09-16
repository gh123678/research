# FP-INDEP-001：双路线独立复现"按状态保守更新缓解早停"的限定结论

> ## ⚠️ 例外声明（必须最显眼）
>
> **本任务单由 Claude 起草**，依据是**用户 2026-09-15 的直接指令**（"仓库收口已完成。下一步应做研究决策……你去起草"）。
> 按 `AGENTS.md` §一，起草正式研究任务**是 GPT 主研究员的唯一职责**；本条是**用户以裁决者身份给出的明确例外授权**。
>
> **因此：**
> - 本任务单**不是** GPT 起草的，**不得**当作 GPT 的判断或认可。
> - 与 `FP-COMPOSE-002` 一样：**这仍是例外，不是先例**。下一份任务单仍应由 GPT 起草。
> - **预审问题**：按 `AGENTS.md` §四，预审应由非起草方完成。起草方是 Claude，而 GPT 在起草时额度不可用；
>   故本任务单进入 `REVIEW` 后，**必须由 GPT（恢复后）或用户本人预审**，在此之前**不执行任何实验**。
>   Claude **没有也绝不能**对自己的草案出具"预审通过"。

- 任务编号：`FP-INDEP-001`
- 版本：v1，2026-09-15
- 状态：`REVIEW`（待非起草方预审；**在预审通过前不得执行**）
- 代码基线：`1da7162abb662142332eaf4493209f73f41fb5c4`（当前 `main`）
- 上游：`docs/2026-09-14-main-question-evidence-map.md`；`FP-COMPOSE-001`/`FP-COMPOSE-002` 的封存与 `FP-COMPOSE-REVIEW-003` 的独立审阅；`FP-SPEC-REVIEW-001` 的规格对应结论；`FP-EARLYSTOP-001` 的 v2 封存（引理 A' 协议）

## 1. 为什么是这个任务（动机与缺口）

项目的限定结论"**在当前有限状态环境和既定采样协议下，按状态保守更新缓解了合取规则的早停问题**"
（`FP-EARLYSTOP-001` v2：`C 胜 A 46/48`、闭合 `74.4%` vs `17.9%`、首步发出 `46/48` vs `13/48`、
A 零发出 35 条中 C 救出 33 条）**至今只有单作者的实现路径**。

此前的独立审查（`FP-COMPOSE-REVIEW-003`）重放了那条路径并 PASS，但它**共享**了
`model.py`、MDP/训练数据生成器与数组生产者——这是审阅方自己声明、并因此拒绝把原任务升级为 `VERIFIED` 的**结构性残余**。
**这种残余不能靠"再跑一轮同型实验"消除，只能在任务设计上消除**：让两个执行者
**从同一份自足规格出发、各自独立实现全部算法件**，再交叉验证。

本任务问的就是这件事。它是**确认性复现**，不是新发现实验；复现对象全部来自既有封存，不引入新环境定义、不改任何结论措辞。

## 2. 研究问题与可证伪假设

**问题**：在**不共享** `model.py`、MDP/训练数据生成器、数组生产者与判决实现的前提下，
两个执行者能否复现该限定结论？以及：规格是否自足到能支撑这种复现？

**预登记假设（任一为假即 FALSIFIED 并原样报告，不调参补救）**：

- **G1（规格自洽）**：两条路线各自的 spec 实现在**全部 24 环境 × 12 步**上产出**逐位一致**的
  环境、训练批、认证批与首访压缩（哈希全等）。
- **G2（基线符合）**：spec 产出与**基线参考实现**（只作 oracle 使用，见 §4）的对应产物**逐位一致**；
  任何不一致→**调查并定位到 spec 还是基线**，**不自动判复现失败**。
- **P1（安全，每路线各自）**：每个实际评估步 `E_Q ≥` 该格实测 `‖Q̂−Q^π‖∞`（独立审计通道求解）；
  每个发出步逐分量不退化；无错误发出。0 违规只是失败检测，不是概率证明。
- **P2（跨路线判决一致）**：48 条记录 × 3 臂 × 2 生产者的**发出序列与逐行/逐格选中 η（含弃权标记）
  在两路线间逐位一致**。
- **P3（跨路线数值一致）**：numpy 路径 `|Δ| ≤ 1e-9`（绝对）；网络路径（float32）`|Δ| ≤ 1e-4`（绝对）；
  `E_Q` 差 ≤ `1e-9`。判决是离散量，**必须逐位一致，不设容差**。
- **R1（限定结论复现，每路线各自）**：相对 `FP-EARLYSTOP-001` v2 封存值（`closure_A=17.9%`、
  `closure_C=74.4%`、首步 `A 13/48`、`C 46/48`、`C>A 46/48`、`C<A 0`、A 零发 35→C 救 33），
  预登记复现带：
  - `closure_C − closure_A ≥ 40` 个百分点；
  - 首步发出 `C ≥ 40/48` 且 `A ≤ 20/48` 且 `C ≥ 2·A`；
  - 配对 `C > A ≥ 40/48`、`C < A ≤ 2/48`；
  - 救出数 `≥ 25`；
  - 三臂（含 B）零违规零退化。
- **R2（生产者对应，每路线各自）**：每 `(规则, 路线, 记录)` 上，**网络与数组的判决序列逐位一致**
  （数值不设按位相等要求；两生产者的 `Q̂` 差按 P3 容差报告）。
- **R3（对 v2 封存的一致性）**：每路线的判决序列与 v2 封存包存的字段（`emitted`、`eta_selected`、
  `states_updated`、`ordered_reasons`、`min_pair_count_observed`）**逐位一致**；`e_q` 差 ≤ `1e-9`；
  臂级汇总（闭合、收益）差 ≤ `0.5` 个百分点。
- **X1（最终交叉）**：两路线的结论表逐格一致（判决）或落在 P3 容差内（数值）。

## 3. 冻结输入与评价协议

**复现对象 = `FP-EARLYSTOP-001` v2（引理 A' 协议）的同一冻结协议**；环境、种子、预算、K 全部沿用：

| 项 | 固定值 |
|---|---|
| 族 | 4 状态 × 3 动作；`GAMMA=0.70`；`REWARD_BOUND=1.5`；`GAP_BONUS=0.5`；`PI_MIN=0.15` |
| 环境 | `mixing ∈ {0.08, 0.5}`；`task_index ∈ 12..23`（24 环境；从未用于前期探索之外，沿用） |
| 路线 | `expected_exact`、`expected_finite`；24 环境 × 2 路线 = **48 条记录** |
| 臂 | `conj_n16k`（A）、`conj_n64k`（B，4× 数据）、`perstate_n16k`（C） |
| 生产者 | `numpy`（float64）与**规格注意力算子**（float32，按 §10.10 实现；**不调用 model.py**） |
| 主种子 / 任务盐 | `20260911` / `77531` |
| 网络参数 | `ALPHA=0.65`；`LAYERS=160`；finite `ZETA=XI=TAU=8.0`；`Q_0=0` |
| 认证批 | 每（记录,步）一批：`65536` 链 × `64` 项，分块 `32768`；A/C 用每对首访的前 `16384` 链，B 用全部 |
| 保留 | 首访协议：每链每对只保留首访；`min_visits=2000`，不足弃权 |
| 风险 | `delta_total=0.05`；`K=12`；`delta_step=delta_total/K=0.05/12` |
| 证书 | **frozen 包络**（§10.8）：`value_bound=5`、`envelope=10`、`y_range=20`、`delta_dir=delta_step/(2d)`、`MP_CONSTANT=7/3`、`ddof=1` |
| η 网格 | `(1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)`，降序首通 |
| 停止 | 某格首次全部行弃权后该格停止；安全异常见 §6 |

**平台**：Python + numpy 2.4.6。两边**可以**共享 numpy 原语（`default_rng`、`choice`、`random`、`dirichlet`、`integers`）——
numpy 是平台，视同 IEEE float64。**不共享的是算法层**：本任务单 §10 的全部算子都必须由双方**各自从规格重写**，
且**生产路径不得 import** 基线模块（`fixed_policy_*`、`fp_*`、`evaluate_*`、`model.py`、`mdps.py`、`verify_*`）。

## 4. 允许方法与禁止事项

**允许**：
- 各自从 §10 规格重写全部算法件，代码结构任意（循环/向量化自选，但 RNG 调用序列必须逐字按 §10.6 钉死）。
- 基线模块**仅作参考 oracle**：双方可各自运行基线代码产出参考哈希/参考输出，用于 G2 符合性检查。
  oracle 产出**不进入**生产路径；其使用必须在清单中逐项记录。
- 模型参考比对：各自用 torch 2.11.0+cpu 调用 `model.py` 两个 `EndToEnd*ExpectedSARSA`，与各自的 spec 网络实现比对（容差 ≤1e-4 绝对）。

**禁止**：
- 生产路径 import 任何基线模块（含 `model.py`）。
- 按结果调整环境/种子/预算/η/精度/口径；不加入额外实验臂；
  **不调用** `L12M`/`L12S`/`L123` 或任何传播构造；本任务的证书是 frozen 包络，不是 L12。
- 不读取对方首次结果，直到各自封存并记录哈希。
- 不修改既有封存（含 v2 包）。

## 5. 产物与准确保存位置

每方（`results/FP-INDEP-001/<actor>/`）：

- `formal/`：`task_results.json`（权威件）、`input_manifest.json`（全部输入内容哈希 + 代码哈希 + 任务单哈希）、
  `config.json`、`environment.json`、运行日志、失败记录；
- `verification/`：本侧 C1 式重放报告。
- `docs/research_branches/FP-INDEP-001/<actor>/`：`premise_audit.md`（§9 门禁）、
  `evaluate.py`、`analyze.py`、`verify.py`、`first_result.md`、`verification_of_other.md`。

## 6. 验收标准、失败判据与停止条件

- **验收**：§2 的 G1、G2、P1、P2、P3、R1、R2、R3、X1 逐条有证据。**双方均为 PASS，任务才到 `VERIFIED`**。
- **失败判据**：G1/G2 哈希不一致 → 定位到 spec 或基线，修复方是**任务单修订**（GPT），不是静默改实现；
  P1 任一确认违规 → 停，保留失败证据；P2/R3 判决不一致 → 判定为复现失败的一种（实现或 spec 错误），原样报告；
  R1 落入带外 → 该限定结论在本复现下**不成立**，原样报告。
- **停止条件**：任一 `H1` 疑似异常（负余量、非有限值、或 `|余量| ≤ 1e-10` 的数值边界点）→
  **暂停该执行者全部正式采样**，按封存输入用 80/120 位独立高精度重算（沿用 `diagnose_boundary.py` 的模式，
  各路线须写自己的高精度复核件）；禁止把科学失败记为基础设施错误、禁止换种子续跑。
- **争议处理**：发现任务单缺陷 → 进入 `BLOCKED_BY_OBJECTION`，等用户裁决。

## 7. 预计耗时与资源

- 每路线：288 条轨迹（48 记录 × 3 臂 × 2 生产者）× ≤12 步；网络前向至多 `3456` 次（4×3 族，预计 ≈1 s/次）；
  认证采样 `576` 批 × `4,194,304` 项 ≈ `2.42e9` 项。单路线估计 **≤ 2 小时**（CPU 单进程）。
- **两路线顺序执行**（先 GPT 后 Claude 或反之，防内存竞争且保持盲态）；超上限则保存已完成与待办，不降低样本量/链数/步数。

## 8. 双方范围与各自验证

- **GPT**：`codex/FP-INDEP-001`；`results/FP-INDEP-001/codex/`。
- **Claude**：`claude/FP-INDEP-001`；`results/FP-INDEP-001/claude/`。
- 双方**各自**：先写 `premise_audit.md`（§9 门禁全 PASS 才可跑）→ 实现 → 封存首次结果并记录哈希 → 再读对方结果 → 写 `verification_of_other.md`。
- 双方的 `verify.py` 都必须**不导入对方新增的判决函数**；重放须从冻结种子逐步重建 π/Q̂、认证输入哈希/计数/E_Q、判决、价值与成本。
- **共同依赖声明**：numpy 平台原语是共享依赖（如同 IEEE float64）；其版本必须一致（2.4.6），
  且双方必须各自对生成器的**抽样法则**做只读核对（沿用 `shared_generator_review` 的做法：起点/动作/转移/奖励各自对照解析律），并记录限制。
- 若一方因额度/可用性不可用，**不得**由另一方同时跑两条路线——那正是本任务要消除的结构。
  届时任务停在 `REVIEW`/`ACTIVE`（未交叉验证），等其恢复。

## 9. 理论门禁（先于任何正式运行；逐项给有证据的 PASS/FAIL）

沿用 `FP-L12-one-sided-proof.md` 与 `FP-COMPOSE-REVIEW-003` 的定理复核，但**针对本任务的 frozen 包络证书**逐条重述：

- **T1** 历史条件：对每格每步列 `H_(k-1)=σ(MDP, train, B_1,…,B_(k-1))`；`Q̂_k/π` 历史可测且未读本步批；
  认证批种子按 `step` 索引（§10.6），故 `B_k ⊥ H_(k-1)`；首访样本在给定容量下 iid（引理 A'，`FP-L12S-REVIEW-001` 已确认）。
- **T2** 证书目标：`frozen` 包络是**固定常数**（`y_range=20`），不随数据变；`range(Y_x) ≤ 2R*/(1−γ) = 10 < 20`
  对这两族成立（R*=1.5）。MP 单侧界按真实、历史可测的 `ρ_x` 符号分情形，每个固定分布只需一个方向；
  `δ_dir = δ_step/(2d)` 对幅值目标至多花 `δ_step/2`（保守）。**不得**把幅值界写成中心区间。
- **T3** 条件性逐分量不退化：同一 sup 事件上 `A_s ≥ LB_s`；未更新行 `A_s=0`；`(I−γP^π')^{-1}` 非负；
  所有候选由旧 π 构造，不得把新策略 Q 替换旧策略 Q。
- **T4** 实现边界：生产者（数组与网络）只读 `(π, train)`，认证只读 `(Q̂, 认证批)`；审计通道（Q^π、v*）
  只用于事后核对，**不得**影响任何决策。
- **T5** 必答"是否把由当前认证数据算出的函数代入只适用于固定函数的浓度不等式"：**没有**——
  本任务证书的包络是常数，浓度的是样本自身的均值/方差。

## 10. 规格附录（自足性成败线：不看任何 .py 也能实现）

> **本附录是任务的核心交付。** 每条都写成数学/伪代码，RNG 调用序列逐字钉死。
> 双方实现时**只看本附录**；与基线的差异由 G2 检查发现。

### 10.1 常量

`S=4, A=3, d=12, γ=0.70, R*=1.5, GAP_BONUS=0.5, PI_MIN=0.15, SEED=20260911, SALT=77531,`
`TRAIN_LENGTH=65536, CERT_CHAINS=65536, CHAIN_LENGTH=64, CHUNK=32768, MIN_VISITS=2000,`
`DELTA_TOTAL=0.05, K=12, DELTA_STEP=0.05/12, ETA_GRID=(1,.5,.2,.1,.05,.02,.01),`
`ALPHA=0.65, LAYERS=160, ZETA=XI=TAU=8.0, VALUE_BOUND=R*/(1−γ)=5`。

### 10.2 环境（对每个 `mixing∈{0.08,0.5}`、`task_index∈12..23`）

```
rng = default_rng([SEED, round(100*mixing), task_index])
# sample_mdp（全部 float32）：
P0[s,a,:] ~ Dirichlet((1,1,1,1))        # rng.dirichlet(ones(4), size=(4,3))
R[s,a,s'] ~ U(-1,1)                     # rng.uniform(-1,1,size=(4,3,4))
p0 ~ Dirichlet((1,1,1,1))               # rng.dirichlet(ones(4))
# make_mdp（转 float64）：
P0 行归一化；sticky[s,a,s]=1；P = (1-mixing)*sticky + mixing*P0
R[:,0,:] += GAP_BONUS
# make_policy：
preferred = rng.integers(0, 3, size=4)
pi = 0.15 处除 pi[s, preferred[s]] = 0.70
```

### 10.3 精确量（审计/起点用）

```
P^π[s,t] = Σ_a π[s,a]·P[s,a,t]
r_sa[s,a] = Σ_s' P[s,a,s']·R[s,a,s']；r_π[s] = Σ_a π[s,a]·r_sa[s,a]
v^π = solve(I − γP^π, r_π)；Q^π[s,a] = r_sa[s,a] + γ·Σ_s' P[s,a,s']·v^π[s']
μ：解 (P^π)ᵀ μ = μ（末行替换为全 1 方程），负值截 0，归一化。
v*：贪心策略迭代（q = r_sa + γ·P v^π；argmax；直至策略不动或 1000 轮）。
```

### 10.4 训练批（只用于产生 Q̂）

**与 10.2 共用同一个 rng、顺序消费**：`start = rng.choice(4, p=μ)`；随后 `rollout` 跑 `TRAIN_LENGTH` 步：
每步 `a = rng.choice(3, p=π[s])`、`s' = rng.choice(4, p=P[s,a])`、`r = R[s,a,s']`。
取 `states=S[0:65536]`、`actions=A[0:65536]`、`rewards=Rew[1:65537]`、`next_states=S[1:65537]`。

### 10.5 认证批（每记录一步一批；与 10.2/10.4 的 rng 流**无关**）

`seed_parts = [SEED, SALT, round(100·mixing), task_index, step]`；`rng = default_rng(seed_parts)`；
分块 `CHUNK=32768` 链。**钉死的调用序列**（每块）：

```
current = rng.choice(4, size=块大小, p=μ)          # 1 次 choice
对 step = 0..63：
    u1 = rng.random(块大小)
    a = Σ_j 1{u1 > cumsum(π[current])[j]}（clip 到 [0,2]）
    u2 = rng.random(块大小)
    s' = Σ_j 1{u2 > cumsum(P[current,a])[j]}（clip 到 [0,3]）
    记录 (current, a, R[current,a,s'], s')；current ← s'
```

链在批内按 `链号×64+步` 排布。`next_actions = actions`。

### 10.6 首访压缩

对每 `(链, 对 x)`：保留该链**首次**访问 x 的那一项；`counts[x] = 访问过 x 的链数`；
`max_chains`（A/C 取 16384，B 取 65536）意为只用前 `max_chains` 条链。

### 10.7 残差

`Y = R[s,a,s'] + γ·Σ_b π[s',b]·Q̂[s',b] − Q̂[s,a]`（对压缩后的每条记录项）。

### 10.8 frozen 包络证书（**非 L12**）

```
若 max|Q̂| > 5        → 弃权，原因 divergence_guard_triggered
若 Q̂ 非有限           → 弃权，原因 numerical_nonfinite
若任一 N_x < 2000     → 弃权，原因 heldout_pair_support_missing
δ_dir = DELTA_STEP/(2·12)；lt = log(2/δ_dir)
每对：mean_x、var_x（ddof=1）
radius_x = sqrt(2·var_x·lt/N_x) + (7/3)·20·lt/max(N_x−1, 1)
eps_x = |mean_x| + radius_x；E_Q = max_x eps_x/(1−γ)
```

### 10.9 判决

候选：`π_η^+(s,a) ∝ π(s,a)·exp(η·Q̂(s,a))`，行归一化（实现：logits=log π + ηQ̂，减去行最大值后 exp、归一化）。
`LB_s = Σ_a Δπ_s(a)·Q̂(s,a) − E_Q·‖Δπ_s‖₁`。
- **A/B（合取）**：降序遍历网格，**首个**满足 `min_s LB_s > 0` 的 `η` → 整格更新为该候选；否则弃权。
- **C（按状态）**：每行独立取首个 `LB_s > 0` 的 `η`；无则该行不动；`η` 记录含"不更新"标记。

### 10.10 生产者

**数组-精确（expected_exact）**：`q=0`；重复 `LAYERS` 次：
`qbar_t = Σ_b π[s'_t,b]·q[s'_t,b]`；`δ_t = r_t + γ·qbar_t − q[s_t,a_t]`；
按对分组求均值 `m_x = Σ_{t∈x} δ_t / count_x`（`count` 为训练批内出现次数）；
`q[x] += α·m_x`（仅 `count_x>0` 的对动）。若 `max|q| > 5` → 停止并标记 diverged。

**数组-有限（expected_finite）**：每层三个有限算子：
`read_x = (e^ξ·q_x + (Σ_y q_y − q_x))/(e^ξ + d − 1)`；
`succ_s = softmax over all tokens y=(u,b) with score ζ·1{u=s}+log π(b|u)`，作用于 q；
`δ_t = r_t + γ·succ[s'_t] − read[current_t]`；
写回：`N_x` = 该步命中对 x 的转移数；`denom_x = N_x·e^τ + (m − N_x)`（`m` 为转移总数）；
`q[x] += α·(e^τ·Σ_{t∈x}δ_t + (Σ_all δ_t − Σ_{t∈x}δ_t))/denom_x`。**无 visited gate，未访问对也会泄漏更新**。
发散守卫同上。

**网络算子（按 FP-SPEC-REVIEW-001 已确立的本地规格实现；不调用 model.py）**：
- 掩码精确版：每个当前 pair 的读取**只**关注其匹配 token（精确等值路由）；写回对**已访问** pair 取匹配转移的均匀平均；
  **未访问** pair 只读零值 null token、更新为 0；后继头**直接构造**权重 `A[t,(u,b)] = 1{u=s'_t}·π(b|s'_t)`
  （合法输入下与 masked softmax 代数等价，但不调用 softmax）。每步重复 `LAYERS` 次。
- 有限版：与数组-有限路线**同一算子**，只是以满支撑 softmax 形式实现：读取/后继/写回三处均为有限 sharpness 的
  完整 token softmax；未访问 query 也产生非零泄漏更新。
- 实现语言/库自选（numpy float32 即可）；与 `model.py` 的比对是**参考 oracle 检查**（容差 ≤1e-4 绝对），不进生产路径。

### 10.11 度量与配对

每（路线,臂,生产者,记录）轨迹：`emitted`、`states_updated`、选中 `η`（含标记）、`E_Q`、逐步 `v` 链、
`total_gain = Σv_final − Σv_initial`、`closure = total_gain/initial_suboptimality`
（`initial_suboptimality = Σv* − Σv^π0`）。**环境是统计单元**；路线作配对比较。
配对规则：同记录内两两比较臂的总收益/闭合（胜/负/平）。

## 11. 起草与预审记录

- 2026-09-15：Claude 按用户直接指令起草（例外，见文首声明）。**未运行任何实验**；附录 10 的每式都对过基线源码。
- **预审缺口**：起草方即 Claude；GPT 额度不可用。故 `REVIEW` 必须由 **GPT（恢复后）或用户**完成。
  在预审通过前，本任务单**不构成执行授权**。
- 执行还需 GPT 跑 `codex/FP-INDEP-001` 路线；**两条路线绝不由同一执行者承担**。
