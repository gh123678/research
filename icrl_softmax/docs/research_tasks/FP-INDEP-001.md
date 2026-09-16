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
- 版本：v3，2026-09-15（v1、v2 各经预审 **OBJECTION**，本版逐项修复；记录见 §11）
- 状态：`REVIEW`（**修订后仍须非起草方重新预审；通过前不得执行**）
- 代码基线：`1da7162abb662142332eaf4493209f73f41fb5c4`（当前 `main`）
- 上游：`FP-EARLYSTOP-001` v2 封存（引理 A' 协议，复现对象）；`FP-COMPOSE-REVIEW-003`（定理层复核）；`FP-SPEC-REVIEW-001`（网络算子的本地规格对应，VERIFIED 仅本审查）；证据对照表 `docs/2026-09-14-main-question-evidence-map.md`

## 1. 为什么是这个任务（动机与缺口）

项目的限定结论"**在当前有限状态环境和既定采样协议下，按状态保守更新缓解了合取规则的早停问题**"
（`FP-EARLYSTOP-001` v2：`C 胜 A 46/48`、闭合 `74.4%` vs `17.9%`、首步发出 `46/48` vs `13/48`、
A 零发出 35 条中 C 救出 33 条）**至今只有单作者的实现路径**。

此前的独立审查重放了那条路径并 PASS，但它**共享**了 `model.py`、MDP/训练数据生成器与数组生产者——
这是审阅方自己声明、并因此拒绝升级 `VERIFIED` 的**结构性残余**。**这种残余不能靠"再跑一轮同型实验"消除，
只能在任务设计上消除**：让两个执行者**从同一份自足规格出发、各自独立实现全部算法件**，再交叉验证。

本任务问的就是这件事。它是**确认性复现**，不是新发现实验；不引入新环境定义、不改任何结论措辞。

## 2. 研究问题与可证伪假设

**问题**：在**不共享** `model.py`、MDP/训练数据生成器、数组生产者与判决实现的前提下，
两个执行者能否复现该限定结论？以及：规格是否自足到能支撑这种复现？

**计数与比较的全部定义（先冻结，杜绝歧义）**：
- 统计单元 = 环境；路线作配对比较。记录 = `(环境, 路线)`，共 48 条。
- **配对胜负**（同记录内两臂比 `closure`）：`win` ⟺ `closure_C > closure_A + 1e-9`；`loss` ⟺ `closure_C < closure_A − 1e-9`；否则 `tie`。`1e-9` 为冻结平局容差。
- **救出**：一条记录满足"A 在全部 12 步从未发出，且 C 至少发出一次"。
- **首步发出**：该臂在该记录的第 1 步 `emitted = true`。

**预登记假设（任一不成立即按 §6 记录为 FALSIFIED 并原样报告，不调参补救）**：

- **G1（规格自洽）**：两条路线各自的 spec 实现在**全部 24 环境 × 12 步**上产出**逐位一致**的
  环境、训练批、认证批与首访压缩（内容哈希全等）。
- **G2（基线符合）**：spec 产出与**基线参考实现**（只作 oracle 使用，见 §4）的对应产物逐位一致。
  不一致时按 §2.1 的**封闭决策树**处理。
- **P1（安全，每路线各自）**：每个实际评估步 `E_Q ≥` 该格实测 `‖Q̂−Q^π‖∞`（独立审计通道求解）；
  每个发出步逐分量退化 `min Δv ≥ −1e-12`；无错误发出。0 违规只是失败检测，不是概率证明。
- **P2（跨路线判决一致）**：48 记录 × 3 臂 × 2 生产者的**发出序列、选中 η（含"不更新"标记）、
  弃权原因序列**在两路线间**逐位一致**。判决是离散量，**必须逐位一致，不设容差**。
- **P3（跨路线数值一致）**：numpy 路径 `|Δ| ≤ 1e-9`（绝对）；网络路径（float32）`|Δ| ≤ 1e-4`（绝对）；
  `E_Q` 差 ≤ `1e-9`。
- **R1（限定结论复现，每路线各自；全部用显式计数/显式均值）**：
  - **`mean(closure_C) − mean(closure_A) ≥ 0.40`**（`closure_x` 为该臂在 48 条记录上的 `fraction_gap_closed`；
    均值按记录数取算术平均；`0.40` 为分数单位，即 40 个百分点。封存对照：`0.744 − 0.179 = 0.565`）；
  - `count(C 首步发出) ≥ 40` 且 `count(A 首步发出) ≤ 20` 且 `count(C 首步发出) ≥ 2 × count(A 首步发出)`；
  - `count(win(C,A)) ≥ 40`；`count(loss(C,A)) ≤ 2`；
  - `count(救出) ≥ 25`；
  - 三臂（含 B）零违规零退化。
  封存对照值：`closure_A 17.9% / closure_C 74.4%`（差 56.5 点）、首步 `A 13 / C 46`、
  `win 46 / loss 0`、救出 `33`。
- **R2（生产者对应，每路线各自）**：每 `(规则, 路线, 记录)` 上，**网络与数组的判决序列逐位一致**
  （两生产者的 `Q̂` 差按 P3 网络容差报告，不要求按位相等）。
- **R3（对 v2 封存的一致性）**：每路线与 v2 封存包逐字段比对（字段与序列化格式见 §6.1）：
  判决类字段**逐位一致**；`e_q`、`min_lb` 差 ≤ `1e-9`；臂级汇总（闭合、收益、初始量）差 ≤ `1e-6`。
- **X1（最终交叉）**：两路线的结论表逐格一致（判决）或落在 P3 容差内（数值）。

### 2.1 G2 的封闭决策树（预审第 3 条的修复；v3 补齐至全集）

G2 发现某输入产物的 spec 哈希 ≠ oracle 哈希时，按下述顺序处理。**每一步都只走向五个终态之一，
不存在"以上皆非"**：

**第 1 步：整数/下标字段是否全部一致？**

- **任一整数/下标字段不同** → 终态 **`G2-SPEC-DEFECT`**（规格的钉死调用序列在结构上无法复现 oracle）。
- **全部一致** → 第 2 步。

**第 2 步：双序重算自检。** 双方各自用**两种**运算次序（规格钉死的次序、oracle 的次序）重算该产物：

- 规格次序**不能**复现 spec 产物，**或** oracle 次序**不能**复现 oracle 产物
  （含"两种次序都无法复现各自产物"的情形）→ 终态 **`G2-SPEC-DEFECT`**
  （规格文本对运算次序的钉死是错的、含糊的，或不可实现）。
- 两种次序各自复现各自产物 → 差异确属**运算次序**，进入第 3 步。

**第 3 步：判决影响探针。** 在**两套产物**上各跑完整判决链（48 记录 × 3 臂 × 2 生产者的全部步），
比对全部判决字段（`emitted`、`eta_selected`、`states_updated`、`ordered_reasons`）：

- **任一判决不同** → 终态 **`G2-DEVIATION-DECISION-IMPACT`**（复现目标歧义 → `BLOCKED_BY_OBJECTION`，用户裁决）。
- **全部判决一致** → 第 4 步。

**第 4 步：浮点幅度分流。**

- `max|Δ| ≤ 1e-12` → 终态 **`G2-DEVIATION-BENIGN-ROUNDING`**：生产继续用 spec 产物，记录偏差位置与幅度；
  R3 照常对 v2 成立。
- `max|Δ| > 1e-12` → 终态 **`G2-DEVIATION-BENIGN-MATERIAL`**：允许继续，但有三条强制要求——
  ① 偏差位置与幅度必须写在报告**最前面**；② R3 的**数值**比对须改为：在 **oracle 产物**上重跑生产链，
  其输出须与 v2 封存在容差内一致（这同时证明生产链本身无误）；③ 若该 oracle 产物重跑不满足 R3 →
  转入终态 **`G2-SPEC-DEFECT`**。

**终态全集（五态）**：`G2-PASS`、`G2-DEVIATION-BENIGN-ROUNDING`、`G2-DEVIATION-BENIGN-MATERIAL`、
`G2-DEVIATION-DECISION-IMPACT`、`G2-SPEC-DEFECT`。后两者为 `BLOCKED_BY_OBJECTION`，不进入生产运行；
两个 BENIGN 终态允许继续但都必须带偏差记录。
**证据要求**：每个终态都要给出定位坐标、双序重算结果、（如适用）两套产物上的判决比对输出位置。

## 3. 冻结输入与评价协议

**复现对象 = `FP-EARLYSTOP-001` v2（引理 A' 协议）的同一冻结协议**；环境、种子、预算、K 全部沿用：

| 项 | 固定值 |
|---|---|
| 族 | 4 状态 × 3 动作；`GAMMA=0.70`；`REWARD_BOUND=1.5`；`GAP_BONUS=0.5`；`PI_MIN=0.15` |
| 环境 | `mixing ∈ {0.08, 0.5}`；`task_index ∈ 12..23`（24 环境；沿用，与 v2 同） |
| 路线 | `expected_exact`、`expected_finite`；24 环境 × 2 路线 = **48 条记录** |
| 臂 | `conj_n16k`（A）、`conj_n64k`（B，4× 数据）、`perstate_n16k`（C） |
| 生产者 | `numpy`（float64）与**规格注意力算子**（float32，按 §10.10 实现；**不调用 model.py**） |
| 主种子 / 任务盐 | `20260911` / `77531` |
| 网络参数 | `ALPHA=0.65`；`LAYERS=160`；finite `ZETA=XI=TAU=8.0`；`Q_0=0` |
| 认证批 | 每（记录,步）一批：`65536` 链 × `64` 项，分块 `32768`；A/C 用每对首访的前 `16384` 链，B 用全部 |
| 保留 | 首访协议：每链每对只保留首访；`min_visits=2000`，不足弃权 |
| 风险 | `delta_total=0.05`；`K=12`；`delta_step=delta_total/K` |
| 证书 | **frozen 包络**（§10.8）：`value_bound=5`、`envelope=10`、`y_range=20`、`delta_dir=delta_step/(2d)`、`MP_CONSTANT=7/3`、`ddof=1` |
| η 网格 | `(1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)`，降序首通 |
| 停止 | 某格首次未发出后该格停止（轨迹不再继续）；安全异常见 §6 |

**平台（两侧必须一致）**：Python 3.13.9、numpy 2.4.6；torch 2.11.0+cpu 仅用于 `model.py` 参考比对。
两边**可以**共享 numpy 原语（`default_rng`、`choice`、`random`、`dirichlet`、`integers`）——numpy 是平台，视同 IEEE float64。
**不共享的是算法层**：§10 的全部算子由双方**各自从规格重写**，
且**生产路径不得 import** 基线模块（`fixed_policy_*`、`fp_*`、`evaluate_*`、`model.py`、`mdps.py`、`verify_*`）。

## 4. 允许方法与禁止事项

**允许**：
- 各自从 §10 规格重写全部算法件，代码结构任意；**但 §10.2–10.6 的 RNG/抽样调用序列逐字钉死，无实现自由**；
  证书/判决/生产者/求解有实现自由，但必须产出规格所定的输出（判决逐位，数值容差见 §2）。
- 基线模块**仅作参考 oracle**：双方可各自运行基线代码产出参考哈希/参考输出，用于 G2 符合性检查与
  R2/R3 的比对输入。oracle 产出**不进入**生产路径；其使用必须在清单中逐项记录。
- 模型参考比对：各自用 torch 调用 `model.py` 两个 `EndToEnd*ExpectedSARSA`，与各自的 spec 网络实现比对（≤1e-4 绝对）。

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
- `verification/`：本侧重放报告。
- `docs/research_branches/FP-INDEP-001/<actor>/`：`premise_audit.md`（§9 门禁）、
  `evaluate.py`、`analyze.py`、`verify.py`、`first_result.md`、`verification_of_other.md`。

## 6. 验收标准、失败判据与停止条件

- **验收**：§2 的 G1、G2、P1、P2、P3、R1、R2、R3、X1 逐条有证据。**双方均为 PASS，任务才到 `VERIFIED`**。
- **失败判据**：G1 不一致 → `G2-SPEC-DEFECT` 路径；G2 按 §2.1 决策树；P1 任一确认违规 → 停，保留失败证据；
  P2/R3 判决不一致 → 复现失败的一种（实现或 spec 错误），原样报告；R1 落入带外 → 该限定结论在本复现下**不成立**，原样报告。
- **停止条件**：任一 `H1` 疑似异常（负余量、非有限值、或 `|余量| ≤ 1e-10` 的数值边界点）→
  **暂停该执行者全部正式采样**，并按下条做高精度复核；禁止把科学失败记为基础设施错误、禁止换种子续跑。
- **高精度复核（自足要求，不指向任何既有文件）**：每路线须**自己实现**对受影响点的独立高精度重算——
  用任意精度十进制（80 位与 120 位，如 mpmath）从同一封存输入重算：逐对残差均值与 `ddof=1` 方差、半径、`E_Q`、
  逐行 LB 表与选中 η、以及 `π_before`/`π_after` 的 `Q^π`/`v^π`（任意精度线性求解）。
  **两个精度一致是强数值一致性检查，不是形式区间证明，也不因此升级任何结果。**
- **争议处理**：发现任务单缺陷 → 进入 `BLOCKED_BY_OBJECTION`，等用户裁决。

### 6.1 R3 的比对字段与序列化格式（预审第 4 条的修复）

v2 封存包每步字段与本任务的对应关系（**v2 的 `e_q` 就是 §10.8 的 `E_Q`**，JSON number，float64）：

> **"逐位"的可复现定义（预审第 5 条的修复）**：JSON 文本本身不保证保留 IEEE-754 位模式，
> 故本任务规定**规范序列化 + 封存完整性检查**：
> - 封存时一律用**可完整往返的序列化**（Python `json.dumps` 的 float64 默认 repr 即满足；或用显式十六进制位模式）；
> - 封存者在封存前必须运行**往返完整性检查**：把写出的 JSON 重新解析为 float64，
>   确认每个浮点字段**按 float64 相等**于内存值（`parsed == original`，含 `-0.0` 与 NaN 的位级语义——本任务不产生这两者）；
>   检查不通过则**不得封存**。
> - 比对时"逐位"= 解析为 float64 后按 `==` 精确相等；`eta_selected` 的每个非 null 元素**必须**
>   恰为冻结网格 `(1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)` 中的常数（网格常数本身按 float64 精确表示，
>   比较时以网格常数判定，不做近似匹配）。

| 字段 | 类型/序列化 | 比对方式 |
|---|---|---|
| `emitted` | JSON bool | **逐位**（见下方"逐位"的可复现定义） |
| `eta_selected` | A/B 臂：JSON number 或 null；C 臂：长度 4 的 JSON 数组，元素 number 或 null | **逐位** |
| `states_updated` | JSON integer | **逐位** |
| `ordered_reasons` | JSON 字符串数组；规范次序仅含本协议可产生的三种原因，按 `(divergence_guard_triggered, heldout_pair_support_missing, numerical_nonfinite)` 排序；证书通过而未发出时为 `["improvement_lcb_nonpositive"]`；发出时为 `[]`。**本协议不产生 `pair_count_mismatch`**（那是已废止的固定计数协议的原因）；若任何记录出现该字符串或其它未列原因 → **协议异常，停止该路线并记录** | **逐位** |
| `min_pair_count_observed` | JSON integer | **逐位** |
| `e_q` | JSON number（float64）或 null | `≤1e-9` |
| `min_lb` | JSON number（发出时为 `min_s LB_s`；未发出时为 `0.0`） | `≤1e-9` |
| `oracle_audit.realized_q_sup_error_vs_current_target_pi` | JSON number | `≤1e-9` |
| `oracle_audit.certificate_violation` | JSON bool | **逐位**（且**必须全为 false**） |
| 发出步的 `value_delta_vs_previous` / `total_value_gain` / `componentwise_nondegrading` | JSON array(4) / number / bool | `≤1e-9` / `≤1e-9` / **逐位** |
| 臂级汇总 `initial_v_sum` / `final_v_sum` / `initial_suboptimality` / `fraction_gap_closed` / `emitted_steps` | JSON number / integer | 数值 `≤1e-6`；计数**逐位** |

## 7. 预计耗时与资源

- 每路线：288 条轨迹（48 记录 × 3 臂 × 2 生产者）× ≤12 步；网络前向至多 `3456` 次（4×3 族）；
  认证采样 `576` 批 × `4,194,304` 项 ≈ `2.42e9` 项。单路线估计 **≤ 2 小时**（CPU 单进程）。
- **两路线顺序执行**（防内存竞争且保持盲态）；超上限则保存已完成与待办，不降低样本量/链数/步数。

## 8. 双方范围与各自验证

- **GPT**：`codex/FP-INDEP-001`；`results/FP-INDEP-001/codex/`。
- **Claude**：`claude/FP-INDEP-001`；`results/FP-INDEP-001/claude/`。
- 双方**各自**：先写 `premise_audit.md`（§9 门禁全 PASS 才可跑）→ 实现 → 封存首次结果并记录哈希 → 再读对方结果 → 写 `verification_of_other.md`。
- 双方的 `verify.py` 都必须**不导入对方新增的判决函数**；重放须从冻结种子逐步重建 π/Q̂、认证输入哈希/计数/E_Q、判决、价值与成本。
- **共同依赖声明**：numpy 平台原语是共享依赖（如同 IEEE float64）；版本必须一致（2.4.6）。
  双方必须各自对生成器的**抽样法则**做只读核对（起点/动作/转移/奖励各自对照解析律），并记录限制。
- 若一方因额度/可用性不可用，**不得**由另一方同时跑两条路线——那正是本任务要消除的结构。
  届时任务停在 `REVIEW`/`ACTIVE`（未交叉验证），等其恢复。

## 9. 理论门禁（先于任何正式运行；逐项给有证据的 PASS/FAIL；本节自足）

> 预审第 1 条的修复：本节与 §10 不再回指任何文档来承载命题内容；出处只作历史注记。

- **T1 历史条件**：对每格每步列 `H_(k-1) = σ(MDP, train, B_1,…,B_(k-1))`。
  `Q̂_k` 与 `π_(k-1)` 只由 `train`、初始策略与前 `k−1` 步判决决定，故 `H_(k-1)` 可测；
  认证批由按 `step` 索引的种子向量（§10.5）独立生成，故 `B_k ⊥ H_(k-1)`；
  残差 `Y_x = R[s,a,s'] + γ·Σ_b π[s',b]Q̂[s',b] − Q̂[s,a]` 在 `(π, Q̂)` 给定后是**固定函数**。
- **引理 A'（首访前提，内联陈述）**：每条链对每对只保留**首访**残差。链的首访指示 `G_c` 由首访前的轨迹决定，
  首访处的残差值在首访时新鲜抽取，故 `Y_c ⊥ 1{G_c}`；条件于 `N_x = n`，保留样本为 `P_x^{⊗n}`。
  **只有 `first_visit` 保留协议被授权**；固定计数（"每对前 n 个"）的停时论证已证伪，不得使用。
- **T2 证书目标（frozen 包络，单侧幅值界）**：本证书保证的是**幅值** `|ρ_x| ≤ |mean_x| + radius_x`，
  **不是**中心区间。按真实、历史可测的 `ρ_x` 的符号分情形：`ρ_x ≥ 0` 时只用 MP 上侧方向
  `ρ_x − mean_x ≤ r_x`；`ρ_x < 0` 时对 `−Y` 用 MP；`ρ_x = 0` 平凡。每个**固定分布**只需 MP 的**一个方向**，
  `δ_dir = δ_step/(2d)` 对幅值目标至多花 `δ_step/2`（保守）；`d·(2δ_dir) = δ_step`，跨步 ≤ `δ_total`。
  半径的经验 Bernstein 形式：`sqrt(2·var_x·lt/N_x) + (7/3)·y_range·lt/max(N_x−1,1)`，`lt = log(2/δ_dir)`，`ddof=1`。
  **包络前提**：`Y_x` 的值域 ≤ `2R*/(1−γ) = 10 < y_range = 20` 对这两族成立（`R*=1.5`），
  该界是**冻结常数**，不含任何浓度步骤。
- **T3 条件性逐分量不退化**：在同一事件 `E = {‖Q̂ − Q^π‖∞ ≤ E_Q}` 上，对任意行更新 `Δπ_s`（即使该行由当步认证数据选出）：
  `A_s = Σ_a Δπ_s(a)·Q^π(s,a) ≥ Σ_a Δπ_s(a)·Q̂(s,a) − E_Q‖Δπ_s‖₁ = LB_s`。算法只采用 `LB_s > 0` 的行，
  其余行原样不动（`Δπ_s = 0 ⟹ A_s = 0`）；所有候选由**旧 π** 构造；`(I−γP^{π'})^{-1} = Σ_t γ^t P^{π't}` 各项非负，故
  `v^{π'} − v^π = (I−γP^{π'})^{-1} A ≥ 0` 逐分量成立。**条件性**结论，条件于 `E`；不消耗额外风险。
- **T4 实现边界**：生产者（数组与网络）只读 `(π, train)`，**不读**认证批；证书只读 `(Q̂, 认证批)`，**不读** MDP 核/真值；
  审计通道（`Q^π`、`v*`）只在隔离的事后核对中使用，**不得**影响任何决策。
- **T5 必答**"是否把由当前认证数据算出的函数代入只适用于固定函数的浓度不等式"：**没有**——
  本证书的包络是常数（不随数据变），被浓度的是样本自身的均值/方差；行与 η 的选择是 `(Q̂, E_Q, 旧 π)` 的确定性函数；
  网络与数组生产者只读 `(train, π)`；本任务**不存在**任何传播构造。

## 10. 规格附录（自足性成败线：不看任何 .py 也能实现）

> **本附录是任务的核心交付。** 每条都写成数学/伪代码；§10.2–10.6 的 RNG/抽样调用序列**逐字钉死**（无实现自由）；
> 其余算子允许实现自由，但输出必须满足规格（判决逐位、数值容差见 §2）。
> 双方实现时**只看本附录**；与基线的差异由 G2 决策树处理。

### 10.1 常量

`S=4, A=3, d=12, γ=0.70, R*=1.5, GAP_BONUS=0.5, PI_MIN=0.15, SEED=20260911, SALT=77531,`
`TRAIN_LENGTH=65536, CERT_CHAINS=65536, CHAIN_LENGTH=64, CHUNK=32768, MIN_VISITS=2000,`
`DELTA_TOTAL=0.05, K=12, DELTA_STEP=0.05/12, ETA_GRID=(1,.5,.2,.1,.05,.02,.01),`
`ALPHA=0.65, LAYERS=160, ZETA=XI=TAU=8.0, VALUE_BOUND=5`。

### 10.2 环境（对每个 `mixing∈{0.08,0.5}`、`task_index∈12..23`）

平台：`numpy 2.4.6`，`default_rng`（PCG64 + SeedSequence）。

```
rng = default_rng([SEED, round(100*mixing), task_index])
# —— 三个抽样都是 float64 输出，**立即** .astype(float32)：
P0 = rng.dirichlet(ones(4), size=(4,3))           # → float32
R  = rng.uniform(-1.0, 1.0, size=(4,3,4))         # → float32
p0 = rng.dirichlet(ones(4))                       # → float32
# —— 随后转 float64，再做归一化与混合：
P0 = asarray(P0, float64)
P0[s,a,t] ← P0[s,a,t] / Σ_{t'} P0[s,a,t']          # 先按 axis=2 求和，再整除；就地、float64
sticky[s,a,s] = 1（其余 0）
P = (1.0 − mixing)·sticky + mixing·P0              # float64
R = asarray(R, float64)
R[s,0,t] ← R[s,0,t] + GAP_BONUS                    # float64，仅动作 0
# —— 初始行为策略（在以上三次抽样之后、同一个 rng）：
preferred = rng.integers(0, 3, size=4)
π = full((4,3), 0.15, dtype=float64)
π[s, preferred[s]] = 0.70                          # = 1 − (3−1)·0.15
# π 严格正、行和为 1，dtype float64
```

### 10.3 精确量（审计/起点用）

```
P^π[s,t] = Σ_a π[s,a]·P[s,a,t]
r_sa[s,a] = Σ_s' P[s,a,s']·R[s,a,s']
r_π[s] = Σ_a π[s,a]·r_sa[s,a]
v^π = solve(I − γP^π, r_π)
Q^π[s,a] = r_sa[s,a] + γ·Σ_s' P[s,a,s']·v^π[s']
μ：A_μ = (P^π)ᵀ − I；A_μ 末行 ← (1,1,1,1)；b = (0,0,0,1)ᵀ；
   μ = solve(A_μ, b)；μ ← max(μ, 0)；μ ← μ/Σμ
v*：greedy 初值全 1/3；重复至多 1000 轮：
   q = r_sa + γ·(P 作用于 v(greedy))；best[s] = argmax_a q[s,a]（**取下标最小者**，numpy argmax 约定）；
   new[s,best[s]]=1，其余 0；若 new 与 greedy 逐元素相等则停。v* = v(greedy)。
```

### 10.4 训练批（只用于产生 Q̂）

**与 10.2 共用同一个 rng、顺序消费**：`start = rng.choice(4, p=μ)`；随后 `rollout` 跑 `TRAIN_LENGTH` 步：
每步 `a = rng.choice(3, p=π[s])`、`s' = rng.choice(4, p=P[s,a])`、`r = R[s,a,s']`。

`rollout` 返回 `S, A_, Rew`（长度均为 `TRAIN_LENGTH+1`），其语义：
`S[0]=start`；`Rew[0]=0` 为**占位**；对 `t = 0..TRAIN_LENGTH−1`：
`A_[t] = rng.choice(3, p=π[S[t]])`、`S[t+1] = rng.choice(4, p=P[S[t], A_[t]])`、`Rew[t+1] = R[S[t], A_[t], S[t+1]]`。
**末项（预审第 4 条的修复，必须钉死）**：循环结束后**再抽一次**动作
`A_[TRAIN_LENGTH] = rng.choice(3, p=π[S[TRAIN_LENGTH]])`——这是**消费 RNG 的确定调用**（使环境 rng 流与基线一致；
本流此后不再被使用，认证批另起独立流）。`Rew[TRAIN_LENGTH]` 保持占位 `0`，`S` 不再有后继。

**存储字段**（长度均为 65536）：
`states = S[0:65536]`；`actions = A_[0:65536]`；`rewards = Rew[1:65537]`；
`next_states = S[1:65537]`；`next_actions = A_[1:65537]`。
即第 `t` 条记录项 = `(S[t], A_[t], Rew[t+1], S[t+1])`。
**`next_actions` 的全部定义**：`next_actions[t] = A_[t+1]`，含 `next_actions[65535] = A_[65536]`（即上句那次额外抽取）。
**它进入训练批的内容哈希**（与基线封存格式一致），但**本任务两条数组路线与两条网络路线都不读取它**；
把它列入哈希仅是为了与基线参考产物可逐位比对，不构成任何判决输入。

### 10.5 认证批（每记录一步一批；与 10.2/10.4 的 rng 流无关）

`seed_parts = [SEED, SALT, round(100·mixing), task_index, step]`；`rng = default_rng(seed_parts)`；
分块 `CHUNK=32768` 链。**钉死的调用序列**（每块、每块大小 `size = min(32768, 剩余链数)`）：

```
policy_cdf = cumsum(π, axis=1)；policy_cdf[:, -1] = 1.0
current = rng.choice(4, size=size, p=μ)            # 1 次 choice
对 step = 0..63：
    u1 = rng.random(size)
    a = Σ_j 1{u1 > policy_cdf[current][j]}（clip 到 [0,2]）
    u2 = rng.random(size)
    # 转移 CDF 每步重算：rows = P[current, a]；cdf2 = cumsum(rows, axis=1)
    s' = Σ_j 1{u2 > cdf2[j]}（clip 到 [0,3]）
    记录 (current, a, R[current,a,s'], s')；current ← s'
```

链在批内按 `链号×64+步` 排布。`next_actions = actions`（同一份数组）。

### 10.6 首访压缩

对每 `(链, 对 x)`：保留该链**首次**访问 `x` 的那一项；`counts[x] = 访问过 x` 的**不同链数**；
`max_chains`（A/C 取 16384，B 取 65536）意为只用前 `max_chains` 条链。保留项按链序排序。

### 10.7 残差

`Y = R[s,a,s'] + γ·Σ_b π[s',b]·Q̂[s',b] − Q̂[s,a]`（对压缩后的每条记录项）。

### 10.8 frozen 包络证书（**非 L12**）

```
若 max|Q̂| > 5        → 弃权，原因 divergence_guard_triggered
若 Q̂ 含非有限值       → 弃权，原因 numerical_nonfinite
若任一 N_x < 2000     → 弃权，原因 heldout_pair_support_missing
（按此顺序逐个判；若某对先触发支持度不足即停止检查后续对）
δ_dir = DELTA_STEP/(2·12)；lt = log(2/δ_dir)
每对：mean_x、var_x（ddof=1）
radius_x = sqrt(2·max(var_x,0)·lt/N_x) + (7/3)·20·lt/max(N_x−1, 1)
eps_x = |mean_x| + radius_x；E_Q = max_x eps_x/(1−γ)
```

弃权原因序列化规范次序（**本协议可产生的全部原因**）：`(divergence_guard_triggered, heldout_pair_support_missing, numerical_nonfinite)`。
**本协议不产生 `pair_count_mismatch`**（它属于已废止的固定计数协议）；
任何记录若出现该字符串或其它未列原因 → **协议异常，停止该路线并记录**。

### 10.9 判决

候选：`π_η^+(s,a) ∝ π(s,a)·exp(η·Q̂(s,a))`，行归一化（实现：logits = log π + ηQ̂，减去行最大值后 exp、归一化）。
`Δπ = π_η^+ − π`；`LB_s = Σ_a Δπ_s(a)·Q̂(s,a) − E_Q·‖Δπ_s‖₁`。
- **A/B（合取）**：降序遍历网格，**首个**满足 `min_s LB_s > 0` 的 `η` → 整格更新为该候选（每行都用同一个 η）；否则弃权。
- **C（按状态）**：每行独立取首个 `LB_s > 0` 的 `η`；无则该行不动；`η` 记录含"不更新"标记。
- 证书弃权（无 `E_Q`）时不做判决，直接弃权，原因取证书的原因序列。
- **某格首次未发出后，该轨迹停止**（不再继续后续步）。

### 10.10 生产者（预审第 1 条的修复：完整内联，含 token 集合/轴/mask/null token/dtype/更新顺序）

**共同结构**：`q` 初值全 0，重复 `LAYERS=160` 层；每层用**同一个训练批**（§10.4），
**不读认证批**。生产者**无发散守卫**；若任一层 `max|q| > VALUE_BOUND=5`，**停止整条路线并记为协议异常**
（这是协议级停止，不是静默发散）。

**A. 数组-精确（expected_exact，float64）**：
每层：`qbar_t = Σ_b π[s'_t,b]·q[s'_t,b]`；`δ_t = r_t + γ·qbar_t − q[s_t,a_t]`；
`count_x` = 训练批内对 `x` 的出现次数；`q[x] += α·(Σ_{t: x_t=x} δ_t)/count_x`（仅 `count_x>0` 的对动，其余不动）。

**B. 数组-有限（expected_finite，float64）**，每层三个算子（`d=12`）：
- 读：`read_x = (e^ξ·q_x + (Σ_y q_y − q_x))/(e^ξ + d − 1)`；
- 后继：`succ_s = softmax over tokens y=(u,b)，score = ζ·1{u=s} + log π(b|u)`，作用于 `q`；
- 残差：`δ_t = r_t + γ·succ[s'_t] − read[x_t]`；
- 写回：`N_x` = 该批命中对 `x` 的转移数；`denom_x = N_x·e^τ + (m − N_x)`（`m` 为转移总数）；
  `q[x] += α·(e^τ·Σ_{t:x_t=x}δ_t + (Σ_all δ_t − Σ_{t:x_t=x}δ_t))/denom_x`。
  **无 visited gate：未访问对也收到非零泄漏更新。**

**C. 网络-掩码精确版（float32）**。token 集合 = `d` 个 pair token（记忆值 = `q` 展平）**+ 1 个零值 null token**。
每层、对当前训练批：
- **读**（精确单例检索）：`current_q[t] = q_flat[x_t]`（等效于对 score `0`（命中）/`−inf`（不命中）做 softmax 得 one-hot）。
- **后继**（直接构造的动作期望头）：`attention[t,(u,b)] = 1{u = s'_t}·π(b|u)`；`qbar_t = Σ_{u,b} attention·q_flat`。
  **说明**：合法输入下它等于"只允许下一状态动作 token、logit 为 log π"的 masked softmax，但**本身不算 logits、不调 softmax**。
- **残差**：`δ_t = r_t + γ·qbar_t − current_q[t]`。
- **写回**（轴 = 来源轴，即"转移 + null"）：来源集合 = `{t : x_t = x} ∪ ({null} 当无 t 命中 x)`；
  score 在允许来源上为 `0`、其余 `−inf`；`attention = softmax`（沿来源轴）；
  `write_x = Σ_sources attention·value`（value 为 `δ_t`，null 为 0）；`update_x = α·write_x`。
  即：已访问对得 `α·mean{δ_t : x_t=x}`；未访问对得 `0`（只读 null token）。
- `q_flat += update`；reshape 为 `(4,3)`。

**D. 网络-有限版（float32）**。token 集合 = `d` 个 pair token，**无 null、无 mask、无 visited gate**。
每层：
- 后继：`score[t,y] = log π(b_y|u_y) + ζ·1{u_y = s'_t}`；`attention = softmax over y`；`succ_t = Σ_y attention·q_flat[y]`。
- 读：`score[t,y] = ξ·1{y = x_t}`；`read_t = softmax over y 作用于 q_flat`。
- 残差：`δ_t = r_t + γ·succ_t − read_t`。
- 写回：`w_t(x) = exp(τ·1{x_t = x}) / (N_x·e^τ + (m − N_x))`；`update_x = α·Σ_t w_t(x)·δ_t`。
- `q_flat += update`。

**dtype 与更新顺序**：网络版全程 `float32`（输入 `states/actions/next_states` 转 int64，`rewards/policy` 转 float32）；
`q` 从 0 起每层整体同步更新（同层内所有量用**同一** `q`）。两个类**无 trainable parameters**。

### 10.11 度量与配对

每（路线,臂,生产者,记录）轨迹：`emitted`、`states_updated`、选中 `η`（含标记）、`E_Q`、逐步 `v` 链、
`total_gain = Σv_final − Σv_initial`、`closure = total_gain/initial_suboptimality`
（`initial_suboptimality = Σv* − Σv^π0`；若 `≤ 0` 则记 `1.0`——本族未发生）。
**环境是统计单元**；路线作配对比较。配对比较定义见 §2 冻结定义（win/loss/tie，`1e-9` 容差）。

## 11. 起草、异议与预审记录

- 2026-09-15：Claude 按用户直接指令起草 **v1**（例外，见文首声明）。**未运行任何实验**；附录 10 的每式都对过基线源码。
- 2026-09-15：**预审 OBJECTION**（五项阻断）：
  1. §10 非完全自足（回指外部文档；网络算子的 token 集合/softmax 轴/mask/null token/dtype/更新顺序未完整定义）；
  2. 环境与训练 RNG 的逐位定义有缺口（float32 转换时点、P0 归一化公式与 dtype、μ 末行右端、v* 平局规则、Rew 语义、π 的完整定义）；
  3. G2 判定状态不封闭（无 spec/oracle 不一致时的决策树与终态）；
  4. R3 的 `e_q` 未定义（与 §10.8 的 `E_Q` 关系及序列化格式不明）；
  5. R1 配对条件语法歧义（`C > A ≥ 40/48`）。
  **v2 逐项修复**：1 → §10.10 全量内联、§9 自足化、§6 高精度复核自足化；
  2 → §10.2/10.3/10.4 逐字钉死（含 dtype、次序、末行方程、argmax 约定、Rew 索引语义、π 数组）；
  3 → 新增 §2.1 封闭决策树（四个终态）；4 → 新增 §6.1 字段/序列化对照表（`e_q` ≡ `E_Q`）；
  5 → §2 改为显式 `count(...)` 计数并冻结 `1e-9` 平局容差。
- **预审缺口仍在**：起草方即 Claude；GPT 额度不可用。v2 须由 **GPT（恢复后）或用户**重新预审；
  通过前本任务单**不构成执行授权**，任务保持 `REVIEW`。（随后发生第二轮预审，见 §12。）

## 12. 第二轮预审记录（v2 → OBJECTION → v3）

- 2026-09-15：**第二轮预审 OBJECTION**（确认 v1 五项已实质修复、基线哈希正确，但新发现五项缺口）：
  1. §2 R1 第一项 `count` 与"均值差"混用 → **v3 改为 `mean(closure_C) − mean(closure_A) ≥ 0.40`（显式分数单位）**；
  2. §2.1 决策树非全集：缺"整数相同、浮点差 >1e-12"分支，缺"双序重算都无法复现各自产物"终态 →
     **v3 补齐为四步流程、五个终态**（含 `BENIGN-ROUNDING`/`BENIGN-MATERIAL` 分流与 oracle 产物重跑要求）；
  3. `pair_count_mismatch` 有规范次序却无生产规则 → **v3 从可产生原因集中删除，并规定出现即协议异常、停路线**；
  4. 训练批 `A_[TRAIN_LENGTH]` 未定义 → **v3 钉死末项抽取（含其 RNG 消费语义）、`next_actions` 全定义及其入哈希但不被任何路线读取的地位**；
  5. "JSON number 按 IEEE 位模式逐位比较"不可复现 → **v3 改为规范序列化 + 封存往返完整性检查 + 解析后 float64 精确相等的可复现定义**。
- 2026-09-15：**预审缺口仍在**：起草方即 Claude；GPT 额度不可用。v3 须由 **GPT（恢复后）或用户**重新预审；
  通过前本任务单**不构成执行授权**，任务保持 `REVIEW`。
- 执行还需 GPT 跑 `codex/FP-INDEP-001` 路线；**两条路线绝不由同一执行者承担**。
