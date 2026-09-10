# FP-ITER-001 v1.1 read-only pre-review

- Date: 2026-09-10. Verdict: APPROVED.
- Reviewed task SHA-256 (pre-activation bytes): F3C0885F9DBA65798B5F8DDC9ECC4BF9AE5096CF07A617947B296689AB3561EF.
- CLI: Claude Code 2.1.138, existing configuration; Read-only built-in tool set,
  dontAsk permission mode, no MCP servers, no slash commands, no session persistence.
- CLI usage metadata identifies `kimi-k2.6` and `k3`; provider is not independently
  verified and no claim of an Anthropic model is made.
- Raw full stream and stderr: results/FP-ITER-001/codex/pre_review_v11.raw.jsonl
  and pre_review_v11.stderr.txt. Prompt: pre_review_v11_prompt.txt in that directory.
- Below is the returned review verbatim. No scientific implementation existed
  at the time of this pre-review. Activation changes lifecycle metadata only.

女仆已完成四份指定文件的只读审阅（仅使用 Read，未做任何修改、未读取其他 actor 的代码或结果）。以下是对 FP-ITER-001 v1.1 的独立预审，不把 v1.0 的历史批准视为有效，逐项独立核对。

## 预审报告：FP-ITER-001 v1.1（REVIEW 状态）

### 逐项检查

**1. 输入与协议冻结 — 通过**
- 策略 π、两组初始 Q（Q0_Z 零初始化、Q0_A 非平凡初始化）、完整批次 C（6 行逐行列出）与缺失批次 M（明确定义为 C 的前 5 行、缺失 pair 11）、γ=0.7、α=0.5、64 步（k=0..64 全部保存）、两个锐度三元组 (0,0,0) 与 (8,8,8) 联合冻结、pair 顺序 00,01,10,11、CPU float64、无随机种子、无提前停止/调参。v1.0 遗漏的全部冻结项均已补齐。人口 MDP（r、P、q_pi）明确标注为 audit-only，不进入网络输入，无采样解释。

**2. 同步固定策略评价 — 通过**
- F0 与 Ff 均为对同一输入 q 的同步仿射更新，策略全程冻结，范围明确限定为 policy evaluation，不含策略改进、在线控制或证书主张。与 ACTIVE_WORKSPACE 中"不激活 FP-ESARSA 矩阵"的边界一致。

**3. 字面注意力可行性与掩码声明 — 通过**
- 有限算子的分数 ξ1{x_t=y}、ζ1{u_t=u}+log π(b|u)、τ1{x=x_t} 均为声明的 one-hot 身份特征的静态点积，不依赖动态 Q；动态量仅经 value 向量传递。固定 WQ/WK/WV/WO、残差连接与固定前馈足以实现，存在可行构造。掩码边界清晰：有限网络只允许静态角色/位置掩码且不得依赖 pair 相等性；精确参考路线允许显式声明的相等/缺席 null 掩码且不计入有限网络能力。掩码声明无模糊空间。

**4. 数据/注意力/误差分离 — 通过**
- e_current（当前对读取）、e_successor（后继路由/跨状态泄漏）、e_write（写入聚合）三项在同一输入 q 处有符号定义；批次差异（q_hat − q_pi）作为独立的 audit 项，不混入注意力误差；条件动作权重恰为 π（误差报零至舍入），跨状态泄漏单独报告。设计文档决策门第 3 条与之一致。

**5. 有符号望远镜恒等式 — 通过（已独立验证代数）**
- 女仆手工展开验证：e_current + e_successor + e_write = αW(r+γSq−Cq) − αW0(r+γS0q−C0q) = Ff(q)−F0(q)，作为向量恒等式精确成立；任务单正确声明范数层面仅为三角不等式上界而非范数等式，且禁止"从有符号望远镜得出范数恒等"。

**6. 缺失对收缩区分 — 通过（已独立验证收缩因子）**
- 完整批 C 覆盖全部 4 个 pair；G0 各行非负（对角元 1−α+αγ·(非负) ≥ 0），行和恰为 1−α(1−γ)=0.85，故"at most 0.85"成立且实为精确值。缺失批 M 中 pair 11 的 W0 行为零、坐标保持不变，任务单正确声明严格全表收缩失效、禁止虚构未观测转移或声称唯一全表经验不动点；条件于保留坐标的数值须明确标注。区分处理正确。

**7. 扰动界与稳态界有效性 — 通过（已独立验证推导）**
- 有限时界 E_{k+1} = c_f·E_k + ‖Ff(q_exact_k)−F0(q_exact_k)‖_∞，E_0=0，是标准的仿射扰动递归，使用诱导 ∞-范数 c_f 有效；c_f≥1 时界仍成立（仅变松），任务单要求报告松弛度。稳态界 ‖q_f_inf−q_hat‖ ≤ ‖Ff(q_hat)−q_hat‖/(1−c_f) 在 c_f<1 时为标准结论；‖q_finite_k−q_hat‖ ≤ c_f^k‖Q0−q_f_inf‖+‖q_f_inf−q_hat‖ 为收缩加三角不等式，成立。

**8. 不动点警告 — 通过**
- 明确声明 c_f<1 仅为唯一不动点与几何收敛的充分条件，c_f≥1 不证明不稳定；谱半径须另行计算，更强谱论断需单独论证；奇异/非收缩情形返回带原因的 unavailable，禁止静默求逆或声称唯一性。"单纯充分收缩检验失败不是发散证据"写入 H5，方向正确。

**9. 容差 — 通过**
- 单步等价与归一化概率用绝对容差 1e-12；重复轨迹、线性求解、分解与界检验用 1e-10·(1+max_abs)。对 float64 规模（4 维仿射、64 步）合理且不过松；要求同时报告原始误差与 pass/fail，严格 JSON 禁止 NaN/Infinity 字面量。

**10. 独立 actor 范围与盲执行 — 通过**
- 双路线（codex/claude 各自目录与 results 子目录）、共同激活 commit、首结果密封前互读禁令、双向可执行验证（逐字引用 AGENTS.md 长任务规则）、明确排除共享科学模块修改、前任任务的委托诊断不构成对本任务双路线规则的豁免。产物清单与复现命令（python -B、ruff）完整。

**11. 停止规则与负面结果 — 通过**
- 允许可复现的科学负面结果通过；禁止检查数据后重调锐度或更改夹具；实现失败回 ACTIVE 由原作者修复；任务定义异议立即 BLOCKED 等用户裁决；禁止 Codex 子代理冒充 Claude；无 push/merge/main 授权。与 AGENTS.md 一致。

### 观察项（非阻断）

- **CLI/后端元数据条款**：任务单末尾要求预审证据识别实际 CLI/后端元数据、不得以可执行文件品牌推断模型提供方。女仆如实披露：本会话运行于 Claude Code CLI，环境元数据报告模型标识为 "k3"；女仆无法超出所报告元数据独立核验后端提供方，此处仅作事实陈述。该条款本身可由执行阶段的 manifest 满足，不构成任务定义缺陷。
- 任务单与设计文档对 M 的描述（"C 的前 5 行" / "missing-pair subset"）一致；C 确实含自环（(0,0)→0）与重复行，验证器相应检查有依据。

### 结论

女仆独立复核了冻结完整性、同步语义、注意力构造可行性、误差分离、望远镜代数、收缩因子（0.85 确为精确值）、扰动与稳态界推导、不动点充分性警告、容差与治理生命周期，未发现阻断性缺陷。任务定义自洽、可证伪、允许负面结果，符合 AGENTS.md 的全部要求。

APPROVED
