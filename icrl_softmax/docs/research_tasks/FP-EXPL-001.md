# FP-EXPL-001：固定探索采样后的组内平均 Q 迭代

## 1. 元数据与授权

- 日期：2026-09-11。作者：GPT。版本：1.1。状态：ACTIVE。
- 分支：codex/FP-EXPL-001 与 claude/FP-EXPL-001。
- 基线：c710e32b24d77085adea134c3f470773037ffbb1。
- 前序 FP-ITER-001 v1.1 已 VERIFIED；FP-ESARSA-001 保持 DRAFT。
- 设计：docs/superpowers/specs/2026-09-11-fixed-exploration-grouped-iteration-design.md。
- 用户已同意保留组内平均、只借鉴探索采样，且明确选择
  “先采样一批，固定数据做迭代”。两个策略均固定。
- 用户已确认书面方案，并指示“你主要交给claude就可以，你负责验收”。
  本任务据此采用 Claude 主执行、GPT 独立验收；这是针对本任务对
  AGENTS.md 默认双路线完整盲态执行的用户裁决例外，不修改仓库通则。
- v1.1 只改变职责和相应产物，不改变 v1.0 的任何科学输入、公式、
  数值容差、覆盖处理或验收强度。Claude 只读预审现已 APPROVED，
  本提交发布 ACTIVE，作为 Claude 主执行与 GPT 验收的共同基线。

## 2. 研究问题与可证伪假设

连续环境轨迹替代手工数据后，已有有限 softmax 构造能否在实际完整
覆盖的批次上实现固定策略组内平均迭代，并解释三种误差？

- H1：冻结采样协议的这一批覆盖全部四个状态动作对。
  这是可失败的诊断，不是全支持行为策略直接保证的事件。
- H2：覆盖成立时，独立直接参考与精确分组注意力一致；
  字面有限网络与独立标量有限公式一致。
- H3：三个同输入阶段误差的有符号分解及有限步扰动界成立。
- H4：精确完整批次算子满足 0.85 收缩界；有限算子的收缩充分条件
  是否成立由实际矩阵决定，不能事先假定。
- H5：两个真实状态价值不同；本批经验固定点与真实固定点是否不同
  是报告指标，不预设非零数据误差为成功条件。

允许负面结果。数学或实现错误不得通过放宽容差、换参数或隐藏失败解决。

## 3. 唯一冻结数值协议

CPU float64。状态和动作均为 0、1；状态动作对顺序 00、01、10、11。
奖励只依赖当前状态动作对，无终止状态。

    gamma = 0.7
    alpha = 0.5
    target_pi = [[0.75, 0.25], [0.25, 0.75]]
    behavior_pi = [[0.5, 0.5], [0.5, 0.5]]
    rewards = [1.0, -0.5, 0.25, 0.25]
    P_next_state = [[0.75, 0.25],
                    [0.25, 0.75],
                    [0.5, 0.5],
                    [0.75, 0.25]]
    initial_state = 0
    batch_length = 64
    seed = 20260911
    Q0 = [0.0, 0.0, 0.0, 0.0]
    updates = 64
    xi = zeta = tau = 8.0

环境相对前序只把状态动作对 (1,1) 的奖励，即 rewards[3]，从 0.75 改成 0.25。
目标策略的两状态平均即时奖励因此为 0.625 和 0.25，不可能对应
常数真实状态价值。这是采样前的非退化理由，不涉及新结果。

使用 numpy.random.Generator(numpy.random.PCG64(seed))。
每条转移按顺序消耗恰好两次 rng.random() 标量调用：

1. 抽 u_action；小于 0.5 时 a=0，否则 a=1。
2. 抽 u_transition；小于 P_next_state[2*s+a,0] 时 s_next=0，否则为 1。
3. 保存 (s,a,rewards[2*s+a],s_next,u_action,u_transition)，令 s=s_next。

不得额外消耗该生成器作测试输入，不得补样、洗牌、重置中途状态、
重采样、选择种子或采到覆盖才停。双方独立生成后要求轨迹完全相同。
记录 Python/NumPy 版本。重放同一冻结协议不算补样。

先保存全部 n_x，检查 min n_x>=1。若失败，输出 COVERAGE_FAILURE、
原始轨迹与缺失项，停止该批 Q 迭代并记录失败；不得自动扩大样本。
此时不满足本任务的构造完成条件，后续任务由用户另行裁决。

覆盖通过后冻结该批，从相同 Q0 保存 k=0,...,64 的三条路线：
直接组内平均、精确分组注意力、有限锐度 8 字面网络。
另用独立标量有限公式验证同一有限轨迹，不增加实验输入。
禁止追加初始值、锐度、样本规模扫描或策略更新。

## 4. 算法和网络边界

N=64，x_t=(s_t,a_t)，u_t=s_next,t。只在覆盖通过的批次执行：

    C0[t,y] = 1{x_t=y}
    S0[t,(u,b)] = 1{u_t=u} target_pi[b|u]
    W0[x,t] = 1{x=x_t}/n_x
    d0_t(q) = r_t + gamma sum_b target_pi[b|u_t] q[u_t,b] - q[x_t]
    F0(q) = q + alpha W0 d0(q)

直接参考独立计算分组残差，不调用注意力实现。精确参考可使用声明的
相等性掩码。有限网络不允许内容相等掩码，必须实现：

    C[t,y] = softmax_y(xi * 1{x_t=y})
    S[t,(u,b)] = softmax_(u,b)(zeta * 1{u_t=u} + log target_pi[b|u])
    W[x,t] = softmax_t(tau * 1{x=x_t})
    Ff(q) = q + alpha W (r + gamma S q - C q)

指标函数描述固定 one-hot 点积，不能代替字面注意力构造。
须声明 H、WQ/WK/WV/WO、scaled-dot-product softmax、残差连接和固定
线性/ReLU scratch 清除。允许沿用前序结构。Claude 独立实现本任务
网络及直接/标量参考；GPT 独立重建公式和输入进行验收，不导入 Claude
实现的算子作为审核计算依据，也不需要重复撰写一套完整网络。

允许静态角色/位置掩码、prompt 初始化和固定位置输出提取。
动态 Q 的读取和写入必须经过字面矩阵；不允许外部索引 Q 代替网络、
有限网络访问门控、内容硬掩码、依赖访问次数的额外修正或隐含 oracle。
权重只可依赖维度、固定 target_pi、gamma、alpha 和锐度；
不能依赖样本结果、奖励值、Q 或真实环境转移概率。
奖励和已观测转移可以作为 prompt 数据，不能硬编码进权重。
每轮仅携带 Q；不可变字段保留，scratch 用固定映射清除。

后继动作平均使用 target_pi，不能换成 behavior_pi。
P 只供环境采样器和独立审计器；q_pi 只供审计，不输入网络。
不实现论文整批时间加权，不直接继承其收敛/样本复杂度保证。
不训练参数、不做 policy improvement、新数据逐轮输入或在线控制。
不修改历史代码/结果、用户学习文档、main 或全局工具配置。

## 5. 推导、指标与验证

独立从公式导出 G0,b0,Gf,bf，不拟合轨迹：

    Gf = I + alpha W (gamma S - C)
    bf = alpha W r
    c_f = ||Gf||_infinity
    c0 = 1-alpha(1-gamma) = 0.85

rho(Gf) 只作数值诊断。c_f>=1 不代表发散；浮点特征值不能单独证明
精确谱性质、唯一固定点或收敛。
同一个 q 上的阶段分解及完整轨迹扰动界必须检查：

    e_current = -alpha W(C-C0)q
    e_successor = alpha gamma W(S-S0)q
    e_write = alpha(W-W0)(r+gamma S0 q-C0 q)
    Ff(q)-F0(q) = e_current + e_successor + e_write
    E_0 = 0
    E_(k+1) = c_f E_k + ||Ff(q_exact,k)-F0(q_exact,k)||_infinity
    ||q_finite,k-q_exact,k||_infinity <= E_k

计算精确经验固定点 q_hat。独立审计器从真实 MDP 的 Bellman 方程
求 q_pi，并报告 V_pi(s)=sum_a target_pi[a|s] q_pi[s,a] 的状态差异。
报告数据偏差及确定性审计上界：

    ||q_hat-q_pi|| <= ||F0(q_pi)-q_pi|| / (1-c0)

这个上界使用审计真值，不称为学习器可计算的统计证书。
仅在 c_f<1 时求 q_f,infinity 并验证唯一性论证和下列各式：

    ||q_f,infinity-q_hat|| <= ||Ff(q_hat)-q_hat||/(1-c_f)
    ||q_finite,k-q_f,infinity|| <= c_f^k ||Q0-q_f,infinity||
    q_finite,k-q_pi
      = (q_finite,k-q_f,infinity)
      + (q_f,infinity-q_hat)
      + (q_hat-q_pi)

所有范数均为 infinity norm。有符号向量分解是等式，范数相加只给上界。
不适用项写 null 和原因，不能默认逆矩阵存在。不要求实际误差逐步单调。

验证至少覆盖：轨迹连续性/抽样阈值重建、策略保持、精确和有限各自
等价性、四个标准基向量的仿射重建、scratch 清除、不可变字段保存、
阶段误差、全部有限步界、适用时的固定点残差和三项分解。
可用基向量和无效输入作单位检查，不能生成新研究数据。

概率和单步等价绝对容差 1e-12；重复轨迹、求解、分解及界用
1e-10*(1+max_abs(left,right))，不等式将该容差加在右侧。
原始抽样值与离散轨迹双路线要求完全一致；其他比较保存原始偏差。

## 6. 产物、复现和资源

以下路径相对各自工作树的 icrl_softmax。Claude 主执行产物为：

- docs/research_branches/FP-EXPL-001/claude/witness.py
- docs/research_branches/FP-EXPL-001/claude/verify.py
- docs/research_branches/FP-EXPL-001/claude/theory.md
- docs/research_branches/FP-EXPL-001/claude/report.md
- docs/research_branches/FP-EXPL-001/claude/verification_of_other.md
- results/FP-EXPL-001/claude/execution_manifest.json
- results/FP-EXPL-001/claude/results.json
- results/FP-EXPL-001/claude/verification.json
- results/FP-EXPL-001/claude/：失败输出、重放证据与审查辅助文件。

GPT 验收产物为：

- docs/research_branches/FP-EXPL-001/codex/verify_claude.py
- docs/research_branches/FP-EXPL-001/codex/verification_of_other.md
- results/FP-EXPL-001/codex/execution_manifest.json
- results/FP-EXPL-001/codex/：CLI 日志、Claude 输出重放、独立核验结果与失败记录。

GPT 另负责本任务、设计/计划、自身预审记录、handoff.md、synthesis.md
和 ACTIVE_WORKSPACE.md。Claude 预审原文保存在
docs/research_branches/FP-EXPL-001/codex/claude_pre_review.md，保留原始日志。
记录底层模型运行时元数据，不能从 CLI 品牌推断模型身份。

Claude 工作树位置为根项目 icrl_softmax/results/FP-EXPL-001/claude_worktree。
双方从同一 ACTIVE 发布提交开始，各自首次运行前记录提交、任务
SHA-256 和环境到 execution_manifest.json。未发布 ACTIVE 前没有
有效执行基线。Claude 封存自己的首轮代码、理论、报告和原始结果后，
GPT 才进行正式验收。GPT 可在等待期间基于冻结任务编写独立核验公式。
本任务不声称存在两套完整盲态网络实现。

在各自 icrl_softmax 目录执行并保存 stdout 和退出码：

    C:\Users\Admin\anaconda3\python.exe -B docs/research_branches/FP-EXPL-001/claude/verify.py
    C:\Users\Admin\anaconda3\python.exe -B docs/research_branches/FP-EXPL-001/claude/witness.py
    C:\Users\Admin\anaconda3\python.exe -m ruff check docs/research_branches/FP-EXPL-001/claude/witness.py docs/research_branches/FP-EXPL-001/claude/verify.py

GPT 先重放上述三条命令，再从自己的工作树运行：

    C:\Users\Admin\anaconda3\python.exe -B docs/research_branches/FP-EXPL-001/codex/verify_claude.py --results <Claude原始results.json绝对路径>

该路径是执行参数，不是待选择的研究输入。审核程序独立重建冻结种子、
全部轨迹、C0/S0/W0/C/S/W、仿射算子、全部 Q 迭代、误差/界和适用的
固定点，并核对 Claude 保存的字面投影与原始结果。
GPT 另行只读审查源代码与证明，确认没有外部动态 Q 查表或隐含 oracle。

witness 的 strict JSON 包括全部抽样值/轨迹、计数、输入、算子、
字面权重/第一步投影、全部 Q、阶段误差、界、固定点和不适用原因。
verify 输出 strict JSON 检查数、失败项和最大偏差。禁止 NaN/Infinity。
全部实际失败须保留；不能按重放结果选择输出或更换种子。

资源：CPU，一条 64 步数据轨迹，各数值路线 64 次更新，无扫描。
预计 Claude 主执行 30--60 分钟，GPT 验收 15--30 分钟，数值执行各数分钟内。
各研究阶段超过 60 分钟则记录进度与交接，不自动扩充协议。
不新增收费服务、提供方或收费类别。

## 7. 生命周期、验收和停止条件

1. DRAFT：v1.0 已书面提交，用户确认并裁决由 Claude 主执行、GPT 验收。
2. REVIEW：Claude 只读检查问题定义、数学目标、网络边界、公平输入、
   盲态、完整性、容差、资源和失败处理，返回 APPROVED 或 OBJECTION。
3. ACTIVE：仅 APPROVED 后发布；Claude 主执行并以 [claude] 提交封存
   源码/理论/报告，记录未跟踪原始结果的 SHA-256。GPT 在自身分支工作。
4. VERIFYING：GPT 独立重建并重放全部输入、原始输出和结论，撰写
   verification_of_other.md。Claude 在其结果封存后复核 GPT 的审核代码、
   证据和结论，撰写自身 verification_of_other.md。
   两份报告仍必须有可检查证据并以 PASS、FAIL 或 OBJECTION 结束。
5. VERIFIED：无未裁决异议、覆盖通过、H2/H3 及适用的 H4 有证据、
   H5 如实评价、双向 PASS 和可复现产物齐全、差异已解释、
   ACTIVE_WORKSPACE.md 已更新。任何 main 合并另需用户批准。

覆盖失败不满足构造完成条件。有限算子未通过收缩充分检查、或数据偏差
为零，可以作为科学负面结果接受，但须验证其余等价性、有限步界并记录
不适用项；不得把“充分条件不成立”扩大为“不收敛”。

实现/推理失败且定义有效时返回 ACTIVE，由原作者修复；定义异议进入
BLOCKED_BY_OBJECTION，受影响工作停止等待用户裁决。
重采样、结果驱动改参数、隐藏 oracle、未声明门控、未记录失败或缺少
独立验证都不能通过。禁止直接合并或推送。

Claude 登录/额度/权限/环境不可用时记录真实阻塞，不能用 Codex 子代理
替代。不修改全局 memory、安全设置或提供方配置。需额外授权时，先保留
可审查文件和具体范围，再说明阻塞。

交接须记录分支/提交/任务版本、已完成/运行中/未完成事项、命令/输出、
不变协议、Claude 准确下一步及 GPT 恢复复核项目。未验证结果保持初步。

## 8. 当前证据和裁决

- 2026-09-11：用户同意保留组内平均，只借鉴探索采样，并明确选择固定一批。
- 64 步、种子、奖励变更、Q0 和锐度是 GPT 事前提出的具体协议；
  本任务尚未产生任何随机轨迹或实验指标。
- 前序验收记录已封存在 c710e32，新任务在 codex/FP-EXPL-001。
- 远端读取因 SSH known_hosts 权限失败，未同步或改 SSH 配置。
- 三份未跟踪用户学习文档和既有 tmp 内容保留，不纳入新任务。
- 尚无 Claude 预审、执行或交叉验证，不使用 VERIFIED 表述。
- 用户随后确认书面方案并指示“可以你主要交给claude就可以，你负责验收”。
  GPT 据此发布 v1.1，DRAFT -> REVIEW。豁免本任务双路线完整盲态执行，
  保留 Claude 预审、原作者修复、GPT 独立验收及 Claude 对验收证据复核。
  本次角色裁决不是数值协议变更，不授权合并、推送或新收费类别。
- 预审启动尝试被 Codex 自动审批拒绝，未创建执行进程。拒绝理由：
  用户已授权主要交给 Claude，但尚未明确授权将本任务私有文档发送到
  现有 HTTPS api.kimi.com:443 目的地。未重试、换工具或发送材料。
  已准备的载荷为 AGENTS.md、本任务、设计、计划及 ACTIVE_WORKSPACE.md
  的本任务当前段落，保存在 results/FP-EXPL-001/codex/pre_review_prompt.txt。
  任务仍为 REVIEW；这是外发授权阻塞，不是 Claude 的任务定义 OBJECTION。
  书面方案与执行分工的批准保持有效。
- 用户随后直接回应列明目的地和载荷的授权请求：“继续”。
  该请求明确列出向现有 api.kimi.com 发送本任务治理文档、任务与设计、
  相关前序构造资料，以及本任务代码和结果，用于预审、执行和验证，并
  告知私有材料由 Kimi 服务处理。此回应授权上述限定传送及必要续接，
  解决先前的外发授权阻塞；不扩大到无关文件、新提供方/收费类别、
  全局配置变更、推送或 main 合并。科学协议 v1.1 不变。
- 只读预审正常完成（工具列表为空、permissionMode=dontAsk、模型元数据 k3），
  共一轮，约 200 秒，返回 APPROVED。原文：
  docs/research_branches/FP-EXPL-001/codex/claude_pre_review.md；
  原始日志：results/FP-EXPL-001/codex/claude_pre_review.raw.jsonl。
  期间因暂时没有输出检查过进程；它在任何停止操作发生前已正常退出，
  没有中止或重复调用。未运行样本或实验。
  本提交发布 REVIEW -> ACTIVE；科学协议与用户裁决不变。
