# FP-SURVIVOR-REVIEW-001：第二轮三项独立审阅
日期 2026-09-14；状态 VERIFYING（审阅完成，发现待作者处理）；冻结基线 325140bee7b51455fa7c3c42458fb9825d34a611。
用户直接授权继续审阅 docs/2026-09-13-third-party-review-brief.md 的第二轮追加范围。
输入：C:/Users/Admin/Desktop/research/icrl_softmax/docs/2026-09-13-third-party-review-brief.md 追加节；原仓库 AGENTS.md、ACTIVE_WORKSPACE.md；追加节指向的任务、推导、代码及本机 results 封存。代码基线的隔离只读副本在 results/FP-SURVIVOR-REVIEW-001/codex_worktree。封存数据只在宿主可读。不要读取本次 codex/ 结果或报告；首次意见前保持独立。
范围三项：1 按状态保守规则及 frozen 同认证数据成本比较 FP-COST-001；2 免费 L1/L12 的前提、方向风险、迭代数字；3 FP-NETX-002 仅 frozen 单元格的生产者决策等价与精度声明。
必答问题，每项单列：这里有没有把“由数据算出的函数”送进只能用于固定函数的浓度不等式？逐项说明被平均函数、条件历史/独立性、事件及风险；若没有必须明说通过此专项检查。该检查通过不意味着其它验收自动通过。
方法：只读代码、推导和封存，独立数学构造/反证，封存指标检查；不得执行代码修改或新实验，不联网、不发外部消息、不调用子代理。不得把零覆盖违规当概率证明。允许指出文档错误但独立重证当前实际标量输出；须区分当前论证与替代论证。
双方工作：GPT 自行审阅三项并用另写程序重算封存统计、必要的小规模只读验证；Claude 在新上下文中独立只读检查三项，最终通过 stdout 返回完整意见。不允许 Claude 修改任何文件/分支。首次结果后双方交叉复核；不能用同一模型新会话冒称异模型验证。
资源：一次本机 Claude Code 会话约 5–15 分钟；GPT 轻量复算，最长约 30 分钟。本次不重抽 8.4e9 项完整阶梯、不改冻结方法或开展新大矩阵。
产物：GPT 归档 stdout 到 results/FP-SURVIVOR-REVIEW-001/claude/read_only_review.md；GPT 报告与可复现脚本在隔离 codex/FP-SURVIVOR-REVIEW-001 分支 docs/research_branches/FP-SURVIVOR-REVIEW-001/codex/，计算输出在宿主 results/FP-SURVIVOR-REVIEW-001/codex/。
验收：先给此审阅任务预审 APPROVED 或 OBJECTION；APPROVED 后进行授权只读审查。每项分别给 PASS/FAIL/OBJECTION，列文件路径行号、推导/数值、限定范围与未验证项。不得因一项 FAIL 停止其它独立审查。若任务级异议，受影响范围 BLOCKED_BY_OBJECTION，等待用户裁决。最后一行限 PASS/FAIL/OBJECTION。
研究问题/可证伪命题：三项在排除 L12S 后是否仍有完整前提链与正确计数、成本及概率口径？失败判据包括数据复用破坏条件化、风险超支且无正确论证、封存数字不符或由有限对比外推一般结论。停止：权限/登录不可用则记录；禁止升级原任务 VERIFIED。审查记录原文不得被改写。
注意追加节可能有过强断言，按证据审查；FP-PROP-CLOSURE-001 §1 不是本次重启传播方向的授权，最多作为相关口径风险附记。
Git：原分支 claude/FP-CENSUS-001；原未跟踪文件全部保留。远端 SSH 因 known_hosts 读取权限失败，未同步。工作在本地冻结基线。

用户追加约束：不做任何依赖 L12S 有效性的新实验；不做第四次传播尝试，除非从新构造开始且先通过预审。本轮仅审查保留臂；无传播尝试。

## 审阅完成记录
状态：VERIFYING（只读审阅完成，原表述发现待作者处置；不标VERIFIED）。
REVIEW：Claude预审APPROVED；ACTIVE：双方独立检查；VERIFYING：双方首次结果归档，Claude交叉复核GPT幅值证明PASS。
GPT复核Claude初判后，指出其把弱幅值命题误作中心区间；Claude明确撤回L12超支结论。最终仍有成本与统计外推、旧证明口径等FAIL发现，详见 docs/research_branches/FP-SURVIVOR-REVIEW-001/codex/review_report.md。
计算证据：完整封存汇总与1536生产者步骤判决/价值重建；35步高档frozen重放；32步真实生产者frozen重放。未全量重抽高档。用户禁止的L12S实验和传播尝试均未执行。
Claude命令为 -p --permission-mode dontAsk --tools Read,Glob,Grep --allowedTools Read,Glob,Grep --no-session-persistence --output-format text；两次退出码0，stderr为空。归档各原始stdout；GPT报告另述复核修正，不改Claude原文。

