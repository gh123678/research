# FP-COMPOSE-REVIEW-003：补齐定理、全量输入/网络重放、扩维算术与远端核查

- 起草：GPT；日期 2026-09-15；v1.1；状态 VERIFIED（GPT 执行完成，Claude 对完整最终产物交叉检查 PASS）。
- 基线：bcec1fe（正式执行前记录完整哈希）；审查对象 FP-COMPOSE-001/002 现存 Claude 产物。
- 用户直接要求补齐上一轮四个缺口；这是对既有封存结果的事后独立验证，不是新的盲态发现实验。GPT 已读过作者结果，不得声称盲态。

## 问题与可证伪命题
T：逐项独立重证 T1–T5、首访随机容量、L12 单侧幅值界、Bellman 残差界、条件性逐分量不退化；追踪实际实现依赖。若前提或推理有反例即 FAIL，不以数值重放支持定理。
R：从冻结种子重建全部 96 条记录训练输入与 1136 个实际唯一认证批，独立重写采样/首访、证书、决策、状态空间真值核；调用原 model.py 的被测网络重跑全部 2267 次前向、两个生产者全部 4534 步。核对冻结包及自身滚动策略轨迹。
D：从封存数据独立计算扩维锚点、指数、半径比值、网格门限与链数倍数，分开评判算术复现和统计/因果外推有效性。
G：只读查询真实远端 refs，对照本地；不得推送、同步覆盖或合并。

## 冻结输入、方法与验收
输入是宿主 icrl_softmax/results/FP-COMPOSE-002/claude/formal/task_results.json 与 input_manifest.json；执行前 SHA256 封存。上游 001 的任务单在其 codex_worktree 内，保留来源路径及内容哈希。
允许共享定义待测对象的 model.py、数组估计器、MDP/训练生成器；新重放控制器不得导入 Claude 新增 evaluate.py/verify.py/analyze.py/dimension_scaling.py。认证采样与首访独立实现，保持冻结 PRNG 抽数顺序以便逐位对照；共享 PRNG 是重放依赖而非独立随机性证明。
默认同环境 torch 6 线程；若为资源效率使用单线程，必须分别记录，先对固定首个输入做 1/6 线程一致性桥接。不得把线程不同当作逐位承诺。
门槛沿用原任务：Qhat 重放差 <=1e-9，EQ/价值 <=1e-10；所有发出/弃权、逐行 eta 必须一致；哈希/计数精确。阈值仅用于复现，任何真实负安全余量/价值减退不被容差放行。出现异常先封存定位，停止受影响重放并原输入高精度诊断，不改种子、K、预算或规则。
T、R、D 算术、D 推论、G 分开报告 PASS/FAIL/OBJECTION；完整记录未完成量。任何定义缺陷提交 OBJECTION；不得自动升级原任务 VERIFIED。

## 分工与独立性
GPT 在 codex/FP-COMPOSE-REVIEW-003 实施全部补审、写独立程序与报告。Claude 仅只读预审任务和最终交叉检查报告/代码，输出到 stdout，由 GPT 原样保存；不改代码，不启动新实验。原 Claude 路线已封存，本轮是验证该路线，无需重新组织双方盲态发现。
Claude 登录/额度/权限不可用则记录阻塞并通知用户；不以 Codex 子代理替代。

## 产物、资源、停止
代码与报告：本工作区 icrl_softmax/docs/research_branches/FP-COMPOSE-REVIEW-003/codex/。
输出：本工作区 icrl_softmax/results/FP-COMPOSE-REVIEW-003/codex/；禁止覆盖作者封存包。
预计 1–2 小时（主要全量网络），上限 3 小时墙钟、2GB 新输出；不新增付费类别，不用外部计算。超限记录已完成/未完成并交接，不降低矩阵。数据读取、原论文核查与远端查询均为只读。
完成后保存命令、哈希、环境、所有异常、分层判定，并更新本隔离工作区 ACTIVE_WORKSPACE.md；不合并 main。

## 审查与证据记录
2026-09-15：GPT 起草并送 Claude 只读预审；尚未执行新重放。
2026-09-15：Claude CLI 只读预审返回 OBJECTION：O1 要求把作者新增程序的枚举导入禁令改成目录级禁令；O2 要求统一“001/002 审查对象”与只列 002 数据的输入范围。O3 建议明确线程桥接失败回到 6 线程；O4 建议补自身哈希、D 输入与授权记录。GPT 不擅自裁决，R/D 停止等待用户；T/G 为未受影响只读项继续。

## v1.1 用户裁决与生效修订（优先于上述 v1 对应条款）
2026-09-15：用户对 proposed_ruling.md 所列四条具体修订回复“批准”。GPT 现发布 v1.1 并恢复执行。基线完整值 bcec1fe4f8ef49b2d1835f706534b97d8a564016。
1. 禁止直接或间接导入 docs/research_branches/FP-COMPOSE-001/claude/ 和 FP-COMPOSE-002/claude/ 下任何程序。仅共享 model.py、基线数组估计器、MDP/训练生成器及必要基线依赖；记录实际模块与 SHA256。认证采样、首访、证书、门限、判决、真值核独立实现。
2. 增补 001 的 formal/task_results.json、input_manifest.json、verification/selfcheck_c1.json（按实际名称记录）并冻结哈希。先核对 001/002 前四步的输入/元数据/输出；相同输入下 002 前四步重放同时对照两套封存包，不重复计作两次独立运行。不一致则记录并停下受影响路线。正式覆盖仍为 002 全 96 记录、1136 唯一认证批、2267 网络前向、两生产者合计4534步，兼核001的1536步。
3. 默认 6 线程。若单/6线程桥接差>1e-9则必须回到6线程；后续单线程差异也不能直接豁免。可以全程6线程。
4. 增补 dimension_scaling.json/md 源哈希，执行前冻结本任务单、代码哈希与命令。Claude 最后检查为对 GPT 审阅的交叉检查，不是对原 Claude 实验的独立佐证。
其余冻结协议与资源上限不变。不修改原作者代码，不合并 main。

## 执行完成与验证门槛

2026-09-15：T、R、D 算术及 G 已按 v1.1 完成；完整命令、来源哈希、异常记录和逐层判定均已存入 `docs/research_branches/FP-COMPOSE-REVIEW-003/codex/` 与 `results/FP-COMPOSE-REVIEW-003/codex/`，最终分层结论见 `final_report.md`。其中扩维推论为 FAIL，已据此撤回越界表述；该 FAIL 不改变本任务定义，故任务依法从 ACTIVE 进入 VERIFYING。

在 Claude 针对**完整** R 结果、最终报告和撤回边界给出可检查的 PASS／FAIL／OBJECTION 前，本任务不得标记 VERIFIED。即使本审查本身最终通过，也不得自动升级 FP-COMPOSE-001 或 FP-COMPOSE-002 为 VERIFIED。

## Claude 最终交叉检查与收束

2026-09-15：在用户明确授权的只读外部审查中，实际 Claude Code 对完整产物给出 **PASS**，无 OBJECTION。原始 stdout 已逐字归档为 `docs/research_branches/FP-COMPOSE-REVIEW-003/codex/claude_final_cross_review.json`（SHA256 `fd824f1816992228cead897e58890db9d07bad6cdac1b244b2173f0de89bf0aa`）。其逐层结论为 T PASS、R PASS、D 算术 PASS、D 推论对“扩维外推 FAIL”之撤回 PASS、G PASS；并明确本次是对 GPT 补审的交叉检查，不能升级 FP-COMPOSE-001/002。

Claude 记录的非阻断保留包括：`theory_review.md` 的三处时效/措辞、R 结果缺机器可读 limitation 字段、共享模块哈希的事后记录、严格 `>1e-10` 重放闸的假阴性风险、001/002 共有错误盲区，以及 Claude 未能亲自重算 SHA256／读取宿主封存包原件／查询远端。这些均已披露或限定，未改变验收判定。GPT 复核该 PASS 与冻结任务的短任务分工一致，故本审查任务从 VERIFYING 收束为 VERIFIED；原任务状态保持不变。

## 补充审计脚本归档（不改变结论）

2026-09-16：为清理宿主工作树，三个曾位于 `icrl_softmax/tmp/` 的只读核对脚本原样副本归档至 `docs/research_branches/FP-COMPOSE-REVIEW-003/codex/supplemental_audits/`。它们是事后诊断辅助材料，不是冻结执行器、独立验证路线或新增验收证据，故不改变本任务的状态与结论：`quantify_range_looseness.py`（SHA256 `e89fa1748ee9be064f8605d3585e966b2760be9a665e3eb26b0f8006aab7929c`）、`tail_delta_distribution.py`（`bd0bfabb4b24b2b899a3f4907cfc60fcceee7beb7ef7bbbdd4ca47d254b8cf2`）和 `verify_review_claims.py`（`43587c7b865f44eb2cdccc97d3635fa09d34beb2c9ffee108dd93a47a4c455fa`）。
