# FP-L12S-REVIEW-001：用户指定两份材料的只读送审
日期：2026-09-14。状态：REVIEW。
用户授权：送审 FP-BOUND-002-l12m-withdrawal-and-split-repair.md（证明修复）与 FP-RERUN-001（撤回后重建的口径）。
冻结基线：05f59890f9d00e3de831beea1c13431a5fa11d17。只读原始仓库 C:/Users/Admin/Desktop/research。
审查材料：
1. icrl_softmax/docs/derivations/FP-BOUND-002-l12m-withdrawal-and-split-repair.md
2. icrl_softmax/docs/research_tasks/FP-RERUN-001.md
3. 第二份的结果报告 icrl_softmax/docs/research_branches/FP-RERUN-001/claude/first_result.md
研究问题：修复是否真正闭合证明，撤回后的任务和结论是否由证据支持？可证伪命题：修复每一步均满足所引浓度不等式条件；重建口径未超出冻结设计和封存证据。
范围：只读审查，不执行修复或重跑实验，不更改任务定义、文件、分支和结果；不联网、不发送外部消息。可读取上述材料依赖的本地代码、推导、封存 JSON。不得启动子代理。资源：一次本机 Claude Code 会话，约 5–15 分钟；如登录、额度、权限不可用则停止记录。
GPT：发起、固定输入、归档 stdout、复核审查意见。Claude：在全新上下文中只读审查两份材料及必要依赖；已有结果由 Claude 编写，不能声称本次替代全部跨模型独立验证。
审查要求：自行推导并检查数据依赖、首访与跨对依赖、逐次传播条件化、风险方向和总账、代码与证明一致性；核对重跑假设与结果、配对批次、同成本截断、作废数字的用途和结案表述。不要只复述原报告；区分证明不成立与算法必错、测量结果与概率保证。
产物：仅通过最终 stdout 输出中文 Markdown 审查报告，由 GPT 归档到 results/FP-L12S-REVIEW-001/codex/claude_review.md；包含每份材料独立判定、逐项检查、准确文件路径和行号、可检查推导或数值证据、影响范围与用户裁决选项。
验收：任务定义预审输出 APPROVED/OBJECTION；既有证明/结果审查分别输出 PASS/FAIL/OBJECTION；整体最后一行限 PASS/FAIL/OBJECTION。证据不足必须明确，不以零覆盖违规替代理论证明。不做全面实验重放，不声称数据全部复现。
停止：若有任务级异议，报告 BLOCKED_BY_OBJECTION；不提出静默执行的修复。等待用户裁决。
环境记录：原分支 claude/FP-CENSUS-001；原未跟踪内容保留（LITERATURE_ALERTS.md、_audit_claims.py、_audit_claims2.py、_backups/、_paper.txt、_paper2.txt、_wall_measure.py、icrl_softmax/_audit_rows.npy、tmp/）。远端 SSH host key 检查因读取权限失败，未同步。隔离 GPT 工作区 results/FP-L12S-REVIEW-001/codex_worktree，分支 codex/FP-L12S-REVIEW-001。

## 执行与证据记录
2026-09-14：Claude 只读预审 APPROVED 后返回报告；该送审进入 ACTIVE、VERIFYING。发现证明和结果报告 FAIL，本次送审已完成，但研究修复未完成，不标 VERIFIED。审查结果与 GPT 复核归档见 docs/research_branches/FP-L12S-REVIEW-001/codex/review_summary.md。当前状态：VERIFYING（意见已归档，相关任务定义冲突待用户裁决）。
命令：送审单通过 stdin 交给 claude -p --permission-mode dontAsk --tools Read,Glob,Grep --allowedTools Read,Glob,Grep --no-session-persistence --output-format text；退出码 0，stderr 为空。仅开放读取工具。未执行全面实验重放。
送审过程中宿主另出现 FP-COST-001 任务及 evaluate/analyze_fp_cost_001.py 未跟踪文件，均视为他人工作保留，不纳入本次范围。
