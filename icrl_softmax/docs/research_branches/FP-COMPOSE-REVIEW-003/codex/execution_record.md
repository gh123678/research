# 执行记录（进行中）

## 授权与冻结
- 用户于 2026-09-15 回复“批准”，批准 proposed_ruling.md 四条具体修订；正式任务发布 v1.1 ACTIVE，O1/O2 解除。
- 原作者基线 bcec1fe4f8ef49b2d1835f706534b97d8a564016。
- 新重放代码冻结提交 cdc5772（完整提交可由 git rev-parse cdc5772 获取）；Codex 分支 codex/FP-COMPOSE-REVIEW-003。
- 封存输入与运行源码的 SHA256 在各输出 manifest.json/result.json 中。宿主 Claude 分支与未跟踪内容原样保留。

## 命令（cwd 为本隔离 worktree 的 icrl_softmax）
```
C:/Users/Admin/anaconda3/python.exe -B docs/research_branches/FP-COMPOSE-REVIEW-003/codex/replay.py --bridge --output results/FP-COMPOSE-REVIEW-003/codex/thread_bridge
C:/Users/Admin/anaconda3/python.exe -B docs/research_branches/FP-COMPOSE-REVIEW-003/codex/replay.py --threads 6 --output results/FP-COMPOSE-REVIEW-003/codex/formal_v1
C:/Users/Admin/anaconda3/python.exe -B docs/research_branches/FP-COMPOSE-REVIEW-003/codex/dimension_review.py
C:/Users/Admin/anaconda3/python.exe -B docs/research_branches/FP-COMPOSE-REVIEW-003/codex/supplement.py
```

## 首个输入线程桥接
F1/0.08/task200，两个网络各重跑 6线程和1线程；输出逐位一致。exact 6线程1.85秒/1线程6.40秒，finite 6线程1.35秒/1线程5.04秒。正式采用原6线程，未改变网络、采样、策略、K或矩阵。

## 已完成检查
- H0：原16个字段以及补充审计的全部字段（仅排除明确改变的 delta_total）1536步零差异；两套输入记录清单相同。
- D：独立锚点、指数、网格门限及成本表复算 PASS；外推上下界声明 FAIL，详 dimension_report.md。
- 补充元数据/价值/成本：supplement.json PASS；重新求最优策略的分母差最大1.391e-11、逐步闭合率差1.622e-12，处于冻结复现门槛内。
- R：完整运行已完成；96/96 记录、1136 批、4534 生产者步骤、2267 网络前向、4510 价值步骤，result.json 完整计数 PASS。

## 异常/重试
- 构建 worktree 与远端查询起初受 Git/SSH 沙箱权限限制；经正常权限升级成功，没有关闭主机密钥检查。
- Claude 中期只读交叉检查第一次命令把 prompt 放在变长 --add-dir 参数后，被 CLI 解析为路径而未收到输入；exit1，无审查、无实验。第二次改 stdin 后启动，输出 claude_interim_review_v2.txt；没有绕过权限。
- 截至此记录，正式重放尚无异常；后续任何 failure.json 原样保留，不覆盖输出目录。

## 证据边界
原封存 batch_hash 只覆盖前1000个 states/next_states，且为SHA256前16位。新程序重抽全部样本，并校验该已存在摘要、全部首访计数和残差统计；同时记录新批四个字段的完整SHA256。不能把原来的短前缀摘要称作完整原始批字节封存。训练MDP/批哈希按原dtype/shape/bytes口径核对。
model.py、基线数组估计器及MDP/训练生成器是共享的待测定义；认证采样、首访保留、证书、决策和状态空间价值核为独立实现。共享模块最终清单见 formal_v1/shared_modules.json，不把共享被测模型说成全新网络实现。

## 完成
2026-09-15：全量 R 退出码 0、verdict PASS。T/D/G 分层终稿见 final_report.md；D 推论 FAIL 的具体撤回项见 dimension_report.md。原任务不升级 VERIFIED。
