# FP-COMPOSE-REVIEW-003 最终报告

日期：2026-09-15；GPT；Codex 分支 `codex/FP-COMPOSE-REVIEW-003`；冻结提交 `bcec1fe4f8ef49b2d1835f706534b97d8a564016`；执行代码提交 `cdc5772`。用户已批准 v1.1 修订。此次审查是事后验证，不是盲态发现。

## 结论

四个缺口的分层结果：

| 层 | 判定 | 含义 |
|---|---|---|
| T1–T5、L12 单侧幅值界、Bellman 残差界、条件性不退化 | **PASS（理想实数/采样协议）** | 数学前提与推理成立；不等于机器浮点形式化证明 |
| 002 全量采样/认证/网络/滚动策略重放 | **PASS（复现层）** | 96/96 记录，独立重写证书/决策/价值核；共享待测生成器与网络实现仍是结构限制 |
| 扩维数字算术 | **PASS** | 独立复算与封存表一致，包括完整 sweep |
| 扩维“真实上限/成本下界/所有省略效应同向” | **FAIL** | 两点混杂外推无法授权这些统计或因果结论 |
| 远端 refs | **PASS** | 只读查询完成，main 已核对；COMPOSE 分支未发布 |

## 全量重放证据

独立重放器 [replay.py](replay.py) 没有导入任何 `FP-COMPOSE-001/002/claude/` 程序；只共享待测的基线模型、MDP/训练生成器和数组生产者。认证采样、首访抽取、L12 标量、逐行判决、状态空间价值求解均独立实现。

命令：

```
C:/Users/Admin/anaconda3/python.exe -B docs/research_branches/FP-COMPOSE-REVIEW-003/codex/replay.py --threads 6 --output results/FP-COMPOSE-REVIEW-003/codex/formal_v1
```

结果文件：[formal_v1/result.json](../../../../results/FP-COMPOSE-REVIEW-003/codex/formal_v1/result.json)。

- 96/96 条记录；1,136 个实际唯一认证批；4,534 个生产者步骤；2,267 次网络前向；4,510 个价值重算。
- 前四步与 001/002 封存记录逐字段一致：1,536 步，差异 0。
- `Qhat` 最大差 `0.0`；`E_Q` 最大差 `0.0`；价值增量最大差 `8.88e-15`；策略最大差 `0.0`。
- 最小安全余量 `+0.103226901473463`；最小价值增量 `+2.8091262649354576e-10`。
- 所有 87,546 项重放断言通过；最终 `verdict=PASS`。
- 6/1 线程桥接在首个固定输入上逐位一致；正式运行使用原 6 线程设置。

补充的价值、成本和审计字段复核见 [supplement.json](../../../../results/FP-COMPOSE-REVIEW-003/codex/supplement.json)，结果 PASS。来源版本快照见 [source_provenance.json](../../../../results/FP-COMPOSE-REVIEW-003/codex/source_provenance.json)。

### 复现边界

这不是完全独立的“从另一套生成器/另一网络实现重做”：`model.py`、MDP/训练生成器和数组生产者是定义待测对象的共享依赖。因此它能发现认证、首访、证书、判决、价值审计和控制流错误，不能排除共享生成器或被测网络实现中的共同概念错误。原封存包保存的批摘要只覆盖前 1,000 个状态字段；本次从冻结种子重抽了完整批并保存了新的整批 SHA256，不能倒推旧包曾完整封存所有原始字节。

## 定理层

原始 Maurer–Pontil 2009 Theorem 4 是单侧经验 Bernstein 界：[原始论文](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf)。对真实残差均值的符号分情况，`|rho| <= |mean| + radius` 每个固定分布只需一个方向；先对随机首访容量逐 `n` 积分，再对 `S·A` 个状态动作对做 union，得到每步 `delta_step`。每步新批与历史条件独立，再对 `K` 步 union，故单条生产者-路线轨迹的风险是 `K*delta_step`；K=12 本任务为 `0.15`。

按状态规则在同一 sup 误差事件上使用 Hölder 不等式；所有候选由旧策略构造，未通过的行不变。`(I-gamma P_new)^(-1)` 非负，因此条件于覆盖事件逐分量不退化。完整推导见 [theory_review.md](theory_review.md)。这不提供 192 条路线记录、两生产者的联合保证，也不提供软件舍入的形式化保证。

## 扩维复算与撤回

[dimension_review.py](dimension_review.py) 独立复算了首步覆盖锚点、两点指数、半径比值、门限扫描、成本表和完整 sweep；[dimension/result.json](../../../../results/FP-COMPOSE-REVIEW-003/codex/dimension/result.json) 顶层 `differences=[]`，包括 sweep 和网格标量。

复算成立的只是模型内算术：两点拟合指数 `-0.625225...`；模型网格上门限先于支持阈值关闭（两点拟合 `d=96` 对 `d=224`，机制指数 `-1` 为 `d=64` 对 `d=96`）。

不能保留的说法：`d≈64–96` 是真实天花板；表中链数倍数是现实成本下界；更大维度的方差、价值范数必然增大、门限必然下降；所有省略效应都同向；两小时预算仍然足够。两校准点同时改变族、状态、动作、mixing、任务和奖励条件；合成锚点还混用了 F1 的 `E_Q` 中位与跨族合并的 `h` 中位。详见 [dimension_report.md](dimension_report.md)。

## 纠正的报告数字

用固定 192 分母的累计价值曲线，独立重算得到：

- 第 6–12 步贡献 **4.309259062744131 个百分点**；
- 第 5–12 步贡献 **7.839101217608224 个百分点**；
- 各步正增量轨迹数为 `[192,192,192,192,192,191,188,186,185,185,180,180]`。

所以“后段达到预登记的 2 点阈值”仍成立；“每条轨迹每一步都继续增长”不成立。小增量也不能说明剩余最优性差距已耗尽；例如某末步轨迹只闭合约 62.68%，仍剩约 37.32%。

## 远端

远端只读结果见 [remote_review.md](remote_review.md) 与 [remote_heads.txt](remote_heads.txt)：

- `origin/main = c1e03dd4e5cd610cf5b8ee0eb57c6b1de0844d0d`，与本地 main 一致；
- 远端没有 `claude/FP-COMPOSE-001`、`FP-COMPOSE-002` 或本次审查分支；
- 本轮没有 fetch、push、merge 或修改远端。

## Claude 交叉检查

Claude 只读检查报告见 [claude_interim_review.txt](claude_interim_review.txt)。其判定：T PASS；D 算术 PASS；D 推论对撤回结论 PASS；R 代码层条件 PASS、结果在当时尚未完成。之后 R 已完成并通过完整计数。Claude 指出的“扩维比较键覆盖不全”已修正并重跑；关于原任务判据的两条建议经冻结任务核对后不适用：本任务暂停阈值是 `1e-10`，不是负值容差；COMPOSE-002 没有 divergence guard 弃权分支。Claude 的最终检查不是对其原实验的独立佐证。

## 最终状态

本审查任务的 T、R、D 算术和 G 证据齐备，但扩维推论已判 FAIL，且 R 仍有共享待测生成器/网络的结构边界。因此不把原 FP-COMPOSE-002 的任务升级为“完全独立验证”或新的 `VERIFIED`；只把上述有限复现和数学结论记为 PASS，并撤回超出证据范围的扩维上限叙述。
