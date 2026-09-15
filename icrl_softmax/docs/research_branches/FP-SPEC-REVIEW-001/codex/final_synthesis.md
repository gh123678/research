# 论文规格与实现对应审查：FP-SPEC-REVIEW-001

日期：2026-09-15。基线：`16ee0f652cde56b4f2ef6e3f1a43b58818bb0108`。
状态：**本审查任务 VERIFIED**。双方最终报告与双向 PASS 已归档；该状态只覆盖本次文献／规格对应审查，不升级任何先前实验任务。

## 结论

**当前实现是有明确规格的本地构造，不是两篇外部论文网络的直接复现。**

两条路线都可从 `FP-ESARSA-001` 的冻结契约明确导出其更新算子。它们继承本地早期 sampled-SARSA 构造的完整 Q-memory、读写与路由结构，再改成固定策略的动作期望；有限版还移除了旧构造的 visited gate。因此，应把问题从“是否忠实复现论文”改为“哪些组成来自论文、哪些是自行定义且另行论证的扩展”。

实现对象并非指称不明。已确立的是**本地规格与实现算子的对应**；不能声称它就是外部论文的原网络，或直接获得外部论文的全部保证。

## 四项关键发现

### 1. 两篇论文分别提供不同的依据，不能拼接成直接继承关系

| 来源 | 原文对象 | 当前实现的差别 |
|---|---|---|
| Xie，arXiv 2605.07333v2，pp.4–6，式 (8)–(19)、Theorem 1 | 轨迹列上的 weighted softmax TD；双记忆行和时间移位；固定策略状态价值评估 | 完整状态—动作 Q-memory；显式读取当前值和策略期望后继值；无 TSM。有限 writer 的 one-hot 内积核确实属于其 score 形式，但完整算子不同 |
| Liang–Lai，arXiv 2605.05755v1，pp.2–4，式 (1)、(2)、(4)、(5)、Theorem 3.1 | 线性注意力实现全局 `α/N` batch semi-gradient SARSA，使用采样后继动作 | 归一化 softmax / 路由注意力；精确版对已访问 pair 使用局部均值，有限版保留组外权重；Expected 后继；不是其训练架构 |
| 本地历史构造，式 (3.1)–(3.9) | sampled-SARSA，精确等值路由；旧有限版保留 visited gate | Expected 后继与无 gate 有限版是后续任务明确声明的扩展 |

**正面对应**也已保留：`φ(x)=√τ e_x` 给出 `〈φ(x),φ(y)〉=τ1{x=y}`，所以有限写回核是 Xie 内积 score 的特例。不能用“one-hot 等值分数不是特征相似度”作为否定依据。

### 2. “论文 Theorem 3.1”有可定位的本地来源，但引用方式含糊

`model.py:926` 的结构化等值 mask 对应本地 `论文_草稿/preliminaries.md` 的 Theorem 3.1，以及独立构造文的式 (3.6)。它不是 Liang–Lai 的同号线性注意力定理。旧有限版的 visited gate 也能在本地构造找到。

正确处理是注明**本地定理来源 + 后续 Expected-SARSA 扩展**。不能据裸引用就断言作者误读了 Liang–Lai。按用户裁决，本次把这项记录在报告，不改 `model.py`。

### 3. 精确动作期望头是代数等价的直接权重构造

`FixedPolicyActionExpectation`（`model.py:888–915`）直接计算

`A[t,(u,b)] = 1{u=s'_t} π(b|s'_t)`，再乘 Q-memory。

在严格正、行归一化策略下，它等于“只允许下一状态动作 token、logit 为 `log π`”的 masked softmax。但该函数本身不计算 logits、不调用 softmax。精确路线的当前值检索和写回确实调用了 softmax。

因此，**算子对应成立，不等于所有 head 都按完整 prompt / projection / softmax 图字面执行**。当前 compact 实现不能被描述为两篇论文或历史完整 token 构造的逐模块复现。均匀权重仍是合法注意力，不能反过来把精确路线的 softmax 称为“只有名义”。

### 4. 有限版不能直接继承旧界和外部收敛结论

- 有限版当前读取和后继读取都会泄漏，真实 `Q^π` 不一定是其固定点。GPT 的两动作、`γ=0` 手算反例经 Claude 独立核算通过；细节见修订报告 §4。
- 若 `n_x=0`，有限 writer 为 `1/N`，更新为 `α` 乘残差均值；精确版更新为零。只令 sharpness 增大不能消除这个差别。
- 旧 sampled-SARSA / gate-assisted 界不能仅凭泄漏质量比较移植给 Expected / gate-free 算子。目标残差与支持规则必须同时匹配。
- `α/N` 与 `α/n_x` 在相同步长下不同；均衡计数 `n_x=N/m` 仍相差 `m` 倍。有限 `τ` 的 visited writer 仍带组外权重，只在适当极限或特殊输入下化为精确均值。

这些边界不推翻封存组合结论：后者针对实际产生的 Q 估计，通过外部证书与按状态规则验证安全更新，不依赖把当前网络认定为 Xie 或 Liang–Lai 原网络。

## 建议采用的主问题措辞

> 在本仓库定义的固定策略 Expected-SARSA 注意力算子实现中，精确路由版与有限-logit 近似版产生 Q 估计，再由外部 L12 证书和按状态保守规则决定策略更新。精确版动作期望采用与 masked softmax 代数等价的直接权重构造。

简写可用“**本仓库构造的网络**”，并链接上述规格。不要简写成“Xie / Liang–Lai 论文的网络”；不要暗示模型内部完成证书、策略选择或学会路由。

## 证据、验证与范围

- [GPT 修订报告](first_result_v2.md)：完整算子推导、覆盖表、代码行号、原文页码与反例。
- [执行记录](execution_record.md)：基线、环境、原始文件指纹、失败读取与独立性边界。
- [GPT 对 Claude 首稿的复核](verification.md)、[对第二稿的复核](verification_v2.md)：失败记录保留，不能以主结论相同代替逐条查错。
- [Claude 最终报告的原文副本](claude_evidence/final_report.md)、[Claude 对 GPT 修订报告的 PASS](claude_evidence/verification_v2.md)、[GPT 对 Claude 最终报告的 PASS](verification_final.md)。原文件保留在 Claude 独立工作区；[归档清单](archive_manifest.json)列出原始及 LF 归一化指纹。

此次为原文、规格、源码和手算的双路线审查，**没有训练、重采样或重跑封存实验**。双方均有历史材料接触；Claude 的 shell 不可用，因此文件指纹由 GPT 补充核对，Claude 的独立核对是读取与代数验证，未伪称独立运行哈希或实验。没有重新证明两篇外部论文的全部定理，也没有验证通用输入、CUDA、可学习性或完整 Transformer 编译构造。

`model.py`、NumPy 参考、封存结果和 `main` 保持不变。`FP-COMPOSE-001/002` 不因本次规格审查升级状态。
