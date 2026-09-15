# `model.py` 实现规格审计

- 日期：2026-09-15
- 审计对象：`model.py` 的 `FixedPolicyActionExpectation`、
  `EndToEndMaskedSoftmaxExpectedSARSA`、`EndToEndFiniteSoftmaxExpectedSARSA`
- 审计基线：当前 `claude/FP-COMPOSE-001` 工作树
- 审计程序：`model_spec_audit.py`
- 结果 JSON：`results/FP-MODEL-REVIEW-001/model_spec_audit.json`
- 既有套件：`verify_fixed_policy_expected_sarsa.py`，**13,550 checks PASS**
- 本次新增检查：**721/721 PASS**，60 个随机 fixture（float64/float32、多个状态/动作维度、随机与强制未访问 pair 两类）

## 结论

### 有效输入的算法对应：PASS

在输入满足任务实际前提时，literal Torch 实现与本地纯 NumPy 参考实现一致：

| 部分 | 审计结果 |
|---|---|
| `FixedPolicyActionExpectation` | 对每个 `next_state` 只在该状态的动作 token 上赋予 `pi(a|s)`，与精确动作期望一致 |
| masked singleton retrieval | 每个当前 pair 只读对应 Q-memory token，随机维度测试与 `run_expected_exact` 一致 |
| masked write-back | 已访问 pair 对其匹配 transition 做均匀平均；未访问 pair 只读零值 null token，更新为 0 |
| finite successor | `log pi + zeta * state-match` 的完整 token softmax，与 `finite_successor_all` 一致 |
| finite read | `xi * pair-match` 的完整 token softmax，与 `finite_read_all` 一致 |
| finite write-back | `tau` finite kernel 的分母、full-support leakage 与 `finite_writeback_all` 一致 |
| 参数/状态 | 两个端到端类均无 trainable parameters，`state_dict()` 为空 |
| 注意力不变量 | 有效输入下权重归一化；finite 三处 attention 严格为正；未访问 finite query 产生非零 leakage |

既有套件的 S13 已在 fixture 上检查 literal Torch 与纯参考的一致性；本次新增 60 个 fixture 扩展了维度、dtype、重复 pair、未访问 pair 和 attention 归一化检查。float32 与 float64 的差异均在对应 dtype 的容差内：本次最大绝对差约为：

- masked：`1.67e-7`
- finite：`4.18e-7`

### 明确的 API/鲁棒性限制

这些不是 FP-COMPOSE-001/002 使用的**有效 policy + sharpness=8**配置上的算法不一致，但应在规格中明确：

1. **policy 合法性没有被运行时验证**：
   `policy` 在文档语义上是概率分布，但实现不拒绝：
   - 非负但未归一化的行；
   - 含 0 的行；
   - 含负值的行。

   本次测试中三类输入都被接受且产生有限输出。对有效概率 policy 没有影响；对非法 policy，输出不再具有 Expected-SARSA 的概率语义。
   **建议**：要么在入口检查严格正值与行和 1，要么把它明确写成调用方前置条件并增加测试。

2. **极大 finite sharpness 的数值边界**：
   sharpness `8, 50, 100, 500` 均能产生有限输出；`1000` 在 finite write-back 的 `math.exp(tau)` 处抛出 `OverflowError: math range error`。
   构造函数当前只检查 `tau > 0`，没有 finite/上界检查，也没有稳定的 log-domain 写法。
   **任务配置 `zeta=xi=tau=8` 不受影响**；这是通用 API 的鲁棒性缺口。

3. **设备一致性没有显式验证**：
   `states/actions/next_states` 只调用 `.long()`，没有显式移动到 `q_values.device`。如果 q 在 GPU 而 index tensor 在 CPU，后续 pair comparison/indexing 会发生 device mismatch；CUDA 本机不可用，因此本项未执行 GPU 运行测试。
   **建议**：入口显式检查所有 tensor 同 device，或统一 `.to(q_values.device)`，并增加 CPU/GPU 测试。

## 未发现的算法问题

- masked 路径没有发现 null-token 泄漏或已访问/未访问 gate 反转。
- finite 路径没有发现意外的 equality mask、visited gate 或零权重。
- `model.py` 的端到端输出与现有参考路线一致，且既有 13,550 项验证也通过。
- 本审计没有修改 `model.py`。

## 审计边界

这份审计验证的是**实现与仓库内纯参考路线的一致性**，不是从论文规格完全独立重写一套网络架构。因此它不能排除：

- 纯参考实现与 `model.py` 共享的概念错误；
- docstring/既有参考路线本身与论文原意不一致；
- CUDA、混合设备、非 CPU backend 的实际行为问题；
- 非法 policy 与极大 sharpness 是否应当属于公开 API 的合法输入。

因此本报告结论应写作：

> `model.py` 在 FP-COMPOSE-001/002 使用的有效输入与 sharpness=8 配置下，端到端实现与本地纯参考路线一致；同时存在未强制执行的 policy 输入契约、极大 sharpness 溢出边界和未测试的跨设备契约。**这补上了实现一致性审计的一部分，但不是论文规格的完全独立证明。**
