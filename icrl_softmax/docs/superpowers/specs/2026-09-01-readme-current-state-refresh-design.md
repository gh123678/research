# README 当前状态重写设计

> 日期：2026-09-01｜状态：用户已批准｜目标文件：`icrl_softmax/README.md`

## 1. 背景

当前 README 自初始提交后未更新，仍把 sampled-SARSA attention 构造、旧控制实验和论文构建流程作为项目主线。项目在 2026-08-31 已完成新的 fixed-policy 阶段：

- Direct-Q 在冻结平稳单轨迹上的 uniform-in-layer 有限样本界；
- V-first no-split 在同一轨迹上的 fixed-ghost 有限样本界；
- state、pair、edge 三条链共享的显式证书；
- 480 项同种子正式复现与旧路线零 mismatch 回归。

README 的目录树还包含已归档的 `diag_*.py`、`train_*.py` 和不可运行的 `evaluate_c3_v3.py` 命令，同时漏掉当前核心实现与理论文档。

文件本身是严格有效的 UTF-8，没有损坏字符。显示异常主要可能来自 Unicode 树形字符、兼容性较差的公式分隔符和混杂的排版符号。

## 2. 采用的方案

采用“当前研究主线优先”的整体重写：

1. README 首屏说明当前 fixed-policy 研究问题、完成状态和适用范围；
2. 诚实区分已闭合结论、经验观察和仍开放问题；
3. 给出可直接运行的快速验证与 480 项正式复现命令；
4. 用普通 Markdown 列表或表格导航核心代码、理论和结果；
5. 把 sampled-SARSA 构造与论文构建降为简短的历史成果章节；
6. 删除活跃目录中已经失效的旧 V2/V3 复现命令。

未采用的方案：

- 新旧研究线等篇幅并列：会继续掩盖当前目标；
- 完全删除旧 SARSA 内容：会丢失仍有价值的前置构造与端到端验证入口。

## 3. README 结构

新版按下列顺序组织：

1. 项目现状与当前研究问题；
2. 已闭合结论；
3. 适用范围与明确限制；
4. 快速验证；
5. 480 项正式复现与分析；
6. 核心代码和证据导航；
7. 已观察到的正式结果；
8. 仍开放的问题；
9. 历史 sampled-SARSA 构造与论文构建；
10. 本地结果、参考论文和归档说明。

## 4. 内容契约

README 的事实以以下文件为准：

- `ACTIVE_WORKSPACE.md`：当前状态、实现和证据入口；
- `docs/research_branches/shared_fixed_policy_finite_sample_theory.md`：定理范围与显式证书；
- `docs/research_branches/direct_q_vs_v_first_report.md`：480 项结果与路线判断；
- 当前脚本的参数接口：复现命令及输出目录。

必须明确写出：

- Direct-Q 与 V-first no-split 的 fixed-policy 结果均已闭合；
- 两条路线没有统一支配关系；
- 保守 pair-coverage 充分条件在 480 项正式配置中的通过率为 0%，这不等于经验覆盖失败或定理错误；
- fully online control、非平稳起步、额外随机 reward noise 和更尖锐的 occupancy-adaptive rate 仍开放；
- `results/` 与 `papers/` 被 Git 忽略，新 clone 不包含正式结果文件；
- 分析脚本需要先生成新结果，并保留旧 baseline 结果目录。

## 5. 格式与编码

- 保持 UTF-8、LF，不添加 BOM；
- 删除 Unicode 目录树字符，改用普通列表或表格；
- 公式采用 GitHub Markdown 更稳定的 `$$ ... $$` 块；
- 关键文档使用相对 Markdown 链接；
- 命令块只包含当前存在且参数有效的入口；
- 统一中英文标点，移除 `attention--FFN` 等转换残留。

## 6. 验证

实施后执行以下只读或轻量检查：

1. 检查 UTF-8 解码、替换字符、控制字符和代码围栏配对；
2. 检查 README 中的相对文件链接均存在；
3. 检查所有列出的 Python 入口均存在；
4. 运行三个当前 fixed-policy 契约验证；
5. 运行历史端到端 SARSA verifier，确保历史章节仍可复现；
6. 检查 Git diff，只允许 README 与实施计划发生预期变化。

正式 480 项扫描成本较高，不因 README 修改而重复运行；README 命令将按已完成的正式配置和脚本参数进行静态核对。

## 7. 验收标准

- 新读者能在首屏识别当前目标、完成状态和结论边界；
- README 不再引用活跃目录中不存在的脚本或结果；
- 当前核心实现、理论文档和验证命令均可找到；
- Windows 和 GitHub 渲染不依赖 Unicode 树形字符或 BOM；
- 历史成果保留但不再冒充当前研究主线。
