# 协作说明（AGENTS.md）

本仓库由两个 AI 编程助手协同维护：**Claude Code**（女仆）和 **Codex (GPT)**。两边都在同一台/两台 Windows 机器上本地运行，共享这个目录。本文件是两边共同遵守的协作规则。

## 仓库结构

- `icrl_softmax/` — 当前活跃的研究项目：softmax 能否做 policy improvement 的 ICRL 研究线
  - `ACTIVE_WORKSPACE.md` — 当前工作状态与待办，**开始工作前必读**
  - `docs/` — 研究文档与夜间报告
  - `results/` — 实验输出（已 gitignore，不进仓库）
  - `papers/` — 参考论文 PDF（已 gitignore）
- `_archive/` — 归档（不进仓库）

## 分支约定

- `main` — 只放双方验证过的稳定结果，不直接提交实验性改动
- `claude/*` — Claude Code 的工作分支，如 `claude/fix-eval-bug`
- `codex/*` — Codex 的工作分支，如 `codex/add-baseline`
- 合并到 `main` 前，由另一方审查 diff

## 提交规范

- 提交信息前缀标注作者：`[claude] ...` 或 `[codex] ...`
- 提交信息说明"为什么改"，不只写"改了什么"
- 一次提交聚焦一件事，便于审查

## 工作方式

1. **串行接力优先**：一方完成并提交后，另一方再开始。避免同时编辑同一文件。
2. 开始工作前：`git pull` 拉最新，读 `icrl_softmax/ACTIVE_WORKSPACE.md` 了解当前状态。
3. 结束工作时：提交并 push，在 `ACTIVE_WORKSPACE.md` 更新状态，方便下一棒接手。
4. 实验输出写入 `results/`（不提交）；代码、文档、小图（`figures/`）提交。
5. 研究方向是探索性的：不要擅自删除对方的实验代码或结果，有疑问在文档里标注讨论。

## 环境

- Windows 11，bash shell（路径用 `/`），Python 为 Anaconda（`C:\Users\Admin\anaconda3`）
- GitHub 远程为私有仓库，主人在两台电脑间用 `git pull/push` 同步进度
