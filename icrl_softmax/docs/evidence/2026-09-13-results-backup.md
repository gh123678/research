# 封存结果证据包（2026-09-13）

阶段 1（证据与勘误）的备份交付。动机：`results/` 被 Git 忽略，封存包此前仅存于本机单一位置（耐久性风险记录于 `docs/2026-09-12-independent-verification-handoff.md`）。

## 内容

- 源：`icrl_softmax/results/`（2209 个文件，757.0 MB，34 个任务目录 + 1 个模型文件）。
- 备份位置（本机第二位置）：`C:\Users\Admin\Desktop\research\_backups\icrl_softmax_results_2026-09-13\`（目录内容为 `results/` 的完整镜像，清单位于其根目录）。
- 哈希清单：`MANIFEST-results-2026-09-13.sha256`（本目录内的入库副本；备份根目录的 `MANIFEST.sha256` 为同一文件）。格式：每行 `<sha256>  <相对路径>`，哈希以**源文件**计算。
- 验证：备份全部 2209 个文件已逐一对照清单重算 SHA256，`verified: 2209, problems: 0`（2026-09-13，本机）。
- 备份时仓库状态：分支 `claude/FP-CENSUS-001`，提交 `a2df7f8`（勘误提交）之后。

## 边界

- 这是同机第二位置备份，防误删/防覆盖，**不防磁盘故障**；异地或离线备份由用户决定。
- 清单覆盖备份时刻的 `results/` 快照；此后新增结果不在其中。
- 恢复方式：将备份目录内容复制回 `icrl_softmax/results/`，并用清单重算校验。
