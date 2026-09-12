# Softmax 注意力与策略改进

研究固定权重的 softmax 注意力网络能否从经验数据估计动作价值，并为策略改进提供依据。

从 [当前研究状态](ACTIVE_WORKSPACE.md) 开始阅读。该页集中说明研究主线、最近结果、证据限制和任务入口。

## 阅读位置

- [当前研究状态](ACTIVE_WORKSPACE.md)：当前主线和最近任务。
- [材料索引](docs/README.md)：说明当前入口与历史研究依据的区别。
- [正式任务](docs/research_tasks/README.md)：各次研究的问题、协议、验收依据和记录。
- [研究证据](docs/research_branches/)：推导、实验报告和验证记录。
- [历史构造依据](论文_草稿/README.md)：仍被引用的早期推导与复核材料。

实验代码保留在本目录，运行产物在 `results/`，参考论文在 `papers/`。复现某次实验时，使用该任务报告记录的版本和命令。

协作规则见仓库根目录 [AGENTS.md](../AGENTS.md)。当前状态统一维护在 `ACTIVE_WORKSPACE.md`，不再使用旧教学稿或旧论文导出件判断项目进度。
