# README 当前状态重写实施计划

> 日期：2026-09-01｜依据：`../specs/2026-09-01-readme-current-state-refresh-design.md`

## 任务 1：重写项目首页

- 把 fixed-policy Direct-Q / V-first 有限样本结果移到首屏；
- 写清适用范围、证据等级和未闭合问题；
- 删除过时目录统计和失效的 V2/V3 命令。

## 任务 2：更新复现入口

- 加入三个当前契约验证命令；
- 加入 480 项正式扫描的精确参数；
- 说明分析脚本对本地旧 baseline 的依赖；
- 把端到端 sampled-SARSA 验证保留为历史成果。

## 任务 3：修复跨平台格式

- 用普通 Markdown 表格和列表替代 Unicode 目录树；
- 使用相对链接导航核心代码与理论文档；
- 保持 UTF-8、LF、无 BOM，并检查异常字符与围栏配对。

## 任务 4：验证

- 静态检查链接、脚本路径、编码和 Markdown 结构；
- 运行 fixed-policy 三个契约验证；
- 运行历史 sampled-SARSA 端到端验证；
- 审查 Git diff，确保修改范围符合设计。
