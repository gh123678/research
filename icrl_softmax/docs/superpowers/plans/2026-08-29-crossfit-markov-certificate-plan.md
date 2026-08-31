# Cross-fit 与 Markov Certificate 执行计划

> 日期：2026-08-29  
> 依据：docs/superpowers/specs/2026-08-29-crossfit-markov-certificate-design.md  
> 状态：已获用户授权执行

## Task 1：核对原始 Markov 浓缩工具

读取 Fan, Jiang & Sun (2021) 的正式 JMLR 版本和 Xie et al. (2026) 的相关附录。

输出：

- docs/research_branches/crossfit_markov_certificate_theory.md

要求：

- 记录原始 theorem 的链类型、初始分布、函数界、谱/mixing 常数与概率形式；
- 区分直接可用、需要 pair-chain 替换、仅作诊断三类；
- 不把 gap mixing proxy 写成已证明的 cross-fit 独立性。

## Task 2：实现 Markov/coverage certificate

新增：

- markov_coverage_certificate.py

核心接口：

- stationary_time_reversal
- edge_chain_transition
- dobrushin_coefficient
- total_variation_mixing_curve
- suggest_crossfit_gap
- kernel_coverage_certificate

契约：

- transition 必须为方阵、非负、行随机；
- forward/reverse stationary relation 数值成立；
- 对称链 forward/reverse 指标一致；
- sticky chain 的建议 gap 不小于 fast-mixing chain；
- gap 截断与 gap_capped 标记一致。

## Task 3：实现两折 cross-fit

新增：

- crossfit_vfirst.py

核心接口：

- crossfit_blocks
- build_crossfit_targets
- crossfit_vfirst_estimate

实现：

- exact matching；
- finite one-hot softmax recovery；
- 无间隙与带间隙；
- pair-count weighted aggregation；
- population-mixture reference；
- V error、recovery error、composition slack 和样本使用诊断。

## Task 4：建立联合契约验证器

新增：

- verify_crossfit_markov_certificate.py

覆盖：

- g=0 与 g>0 样本计数；
- 跨折 target 无泄漏；
- 拼接 targets 与计数合并等价；
- oracle-V population mixture；
- 某一折缺 pair、两折缺 pair；
- rare-action、deterministic-transition、zero-gap；
- random、sticky、symmetric chain certificate。

运行：

    python verify_crossfit_markov_certificate.py

验收：所有断言通过并打印 PASS。

## Task 5：扩展 fixed-policy 实验入口

修改：

- evaluate_fixed_policy_q_routes.py

新增路线：

- vfirst_crossfit_exact
- vfirst_crossfit_softmax
- vfirst_crossfit_gap_exact
- vfirst_crossfit_gap_softmax

新增输出字段：

- recovery_transitions_used
- discarded_gap_transitions
- fold_a_v_sup_error
- fold_b_v_sup_error
- dependency_tv_at_gap
- gap_raw
- gap_used
- gap_capped
- direct_kernel_threshold_slack
- state/pair/edge mixing certificate

结果写入新目录 results/fixed_policy_q_routes_crossfit，不覆盖上一轮结果。

## Task 6：烟雾测试与回归

烟雾运行：

    python evaluate_fixed_policy_q_routes.py --tasks 2 --trajectory-lengths 128 512 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --iterations 80 --output-dir results/fixed_policy_q_routes_crossfit_smoke

回归：

    python verify_fixed_policy_q_routes.py
    python verify_end_to_end_sarsa.py
    python -m py_compile verify_crossfit_markov_certificate.py crossfit_vfirst.py markov_coverage_certificate.py evaluate_fixed_policy_q_routes.py
    python -m ruff check verify_crossfit_markov_certificate.py crossfit_vfirst.py markov_coverage_certificate.py evaluate_fixed_policy_q_routes.py

## Task 7：聚焦正式扫描

运行：

    python evaluate_fixed_policy_q_routes.py --tasks 30 --trajectory-lengths 256 1024 4096 16384 --n-actions 4 --pi-mins 0.05 --betas 8 --mixing 0.08 0.5 --gap-bonuses 0 0.5 --iterations 160 --output-dir results/fixed_policy_q_routes_crossfit

输出：

- config.json
- task_results.json
- summary.json
- crossfit_error.png
- recovery_budget.png
- certificate_vs_error.png

## Task 8：更新结论

修改：

- docs/research_branches/direct_q_vs_v_first_report.md

补充：

- cross-fit 与 no-split/split/Direct-Q 的正式对照；
- 95% Student-t CI；
- 配对胜率与 coverage 条件结果；
- gap_capped 比例和 certificate 满足率；
- 是否值得把 cross-fit 放入 blockwise control 的新决定。

论文主文保持不变。
