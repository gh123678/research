# icrl_softmax

研究问题：标准 softmax attention 在什么明确的结构条件下可以实现 sampled-SARSA 更新，以及这种构造与 softmax 控制近似、线性注意力参数更新分别有什么边界。

当前构造性结论：一个固定权重、两块 residual attention--FFN 构造可以对同一状态--动作对的样本执行精确的组内平均更新

\[
Q^+(x)=Q(x)+\frac{\alpha}{n_x}\sum_{k:X_k=x}
\bigl(R_k+\gamma Q(X'_k)-Q(X_k)\bigr).
\]

这里的“精确”有严格前提：equality-routing mask 与 visited-query/null-token gate 由外部提供，网络无 layer normalization 和 dropout，注意力是双向结构化注意力而不是 decoder-only causal Transformer；单个固定块对应固定步长 \(\alpha\)。因此，这不是“通用预训练 Transformer 会自行发现 routing”的结论，也不是全批次 \(\alpha/N\) 更新（除非所有组计数恰好相同）。

项目同时保留 softmax 控制近似：softmax 正权重不会消除 value 中 TD error 的正负号；动作 softmax 与状态--动作写回核的有限温度会引入可控偏差。旧 V2/V5 结果仅作为“softmax 评估 + 外部控制读出”基线。

## 目录结构

```
icrl_softmax/
├── *.py             # 全部脚本（平铺 import，勿移动）
│   ├── model.py / mdps.py / evaluate.py    # 核心库
│   ├── train_*.py / evaluate_*.py          # 训练 / 评估
│   ├── diag_*.py                           # 诊断（23 个）
│   └── verify_*.py                         # 验证
├── results/         # 全部运行产物（原平铺在根目录）
│   ├── out_*        # 训练存档（checkpoint + loss.npy + config.json）
│   ├── eval_*       # 闭环评估输出（returns.json + closed_loop.png）
│   ├── tauscan_*    # 温度扫描
│   └── scan/        # 早期温度扫描
├── docs/            # 报告与调研（历史档案，路径保留原样）
│   ├── 夜间工作报告_0810-0811.md
│   ├── idea_评估_icrl_softmax.md / .pdf
│   ├── 文献调研_softmax_policy_improvement.md / .pdf
│   └── 完整构造与双语构建复现.md
├── 论文_草稿/       # 论文正文 + 验证文档
│   ├── full_paper.md                # 由分节源文件生成的英文全文
│   ├── full_paper_中文版.md          # 手工同步维护的中文全文
│   ├── abstract.md / introduction.md / method_experiments.md / ...（分节）
│   ├── 验证_复现核对指南.md          # 每个命令的期望数字
│   ├── 验证_复核报告_0818.md         # 存档核对 + 独立重跑双重验证
│   └── 验证_公式代码对照表.md
├── verify_end_to_end_sarsa.py       # literal / compact / reference 三路构造核对
└── tools/                            # 全文生成、双语检查与 md→pdf 构建脚本
```

## 复现

从本目录运行，结果路径已统一带 `results/` 前缀。完整命令与期望数字见：

- `论文_草稿/验证_复现核对指南.md`（命令清单）
- `论文_草稿/full_paper.md` 附录 A

### 1. 完整构造的逐矩阵验证

```bash
python verify_end_to_end_sarsa.py
```

该脚本实际建立列为 token 的 \(H_0\in\mathbb{R}^{d\times L}\)，其中 \(d=2m+8\)、\(L=m+N+1\)，并显式执行各头的 \(W_Q,W_K,W_V,W_O\)、query-by-source mask、标准指数 softmax、两单位 ReLU residual FFN 与写回矩阵。它把 literal 结果同时与 compact attention 实现和直接 SARSA reference 对照；默认测试应以 `PASS end-to-end softmax SARSA construction` 结束。详见 `docs/完整构造与双语构建复现.md`。

### 2. 大规模 control sweep 的范围

```bash
python evaluate_softmax_sarsa_control.py
```

该 sweep 为节省计算，直接用数组索引读取 \(Q(s,a)\) 和 \(Q(s',a')\)，然后检验分组写回核与闭环控制表现；它**不是**完整 \(d\times L\) prompt、检索头和 FFN 的端到端实现。完整 prompt 构造由上一节的 literal verifier 单独覆盖，不应把两类证据合并表述。

### 3. 双语正文一致性与 PDF

先生成英文全文，再检查英文分节源、生成后的英文全文与中文全文：

```bash
python tools/build_fullpaper.py
python tools/verify_bilingual_manuscript.py
python tools/verify_bilingual_manuscript.py --json
```

英文和中文 PDF 必须串行构建，因为构建器共用临时文件：

```bash
python tools/build_pdf.py --src 论文_草稿/full_paper.md --out 论文_草稿/full_paper.pdf --lang en
python tools/build_pdf.py --src 论文_草稿/full_paper_中文版.md --out 论文_草稿/full_paper_中文版.pdf --lang zh
```

不要直接编辑 `论文_草稿/full_paper.md`；它会被 `tools/build_fullpaper.py` 覆盖。详细命令、检查层次和定理适用范围见 `docs/完整构造与双语构建复现.md`。

### 4. 其他实验

两阶段 softmax Q-control 近似仍可用下列命令复现：

```bash
python verify_two_stage_q_control.py
python evaluate_two_stage_q_control.py --tasks 20 --steps 200 --sweep-tasks 10 --sweep-steps 20
```

输出位于 `results/two_stage_q_control/`。深度重构设计与实施计划位于 `docs/superpowers/`。

历史 V2 外部读出结果（仅作模块化基线）：

```bash
python evaluate_c3_v3.py --ckpt results/out_c3_v3/model.pt --self-eval 0 --update-steps 400
# 期望 [model] final ≈ 0.4266
```

> 注：旧脚本默认 `--outdir`（如 `eval_v3`）仍会写到根目录；复现时建议显式传 `--outdir results/...` 保持整洁。历史报告 `docs/夜间工作报告_0810-0811.md` 中的路径是当时原样（未加 `results/` 前缀），仅作存档参考。
