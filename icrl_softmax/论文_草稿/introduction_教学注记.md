# Introduction 教学注记（新主线）

## 引言的六段逻辑

1. **两条前作路线**：Liang–Lai 用 linear attention 做参数化 policy improvement；Xie 用 softmax 做固定策略 TD evaluation。
2. **纠正旧问题**：softmax 正权重不会抹掉 value 中 `delta` 的负号。
3. **真实难点**：参数路线要动态生成 `deltaphi`；Q-learning 路线要生成 greedy target。
4. **方法选择**：不强迫 softmax 重现 `deltaphi`，而是 token 化 `Q(s,a)`，直接写回查询值。
5. **理论与实验**：第一阶段误差 `log|A|/beta`，第二阶段有 kernel/context 误差；实验逐项隔离。
6. **克制声明**：不声称单头参数化 Q-learning、不声称 max 对所有改进必要、不使用宽泛 first claim。

## 读引言时只问三个问题

- 论文维护的是参数 `w`，还是每个 query 的 `Q(s,a)`？
- max、delta 和写回分别是在 attention 内、prompt 中还是外部？
- 理论误差究竟来自温度、kernel、采样，还是未经证明的“表达性不可能”？

更完整推导见 `paper_教学注记.md`。
