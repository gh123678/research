"""单层线性注意力 ICRL 块（03 论文式 1），含 Proposition 3.1 的结构化参数化。

H_out = H + (1/n) * V H (H^T P H),  P := Q^T K,  D = 3d + 2
输出 TF_theta(H) = f_read(H_out) = H_out 最后一列的后 d 个元素（更新后的 w）。

按 Proposition 3.1，梯度流下不变（对输出无影响）的子块被冻结为零：
- P11, P21, P22
- V11, V12, V22
- V21 的第一行
只有 P12 (2d+1, d+1) 和 V21 的后 d 行（记作 V21bar, (d, 2d+1)）可学习。
"""
import math

import torch
import torch.nn as nn


class LinearAttnICRL(nn.Module):
    def __init__(self, d, gain=0.1):
        super().__init__()
        self.d = d
        self.D = 3 * d + 2
        self.P12 = nn.Parameter(torch.empty(2 * d + 1, d + 1))
        self.V21 = nn.Parameter(torch.empty(d, 2 * d + 1))
        nn.init.xavier_normal_(self.P12, gain=gain)
        nn.init.xavier_normal_(self.V21, gain=gain)
        # 惰性子块（固定 0）
        self.register_buffer("P_full", torch.zeros(self.D, self.D))
        self.register_buffer("V_full", torch.zeros(self.D, self.D))

    def _assemble(self):
        P = self.P_full.clone()
        V = self.V_full.clone()
        P[: 2 * self.d + 1, 2 * self.d + 1:] = self.P12
        V[2 * self.d + 2:, : 2 * self.d + 1] = self.V21  # V21 第一行恒为 0
        return P, V

    def forward(self, H, n):
        P, V = self._assemble()
        Hout = H + (1.0 / n) * (V @ H) @ (H.transpose(-2, -1) @ P @ H)
        return Hout[-self.d:, -1]


class SingleHeadTSM(nn.Module):
    """04 论文单头 + TSM softmax TD 块（式 18-19），策略评估架构。

    prompt Z (D, n+1)，D = d_feat + 3：特征 d 行 + 奖励 1 行 + 双 memory 行
    （target 在前、current 在后，对应式 (10) 的 Z=[X; R; 0; 0]）。

    层更新（式 18-19）：
        Z_half = Z + V Z softmax(Z^T A Z + M)      # M 禁 query 列作 source
        target_new[:-1] = gamma * current_half[1:]   # TSM 列左移：target_j = γ·v(S_{j+1})
        其余行保留
    输出：current-memory 行 query 列 = v_L(S_n)（式 16，0-based 行 d_feat+2）。

    Theorem 1 的理想构造（式 12）作为默认初始化：
        A 左上 d×d = I_d（score 只看特征内积），V 最后一行 [r, γv', v] = [1, 1, -1]。
    """
    def __init__(self, d_feat, L, gamma, train_from_ideal=True, gain=0.05):
        super().__init__()
        self.d_feat = d_feat
        self.D = d_feat + 3
        self.L = L
        self.gamma = gamma
        self.V = nn.Parameter(torch.empty(self.D, self.D))
        self.A = nn.Parameter(torch.empty(self.D, self.D))
        self._init_ideal()
        if not train_from_ideal:
            with torch.no_grad():
                self.V.normal_(0.0, gain)
                self.A.normal_(0.0, gain)

    def _init_ideal(self):
        d, D = self.d_feat, self.D
        with torch.no_grad():
            self.A.zero_()
            self.A[:d, :d] = torch.eye(d)
            self.V.zero_()
            # V 最后一行（current 行 d+2）在 [r, target, current] 列上 = [1, 1, -1]
            self.V[d + 2, d] = 1.0
            self.V[d + 2, d + 1] = 1.0
            self.V[d + 2, d + 2] = -1.0

    def _attention(self, Z):
        scores = Z.transpose(-2, -1) @ self.A @ Z        # (n+1, n+1)
        ncols = scores.shape[-1]
        M = torch.zeros(ncols, ncols, device=scores.device)
        M[ncols - 1, :] = float("-inf")                  # query 列不作 source
        attn = torch.softmax(scores + M, dim=0)          # 列归一化
        return attn

    def _layer(self, Z):
        attn = self._attention(Z)
        Z_half = Z + self.V @ Z @ attn
        Z_new = Z_half.clone()
        target_new = torch.zeros_like(Z_half[self.d_feat + 1])
        # TSM 列左移：target_j = γ·current_{j+1}（target 取后继列值）
        target_new[:-1] = self.gamma * Z_half[self.d_feat + 2, 1:]
        Z_new[self.d_feat + 1] = target_new
        return Z_new

    def forward(self, Z, n):
        # Z (D, n+1)；unroll L 层共享参数
        for _ in range(self.L):
            Z = self._layer(Z)
        return Z[self.d_feat + 2, -1]                     # current-memory query 列


class SoftmaxSARSA(nn.Module):
    """C2：04 注意力机制 + 03 的 SARSA teacher 目标。

    prompt Z (D, n+1)，D = 3d+2（同 03 式 4）：
    - 行 0..2d：特征 [phi_i; gamma*phi+_i; r_{i+1}]（固定）
    - 行 2d+1..3d+1：w~ = [1; w]，只在 query 列（最后一列）非零

    与 Stage 1（SoftmaxAttnICRL）的本质区别：
    - A 只选特征行（I_{2d+1}），attention score 不依赖动态 w → 权重稳定
    - w 通过 V 的线性组合参与输出，不污染注意力
    输出：更新后的 w = Hout 后 d 行 query 列。

    默认 fix_A：A 固定为特征内积核（score = <x_i, x_j>），V 全参数可学。
    """
    def __init__(self, d, gain=0.1, temperature=1.0, fix_A=True):
        super().__init__()
        self.d = d
        self.D = 3 * d + 2
        self.temperature = temperature
        self.fix_A = fix_A
        self.V = nn.Parameter(torch.empty(self.D, self.D))
        nn.init.xavier_normal_(self.V, gain=gain)
        if fix_A:
            self.register_buffer("A", torch.zeros(self.D, self.D))
            self.A[: 2 * d + 1, : 2 * d + 1] = torch.eye(2 * d + 1)
        else:
            self.A = nn.Parameter(torch.zeros(self.D, self.D))
            with torch.no_grad():
                self.A[: 2 * d + 1, : 2 * d + 1] = torch.eye(2 * d + 1)

    def forward(self, H, n):
        scores = H.transpose(-2, -1) @ self.A @ H          # (n+1, n+1)
        if self.temperature != 1.0:
            scores = scores / self.temperature
        M = torch.zeros(scores.shape[-1], scores.shape[-1], device=H.device)
        M[-1, :] = float("-inf")                            # query 列不作 source
        attn = torch.softmax(scores + M, dim=0)             # 列归一化
        Hout = H + self.V @ H @ attn
        return Hout[2 * self.d + 2:, -1]


class SoftmaxEvalImprove(nn.Module):
    """C3（历史基线）：softmax TD 评估 + 块外 semi-gradient 读出。

    核心动机（numpy 验证结论）：
    - 单层 softmax attention（线性 V、固定核、w 不进 scores）无法精确实现
      SARSA 的 Δw = Σ_t δ_t φ_t（需 bilinear w^T φ + 逐项乘积聚合）。
    - 但"渐进在线 memory-TD 评估（跨帧，每帧一步）+ on-policy 半梯度改进"
      能闭环改进（verify_eval_online：0.12→0.23）。
    - 评估用 memory 行（w 无关）；改进 Δw = Σ δ_t φ_t 由 forward 末尾的
      显式张量读出完成，不属于 attention 矩阵内部运算。

    prompt Z (D, n+1)，D = d+3：
    - 行 0..d-1：特征 φ(S_t,A_t)（固定）
    - 行 d：奖励 R_{t+1}
    - 行 d+1：target memory（TSM 内部）
    - 行 d+2：current memory = v_state[S_t,A_t]（跨帧评估状态，输入给定）
    层更新（04 式 18-19）：
        Zhalf = Z + V Z softmax(Z^T A Z + M)   # M 禁 query 列作 source
        target ← γ·current（TSM 列左移）
    输出：
        v_pred = 更新后 current-memory 行（评估）
        dw = (α/n) Σ_t (R_{t+1} + γ v_{t+1} − v_t) φ(S_t,A_t)（改进）
    闭环：w ← w + dw；v_state ← v_pred（跨帧）。
    """
    def __init__(self, d, L, gamma, alpha, temperature=1.0):
        super().__init__()
        self.d = d
        self.D = d + 3
        self.L = L
        self.gamma = gamma
        self.alpha = alpha
        self.temperature = temperature
        self.V = nn.Parameter(torch.empty(self.D, self.D))
        self.A = nn.Parameter(torch.empty(self.D, self.D))
        nn.init.xavier_normal_(self.V, gain=0.05)
        with torch.no_grad():
            A0 = torch.zeros(self.D, self.D)
            A0[:d, :d] = torch.eye(d)          # 默认特征内积核
            self.A.data = A0
            self.V.data.zero_()

    def _attention(self, Z):
        scores = Z.transpose(-2, -1) @ self.A @ Z
        if self.temperature != 1.0:
            scores = scores / self.temperature
        ncols = scores.shape[-1]
        M = torch.zeros(ncols, ncols, device=scores.device)
        M[ncols - 1, :] = float("-inf")
        return torch.softmax(scores + M, dim=0)

    def _layer(self, Z):
        attn = self._attention(Z)
        Zhalf = Z + self.V @ Z @ attn
        Znew = Zhalf.clone()
        target_new = torch.zeros_like(Zhalf[self.d + 1])
        target_new[:-1] = self.gamma * Zhalf[self.d + 2, 1:]   # TSM 列左移
        Znew[self.d + 1] = target_new
        return Znew

    def forward(self, Z, n):
        for _ in range(self.L):
            Z = self._layer(Z)
        v = Z[self.d + 2, :]                    # (n+1,) 更新后 current mem
        phi = Z[:self.d, :n]                    # (d, n)
        r = Z[self.d, :n]                       # (n,)
        delta = r + self.gamma * v[1:] - v[:n]  # (n,) TD 残差（改进用）
        dw = (self.alpha / n) * (phi @ delta)   # (d,)
        return dw, v


class SoftmaxEvalImproveV2(nn.Module):
    """C3-v2（历史基线）：attention 评估 + 外部 max target/改进读出。

    历史实验发现该 Q-learning 路线优于当时的 on-policy 基线，但 E[δ]=0 at Q^π
    只表示策略评估完成，并不构成一般的“策略改进死锁”；SARSA、actor--critic 和
    evaluate-then-greedify 都可改进策略。这里的 max 仅服务于 Q-learning 路线。

    prompt Z (d+4, n+1)：
      行 0..d-1：特征 φ(S_t,A_t)（固定）
      行 d：奖励 R_{t+1}
      行 d+1：max 目标 M_t（由块外计算后作为输入给定）
      行 d+2：target mem（TSM 内部）
      行 d+3：current mem = v_state(S_t,A_t)
    层更新（TSM，同 04）：target ← γ·current 列左移。
    输出：
      v_pred = 更新后 current 行（评估）
      dw = (α/n) Σ (R_{t+1} + γ M_t − v_t) φ(S_t,A_t)（改进读出，fullv 形式用 v 行）
    teacher 监督 dw_tgt = (α/n) Σ (R_{t+1} + γ M_t − φ(S_t,A_t)·w) φ(S_t,A_t)
    （improve_max，含 φw）。因此该类是 modular baseline，不是层内 max 或 δφ 构造。
    """
    def __init__(self, d, L, gamma, alpha, temperature=1.0):
        super().__init__()
        self.d = d
        self.D = d + 4
        self.L = L
        self.gamma = gamma
        self.alpha = alpha
        self.temperature = temperature
        self.V = nn.Parameter(torch.empty(self.D, self.D))
        self.A = nn.Parameter(torch.empty(self.D, self.D))
        nn.init.xavier_normal_(self.V, gain=0.05)
        with torch.no_grad():
            A0 = torch.zeros(self.D, self.D)
            A0[:d, :d] = torch.eye(d)
            self.A.data = A0
            self.V.data.zero_()

    def _attention(self, Z):
        scores = Z.transpose(-2, -1) @ self.A @ Z
        if self.temperature != 1.0:
            scores = scores / self.temperature
        ncols = scores.shape[-1]
        M = torch.zeros(ncols, ncols, device=scores.device)
        M[ncols - 1, :] = float("-inf")
        return torch.softmax(scores + M, dim=0)

    def _layer(self, Z):
        attn = self._attention(Z)
        Zhalf = Z + self.V @ Z @ attn
        Znew = Zhalf.clone()
        target_new = torch.zeros_like(Zhalf[self.d + 2])
        target_new[:-1] = self.gamma * Zhalf[self.d + 3, 1:]
        Znew[self.d + 2] = target_new
        return Znew

    def forward(self, Z, n):
        for _ in range(self.L):
            Z = self._layer(Z)
        v = Z[self.d + 3, :]                    # (n+1,) current mem
        phi = Z[:self.d, :n]                    # (d, n)
        r = Z[self.d, :n]                       # (n,)
        M = Z[self.d + 1, :n]                   # (n,) max 目标
        delta = r + self.gamma * M - v[:n]      # (n,) fullv 形式读出
        dw = (self.alpha / n) * (phi @ delta)   # (d,)
        return dw, v


class SoftmaxEvalImproveV3(nn.Module):
    """C3-v4：同 V2，但 prompt 加 w 块（d 行）——测能否从 [φ,w] 重推导 φ·w。

    diag_selfeval：V2 模型是完美 copy 器（|v_pred−v_in|≈0.01），v_state 漂移是
    copy 误差累积。根因：w 不在输入，模型无法重推导 φ·w，只能复制 v_in。

    V3 加 w 块（D=2d+4），训练时 v_in 注入噪声、v_tgt=φ·w → 逼模型用 w 重推导。
    该特定单头布局中，value projection 为线性且 score kernel 默认只看特征；实验
    测量它能否重建 φ·w。vloss 平台只说明此布局/训练协议存在动态乘积瓶颈，不能
    推广成任意深度、多头或其他 tokenisation 的普遍不可能性。

    prompt Z (2d+4, n+1)：
      0..d-1：φ   d：r   d+1：M   d+2：target mem   d+3：current mem
      d+4..2d+3：w（d 行）
    """
    def __init__(self, d, L, gamma, alpha, temperature=1.0):
        super().__init__()
        self.d = d
        self.D = 2 * d + 4
        self.L = L
        self.gamma = gamma
        self.alpha = alpha
        self.temperature = temperature
        self.V = nn.Parameter(torch.empty(self.D, self.D))
        self.A = nn.Parameter(torch.empty(self.D, self.D))
        nn.init.xavier_normal_(self.V, gain=0.05)
        with torch.no_grad():
            A0 = torch.zeros(self.D, self.D)
            A0[:d, :d] = torch.eye(d)
            self.A.data = A0
            self.V.data.zero_()

    def _attention(self, Z):
        scores = Z.transpose(-2, -1) @ self.A @ Z
        if self.temperature != 1.0:
            scores = scores / self.temperature
        ncols = scores.shape[-1]
        M = torch.zeros(ncols, ncols, device=scores.device)
        M[ncols - 1, :] = float("-inf")
        return torch.softmax(scores + M, dim=0)

    def _layer(self, Z):
        attn = self._attention(Z)
        Zhalf = Z + self.V @ Z @ attn
        Znew = Zhalf.clone()
        target_new = torch.zeros_like(Zhalf[self.d + 2])
        target_new[:-1] = self.gamma * Zhalf[self.d + 3, 1:]
        Znew[self.d + 2] = target_new
        return Znew

    def forward(self, Z, n):
        for _ in range(self.L):
            Z = self._layer(Z)
        v = Z[self.d + 3, :]
        phi = Z[:self.d, :n]
        r = Z[self.d, :n]
        M = Z[self.d + 1, :n]
        delta = r + self.gamma * M - v[:n]
        dw = (self.alpha / n) * (phi @ delta)
        return dw, v


class SoftmaxEvalImproveV4(SoftmaxEvalImproveV2):
    """C3-v4：权重内凸组合读出（无 δ 线性缩放）。

    测：受限的 pure feature convex readout 能否拟合一般 signed update。
    读出直接用 attention 权重（query 列对 source 的 softmax 权重）做 φ 的凸组合，
    无 δ 前乘。§17.6 numpy 预测：纯凸组合（Σ softmax(δ/τ)φ，无 δ 缩放）死到 0.05
    级——单层 softmax 的读出天然是 softmax 凸组合，无法表达未归一化的 δ 缩放。

    对照：V2（块外线性读出 (α/n)Φᵀδ，活）、V5（块外 score-scaled，活）。
    若 V4 死而 V2/V5 活，只能说明裸特征凸读出缺少 signed carrier；Xie 型 value
    token 已携带 δ 的 softmax attention 不受这一反例排除。
    """
    def forward(self, Z, n):
        for _ in range(self.L):
            Z = self._layer(Z)
        v = Z[self.d + 3, :]
        phi = Z[:self.d, :n]
        r = Z[self.d, :n]
        M = Z[self.d + 1, :n]
        delta = r + self.gamma * M - v[:n]
        attn = self._attention(Z)               # 真实 attention 权重 (n+1, n+1)
        a = attn[-1, :n]                        # query 列对 source 的权重（凸组合）
        dw = self.alpha * (phi @ a)             # 无 δ 缩放
        return dw, v


class SoftmaxEvalImproveV5(SoftmaxEvalImproveV2):
    """C3-v5：块外 score-scaled softmax 读出（带 δ 缩放）。

    dw = α Σ_j δ_j · softmax(δ_j/τ) · φ_j，即 numpy att_td_lin 的真实块版：
    真实评估（v≈φ·w，attention 块内）+ 块外 score-scaled 改进读出。
    τ→∞ 退化为半梯度 (α/n)Σδφ（= V2）。测真实块能否闭环 score-scaled 改进
    （预期活，稠密 gridworld τ=10 达天花板 ~5.2-5.5）。
    """
    def forward(self, Z, n):
        for _ in range(self.L):
            Z = self._layer(Z)
        v = Z[self.d + 3, :]
        phi = Z[:self.d, :n]
        r = Z[self.d, :n]
        M = Z[self.d + 1, :n]
        delta = r + self.gamma * M - v[:n]
        d_ = delta - delta.max()
        a = torch.softmax(d_ / self.temperature, dim=0)
        dw = self.alpha * ((delta * a)[:, None] * phi.t()).sum(0)
        return dw, v


class SoftmaxEvalImproveV4b(SoftmaxEvalImproveV2):
    """C3-v4b：凸组合读出 + 可学全局缩放（归因诊断）。

    V4（凸组合，无缩放）闭环死 2.50 < random 3.15，但训练 loss→0（拟合成功）。
    归因问题：死因是量级（凸组合缺 (1/n) 步长）、方向（非负权重不能负贡献），
    还是闭环自举路径泛化？加可学全局缩放 γ（只修整体量级，不能逐 token 修符号）：
      - 若 V4b 活 → 量级主因，凸组合本身可行；
      - 若 V4b 仍死 → 逐 token 的 δ 线性缩放不可少（方向/泛化主导）。
    """
    def __init__(self, d, L, gamma, alpha, temperature=1.0):
        super().__init__(d, L, gamma, alpha, temperature)
        self.readout_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, Z, n):
        for _ in range(self.L):
            Z = self._layer(Z)
        v = Z[self.d + 3, :]
        phi = Z[:self.d, :n]
        r = Z[self.d, :n]
        M = Z[self.d + 1, :n]
        delta = r + self.gamma * M - v[:n]
        attn = self._attention(Z)
        a = attn[-1, :n]
        dw = self.readout_scale * (phi @ a)
        return dw, v


class SoftmaxAttnICRL(nn.Module):
    """04 论文式 softmax 注意力 ICRL 块（Stage 1）。

    H_out = H + V H softmax(H^T P H + M)
    - M 为掩码：query 列（最后一列）不作为 source，M[-1, :] = -inf
    - softmax 沿列归一化（dim=0），对应 04 论文式 2
    - 输出取最后一列的后 d 个元素（更新后的 w），与 03 兼容

    参数化默认沿用 03 的结构化形式（P12, V21 可学习），但注意：
    Proposition 3.1 是线性注意力的性质，softmax 下惰性子块未必保持为 0，
    因此提供 freeze_inert 开关，必要时放开全部参数（full=True）。
    """

    def __init__(self, d, gain=0.1, freeze_inert=True, full=False, temperature=1.0):
        super().__init__()
        self.d = d
        self.D = 3 * d + 2
        self.freeze_inert = freeze_inert
        self.temperature = temperature
        if full or not freeze_inert:
            self.P = nn.Parameter(torch.empty(self.D, self.D))
            self.V = nn.Parameter(torch.empty(self.D, self.D))
            nn.init.xavier_normal_(self.P, gain=gain)
            nn.init.xavier_normal_(self.V, gain=gain)
        else:
            self.P12 = nn.Parameter(torch.empty(2 * d + 1, d + 1))
            self.V21 = nn.Parameter(torch.empty(d, 2 * d + 1))
            nn.init.xavier_normal_(self.P12, gain=gain)
            nn.init.xavier_normal_(self.V21, gain=gain)
            self.register_buffer("P_full", torch.zeros(self.D, self.D))
            self.register_buffer("V_full", torch.zeros(self.D, self.D))

    def _assemble(self):
        if hasattr(self, "P"):  # full 模式
            return self.P, self.V
        P = self.P_full.clone()
        V = self.V_full.clone()
        P[: 2 * self.d + 1, 2 * self.d + 1:] = self.P12
        V[2 * self.d + 2:, : 2 * self.d + 1] = self.V21
        return P, V

    def forward(self, H, n):
        P, V = self._assemble()
        ncols = H.shape[-1]
        scores = H.transpose(-2, -1) @ P @ H          # (n+1, n+1)
        if self.temperature != 1.0:
            scores = scores / self.temperature
        M = torch.zeros(ncols, ncols, device=H.device)
        M[-1, :] = float("-inf")                       # query 列不作 source
        attn = torch.softmax(scores + M, dim=0)        # 列归一化（04 式 2）
        Hout = H + (V @ H) @ attn
        return Hout[-self.d:, -1]


class GroupedSoftmaxMax(nn.Module):
    """用标准 softmax attention 对每个下一状态的动作 Q token 做近似 max。

    输入 ``q_actions`` 的最后一维枚举动作。attention score 为 ``beta * Q``，
    value 也是 ``Q``，因此输出

        M_beta(Q) = sum_a softmax(beta * Q)_a * Q_a.

    该模块不调用 ``torch.max``；返回实际动作 attention，供公式级核验。
    """

    def __init__(self, beta=10.0):
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        self.beta = float(beta)

    def forward(self, q_actions, valid_mask=None):
        if q_actions.ndim < 1:
            raise ValueError("q_actions must have an action dimension")
        scores = self.beta * q_actions
        if valid_mask is not None:
            if valid_mask.shape != q_actions.shape:
                raise ValueError("valid_mask must have the same shape as q_actions")
            if not torch.all(valid_mask.any(dim=-1)):
                raise ValueError("each action group must contain at least one valid action")
            scores = scores.masked_fill(~valid_mask, float("-inf"))
        action_attn = torch.softmax(scores, dim=-1)
        softmax_max = (action_attn * q_actions).sum(dim=-1)
        return softmax_max, action_attn


class EndToEndMaskedSoftmaxSARSA(nn.Module):
    """从原始 transition 字段端到端实现精确 masked batch SARSA。

    两个 retrieval heads 以当前/下一 state-action one-hot 为 query，从完整
    Q-memory token 集合精确读取两个 Q 值。固定线性坐标投影形成 sampled-SARSA
    residual。write-back head 对同一当前 pair 的 residual 做 softmax 均值；
    未访问 query 只关注一个零值 null token。

    该模块没有可训练参数。布尔 mask 对应论文 Theorem 3.1 的结构化等值 mask。
    """

    def __init__(self, gamma=0.5, alpha=0.5):
        super().__init__()
        if not 0 <= gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    @staticmethod
    def _validate_inputs(
        q_values, states, actions, rewards, next_states, next_actions
    ):
        if q_values.ndim != 2:
            raise ValueError("q_values must have shape (n_states, n_actions)")
        vectors = (states, actions, rewards, next_states, next_actions)
        if any(vector.ndim != 1 for vector in vectors):
            raise ValueError("transition fields must be one-dimensional")
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("transition fields must have equal length")
        if len(states) == 0:
            raise ValueError("at least one transition is required")

        n_states, n_actions = q_values.shape
        integer_fields = (
            (states, n_states, "states"),
            (actions, n_actions, "actions"),
            (next_states, n_states, "next_states"),
            (next_actions, n_actions, "next_actions"),
        )
        for values, upper, name in integer_fields:
            if torch.any(values < 0) or torch.any(values >= upper):
                raise ValueError(f"{name} contain an out-of-range index")

    @staticmethod
    def _masked_singleton_retrieval(memory_values, pair_queries):
        """Standard softmax over the singleton memory source matching each query."""
        n_pairs = memory_values.numel()
        pair_axis = torch.arange(n_pairs, device=memory_values.device)
        allowed = pair_queries[:, None] == pair_axis[None, :]
        scores = torch.zeros(
            allowed.shape, dtype=memory_values.dtype, device=memory_values.device
        )
        scores = scores.masked_fill(~allowed, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        retrieved = attention @ memory_values
        return retrieved, attention

    def forward(self, q_values, states, actions, rewards, next_states, next_actions):
        self._validate_inputs(
            q_values, states, actions, rewards, next_states, next_actions
        )
        n_states, n_actions = q_values.shape
        states = states.long()
        actions = actions.long()
        next_states = next_states.long()
        next_actions = next_actions.long()
        rewards = rewards.to(dtype=q_values.dtype, device=q_values.device)

        current_pairs = states * n_actions + actions
        next_pairs = next_states * n_actions + next_actions
        memory_values = q_values.reshape(-1)

        current_q, current_attention = self._masked_singleton_retrieval(
            memory_values, current_pairs
        )
        next_q, next_attention = self._masked_singleton_retrieval(
            memory_values, next_pairs
        )
        td_values = rewards + self.gamma * next_q - current_q

        n_pairs = memory_values.numel()
        pair_axis = torch.arange(n_pairs, device=q_values.device)
        match = current_pairs[:, None] == pair_axis[None, :]
        visited = match.any(dim=0)

        # Context rows are followed by one zero-value null token. A visited query
        # admits exactly its matching transitions; an unvisited query admits only
        # the null token. Softmax is therefore always defined.
        allowed = torch.cat((match, (~visited)[None, :]), dim=0)
        scores = torch.zeros(
            allowed.shape, dtype=q_values.dtype, device=q_values.device
        )
        scores = scores.masked_fill(~allowed, float("-inf"))
        write_attention = torch.softmax(scores, dim=0)
        source_values = torch.cat(
            (td_values, torch.zeros(1, dtype=q_values.dtype, device=q_values.device))
        )
        signed_write = (source_values[:, None] * write_attention).sum(dim=0)
        update = self.alpha * signed_write
        q_new = memory_values + update

        diagnostics = {
            "current_pairs": current_pairs,
            "next_pairs": next_pairs,
            "current_q": current_q,
            "next_q": next_q,
            "td_values": td_values,
            "current_attention": current_attention,
            "next_attention": next_attention,
            "write_attention": write_attention,
            "visited": visited,
            "update": update.reshape_as(q_values),
        }
        return q_new.reshape_as(q_values), diagnostics


class EndToEndFiniteSoftmaxSARSA(nn.Module):
    """有限-logit 的 Q-retrieval + signed SARSA write-back 近似。

    Retrieval heads 在完整 Q-memory 上使用 one-hot dot-product softmax；
    write-back head 仅对已访问 query 发起有限-sharpness softmax。后者保留
    论文声明的 visited-query gate，因为有限未掩码 softmax 无法自行产生零更新。
    """

    def __init__(self, gamma=0.5, alpha=0.5, retrieval_beta=12.0, kernel_beta=12.0):
        super().__init__()
        if not 0 <= gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")
        if alpha <= 0 or retrieval_beta <= 0 or kernel_beta <= 0:
            raise ValueError("alpha and sharpness parameters must be positive")
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.retrieval_beta = float(retrieval_beta)
        self.kernel_beta = float(kernel_beta)

    def forward(self, q_values, states, actions, rewards, next_states, next_actions):
        EndToEndMaskedSoftmaxSARSA._validate_inputs(
            q_values, states, actions, rewards, next_states, next_actions
        )
        n_states, n_actions = q_values.shape
        states = states.long()
        actions = actions.long()
        next_states = next_states.long()
        next_actions = next_actions.long()
        rewards = rewards.to(dtype=q_values.dtype, device=q_values.device)

        current_pairs = states * n_actions + actions
        next_pairs = next_states * n_actions + next_actions
        memory_values = q_values.reshape(-1)
        n_pairs = memory_values.numel()
        features = torch.eye(n_pairs, dtype=q_values.dtype, device=q_values.device)

        memory_keys = features
        current_queries = features[current_pairs]
        next_queries = features[next_pairs]
        current_attention = torch.softmax(
            self.retrieval_beta * (current_queries @ memory_keys.transpose(0, 1)),
            dim=-1,
        )
        next_attention = torch.softmax(
            self.retrieval_beta * (next_queries @ memory_keys.transpose(0, 1)),
            dim=-1,
        )
        current_q = current_attention @ memory_values
        next_q = next_attention @ memory_values
        td_values = rewards + self.gamma * next_q - current_q

        unique_pairs = torch.unique(current_pairs, sorted=True)
        context_features = features[current_pairs]
        query_features = features[unique_pairs]
        scores = self.kernel_beta * (
            context_features @ query_features.transpose(0, 1)
        )
        write_attention = torch.softmax(scores, dim=0)
        signed_write = (td_values[:, None] * write_attention).sum(dim=0)

        q_new = memory_values.clone()
        update = self.alpha * signed_write
        q_new[unique_pairs] = q_new[unique_pairs] + update
        full_update = torch.zeros_like(memory_values)
        full_update[unique_pairs] = update

        diagnostics = {
            "current_pairs": current_pairs,
            "next_pairs": next_pairs,
            "current_q": current_q,
            "next_q": next_q,
            "td_values": td_values,
            "current_attention": current_attention,
            "next_attention": next_attention,
            "write_attention": write_attention,
            "query_pairs": unique_pairs,
            "update": full_update.reshape_as(q_values),
        }
        return q_new.reshape_as(q_values), diagnostics


class KernelizedSoftmaxQTD(nn.Module):
    """Xie 型 softmax kernel 对 signed TD values 的 action-value 写回。

    ``context_features`` 是产生 TD error 的状态—动作 token，
    ``query_features`` 是需要更新的状态—动作查询 token。对每个查询，softmax
    沿 context 维归一化；TD error 位于 value 中，可以为正也可以为负。
    """

    def __init__(self, alpha=0.5, kernel_beta=20.0):
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if kernel_beta <= 0:
            raise ValueError("kernel_beta must be positive")
        self.alpha = float(alpha)
        self.kernel_beta = float(kernel_beta)

    def forward(self, q_queries, td_values, context_features, query_features):
        if q_queries.ndim != 1 or td_values.ndim != 1:
            raise ValueError("q_queries and td_values must be one-dimensional")
        if context_features.ndim != 2 or query_features.ndim != 2:
            raise ValueError("context_features and query_features must be matrices")
        if context_features.shape[0] != td_values.shape[0]:
            raise ValueError("one context feature is required per TD value")
        if query_features.shape[0] != q_queries.shape[0]:
            raise ValueError("one query feature is required per Q query")
        if context_features.shape[1] != query_features.shape[1]:
            raise ValueError("context and query feature dimensions must match")

        scores = self.kernel_beta * (context_features @ query_features.transpose(0, 1))
        kernel_attn = torch.softmax(scores, dim=0)
        signed_write = (td_values[:, None] * kernel_attn).sum(dim=0)
        update = self.alpha * signed_write
        return q_queries + update, update, kernel_attn


class TwoStageSoftmaxQControl(nn.Module):
    """两阶段标准 softmax Bellman-optimality 近似。

    第一阶段在下一状态的动作 token 上计算 ``M_beta``；第二阶段把
    ``delta = r + gamma*M_beta - Q(s,a)`` 作为 signed value 做 kernel 写回。
    默认 one-hot 状态—动作特征使 kernel leakage 可直接由 sharpness 控制。
    """

    def __init__(self, gamma=0.5, alpha=0.5, action_beta=10.0, kernel_beta=20.0):
        super().__init__()
        if not 0 <= gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")
        self.gamma = float(gamma)
        self.softmax_max = GroupedSoftmaxMax(action_beta)
        self.kernel_qtd = KernelizedSoftmaxQTD(alpha, kernel_beta)

    @staticmethod
    def _default_features(q_values):
        n_queries = q_values.numel()
        return torch.eye(n_queries, dtype=q_values.dtype, device=q_values.device)

    def forward(
        self,
        q_values,
        context_sa,
        rewards,
        next_states,
        state_action_features=None,
    ):
        if q_values.ndim != 2:
            raise ValueError("q_values must have shape (n_states, n_actions)")
        if context_sa.ndim != 2 or context_sa.shape[1] != 2:
            raise ValueError("context_sa must have shape (n_context, 2)")
        if rewards.ndim != 1 or next_states.ndim != 1:
            raise ValueError("rewards and next_states must be vectors")
        if not (len(context_sa) == len(rewards) == len(next_states)):
            raise ValueError("context_sa, rewards and next_states must have equal length")

        n_states, n_actions = q_values.shape
        states = context_sa[:, 0].long()
        actions = context_sa[:, 1].long()
        next_states = next_states.long()
        if torch.any(states < 0) or torch.any(states >= n_states):
            raise ValueError("context state index out of range")
        if torch.any(actions < 0) or torch.any(actions >= n_actions):
            raise ValueError("context action index out of range")
        if torch.any(next_states < 0) or torch.any(next_states >= n_states):
            raise ValueError("next state index out of range")

        next_q_tokens = q_values[next_states]
        softmax_max, action_attn = self.softmax_max(next_q_tokens)
        current_q = q_values[states, actions]
        td_values = rewards + self.gamma * softmax_max - current_q

        if state_action_features is None:
            state_action_features = self._default_features(q_values)
        if state_action_features.ndim != 2 or state_action_features.shape[0] != q_values.numel():
            raise ValueError("state_action_features must have n_states*n_actions rows")
        flat_indices = states * n_actions + actions
        context_features = state_action_features[flat_indices]
        q_new_flat, update_flat, kernel_attn = self.kernel_qtd(
            q_values.reshape(-1), td_values, context_features, state_action_features
        )
        diagnostics = {
            "softmax_max": softmax_max,
            "action_attn": action_attn,
            "td_values": td_values,
            "kernel_attn": kernel_attn,
            "update": update_flat.reshape_as(q_values),
        }
        return q_new_flat.reshape_as(q_values), diagnostics


class OracleMaxKernelQTD(nn.Module):
    """与 ``TwoStageSoftmaxQControl`` 相同，但用外部 exact max 作 oracle 对照。"""

    def __init__(self, gamma=0.5, alpha=0.5, kernel_beta=20.0):
        super().__init__()
        if not 0 <= gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")
        self.gamma = float(gamma)
        self.kernel_qtd = KernelizedSoftmaxQTD(alpha, kernel_beta)

    def forward(
        self,
        q_values,
        context_sa,
        rewards,
        next_states,
        state_action_features=None,
    ):
        n_states, n_actions = q_values.shape
        states = context_sa[:, 0].long()
        actions = context_sa[:, 1].long()
        next_states = next_states.long()
        exact_max = torch.amax(q_values[next_states], dim=-1)
        td_values = rewards + self.gamma * exact_max - q_values[states, actions]
        if state_action_features is None:
            state_action_features = torch.eye(
                q_values.numel(), dtype=q_values.dtype, device=q_values.device
            )
        flat_indices = states * n_actions + actions
        q_new_flat, update_flat, kernel_attn = self.kernel_qtd(
            q_values.reshape(-1),
            td_values,
            state_action_features[flat_indices],
            state_action_features,
        )
        diagnostics = {
            "exact_max": exact_max,
            "td_values": td_values,
            "kernel_attn": kernel_attn,
            "update": update_flat.reshape_as(q_values),
        }
        return q_new_flat.reshape_as(q_values), diagnostics


class LiangLaiOperatorBaseline(nn.Module):
    """Liang–Lai 定理对应的参数化 semi-gradient operator baseline。

    该类核验 ``Delta w = alpha/n * sum delta_i phi_i``，不声称复现其端到端
    teacher-mimicking 训练动力学。
    """

    def __init__(self, gamma=0.5, alpha=0.5):
        super().__init__()
        if not 0 <= gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def forward(self, w, phi, phi_next, rewards):
        if phi.ndim != 2 or phi_next.shape != phi.shape:
            raise ValueError("phi and phi_next must have the same matrix shape")
        if w.ndim != 1 or phi.shape[1] != w.shape[0]:
            raise ValueError("w dimension must match the feature dimension")
        if rewards.ndim != 1 or rewards.shape[0] != phi.shape[0]:
            raise ValueError("one reward is required per transition")
        td_values = rewards + self.gamma * (phi_next @ w) - (phi @ w)
        delta_w = (self.alpha / phi.shape[0]) * (phi.transpose(0, 1) @ td_values)
        return w + delta_w, {"td_values": td_values, "delta_w": delta_w}


class FixedPolicyActionExpectation(nn.Module):
    """精确策略动作期望头：qbar_t = sum_a pi(a | s_next) Q(s_next, a)。

    对每个下一状态 query，attention 权重只落在该状态的动作 token 上，权重
    等于策略概率。该头无 mask、无可训练参数，供 Expected SARSA 的后继项使用。
    """

    def forward(self, q_values, next_states, policy):
        if q_values.ndim != 2:
            raise ValueError("q_values must have shape (n_states, n_actions)")
        n_states, n_actions = q_values.shape
        if policy.shape != q_values.shape:
            raise ValueError("policy must share the Q shape")
        next_states = next_states.long()
        if torch.any(next_states < 0) or torch.any(next_states >= n_states):
            raise ValueError("next_states contain an out-of-range index")

        n_transitions = next_states.shape[0]
        one_hot = torch.zeros(
            (n_transitions, n_states), dtype=q_values.dtype, device=q_values.device
        )
        one_hot.scatter_(1, next_states[:, None], 1.0)
        next_policy = policy[next_states]
        attention = (one_hot[:, :, None] * next_policy[:, None, :]).reshape(
            n_transitions, n_states * n_actions
        )
        memory_values = q_values.reshape(-1)
        return attention @ memory_values, attention


class EndToEndMaskedSoftmaxExpectedSARSA(nn.Module):
    """精确 masked batch Expected SARSA 的端到端实现。

    两个 retrieval heads 以当前 / 下一状态-action one-hot 为 query，从完整
    Q-memory token 集合精确读取 Q 值。后继值由精确策略动作期望头给出，而不是
    采样动作。write-back head 对同一当前 pair 的 Expected-SARSA residual 做
    softmax 均值；未访问 query 只关注一个零值 null token。

    该模块没有可训练参数。布尔 mask 对应论文 Theorem 3.1 的结构化等值 mask。
    """

    def __init__(self, gamma=0.5, alpha=0.5):
        super().__init__()
        if not 0 <= gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    @staticmethod
    def _validate_inputs(q_values, states, actions, rewards, next_states):
        if q_values.ndim != 2:
            raise ValueError("q_values must have shape (n_states, n_actions)")
        vectors = (states, actions, rewards, next_states)
        if any(vector.ndim != 1 for vector in vectors):
            raise ValueError("transition fields must be one-dimensional")
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("transition fields must have equal length")
        if len(states) == 0:
            raise ValueError("at least one transition is required")

        n_states, n_actions = q_values.shape
        integer_fields = (
            (states, n_states, "states"),
            (actions, n_actions, "actions"),
            (next_states, n_states, "next_states"),
        )
        for values, upper, name in integer_fields:
            if torch.any(values < 0) or torch.any(values >= upper):
                raise ValueError(f"{name} contain an out-of-range index")

    @staticmethod
    def _masked_singleton_retrieval(memory_values, pair_queries):
        """Standard softmax over the singleton memory source matching each query."""
        n_pairs = memory_values.numel()
        pair_axis = torch.arange(n_pairs, device=memory_values.device)
        allowed = pair_queries[:, None] == pair_axis[None, :]
        scores = torch.zeros(
            allowed.shape, dtype=memory_values.dtype, device=memory_values.device
        )
        scores = scores.masked_fill(~allowed, float("-inf"))
        attention = torch.softmax(scores, dim=-1)
        return attention @ memory_values, attention

    def forward(self, q_values, states, actions, rewards, next_states, policy):
        self._validate_inputs(q_values, states, actions, rewards, next_states)
        n_states, n_actions = q_values.shape
        states = states.long()
        actions = actions.long()
        next_states = next_states.long()
        rewards = rewards.to(dtype=q_values.dtype, device=q_values.device)
        policy = policy.to(dtype=q_values.dtype, device=q_values.device)

        current_pairs = states * n_actions + actions
        memory_values = q_values.reshape(-1)
        current_q, current_attention = self._masked_singleton_retrieval(
            memory_values, current_pairs
        )
        qbar, action_attention = FixedPolicyActionExpectation()(
            q_values, next_states, policy
        )
        residuals = rewards + self.gamma * qbar - current_q

        n_pairs = memory_values.numel()
        pair_axis = torch.arange(n_pairs, device=q_values.device)
        match = current_pairs[:, None] == pair_axis[None, :]
        visited = match.any(dim=0)

        # Context rows are followed by one zero-value null token. A visited query
        # admits exactly its matching transitions; an unvisited query admits only
        # the null token. Softmax is therefore always defined.
        allowed = torch.cat((match, (~visited)[None, :]), dim=0)
        scores = torch.zeros(
            allowed.shape, dtype=q_values.dtype, device=q_values.device
        )
        scores = scores.masked_fill(~allowed, float("-inf"))
        write_attention = torch.softmax(scores, dim=0)
        source_values = torch.cat(
            (residuals, torch.zeros(1, dtype=q_values.dtype, device=q_values.device))
        )
        signed_write = (source_values[:, None] * write_attention).sum(dim=0)
        update = self.alpha * signed_write
        q_new = memory_values + update

        diagnostics = {
            "current_pairs": current_pairs,
            "current_q": current_q,
            "qbar": qbar,
            "residuals": residuals,
            "current_attention": current_attention,
            "action_attention": action_attention,
            "write_attention": write_attention,
            "visited": visited,
            "update": update.reshape_as(q_values),
        }
        return q_new.reshape_as(q_values), diagnostics


class EndToEndFiniteSoftmaxExpectedSARSA(nn.Module):
    """有限-logit 的 Expected SARSA 实现：无等值 mask、无 visited gate。

    后继、读取、写回三处都使用有限 sharpness 的 softmax，覆盖完整 token 集合，
    因此每个 attention 都是满支撑的严格正权重；未访问 query 也会收到可报告的
    泄漏更新。该模块没有可训练参数。
    """

    def __init__(self, gamma=0.5, alpha=0.5, zeta=8.0, xi=8.0, tau=8.0):
        super().__init__()
        if not 0 <= gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")
        if alpha <= 0 or zeta <= 0 or xi <= 0 or tau <= 0:
            raise ValueError("alpha and sharpness parameters must be positive")
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.zeta = float(zeta)
        self.xi = float(xi)
        self.tau = float(tau)

    def forward(self, q_values, states, actions, rewards, next_states, policy):
        EndToEndMaskedSoftmaxExpectedSARSA._validate_inputs(
            q_values, states, actions, rewards, next_states
        )
        n_states, n_actions = q_values.shape
        n_pairs = n_states * n_actions
        states = states.long()
        actions = actions.long()
        next_states = next_states.long()
        rewards = rewards.to(dtype=q_values.dtype, device=q_values.device)
        policy = policy.to(dtype=q_values.dtype, device=q_values.device)

        memory_values = q_values.reshape(-1)
        current_pairs = states * n_actions + actions
        n_transitions = states.shape[0]
        token_axis = torch.arange(n_pairs, device=q_values.device)

        # Successor: finite sharpness on the matched next-state block.
        log_policy = torch.log(policy).reshape(-1)
        successor_block = token_axis // n_actions
        successor_mask = (successor_block[None, :] == next_states[:, None]).to(
            q_values.dtype
        )
        successor_scores = log_policy[None, :] + self.zeta * successor_mask
        action_attention = torch.softmax(successor_scores, dim=1)
        successor = action_attention @ memory_values

        # Read: finite sharpness on the matched current-pair token.
        read_mask = (token_axis[None, :] == current_pairs[:, None]).to(q_values.dtype)
        read_attention = torch.softmax(self.xi * read_mask, dim=1)
        read = read_attention @ memory_values

        residuals = rewards + self.gamma * successor - read

        # Write: full-support finite kernel over every query, no visited gate.
        match = (current_pairs[:, None] == token_axis[None, :]).to(q_values.dtype)
        counts = match.sum(dim=0)
        numerator = torch.exp(self.tau * match)
        denominator = counts * math.exp(self.tau) + (n_transitions - counts)
        write_attention = numerator / denominator[None, :]
        update = self.alpha * (write_attention.transpose(0, 1) @ residuals)
        q_new = memory_values + update

        diagnostics = {
            "current_pairs": current_pairs,
            "successor": successor,
            "read": read,
            "residuals": residuals,
            "action_attention": action_attention,
            "read_attention": read_attention,
            "write_attention": write_attention,
            "update": update.reshape_as(q_values),
        }
        return q_new.reshape_as(q_values), diagnostics
