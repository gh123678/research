"""随机表格 MDP、特征与轨迹采样，对齐 03 论文（Liang & Lai, arXiv:2605.05755）。

任务配置（03 论文 Section 4.1）：
- nS=9 状态, nA=4 动作, gamma=0.5
- 转移 P(·|s,a) ~ Dirichlet(1,...,1)
- 奖励 r(s,a,s') ~ Unif(-1,1)
- 初始分布 p0 ~ Dirichlet(1,...,1)
- 随机特征 phi: S x A -> R^d, d = nS*nA, 元素 Unif(-1,1)
"""
import numpy as np


def sample_mdp(nS, nA, gamma, rng=None):
    """采样一个随机 MDP，返回 dict：P (nS,nA,nS), R (nS,nA,nS), p0 (nS,)。"""
    if rng is None:
        rng = np.random.default_rng()
    P = rng.dirichlet(np.ones(nS), size=(nS, nA)).astype(np.float32)
    R = rng.uniform(-1.0, 1.0, size=(nS, nA, nS)).astype(np.float32)
    p0 = rng.dirichlet(np.ones(nS)).astype(np.float32)
    return {"nS": nS, "nA": nA, "gamma": gamma, "P": P, "R": R, "p0": p0}


def sample_features(nS, nA, d, rng=None):
    """每个 MDP 独立采样的随机特征 phi: (nS, nA, d)，元素 Unif(-1,1)。"""
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(-1.0, 1.0, size=(nS, nA, d)).astype(np.float32)


def eps_greedy_probs(phi, w, eps):
    """epsilon-greedy 行为策略概率矩阵 (nS, nA)，Q(s,a)=phi[s,a] @ w。"""
    nS, nA = phi.shape[0], phi.shape[1]
    Q = phi @ w  # (nS, nA)
    greedy = np.argmax(Q, axis=1)
    probs = np.full((nS, nA), eps / nA, dtype=np.float32)
    probs[np.arange(nS), greedy] += (1.0 - eps)
    return probs


def rollout(mdp, probs, start, n, rng=None):
    """采 n 步轨迹。

    返回 S (n+1,), A (n+1,), Rew (n+1,)：
    - S[t], A[t] 是第 t 时刻状态/动作，S[0]=start
    - A[n] 是在 S[n] 上额外采的动作（供 phi_plus_{n-1} 使用）
    - Rew[t] = r_t（Rew[0]=0 为占位）
    """
    if rng is None:
        rng = np.random.default_rng()
    P, R = mdp["P"], mdp["R"]
    nS, nA = mdp["nS"], mdp["nA"]
    S = np.zeros(n + 1, dtype=np.int64)
    A = np.zeros(n + 1, dtype=np.int64)
    Rew = np.zeros(n + 1, dtype=np.float32)
    S[0] = start
    for t in range(n):
        a = rng.choice(nA, p=probs[S[t]])
        A[t] = a
        s_next = rng.choice(nS, p=P[S[t], a])
        S[t + 1] = s_next
        Rew[t + 1] = R[S[t], a, s_next]
    A[n] = rng.choice(nA, p=probs[S[n]])
    return S, A, Rew


def sample_mrp_chain(N, gamma, d, rng=None):
    """04 论文风格的确定性链式 MRP（Boyan 风格简化版，策略评估设置）。

    状态 0..N-1，转移 s->s+1 确定性（N-1 吸收），奖励 r(s) 随机 Unif(-1,1)，
    特征 x(s) 随机 Unif(-1,1)^d。返回解析价值 v(s)=sum_t gamma^t r(s+t)。
    """
    if rng is None:
        rng = np.random.default_rng()
    r = rng.uniform(-1.0, 1.0, size=N).astype(np.float32)
    x = rng.uniform(-1.0, 1.0, size=(N, d)).astype(np.float32)
    v = np.zeros(N, dtype=np.float64)
    for s in range(N):
        acc, g = 0.0, 1.0
        for t in range(N - s):
            acc += g * r[min(s + t, N - 1)]
            g *= gamma
        v[s] = acc
    return {"N": N, "gamma": gamma, "r": r, "x": x, "v": v.astype(np.float32)}


def mrp_chain_rollout(mrp, n, start=0):
    """确定性链 MRP 上采 n 个 transition。

    返回 S (n+1,), R (n,)：S[0]=start，S[t+1]=min(S[t]+1,N-1)，
    R[t]=r(S[t])（04 式(10)：R_k = r(S_{k-1})，MRP 奖励只依赖出发状态）。
    """
    N = mrp["N"]
    r = mrp["r"]
    S = np.zeros(n + 1, dtype=np.int64)
    S[0] = start
    for t in range(n):
        S[t + 1] = min(S[t] + 1, N - 1)
    R = r[S[:n]].astype(np.float32)
    return S, R
