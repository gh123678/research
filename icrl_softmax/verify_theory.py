"""数值核验论文理论章节的有限温度界与表格均值递推。

对应 preliminaries.md 新增的命题，每个命题打印一组可写进论文的数字：

  Lemma 3.1   softmax(x/τ)_j = 1/n + (x_j−x̄)/(nτ) + O(1/τ²)
              → 展开误差随 τ 翻倍下降约 4 倍（∝ 1/τ²）。
  Theorem 3.2 g_τ = Σ_j δ_j·softmax(δ/τ)_j·φ_j → (1/n)Σ_j δ_j φ_j（τ→∞）
              → 相对误差随 τ 单调下降至 0。
  Prop 3.3    凸组合读出 ĝ = Σ a_j φ_j 无法表达带符号半梯度目标
              → 反例 φ=(1,2), δ=(1,−1) 得 g_∞=−1/2 ∉ [1,2]。
  Corollary 3.4  有限温度的方向充分条件与 Lipschitz 性。
  Theorem 3.7  固定温度的扰动 Bellman 递推进入 O(1/tau) 邻域；
                 可和的升温日程回到无扰动极限。

数值输出是公式和常数的 sanity check，不替代 markdown 中的证明。
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis, keepdims=True)


def main():
    rng = np.random.default_rng(0)
    n, d = 20, 36

    # ---- Lemma 3.1: 一阶展开误差 ∝ 1/τ² ----
    print("Lemma 3.1: softmax(x/tau)_j = 1/n + (x_j-xbar)/(n*tau) + O(1/tau^2)")
    x = rng.normal(size=n)
    xbar = x.mean()
    print("   tau   |  max|softmax - approx|   |  err_prev/err")
    prev = None
    for tau in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
        a = softmax(x / tau)
        approx = 1 / n + (x - xbar) / (n * tau)
        err = float(np.abs(a - approx).max())
        ratio = (prev / err) if prev else float("nan")
        print("  %6.1f |   %18.6e   |   %8.2f" % (tau, err, ratio))
        prev = err
    print("  (tau 翻倍 -> 误差应降约 4 倍，即 ratio≈4)")

    # ---- Theorem 3.2: g_τ → (1/n)Σδφ ----
    print("\nTheorem 3.2: g_tau = sum_j delta_j*softmax(delta/tau)_j*phi_j -> (1/n)sum delta_j phi_j")
    delta = rng.normal(size=n)
    phi = rng.normal(size=(n, d))
    g_inf = (1.0 / n) * (delta @ phi)
    print("   tau   |  rel err ||g_tau - g_inf|| / ||g_inf||")
    for tau in [0.3, 1.0, 5.0, 20.0, 200.0, 2000.0]:
        a = softmax(delta / tau)
        g_tau = (delta * a) @ phi
        rel = float(np.linalg.norm(g_tau - g_inf) / np.linalg.norm(g_inf))
        print("  %6.1f |   %26.6e" % (tau, rel))
    print("  (tau -> inf 时相对误差 -> 0)")

    # ---- Proposition 3.3: 凸组合死锁反例 ----
    print("\nProposition 3.3: convex-combination readout cannot express a signed update")
    phi_c = np.array([1.0, 2.0])
    delta_c = np.array([1.0, -1.0])
    g_inf_c = 0.5 * (delta_c @ phi_c)
    print("  phi=(1,2), delta=(1,-1) -> g_inf = (1*1 + (-1)*2)/2 = %.4f" % g_inf_c)
    print("  conv{1,2} = [1,2];  g_inf=%.4f is %s [1,2]" %
          (g_inf_c, "outside" if (g_inf_c < 1.0 or g_inf_c > 2.0) else "inside"))
    print("  any convex readout g_hat = a*1 + (1-a)*2 lies in [1,2], never %.4f" % g_inf_c)

    # ---- Proposition 3.4: 有界 vs 无界 ----
    print("\nProposition 3.4: softmax output is bounded; phi^T w is unbounded")
    v = rng.uniform(0.0, 1.0, size=(n, d))
    a = rng.random(n)
    a /= a.sum()
    o = a @ v
    print("  convex output coord range [%.3f, %.3f] inside value range [%.3f, %.3f]"
          % (o.min(), o.max(), v.min(), v.max()))
    phi_p = rng.normal(size=d)
    for wscale in [1.0, 10.0, 100.0]:
        w = wscale * rng.normal(size=d)
        print("  ||w|| ~ %5.1f  ->  phi^T w = %10.2f   (unbounded in w)"
              % (wscale, float(phi_p @ w)))

    # ---- finite-temperature bound / direction / Lipschitz ----
    print("\nFinite-temperature bounds (Theorem 3.2 / Corollary 3.4)")
    B = float(np.abs(delta).max())
    F = float(np.linalg.norm(phi, axis=1).max())
    for tau in [2 * B, 5 * B, 20 * B]:
        p = softmax(delta / tau)
        g_tau = (delta * p) @ phi
        err = float(np.linalg.norm(g_tau - g_inf))
        bound = F * B * (np.exp(2 * B / tau) - 1)
        dot = float(g_tau @ g_inf)
        print("  tau=%7.3f  err=%9.3e <= bound=%9.3e  <g_tau,g_inf>=%9.3e"
              % (tau, err, bound, dot))

    delta2 = np.clip(delta + 0.05 * rng.normal(size=n), -B, B)
    tau = 5 * B
    lhs = float(np.linalg.norm((delta * softmax(delta / tau)) @ phi -
                               (delta2 * softmax(delta2 / tau)) @ phi))
    rhs = F * (1 + 2 * B / tau) * float(np.abs(delta - delta2).max())
    print("  Lipschitz check: %9.3e <= %9.3e" % (lhs, rhs))

    # ---- perturbed Bellman recursion: a scalar gamma-contraction ----
    print("\nTheorem 3.7 scalar Bellman-perturbation sanity check")
    gamma, alpha, eta = 0.8, 0.2, 0.04
    q = 1.0  # Q*=0 and TQ=gamma Q
    for _ in range(400):
        q = (1 - alpha) * q + alpha * gamma * q + alpha * eta
    print("  fixed perturbation: |Q-Q*|=%8.5f, theorem radius eta/(1-gamma)=%8.5f"
          % (abs(q), eta / (1 - gamma)))

    q = 1.0
    bias_mass = 0.0
    for k in range(1, 20001):
        a = 1.0 / (k + 20) ** 0.7
        tau_k = (k + 20) ** 0.5
        e = 1.0 / tau_k
        bias_mass += a * e
        q = (1 - a) * q + a * gamma * q + a * e
    print("  annealed perturbation: |Q-Q*|=%8.5f, accumulated alpha/tau=%8.5f"
          % (abs(q), bias_mass))


if __name__ == "__main__":
    main()
