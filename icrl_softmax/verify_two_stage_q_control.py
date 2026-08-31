"""公式级核验：两阶段 softmax Q-control 的每个算子与理论界。"""

import math

import torch

from model import (
    GroupedSoftmaxMax,
    KernelizedSoftmaxQTD,
    LiangLaiOperatorBaseline,
    OracleMaxKernelQTD,
    TwoStageSoftmaxQControl,
)


def check_softmax_max_bound():
    torch.manual_seed(0)
    q = 3.0 * torch.randn(128, 7, dtype=torch.float64)
    for beta in (0.5, 1.0, 3.0, 10.0, 30.0):
        value, attn = GroupedSoftmaxMax(beta)(q)
        gap = torch.amax(q, dim=-1) - value
        bound = math.log(q.shape[-1]) / beta
        assert torch.all(gap >= -1e-12)
        assert float(gap.max()) <= bound + 1e-12
        assert torch.allclose(attn.sum(-1), torch.ones(128, dtype=q.dtype), atol=1e-12)
    print("PASS softmax-max entropy bound")


def check_signed_kernel_write():
    q = torch.zeros(2, dtype=torch.float64)
    td = torch.tensor([2.0, -3.0], dtype=torch.float64)
    eye = torch.eye(2, dtype=torch.float64)
    q_new, update, attn = KernelizedSoftmaxQTD(alpha=1.0, kernel_beta=40.0)(
        q, td, eye, eye
    )
    assert torch.allclose(attn.sum(0), torch.ones(2, dtype=q.dtype), atol=1e-12)
    assert update[0] > 0 and update[1] < 0
    assert torch.allclose(q_new, td, atol=1e-12)
    print("PASS signed TD values survive positive kernel weights")


def full_context(n_states, n_actions):
    return torch.tensor(
        [(s, a) for s in range(n_states) for a in range(n_actions)], dtype=torch.long
    )


def check_two_stage_oracle_gap():
    torch.manual_seed(1)
    dtype = torch.float64
    n_states, n_actions = 5, 4
    q = torch.randn(n_states, n_actions, dtype=dtype)
    context = full_context(n_states, n_actions)
    rewards = torch.randn(len(context), dtype=dtype)
    next_states = torch.arange(len(context)) % n_states
    alpha, gamma, beta = 0.4, 0.7, 6.0
    internal = TwoStageSoftmaxQControl(
        gamma=gamma, alpha=alpha, action_beta=beta, kernel_beta=40.0
    )
    oracle = OracleMaxKernelQTD(gamma=gamma, alpha=alpha, kernel_beta=40.0)
    q_internal, d_internal = internal(q, context, rewards, next_states)
    q_oracle, _ = oracle(q, context, rewards, next_states)
    theoretical = alpha * gamma * math.log(n_actions) / beta
    actual = float(torch.max(torch.abs(q_internal - q_oracle)))
    assert actual <= theoretical + 1e-10
    assert d_internal["action_attn"].shape == (len(context), n_actions)
    assert d_internal["kernel_attn"].shape == (len(context), n_states * n_actions)
    print(f"PASS internal/oracle gap: actual={actual:.6f}, bound={theoretical:.6f}")


def check_liang_lai_operator():
    torch.manual_seed(2)
    dtype = torch.float64
    n, d = 11, 6
    w = torch.randn(d, dtype=dtype)
    phi = torch.randn(n, d, dtype=dtype)
    phi_next = torch.randn(n, d, dtype=dtype)
    rewards = torch.randn(n, dtype=dtype)
    gamma, alpha = 0.6, 0.3
    w_new, diagnostics = LiangLaiOperatorBaseline(gamma, alpha)(
        w, phi, phi_next, rewards
    )
    delta = rewards + gamma * (phi_next @ w) - phi @ w
    expected = w + (alpha / n) * phi.T @ delta
    assert torch.allclose(w_new, expected, atol=1e-12)
    assert torch.allclose(diagnostics["td_values"], delta, atol=1e-12)
    print("PASS Liang-Lai operator equivalence")


if __name__ == "__main__":
    check_softmax_max_bound()
    check_signed_kernel_write()
    check_two_stage_oracle_gap()
    check_liang_lai_operator()
    print("ALL TWO-STAGE Q-CONTROL CHECKS PASSED")
