"""Formula-level verification for Direct-Q and V-first fixed-policy routes.

This script is intentionally small and deterministic. It checks the contracts in
the two theory notes without claiming a finite-sample theorem:

1. fixed-policy Q is the value function of the state-action Markov chain;
2. the Direct-Q population operator fixes Q^pi and obeys the diagonal bound;
3. finite softmax pair write-back stays inside its one-step leakage bound;
4. population V-to-Q recovery is gamma-Lipschitz;
5. the sample-split ratio decomposition covers the observed recovery error;
6. a sufficiently rare action can remain unobserved, exposing the coverage barrier.
"""

from __future__ import annotations

import numpy as np

from mdps import rollout, sample_mdp


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Solve mu^T P = mu^T for a finite ergodic row-stochastic matrix."""
    n = transition.shape[0]
    system = transition.T - np.eye(n)
    rhs = np.zeros(n)
    system[-1] = 1.0
    rhs[-1] = 1.0
    mu = np.linalg.solve(system, rhs)
    mu = np.maximum(mu, 0.0)
    return mu / mu.sum()


def policy_quantities(
    mdp: dict[str, np.ndarray | int | float], policy: np.ndarray
) -> dict[str, np.ndarray]:
    """Return exact P^pi, V^pi, Q^pi, pair transition, and occupancies."""
    p = np.asarray(mdp["P"], dtype=np.float64)
    p = p / p.sum(axis=2, keepdims=True)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    gamma = float(mdp["gamma"])
    n_states = int(mdp["nS"])
    n_actions = int(mdp["nA"])

    p_pi = np.einsum("sa,san->sn", policy, p)
    reward_sa = np.sum(p * reward, axis=2)
    reward_pi = np.sum(policy * reward_sa, axis=1)
    v_pi = np.linalg.solve(np.eye(n_states) - gamma * p_pi, reward_pi)
    q_pi = reward_sa + gamma * np.einsum("san,n->sa", p, v_pi)

    n_pairs = n_states * n_actions
    p_pair = np.zeros((n_pairs, n_pairs), dtype=np.float64)
    for state in range(n_states):
        for action in range(n_actions):
            source = state * n_actions + action
            for next_state in range(n_states):
                start = next_state * n_actions
                p_pair[source, start : start + n_actions] = (
                    p[state, action, next_state] * policy[next_state]
                )

    mu_state = stationary_distribution(p_pi)
    mu_pair_formula = (mu_state[:, None] * policy).reshape(-1)
    mu_pair_solved = stationary_distribution(p_pair)

    return {
        "p_pi": p_pi,
        "reward_sa": reward_sa,
        "v_pi": v_pi,
        "q_pi": q_pi,
        "p_pair": p_pair,
        "mu_state": mu_state,
        "mu_pair": mu_pair_formula,
        "mu_pair_solved": mu_pair_solved,
    }


def one_hot_population_kernel(mu: np.ndarray, sharpness: float) -> np.ndarray:
    """M(x,z) proportional to mu(z) exp(sharpness * 1{x=z})."""
    n = mu.size
    weights = np.broadcast_to(mu, (n, n)).copy()
    weights[np.arange(n), np.arange(n)] *= np.exp(sharpness)
    return weights / weights.sum(axis=1, keepdims=True)


def required_sharpness(mu: float, diagonal_target: float) -> float:
    return float(
        np.log(
            diagonal_target
            * (1.0 - mu)
            / ((1.0 - diagonal_target) * mu)
        )
    )


def direct_q_population_update(
    q: np.ndarray,
    reward_pair: np.ndarray,
    p_pair: np.ndarray,
    kernel: np.ndarray,
    gamma: float,
    alpha: float,
) -> np.ndarray:
    residual = reward_pair + gamma * (p_pair @ q) - q
    return q + alpha * (kernel @ residual)


def exact_pair_mean_update(
    q: np.ndarray,
    current_pairs: np.ndarray,
    next_pairs: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residuals = rewards + gamma * q[next_pairs] - q[current_pairs]
    updated = q.copy()
    counts = np.bincount(current_pairs, minlength=q.size)
    for pair in np.flatnonzero(counts):
        updated[pair] += alpha * residuals[current_pairs == pair].mean()
    return updated, residuals, counts


def finite_pair_kernel_update(
    q: np.ndarray,
    current_pairs: np.ndarray,
    residuals: np.ndarray,
    sharpness: float,
    alpha: float,
) -> np.ndarray:
    updated = q.copy()
    visited = np.unique(current_pairs)
    for query in visited:
        logits = sharpness * (current_pairs == query).astype(np.float64)
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        updated[query] += alpha * float(weights @ residuals)
    return updated


def tabular_v_from_context(
    states: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    n_states: int,
    iterations: int = 200,
) -> np.ndarray:
    """Repeated exact state matching: the tabular special case of softmax V-TD."""
    current = states[:-1]
    following = states[1:]
    counts = np.bincount(current, minlength=n_states)
    if np.any(counts == 0):
        raise AssertionError("V context must cover every state in this fixture")

    value = np.zeros(n_states, dtype=np.float64)
    for _ in range(iterations):
        residuals = rewards + gamma * value[following] - value[current]
        next_value = value.copy()
        for state in range(n_states):
            next_value[state] += residuals[current == state].mean()
        value = next_value
    return value


def main() -> None:
    rng = np.random.default_rng(20260829)
    gamma = 0.70
    mdp = sample_mdp(nS=3, nA=2, gamma=gamma, rng=rng)
    policy = np.array(
        [[0.72, 0.28], [0.61, 0.39], [0.77, 0.23]], dtype=np.float64
    )
    exact = policy_quantities(mdp, policy)

    q_pi = exact["q_pi"].reshape(-1)
    v_pi = exact["v_pi"]
    p_pair = exact["p_pair"]
    reward_pair = exact["reward_sa"].reshape(-1)
    mu_pair = exact["mu_pair"]

    pair_stationary_error = float(
        np.max(np.abs(exact["mu_pair"] - exact["mu_pair_solved"]))
    )
    bellman_error = float(
        np.max(np.abs(q_pi - (reward_pair + gamma * (p_pair @ q_pi))))
    )
    assert pair_stationary_error < 1e-10
    assert bellman_error < 1e-10

    print("[A1] state-action MRP")
    print(f"  stationary formula error: {pair_stationary_error:.3e}")
    print(f"  Q^pi Bellman error:       {bellman_error:.3e}")

    margin = 0.04
    diagonal_target = (1.0 + gamma) / 2.0 + margin
    eta_needed = max(
        required_sharpness(float(mu), diagonal_target) for mu in mu_pair
    )
    eta = eta_needed + 0.30
    pair_kernel = one_hot_population_kernel(mu_pair, eta)
    minimum_diagonal = float(np.diag(pair_kernel).min())
    alpha = 0.65

    population_fixed_error = float(
        np.max(
            np.abs(
                direct_q_population_update(
                    q_pi, reward_pair, p_pair, pair_kernel, gamma, alpha
                )
                - q_pi
            )
        )
    )
    linear = (
        np.eye(q_pi.size)
        - alpha * pair_kernel
        + alpha * gamma * pair_kernel @ p_pair
    )
    actual_operator_norm = float(np.abs(linear).sum(axis=1).max())
    diagonal_bound = float(
        1.0 - alpha * (2.0 * minimum_diagonal - (1.0 + gamma))
    )
    assert minimum_diagonal >= diagonal_target - 1e-12
    assert population_fixed_error < 1e-10
    assert actual_operator_norm <= diagonal_bound + 1e-12
    assert diagonal_bound < 1.0

    q_iterate = rng.normal(size=q_pi.size)
    for _ in range(25):
        previous_error = float(np.max(np.abs(q_iterate - q_pi)))
        q_iterate = direct_q_population_update(
            q_iterate, reward_pair, p_pair, pair_kernel, gamma, alpha
        )
        current_error = float(np.max(np.abs(q_iterate - q_pi)))
        assert current_error <= diagonal_bound * previous_error + 1e-12

    print("[A2] Direct-Q population contraction")
    print(f"  required eta:             {eta_needed:.4f}")
    print(f"  used eta:                 {eta:.4f}")
    print(f"  minimum kernel diagonal:  {minimum_diagonal:.6f}")
    print(f"  actual operator norm:     {actual_operator_norm:.6f}")
    print(f"  diagonal upper bound:     {diagonal_bound:.6f}")
    print(f"  Q^pi fixed-point error:   {population_fixed_error:.3e}")

    trajectory_rng = np.random.default_rng(7)
    start = int(trajectory_rng.choice(int(mdp["nS"]), p=exact["mu_state"]))
    states, actions, rewards = rollout(
        mdp, policy, start=start, n=2000, rng=trajectory_rng
    )
    current_pairs = states[:-1] * int(mdp["nA"]) + actions[:-1]
    next_pairs = states[1:] * int(mdp["nA"]) + actions[1:]
    q_start = rng.normal(scale=0.25, size=q_pi.size)
    write_alpha = 0.4
    exact_write, residuals, counts = exact_pair_mean_update(
        q_start,
        current_pairs,
        next_pairs,
        rewards[1:],
        gamma,
        write_alpha,
    )
    assert float(np.min(residuals)) < 0.0 < float(np.max(residuals))
    write_eta = 9.0
    soft_write = finite_pair_kernel_update(
        q_start, current_pairs, residuals, write_eta, write_alpha
    )
    residual_bound = float(np.max(np.abs(residuals)))
    observed_write_error = np.abs(soft_write - exact_write)
    leakage_bounds = np.zeros_like(observed_write_error)
    n_transitions = current_pairs.size
    for pair in np.flatnonzero(counts):
        leakage = (
            2.0
            * (n_transitions - counts[pair])
            / (counts[pair] * np.exp(write_eta) + n_transitions - counts[pair])
        )
        leakage_bounds[pair] = write_alpha * residual_bound * leakage
    assert np.all(observed_write_error <= leakage_bounds + 1e-12)

    print("[A3] finite pair-kernel write-back")
    print(f"  max observed error:       {observed_write_error.max():.3e}")
    print(f"  max leakage bound:        {leakage_bounds.max():.3e}")

    v_perturbation = rng.normal(scale=0.08, size=v_pi.size)
    v_candidate = v_pi + v_perturbation
    p = np.asarray(mdp["P"], dtype=np.float64)
    p /= p.sum(axis=2, keepdims=True)
    q_from_v = exact["reward_sa"] + gamma * np.einsum(
        "san,n->sa", p, v_candidate
    )
    recovery_error = float(np.max(np.abs(q_from_v - exact["q_pi"])))
    recovery_bound = gamma * float(np.max(np.abs(v_perturbation)))
    assert recovery_error <= recovery_bound + 1e-12

    print("[B1] population V-to-Q recovery")
    print(f"  observed Q error:         {recovery_error:.3e}")
    print(f"  gamma * V error:          {recovery_bound:.3e}")

    split_rng = np.random.default_rng(19)
    start = int(split_rng.choice(int(mdp["nS"]), p=exact["mu_state"]))
    split_length = 80000
    split_states, split_actions, split_rewards = rollout(
        mdp, policy, start=start, n=split_length, rng=split_rng
    )
    half = split_length // 2
    v_hat = tabular_v_from_context(
        split_states[: half + 1],
        split_rewards[1 : half + 1],
        gamma,
        int(mdp["nS"]),
    )
    reward_limit = float(np.max(np.abs(mdp["R"])))
    value_limit = reward_limit / (1.0 - gamma)
    v_hat = np.clip(v_hat, -value_limit, value_limit)

    recovery_states = split_states[half:-1]
    recovery_actions = split_actions[half:-1]
    recovery_next_states = split_states[half + 1 :]
    recovery_rewards = split_rewards[half + 1 :]
    recovery_pairs = (
        recovery_states * int(mdp["nA"]) + recovery_actions
    )
    targets = recovery_rewards + gamma * v_hat[recovery_next_states]
    recovery_counts = np.bincount(recovery_pairs, minlength=q_pi.size)
    assert np.all(recovery_counts > 0)

    q_hat = np.zeros_like(q_pi)
    a_empirical = np.zeros_like(q_pi)
    b_empirical = recovery_counts / recovery_pairs.size
    for pair in range(q_pi.size):
        mask = recovery_pairs == pair
        q_hat[pair] = targets[mask].mean()
        a_empirical[pair] = targets[mask].sum() / recovery_pairs.size

    q_v_hat = (
        exact["reward_sa"]
        + gamma * np.einsum("san,n->sa", p, v_hat)
    ).reshape(-1)
    eps_a = np.abs(a_empirical - mu_pair * q_v_hat)
    eps_b = np.abs(b_empirical - mu_pair)
    target_limit = reward_limit + gamma * value_limit
    assert np.all(eps_b <= mu_pair / 2.0)
    ratio_bounds = 2.0 * (eps_a + target_limit * eps_b) / mu_pair
    observed_ratio_error = np.abs(q_hat - q_v_hat)
    assert np.all(observed_ratio_error <= ratio_bounds + 1e-12)

    total_error = float(np.max(np.abs(q_hat - q_pi)))
    composed_bound = (
        gamma * float(np.max(np.abs(v_hat - v_pi)))
        + float(np.max(observed_ratio_error))
    )
    assert total_error <= composed_bound + 1e-12

    print("[B2] sample-split finite recovery")
    print(f"  V-stage sup error:        {np.max(np.abs(v_hat-v_pi)):.3e}")
    print(f"  recovery-only sup error:  {observed_ratio_error.max():.3e}")
    print(f"  ratio bound:              {ratio_bounds.max():.3e}")
    print(f"  total Q sup error:        {total_error:.3e}")
    print(f"  composed bound:           {composed_bound:.3e}")

    rare_policy = np.array(
        [[1.0 - 1e-12, 1e-12]] * int(mdp["nS"]), dtype=np.float64
    )
    rare_rng = np.random.default_rng(123)
    rare_states, rare_actions, _ = rollout(
        mdp, rare_policy, start=0, n=200, rng=rare_rng
    )
    rare_pairs = (
        rare_states[:-1] * int(mdp["nA"]) + rare_actions[:-1]
    )
    rare_counts = np.bincount(rare_pairs, minlength=q_pi.size)
    unseen_rare_actions = int(np.sum(rare_counts[1::2] == 0))
    assert unseen_rare_actions == int(mdp["nS"])

    print("[B3] action-coverage barrier")
    print(
        "  unseen rare-action pairs: "
        f"{unseen_rare_actions}/{int(mdp['nS'])}"
    )

    deterministic_p = np.zeros((2, 2, 2), dtype=np.float64)
    deterministic_p[0, 0, 0] = 1.0
    deterministic_p[0, 1, 1] = 1.0
    deterministic_p[1, 0, 1] = 1.0
    deterministic_p[1, 1, 0] = 1.0
    deterministic_r = np.zeros_like(deterministic_p)
    deterministic_r[0, 0, 0] = 0.4
    deterministic_r[0, 1, 1] = -0.2
    deterministic_r[1, 0, 1] = 0.1
    deterministic_r[1, 1, 0] = 0.7
    deterministic_mdp = {
        "nS": 2,
        "nA": 2,
        "gamma": 0.6,
        "P": deterministic_p,
        "R": deterministic_r,
        "p0": np.array([0.5, 0.5]),
    }
    deterministic_policy = np.array([[0.6, 0.4], [0.55, 0.45]])
    deterministic_exact = policy_quantities(
        deterministic_mdp, deterministic_policy
    )
    deterministic_q = deterministic_exact["q_pi"].reshape(-1)
    deterministic_bellman_error = float(
        np.max(
            np.abs(
                deterministic_q
                - (
                    deterministic_exact["reward_sa"].reshape(-1)
                    + 0.6 * deterministic_exact["p_pair"] @ deterministic_q
                )
            )
        )
    )
    assert deterministic_bellman_error < 1e-12

    tied_transition = np.repeat(
        np.array([[[0.8, 0.2]], [[0.3, 0.7]]], dtype=np.float64),
        2,
        axis=1,
    )
    tied_reward = np.repeat(
        np.array([[[0.2, -0.1]], [[-0.3, 0.4]]], dtype=np.float64),
        2,
        axis=1,
    )
    tied_mdp = {
        "nS": 2,
        "nA": 2,
        "gamma": 0.6,
        "P": tied_transition,
        "R": tied_reward,
        "p0": np.array([0.5, 0.5]),
    }
    tied_exact = policy_quantities(
        tied_mdp, np.array([[0.7, 0.3], [0.4, 0.6]])
    )
    maximum_action_gap = float(
        np.max(np.abs(tied_exact["q_pi"][:, 0] - tied_exact["q_pi"][:, 1]))
    )
    assert maximum_action_gap < 1e-12

    print("[C1] deterministic-transition fixture")
    print(f"  Q^pi Bellman error:       {deterministic_bellman_error:.3e}")
    print("[C2] zero action-gap fixture")
    print(f"  maximum action gap:       {maximum_action_gap:.3e}")
    print("PASS fixed-policy Direct-Q and V-first formula verification")


if __name__ == "__main__":
    main()
