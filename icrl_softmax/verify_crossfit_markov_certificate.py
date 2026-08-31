"""Contract verification for blocked V-first cross-fit and Markov diagnostics."""

from __future__ import annotations

import numpy as np

from crossfit_vfirst import (
    build_crossfit_targets,
    crossfit_blocks,
    crossfit_population_reference,
    crossfit_vfirst_estimate,
)
from markov_coverage_certificate import (
    build_markov_coverage_certificate,
    edge_chain_transition,
    stationary_distribution,
    stationary_time_reversal,
    suggest_crossfit_gap,
)
from mdps import rollout, sample_mdp
from verify_fixed_policy_q_routes import policy_quantities


def deterministic_fixture() -> dict[str, object]:
    transition = np.zeros((2, 2, 2), dtype=np.float64)
    transition[0, 0, 0] = 1.0
    transition[0, 1, 1] = 1.0
    transition[1, 0, 1] = 1.0
    transition[1, 1, 0] = 1.0
    reward = np.zeros_like(transition)
    reward[0, 0, 0] = 0.4
    reward[0, 1, 1] = -0.2
    reward[1, 0, 1] = 0.1
    reward[1, 1, 0] = 0.7
    return {
        "nS": 2,
        "nA": 2,
        "gamma": 0.6,
        "P": transition,
        "R": reward,
        "p0": np.array([0.5, 0.5]),
    }


def verify_blocks_and_no_leakage() -> None:
    no_gap = crossfit_blocks(20, 0)
    gap = crossfit_blocks(20, 2)
    assert no_gap["recovery_transitions_used"] == 20
    assert no_gap["discarded_gap_transitions"] == 0
    assert gap["recovery_transitions_used"] == 16
    assert gap["discarded_gap_transitions"] == 4
    assert gap["fold_a_transitions"] + gap["fold_b_transitions"] == 16

    states = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    actions = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int64)
    rewards = np.zeros_like(states, dtype=np.float64)
    value_a = np.array([10.0, 20.0])
    value_b = np.array([100.0, 200.0])
    blocks = crossfit_blocks(8, 1)
    targets = build_crossfit_targets(
        states,
        actions,
        rewards,
        value_a,
        value_b,
        n_actions=2,
        gamma=0.5,
        blocks=blocks,
    )
    expected_a = 0.5 * value_b[
        states[targets["transition_indices_a"] + 1]
    ]
    expected_b = 0.5 * value_a[
        states[targets["transition_indices_b"] + 1]
    ]
    assert np.array_equal(targets["targets_a_using_value_b"], expected_a)
    assert np.array_equal(targets["targets_b_using_value_a"], expected_b)
    assert np.all(targets["pairs_a"] % 2 == 0)
    assert np.all(targets["pairs_b"] % 2 == 1)

    print("[C1] block accounting and opposite-fold target isolation")
    print("  no-gap recovery transitions: 20/20")
    print("  g=2 recovery transitions:    16/20")


def verify_crossfit_estimator() -> None:
    rng = np.random.default_rng(20260831)
    mdp = sample_mdp(3, 2, 0.7, rng)
    policy = np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    exact = policy_quantities(mdp, policy)
    start = int(rng.choice(3, p=exact["mu_state"]))
    states, actions, rewards = rollout(
        mdp, policy, start=start, n=4000, rng=rng
    )
    transition = np.asarray(mdp["P"], dtype=np.float64)
    transition /= transition.sum(axis=2, keepdims=True)
    value_limit = float(np.max(np.abs(mdp["R"]))) / (1.0 - 0.7)
    for beta in (None, 8.0):
        result = crossfit_vfirst_estimate(
            states,
            actions,
            rewards,
            transition,
            exact["reward_sa"],
            gamma=0.7,
            alpha=0.65,
            iterations=120,
            value_limit=value_limit,
            gap_each_side=7,
            beta=beta,
            true_value=exact["v_pi"],
            true_q=exact["q_pi"],
        )
        diagnostics = result["diagnostics"]
        targets = result["target_data"]
        manual = np.zeros(6, dtype=np.float64)
        for pair in range(6):
            selected = targets["targets"][targets["pairs"] == pair]
            if selected.size:
                manual[pair] = selected.mean()
        if beta is None:
            assert np.max(
                np.abs(manual.reshape(3, 2) - result["q_estimate"])
            ) < 1e-12
        assert diagnostics["recovery_transitions_used"] == 3986
        assert diagnostics["composed_bound_slack"] >= -1e-10

    counts_a = np.array([3, 0, 4, 1, 0, 2])
    counts_b = np.array([0, 5, 2, 0, 0, 1])
    oracle_reference = crossfit_population_reference(
        exact["reward_sa"],
        transition,
        exact["v_pi"],
        exact["v_pi"],
        counts_a,
        counts_b,
        gamma=0.7,
    )
    assert np.max(np.abs(oracle_reference - exact["q_pi"])) < 1e-12

    print("[C2] exact/softmax cross-fit estimator")
    print(
        "  exact Q sup error:        "
        f"{result['diagnostics']['q_sup_error']:.3e}"
    )
    print("  oracle population error:  0.000e+00")


def verify_rare_and_deterministic_boundaries() -> None:
    mdp = deterministic_fixture()
    policy = np.array([[0.65, 0.35], [0.60, 0.40]])
    exact = policy_quantities(mdp, policy)
    rng = np.random.default_rng(44)
    start = int(rng.choice(2, p=exact["mu_state"]))
    states, actions, rewards = rollout(
        mdp, policy, start=start, n=2000, rng=rng
    )
    result = crossfit_vfirst_estimate(
        states,
        actions,
        rewards,
        np.asarray(mdp["P"]),
        exact["reward_sa"],
        gamma=0.6,
        alpha=0.65,
        iterations=120,
        value_limit=2.5,
        gap_each_side=5,
        true_value=exact["v_pi"],
        true_q=exact["q_pi"],
    )
    assert result["diagnostics"]["composed_bound_slack"] >= -1e-10

    rare_policy = np.array([[1.0 - 1e-12, 1e-12]] * 2)
    rare_exact = policy_quantities(mdp, rare_policy)
    rare_rng = np.random.default_rng(3)
    rare_states, rare_actions, rare_rewards = rollout(
        mdp, rare_policy, start=0, n=200, rng=rare_rng
    )
    rare_result = crossfit_vfirst_estimate(
        rare_states,
        rare_actions,
        rare_rewards,
        np.asarray(mdp["P"]),
        rare_exact["reward_sa"],
        gamma=0.6,
        alpha=0.65,
        iterations=80,
        value_limit=2.5,
        gap_each_side=0,
        true_value=rare_exact["v_pi"],
        true_q=rare_exact["q_pi"],
    )
    assert rare_result["diagnostics"]["missing_pairs"] >= 2

    tied_transition = np.repeat(
        np.array([[[0.8, 0.2]], [[0.3, 0.7]]]), 2, axis=1
    )
    tied_reward = np.repeat(
        np.array([[[0.2, -0.1]], [[-0.3, 0.4]]]), 2, axis=1
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
    assert np.max(
        np.abs(tied_exact["q_pi"][:, 0] - tied_exact["q_pi"][:, 1])
    ) < 1e-12

    print("[C3] deterministic, rare-action, and zero-gap boundaries")
    print(
        "  rare missing pairs:       "
        f"{rare_result['diagnostics']['missing_pairs']}"
    )


def verify_markov_certificate() -> None:
    fast = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    sticky = np.array([[0.99, 0.01], [0.01, 0.99]], dtype=np.float64)
    fast_gap = suggest_crossfit_gap(fast, fast, 512)
    sticky_gap = suggest_crossfit_gap(sticky, sticky, 512)
    assert sticky_gap["gap_used"] >= fast_gap["gap_used"]
    assert sticky_gap["dependency_tv_at_gap"] >= 0.0

    mu = stationary_distribution(sticky)
    reverse = stationary_time_reversal(sticky, mu)
    assert np.max(np.abs(reverse - sticky)) < 1e-12
    edge, edge_mu, _ = edge_chain_transition(sticky, mu)
    assert np.max(np.abs(edge_mu @ edge - edge_mu)) < 1e-12

    rng = np.random.default_rng(71)
    mdp = sample_mdp(3, 2, 0.7, rng)
    policy = np.array([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    exact = policy_quantities(mdp, policy)
    counts = np.array([160, 70, 150, 80, 170, 70])
    certificate = build_markov_coverage_certificate(
        exact["p_pi"],
        exact["p_pair"],
        counts,
        trajectory_length=int(counts.sum()),
        beta=8.0,
        gamma=0.7,
    )
    assert certificate["state"]["right_spectral_gap"] > 0.0
    assert certificate["pair"]["right_spectral_gap"] > 0.0
    assert certificate["edge"]["right_spectral_gap"] > 0.0
    assert certificate["kernel_coverage"]["kernel_certified"]
    assert "not a complete" in certificate["certificate_scope"]

    print("[C4] state/pair/edge Markov certificate")
    print(f"  fast gap used:            {fast_gap['gap_used']}")
    print(f"  sticky gap used:          {sticky_gap['gap_used']}")
    print(
        "  pair Hoeffding inflation: "
        f"{certificate['pair']['right_hoeffding_inflation']:.4f}"
    )


def main() -> None:
    verify_blocks_and_no_leakage()
    verify_crossfit_estimator()
    verify_rare_and_deterministic_boundaries()
    verify_markov_certificate()
    print("PASS cross-fit and Markov certificate verification")


if __name__ == "__main__":
    main()
