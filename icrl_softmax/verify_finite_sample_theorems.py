"""Contract verification for shared fixed-policy finite-sample certificates."""

from __future__ import annotations

import json
import math

import numpy as np

from fixed_policy_finite_sample_certificate import (
    direct_q_uniform_bound,
    observed_ghost_residuals,
    one_hot_diagonal_lower_bound,
    right_hoeffding_inflation,
    shared_event_radii,
    state_value_uniform_bound,
    strict_json_ready,
    vfirst_nosplit_bound,
)


def fast_shared_event(
    *,
    beta_irrelevant_scale: float = 1.0,
) -> dict[str, object]:
    """Return a small uniform-chain event with comfortable positive coverage."""
    return shared_event_radii(
        trajectory_length=1_000_000,
        n_states=2,
        n_pairs=4,
        delta=0.05,
        value_bound=2.5 * beta_irrelevant_scale,
        state_stationary_min=0.5,
        pair_stationary_min=0.25,
        state_right_inflation=1.0,
        pair_right_inflation=1.0,
        edge_right_inflation=1.0,
    )


def verify_shared_radius_arithmetic() -> None:
    event = shared_event_radii(
        trajectory_length=10_000,
        n_states=2,
        n_pairs=4,
        delta=0.05,
        value_bound=5.0,
        state_stationary_min=0.5,
        pair_stationary_min=0.25,
        state_right_inflation=1.0,
        pair_right_inflation=3.0,
        edge_right_inflation=2.0,
    )
    expected_m = 2 * 2 + 3 * 4
    expected_log = math.log(2.0 * expected_m / 0.05)
    assert event["M"] == expected_m
    assert math.isclose(event["L_delta"], expected_log)
    assert math.isclose(event["b_S"], math.sqrt(expected_log / 20_000.0))
    assert math.isclose(
        event["b_X"], math.sqrt(3.0 * expected_log / 20_000.0)
    )
    assert math.isclose(
        event["e_E"], 5.0 * math.sqrt(4.0 * expected_log / 10_000.0)
    )
    assert math.isclose(event["epsilon_S"], event["e_E"] / event["u_S"])
    assert math.isclose(event["epsilon_X"], event["e_E"] / event["u_X"])
    assert event["status"] == "high_probability_certified"
    assert event["failure_reasons"] == []

    assert right_hoeffding_inflation(-0.4) == 1.0
    assert math.isclose(right_hoeffding_inflation(0.5), 3.0)
    assert math.isclose(
        one_hot_diagonal_lower_bound(0.2, math.log(4.0)), 0.5
    )
    assert one_hot_diagonal_lower_bound(1.0, 1000.0) == 1.0
    assert math.isfinite(one_hot_diagonal_lower_bound(0.2, 1000.0))
    try:
        right_hoeffding_inflation(1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("lambda_right >= 1 must be rejected")

    print("[F1] shared radius arithmetic and one global delta allocation")


def residual_fixture() -> dict[str, np.ndarray | float | int]:
    current_states = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    next_states = np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.int64)
    current_pairs = np.array([0, 1, 2, 3, 0, 2, 1, 3], dtype=np.int64)
    next_pairs = np.array([1, 2, 3, 0, 1, 3, 2, 0], dtype=np.int64)
    rewards = np.array([0.2, -0.1, 0.4, 0.0, -0.3, 0.5, 0.1, 0.2])
    return {
        "current_states": current_states,
        "next_states": next_states,
        "current_pairs": current_pairs,
        "next_pairs": next_pairs,
        "rewards": rewards,
        "true_value": np.array([0.35, -0.15]),
        "true_q": np.array([0.25, -0.05, 0.4, -0.2]),
        "gamma": 0.6,
        "n_states": 2,
        "n_pairs": 4,
    }


def verify_direct_q_residual_identity() -> None:
    fixture = residual_fixture()
    observed = observed_ghost_residuals(**fixture)
    pair_means = np.asarray(observed["pair_residual_means"])
    counts = np.asarray(observed["pair_counts"])
    beta = 2.3
    sharp = math.exp(beta)
    kernel = np.empty((4, 4), dtype=np.float64)
    for query in range(4):
        weights = counts.astype(np.float64)
        weights[query] *= sharp
        kernel[query] = weights / weights.sum()

    residuals = (
        fixture["rewards"]
        + fixture["gamma"] * fixture["true_q"][fixture["next_pairs"]]
        - fixture["true_q"][fixture["current_pairs"]]
    )
    token_writes = np.empty(4, dtype=np.float64)
    for query in range(4):
        token_weights = np.where(fixture["current_pairs"] == query, sharp, 1.0)
        token_writes[query] = np.dot(token_weights, residuals) / token_weights.sum()

    assert np.max(np.abs(token_writes - kernel @ pair_means)) < 1e-12
    assert observed["missing_pairs"] == 0
    print("[F2] Direct-Q fixed-Q residual identity")


def verify_uniform_recurrences_and_clipping() -> None:
    event = fast_shared_event()
    gamma = 0.6
    alpha = 0.7
    initial_error = 2.0
    for matching, beta in (("exact", None), ("softmax", 8.0)):
        for layers in (0, 1, 5, 20):
            bound = direct_q_uniform_bound(
                event,
                matching=matching,
                gamma=gamma,
                alpha=alpha,
                iterations_used=layers,
                initial_error=initial_error,
                beta=beta,
            )
            assert bound["status"] == "high_probability_certified"
            rho = bound["rho"]
            residual = event["epsilon_X"]
            error = initial_error
            for _ in range(layers):
                error = rho * error + alpha * residual
            assert math.isclose(bound["total_bound"], error, rel_tol=1e-12)

    state_bound = state_value_uniform_bound(
        event,
        matching="exact",
        gamma=gamma,
        alpha=alpha,
        iterations_used=7,
        initial_error=1.5,
        beta=12345.0,
    )
    assert state_bound["status"] == "high_probability_certified"
    truth = np.array([-2.0, -0.2, 1.4, 2.0])
    proposed = np.array([-8.0, 0.3, 7.0, 1.5])
    clipped = np.clip(proposed, -2.5, 2.5)
    assert np.max(np.abs(clipped - truth)) <= np.max(np.abs(proposed - truth))
    assert state_bound["clipping_nonexpansive"]
    print("[F3] exact/softmax all-layer recurrences and state clipping")


def verify_nosplit_ghost_decomposition() -> None:
    fixture = residual_fixture()
    observed = observed_ghost_residuals(**fixture)
    rewards = fixture["rewards"]
    current_pairs = fixture["current_pairs"]
    next_states = fixture["next_states"]
    true_value = fixture["true_value"]
    true_q = fixture["true_q"]
    gamma = fixture["gamma"]

    # Deliberately construct V-hat from the very same rewards.
    value_estimate = true_value + np.array([rewards.mean(), -rewards.mean()])
    value_error = float(np.max(np.abs(value_estimate - true_value)))
    recovered = np.zeros(4, dtype=np.float64)
    for pair in range(4):
        selected = current_pairs == pair
        recovered[pair] = np.mean(
            rewards[selected] + gamma * value_estimate[next_states[selected]]
        )
    actual_error = float(np.max(np.abs(recovered - true_q)))
    ghost_bound = float(observed["recovery_residual_sup"])
    assert actual_error <= gamma * value_error + ghost_bound + 1e-12

    event = fast_shared_event()
    state_bound = state_value_uniform_bound(
        event,
        matching="exact",
        gamma=gamma,
        alpha=0.7,
        iterations_used=10,
        initial_error=1.0,
    )
    combined = vfirst_nosplit_bound(
        event,
        state_bound,
        recovery_matching="exact",
        gamma=gamma,
        value_bound=2.5,
    )
    assert math.isclose(
        combined["total_bound"],
        gamma * state_bound["total_bound"] + event["epsilon_X"],
    )
    print("[F4] same-sample V-first ghost-target decomposition")


def verify_softmax_recovery_leakage() -> None:
    fixture = residual_fixture()
    targets = fixture["rewards"] + fixture["gamma"] * fixture["true_value"][
        fixture["next_states"]
    ]
    groups = fixture["current_pairs"]
    beta = 1.7
    sharp = math.exp(beta)
    exact = np.array([targets[groups == pair].mean() for pair in range(4)])
    soft = np.empty(4, dtype=np.float64)
    diagonal = np.empty(4, dtype=np.float64)
    for pair in range(4):
        weights = np.where(groups == pair, sharp, 1.0)
        soft[pair] = np.dot(weights, targets) / weights.sum()
        diagonal[pair] = weights[groups == pair].sum() / weights.sum()
    value_bound = 2.5
    assert np.max(np.abs(soft - exact)) <= (
        2.0 * value_bound * np.max(1.0 - diagonal) + 1e-12
    )

    event = fast_shared_event()
    state_bound = state_value_uniform_bound(
        event,
        matching="exact",
        gamma=0.6,
        alpha=0.7,
        iterations_used=10,
        initial_error=1.0,
    )
    combined = vfirst_nosplit_bound(
        event,
        state_bound,
        recovery_matching="softmax",
        gamma=0.6,
        value_bound=value_bound,
        beta=8.0,
    )
    expected = 2.0 * value_bound * (
        1.0 - one_hot_diagonal_lower_bound(event["u_X"], 8.0)
    )
    assert math.isclose(combined["softmax_leakage"], expected)

    # Recovery needs a finite leakage bound, not Direct-Q's contraction margin.
    low_diagonal = vfirst_nosplit_bound(
        event,
        state_bound,
        recovery_matching="softmax",
        gamma=0.6,
        value_bound=value_bound,
        beta=-10.0,
    )
    assert low_diagonal["status"] == "high_probability_certified"
    assert low_diagonal["softmax_leakage"] is not None
    assert low_diagonal["recovery_kernel_diagonal_lower_bound"] < 0.8
    print("[F5] finite-softmax recovery leakage")


def verify_rejection_codes_and_beta_independence() -> None:
    uncovered = shared_event_radii(
        trajectory_length=10,
        n_states=3,
        n_pairs=6,
        delta=0.05,
        value_bound=5.0,
        state_stationary_min=0.1,
        pair_stationary_min=0.01,
        state_right_inflation=1.0,
        pair_right_inflation=1.0,
        edge_right_inflation=1.0,
    )
    assert "state_coverage_failed" in uncovered["failure_reasons"]
    assert "pair_coverage_failed" in uncovered["failure_reasons"]
    assert uncovered["epsilon_S"] is None
    assert uncovered["epsilon_X"] is None
    rejected = direct_q_uniform_bound(
        uncovered,
        matching="exact",
        gamma=0.7,
        alpha=0.6,
        iterations_used=5,
        initial_error=1.0,
    )
    assert rejected["status"] == "not_certified"
    assert rejected["failure_reasons"] == ["pair_coverage_failed"]
    assert rejected["margin"] is None
    assert rejected["total_bound"] is None
    missing_path = direct_q_uniform_bound(
        uncovered,
        matching="exact",
        gamma=0.7,
        alpha=0.6,
        iterations_used=5,
        initial_error=1.0,
        missing_pairs=1,
        evidence_level="pathwise",
        residual_bound=None,
    )
    assert missing_path["failure_reasons"] == ["pair_coverage_failed"]
    rejected_state = state_value_uniform_bound(
        uncovered,
        matching="exact",
        gamma=0.7,
        alpha=0.6,
        iterations_used=5,
        initial_error=1.0,
    )
    assert rejected_state["failure_reasons"] == ["state_coverage_failed"]

    event = fast_shared_event()
    low_margin = direct_q_uniform_bound(
        event,
        matching="softmax",
        gamma=0.9,
        alpha=0.5,
        iterations_used=5,
        initial_error=1.0,
        beta=-10.0,
    )
    assert low_margin["failure_reasons"] == [
        "pair_kernel_margin_nonpositive"
    ]
    low_state_margin = state_value_uniform_bound(
        event,
        matching="softmax",
        gamma=0.9,
        alpha=0.5,
        iterations_used=5,
        initial_error=1.0,
        beta=-10.0,
    )
    assert low_state_margin["failure_reasons"] == [
        "state_kernel_margin_nonpositive"
    ]
    diverged = direct_q_uniform_bound(
        event,
        matching="exact",
        gamma=0.7,
        alpha=0.6,
        iterations_used=5,
        initial_error=1.0,
        divergence_guard_triggered=True,
    )
    assert diverged["failure_reasons"] == ["divergence_guard_triggered"]

    spectral = shared_event_radii(
        trajectory_length=1000,
        n_states=2,
        n_pairs=2,
        delta=0.05,
        value_bound=2.5,
        state_stationary_min=0.5,
        pair_stationary_min=0.5,
        state_right_inflation=1.0,
        pair_right_inflation=math.inf,
        edge_right_inflation=1.0,
    )
    assert spectral["failure_reasons"] == ["spectral_condition_failed"]

    state_bound = state_value_uniform_bound(
        event,
        matching="exact",
        gamma=0.6,
        alpha=0.7,
        iterations_used=4,
        initial_error=1.0,
    )
    mismatch = vfirst_nosplit_bound(
        event,
        state_bound,
        recovery_matching="exact",
        gamma=0.6,
        value_bound=2.5,
        algorithm_mode="crossfit",
    )
    assert mismatch["failure_reasons"] == ["algorithm_mode_mismatch"]

    truncated = shared_event_radii(
        trajectory_length=1_000_000,
        n_states=2,
        n_pairs=2,
        delta=0.05,
        value_bound=2.5,
        state_stationary_min=0.5,
        pair_stationary_min=0.5,
        state_right_inflation=1.0,
        pair_right_inflation=1.0,
        edge_right_inflation=1.0,
        numerical_support_truncated=True,
    )
    assert truncated["failure_reasons"] == ["numerical_support_truncated"]

    exact_low = direct_q_uniform_bound(
        event,
        matching="exact",
        gamma=0.6,
        alpha=0.7,
        iterations_used=9,
        initial_error=1.0,
        beta=-1e6,
    )
    exact_high = direct_q_uniform_bound(
        event,
        matching="exact",
        gamma=0.6,
        alpha=0.7,
        iterations_used=9,
        initial_error=1.0,
        beta=1e6,
    )
    assert exact_low == exact_high
    print("[F6] support, margin, spectral, mode, and exact-beta contracts")


def verify_early_stopping_and_strict_json() -> None:
    event = fast_shared_event()
    stopped = direct_q_uniform_bound(
        event,
        matching="exact",
        gamma=0.6,
        alpha=0.5,
        iterations_used=3,
        initial_error=2.0,
    )
    nominal = direct_q_uniform_bound(
        event,
        matching="exact",
        gamma=0.6,
        alpha=0.5,
        iterations_used=100,
        initial_error=2.0,
    )
    assert stopped["iterations"] == 3
    assert stopped["optimization_term"] > nominal["optimization_term"]

    payload = strict_json_ready(
        {
            "finite": np.float64(1.25),
            "bad": np.nan,
            "nested": [np.inf, -np.inf, np.array([2.0, np.nan])],
        }
    )
    assert payload == {
        "finite": 1.25,
        "bad": None,
        "nested": [None, None, [2.0, None]],
    }
    json.dumps(payload, allow_nan=False)
    print("[F7] actual stopping depth and strict JSON serialization")


def verify_fast_mixing_complete_certificate() -> None:
    event = fast_shared_event()
    direct = direct_q_uniform_bound(
        event,
        matching="softmax",
        gamma=0.6,
        alpha=0.7,
        iterations_used=40,
        initial_error=2.5,
        beta=8.0,
    )
    state = state_value_uniform_bound(
        event,
        matching="softmax",
        gamma=0.6,
        alpha=0.7,
        iterations_used=40,
        initial_error=2.5,
        beta=8.0,
    )
    vfirst = vfirst_nosplit_bound(
        event,
        state,
        recovery_matching="softmax",
        gamma=0.6,
        value_bound=2.5,
        beta=8.0,
    )
    for certificate in (event, direct, state, vfirst):
        assert certificate["status"] == "high_probability_certified"
        assert certificate["failure_reasons"] == []
    assert 0.0 <= direct["rho"] < 1.0
    assert 0.0 <= state["rho"] < 1.0
    assert vfirst["total_bound"] >= 0.0
    pathwise = direct_q_uniform_bound(
        event,
        matching="softmax",
        gamma=0.6,
        alpha=0.7,
        iterations_used=4,
        initial_error=1.0,
        beta=8.0,
        evidence_level="pathwise",
        residual_bound=0.1,
        kernel_diagonal_lower_bound=0.99,
    )
    diagnostic = direct_q_uniform_bound(
        event,
        matching="exact",
        gamma=0.6,
        alpha=0.7,
        iterations_used=4,
        initial_error=1.0,
        evidence_level="diagnostic",
        residual_bound=0.1,
    )
    assert pathwise["status"] == "pathwise_bound_verified"
    assert diagnostic["status"] == "diagnostic_only"
    json.dumps(strict_json_ready({"event": event, "routes": [direct, state, vfirst]}), allow_nan=False)
    print("[F8] uniform fast-mixing fixture passes the complete certificate")


def main() -> None:
    verify_shared_radius_arithmetic()
    verify_direct_q_residual_identity()
    verify_uniform_recurrences_and_clipping()
    verify_nosplit_ghost_decomposition()
    verify_softmax_recovery_leakage()
    verify_rejection_codes_and_beta_independence()
    verify_early_stopping_and_strict_json()
    verify_fast_mixing_complete_certificate()
    print("PASS shared fixed-policy finite-sample theorem verification")


if __name__ == "__main__":
    main()
