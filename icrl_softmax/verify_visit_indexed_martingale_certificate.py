"""Contract checks for the visit-indexed martingale certificate."""

from __future__ import annotations

import json
import math

import numpy as np

from visit_indexed_martingale_certificate import (
    STATUS_NOT_CERTIFIED,
    STATUS_SELECTIVE_HIGH_PROBABILITY,
    build_visit_indexed_certificate,
    empirical_one_hot_diagonals,
    simultaneous_hoeffding_radius,
    strict_json_ready,
)


def expect_value_error(function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def fixture(
    *,
    state_counts: list[int] | None = None,
    pair_counts: list[int] | None = None,
    beta: float = 8.0,
    fixed_context: bool = True,
    synchronous_update: bool = True,
    algorithm_mode: str = "fixed_policy_synchronous",
) -> dict:
    states = [4, 4] if state_counts is None else state_counts
    pairs = [2, 2, 2, 2] if pair_counts is None else pair_counts
    return build_visit_indexed_certificate(
        state_counts=states,
        pair_counts=pairs,
        trajectory_length=sum(states),
        reward_bound=1.5,
        gamma=0.7,
        alpha=0.65,
        beta=beta,
        delta=0.05,
        direct_exact_iterations=7,
        direct_softmax_iterations=6,
        state_exact_iterations=5,
        state_softmax_iterations=4,
        direct_exact_diverged=False,
        direct_softmax_diverged=False,
        fixed_context=fixed_context,
        synchronous_update=synchronous_update,
        algorithm_mode=algorithm_mode,
    )


def verify_radius_and_risk_accounting() -> None:
    radius = simultaneous_hoeffding_radius(
        4,
        reward_bound=1.0,
        gamma=0.5,
        n_groups=5,
        horizon=10,
        delta=0.1,
    )
    expected = 2.0 * math.sqrt(2.0 * math.log(2.0 * 5.0 * 10.0 / 0.1) / 4.0)
    assert math.isclose(radius, expected, rel_tol=1e-15, abs_tol=1e-15)
    r1 = simultaneous_hoeffding_radius(
        1,
        reward_bound=1.0,
        gamma=0.5,
        n_groups=5,
        horizon=10,
        delta=0.1,
    )
    r9 = simultaneous_hoeffding_radius(
        9,
        reward_bound=1.0,
        gamma=0.5,
        n_groups=5,
        horizon=10,
        delta=0.1,
    )
    assert r1 > radius > r9
    assert math.isclose(r1 / r9, 3.0, rel_tol=1e-15, abs_tol=1e-15)

    certificate = fixture()
    event = certificate["event"]
    assert event["n_groups"] == 2 + 2 * 4
    risk = event["risk_allocation"]
    assert math.isclose(
        risk["one_sided_failure_probability"],
        0.05 / (2.0 * event["n_groups"] * 8.0),
    )
    assert math.isclose(
        risk["two_sided_group_count_failure_probability"],
        0.05 / (event["n_groups"] * 8.0),
    )
    assert math.isclose(risk["total_failure_probability"], 0.05)
    assert event["conditional_interval_width"] == 2.0 * event["value_bound"]


def verify_count_selection_and_support() -> None:
    certificate = fixture(state_counts=[3, 5], pair_counts=[1, 2, 2, 3])
    event = certificate["event"]
    assert event["state_bellman"]["counts"] == [3, 5]
    assert event["pair_bellman"]["counts"] == [1, 2, 2, 3]
    assert event["recovery"]["counts"] == [1, 2, 2, 3]
    selected = event["pair_bellman"]["radius_by_group"][3]
    direct = simultaneous_hoeffding_radius(
        3,
        reward_bound=1.5,
        gamma=0.7,
        n_groups=10,
        horizon=8,
        delta=0.05,
    )
    assert selected == direct
    assert event["pair_bellman"]["max_radius"] == event["pair_bellman"][
        "radius_by_group"
    ][0]

    missing = fixture(state_counts=[2, 6], pair_counts=[0, 2, 3, 3])
    assert missing["event"]["pair_bellman"]["radius_by_group"][0] is None
    assert missing["routes"]["direct_exact"]["status"] == STATUS_NOT_CERTIFIED
    assert "pair_support_missing" in missing["routes"]["direct_exact"][
        "failure_reasons"
    ]
    assert missing["routes"]["vfirst_nosplit_exact"]["status"] == (
        STATUS_NOT_CERTIFIED
    )


def verify_empirical_diagonals_and_route_composition() -> None:
    diagonals = empirical_one_hot_diagonals([2, 2, 2, 2], horizon=8, beta=8.0)
    sharp = math.exp(8.0)
    expected_diagonal = 2.0 * sharp / (2.0 * sharp + 6.0)
    assert all(
        math.isclose(value, expected_diagonal, rel_tol=1e-15, abs_tol=1e-15)
        for value in diagonals
    )

    certificate = fixture()
    bound = certificate["certificate_inputs"]["declared"]["value_bound"]
    assert bound == 1.5 / (1.0 - 0.7)
    direct = certificate["routes"]["direct_exact"]
    radius = certificate["event"]["pair_bellman"]["max_radius"]
    rho = 1.0 - 0.65 * (1.0 - 0.7)
    expected = rho**7 * bound + (1.0 - rho**7) * radius / (1.0 - 0.7)
    assert math.isclose(direct["total_bound"], expected, rel_tol=1e-14)
    assert direct["initial_error"] == bound
    assert direct["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY

    soft = certificate["routes"]["direct_softmax"]
    assert soft["kernel_diagonal_lower_bound"] == min(diagonals)
    assert soft["margin"] > 0.0
    assert soft["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY

    state = certificate["state_value"]["exact"]
    vfirst = certificate["routes"]["vfirst_nosplit_exact"]
    recovery = certificate["event"]["recovery"]["max_radius"]
    assert math.isclose(
        vfirst["total_bound"], 0.7 * state["total_bound"] + recovery
    )
    soft_state = certificate["state_value"]["softmax"]
    soft_vfirst = certificate["routes"]["vfirst_nosplit_softmax"]
    pair_diagonal = soft_vfirst["recovery_kernel_diagonal_lower_bound"]
    leakage = 2.0 * bound * (1.0 - pair_diagonal)
    assert math.isclose(soft_vfirst["softmax_leakage"], leakage)
    assert math.isclose(
        soft_vfirst["total_bound"],
        0.7 * soft_state["total_bound"] + recovery + leakage,
    )

    lower_beta = fixture(beta=2.0)
    assert lower_beta["routes"]["direct_exact"]["total_bound"] == direct[
        "total_bound"
    ]
    assert lower_beta["routes"]["vfirst_nosplit_exact"]["total_bound"] == (
        vfirst["total_bound"]
    )


def verify_rejections_and_ordering() -> None:
    bad_margin = fixture(beta=0.0)
    assert bad_margin["routes"]["direct_softmax"]["status"] == (
        STATUS_NOT_CERTIFIED
    )
    assert bad_margin["routes"]["direct_softmax"]["failure_reasons"] == [
        "pair_kernel_margin_nonpositive"
    ]
    assert bad_margin["state_value"]["softmax"]["failure_reasons"] == [
        "state_kernel_margin_nonpositive"
    ]
    assert bad_margin["routes"]["vfirst_nosplit_softmax"]["failure_reasons"] == [
        "state_kernel_margin_nonpositive"
    ]

    multi = fixture(beta=0.0, fixed_context=False)
    assert multi["routes"]["direct_softmax"]["failure_reasons"] == [
        "algorithm_mode_mismatch",
        "pair_kernel_margin_nonpositive",
    ]
    assert multi["routes"]["direct_softmax"]["status"] == STATUS_NOT_CERTIFIED
    mode = fixture(algorithm_mode="changing_policy")
    assert mode["routes"]["direct_exact"]["failure_reasons"] == [
        "algorithm_mode_mismatch"
    ]

    expect_value_error(fixture, pair_counts=[2, 3, 3])
    expect_value_error(fixture, pair_counts=[1, 1, 3, 3])
    expect_value_error(fixture, fixed_context="yes")

    common = dict(
        count=1,
        reward_bound=1.0,
        gamma=0.5,
        n_groups=5,
        horizon=10,
    )
    expect_value_error(simultaneous_hoeffding_radius, delta=0.0, **common)
    expect_value_error(simultaneous_hoeffding_radius, delta=1.0, **common)
    expect_value_error(simultaneous_hoeffding_radius, delta=math.nan, **common)
    expect_value_error(
        build_visit_indexed_certificate,
        state_counts=[5, 5],
        pair_counts=[2, 2, 3, 3],
        trajectory_length=10,
        reward_bound=math.nan,
        gamma=0.7,
        alpha=0.65,
        beta=8.0,
        delta=0.05,
        direct_exact_iterations=1,
        direct_softmax_iterations=1,
        state_exact_iterations=1,
        state_softmax_iterations=1,
    )


def verify_strict_json_and_oracle_separation() -> None:
    converted = strict_json_ready(
        {"array": np.asarray([1.0, np.nan]), "infinite": math.inf}
    )
    assert converted == {"array": [1.0, None], "infinite": None}
    json.dumps(converted, allow_nan=False)

    certificate = fixture()
    json.dumps(strict_json_ready(certificate), allow_nan=False)
    banned = {
        "occupancy",
        "transition_matrix",
        "spectral_gap",
        "q_pi",
        "v_pi",
        "true_residual",
        "true_initial_error",
        "oracle_audit",
    }

    def keys(value) -> set[str]:
        if isinstance(value, dict):
            result = {str(key).lower() for key in value}
            for child in value.values():
                result.update(keys(child))
            return result
        if isinstance(value, list):
            result: set[str] = set()
            for child in value:
                result.update(keys(child))
            return result
        return set()

    all_keys = keys(certificate)
    assert not (banned & all_keys)
    assert certificate["optional_variance_adaptive"]["status"] == (
        "not_implemented_no_observable_variance_proxy"
    )
    assert certificate["optional_variance_adaptive"]["selected"] is False
    assert certificate["event"]["delta"] == 0.05


def verify_wrong_filtration_counterexample() -> None:
    # One state, two equiprobable actions, rewards +1 and -1, gamma=0.
    # E[D^V | S] = 0, but E[D^V | S,A] equals the sampled reward.
    action_rewards = np.asarray([1.0, -1.0])
    policy = np.asarray([0.5, 0.5])
    assert float(policy @ action_rewards) == 0.0
    assert action_rewards[0] != 0.0 and action_rewards[1] != 0.0


def verify_unadjusted_posthoc_min_counterexample() -> None:
    # Two valid delta-level certificates may fail on disjoint events. Selecting
    # their minimum after seeing both fails on the union, whose mass is 2 delta.
    delta = 0.05
    failure_probabilities = np.asarray([delta, delta, 1.0 - 2.0 * delta])
    assert math.isclose(float(failure_probabilities.sum()), 1.0)
    selected_min_failure = float(failure_probabilities[:2].sum())
    assert selected_min_failure > delta
    assert math.isclose(selected_min_failure, 2.0 * delta)


def main() -> None:
    verify_radius_and_risk_accounting()
    verify_count_selection_and_support()
    verify_empirical_diagonals_and_route_composition()
    verify_rejections_and_ordering()
    verify_strict_json_and_oracle_separation()
    verify_wrong_filtration_counterexample()
    verify_unadjusted_posthoc_min_counterexample()
    print("PASS visit-indexed martingale certificate contracts")


if __name__ == "__main__":
    main()
