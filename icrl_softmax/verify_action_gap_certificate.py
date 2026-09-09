"""Deterministic contract checks for the FP-ADV-001 pure certificate."""

from __future__ import annotations

import inspect
import itertools
import json
import math

import numpy as np

from action_gap_certificate import (
    FAILURE_REASONS,
    build_action_gap_certificate,
    canonical_reasons,
    empirical_successor_row,
    exact_local_uncertainty,
    global_uncertainty,
    softmax_effective_successor_row,
    softmax_local_uncertainty,
    strict_json_ready,
    total_variation,
)


def _expect_value_error(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def _route_bound(total: float | None, emitted: bool = True) -> dict[str, object]:
    return {
        "finite_bound_emitted": bool(emitted and total is not None),
        "total_bound": total,
        "failure_reasons": [] if emitted else ["pair_support_missing"],
    }


def _base_inputs() -> dict[str, object]:
    estimates = np.asarray([[2.0, 1.0, 0.0]], dtype=np.float64)
    return {
        "policy": [[0.5, 0.4, 0.1]],
        "pair_counts": [5, 5, 5],
        "pair_successor_counts": [[5], [5], [5]],
        "q_estimates": {
            "vfirst_exact": estimates,
            "vfirst_softmax": estimates,
            "direct_exact": estimates,
            "direct_softmax": estimates,
        },
        "state_value_bounds": {
            "exact": _route_bound(0.05),
            "softmax": _route_bound(0.05),
        },
        "global_q_bounds": {
            "vfirst_exact": _route_bound(0.2),
            "vfirst_softmax": _route_bound(0.2),
            "direct_exact": _route_bound(0.2),
            "direct_softmax": _route_bound(0.2),
        },
        "recovery_radii": [0.05, 0.05, 0.05],
        "trajectory_length": 15,
        "reward_bound": 1.0,
        "gamma": 0.7,
        "beta": 8.0,
        "pi_min": 0.1,
        "transfer_fraction": 0.5,
        "algorithm_mode": "fixed_policy_synchronous",
        "divergence_guards": {"direct_exact": False, "direct_softmax": False},
    }


def test_total_variation_and_exact_formula() -> None:
    same = [0.25, 0.75]
    disjoint_a = [1.0, 0.0]
    disjoint_b = [0.0, 1.0]
    assert total_variation(same, same) == 0.0
    assert total_variation(disjoint_a, disjoint_b) == 1.0
    assert empirical_successor_row([2, 6], 8) == [0.25, 0.75]
    common = exact_local_uncertainty(0.2, 0.3, 0.7, 0.4, same, same)
    extreme = exact_local_uncertainty(
        0.2, 0.3, 0.7, 0.4, disjoint_a, disjoint_b
    )
    assert math.isclose(common, 0.5)
    assert math.isclose(extreme, 0.5 + 2.0 * 0.7 * 0.4)


def test_span_tv_inequality() -> None:
    p = np.asarray([0.5, 0.3, 0.2])
    q = np.asarray([0.1, 0.2, 0.7])
    e = np.asarray([-0.4, 0.2, 0.3])
    lhs = abs(float((p - q) @ e))
    tv_span = total_variation(p, q) * float(np.max(e) - np.min(e))
    sup = 2.0 * total_variation(p, q) * float(np.max(np.abs(e)))
    assert lhs <= tv_span + 1e-15
    assert tv_span <= sup + 1e-15


def test_exhaustive_finite_distribution_span_bound() -> None:
    """Exercise the TV/span inequality over a small integer distribution grid."""
    for denominator in range(1, 5):
        compositions = [
            counts
            for counts in itertools.product(range(denominator + 1), repeat=3)
            if sum(counts) == denominator
        ]
        errors = tuple(itertools.product((-1.0, 0.0, 1.0), repeat=3))
        for left, right in itertools.product(compositions, repeat=2):
            p = empirical_successor_row(left, denominator)
            q = empirical_successor_row(right, denominator)
            tv = total_variation(p, q)
            for error in errors:
                span = max(error) - min(error)
                assert abs(float(np.dot(np.asarray(p) - np.asarray(q), error))) <= (
                    tv * span + 1e-15
                )


def test_softmax_effective_row_matches_dense_weights() -> None:
    group_successors = [2, 1]
    total_successors = [4, 6]
    beta = 2.0
    result = softmax_effective_successor_row(
        group_successors,
        total_successors,
        group_count=3,
        trajectory_length=10,
        beta=beta,
    )
    sharp = math.exp(beta)
    expected_counts = np.asarray(group_successors) * sharp + (
        np.asarray(total_successors) - np.asarray(group_successors)
    )
    expected = expected_counts / float(np.sum(expected_counts))
    expected_kappa = 3.0 * sharp / (3.0 * sharp + 7.0)
    assert math.isclose(float(result["kappa"]), expected_kappa, rel_tol=1e-15)
    assert np.allclose(result["row"], expected, rtol=1e-15, atol=1e-15)

    all_group = softmax_effective_successor_row(
        [4, 6], [4, 6], group_count=10, trajectory_length=10, beta=1000.0
    )
    assert all_group["kappa"] == 1.0
    assert np.allclose(all_group["row"], [0.4, 0.6])


def test_softmax_contamination_and_global_formula() -> None:
    uncertainty = softmax_local_uncertainty(
        radius_a=0.2,
        radius_b=0.4,
        kappa_a=0.9,
        kappa_b=0.8,
        reward_bound=1.0,
        gamma=0.7,
        value_bound=0.3,
        row_a=[1.0, 0.0],
        row_b=[0.0, 1.0],
    )
    b = 1.0 / (1.0 - 0.7)
    expected = (
        0.9 * 0.2
        + 2.0 * b * 0.1
        + 0.8 * 0.4
        + 2.0 * b * 0.2
        + 2.0 * 0.7 * 0.3
    )
    assert math.isclose(uncertainty, expected, rel_tol=1e-15)
    assert global_uncertainty(0.25) == 0.5


def test_positive_update_floor_tie_break_and_theta() -> None:
    certificate = build_action_gap_certificate(**_base_inputs())
    for route in certificate["routes"].values():
        assert route["receiver_by_state"] == [0]
        assert route["status"] == "safe_update_emitted"
        assert np.allclose(route["policy_plus"], [[0.65, 0.25, 0.1]])
        assert route["eligible_donor_count"] == 1
        assert route["total_transferred_mass"] == 0.15000000000000002
        assert route["bellman_lcb_by_state"][0] > 0.0

    tied = _base_inputs()
    tied_estimate = np.asarray([[2.0, 2.0, 0.0]])
    tied["q_estimates"] = {
        key: tied_estimate for key in (
            "vfirst_exact",
            "vfirst_softmax",
            "direct_exact",
            "direct_softmax",
        )
    }
    tied_certificate = build_action_gap_certificate(**tied)
    assert all(
        route["receiver_by_state"] == [0]
        for route in tied_certificate["routes"].values()
    )
    _expect_value_error(
        build_action_gap_certificate, **{**_base_inputs(), "transfer_fraction": 0.4}
    )


def test_multiple_donor_transfer_is_exact_and_conservative() -> None:
    inputs = _base_inputs()
    inputs["policy"] = [[0.5, 0.3, 0.2]]
    certificate = build_action_gap_certificate(**inputs)
    for route in certificate["routes"].values():
        assert route["eligible_donor_count"] == 2
        assert np.allclose(route["policy_plus"], [[0.65, 0.2, 0.15]])
        assert math.isclose(route["total_transferred_mass"], 0.15)
        assert math.isclose(sum(route["policy_plus"][0]), 1.0)


def test_partial_support_is_local_and_unvisited_receiver_abstains() -> None:
    values = np.asarray([[2.0, 0.0], [1.0, 0.0]])
    inputs = _base_inputs()
    inputs.update(
        {
            "policy": [[0.7, 0.3], [0.7, 0.3]],
            "pair_counts": [5, 5, 5, 0],
            "pair_successor_counts": [[5, 0], [5, 0], [0, 5], [0, 0]],
            "q_estimates": {key: values for key in inputs["q_estimates"]},
            "recovery_radii": [0.05, 0.05, 0.05, None],
            "trajectory_length": 15,
            "global_q_bounds": {
                key: _route_bound(None, emitted=False)
                for key in inputs["global_q_bounds"]
            },
        }
    )
    certificate = build_action_gap_certificate(**inputs)
    local = certificate["routes"]["vfirst_local_exact"]
    assert local["state_results"][0]["eligible_donor_count"] == 1
    assert local["state_results"][1]["eligible_donor_count"] == 0
    assert local["status"] == "safe_update_emitted"
    assert certificate["routes"]["vfirst_global_exact"]["status"] == "abstained"

    unvisited_receiver = dict(inputs)
    unvisited_values = np.asarray([[3.0, 0.0], [0.0, 4.0]])
    unvisited_receiver["q_estimates"] = {
        key: unvisited_values for key in inputs["q_estimates"]
    }
    certificate = build_action_gap_certificate(**unvisited_receiver)
    second = certificate["routes"]["vfirst_local_exact"]["state_results"][1]
    assert second["receiver"] == 1
    assert "candidate_pair_unvisited" in second["failure_reasons"]


def test_no_donor_returns_original_policy_exactly() -> None:
    inputs = _base_inputs()
    estimates = np.asarray([[0.02, 0.01, 0.0]], dtype=np.float64)
    inputs["q_estimates"] = {key: estimates for key in inputs["q_estimates"]}
    certificate = build_action_gap_certificate(**inputs)
    for route in certificate["routes"].values():
        assert route["eligible_donor_count"] == 0
        assert route["policy_plus"] == inputs["policy"]


def test_ordered_reasons_divergence_and_mode() -> None:
    scrambled = [
        "gap_lcb_nonpositive",
        "numerical_nonfinite",
        "algorithm_mode_mismatch",
        "candidate_pair_unvisited",
        "algorithm_mode_mismatch",
    ]
    assert canonical_reasons(scrambled) == [
        "algorithm_mode_mismatch",
        "candidate_pair_unvisited",
        "numerical_nonfinite",
        "gap_lcb_nonpositive",
    ]
    assert tuple(FAILURE_REASONS) == (
        "algorithm_mode_mismatch",
        "divergence_guard_triggered",
        "state_certificate_not_emitted",
        "candidate_pair_unvisited",
        "recovery_radius_unavailable",
        "attention_mass_invalid",
        "effective_transition_row_invalid",
        "numerical_nonfinite",
        "gap_lcb_nonpositive",
        "no_transferable_mass",
    )

    mismatched = _base_inputs()
    mismatched["algorithm_mode"] = "asynchronous"
    certificate = build_action_gap_certificate(**mismatched)
    assert all(
        route["failure_reasons"][0] == "algorithm_mode_mismatch"
        for route in certificate["routes"].values()
    )

    diverged = _base_inputs()
    diverged["divergence_guards"] = {
        "direct_exact": True,
        "direct_softmax": False,
    }
    certificate = build_action_gap_certificate(**diverged)
    assert certificate["routes"]["direct_global_exact"]["failure_reasons"][0] == (
        "divergence_guard_triggered"
    )


def test_local_global_dominance_and_bellman_fixture() -> None:
    certificate = build_action_gap_certificate(**_base_inputs())
    dominance = certificate["local_global_dominance"]
    assert dominance["exact"]["checked_comparisons"] > 0
    assert dominance["exact"]["all_local_penalties_nonincreasing"] is True
    assert dominance["softmax"]["all_local_penalties_nonincreasing"] is True

    route = certificate["routes"]["vfirst_local_exact"]
    old_policy = np.asarray(_base_inputs()["policy"])
    new_policy = np.asarray(route["policy_plus"])
    true_q = np.asarray([[2.2, 1.0, 0.0]])
    v_pi = np.sum(old_policy * true_q, axis=1)
    bellman_change = np.sum((new_policy - old_policy) * true_q, axis=1)
    assert np.all(bellman_change >= np.asarray(route["bellman_lcb_by_state"]) - 1e-12)
    assert np.all(bellman_change >= 0.0)
    assert np.allclose(v_pi + bellman_change, np.sum(new_policy * true_q, axis=1))


def test_strict_json_and_no_oracle_interface() -> None:
    signature = inspect.signature(build_action_gap_certificate)
    forbidden = ("true", "oracle", "occupancy", "kernel", "return")
    assert not any(
        token in name.lower() for name in signature.parameters for token in forbidden
    )
    payload = strict_json_ready(
        {"array": np.asarray([1.0, np.nan]), "positive": math.inf, "ok": 2}
    )
    assert payload == {"array": [1.0, None], "positive": None, "ok": 2}
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)
    assert "NaN" not in encoded and "Infinity" not in encoded

    _expect_value_error(
        build_action_gap_certificate,
        **{**_base_inputs(), "q_estimates": {"oracle_q": [[1.0, 0.0, 0.0]]}},
    )


def test_invalid_inputs() -> None:
    _expect_value_error(total_variation, [0.5, 0.4], [0.5, 0.5])
    _expect_value_error(empirical_successor_row, [0, 0], 0)
    _expect_value_error(
        softmax_effective_successor_row,
        [1, 0],
        [1, 1],
        group_count=1,
        trajectory_length=1,
        beta=8.0,
    )
    _expect_value_error(
        build_action_gap_certificate, **{**_base_inputs(), "pi_min": 0.34}
    )
    _expect_value_error(
        build_action_gap_certificate,
        **{**_base_inputs(), "pair_counts": [5, 5, 4]},
    )
    _expect_value_error(
        build_action_gap_certificate,
        **{**_base_inputs(), "pair_counts": [5.0, 5.0, 5.0]},
    )
    _expect_value_error(
        build_action_gap_certificate,
        **{
            **_base_inputs(),
            "pair_successor_counts": [[5.0], [5.0], [5.0]],
        },
    )
    _expect_value_error(
        build_action_gap_certificate,
        **{**_base_inputs(), "recovery_radii": [math.nan, 0.05, 0.05]},
    )
    bad_q = np.asarray([[math.nan, 1.0, 0.0]])
    _expect_value_error(
        build_action_gap_certificate,
        **{
            **_base_inputs(),
            "q_estimates": {key: bad_q for key in _base_inputs()["q_estimates"]},
        },
    )


def main() -> None:
    test_total_variation_and_exact_formula()
    test_span_tv_inequality()
    test_exhaustive_finite_distribution_span_bound()
    test_softmax_effective_row_matches_dense_weights()
    test_softmax_contamination_and_global_formula()
    test_positive_update_floor_tie_break_and_theta()
    test_multiple_donor_transfer_is_exact_and_conservative()
    test_partial_support_is_local_and_unvisited_receiver_abstains()
    test_no_donor_returns_original_policy_exactly()
    test_ordered_reasons_divergence_and_mode()
    test_local_global_dominance_and_bellman_fixture()
    test_strict_json_and_no_oracle_interface()
    test_invalid_inputs()
    print("action-gap certificate checks passed")


if __name__ == "__main__":
    main()
