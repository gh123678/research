"""Deterministic contract checks for FP-KERN-001."""

# FP-KERN-001 canonical route code: Codex corrected route.
#
# Added 2026-09-11, outside the sealed route, when the verified FP-KERN-001 and
# FP-KERN-002 tasks were merged into main under explicit user approval.
#
# This file is the Codex route implementation, sealed at
#   640a3f8  original formal seal
#   5af3dc6  corrective implementation and smoke seal
#   1001d23  corrected formal seal (authoritative 480-record corpus)
# Claude's independent route implementation is preserved byte-for-byte at
#   docs/research_branches/FP-KERN-001/claude/route/
# See that directory's README.md: both routes used these identical top-level
# module names, so only one implementation can occupy this path in main.
# The statement above the imports is provenance only; it is not part of the
# sealed program and changes no formula, matrix, seed, threshold, or decision
# rule.

from __future__ import annotations

import copy

import numpy as np

from analyze_kernel_state_generalization import (
    _false_improvement_counts,
    _record_action_correlations,
)
from kernel_generalization_mdps import make_hidden_cluster_mdp
from kernel_state_generalization import (
    build_kernel_routes,
    effective_sample_size,
    median_positive,
)


def fixture_observables() -> dict[str, object]:
    q_signature = np.array(
        [
            [9.0, 0.00, 0.00, 0.00],
            [-9.0, 0.05, 0.05, 0.05],
            [3.0, 1.00, 1.00, 1.00],
            [2.0, 2.00, 2.00, 2.00],
        ],
        dtype=np.float64,
    )
    signature_counts = np.ones((4, 4), dtype=np.int64)
    target_counts = np.array(
        [
            [0, 2, 2, 2],
            [4, 0, 0, 0],
            [3, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    target_sums = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [8.0, 0.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    policy = np.full((4, 4), 0.05, dtype=np.float64)
    policy[:, 0] = 0.85
    return {
        "signature_q": q_signature,
        "signature_counts": signature_counts,
        "target_sums": target_sums,
        "target_counts": target_counts,
        "current_policy": policy,
        "pi_min": 0.05,
        "value_bound": 10.0,
    }


def estimate_matrix(result: dict[str, object], route: str) -> np.ndarray:
    raw = result["routes"][route]["estimate"]  # type: ignore[index]
    return np.array(
        [[np.nan if value is None else float(value) for value in row] for row in raw],
        dtype=np.float64,
    )


def verify_median_and_ess() -> None:
    assert median_positive([0.0, 3.0, 1.0, 2.0]) == 2.0
    assert median_positive([1.0, 3.0]) == 2.0
    assert median_positive([0.0, -1.0]) is None
    observed = effective_sample_size(
        np.array([1.0, 0.5]), np.array([2, 4], dtype=np.int64)
    )
    expected = (2.0 + 2.0) ** 2 / (2.0 + 1.0)
    assert np.isclose(observed, expected)


def verify_target_action_exclusion_and_zero_recovery() -> None:
    observable = fixture_observables()
    first = build_kernel_routes(observable)
    modified = copy.deepcopy(observable)
    modified_q = np.asarray(modified["signature_q"], dtype=np.float64).copy()
    modified_q[:, 0] += np.array([1000.0, -300.0, 777.0, -999.0])
    modified["signature_q"] = modified_q
    second = build_kernel_routes(modified)

    first_meta = first["routes"]["leave_one_action_out_kernel"]["kernel"]  # type: ignore[index]
    second_meta = second["routes"]["leave_one_action_out_kernel"]["kernel"]  # type: ignore[index]
    assert first_meta["distance_by_action"][0] == second_meta["distance_by_action"][0]
    assert first_meta["bandwidth_by_action"][0] == second_meta["bandwidth_by_action"][0]
    assert np.isfinite(estimate_matrix(first, "leave_one_action_out_kernel")[0, 0])
    assert estimate_matrix(first, "local_unpooled")[0, 0] != estimate_matrix(
        first, "local_unpooled"
    )[0, 0]
    same_reason = first["routes"]["same_action_anchor_kernel"]["reason"][0][0]  # type: ignore[index]
    assert same_reason == "target_source_unavailable"
    assert first_meta["signature_eligible"][0][0] is True

    weights = first_meta["weight_by_target"][0][0]
    assert np.isclose(float(weights[0]), 1.0)
    assert float(weights[1]) > float(weights[2]) > float(weights[3])
    assert not any(
        forbidden in str(first).lower()
        for forbidden in ("true_q", "oracle", "cluster_by_state", "occupancy")
    )


def verify_self_only_reduces_to_local() -> None:
    observable = fixture_observables()
    target_counts = np.zeros((4, 4), dtype=np.int64)
    target_sums = np.zeros((4, 4), dtype=np.float64)
    target_counts[0, 1] = 3
    target_sums[0, 1] = 6.0
    observable["target_counts"] = target_counts
    observable["target_sums"] = target_sums
    result = build_kernel_routes(observable)
    local = estimate_matrix(result, "local_unpooled")[0, 1]
    primary = estimate_matrix(result, "leave_one_action_out_kernel")[0, 1]
    assert np.isclose(local, 2.0)
    assert np.isclose(primary, local)
    ess = result["routes"]["leave_one_action_out_kernel"]["effective_sample_size"][0][1]  # type: ignore[index]
    assert np.isclose(float(ess), 3.0)


def verify_sparse_positive_self_weight_without_signature_support() -> None:
    observable = fixture_observables()
    signature_counts = np.asarray(
        observable["signature_counts"], dtype=np.int64
    ).copy()
    signature_q = np.asarray(observable["signature_q"], dtype=np.float64).copy()
    signature_counts[0, 1:] = np.array([1, 0, 0])
    signature_q[0, 2:] = 0.0
    target_counts = np.zeros((4, 4), dtype=np.int64)
    target_sums = np.zeros((4, 4), dtype=np.float64)
    target_counts[0, 0] = 3
    target_sums[0, 0] = 6.0
    observable["signature_counts"] = signature_counts
    observable["signature_q"] = signature_q
    observable["target_counts"] = target_counts
    observable["target_sums"] = target_sums

    result = build_kernel_routes(observable)
    local = estimate_matrix(result, "local_unpooled")[0, 0]
    primary = estimate_matrix(result, "leave_one_action_out_kernel")[0, 0]
    weights = result["routes"]["leave_one_action_out_kernel"]["kernel"][  # type: ignore[index]
        "weight_by_target"
    ][0][0]
    assert np.isclose(local, 2.0)
    assert np.isclose(primary, local)
    assert np.isclose(float(weights[0]), 1.0)
    assert np.allclose(np.asarray(weights[1:], dtype=np.float64), 0.0)


def verify_one_sided_false_improvement() -> None:
    true_q = np.array([[2.0, 0.0]], dtype=np.float64)
    reversed_estimate = np.array([[0.0, 3.0]], dtype=np.float64)
    primary_false, baseline_false, comparisons = _false_improvement_counts(
        true_q, reversed_estimate, reversed_estimate
    )
    assert comparisons == 1
    assert primary_false == 0
    assert baseline_false == 0

    true_q = np.array([[0.0, 2.0]], dtype=np.float64)
    positive_estimate = np.array([[3.0, 0.0]], dtype=np.float64)
    primary_false, baseline_false, comparisons = _false_improvement_counts(
        true_q, positive_estimate, positive_estimate
    )
    assert comparisons == 1
    assert primary_false == 1
    assert baseline_false == 1


def verify_record_action_spearman_granularity() -> None:
    true_q = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [3.0, 3.0],
        ],
        dtype=np.float64,
    )
    distances = np.full((2, 3, 3), np.nan, dtype=np.float64)
    distances[0, 0, 1] = distances[0, 1, 0] = 1.0
    distances[0, 0, 2] = distances[0, 2, 0] = 3.0
    distances[0, 1, 2] = distances[0, 2, 1] = 2.0
    distances[1, 0, 1] = distances[1, 1, 0] = 3.0
    distances[1, 0, 2] = distances[1, 2, 0] = 1.0
    distances[1, 1, 2] = distances[1, 2, 1] = 2.0
    correlations = _record_action_correlations(distances.tolist(), true_q)
    assert len(correlations) == 2
    assert np.isclose(correlations[0], 1.0)
    assert np.isclose(correlations[1], -1.0)


def verify_state_permutation_equivariance() -> None:
    observable = fixture_observables()
    result = build_kernel_routes(observable)
    permutation = np.array([2, 0, 3, 1], dtype=np.int64)
    inverse = np.argsort(permutation)
    permuted = copy.deepcopy(observable)
    for key in (
        "signature_q",
        "signature_counts",
        "target_sums",
        "target_counts",
        "current_policy",
    ):
        permuted[key] = np.asarray(observable[key])[permutation]
    permuted_result = build_kernel_routes(permuted)
    for route in (
        "local_unpooled",
        "action_only_pool",
        "same_action_anchor_kernel",
        "leave_one_action_out_kernel",
    ):
        original = estimate_matrix(result, route)
        restored = estimate_matrix(permuted_result, route)[inverse]
        assert np.allclose(original, restored, equal_nan=True)


def verify_validation_and_oracle_rejection() -> None:
    observable = fixture_observables()
    bad = copy.deepcopy(observable)
    bad["target_counts"] = np.asarray(bad["target_counts"], dtype=np.float64)
    try:
        build_kernel_routes(bad)
    except ValueError as error:
        assert "integer" in str(error)
    else:
        raise AssertionError("floating counts were accepted")

    bad = copy.deepcopy(observable)
    bad["true_q"] = np.zeros((4, 4))
    try:
        build_kernel_routes(bad)
    except ValueError as error:
        assert "unexpected" in str(error) or "prohibited" in str(error)
    else:
        raise AssertionError("oracle input was accepted")

    bad = copy.deepcopy(observable)
    sums = np.asarray(bad["target_sums"], dtype=np.float64).copy()
    sums[0, 0] = np.nan
    bad["target_sums"] = sums
    try:
        build_kernel_routes(bad)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite sums were accepted")

    bad = copy.deepcopy(observable)
    policy = np.asarray(bad["current_policy"], dtype=np.float64).copy()
    policy[0, 0] -= 0.1
    bad["current_policy"] = policy
    try:
        build_kernel_routes(bad)
    except ValueError as error:
        assert "sum" in str(error) or "pi_min" in str(error)
    else:
        raise AssertionError("invalid policy was accepted")

    bad = copy.deepcopy(observable)
    counts = np.asarray(bad["target_counts"], dtype=np.int64).copy()
    counts[0, 0] = -1
    bad["target_counts"] = counts
    try:
        build_kernel_routes(bad)
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("negative count was accepted")


def verify_degenerate_bandwidth_abstains() -> None:
    observable = fixture_observables()
    observable["signature_q"] = np.zeros((4, 4), dtype=np.float64)
    result = build_kernel_routes(observable)
    primary = result["routes"]["leave_one_action_out_kernel"]  # type: ignore[index]
    assert primary["kernel"]["bandwidth_by_action"] == [None, None, None, None]
    assert primary["reason"][0][0] == "bandwidth_unavailable"


def verify_hidden_cluster_generator() -> None:
    kwargs = {
        "n_states": 6,
        "n_actions": 4,
        "gamma": 0.7,
        "mixing": 0.08,
        "gap_bonus": 0.5,
    }
    first, first_audit = make_hidden_cluster_mdp(
        **kwargs, rng=np.random.default_rng(123)
    )
    second, second_audit = make_hidden_cluster_mdp(
        **kwargs, rng=np.random.default_rng(123)
    )
    for key in ("P", "R", "p0"):
        assert np.array_equal(np.asarray(first[key]), np.asarray(second[key]))
    assert first_audit == second_audit
    transition = np.asarray(first["P"], dtype=np.float64)
    reward = np.asarray(first["R"], dtype=np.float64)
    assert np.allclose(transition.sum(axis=2), 1.0)
    assert np.all(transition >= 0.0)
    assert np.max(np.abs(reward)) <= 1.5 + 1e-12
    labels = np.asarray(first_audit["cluster_by_state"], dtype=np.int64)
    assert sorted(np.bincount(labels).tolist()) == [3, 3]
    assert first_audit["prototype_weight"] == 0.9
    assert first_audit["independent_weight"] == 0.1
    try:
        make_hidden_cluster_mdp(
            5, 4, 0.7, 0.08, 0.5, np.random.default_rng(1)
        )
    except ValueError as error:
        assert "freezes" in str(error)
    else:
        raise AssertionError("invalid hidden-cluster dimensions were accepted")


def main() -> None:
    verify_median_and_ess()
    verify_target_action_exclusion_and_zero_recovery()
    verify_self_only_reduces_to_local()
    verify_sparse_positive_self_weight_without_signature_support()
    verify_one_sided_false_improvement()
    verify_record_action_spearman_granularity()
    verify_state_permutation_equivariance()
    verify_validation_and_oracle_rejection()
    verify_degenerate_bandwidth_abstains()
    verify_hidden_cluster_generator()
    print("kernel state-generalization checks passed")


if __name__ == "__main__":
    main()
