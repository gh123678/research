"""Deterministic verifier for FP-KERN-002 reused-record diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analyze_kernel_reuse_diagnostics import (
    COMMON_INPUT_DEFAULT,
    adjusted_rand_index,
    balanced_partitions,
    classify_diagnostic,
    false_improvement_counts,
    observable_balanced_cluster,
    observable_balanced_partition,
    oracle_generator_cluster,
    oracle_q_nearest2,
    validate_common_input,
)


def _observables() -> dict[str, object]:
    signature_q = np.zeros((6, 4), dtype=np.float64)
    signature_q[:3, 1:] = np.array(
        [[0.00, 0.02, -0.01], [0.01, 0.00, 0.02], [-0.02, 0.01, 0.00]]
    )
    signature_q[3:, 1:] = np.array(
        [[1.99, 2.01, 2.00], [2.02, 1.98, 2.01], [2.00, 2.02, 1.99]]
    )
    counts = np.full((6, 4), 5, dtype=np.int64)
    target_counts = np.ones((6, 4), dtype=np.int64)
    target_counts[0, 0] = 0
    target_sums = np.arange(24, dtype=np.float64).reshape(6, 4) / 10.0
    target_sums[target_counts == 0] = 0.0
    return {
        "signature_q": signature_q.tolist(),
        "signature_counts": counts.tolist(),
        "target_sums": target_sums.tolist(),
        "target_counts": target_counts.tolist(),
        "current_policy": np.full((6, 4), 0.25).tolist(),
        "pi_min": 0.05,
        "value_bound": 3.0,
    }


def _estimate(route: dict[str, object]) -> np.ndarray:
    raw = route["estimate"]
    return np.asarray(
        [[np.nan if value is None else float(value) for value in row] for row in raw],
        dtype=np.float64,
    )


def test_balanced_partitions() -> None:
    partitions = balanced_partitions(6)
    assert len(partitions) == 10
    assert len(set(partitions)) == 10
    for left, right in partitions:
        assert len(left) == len(right) == 3
        assert left[0] == 0
        assert set(left).isdisjoint(right)
        assert set(left) | set(right) == set(range(6))


def test_observable_partition_and_target_exclusion() -> None:
    observables = _observables()
    route = observable_balanced_cluster(observables)
    partition = route["partition_by_action"][0]
    assert partition["status"] == "ok"
    assert partition["groups"] == [[0, 1, 2], [3, 4, 5]]
    estimate = _estimate(route)
    expected_zero = (0.4 + 0.8) / 2.0
    assert np.isclose(estimate[0, 0], expected_zero)
    assert 0 not in route["source_states"][0][0]

    changed = _observables()
    changed_q = np.asarray(changed["signature_q"], dtype=np.float64)
    changed_q[:, 0] = np.array([3.0, -3.0, 2.5, -2.5, 2.0, -2.0])
    changed["signature_q"] = changed_q.tolist()
    repeated = observable_balanced_cluster(changed)
    assert repeated["partition_by_action"][0] == partition
    assert _estimate(repeated)[:, 0].tolist() == estimate[:, 0].tolist()
    for state in range(6):
        assert repeated["source_states"][state][0] == route["source_states"][state][0]

    prohibited = dict(observables)
    prohibited["true_q"] = np.zeros((6, 4)).tolist()
    try:
        observable_balanced_cluster(prohibited)
    except ValueError as error:
        assert "prohibited" in str(error)
    else:
        raise AssertionError("observable route accepted true_q")


def test_tie_and_missing_distance() -> None:
    distances = np.full((6, 6), np.nan, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    partition = observable_balanced_partition(distances)
    assert partition["status"] == "partition_tie"
    assert partition["missing_distance_replacement"] == 1.0

    tied = _observables()
    tied["signature_q"] = np.zeros((6, 4), dtype=np.float64).tolist()
    route = observable_balanced_cluster(tied)
    assert all(item["status"] == "partition_tie" for item in route["partition_by_action"])
    assert all(reason == "partition_tie" for row in route["reason"] for reason in row)


def test_oracle_routes() -> None:
    true_q = np.tile(np.arange(6, dtype=np.float64)[:, None], (1, 4))
    target_counts = np.ones((6, 4), dtype=np.int64)
    target_counts[0, 0] = 0
    target_sums = target_counts.astype(np.float64) * np.arange(24).reshape(6, 4)
    q_route = oracle_q_nearest2(true_q, target_sums, target_counts)
    assert q_route["source_states"][0][0] == [1, 2]
    assert np.isclose(_estimate(q_route)[0, 0], (4.0 + 8.0) / 2.0)

    clusters = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    cluster_route = oracle_generator_cluster(clusters, target_sums, target_counts)
    assert cluster_route["source_states"][0][0] == [1, 2]
    assert np.isclose(_estimate(cluster_route)[0, 0], (4.0 + 8.0) / 2.0)


def test_cluster_metrics_and_false_improvement() -> None:
    truth = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    assert adjusted_rand_index(truth, truth) == 1.0
    assert adjusted_rand_index(truth, 1 - truth) == 1.0

    true_q = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    route = np.array([[2.0, 1.0, 0.0]], dtype=np.float64)
    baseline = np.array([[0.0, 1.0, 2.0]], dtype=np.float64)
    route_false, baseline_false, comparisons = false_improvement_counts(
        true_q, route, baseline
    )
    assert (route_false, baseline_false, comparisons) == (3, 0, 3)


def test_decision_table() -> None:
    for peer in (False, True):
        for generator in (False, True):
            for observable_hidden in (False, True):
                for observable_current in (False, True):
                    result = classify_diagnostic(
                        invalid=False,
                        peer_headroom=peer,
                        generator_structure_useful=generator,
                        observable_hidden=observable_hidden,
                        observable_current=observable_current,
                    )
                    if observable_hidden and observable_current:
                        expected = "GENERAL_PROMISING"
                    elif observable_hidden:
                        expected = "STRUCTURE_CONDITIONAL_PROMISING"
                    elif generator:
                        expected = "REPRESENTATION_GAP"
                    elif peer:
                        expected = "GENERATOR_STRUCTURE_MISALIGNED"
                    else:
                        expected = "NO_BORROWING_EVIDENCE"
                    assert result == expected
    assert (
        classify_diagnostic(
            invalid=True,
            peer_headroom=True,
            generator_structure_useful=True,
            observable_hidden=True,
            observable_current=True,
        )
        == "INVALID_INPUT"
    )


def test_permutation_equivariance() -> None:
    observables = _observables()
    original = observable_balanced_cluster(observables)
    permutation = np.array([3, 4, 5, 0, 1, 2], dtype=np.int64)
    permuted: dict[str, object] = {}
    for key, value in observables.items():
        array = np.asarray(value)
        if array.shape == (6, 4):
            permuted[key] = array[permutation].tolist()
        else:
            permuted[key] = value
    observed = observable_balanced_cluster(permuted)
    inverse = np.argsort(permutation)
    assert np.allclose(
        _estimate(observed)[inverse], _estimate(original), equal_nan=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=COMMON_INPUT_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_balanced_partitions()
    test_observable_partition_and_target_exclusion()
    test_tie_and_missing_distance()
    test_oracle_routes()
    test_cluster_metrics_and_false_improvement()
    test_decision_table()
    test_permutation_equivariance()
    validated = validate_common_input(args.input_dir)
    assert validated["record_count"] == 480
    print("PASS FP-KERN-002 verifier and frozen common-input checks")


if __name__ == "__main__":
    main()
