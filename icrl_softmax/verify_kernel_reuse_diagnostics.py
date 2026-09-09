"""Deterministic contract checks for FP-KERN-002 (Claude independent route).

Verifier-first: this module is created before
``analyze_kernel_reuse_diagnostics.py`` and must fail with an import error
until the analyzer exists. Fixtures cover the frozen route, metric, and
decision-table contracts from ``docs/research_tasks/FP-KERN-002.md``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

from analyze_kernel_reuse_diagnostics import (
    MISSING_DISTANCE,
    adjusted_rand_index,
    balanced_partitions,
    classify,
    effective_sample_size,
    false_improvement_counts,
    generator_cluster_sources,
    leave_one_action_out_distances,
    mean_ci,
    median_positive,
    observable_balanced_cluster_route,
    oracle_q_nearest2_peers,
    pool_estimate,
    replay_predecessor_routes,
    select_partition,
    select_smoke_records,
    validate_manifest,
    validate_observables,
)

INPUT_DIR = Path(__file__).resolve().parents[2] / "input"

EXPECTED_CONFIG_HASH = (
    "78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a"
)
EXPECTED_RECORDS_HASH = (
    "9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1"
)


def fixture_observables() -> dict[str, object]:
    """Six states, four actions; states {0,1,2} and {3,4,5} are similar."""
    signature_q = np.zeros((6, 4), dtype=np.float64)
    signature_q[:3] = np.array([1.0, 0.2, -0.4, 0.7])
    signature_q[3:] = np.array([-1.0, -0.2, 0.4, -0.7])
    signature_q[1] += np.array([0.05, -0.03, 0.02, -0.01])
    signature_q[2] += np.array([-0.04, 0.01, -0.05, 0.03])
    signature_q[4] += np.array([0.02, 0.04, -0.01, -0.05])
    signature_q[5] += np.array([-0.03, 0.05, 0.01, 0.02])
    signature_counts = np.full((6, 4), 5, dtype=np.int64)
    target_counts = np.array(
        [
            [0, 3, 1, 2],
            [2, 0, 4, 1],
            [1, 2, 0, 0],
            [3, 1, 2, 0],
            [0, 0, 1, 3],
            [4, 2, 0, 1],
        ],
        dtype=np.int64,
    )
    rng_values = np.linspace(-2.0, 2.0, num=24).reshape(6, 4)
    target_sums = rng_values * target_counts
    target_sums = np.where(target_counts > 0, target_sums, 0.0)
    policy = np.full((6, 4), 0.05, dtype=np.float64)
    policy[:, 0] = 0.85
    return {
        "signature_q": signature_q,
        "signature_counts": signature_counts,
        "target_sums": target_sums,
        "target_counts": target_counts,
        "current_policy": policy,
        "pi_min": 0.05,
        "value_bound": 10.0,
    }


def verify_manifest_validation() -> None:
    manifest = {
        "schema_version": 1,
        "task_id": "FP-KERN-002",
        "source_task_id": "FP-KERN-001",
        "source_verified_commit": "403884ae6bde46c7c3578ae01d77422ed03faf05",
        "source_corrected_formal_seal": "1001d23273bdf29b92d9b84a3f3956e83819da4a",
        "files": {
            "config.json": {"size_bytes": 1, "sha256": EXPECTED_CONFIG_HASH},
            "task_results.json": {"size_bytes": 2, "sha256": EXPECTED_RECORDS_HASH},
        },
        "expected_records": 480,
        "expected_family_counts": {"current_unstructured": 240, "hidden_cluster": 240},
        "expected_matrix": {
            "tasks_per_cell": 15,
            "trajectory_lengths": [256, 1024, 4096, 16384],
            "mixing": [0.08, 0.5],
            "gap_bonuses": [0.0, 0.5],
            "seed": 20260909,
        },
        "frozen_after_creation": True,
    }
    validate_manifest(copy.deepcopy(manifest))
    tampered = copy.deepcopy(manifest)
    tampered["files"]["config.json"]["sha256"] = "0" * 64
    try:
        validate_manifest(tampered)
    except ValueError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("tampered manifest hash was accepted")
    tampered = copy.deepcopy(manifest)
    tampered["expected_records"] = 479
    try:
        validate_manifest(tampered)
    except ValueError as error:
        assert "480" in str(error)
    else:
        raise AssertionError("tampered record count was accepted")
    tampered = copy.deepcopy(manifest)
    tampered["frozen_after_creation"] = False
    try:
        validate_manifest(tampered)
    except ValueError:
        pass
    else:
        raise AssertionError("unfrozen manifest was accepted")


def verify_frozen_input_identity() -> None:
    import hashlib
    import json

    config_path = INPUT_DIR / "config.json"
    records_path = INPUT_DIR / "task_results.json"
    manifest_path = INPUT_DIR / "source_manifest.json"
    for path in (config_path, records_path, manifest_path):
        if not path.is_file():
            raise AssertionError(f"missing frozen input file: {path}")
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (
            ("config.json", config_path),
            ("task_results.json", records_path),
        )
    }
    assert hashes["config.json"] == EXPECTED_CONFIG_HASH
    assert hashes["task_results.json"] == EXPECTED_RECORDS_HASH
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    assert manifest["files"]["config.json"]["sha256"] == hashes["config.json"]
    assert manifest["files"]["task_results.json"]["sha256"] == hashes["task_results.json"]

    records = json.loads(records_path.read_text(encoding="utf-8"))
    assert isinstance(records, list) and len(records) == 480
    families = {"current_unstructured": 0, "hidden_cluster": 0}
    cells: set[tuple[str, int, float, float, int]] = set()
    for index, record in enumerate(records):
        assert int(record["record_index"]) == index
        assert record["task_id"] == "FP-KERN-001"
        family = str(record["environment_family"])
        families[family] += 1
        cells.add(
            (
                family,
                int(record["trajectory_length"]),
                float(record["mixing"]),
                float(record["gap_bonus"]),
                int(record["task_index"]),
            )
        )
        assert int(record["seed_components"][0]) == 20260909
    assert families == {"current_unstructured": 240, "hidden_cluster": 240}
    assert len(cells) == 480


def verify_observable_boundary_rejection() -> None:
    observable = fixture_observables()
    for forbidden_key in (
        "true_q",
        "oracle_audit",
        "cluster_by_state",
        "realized_error",
        "exact_return",
    ):
        bad = copy.deepcopy(observable)
        bad[forbidden_key] = np.zeros((6, 4))
        try:
            observable_balanced_cluster_route(bad)
        except ValueError as error:
            assert "prohibited" in str(error) or "unexpected" in str(error)
        else:
            raise AssertionError(f"oracle input {forbidden_key} was accepted")
    missing = copy.deepcopy(observable)
    del missing["target_sums"]
    try:
        observable_balanced_cluster_route(missing)
    except ValueError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("incomplete observable input was accepted")


def verify_target_action_exclusion() -> None:
    observable = fixture_observables()
    first = observable_balanced_cluster_route(observable)
    for action in range(4):
        modified = copy.deepcopy(observable)
        modified_q = np.asarray(modified["signature_q"], dtype=np.float64).copy()
        modified_q[:, action] += np.array([513.0, -257.0, 129.0, -65.0, 33.0, -17.0])
        modified["signature_q"] = modified_q
        second = observable_balanced_cluster_route(modified)
        before = first["partition_by_action"][action]
        after = second["partition_by_action"][action]
        assert before == after, f"target action {action} leaked into its partition"


def verify_oracle_q_nearest2_order() -> None:
    true_q = np.zeros((6, 4), dtype=np.float64)
    true_q[:, 0] = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    target_counts = np.zeros((6, 4), dtype=np.int64)
    target_counts[:, 0] = np.array([1, 1, 1, 0, 1, 1])
    peers = oracle_q_nearest2_peers(true_q, target_counts, 0, 0)
    assert peers == [1, 2]
    # State 3 has zero target count; |3 - q| ties at distance 1 for states
    # 2 and 4, and the frozen order breaks the tie by state index.
    peers = oracle_q_nearest2_peers(true_q, target_counts, 3, 0)
    assert peers == [2, 4]
    # Zero-count peers are never selected, and at most two peers are taken.
    target_counts[5, 0] = 0
    peers = oracle_q_nearest2_peers(true_q, target_counts, 1, 0)
    assert peers == [0, 2]
    assert len(peers) <= 2


def verify_ten_balanced_partitions() -> None:
    partitions = balanced_partitions(6)
    assert len(partitions) == 10
    seen = set()
    for first, second in partitions:
        assert len(first) == 3 and len(second) == 3
        assert 0 in first
        assert sorted(first + second) == [0, 1, 2, 3, 4, 5]
        canonical = tuple(sorted(first))
        assert canonical not in seen
        seen.add(canonical)


def verify_missing_distance_replacement() -> None:
    signature_q = np.zeros((6, 4), dtype=np.float64)
    signature_q[:3] = np.array([1.0, 0.2, -0.4, 0.7])
    signature_q[3:] = np.array([-1.0, -0.2, 0.4, -0.7])
    signature_counts = np.zeros((6, 4), dtype=np.int64)
    signature_counts[:, 0] = 5
    signature_counts[0, 1] = 5
    signature_counts[1, 2] = 5
    distances, common_counts, _ = leave_one_action_out_distances(
        signature_q, signature_counts, 10.0
    )
    # Target action 3: every state pair has fewer than two common positive
    # non-target signature actions, so all off-diagonal distances are NaN.
    off_diagonal = [
        distances[3, left, right]
        for left in range(6)
        for right in range(6)
        if left != right
    ]
    assert all(not np.isfinite(value) for value in off_diagonal)
    assert int(common_counts[3, 0, 1]) == 1
    selection = select_partition(np.asarray(distances[3]))
    assert selection["status"] == "partition_tie"
    # With every pair replaced by exactly MISSING_DISTANCE, all ten scores
    # equal MISSING_DISTANCE ** 2, which forces the frozen tie abstention.
    assert MISSING_DISTANCE == 1.0
    for score in selection["scores"]:
        assert score == MISSING_DISTANCE**2


def verify_sorted_scoring_and_unique_minimum() -> None:
    observable = fixture_observables()
    (
        signature_q,
        signature_counts,
        _,
        _,
        _,
        _,
        value_bound,
    ) = validate_observables(observable)
    distances, _, _ = leave_one_action_out_distances(
        signature_q, signature_counts, value_bound
    )
    selection = select_partition(np.asarray(distances[0]))
    assert selection["status"] == "ok"
    first, second = selection["groups"]
    assert 0 in first
    assert sorted(first) == [0, 1, 2]
    assert sorted(second) == [3, 4, 5]
    # The score is the arithmetic mean of the six sorted squared
    # within-group distances after missing-distance replacement.
    replaced = np.asarray(distances[0], dtype=np.float64).copy()
    replaced[~np.isfinite(replaced)] = MISSING_DISTANCE
    within = [
        replaced[left, right] ** 2
        for group in (first, second)
        for left, right in (
            (group[0], group[1]),
            (group[0], group[2]),
            (group[1], group[2]),
        )
    ]
    expected = float(np.mean(sorted(within)))
    assert selection["score"] == expected
    # An exact tie across minima abstains instead of choosing by state index.
    tie = np.zeros((6, 6), dtype=np.float64)
    selection = select_partition(tie)
    assert selection["status"] == "partition_tie"
    assert selection["groups"] is None


def verify_state_permutation_equivariance() -> None:
    observable = fixture_observables()
    first = observable_balanced_cluster_route(observable)
    permutation = np.array([3, 0, 5, 1, 4, 2], dtype=np.int64)
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
    second = observable_balanced_cluster_route(permuted)
    for action in range(4):
        before = first["partition_by_action"][action]
        after = second["partition_by_action"][action]
        assert before["status"] == after["status"]
        if before["status"] != "ok":
            continue
        moved_groups = [
            tuple(sorted(int(permutation[state]) for state in group))
            for group in after["groups"]
        ]
        original_groups = [tuple(sorted(group)) for group in before["groups"]]
        assert sorted(moved_groups) == sorted(original_groups)
    original_estimates = np.array(
        [[v if v is not None else np.nan for v in row] for row in first["estimate"]]
    )
    moved_estimates = np.array(
        [[v if v is not None else np.nan for v in row] for row in second["estimate"]]
    )[inverse]
    assert np.allclose(original_estimates, moved_estimates, equal_nan=True)


def verify_group_pooling_and_abstention() -> None:
    target_sums = np.array([[2.0, 0.0], [4.0, 3.0], [6.0, 0.0]], dtype=np.float64)
    target_counts = np.array([[1, 0], [2, 1], [3, 0]], dtype=np.int64)
    estimate, reason, denominator = pool_estimate(
        target_sums, target_counts, 0, [0, 1, 2]
    )
    # Positive-count targets include themselves: (2+4+6)/(1+2+3) = 2.
    assert reason == "ok" and estimate == 2.0 and denominator == 6.0
    estimate, reason, denominator = pool_estimate(
        target_sums, target_counts, 0, [1, 2]
    )
    # Peers only: (4+6)/(2+3) for action 0.
    assert reason == "ok" and estimate == 2.0 and denominator == 5.0
    estimate, reason, denominator = pool_estimate(
        target_sums, target_counts, 1, [0, 2]
    )
    assert reason == "target_source_unavailable" and estimate is None
    assert denominator == 0.0
    bad_sums = target_sums.copy()
    bad_sums[1, 0] = np.inf
    estimate, reason, _ = pool_estimate(bad_sums, target_counts, 0, [0, 1, 2])
    assert estimate is None and reason != "ok"


def verify_generator_cluster_sources() -> None:
    labels = [1, 1, 0, 0, 1, 0]
    sources = generator_cluster_sources(labels)
    assert sources[0] == [0, 1, 4]
    assert sources[2] == [2, 3, 5]
    try:
        generator_cluster_sources([0, 0, 0, 1, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("unbalanced generator labels were accepted")


def verify_ari_and_peer_precision() -> None:
    assert adjusted_rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    value = adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1])
    assert np.isclose(value, -0.5)
    assert adjusted_rand_index([0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]) == 1.0

    from analyze_kernel_reuse_diagnostics import peer_precision_counts

    groups = ((0, 1, 2), (3, 4, 5))
    labels = [0, 0, 1, 1, 1, 0]
    matches, total = peer_precision_counts(groups, labels)
    assert total == 12
    # Matches: state0->1, state1->0, state3->4, state4->3; all others differ.
    assert matches == 4
    matches, total = peer_precision_counts(groups, [0, 0, 0, 1, 1, 1])
    assert (matches, total) == (12, 12)


def verify_metric_primitives() -> None:
    assert median_positive([0.0, 3.0, 1.0, 2.0]) == 2.0
    assert median_positive([1.0, 3.0]) == 2.0
    assert median_positive([0.0, -1.0]) is None
    observed = effective_sample_size(
        np.array([1.0, 0.5]), np.array([2, 4], dtype=np.int64)
    )
    assert np.isclose(observed, 16.0 / 3.0)
    summary = mean_ci([1.0])
    assert summary["n"] == 1 and summary["lower"] is None
    summary = mean_ci([])
    assert summary["n"] == 0 and summary["mean"] is None

    true_q = np.array([[2.0, 0.0]], dtype=np.float64)
    reversed_estimate = np.array([[0.0, 3.0]], dtype=np.float64)
    primary_false, _, comparisons = false_improvement_counts(
        true_q, reversed_estimate, reversed_estimate
    )
    assert (primary_false, comparisons) == (0, 1)
    true_q = np.array([[0.0, 2.0]], dtype=np.float64)
    positive_estimate = np.array([[3.0, 0.0]], dtype=np.float64)
    primary_false, _, comparisons = false_improvement_counts(
        true_q, positive_estimate, positive_estimate
    )
    assert (primary_false, comparisons) == (1, 1)


def verify_decision_table() -> None:
    base = {
        "peer_headroom": False,
        "generator_structure_useful": False,
        "observable_structure_useful": False,
    }
    assert classify(False, base, False) == "INVALID_INPUT"
    gates = {**base, "observable_structure_useful": True}
    assert classify(True, gates, True) == "GENERAL_PROMISING"
    assert classify(True, gates, False) == "STRUCTURE_CONDITIONAL_PROMISING"
    gates = {**base, "generator_structure_useful": True}
    assert classify(True, gates, False) == "REPRESENTATION_GAP"
    assert classify(True, gates, True) == "REPRESENTATION_GAP"
    gates = {**base, "peer_headroom": True}
    assert classify(True, gates, False) == "GENERATOR_STRUCTURE_MISALIGNED"
    assert classify(True, base, False) == "NO_BORROWING_EVIDENCE"
    # The observable-route pass takes precedence over every oracle gate.
    gates = {
        "peer_headroom": True,
        "generator_structure_useful": True,
        "observable_structure_useful": True,
    }
    assert classify(True, gates, False) == "STRUCTURE_CONDITIONAL_PROMISING"


def verify_predecessor_replay_schema() -> None:
    observable = fixture_observables()
    replay = replay_predecessor_routes(observable)
    assert replay["route_order"] == [
        "local_unpooled",
        "action_only_pool",
        "same_action_anchor_kernel",
        "leave_one_action_out_kernel",
    ]
    assert replay["shape"] == {"n_states": 6, "n_actions": 4}
    kernel = replay["routes"]["leave_one_action_out_kernel"]["kernel"]
    for key in (
        "bandwidth_by_action",
        "distance_by_action",
        "weight_by_target",
        "denominator",
        "signature_eligible",
        "common_action_count_by_action",
    ):
        assert key in kernel
    local = replay["routes"]["local_unpooled"]["estimate"]
    # Zero-count pairs abstain locally; positive pairs equal the pair mean.
    assert local[0][0] is None
    expected_local = float(np.linspace(-2.0, 2.0, num=24).reshape(6, 4)[0, 1])
    assert np.isclose(float(local[0][1]), expected_local)


def verify_smoke_subset() -> None:
    import json

    records = json.loads((INPUT_DIR / "task_results.json").read_text(encoding="utf-8"))
    smoke = select_smoke_records(records)
    assert len(smoke) == 16
    seen = {
        (
            record["environment_family"],
            record["trajectory_length"],
            record["mixing"],
            record["gap_bonus"],
        )
        for record in smoke
    }
    assert len(seen) == 16
    assert all(record["task_index"] == 0 for record in smoke)
    assert {record["trajectory_length"] for record in smoke} == {256, 1024}


def main() -> None:
    checks = [
        verify_manifest_validation,
        verify_observable_boundary_rejection,
        verify_target_action_exclusion,
        verify_oracle_q_nearest2_order,
        verify_ten_balanced_partitions,
        verify_missing_distance_replacement,
        verify_sorted_scoring_and_unique_minimum,
        verify_state_permutation_equivariance,
        verify_group_pooling_and_abstention,
        verify_generator_cluster_sources,
        verify_ari_and_peer_precision,
        verify_metric_primitives,
        verify_decision_table,
        verify_predecessor_replay_schema,
    ]
    only_fixtures = "--fixtures-only" in sys.argv
    for check in checks:
        check()
    if not only_fixtures:
        verify_frozen_input_identity()
        verify_smoke_subset()
    print(f"kernel reuse diagnostic checks passed ({len(checks)} fixtures)")


if __name__ == "__main__":
    main()
