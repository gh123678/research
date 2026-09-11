"""Reused-record oracle and learnability diagnostics for FP-KERN-002."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.stats import t as student_t

from analyze_kernel_state_generalization import analyze as predecessor_analyze
from kernel_state_generalization import EXPECTED_INPUTS, build_kernel_routes


TASK_ID = "FP-KERN-002"
TASK_VERSION = "1.0"
COMMON_EXECUTION_START = "ffdf26b029efde08ea794454a7b5da890108c355"
COMMON_INPUT_DEFAULT = Path(
    r"C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-KERN-002\input"
)
EXPECTED_CONFIG_SHA256 = (
    "78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a"
)
EXPECTED_RECORDS_SHA256 = (
    "9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1"
)
EXPECTED_MANIFEST_SHA256 = (
    "670648f7a2761d919f58b25881db45dc6d9d49d4fec25e307c6fe72bf8b31966"
)
FROZEN_PREDECESSOR_SUMMARY_FILE_SHA256 = (
    "0c5d1b039720acb3b3710846f1e490589a07f5eb7fd6f2ed859ba3520515b640"
)
EXPECTED_PREDECESSOR_CANONICAL_SUMMARY_SHA256 = (
    "e6327594ccca29e02eeb718849be09d7445a56f1ccbc2d53ee8129d79d2da9d0"
)
FAMILIES = ("current_unstructured", "hidden_cluster")
DIAGNOSTIC_ROUTES = (
    "oracle_q_nearest2",
    "oracle_generator_cluster",
    "observable_balanced_cluster",
)
PROHIBITED_OBSERVABLE_FRAGMENTS = (
    "true",
    "oracle",
    "cluster",
    "prototype",
    "realized",
    "error",
)


def strict_load(path: Path) -> Any:
    """Load strict JSON, rejecting duplicate keys and nonfinite constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError("nonfinite JSON output")
        return converted
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported JSON type: {type(value).__name__}")


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _json_ready(payload),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(payload))
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _matrix_cells(records: Sequence[dict[str, Any]]) -> Counter[tuple[Any, ...]]:
    return Counter(
        (
            record["environment_family"],
            int(record["trajectory_length"]),
            float(record["mixing"]),
            float(record["gap_bonus"]),
        )
        for record in records
    )


def validate_common_input(input_dir: Path) -> dict[str, Any]:
    """Validate the immutable common input and its complete matrix identity."""
    input_dir = input_dir.resolve()
    manifest_path = input_dir / "source_manifest.json"
    config_path = input_dir / "config.json"
    records_path = input_dir / "task_results.json"
    for path in (manifest_path, config_path, records_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    observed_hashes = {
        "source_manifest.json": sha256(manifest_path),
        "config.json": sha256(config_path),
        "task_results.json": sha256(records_path),
    }
    expected_hashes = {
        "source_manifest.json": EXPECTED_MANIFEST_SHA256,
        "config.json": EXPECTED_CONFIG_SHA256,
        "task_results.json": EXPECTED_RECORDS_SHA256,
    }
    if observed_hashes != expected_hashes:
        raise AssertionError(
            f"common-input hash mismatch: {observed_hashes!r} != {expected_hashes!r}"
        )
    manifest = strict_load(manifest_path)
    if manifest["task_id"] != TASK_ID:
        raise AssertionError("manifest task mismatch")
    if manifest["common_execution_start_commit"] != COMMON_EXECUTION_START:
        raise AssertionError("manifest common-start mismatch")
    if not manifest.get("frozen_after_creation"):
        raise AssertionError("manifest does not freeze common input")
    config = strict_load(config_path)
    records = strict_load(records_path)
    if config.get("task_id") != "FP-KERN-001" or config.get("mode") != "formal":
        raise AssertionError("predecessor config identity mismatch")
    if not isinstance(records, list) or len(records) != 480:
        raise AssertionError("common input must contain exactly 480 records")
    for index, record in enumerate(records):
        if int(record["record_index"]) != index:
            raise AssertionError("record index mismatch")
    family_counts = Counter(record["environment_family"] for record in records)
    if family_counts != Counter({"current_unstructured": 240, "hidden_cluster": 240}):
        raise AssertionError(f"family count mismatch: {family_counts!r}")
    expected_cells = Counter(
        {
            (family, length, mixing, gap): 15
            for family in FAMILIES
            for length in (256, 1024, 4096, 16384)
            for mixing in (0.08, 0.50)
            for gap in (0.0, 0.50)
        }
    )
    if _matrix_cells(records) != expected_cells:
        raise AssertionError("frozen 32-cell matrix mismatch")
    return {
        "status": "PASS",
        "input_dir": str(input_dir),
        "record_count": len(records),
        "family_counts": dict(family_counts),
        "hashes": observed_hashes,
    }


def load_common_input(
    input_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    validation = validate_common_input(input_dir)
    config = strict_load(input_dir.resolve() / "config.json")
    records = strict_load(input_dir.resolve() / "task_results.json")
    return config, records, validation


def _assert_nested_exact(observed: Any, expected: Any, path: str = "root") -> None:
    if type(observed) is not type(expected):
        raise AssertionError(
            f"type mismatch at {path}: {type(observed).__name__} != "
            f"{type(expected).__name__}"
        )
    if isinstance(expected, dict):
        if list(observed) != list(expected):
            raise AssertionError(f"dictionary-key mismatch at {path}")
        for key in expected:
            _assert_nested_exact(observed[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if len(observed) != len(expected):
            raise AssertionError(f"list-length mismatch at {path}")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            _assert_nested_exact(left, right, f"{path}[{index}]")
        return
    if observed != expected:
        raise AssertionError(f"value mismatch at {path}: {observed!r} != {expected!r}")


def reproduce_predecessor(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay every old route and the complete predecessor summary."""
    for record in records:
        rebuilt = build_kernel_routes(record["observable_inputs"])
        _assert_nested_exact(
            rebuilt,
            record["kernel_generalization"],
            f"record[{record['record_index']}].kernel_generalization",
        )
    summary, analysis = predecessor_analyze(records)
    observed_summary_hash = _payload_sha256(summary)
    if observed_summary_hash != EXPECTED_PREDECESSOR_CANONICAL_SUMMARY_SHA256:
        raise AssertionError(
            "predecessor summary mismatch: "
            f"{observed_summary_hash} != "
            f"{EXPECTED_PREDECESSOR_CANONICAL_SUMMARY_SHA256}"
        )
    if summary["classification"] != "NOT_SUPPORTED":
        raise AssertionError("predecessor classification mismatch")
    return {
        "status": "PASS",
        "records": len(records),
        "route_reconstruction_mismatches": 0,
        "frozen_summary_file_sha256": FROZEN_PREDECESSOR_SUMMARY_FILE_SHA256,
        "canonical_summary_sha256": observed_summary_hash,
        "canonicalization_note": (
            "The frozen file uses Windows CRLF text translation; the canonical "
            "logical payload uses UTF-8 with LF and no trailing newline."
        ),
        "classification": summary["classification"],
        "family_screens": summary["family_screens"],
        "analysis": analysis,
    }


def _validate_observables(
    observables: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    keys = set(observables)
    extra = sorted(keys - EXPECTED_INPUTS)
    missing = sorted(EXPECTED_INPUTS - keys)
    if extra:
        if any(
            fragment in key.lower()
            for key in extra
            for fragment in PROHIBITED_OBSERVABLE_FRAGMENTS
        ):
            raise ValueError(f"prohibited observable input: {extra}")
        raise ValueError(f"unexpected observable input: {extra}")
    if missing:
        raise ValueError(f"missing observable input: {missing}")
    signature_q = np.asarray(observables["signature_q"], dtype=np.float64)
    raw_signature_counts = np.asarray(observables["signature_counts"])
    target_sums = np.asarray(observables["target_sums"], dtype=np.float64)
    raw_target_counts = np.asarray(observables["target_counts"])
    if not np.issubdtype(raw_signature_counts.dtype, np.integer):
        raise ValueError("signature counts must have integer dtype")
    if not np.issubdtype(raw_target_counts.dtype, np.integer):
        raise ValueError("target counts must have integer dtype")
    signature_counts = raw_signature_counts.astype(np.int64, copy=False)
    target_counts = raw_target_counts.astype(np.int64, copy=False)
    if signature_q.shape != (6, 4):
        raise ValueError("observable signature shape must equal (6, 4)")
    for name, array in (
        ("signature_counts", signature_counts),
        ("target_sums", target_sums),
        ("target_counts", target_counts),
    ):
        if array.shape != signature_q.shape:
            raise ValueError(f"{name} shape mismatch")
    if np.any(signature_counts < 0) or np.any(target_counts < 0):
        raise ValueError("counts must be nonnegative")
    if not np.all(np.isfinite(signature_q)) or not np.all(np.isfinite(target_sums)):
        raise ValueError("observable estimates must be finite")
    if np.any(np.abs(signature_q[signature_counts == 0]) > 1e-15):
        raise ValueError("unvisited signature values must equal zero")
    if np.any(np.abs(target_sums[target_counts == 0]) > 1e-12):
        raise ValueError("zero-count target sums must equal zero")
    value_bound = float(observables["value_bound"])
    if not np.isfinite(value_bound) or value_bound <= 0.0:
        raise ValueError("value_bound must be finite and positive")
    return signature_q, signature_counts, target_sums, target_counts, value_bound


def leave_one_action_out_distances(
    signature_q: np.ndarray,
    signature_counts: np.ndarray,
    value_bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    distances = np.full((4, 6, 6), np.nan, dtype=np.float64)
    common_counts = np.zeros((4, 6, 6), dtype=np.int64)
    for action in range(4):
        for state in range(6):
            distances[action, state, state] = 0.0
            for other in range(state + 1, 6):
                common = (signature_counts[state] > 0) & (
                    signature_counts[other] > 0
                )
                common[action] = False
                count = int(np.sum(common))
                common_counts[action, state, other] = count
                common_counts[action, other, state] = count
                if count < 2:
                    continue
                scaled = (
                    signature_q[state, common] - signature_q[other, common]
                ) / (2.0 * value_bound)
                distance = float(np.sqrt(np.mean(np.square(scaled))))
                if np.isfinite(distance):
                    distances[action, state, other] = distance
                    distances[action, other, state] = distance
    return distances, common_counts


def balanced_partitions(n_states: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    if n_states != 6:
        raise ValueError("FP-KERN-002 requires exactly six states")
    result: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    universe = set(range(n_states))
    for peers in itertools.combinations(range(1, n_states), 2):
        left = (0, *peers)
        right = tuple(sorted(universe - set(left)))
        result.append((left, right))
    return result


def observable_balanced_partition(distances: np.ndarray) -> dict[str, Any]:
    distance_array = np.asarray(distances, dtype=np.float64)
    if distance_array.shape != (6, 6):
        raise ValueError("partition distance matrix must equal (6, 6)")
    if not np.allclose(
        distance_array,
        distance_array.T,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise ValueError("partition distances must be exactly symmetric")
    completed = distance_array.copy()
    missing = ~np.isfinite(completed)
    np.fill_diagonal(missing, False)
    completed[missing] = 1.0
    np.fill_diagonal(completed, 0.0)
    if np.any(completed < 0.0) or np.any(completed > 1.0 + 1e-12):
        raise ValueError("finite normalized distances must lie in [0, 1]")
    candidates: list[dict[str, Any]] = []
    for left, right in balanced_partitions(6):
        squared = [
            float(completed[first, second] ** 2)
            for group in (left, right)
            for first, second in itertools.combinations(group, 2)
        ]
        squared.sort()
        score = float(np.mean(np.asarray(squared, dtype=np.float64)))
        candidates.append(
            {"groups": [list(left), list(right)], "score": score}
        )
    minimum = min(item["score"] for item in candidates)
    winners = [item for item in candidates if item["score"] == minimum]
    common = {
        "missing_distance_replacement": 1.0,
        "missing_pair_count": int(np.sum(np.triu(missing, k=1))),
        "candidate_scores": candidates,
    }
    if len(winners) != 1:
        return {
            "status": "partition_tie",
            "groups": None,
            "score": minimum,
            "tie_count": len(winners),
            **common,
        }
    return {
        "status": "ok",
        "groups": winners[0]["groups"],
        "score": minimum,
        "tie_count": 1,
        **common,
    }


def _optional_matrix(values: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in values
    ]


def _empty_sources() -> list[list[list[int]]]:
    return [[[] for _ in range(4)] for _ in range(6)]


def _route_from_sources(
    target_sums: np.ndarray,
    target_counts: np.ndarray,
    sources: list[list[list[int]]],
    preset_reason: list[list[str]] | None = None,
) -> dict[str, Any]:
    estimates = np.full((6, 4), np.nan, dtype=np.float64)
    denominators = np.zeros((6, 4), dtype=np.float64)
    reasons = (
        [["target_source_unavailable" for _ in range(4)] for _ in range(6)]
        if preset_reason is None
        else [row.copy() for row in preset_reason]
    )
    for state in range(6):
        for action in range(4):
            if reasons[state][action] == "partition_tie":
                continue
            selected = sources[state][action]
            denominator = float(np.sum(target_counts[selected, action]))
            denominators[state, action] = denominator
            if denominator <= 0.0 or not np.isfinite(denominator):
                reasons[state][action] = "target_source_unavailable"
                continue
            estimate = float(np.sum(target_sums[selected, action]) / denominator)
            if not np.isfinite(estimate):
                reasons[state][action] = "estimate_nonfinite"
                continue
            estimates[state, action] = estimate
            reasons[state][action] = "ok"
    return {
        "status": "ok",
        "estimate": _optional_matrix(estimates),
        "reason": reasons,
        "source_states": sources,
        "denominator": _optional_matrix(denominators),
    }


def oracle_q_nearest2(
    true_q: np.ndarray,
    target_sums: np.ndarray,
    target_counts: np.ndarray,
) -> dict[str, Any]:
    true_q = np.asarray(true_q, dtype=np.float64)
    target_sums = np.asarray(target_sums, dtype=np.float64)
    target_counts = np.asarray(target_counts)
    if true_q.shape != (6, 4) or target_sums.shape != (6, 4):
        raise ValueError("oracle-Q arrays must equal (6, 4)")
    if target_counts.shape != (6, 4) or not np.issubdtype(
        target_counts.dtype, np.integer
    ):
        raise ValueError("oracle-Q target counts must be an integer (6, 4) matrix")
    if not np.all(np.isfinite(true_q)) or not np.all(np.isfinite(target_sums)):
        raise ValueError("oracle-Q inputs must be finite")
    if np.any(target_counts < 0):
        raise ValueError("oracle-Q target counts must be nonnegative")
    sources = _empty_sources()
    for state in range(6):
        for action in range(4):
            peers = [
                other
                for other in range(6)
                if other != state and target_counts[other, action] > 0
            ]
            peers.sort(
                key=lambda other: (
                    abs(float(true_q[state, action] - true_q[other, action])),
                    other,
                )
            )
            selected = peers[:2]
            if target_counts[state, action] > 0:
                selected = [state, *selected]
            sources[state][action] = selected
    route = _route_from_sources(target_sums, target_counts, sources)
    route["oracle_use"] = "true_q_peer_ranking_only"
    return route


def oracle_generator_cluster(
    cluster_by_state: np.ndarray,
    target_sums: np.ndarray,
    target_counts: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(cluster_by_state)
    target_sums = np.asarray(target_sums, dtype=np.float64)
    target_counts = np.asarray(target_counts)
    if labels.shape != (6,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("cluster labels must be an integer six-vector")
    unique, sizes = np.unique(labels, return_counts=True)
    if unique.size != 2 or sorted(sizes.tolist()) != [3, 3]:
        raise ValueError("generator clusters must be two balanced groups")
    if target_sums.shape != (6, 4) or target_counts.shape != (6, 4):
        raise ValueError("generator-cluster target arrays must equal (6, 4)")
    sources = _empty_sources()
    for state in range(6):
        group = np.flatnonzero(labels == labels[state]).tolist()
        for action in range(4):
            sources[state][action] = [
                other for other in group if target_counts[other, action] > 0
            ]
    route = _route_from_sources(target_sums, target_counts, sources)
    route["oracle_use"] = "generator_cluster_labels_only"
    route["groups"] = [
        np.flatnonzero(labels == label).astype(int).tolist() for label in unique
    ]
    return route


def observable_balanced_cluster(observables: Mapping[str, Any]) -> dict[str, Any]:
    (
        signature_q,
        signature_counts,
        target_sums,
        target_counts,
        value_bound,
    ) = _validate_observables(observables)
    distances, common_counts = leave_one_action_out_distances(
        signature_q, signature_counts, value_bound
    )
    partitions = [
        observable_balanced_partition(distances[action]) for action in range(4)
    ]
    sources = _empty_sources()
    reasons = [["target_source_unavailable" for _ in range(4)] for _ in range(6)]
    for action, partition in enumerate(partitions):
        if partition["status"] != "ok":
            for state in range(6):
                reasons[state][action] = "partition_tie"
            continue
        groups = partition["groups"]
        group_by_state = {
            state: group for group in groups for state in group
        }
        for state in range(6):
            sources[state][action] = [
                other
                for other in group_by_state[state]
                if target_counts[other, action] > 0
            ]
    route = _route_from_sources(target_sums, target_counts, sources, reasons)
    route["partition_by_action"] = partitions
    route["distance_by_action"] = [
        _optional_matrix(distances[action]) for action in range(4)
    ]
    route["common_action_count_by_action"] = common_counts.tolist()
    route["observable_input_keys"] = sorted(EXPECTED_INPUTS)
    return route


def _estimate_array(route: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [np.nan if value is None else float(value) for value in row]
            for row in route["estimate"]
        ],
        dtype=np.float64,
    )


def _old_estimate(record: dict[str, Any], route_name: str) -> np.ndarray:
    return _estimate_array(record["kernel_generalization"]["routes"][route_name])


def false_improvement_counts(
    true_q: np.ndarray,
    route: np.ndarray,
    baseline: np.ndarray,
) -> tuple[int, int, int]:
    if true_q.shape != route.shape or true_q.shape != baseline.shape:
        raise ValueError("false-improvement arrays must share shape")
    route_false = 0
    baseline_false = 0
    comparisons = 0
    for state in range(true_q.shape[0]):
        for left in range(true_q.shape[1]):
            for right in range(left + 1, true_q.shape[1]):
                if not (
                    np.isfinite(route[state, left])
                    and np.isfinite(route[state, right])
                    and np.isfinite(baseline[state, left])
                    and np.isfinite(baseline[state, right])
                ):
                    continue
                truth = float(true_q[state, left] - true_q[state, right])
                route_gap = float(route[state, left] - route[state, right])
                baseline_gap = float(baseline[state, left] - baseline[state, right])
                route_false += int(route_gap > 0.0 and truth <= 0.0)
                baseline_false += int(baseline_gap > 0.0 and truth <= 0.0)
                comparisons += 1
    return route_false, baseline_false, comparisons


def _mean_ci(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    mean = float(np.mean(array))
    if array.size < 2:
        return {"n": 1, "mean": mean, "lower": None, "upper": None}
    standard_error = float(np.std(array, ddof=1) / np.sqrt(array.size))
    critical = float(student_t.ppf(0.975, df=array.size - 1))
    return {
        "n": int(array.size),
        "mean": mean,
        "lower": mean - critical * standard_error,
        "upper": mean + critical * standard_error,
    }


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def route_screen(
    records: Sequence[dict[str, Any]],
    diagnostic_records: Sequence[dict[str, Any]],
    route_name: str,
) -> dict[str, Any]:
    by_index = {int(item["record_index"]): item for item in diagnostic_records}
    zero_eligible = 0
    zero_covered = 0
    zero_route: list[float] = []
    zero_baseline: list[float] = []
    zero_improvement: list[float] = []
    sparse_route: list[float] = []
    sparse_baseline: list[float] = []
    sparse_improvement: list[float] = []
    top_route: list[float] = []
    top_baseline: list[float] = []
    top_improvement: list[float] = []
    route_false = 0
    baseline_false = 0
    false_denominator = 0
    zero_empty = 0
    sparse_empty = 0
    top_empty = 0
    for record in records:
        diagnostic = by_index[int(record["record_index"])]
        route_record = diagnostic["routes"][route_name]
        if route_record.get("status") == "not_applicable_family":
            raise ValueError(f"route {route_name} is not applicable")
        estimate = _estimate_array(route_record)
        action_pool = _old_estimate(record, "action_only_pool")
        local = _old_estimate(record, "local_unpooled")
        counts = np.asarray(record["observable_inputs"]["target_counts"], dtype=np.int64)
        true_q = np.asarray(record["oracle_audit"]["true_q"], dtype=np.float64)
        eligible = np.asarray(
            record["kernel_generalization"]["routes"]
            ["leave_one_action_out_kernel"]["kernel"]["signature_eligible"],
            dtype=bool,
        )
        zero_mask = counts == 0
        zero_eligible += int(np.sum(zero_mask & eligible))
        zero_covered += int(np.sum(zero_mask & eligible & np.isfinite(estimate)))
        zero_common = zero_mask & np.isfinite(estimate) & np.isfinite(action_pool)
        if np.any(zero_common):
            route_rmse = _rmse(estimate[zero_common] - true_q[zero_common])
            baseline_rmse = _rmse(action_pool[zero_common] - true_q[zero_common])
            zero_route.append(route_rmse)
            zero_baseline.append(baseline_rmse)
            zero_improvement.append(baseline_rmse - route_rmse)
        else:
            zero_empty += 1
        sparse_common = (
            (counts >= 1)
            & (counts <= 4)
            & np.isfinite(estimate)
            & np.isfinite(local)
        )
        if np.any(sparse_common):
            route_rmse = _rmse(estimate[sparse_common] - true_q[sparse_common])
            baseline_rmse = _rmse(local[sparse_common] - true_q[sparse_common])
            sparse_route.append(route_rmse)
            sparse_baseline.append(baseline_rmse)
            sparse_improvement.append(baseline_rmse - route_rmse)
        else:
            sparse_empty += 1
        sparse_states = np.any(counts <= 4, axis=1)
        complete = (
            np.all(np.isfinite(estimate), axis=1)
            & np.all(np.isfinite(action_pool), axis=1)
            & sparse_states
        )
        if np.any(complete):
            truth_actions = np.argmax(true_q[complete], axis=1)
            route_accuracy = float(
                np.mean(np.argmax(estimate[complete], axis=1) == truth_actions)
            )
            baseline_accuracy = float(
                np.mean(np.argmax(action_pool[complete], axis=1) == truth_actions)
            )
            top_route.append(route_accuracy)
            top_baseline.append(baseline_accuracy)
            top_improvement.append(route_accuracy - baseline_accuracy)
        else:
            top_empty += 1
        observed_false, control_false, comparisons = false_improvement_counts(
            true_q, estimate, action_pool
        )
        route_false += observed_false
        baseline_false += control_false
        false_denominator += comparisons
    zero_ci = _mean_ci(zero_improvement)
    sparse_ci = _mean_ci(sparse_improvement)
    top_ci = _mean_ci(top_improvement)
    zero_relative = (
        float(np.mean(zero_improvement)) / float(np.mean(zero_baseline))
        if zero_baseline and float(np.mean(zero_baseline)) > 0.0
        else None
    )
    sparse_relative = (
        float(np.mean(sparse_improvement)) / float(np.mean(sparse_baseline))
        if sparse_baseline and float(np.mean(sparse_baseline)) > 0.0
        else None
    )
    coverage = zero_covered / zero_eligible if zero_eligible else None
    route_false_rate = route_false / false_denominator if false_denominator else None
    baseline_false_rate = (
        baseline_false / false_denominator if false_denominator else None
    )
    criteria = {
        "zero_eligible_coverage_at_least_50pct": (
            coverage is not None and coverage >= 0.50
        ),
        "zero_rmse_improves_10pct_and_ci_positive": (
            zero_relative is not None
            and zero_relative >= 0.10
            and zero_ci["lower"] is not None
            and zero_ci["lower"] > 0.0
        ),
        "sparse_rmse_improves_10pct_and_ci_positive": (
            sparse_relative is not None
            and sparse_relative >= 0.10
            and sparse_ci["lower"] is not None
            and sparse_ci["lower"] > 0.0
        ),
        "top_action_improves_5pp_and_ci_positive": (
            top_ci["mean"] is not None
            and top_ci["mean"] >= 0.05
            and top_ci["lower"] is not None
            and top_ci["lower"] > 0.0
        ),
        "false_improvement_within_1pp": (
            route_false_rate is not None
            and baseline_false_rate is not None
            and route_false_rate <= baseline_false_rate + 0.01
        ),
    }
    return {
        "route": route_name,
        "records": len(records),
        "zero_coverage": {
            "eligible": zero_eligible,
            "covered": zero_covered,
            "rate": coverage,
        },
        "zero_rmse": {
            "route_mean": float(np.mean(zero_route)) if zero_route else None,
            "baseline_mean": float(np.mean(zero_baseline)) if zero_baseline else None,
            "relative_improvement": zero_relative,
            "paired_improvement": zero_ci,
            "empty_records": zero_empty,
        },
        "sparse_rmse": {
            "route_mean": float(np.mean(sparse_route)) if sparse_route else None,
            "baseline_mean": (
                float(np.mean(sparse_baseline)) if sparse_baseline else None
            ),
            "relative_improvement": sparse_relative,
            "paired_improvement": sparse_ci,
            "empty_records": sparse_empty,
        },
        "top_action": {
            "route_mean": float(np.mean(top_route)) if top_route else None,
            "baseline_mean": float(np.mean(top_baseline)) if top_baseline else None,
            "paired_improvement": top_ci,
            "empty_records": top_empty,
        },
        "false_improvement": {
            "comparisons": false_denominator,
            "route_count": route_false,
            "baseline_count": baseline_false,
            "route_rate": route_false_rate,
            "baseline_rate": baseline_false_rate,
        },
        "criteria": criteria,
        "screen_pass": all(criteria.values()),
    }


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    truth = np.asarray(labels_true)
    predicted = np.asarray(labels_pred)
    if truth.ndim != 1 or truth.shape != predicted.shape or truth.size < 2:
        raise ValueError("ARI labels must be equal nontrivial vectors")
    truth_values = np.unique(truth)
    predicted_values = np.unique(predicted)
    contingency = np.zeros(
        (truth_values.size, predicted_values.size), dtype=np.int64
    )
    for row, truth_value in enumerate(truth_values):
        for column, predicted_value in enumerate(predicted_values):
            contingency[row, column] = int(
                np.sum((truth == truth_value) & (predicted == predicted_value))
            )

    def choose_two(values: np.ndarray) -> float:
        return float(np.sum(values * (values - 1) / 2.0))

    index = choose_two(contingency)
    row_pairs = choose_two(np.sum(contingency, axis=1))
    column_pairs = choose_two(np.sum(contingency, axis=0))
    total_pairs = truth.size * (truth.size - 1) / 2.0
    expected = row_pairs * column_pairs / total_pairs
    maximum = 0.5 * (row_pairs + column_pairs)
    denominator = maximum - expected
    if denominator == 0.0:
        return 1.0 if np.array_equal(truth[:, None] == truth, predicted[:, None] == predicted) else 0.0
    return float((index - expected) / denominator)


def _labels_from_groups(groups: Sequence[Sequence[int]]) -> np.ndarray:
    labels = np.full(6, -1, dtype=np.int64)
    for label, group in enumerate(groups):
        labels[np.asarray(group, dtype=np.int64)] = label
    if np.any(labels < 0):
        raise AssertionError("partition omits a state")
    return labels


def _benefit_recovery(
    records: Sequence[dict[str, Any]],
    diagnostic_records: Sequence[dict[str, Any]],
    bin_name: str,
) -> dict[str, Any]:
    by_index = {int(item["record_index"]): item for item in diagnostic_records}
    observable_improvements: list[float] = []
    generator_improvements: list[float] = []
    for record in records:
        diagnostic = by_index[int(record["record_index"])]
        observable = _estimate_array(
            diagnostic["routes"]["observable_balanced_cluster"]
        )
        generator = _estimate_array(
            diagnostic["routes"]["oracle_generator_cluster"]
        )
        counts = np.asarray(record["observable_inputs"]["target_counts"], dtype=np.int64)
        truth = np.asarray(record["oracle_audit"]["true_q"], dtype=np.float64)
        if bin_name == "zero":
            baseline = _old_estimate(record, "action_only_pool")
            count_mask = counts == 0
        elif bin_name == "1-4":
            baseline = _old_estimate(record, "local_unpooled")
            count_mask = (counts >= 1) & (counts <= 4)
        else:
            raise ValueError(f"unknown recovery bin: {bin_name}")
        common = (
            count_mask
            & np.isfinite(observable)
            & np.isfinite(generator)
            & np.isfinite(baseline)
        )
        if not np.any(common):
            continue
        baseline_rmse = _rmse(baseline[common] - truth[common])
        observable_rmse = _rmse(observable[common] - truth[common])
        generator_rmse = _rmse(generator[common] - truth[common])
        observable_improvements.append(baseline_rmse - observable_rmse)
        generator_improvements.append(baseline_rmse - generator_rmse)
    numerator = (
        float(np.mean(observable_improvements)) if observable_improvements else None
    )
    denominator = (
        float(np.mean(generator_improvements)) if generator_improvements else None
    )
    ratio = (
        numerator / denominator
        if numerator is not None
        and denominator is not None
        and np.isfinite(numerator)
        and np.isfinite(denominator)
        and denominator > 0.0
        else None
    )
    return {
        "bin": bin_name,
        "records": len(observable_improvements),
        "observable_mean_improvement": numerator,
        "generator_mean_improvement": denominator,
        "recovery_ratio": ratio,
    }


def cluster_diagnostics(
    records: Sequence[dict[str, Any]],
    diagnostic_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_index = {int(item["record_index"]): item for item in diagnostic_records}
    aris: list[float] = []
    tied_record_actions = 0
    peer_matches = 0
    peer_total = 0
    for record in records:
        diagnostic = by_index[int(record["record_index"])]
        labels = np.asarray(
            record["oracle_audit"]["generator"]["cluster_by_state"],
            dtype=np.int64,
        )
        partitions = diagnostic["routes"]["observable_balanced_cluster"]
        partitions = partitions["partition_by_action"]
        for partition in partitions:
            if partition["status"] != "ok":
                tied_record_actions += 1
                continue
            predicted = _labels_from_groups(partition["groups"])
            aris.append(adjusted_rand_index(labels, predicted))
            for group in partition["groups"]:
                for state in group:
                    for peer in group:
                        if peer == state:
                            continue
                        peer_total += 1
                        peer_matches += int(labels[state] == labels[peer])
    return {
        "adjusted_rand_index": _mean_ci(aris),
        "partition_tie_record_actions": tied_record_actions,
        "emitted_record_actions": len(aris),
        "same_cluster_peer_precision": {
            "matches": peer_matches,
            "assignments": peer_total,
            "rate": peer_matches / peer_total if peer_total else None,
        },
        "oracle_benefit_recovery": {
            "zero": _benefit_recovery(records, diagnostic_records, "zero"),
            "1-4": _benefit_recovery(records, diagnostic_records, "1-4"),
        },
    }


def build_diagnostic_record(record: dict[str, Any]) -> dict[str, Any]:
    observables = record["observable_inputs"]
    target_sums = np.asarray(observables["target_sums"], dtype=np.float64)
    target_counts = np.asarray(observables["target_counts"], dtype=np.int64)
    true_q = np.asarray(record["oracle_audit"]["true_q"], dtype=np.float64)
    routes: dict[str, Any] = {
        "oracle_q_nearest2": oracle_q_nearest2(
            true_q, target_sums, target_counts
        ),
        "observable_balanced_cluster": observable_balanced_cluster(observables),
    }
    if record["environment_family"] == "hidden_cluster":
        clusters = np.asarray(
            record["oracle_audit"]["generator"]["cluster_by_state"],
            dtype=np.int64,
        )
        routes["oracle_generator_cluster"] = oracle_generator_cluster(
            clusters, target_sums, target_counts
        )
    else:
        routes["oracle_generator_cluster"] = {
            "status": "not_applicable_family"
        }
    return {
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "source_task_id": "FP-KERN-001",
        "record_index": int(record["record_index"]),
        "environment_family": record["environment_family"],
        "trajectory_length": int(record["trajectory_length"]),
        "mixing": float(record["mixing"]),
        "gap_bonus": float(record["gap_bonus"]),
        "task_index": int(record["task_index"]),
        "routes": {name: routes[name] for name in DIAGNOSTIC_ROUTES},
    }


def classify_diagnostic(
    *,
    invalid: bool,
    peer_headroom: bool,
    generator_structure_useful: bool,
    observable_hidden: bool,
    observable_current: bool,
) -> str:
    if invalid:
        return "INVALID_INPUT"
    if observable_hidden and observable_current:
        return "GENERAL_PROMISING"
    if observable_hidden:
        return "STRUCTURE_CONDITIONAL_PROMISING"
    if generator_structure_useful:
        return "REPRESENTATION_GAP"
    if peer_headroom:
        return "GENERATOR_STRUCTURE_MISALIGNED"
    return "NO_BORROWING_EVIDENCE"


def analyze_diagnostics(
    records: list[dict[str, Any]],
    diagnostic_records: list[dict[str, Any]],
    baseline_reproduction: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    records_by_family = {
        family: [record for record in records if record["environment_family"] == family]
        for family in FAMILIES
    }
    diagnostic_by_family = {
        family: [
            record
            for record in diagnostic_records
            if record["environment_family"] == family
        ]
        for family in FAMILIES
    }
    screens: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        screens[family] = {
            "oracle_q_nearest2": route_screen(
                records_by_family[family],
                diagnostic_by_family[family],
                "oracle_q_nearest2",
            ),
            "observable_balanced_cluster": route_screen(
                records_by_family[family],
                diagnostic_by_family[family],
                "observable_balanced_cluster",
            ),
        }
    screens["hidden_cluster"]["oracle_generator_cluster"] = route_screen(
        records_by_family["hidden_cluster"],
        diagnostic_by_family["hidden_cluster"],
        "oracle_generator_cluster",
    )
    screens["current_unstructured"]["oracle_generator_cluster"] = {
        "status": "not_applicable_family"
    }
    peer_headroom = bool(
        screens["hidden_cluster"]["oracle_q_nearest2"]["screen_pass"]
    )
    generator_useful = bool(
        screens["hidden_cluster"]["oracle_generator_cluster"]["screen_pass"]
    )
    observable_hidden = bool(
        screens["hidden_cluster"]["observable_balanced_cluster"]["screen_pass"]
    )
    observable_current = bool(
        screens["current_unstructured"]["observable_balanced_cluster"]["screen_pass"]
    )
    classification = classify_diagnostic(
        invalid=False,
        peer_headroom=peer_headroom,
        generator_structure_useful=generator_useful,
        observable_hidden=observable_hidden,
        observable_current=observable_current,
    )
    cluster = cluster_diagnostics(
        records_by_family["hidden_cluster"],
        diagnostic_by_family["hidden_cluster"],
    )
    summary = {
        "status": "PASS",
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "records": len(records),
        "classification": classification,
        "gates": {
            "peer_headroom": peer_headroom,
            "generator_structure_useful": generator_useful,
            "observable_structure_useful": observable_hidden,
            "observable_current_family_pass": observable_current,
        },
        "family_screens": screens,
        "cluster_diagnostics": cluster,
        "predecessor_classification": baseline_reproduction["classification"],
    }
    analysis = {
        "status": "PASS",
        "task_id": TASK_ID,
        "records": len(records),
        "classification": classification,
        "invalid_input": False,
        "baseline_reproduction": {
            "status": baseline_reproduction["status"],
            "route_reconstruction_mismatches": baseline_reproduction[
                "route_reconstruction_mismatches"
            ],
            "canonical_summary_sha256": baseline_reproduction[
                "canonical_summary_sha256"
            ],
        },
        "decision_rule": summary["gates"],
    }
    return summary, analysis


def _select_smoke(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if int(record["task_index"]) == 0
        and int(record["trajectory_length"]) in (256, 1024)
    ]
    if len(selected) != 16:
        raise AssertionError(f"smoke subset must contain 16 records, got {len(selected)}")
    return selected


def build_result(
    records: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    diagnostic_records = [build_diagnostic_record(record) for record in records]
    summary, analysis = analyze_diagnostics(records, diagnostic_records, baseline)
    return diagnostic_records, summary, analysis


def _environment_payload(mode: str) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "mode": mode,
        "created_at": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "script_sha256": sha256(Path(__file__).resolve()),
    }


def _prepare_empty_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"output directory is nonempty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_result_bundle(
    output_dir: Path,
    input_dir: Path,
    mode: str,
    baseline: dict[str, Any],
    diagnostic_records: list[dict[str, Any]],
    summary: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, str]:
    output_dir = _prepare_empty_output(output_dir)
    shutil.copyfile(
        input_dir.resolve() / "source_manifest.json",
        output_dir / "source_manifest.json",
    )
    _write_json_atomic(output_dir / "baseline_reproduction.json", baseline)
    _write_json_atomic(output_dir / "diagnostic_records.json", diagnostic_records)
    _write_json_atomic(output_dir / "summary.json", summary)
    _write_json_atomic(output_dir / "analysis.json", analysis)
    _write_json_atomic(output_dir / "environment.json", _environment_payload(mode))
    (output_dir / "commands.log").write_text(
        " ".join(sys.argv) + "\n", encoding="utf-8"
    )
    (output_dir / "checks.log").write_text(
        f"PASS common input and Stage 0; records={len(diagnostic_records)}\n"
        f"PASS FP-KERN-002 {mode}; classification={summary['classification']}\n",
        encoding="utf-8",
    )
    artifact_names = (
        "source_manifest.json",
        "baseline_reproduction.json",
        "diagnostic_records.json",
        "summary.json",
        "analysis.json",
        "environment.json",
        "commands.log",
        "checks.log",
    )
    hashes = {name: sha256(output_dir / name) for name in artifact_names}
    _write_json_atomic(output_dir / "artifact_hashes.json", hashes)
    return hashes


def verify_result_bundle(output_dir: Path, input_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    baseline = strict_load(output_dir / "baseline_reproduction.json")
    observed_records = strict_load(output_dir / "diagnostic_records.json")
    observed_summary = strict_load(output_dir / "summary.json")
    observed_analysis = strict_load(output_dir / "analysis.json")
    hashes = strict_load(output_dir / "artifact_hashes.json")
    for name, expected in hashes.items():
        if sha256(output_dir / name) != expected:
            raise AssertionError(f"artifact hash mismatch: {name}")
    config, source_records, _ = load_common_input(input_dir)
    del config
    if len(observed_records) == 16:
        selected = _select_smoke(source_records)
    elif len(observed_records) == 480:
        selected = source_records
    else:
        raise AssertionError("diagnostic bundle must contain 16 or 480 records")
    rebuilt_baseline = reproduce_predecessor(source_records)
    rebuilt_records, rebuilt_summary, rebuilt_analysis = build_result(
        selected, rebuilt_baseline
    )
    _assert_nested_exact(baseline, rebuilt_baseline, "baseline_reproduction")
    _assert_nested_exact(observed_records, rebuilt_records, "diagnostic_records")
    _assert_nested_exact(observed_summary, rebuilt_summary, "summary")
    _assert_nested_exact(observed_analysis, rebuilt_analysis, "analysis")
    return {
        "status": "PASS",
        "records": len(observed_records),
        "classification": observed_summary["classification"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=COMMON_INPUT_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--mode", choices=("stage0", "smoke", "formal", "verify"), required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "verify":
        if args.output_dir is None:
            raise ValueError("verify mode requires --output-dir")
        result = verify_result_bundle(args.output_dir, args.input_dir)
        print(
            f"PASS FP-KERN-002 strict bundle verification; "
            f"records={result['records']}; classification={result['classification']}"
        )
        return
    _, all_records, validation = load_common_input(args.input_dir)
    baseline = reproduce_predecessor(all_records)
    if args.mode == "stage0":
        print(
            "PASS FP-KERN-002 Stage 0; "
            f"records={validation['record_count']}; "
            f"classification={baseline['classification']}"
        )
        return
    selected = _select_smoke(all_records) if args.mode == "smoke" else all_records
    diagnostic_records, summary, analysis = build_result(selected, baseline)
    if args.output_dir is None:
        raise ValueError(f"{args.mode} mode requires --output-dir")
    hashes = write_result_bundle(
        args.output_dir,
        args.input_dir,
        args.mode,
        baseline,
        diagnostic_records,
        summary,
        analysis,
    )
    print(
        f"PASS FP-KERN-002 {args.mode}; records={len(selected)}; "
        f"classification={summary['classification']}; artifacts={len(hashes) + 1}"
    )


if __name__ == "__main__":
    main()
