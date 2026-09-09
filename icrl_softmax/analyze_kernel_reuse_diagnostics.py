"""FP-KERN-002 reused-record oracle and learnability diagnostic.

Claude independent implementation. Pure layer first (no file I/O), then
strict adapters. Consumes only the frozen common input copy of the sealed
FP-KERN-001 corpus; generates no trajectories and modifies no old artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from scipy.stats import t as student_t


TASK_ID = "FP-KERN-002"
SOURCE_TASK_ID = "FP-KERN-001"
EXPECTED_CONFIG_SHA256 = (
    "78aa1bcb5bd2529ab7346412a818ec95e2058deb07dff6436777424e074fb33a"
)
EXPECTED_RECORDS_SHA256 = (
    "9f3e777e819fb64625bc2c119bdc5ad462a62277b2ff337b04d2f360253c5da1"
)
EXPECTED_RECORDS = 480
EXPECTED_FAMILY_COUNTS = {"current_unstructured": 240, "hidden_cluster": 240}
FAMILIES = ("current_unstructured", "hidden_cluster")
TRAJECTORY_LENGTHS = (256, 1024, 4096, 16384)
MIXING_VALUES = (0.08, 0.5)
GAP_BONUSES = (0.0, 0.5)
TASKS_PER_CELL = 15
FROZEN_SEED = 20260909
N_STATES = 6
N_ACTIONS = 4

MISSING_DISTANCE = 1.0
PEER_COUNT = 2
PREDECESSOR_ROUTE_ORDER = [
    "local_unpooled",
    "action_only_pool",
    "same_action_anchor_kernel",
    "leave_one_action_out_kernel",
]
DIAGNOSTIC_ROUTE_ORDER = [
    "oracle_q_nearest2",
    "oracle_generator_cluster",
    "observable_balanced_cluster",
]

OBSERVABLE_KEYS = frozenset(
    {
        "signature_q",
        "signature_counts",
        "target_sums",
        "target_counts",
        "current_policy",
        "pi_min",
        "value_bound",
    }
)
PROHIBITED_INPUT_FRAGMENTS = (
    "true",
    "oracle",
    "cluster",
    "occupancy",
    "stationary",
    "exact_return",
    "realized_error",
    "kernel_matrix",
    "generator",
    "prototype",
)

SMOKE_LENGTHS = (256, 1024)
SMOKE_TASK_INDEX = 0


# ---------------------------------------------------------------------------
# Observable validation
# ---------------------------------------------------------------------------


def validate_observables(
    observables: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    keys = set(observables)
    extra = sorted(keys - OBSERVABLE_KEYS)
    missing = sorted(OBSERVABLE_KEYS - keys)
    if extra:
        lowered = [key.lower() for key in extra]
        if any(
            fragment in key for key in lowered for fragment in PROHIBITED_INPUT_FRAGMENTS
        ):
            raise ValueError(f"prohibited observable input: {extra}")
        raise ValueError(f"unexpected observable input: {extra}")
    if missing:
        raise ValueError(f"missing observable input: {missing}")

    raw_signature_counts = np.asarray(observables["signature_counts"])
    raw_target_counts = np.asarray(observables["target_counts"])
    if not np.issubdtype(raw_signature_counts.dtype, np.integer):
        raise ValueError("signature counts must have integer dtype")
    if not np.issubdtype(raw_target_counts.dtype, np.integer):
        raise ValueError("target counts must have integer dtype")
    signature_q = np.asarray(observables["signature_q"], dtype=np.float64)
    signature_counts = raw_signature_counts.astype(np.int64, copy=False)
    target_sums = np.asarray(observables["target_sums"], dtype=np.float64)
    target_counts = raw_target_counts.astype(np.int64, copy=False)
    policy = np.asarray(observables["current_policy"], dtype=np.float64)
    if signature_q.ndim != 2:
        raise ValueError("signature_q must be a matrix")
    n_states, n_actions = signature_q.shape
    if n_states < 2 or n_actions < 3:
        raise ValueError("at least two states and three actions are required")
    for name, array in (
        ("signature_counts", signature_counts),
        ("target_sums", target_sums),
        ("target_counts", target_counts),
        ("current_policy", policy),
    ):
        if array.shape != signature_q.shape:
            raise ValueError(f"{name} shape differs from signature_q")
    if np.any(signature_counts < 0) or np.any(target_counts < 0):
        raise ValueError("counts must be nonnegative")
    if not np.all(np.isfinite(signature_q)) or not np.all(np.isfinite(target_sums)):
        raise ValueError("signature estimates and target sums must be finite")
    if np.any(np.abs(signature_q[signature_counts == 0]) > 1e-15):
        raise ValueError("unvisited signature entries must be zero")
    if np.any(np.abs(target_sums[target_counts == 0]) > 1e-12):
        raise ValueError("zero-count target sums must be zero")
    if not np.all(np.isfinite(policy)) or np.any(policy < 0.0):
        raise ValueError("current policy must be finite and nonnegative")
    if not np.allclose(policy.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("current policy rows must sum to one")
    pi_min = float(observables["pi_min"])
    value_bound = float(observables["value_bound"])
    if not np.isfinite(pi_min) or not 0.0 < pi_min < 1.0 / n_actions:
        raise ValueError("pi_min must lie in (0, 1 / n_actions)")
    if np.any(policy < pi_min - 1e-12):
        raise ValueError("current policy violates pi_min")
    if not np.isfinite(value_bound) or value_bound <= 0.0:
        raise ValueError("value_bound must be finite and positive")
    return (
        signature_q,
        signature_counts,
        target_sums,
        target_counts,
        policy,
        pi_min,
        value_bound,
    )


# ---------------------------------------------------------------------------
# Shared numeric primitives
# ---------------------------------------------------------------------------


def median_positive(values: Sequence[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    selected = np.sort(array[np.isfinite(array) & (array > 0.0)])
    if selected.size == 0:
        return None
    middle = selected.size // 2
    if selected.size % 2:
        return float(selected[middle])
    return float((selected[middle - 1] + selected[middle]) / 2.0)


def effective_sample_size(weights: np.ndarray, counts: np.ndarray) -> float | None:
    raw_counts = np.asarray(counts)
    if not np.issubdtype(raw_counts.dtype, np.integer):
        raise ValueError("counts must have integer dtype")
    count_array = raw_counts.astype(np.int64, copy=False)
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != count_array.shape:
        raise ValueError("weights and counts must share shape")
    if np.any(count_array < 0) or not np.all(np.isfinite(weight_array)):
        raise ValueError("ESS inputs must be finite and nonnegative")
    if np.any(weight_array < 0.0):
        raise ValueError("ESS weights must be nonnegative")
    numerator_base = float(np.sum(weight_array * count_array))
    denominator = float(np.sum(weight_array * weight_array * count_array))
    if numerator_base <= 0.0 or denominator <= 0.0 or not np.isfinite(denominator):
        return None
    value = numerator_base * numerator_base / denominator
    return value if np.isfinite(value) and value > 0.0 else None


def mean_ci(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    array = np.asarray(list(values), dtype=np.float64)
    mean = float(np.mean(array))
    if array.size < 2:
        return {"n": int(array.size), "mean": mean, "lower": None, "upper": None}
    sem = float(np.std(array, ddof=1) / np.sqrt(array.size))
    critical = float(student_t.ppf(0.975, df=array.size - 1))
    return {
        "n": int(array.size),
        "mean": mean,
        "lower": mean - critical * sem,
        "upper": mean + critical * sem,
    }


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(errors, dtype=np.float64)))))


# ---------------------------------------------------------------------------
# Stage 0: exact predecessor route replay from saved observable fields
# ---------------------------------------------------------------------------


def leave_one_action_out_distances(
    signature_q: np.ndarray,
    signature_counts: np.ndarray,
    value_bound: float,
) -> tuple[np.ndarray, np.ndarray, list[float | None]]:
    n_states, n_actions = signature_q.shape
    distances = np.full((n_actions, n_states, n_states), np.nan, dtype=np.float64)
    common_counts = np.zeros((n_actions, n_states, n_states), dtype=np.int64)
    bandwidths: list[float | None] = []
    for action in range(n_actions):
        for state in range(n_states):
            for other in range(n_states):
                if state == other:
                    distances[action, state, other] = 0.0
                    continue
                common = (signature_counts[state] > 0) & (signature_counts[other] > 0)
                common[action] = False
                common_count = int(np.sum(common))
                common_counts[action, state, other] = common_count
                if common_count < 2:
                    continue
                scaled = (
                    signature_q[state, common] - signature_q[other, common]
                ) / (2.0 * value_bound)
                distance = float(np.sqrt(np.mean(scaled * scaled)))
                if np.isfinite(distance):
                    distances[action, state, other] = distance
        upper = [
            float(distances[action, left, right])
            for left in range(n_states)
            for right in range(left + 1, n_states)
            if np.isfinite(distances[action, left, right])
        ]
        bandwidths.append(median_positive(upper))
    return distances, common_counts, bandwidths


def _same_action_distances(
    signature_q: np.ndarray,
    signature_counts: np.ndarray,
    value_bound: float,
) -> tuple[np.ndarray, list[float | None]]:
    n_states, n_actions = signature_q.shape
    distances = np.full((n_actions, n_states, n_states), np.nan, dtype=np.float64)
    bandwidths: list[float | None] = []
    for action in range(n_actions):
        for state in range(n_states):
            if signature_counts[state, action] <= 0:
                continue
            for other in range(n_states):
                if signature_counts[other, action] <= 0:
                    continue
                distances[action, state, other] = abs(
                    float(signature_q[state, action] - signature_q[other, action])
                ) / (2.0 * value_bound)
        upper = [
            float(distances[action, left, right])
            for left in range(n_states)
            for right in range(left + 1, n_states)
            if np.isfinite(distances[action, left, right])
        ]
        bandwidths.append(median_positive(upper))
    return distances, bandwidths


def _nullable_matrix(values: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in values
    ]


def _nullable_cube(values: np.ndarray) -> list[list[list[float | None]]]:
    return [
        [
            [float(value) if np.isfinite(value) else None for value in row]
            for row in plane
        ]
        for plane in values
    ]


def _diagnostic_policy(
    estimates: np.ndarray, current_policy: np.ndarray, pi_min: float
) -> dict[str, Any]:
    n_states, n_actions = estimates.shape
    updated = current_policy.copy()
    complete_rows: list[bool] = []
    selected: list[int | None] = []
    for state in range(n_states):
        complete = bool(np.all(np.isfinite(estimates[state])))
        complete_rows.append(complete)
        if not complete:
            selected.append(None)
            continue
        action = int(np.argmax(estimates[state]))
        selected.append(action)
        updated[state] = pi_min
        updated[state, action] = 1.0 - (n_actions - 1) * pi_min
    if not np.allclose(updated.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("diagnostic policy row sum failure")
    return {
        "complete_rows": complete_rows,
        "selected_action": selected,
        "policy": updated.tolist(),
    }


def _plain_route(
    estimates: np.ndarray,
    reasons: list[list[str]],
    ess: np.ndarray,
    policy: np.ndarray,
    pi_min: float,
) -> dict[str, Any]:
    return {
        "estimate": _nullable_matrix(estimates),
        "reason": reasons,
        "effective_sample_size": _nullable_matrix(ess),
        "diagnostic_policy": _diagnostic_policy(estimates, policy, pi_min),
    }


def _kernel_route_replay(
    distances: np.ndarray,
    bandwidths: list[float | None],
    target_sums: np.ndarray,
    target_counts: np.ndarray,
    policy: np.ndarray,
    pi_min: float,
    *,
    primary: bool,
    common_counts: np.ndarray | None = None,
) -> dict[str, Any]:
    n_states, n_actions = target_counts.shape
    estimates = np.full((n_states, n_actions), np.nan, dtype=np.float64)
    ess = np.full_like(estimates, np.nan)
    weights = np.full((n_states, n_actions, n_states), np.nan, dtype=np.float64)
    denominators = np.full_like(estimates, np.nan)
    eligible = np.zeros((n_states, n_actions), dtype=bool)
    reasons = [
        ["insufficient_common_actions" for _ in range(n_actions)]
        for _ in range(n_states)
    ]

    for state in range(n_states):
        for action in range(n_actions):
            finite_neighbors = [
                other
                for other in range(n_states)
                if other != state and np.isfinite(distances[action, state, other])
            ]
            self_available = bool(np.isfinite(distances[action, state, state]))
            if (primary and target_counts[state, action] == 0 and not finite_neighbors) or (
                not primary and not self_available
            ):
                reasons[state][action] = "insufficient_common_actions"
                continue
            bandwidth = bandwidths[action]
            if bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0.0:
                reasons[state][action] = "bandwidth_unavailable"
                continue
            eligible[state, action] = True
            target_total = int(np.sum(target_counts[:, action]))
            if (not primary and target_counts[state, action] == 0) or (
                primary and target_total == 0
            ):
                reasons[state][action] = "target_source_unavailable"
                continue
            finite_sources = np.isfinite(distances[action, state])
            kernel = np.zeros(n_states, dtype=np.float64)
            kernel[finite_sources] = np.exp(
                -np.square(distances[action, state, finite_sources])
                / (2.0 * bandwidth * bandwidth)
            )
            weights[state, action] = kernel
            denominator = float(np.sum(kernel * target_counts[:, action]))
            denominators[state, action] = denominator
            if denominator <= 0.0 or not np.isfinite(denominator):
                reasons[state][action] = "kernel_denominator_invalid"
                continue
            estimate = float(np.sum(kernel * target_sums[:, action]) / denominator)
            observed_ess = effective_sample_size(kernel, target_counts[:, action])
            if observed_ess is None:
                reasons[state][action] = "kernel_denominator_invalid"
                continue
            if not np.isfinite(estimate):
                reasons[state][action] = "estimate_nonfinite"
                continue
            estimates[state, action] = estimate
            ess[state, action] = observed_ess
            reasons[state][action] = "ok"

    result = _plain_route(estimates, reasons, ess, policy, pi_min)
    kernel_record: dict[str, Any] = {
        "bandwidth_by_action": bandwidths,
        "distance_by_action": _nullable_cube(distances),
        "weight_by_target": _nullable_cube(weights),
        "denominator": _nullable_matrix(denominators),
        "signature_eligible": eligible.tolist(),
    }
    if common_counts is not None:
        kernel_record["common_action_count_by_action"] = common_counts.tolist()
    result["kernel"] = kernel_record
    return result


def replay_predecessor_routes(observables: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the four frozen FP-KERN-001 routes from observable fields only."""
    (
        signature_q,
        signature_counts,
        target_sums,
        target_counts,
        policy,
        pi_min,
        value_bound,
    ) = validate_observables(observables)
    n_states, n_actions = signature_q.shape

    local = np.full((n_states, n_actions), np.nan, dtype=np.float64)
    local_ess = np.full_like(local, np.nan)
    local_reasons = [
        ["target_source_unavailable" for _ in range(n_actions)]
        for _ in range(n_states)
    ]
    visited = target_counts > 0
    local[visited] = target_sums[visited] / target_counts[visited]
    local_ess[visited] = target_counts[visited]
    for state, action in np.argwhere(visited):
        local_reasons[int(state)][int(action)] = "ok"

    pooled = np.full_like(local, np.nan)
    pooled_ess = np.full_like(local, np.nan)
    pooled_reasons = [
        ["target_source_unavailable" for _ in range(n_actions)]
        for _ in range(n_states)
    ]
    for action in range(n_actions):
        total_count = int(np.sum(target_counts[:, action]))
        if total_count <= 0:
            continue
        estimate = float(np.sum(target_sums[:, action]) / total_count)
        pooled[:, action] = estimate
        pooled_ess[:, action] = total_count
        for state in range(n_states):
            pooled_reasons[state][action] = "ok"

    loao_distances, common_counts, loao_bandwidths = leave_one_action_out_distances(
        signature_q, signature_counts, value_bound
    )
    same_distances, same_bandwidths = _same_action_distances(
        signature_q, signature_counts, value_bound
    )
    routes = {
        "local_unpooled": _plain_route(local, local_reasons, local_ess, policy, pi_min),
        "action_only_pool": _plain_route(
            pooled, pooled_reasons, pooled_ess, policy, pi_min
        ),
        "same_action_anchor_kernel": _kernel_route_replay(
            same_distances,
            same_bandwidths,
            target_sums,
            target_counts,
            policy,
            pi_min,
            primary=False,
        ),
        "leave_one_action_out_kernel": _kernel_route_replay(
            loao_distances,
            loao_bandwidths,
            target_sums,
            target_counts,
            policy,
            pi_min,
            primary=True,
            common_counts=common_counts,
        ),
    }
    return {
        "route_order": list(PREDECESSOR_ROUTE_ORDER),
        "shape": {"n_states": n_states, "n_actions": n_actions},
        "routes": routes,
    }


def assert_exact_route_replay(replayed: Any, serialized: Any, path: str = "root") -> None:
    """Exact equality for every serialized route value (floats included)."""
    if isinstance(serialized, dict):
        if not isinstance(replayed, dict) or set(replayed) != set(serialized):
            raise AssertionError(f"dictionary mismatch at {path}")
        for key in serialized:
            assert_exact_route_replay(replayed[key], serialized[key], f"{path}.{key}")
        return
    if isinstance(serialized, list):
        if not isinstance(replayed, list) or len(replayed) != len(serialized):
            raise AssertionError(f"list mismatch at {path}")
        for index, (left, right) in enumerate(zip(replayed, serialized, strict=True)):
            assert_exact_route_replay(left, right, f"{path}[{index}]")
        return
    if isinstance(serialized, bool) or serialized is None or isinstance(serialized, str):
        if replayed != serialized or type(replayed) is not type(serialized):
            raise AssertionError(
                f"value mismatch at {path}: {replayed!r} != {serialized!r}"
            )
        return
    if isinstance(serialized, int):
        if replayed != serialized:
            raise AssertionError(f"integer mismatch at {path}")
        return
    if isinstance(serialized, float):
        if not isinstance(replayed, (int, float)) or float(replayed) != serialized:
            raise AssertionError(
                f"exact float mismatch at {path}: {replayed!r} != {serialized!r}"
            )
        return
    raise TypeError(f"unsupported comparison value at {path}: {type(serialized).__name__}")


def assert_close_nested(observed: Any, expected: Any, path: str = "root") -> None:
    """Absolute-tolerance 1e-12 comparison for recomputed metric summaries."""
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(observed) != set(expected):
            raise AssertionError(f"dictionary mismatch at {path}")
        for key in expected:
            assert_close_nested(observed[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            raise AssertionError(f"list mismatch at {path}")
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            assert_close_nested(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if observed != expected:
            raise AssertionError(f"value mismatch at {path}: {observed!r} != {expected!r}")
        return
    if isinstance(expected, int) and not isinstance(expected, bool):
        if observed != expected:
            raise AssertionError(f"integer mismatch at {path}")
        return
    if isinstance(expected, float):
        if not isinstance(observed, (int, float)) or not np.isclose(
            float(observed), expected, rtol=1e-12, atol=1e-12
        ):
            raise AssertionError(
                f"numeric mismatch at {path}: {observed!r} != {expected!r}"
            )
        return
    raise TypeError(f"unsupported comparison value at {path}: {type(expected).__name__}")


# ---------------------------------------------------------------------------
# Frozen diagnostic routes
# ---------------------------------------------------------------------------


def pool_estimate(
    target_sums: np.ndarray,
    target_counts: np.ndarray,
    action: int,
    source_states: Sequence[int],
) -> tuple[float | None, str, float]:
    """Common count-weighted pooled estimate over the declared source set."""
    sources = sorted(int(state) for state in source_states)
    numerator = 0.0
    denominator = 0.0
    for source in sources:
        numerator += float(target_sums[source, action])
        denominator += float(target_counts[source, action])
    if denominator <= 0.0:
        return None, "target_source_unavailable", denominator
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return None, "pool_denominator_invalid", denominator
    estimate = numerator / denominator
    if not np.isfinite(estimate):
        return None, "pool_denominator_invalid", denominator
    return estimate, "ok", denominator


def oracle_q_nearest2_peers(
    true_q: np.ndarray, target_counts: np.ndarray, state: int, action: int
) -> list[int]:
    """Frozen favorable peer ranking: |dQ| then state index, at most two."""
    candidates = [
        other
        for other in range(true_q.shape[0])
        if other != state and target_counts[other, action] > 0
    ]
    candidates.sort(
        key=lambda other: (
            abs(float(true_q[state, action]) - float(true_q[other, action])),
            other,
        )
    )
    return candidates[:PEER_COUNT]


def oracle_q_nearest2_route(
    observables: Mapping[str, Any], true_q: np.ndarray
) -> dict[str, Any]:
    (
        _,
        _,
        target_sums,
        target_counts,
        _,
        _,
        _,
    ) = validate_observables(observables)
    true_q = np.asarray(true_q, dtype=np.float64)
    if true_q.shape != target_counts.shape:
        raise ValueError("true_q shape differs from target_counts")
    n_states, n_actions = target_counts.shape
    estimates: list[list[float | None]] = []
    reasons: list[list[str]] = []
    peers_out: list[list[list[int]]] = []
    sources_out: list[list[list[int]]] = []
    denominators: list[list[float | None]] = []
    for state in range(n_states):
        row_estimates: list[float | None] = []
        row_reasons: list[str] = []
        row_peers: list[list[int]] = []
        row_sources: list[list[int]] = []
        row_denominators: list[float | None] = []
        for action in range(n_actions):
            peers = oracle_q_nearest2_peers(true_q, target_counts, state, action)
            sources = sorted(
                peers + ([state] if target_counts[state, action] > 0 else [])
            )
            estimate, reason, denominator = pool_estimate(
                target_sums, target_counts, action, sources
            )
            row_estimates.append(estimate)
            row_reasons.append(reason)
            row_peers.append(peers)
            row_sources.append(sources if reason == "ok" else sources)
            row_denominators.append(denominator)
        estimates.append(row_estimates)
        reasons.append(row_reasons)
        peers_out.append(row_peers)
        sources_out.append(row_sources)
        denominators.append(row_denominators)
    return {
        "estimate": estimates,
        "reason": reasons,
        "peers": peers_out,
        "source_states": sources_out,
        "denominator": denominators,
    }


def generator_cluster_sources(cluster_by_state: Sequence[int]) -> list[list[int]]:
    labels = [int(label) for label in cluster_by_state]
    n_states = len(labels)
    if n_states != N_STATES:
        raise ValueError("generator cluster labels must cover six states")
    groups: dict[int, list[int]] = {}
    for state, label in enumerate(labels):
        groups.setdefault(label, []).append(state)
    if len(groups) != 2 or sorted(len(members) for members in groups.values()) != [3, 3]:
        raise ValueError("generator labels must form two balanced clusters")
    return [sorted(groups[labels[state]]) for state in range(n_states)]


def oracle_generator_cluster_route(
    observables: Mapping[str, Any], cluster_by_state: Sequence[int]
) -> dict[str, Any]:
    (
        _,
        _,
        target_sums,
        target_counts,
        _,
        _,
        _,
    ) = validate_observables(observables)
    sources_by_state = generator_cluster_sources(cluster_by_state)
    n_states, n_actions = target_counts.shape
    estimates: list[list[float | None]] = []
    reasons: list[list[str]] = []
    sources_out: list[list[list[int]]] = []
    denominators: list[list[float | None]] = []
    for state in range(n_states):
        row_estimates: list[float | None] = []
        row_reasons: list[str] = []
        row_sources: list[list[int]] = []
        row_denominators: list[float | None] = []
        for action in range(n_actions):
            cluster = sources_by_state[state]
            if target_counts[state, action] > 0:
                sources = list(cluster)
            else:
                sources = [source for source in cluster if source != state]
            estimate, reason, denominator = pool_estimate(
                target_sums, target_counts, action, sources
            )
            row_estimates.append(estimate)
            row_reasons.append(reason)
            row_sources.append(sources)
            row_denominators.append(denominator)
        estimates.append(row_estimates)
        reasons.append(row_reasons)
        sources_out.append(row_sources)
        denominators.append(row_denominators)
    return {
        "status": "ok",
        "estimate": estimates,
        "reason": reasons,
        "source_states": sources_out,
        "denominator": denominators,
    }


def balanced_partitions(n_states: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All unique balanced two-group partitions; state zero in group one."""
    if n_states != N_STATES:
        raise ValueError("the frozen diagnostic uses six states")
    partitions = []
    others = list(range(1, n_states))
    for left in range(len(others)):
        for right in range(left + 1, len(others)):
            first = tuple(sorted((0, others[left], others[right])))
            second = tuple(state for state in range(n_states) if state not in first)
            partitions.append((first, second))
    return partitions


def _partition_score(distance_matrix: np.ndarray, partition: tuple[tuple[int, ...], ...]) -> float:
    replaced = np.where(np.isfinite(distance_matrix), distance_matrix, MISSING_DISTANCE)
    within: list[float] = []
    for group in partition:
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                within.append(
                    float(replaced[group[left_index], group[right_index]]) ** 2
                )
    return float(np.mean(np.sort(np.asarray(within, dtype=np.float64))))


def select_partition(distance_matrix: np.ndarray) -> dict[str, Any]:
    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.shape != (N_STATES, N_STATES):
        raise ValueError("partition distances must be a six-by-six matrix")
    partitions = balanced_partitions(N_STATES)
    scores = [_partition_score(distances, partition) for partition in partitions]
    minimum = min(scores)
    winners = [index for index, score in enumerate(scores) if score == minimum]
    if len(winners) != 1:
        return {
            "status": "partition_tie",
            "groups": None,
            "score": None,
            "scores": scores,
        }
    first, second = partitions[winners[0]]
    return {
        "status": "ok",
        "groups": [list(first), list(second)],
        "score": minimum,
        "scores": scores,
    }


def observable_balanced_cluster_route(observables: Mapping[str, Any]) -> dict[str, Any]:
    (
        signature_q,
        signature_counts,
        target_sums,
        target_counts,
        _,
        _,
        value_bound,
    ) = validate_observables(observables)
    if signature_q.shape != (N_STATES, N_ACTIONS):
        raise ValueError("the frozen diagnostic uses six states and four actions")
    distances, _, _ = leave_one_action_out_distances(
        signature_q, signature_counts, value_bound
    )
    estimates: list[list[float | None]] = []
    reasons: list[list[str]] = []
    sources_out: list[list[list[int]]] = []
    denominators: list[list[float | None]] = []
    partitions_out: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for action in range(N_ACTIONS):
        selections.append(select_partition(np.asarray(distances[action])))
    for state in range(N_STATES):
        row_estimates: list[float | None] = []
        row_reasons: list[str] = []
        row_sources: list[list[int]] = []
        row_denominators: list[float | None] = []
        for action in range(N_ACTIONS):
            selection = selections[action]
            if selection["status"] != "ok":
                row_estimates.append(None)
                row_reasons.append("partition_tie")
                row_sources.append([])
                row_denominators.append(None)
                continue
            group = next(
                group for group in selection["groups"] if state in group
            )
            if target_counts[state, action] > 0:
                sources = sorted(group)
            else:
                sources = sorted(source for source in group if source != state)
            estimate, reason, denominator = pool_estimate(
                target_sums, target_counts, action, sources
            )
            row_estimates.append(estimate)
            row_reasons.append(reason)
            row_sources.append(sources)
            row_denominators.append(denominator)
        estimates.append(row_estimates)
        reasons.append(row_reasons)
        sources_out.append(row_sources)
        denominators.append(row_denominators)
    for action in range(N_ACTIONS):
        partitions_out.append(selections[action])
    return {
        "estimate": estimates,
        "reason": reasons,
        "source_states": sources_out,
        "denominator": denominators,
        "partition_by_action": partitions_out,
    }


# ---------------------------------------------------------------------------
# Metrics (frozen predecessor definitions, parameterized by route)
# ---------------------------------------------------------------------------


def false_improvement_counts(
    true_q: np.ndarray, primary: np.ndarray, baseline: np.ndarray
) -> tuple[int, int, int]:
    if true_q.shape != primary.shape or true_q.shape != baseline.shape:
        raise ValueError("false-improvement arrays must have one common shape")
    primary_false = 0
    baseline_false = 0
    comparisons = 0
    for state in range(true_q.shape[0]):
        for left in range(true_q.shape[1]):
            for right in range(left + 1, true_q.shape[1]):
                if not (
                    np.isfinite(primary[state, left])
                    and np.isfinite(primary[state, right])
                    and np.isfinite(baseline[state, left])
                    and np.isfinite(baseline[state, right])
                ):
                    continue
                true_difference = float(true_q[state, left] - true_q[state, right])
                primary_difference = float(primary[state, left] - primary[state, right])
                baseline_difference = float(
                    baseline[state, left] - baseline[state, right]
                )
                primary_false += int(
                    primary_difference > 0.0 and true_difference <= 0.0
                )
                baseline_false += int(
                    baseline_difference > 0.0 and true_difference <= 0.0
                )
                comparisons += 1
    return primary_false, baseline_false, comparisons


def _estimate_array(route_output: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [np.nan if value is None else float(value) for value in row]
            for row in route_output["estimate"]
        ],
        dtype=np.float64,
    )


def _serialized_estimate_array(record: dict[str, Any], route: str) -> np.ndarray:
    return _estimate_array(record["kernel_generalization"]["routes"][route])


def route_family_screen(
    records: list[dict[str, Any]],
    primary_estimates: list[np.ndarray],
    eligibility: list[np.ndarray],
) -> dict[str, Any]:
    """Frozen five-item screen for one diagnostic route within one family."""
    zero_eligible = 0
    zero_covered = 0
    zero_primary: list[float] = []
    zero_baseline: list[float] = []
    zero_improvement: list[float] = []
    sparse_primary: list[float] = []
    sparse_baseline: list[float] = []
    sparse_improvement: list[float] = []
    top_improvement: list[float] = []
    top_primary_values: list[float] = []
    top_baseline_values: list[float] = []
    primary_false = 0
    baseline_false = 0
    false_denominator = 0
    zero_empty_records = 0
    sparse_empty_records = 0
    top_empty_records = 0

    for record, primary, eligible in zip(
        records, primary_estimates, eligibility, strict=True
    ):
        counts = np.asarray(record["observable_inputs"]["target_counts"], dtype=np.int64)
        true_q = np.asarray(record["oracle_audit"]["true_q"], dtype=np.float64)
        action_pool = _serialized_estimate_array(record, "action_only_pool")
        local = _serialized_estimate_array(record, "local_unpooled")
        zero_mask_all = counts == 0
        zero_eligible += int(np.sum(zero_mask_all & eligible))
        zero_covered += int(np.sum(zero_mask_all & eligible & np.isfinite(primary)))

        zero_common = zero_mask_all & np.isfinite(primary) & np.isfinite(action_pool)
        if np.any(zero_common):
            primary_rmse = rmse(primary[zero_common] - true_q[zero_common])
            baseline_rmse = rmse(action_pool[zero_common] - true_q[zero_common])
            zero_primary.append(primary_rmse)
            zero_baseline.append(baseline_rmse)
            zero_improvement.append(baseline_rmse - primary_rmse)
        else:
            zero_empty_records += 1

        sparse_common = (
            (counts >= 1) & (counts <= 4) & np.isfinite(primary) & np.isfinite(local)
        )
        if np.any(sparse_common):
            primary_rmse = rmse(primary[sparse_common] - true_q[sparse_common])
            baseline_rmse = rmse(local[sparse_common] - true_q[sparse_common])
            sparse_primary.append(primary_rmse)
            sparse_baseline.append(baseline_rmse)
            sparse_improvement.append(baseline_rmse - primary_rmse)
        else:
            sparse_empty_records += 1

        sparse_states = np.any(counts <= 4, axis=1)
        complete = (
            np.all(np.isfinite(primary), axis=1)
            & np.all(np.isfinite(action_pool), axis=1)
            & sparse_states
        )
        if np.any(complete):
            truth_actions = np.argmax(true_q[complete], axis=1)
            primary_accuracy = float(
                np.mean(np.argmax(primary[complete], axis=1) == truth_actions)
            )
            baseline_accuracy = float(
                np.mean(np.argmax(action_pool[complete], axis=1) == truth_actions)
            )
            top_primary_values.append(primary_accuracy)
            top_baseline_values.append(baseline_accuracy)
            top_improvement.append(primary_accuracy - baseline_accuracy)
        else:
            top_empty_records += 1

        record_primary_false, record_baseline_false, record_comparisons = (
            false_improvement_counts(true_q, primary, action_pool)
        )
        primary_false += record_primary_false
        baseline_false += record_baseline_false
        false_denominator += record_comparisons

    coverage = zero_covered / zero_eligible if zero_eligible else None
    zero_ci = mean_ci(zero_improvement)
    sparse_ci = mean_ci(sparse_improvement)
    top_ci = mean_ci(top_improvement)
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
    primary_false_rate = primary_false / false_denominator if false_denominator else None
    baseline_false_rate = (
        baseline_false / false_denominator if false_denominator else None
    )
    criteria = {
        "zero_eligible_coverage_at_least_50pct": coverage is not None and coverage >= 0.50,
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
            primary_false_rate is not None
            and baseline_false_rate is not None
            and primary_false_rate <= baseline_false_rate + 0.01
        ),
    }
    return {
        "records": len(records),
        "zero_coverage": {
            "eligible": zero_eligible,
            "covered": zero_covered,
            "rate": coverage,
        },
        "zero_rmse": {
            "primary_mean": float(np.mean(zero_primary)) if zero_primary else None,
            "baseline_mean": float(np.mean(zero_baseline)) if zero_baseline else None,
            "relative_improvement": zero_relative,
            "paired_improvement": zero_ci,
            "empty_records": zero_empty_records,
        },
        "sparse_rmse": {
            "primary_mean": float(np.mean(sparse_primary)) if sparse_primary else None,
            "baseline_mean": float(np.mean(sparse_baseline)) if sparse_baseline else None,
            "relative_improvement": sparse_relative,
            "paired_improvement": sparse_ci,
            "empty_records": sparse_empty_records,
        },
        "top_action": {
            "primary_mean": float(np.mean(top_primary_values))
            if top_primary_values
            else None,
            "baseline_mean": float(np.mean(top_baseline_values))
            if top_baseline_values
            else None,
            "paired_improvement": top_ci,
            "empty_records": top_empty_records,
        },
        "false_improvement": {
            "comparisons": false_denominator,
            "primary_count": primary_false,
            "baseline_count": baseline_false,
            "primary_rate": primary_false_rate,
            "baseline_rate": baseline_false_rate,
        },
        "criteria": criteria,
        "screen_pass": all(criteria.values()),
    }


def _record_action_correlations(
    distances: list[list[list[float | None]]], true_q: np.ndarray
) -> list[float]:
    correlations: list[float] = []
    for action in range(true_q.shape[1]):
        observed_distance: list[float] = []
        observed_difference: list[float] = []
        for left in range(true_q.shape[0]):
            for right in range(left + 1, true_q.shape[0]):
                distance = distances[action][left][right]
                if distance is None or not np.isfinite(float(distance)):
                    continue
                observed_distance.append(float(distance))
                observed_difference.append(
                    abs(float(true_q[left, action] - true_q[right, action]))
                )
        if len(observed_distance) < 3:
            continue
        correlation = float(spearmanr(observed_distance, observed_difference).statistic)
        if np.isfinite(correlation):
            correlations.append(correlation)
    return correlations


def predecessor_family_screen(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The FP-KERN-001 screen exactly, with the saved LOAO route as primary."""
    zero_eligible = 0
    zero_covered = 0
    zero_primary: list[float] = []
    zero_baseline: list[float] = []
    zero_improvement: list[float] = []
    sparse_primary: list[float] = []
    sparse_baseline: list[float] = []
    sparse_improvement: list[float] = []
    top_improvement: list[float] = []
    top_primary_values: list[float] = []
    top_baseline_values: list[float] = []
    primary_false = 0
    baseline_false = 0
    false_denominator = 0
    correlations: list[float] = []
    zero_empty_records = 0
    sparse_empty_records = 0
    top_empty_records = 0

    for record in records:
        counts = np.asarray(record["observable_inputs"]["target_counts"], dtype=np.int64)
        true_q = np.asarray(record["oracle_audit"]["true_q"], dtype=np.float64)
        primary = _serialized_estimate_array(record, "leave_one_action_out_kernel")
        action_pool = _serialized_estimate_array(record, "action_only_pool")
        local = _serialized_estimate_array(record, "local_unpooled")
        eligible = np.asarray(
            record["kernel_generalization"]["routes"]["leave_one_action_out_kernel"][
                "kernel"
            ]["signature_eligible"],
            dtype=bool,
        )
        zero_mask_all = counts == 0
        zero_eligible += int(np.sum(zero_mask_all & eligible))
        zero_covered += int(np.sum(zero_mask_all & eligible & np.isfinite(primary)))

        zero_common = zero_mask_all & np.isfinite(primary) & np.isfinite(action_pool)
        if np.any(zero_common):
            primary_rmse = rmse(primary[zero_common] - true_q[zero_common])
            baseline_rmse = rmse(action_pool[zero_common] - true_q[zero_common])
            zero_primary.append(primary_rmse)
            zero_baseline.append(baseline_rmse)
            zero_improvement.append(baseline_rmse - primary_rmse)
        else:
            zero_empty_records += 1

        sparse_common = (
            (counts >= 1) & (counts <= 4) & np.isfinite(primary) & np.isfinite(local)
        )
        if np.any(sparse_common):
            primary_rmse = rmse(primary[sparse_common] - true_q[sparse_common])
            baseline_rmse = rmse(local[sparse_common] - true_q[sparse_common])
            sparse_primary.append(primary_rmse)
            sparse_baseline.append(baseline_rmse)
            sparse_improvement.append(baseline_rmse - primary_rmse)
        else:
            sparse_empty_records += 1

        sparse_states = np.any(counts <= 4, axis=1)
        complete = (
            np.all(np.isfinite(primary), axis=1)
            & np.all(np.isfinite(action_pool), axis=1)
            & sparse_states
        )
        if np.any(complete):
            truth_actions = np.argmax(true_q[complete], axis=1)
            primary_accuracy = float(
                np.mean(np.argmax(primary[complete], axis=1) == truth_actions)
            )
            baseline_accuracy = float(
                np.mean(np.argmax(action_pool[complete], axis=1) == truth_actions)
            )
            top_primary_values.append(primary_accuracy)
            top_baseline_values.append(baseline_accuracy)
            top_improvement.append(primary_accuracy - baseline_accuracy)
        else:
            top_empty_records += 1

        record_primary_false, record_baseline_false, record_comparisons = (
            false_improvement_counts(true_q, primary, action_pool)
        )
        primary_false += record_primary_false
        baseline_false += record_baseline_false
        false_denominator += record_comparisons

        distances = record["kernel_generalization"]["routes"][
            "leave_one_action_out_kernel"
        ]["kernel"]["distance_by_action"]
        correlations.extend(_record_action_correlations(distances, true_q))

    coverage = zero_covered / zero_eligible if zero_eligible else None
    zero_ci = mean_ci(zero_improvement)
    sparse_ci = mean_ci(sparse_improvement)
    top_ci = mean_ci(top_improvement)
    correlation_ci = mean_ci(correlations)
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
    primary_false_rate = primary_false / false_denominator if false_denominator else None
    baseline_false_rate = (
        baseline_false / false_denominator if false_denominator else None
    )
    criteria = {
        "zero_eligible_coverage_at_least_50pct": coverage is not None and coverage >= 0.50,
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
            primary_false_rate is not None
            and baseline_false_rate is not None
            and primary_false_rate <= baseline_false_rate + 0.01
        ),
    }
    return {
        "records": len(records),
        "zero_coverage": {
            "eligible": zero_eligible,
            "covered": zero_covered,
            "rate": coverage,
        },
        "zero_rmse": {
            "primary_mean": float(np.mean(zero_primary)) if zero_primary else None,
            "baseline_mean": float(np.mean(zero_baseline)) if zero_baseline else None,
            "relative_improvement": zero_relative,
            "paired_improvement": zero_ci,
            "empty_records": zero_empty_records,
        },
        "sparse_rmse": {
            "primary_mean": float(np.mean(sparse_primary)) if sparse_primary else None,
            "baseline_mean": float(np.mean(sparse_baseline)) if sparse_baseline else None,
            "relative_improvement": sparse_relative,
            "paired_improvement": sparse_ci,
            "empty_records": sparse_empty_records,
        },
        "top_action": {
            "primary_mean": float(np.mean(top_primary_values))
            if top_primary_values
            else None,
            "baseline_mean": float(np.mean(top_baseline_values))
            if top_baseline_values
            else None,
            "paired_improvement": top_ci,
            "empty_records": top_empty_records,
        },
        "false_improvement": {
            "comparisons": false_denominator,
            "primary_count": primary_false,
            "baseline_count": baseline_false,
            "primary_rate": primary_false_rate,
            "baseline_rate": baseline_false_rate,
        },
        "signature_q_difference_spearman": {
            "record_action_summary": correlation_ci,
            "passes_secondary_hypothesis": (
                correlation_ci["mean"] is not None
                and correlation_ci["mean"] > 0.0
                and correlation_ci["lower"] is not None
                and correlation_ci["lower"] > 0.0
            ),
        },
        "criteria": criteria,
        "screen_pass": all(criteria.values()),
    }


def predecessor_route_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for route in PREDECESSOR_ROUTE_ORDER:
        errors_by_bin: dict[str, list[float]] = {
            "0": [],
            "1-4": [],
            "5-16": [],
            "17+": [],
        }
        estimates = 0
        total = 0
        return_changes: list[float] = []
        for record in records:
            counts = np.asarray(record["observable_inputs"]["target_counts"], dtype=np.int64)
            estimate = _serialized_estimate_array(record, route)
            truth = np.asarray(record["oracle_audit"]["true_q"], dtype=np.float64)
            finite = np.isfinite(estimate)
            estimates += int(np.sum(finite))
            total += int(finite.size)
            for state, action in np.argwhere(finite):
                count = int(counts[state, action])
                label = (
                    "0"
                    if count == 0
                    else "1-4"
                    if count <= 4
                    else "5-16"
                    if count <= 16
                    else "17+"
                )
                errors_by_bin[label].append(
                    float(estimate[state, action] - truth[state, action])
                )
            return_changes.append(
                float(record["oracle_audit"]["routes"][route]["return_change"])
            )
        output[route] = {
            "coverage": estimates / total if total else None,
            "estimated_pairs": estimates,
            "total_pairs": total,
            "count_bins": {
                label: {
                    "n": len(values),
                    "rmse": rmse(np.asarray(values)) if values else None,
                    "mae": float(np.mean(np.abs(values))) if values else None,
                }
                for label, values in errors_by_bin.items()
            },
            "diagnostic_return_change": mean_ci(return_changes),
        }
    return output


# ---------------------------------------------------------------------------
# Secondary cluster diagnostics (hidden family only)
# ---------------------------------------------------------------------------


def adjusted_rand_index(labels_a: Sequence[int], labels_b: Sequence[int]) -> float:
    a = np.asarray(list(labels_a), dtype=np.int64)
    b = np.asarray(list(labels_b), dtype=np.int64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("ARI inputs must share a one-dimensional shape")
    n = int(a.size)
    if n < 2:
        raise ValueError("ARI requires at least two elements")
    contingency: dict[tuple[int, int], int] = {}
    for left, right in zip(a, b, strict=True):
        key = (int(left), int(right))
        contingency[key] = contingency.get(key, 0) + 1

    def choose_two(value: int) -> float:
        return value * (value - 1) / 2.0

    sum_cells = sum(choose_two(count) for count in contingency.values())
    a_counts: dict[int, int] = {}
    b_counts: dict[int, int] = {}
    for label in a:
        a_counts[int(label)] = a_counts.get(int(label), 0) + 1
    for label in b:
        b_counts[int(label)] = b_counts.get(int(label), 0) + 1
    sum_a = sum(choose_two(count) for count in a_counts.values())
    sum_b = sum(choose_two(count) for count in b_counts.values())
    total = choose_two(n)
    expected = sum_a * sum_b / total
    maximum = (sum_a + sum_b) / 2.0
    if maximum == expected:
        return 1.0 if sum_cells == maximum else 0.0
    return (sum_cells - expected) / (maximum - expected)


def peer_precision_counts(
    groups: Sequence[Sequence[int]], labels: Sequence[int]
) -> tuple[int, int]:
    matches = 0
    total = 0
    for group in groups:
        for state in group:
            for peer in group:
                if peer == state:
                    continue
                total += 1
                matches += int(int(labels[state]) == int(labels[peer]))
    return matches, total


def cluster_recovery_diagnostics(
    records: list[dict[str, Any]],
    observable_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    ari_values: list[float] = []
    peer_matches = 0
    peer_total = 0
    tied_record_actions = 0
    emitted_record_actions = 0
    for record, output in zip(records, observable_outputs, strict=True):
        labels = record["oracle_audit"]["generator"]["cluster_by_state"]
        for action in range(N_ACTIONS):
            selection = output["partition_by_action"][action]
            if selection["status"] != "ok":
                tied_record_actions += 1
                continue
            emitted_record_actions += 1
            groups = selection["groups"]
            partition_labels = [0] * N_STATES
            for state in groups[1]:
                partition_labels[state] = 1
            ari_values.append(adjusted_rand_index(partition_labels, labels))
            matches, total = peer_precision_counts(groups, labels)
            peer_matches += matches
            peer_total += total
    ari_summary = mean_ci(ari_values)
    return {
        "emitted_record_actions": emitted_record_actions,
        "excluded_partition_ties": tied_record_actions,
        "adjusted_rand_index": {
            "summary": ari_summary,
            "available": ari_summary["n"] >= 2,
        },
        "peer_precision": {
            "matches": peer_matches,
            "assignments": peer_total,
            "micro_average": (peer_matches / peer_total) if peer_total else None,
        },
    }


def oracle_benefit_recovery(
    records: list[dict[str, Any]],
    observable_estimates: list[np.ndarray],
    generator_estimates: list[np.ndarray],
) -> dict[str, Any]:
    """Frozen per-bin oracle-benefit recovery on exact common record/pair sets."""
    result: dict[str, Any] = {}
    for bin_name, baseline_route, mask_fn in (
        ("0", "action_only_pool", lambda counts: counts == 0),
        ("1-4", "local_unpooled", lambda counts: (counts >= 1) & (counts <= 4)),
    ):
        observable_improvements: list[float] = []
        generator_improvements: list[float] = []
        contributing_records = 0
        for record, observable, generator in zip(
            records, observable_estimates, generator_estimates, strict=True
        ):
            counts = np.asarray(record["observable_inputs"]["target_counts"], dtype=np.int64)
            true_q = np.asarray(record["oracle_audit"]["true_q"], dtype=np.float64)
            baseline = _serialized_estimate_array(record, baseline_route)
            common = (
                mask_fn(counts)
                & np.isfinite(observable)
                & np.isfinite(generator)
                & np.isfinite(baseline)
            )
            if not np.any(common):
                continue
            contributing_records += 1
            observable_rmse = rmse(observable[common] - true_q[common])
            generator_rmse = rmse(generator[common] - true_q[common])
            baseline_rmse = rmse(baseline[common] - true_q[common])
            observable_improvements.append(baseline_rmse - observable_rmse)
            generator_improvements.append(baseline_rmse - generator_rmse)
        numerator = (
            float(np.mean(observable_improvements)) if observable_improvements else None
        )
        denominator = (
            float(np.mean(generator_improvements)) if generator_improvements else None
        )
        available = (
            numerator is not None
            and denominator is not None
            and np.isfinite(denominator)
            and denominator > 0.0
        )
        result[bin_name] = {
            "baseline_route": baseline_route,
            "contributing_records": contributing_records,
            "mean_observable_improvement": numerator,
            "mean_generator_improvement": denominator,
            "recovery": (numerator / denominator) if available else None,
            "available": bool(available),
        }
    return result


# ---------------------------------------------------------------------------
# Ordered final classification
# ---------------------------------------------------------------------------


def classify(
    stage0_ok: bool, hidden_gates: Mapping[str, bool], observable_current_pass: bool
) -> str:
    if not stage0_ok:
        return "INVALID_INPUT"
    observable_hidden = bool(hidden_gates["observable_structure_useful"])
    generator_hidden = bool(hidden_gates["generator_structure_useful"])
    peer_hidden = bool(hidden_gates["peer_headroom"])
    if observable_hidden and observable_current_pass:
        return "GENERAL_PROMISING"
    if observable_hidden:
        return "STRUCTURE_CONDITIONAL_PROMISING"
    if generator_hidden:
        return "REPRESENTATION_GAP"
    if peer_hidden:
        return "GENERATOR_STRUCTURE_MISALIGNED"
    return "NO_BORROWING_EVIDENCE"


# ---------------------------------------------------------------------------
# Strict IO and frozen-input validation
# ---------------------------------------------------------------------------


def strict_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=no_duplicates,
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError("nonfinite output")
        return converted
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(type(value).__name__)


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if int(manifest["schema_version"]) != 1:
        raise ValueError("unsupported manifest schema version")
    if manifest["task_id"] != TASK_ID:
        raise ValueError("manifest task identity mismatch")
    if manifest["source_task_id"] != SOURCE_TASK_ID:
        raise ValueError("manifest source task mismatch")
    if (
        manifest["source_verified_commit"]
        != "403884ae6bde46c7c3578ae01d77422ed03faf05"
    ):
        raise ValueError("manifest source commit mismatch")
    files = manifest["files"]
    expected_hashes = {
        "config.json": EXPECTED_CONFIG_SHA256,
        "task_results.json": EXPECTED_RECORDS_SHA256,
    }
    if set(files) != set(expected_hashes):
        raise ValueError("manifest file set mismatch")
    for name, expected in expected_hashes.items():
        if files[name]["sha256"] != expected:
            raise ValueError(f"manifest hash mismatch for {name}")
    if int(manifest["expected_records"]) != EXPECTED_RECORDS:
        raise ValueError(f"manifest must expect {EXPECTED_RECORDS} records")
    if manifest["expected_family_counts"] != EXPECTED_FAMILY_COUNTS:
        raise ValueError("manifest family counts mismatch")
    matrix = manifest["expected_matrix"]
    if (
        int(matrix["tasks_per_cell"]) != TASKS_PER_CELL
        or [int(v) for v in matrix["trajectory_lengths"]] != list(TRAJECTORY_LENGTHS)
        or [float(v) for v in matrix["mixing"]] != list(MIXING_VALUES)
        or [float(v) for v in matrix["gap_bonuses"]] != list(GAP_BONUSES)
        or int(matrix["seed"]) != FROZEN_SEED
    ):
        raise ValueError("manifest matrix mismatch")
    if manifest.get("frozen_after_creation") is not True:
        raise ValueError("manifest is not marked frozen")


def verify_record_matrix(records: list[dict[str, Any]], config: Mapping[str, Any]) -> None:
    if config["task_id"] != SOURCE_TASK_ID:
        raise ValueError("config task identity mismatch")
    if config["routes"] != PREDECESSOR_ROUTE_ORDER:
        raise ValueError("config route order mismatch")
    if int(config["seed"]) != FROZEN_SEED:
        raise ValueError("config seed mismatch")
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"record count must equal {EXPECTED_RECORDS}")
    family_counts = {family: 0 for family in FAMILIES}
    cells: set[tuple[str, int, float, float, int]] = set()
    for index, record in enumerate(records):
        if int(record["record_index"]) != index:
            raise ValueError("record index mismatch")
        if record["task_id"] != SOURCE_TASK_ID:
            raise ValueError("record task identity mismatch")
        family = str(record["environment_family"])
        if family not in family_counts:
            raise ValueError(f"unknown family: {family}")
        family_counts[family] += 1
        cell = (
            family,
            int(record["trajectory_length"]),
            float(record["mixing"]),
            float(record["gap_bonus"]),
            int(record["task_index"]),
        )
        if cell in cells:
            raise ValueError(f"duplicate matrix cell: {cell}")
        cells.add(cell)
        if int(record["seed_components"][0]) != FROZEN_SEED:
            raise ValueError("record seed mismatch")
    if family_counts != EXPECTED_FAMILY_COUNTS:
        raise ValueError("family counts mismatch")
    expected_cells = {
        (family, length, mixing, gap, task_index)
        for family in FAMILIES
        for length in TRAJECTORY_LENGTHS
        for mixing in MIXING_VALUES
        for gap in GAP_BONUSES
        for task_index in range(TASKS_PER_CELL)
    }
    if cells != expected_cells:
        raise ValueError("matrix cells do not equal the frozen grid")


def select_smoke_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if int(record["task_index"]) == SMOKE_TASK_INDEX
        and int(record["trajectory_length"]) in SMOKE_LENGTHS
    ]
    if len(selected) != 16:
        raise ValueError("the frozen smoke subset must contain exactly 16 records")
    return selected


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_stage0(
    records: list[dict[str, Any]],
    *,
    compare_to_sealed: bool,
    source_directory: Path | None,
) -> dict[str, Any]:
    """Replay all four predecessor routes exactly and reproduce the old result."""
    replay_mismatches = 0
    first_mismatch: str | None = None
    for record in records:
        replayed = replay_predecessor_routes(record["observable_inputs"])
        try:
            assert_exact_route_replay(
                replayed, record["kernel_generalization"], f"record[{record['record_index']}]"
            )
        except AssertionError as error:
            replay_mismatches += 1
            if first_mismatch is None:
                first_mismatch = str(error)
    if replay_mismatches:
        raise AssertionError(
            f"Stage 0 exact route replay failed for {replay_mismatches} records; "
            f"first: {first_mismatch}"
        )

    by_family = {
        family: [record for record in records if record["environment_family"] == family]
        for family in FAMILIES
    }
    screens = {
        family: predecessor_family_screen(selected)
        for family, selected in by_family.items()
    }
    route_metrics = {
        family: predecessor_route_metrics(selected)
        for family, selected in by_family.items()
    }
    hidden_pass = bool(screens["hidden_cluster"]["screen_pass"])
    current_pass = bool(screens["current_unstructured"]["screen_pass"])
    if not hidden_pass:
        classification = "NOT_SUPPORTED"
    elif current_pass:
        classification = "GENERAL_FEASIBLE"
    else:
        classification = "STRUCTURE_CONDITIONAL"

    evidence: dict[str, Any] = {
        "status": "PASS",
        "records_replayed": len(records),
        "exact_route_replay": "exact",
        "replayed_classification": classification,
        "decision_rule": {
            "hidden_cluster_pass": hidden_pass,
            "current_unstructured_pass": current_pass,
        },
        "family_screens": screens,
        "route_metrics": route_metrics,
    }

    if compare_to_sealed and source_directory is not None:
        sealed_summary = strict_load(source_directory / "summary.json")
        sealed_analysis = strict_load(source_directory / "analysis.json")
        assert_close_nested(screens, sealed_summary["family_screens"], "family_screens")
        assert_close_nested(
            route_metrics, sealed_summary["route_metrics"], "route_metrics"
        )
        if sealed_summary["classification"] != classification:
            raise AssertionError("sealed classification mismatch")
        if sealed_summary["classification"] != "NOT_SUPPORTED":
            raise AssertionError("sealed predecessor classification is not NOT_SUPPORTED")
        assert_close_nested(
            {
                "record_count": len(records),
                "classification": classification,
                "decision_rule": evidence["decision_rule"],
            },
            {
                "record_count": sealed_analysis["record_count"],
                "classification": sealed_analysis["classification"],
                "decision_rule": sealed_analysis["decision_rule"],
            },
            "analysis",
        )
        evidence["sealed_summary_comparison"] = "atol_1e-12_match"
        evidence["sealed_summary_path"] = str(source_directory / "summary.json")
    return evidence


def run_diagnostic_routes(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for record in records:
        observables = record["observable_inputs"]
        true_q = np.asarray(record["oracle_audit"]["true_q"], dtype=np.float64)
        q_route = oracle_q_nearest2_route(observables, true_q)
        if record["environment_family"] == "hidden_cluster":
            cluster_route = oracle_generator_cluster_route(
                observables, record["oracle_audit"]["generator"]["cluster_by_state"]
            )
        else:
            cluster_route = {"status": "not_applicable_family"}
        observable_route = observable_balanced_cluster_route(observables)
        outputs.append(
            {
                "record_index": int(record["record_index"]),
                "environment_family": record["environment_family"],
                "trajectory_length": int(record["trajectory_length"]),
                "mixing": float(record["mixing"]),
                "gap_bonus": float(record["gap_bonus"]),
                "task_index": int(record["task_index"]),
                "routes": {
                    "oracle_q_nearest2": q_route,
                    "oracle_generator_cluster": cluster_route,
                    "observable_balanced_cluster": observable_route,
                },
            }
        )
    return outputs


def analyze_diagnostics(
    records: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    stage0_ok: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_family_records = {
        family: [record for record in records if record["environment_family"] == family]
        for family in FAMILIES
    }
    by_family_outputs = {
        family: [
            output
            for record, output in zip(records, outputs, strict=True)
            if record["environment_family"] == family
        ]
        for family in FAMILIES
    }

    eligibility: dict[int, np.ndarray] = {}
    for record in records:
        replayed = replay_predecessor_routes(record["observable_inputs"])
        eligibility[int(record["record_index"])] = np.asarray(
            replayed["routes"]["leave_one_action_out_kernel"]["kernel"][
                "signature_eligible"
            ],
            dtype=bool,
        )

    family_screens: dict[str, Any] = {}
    for family in FAMILIES:
        family_records = by_family_records[family]
        family_outputs = by_family_outputs[family]
        family_eligibility = [
            eligibility[int(record["record_index"])] for record in family_records
        ]
        screens: dict[str, Any] = {}
        for route_name in ("oracle_q_nearest2", "observable_balanced_cluster"):
            estimates = [
                _estimate_array(output["routes"][route_name])
                for output in family_outputs
            ]
            screens[route_name] = route_family_screen(
                family_records, estimates, family_eligibility
            )
        if family == "hidden_cluster":
            estimates = [
                _estimate_array(output["routes"]["oracle_generator_cluster"])
                for output in family_outputs
            ]
            screens["oracle_generator_cluster"] = route_family_screen(
                family_records, estimates, family_eligibility
            )
        else:
            screens["oracle_generator_cluster"] = "not_applicable_family"
        family_screens[family] = screens

    hidden = family_screens["hidden_cluster"]
    gates = {
        "peer_headroom": bool(hidden["oracle_q_nearest2"]["screen_pass"]),
        "generator_structure_useful": bool(
            hidden["oracle_generator_cluster"]["screen_pass"]
        ),
        "observable_structure_useful": bool(
            hidden["observable_balanced_cluster"]["screen_pass"]
        ),
    }
    observable_current_pass = bool(
        family_screens["current_unstructured"]["observable_balanced_cluster"]["screen_pass"]
    )
    classification = classify(stage0_ok, gates, observable_current_pass)

    summary = {
        "status": "PASS" if stage0_ok else "INVALID_INPUT",
        "task_id": TASK_ID,
        "records": len(records),
        "classification": classification,
        "hidden_family_gates": gates,
        "observable_current_family_pass": observable_current_pass,
        "family_screens": family_screens,
    }

    hidden_records = by_family_records["hidden_cluster"]
    hidden_outputs = by_family_outputs["hidden_cluster"]
    hidden_observable_outputs = [
        output["routes"]["observable_balanced_cluster"] for output in hidden_outputs
    ]
    secondary = cluster_recovery_diagnostics(hidden_records, hidden_observable_outputs)
    recovery = oracle_benefit_recovery(
        hidden_records,
        [_estimate_array(output) for output in hidden_observable_outputs],
        [
            _estimate_array(output["routes"]["oracle_generator_cluster"])
            for output in hidden_outputs
        ],
    )

    abstentions: dict[str, dict[str, int]] = {}
    for route_name in DIAGNOSTIC_ROUTE_ORDER:
        reason_counts: dict[str, int] = {}
        for output in outputs:
            route = output["routes"][route_name]
            if route.get("status") == "not_applicable_family":
                reason_counts["not_applicable_family"] = reason_counts.get(
                    "not_applicable_family", 0
                ) + 1
                continue
            for row in route["reason"]:
                for reason in row:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        abstentions[route_name] = reason_counts

    analysis = {
        "status": summary["status"],
        "task_id": TASK_ID,
        "record_count": len(records),
        "stage0_ok": stage0_ok,
        "classification": classification,
        "decision_rule": {
            "peer_headroom": gates["peer_headroom"],
            "generator_structure_useful": gates["generator_structure_useful"],
            "observable_structure_useful": gates["observable_structure_useful"],
            "observable_current_pass": observable_current_pass,
            "ordered_rule": [
                "INVALID_INPUT",
                "GENERAL_PROMISING",
                "STRUCTURE_CONDITIONAL_PROMISING",
                "REPRESENTATION_GAP",
                "GENERATOR_STRUCTURE_MISALIGNED",
                "NO_BORROWING_EVIDENCE",
            ],
        },
        "secondary_hidden_diagnostics": secondary,
        "oracle_benefit_recovery": recovery,
        "abstention_counts": abstentions,
    }
    return summary, analysis


def environment_payload() -> dict[str, Any]:
    import scipy

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("stage0", "smoke", "full"), default="full"
    )
    parser.add_argument("--write-results", action="store_true")
    parser.add_argument("--invocation", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    result_dir = args.result_dir.resolve()
    config_path = input_dir / "config.json"
    records_path = input_dir / "task_results.json"
    manifest_path = input_dir / "source_manifest.json"
    before_hashes = {
        path.name: sha256_file(path)
        for path in (config_path, records_path, manifest_path)
    }
    if before_hashes["config.json"] != EXPECTED_CONFIG_SHA256:
        raise AssertionError("config hash differs from the frozen task value")
    if before_hashes["task_results.json"] != EXPECTED_RECORDS_SHA256:
        raise AssertionError("records hash differs from the frozen task value")
    config = strict_load(config_path)
    records = strict_load(records_path)
    manifest = strict_load(manifest_path)
    validate_manifest(manifest)
    if manifest["files"]["config.json"]["sha256"] != before_hashes["config.json"]:
        raise AssertionError("manifest config hash differs from the frozen copy")
    if (
        manifest["files"]["task_results.json"]["sha256"]
        != before_hashes["task_results.json"]
    ):
        raise AssertionError("manifest records hash differs from the frozen copy")
    verify_record_matrix(records, config)

    source_directory = Path(manifest["source_directory"])
    selected = records if args.mode != "smoke" else select_smoke_records(records)
    stage0_evidence = run_stage0(
        selected,
        compare_to_sealed=args.mode != "smoke",
        source_directory=source_directory if args.mode != "smoke" else None,
    )
    stage0_ok = stage0_evidence["status"] == "PASS"

    if args.mode == "stage0":
        print(
            f"PASS Stage 0 exact predecessor reproduction on {len(selected)} "
            f"records; classification={stage0_evidence['replayed_classification']}"
        )
        return

    outputs = run_diagnostic_routes(selected)
    summary, analysis = analyze_diagnostics(selected, outputs, stage0_ok)
    if args.mode == "smoke":
        summary = {**summary, "mode": "smoke", "records": len(selected)}
        analysis = {**analysis, "mode": "smoke"}

    after_hashes = {
        path.name: sha256_file(path)
        for path in (config_path, records_path, manifest_path)
    }
    if before_hashes != after_hashes:
        raise AssertionError("analyzer modified frozen input artifacts")
    analysis["frozen_input_hashes"] = before_hashes

    if args.write_results:
        if result_dir.exists() and any(result_dir.iterdir()):
            raise AssertionError(
                f"result directory must be absent or empty before generation: {result_dir}"
            )
        result_dir.mkdir(parents=True, exist_ok=True)
        manifest_out = {
            "manifest": manifest,
            "verified_hashes": before_hashes,
            "verified_by": "claude",
            "verification": "PASS",
        }
        write_json_atomic(result_dir / "source_manifest.json", manifest_out)
        write_json_atomic(result_dir / "baseline_reproduction.json", stage0_evidence)
        write_json_atomic(
            result_dir / "diagnostic_records.json",
            {
                "schema_version": 1,
                "task_id": TASK_ID,
                "route": "claude",
                "mode": args.mode,
                "record_count": len(selected),
                "diagnostic_route_order": DIAGNOSTIC_ROUTE_ORDER,
                "records": outputs,
            },
        )
        write_json_atomic(result_dir / "summary.json", summary)
        write_json_atomic(result_dir / "analysis.json", analysis)
        write_json_atomic(result_dir / "environment.json", environment_payload())
        commands = [
            "python verify_kernel_reuse_diagnostics.py",
            "ruff check analyze_kernel_reuse_diagnostics.py "
            "verify_kernel_reuse_diagnostics.py",
            (
                "python analyze_kernel_reuse_diagnostics.py --input-dir <frozen input> "
                "--result-dir <unused> --mode stage0"
            ),
            (
                "python analyze_kernel_reuse_diagnostics.py --input-dir <frozen input> "
                "--result-dir <smoke dir> --mode smoke --write-results"
            ),
            "python verify_kernel_state_generalization.py",
            args.invocation or "python analyze_kernel_reuse_diagnostics.py (formal)",
        ]
        (result_dir / "commands.log").write_text(
            "\n".join(commands) + "\n", encoding="utf-8"
        )
        with (result_dir / "checks.log").open("a", encoding="utf-8") as handle:
            handle.write(
                f"PASS stage0 exact replay on {len(selected)} records; "
                f"classification={summary['classification']}\n"
            )
        bundle_hashes = {
            path.name: sha256_file(path)
            for path in sorted(result_dir.iterdir())
            if path.is_file()
        }
        write_json_atomic(result_dir / "output_hashes.json", bundle_hashes)

    print(
        f"PASS FP-KERN-002 {args.mode} diagnostic on {len(selected)} records; "
        f"classification={summary['classification']}"
    )


if __name__ == "__main__":
    main()
