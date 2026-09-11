"""Pure observable-data kernel routes for FP-KERN-001."""

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

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ROUTE_NAMES = (
    "local_unpooled",
    "action_only_pool",
    "same_action_anchor_kernel",
    "leave_one_action_out_kernel",
)

EXPECTED_INPUTS = {
    "signature_q",
    "signature_counts",
    "target_sums",
    "target_counts",
    "current_policy",
    "pi_min",
    "value_bound",
}

PROHIBITED_INPUT_FRAGMENTS = (
    "true",
    "oracle",
    "cluster",
    "occupancy",
    "stationary",
    "exact_return",
    "realized_error",
    "kernel_matrix",
)


def median_positive(values: Sequence[float]) -> float | None:
    """Return the sorted-float64 positive median, or None."""
    array = np.asarray(list(values), dtype=np.float64)
    selected = np.sort(array[np.isfinite(array) & (array > 0.0)])
    if selected.size == 0:
        return None
    middle = selected.size // 2
    if selected.size % 2:
        return float(selected[middle])
    return float((selected[middle - 1] + selected[middle]) / 2.0)


def effective_sample_size(weights: np.ndarray, counts: np.ndarray) -> float | None:
    """Observation-weighted ESS: (sum K*N)^2 / sum K^2*N."""
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


def _validate_observables(
    observables: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    keys = set(observables)
    extra = sorted(keys - EXPECTED_INPUTS)
    missing = sorted(EXPECTED_INPUTS - keys)
    if extra:
        if any(fragment in key.lower() for key in extra for fragment in PROHIBITED_INPUT_FRAGMENTS):
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
    shape = signature_q.shape
    if shape[0] < 2 or shape[1] < 3:
        raise ValueError("at least two states and three actions are required")
    for name, array in (
        ("signature_counts", signature_counts),
        ("target_sums", target_sums),
        ("target_counts", target_counts),
        ("current_policy", policy),
    ):
        if array.shape != shape:
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
    if not np.isfinite(pi_min) or not 0.0 < pi_min < 1.0 / shape[1]:
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


def _optional_matrix(values: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in values
    ]


def _optional_cube(values: np.ndarray) -> list[list[list[float | None]]]:
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
        "estimate": _optional_matrix(estimates),
        "reason": reasons,
        "effective_sample_size": _optional_matrix(ess),
        "diagnostic_policy": _diagnostic_policy(estimates, policy, pi_min),
    }


def _leave_one_action_out_distances(
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
                common = (signature_counts[state] > 0) & (
                    signature_counts[other] > 0
                )
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


def _kernel_route(
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
    reasons = [["insufficient_common_actions" for _ in range(n_actions)] for _ in range(n_states)]

    for state in range(n_states):
        for action in range(n_actions):
            finite_neighbors = [
                other
                for other in range(n_states)
                if other != state and np.isfinite(distances[action, state, other])
            ]
            self_available = bool(np.isfinite(distances[action, state, state]))
            if (
                primary
                and target_counts[state, action] == 0
                and not finite_neighbors
            ) or (not primary and not self_available):
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
        "distance_by_action": _optional_cube(distances),
        "weight_by_target": _optional_cube(weights),
        "denominator": _optional_matrix(denominators),
        "signature_eligible": eligible.tolist(),
    }
    if common_counts is not None:
        kernel_record["common_action_count_by_action"] = common_counts.tolist()
    result["kernel"] = kernel_record
    return result


def build_kernel_routes(observables: Mapping[str, Any]) -> dict[str, Any]:
    """Build all four frozen routes from observable aggregates only."""
    (
        signature_q,
        signature_counts,
        target_sums,
        target_counts,
        policy,
        pi_min,
        value_bound,
    ) = _validate_observables(observables)
    n_states, n_actions = signature_q.shape

    local = np.full((n_states, n_actions), np.nan, dtype=np.float64)
    local_ess = np.full_like(local, np.nan)
    local_reasons = [["target_source_unavailable" for _ in range(n_actions)] for _ in range(n_states)]
    visited = target_counts > 0
    local[visited] = target_sums[visited] / target_counts[visited]
    local_ess[visited] = target_counts[visited]
    for state, action in np.argwhere(visited):
        local_reasons[int(state)][int(action)] = "ok"

    pooled = np.full_like(local, np.nan)
    pooled_ess = np.full_like(local, np.nan)
    pooled_reasons = [["target_source_unavailable" for _ in range(n_actions)] for _ in range(n_states)]
    for action in range(n_actions):
        total_count = int(np.sum(target_counts[:, action]))
        if total_count <= 0:
            continue
        estimate = float(np.sum(target_sums[:, action]) / total_count)
        pooled[:, action] = estimate
        pooled_ess[:, action] = total_count
        for state in range(n_states):
            pooled_reasons[state][action] = "ok"

    loao_distances, common_counts, loao_bandwidths = _leave_one_action_out_distances(
        signature_q, signature_counts, value_bound
    )
    same_distances, same_bandwidths = _same_action_distances(
        signature_q, signature_counts, value_bound
    )
    routes = {
        "local_unpooled": _plain_route(
            local, local_reasons, local_ess, policy, pi_min
        ),
        "action_only_pool": _plain_route(
            pooled, pooled_reasons, pooled_ess, policy, pi_min
        ),
        "same_action_anchor_kernel": _kernel_route(
            same_distances,
            same_bandwidths,
            target_sums,
            target_counts,
            policy,
            pi_min,
            primary=False,
        ),
        "leave_one_action_out_kernel": _kernel_route(
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
        "route_order": list(ROUTE_NAMES),
        "shape": {"n_states": n_states, "n_actions": n_actions},
        "routes": routes,
    }
