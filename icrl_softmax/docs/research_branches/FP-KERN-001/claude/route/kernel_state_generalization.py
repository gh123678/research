"""Pure data-derived cross-state kernel routes for FP-KERN-001.

This module is I/O-free. Every mask, distance, bandwidth, weight, count,
denominator, effective sample size, estimate, and abstention is derived only
from the declared observable inputs. Oracle quantities (true Q/V, transitions,
rewards, occupancies, cluster labels, exact returns) are rejected at the
input boundary and never influence route outputs.

Frozen formulas (task FP-KERN-001, version 1.0):

- q_sig(s,b) = mean[R + gamma V_hat(S') | S=s, A=b] on the signature stream;
- common support C_a(s,s') = {b != a : N_sig(s,b) > 0 and N_sig(s',b) > 0},
  requiring |C_a| >= 2 for any cross-state comparison;
- d_a(s,s')^2 = mean_{b in C_a} [(q_sig(s,b) - q_sig(s',b)) / (2B)]^2 with
  B = R_star / (1 - gamma);
- h_a = median of the finite positive eligible distances (sorted float64,
  numpy.median semantics); a missing or nonpositive median abstains;
- K_a(s,s') = exp(-d_a(s,s')^2 / (2 h_a^2)); the target state participates
  with weight one;
- Q_hat(s,a) = sum_s' K_a(s,s') Y_sum(s',a) / sum_s' K_a(s,s') N(s',a);
- ESS(s,a) = [sum_s' K N]^2 / sum_s' K^2 N.

The shared V_hat is a common nuisance estimate built from the independent
value stream; it is not a target-action signature coordinate.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


ROUTE_LOCAL = "local_unpooled"
ROUTE_POOL = "action_only_pool"
ROUTE_ANCHOR = "same_action_anchor_kernel"
ROUTE_PRIMARY = "leave_one_action_out_kernel"
ROUTES = (ROUTE_LOCAL, ROUTE_POOL, ROUTE_ANCHOR, ROUTE_PRIMARY)

REASON_SHAPE = "shape_or_dtype_invalid"
REASON_COUNT = "count_inconsistent"
REASON_VALUE = "value_estimate_nonfinite"
REASON_SUPPORT = "insufficient_common_actions"
REASON_BANDWIDTH = "bandwidth_unavailable"
REASON_TARGET = "target_source_unavailable"
REASON_DENOMINATOR = "kernel_denominator_invalid"
REASON_ESTIMATE = "estimate_nonfinite"
REASON_POLICY = "route_incomplete_for_policy"
REASON_ORDER = (
    REASON_SHAPE,
    REASON_COUNT,
    REASON_VALUE,
    REASON_SUPPORT,
    REASON_BANDWIDTH,
    REASON_TARGET,
    REASON_DENOMINATOR,
    REASON_ESTIMATE,
    REASON_POLICY,
)

OBSERVABLE_KEYS = (
    "n_states",
    "n_actions",
    "gamma",
    "reward_bound",
    "policy",
    "value_estimate",
    "signature_counts",
    "signature_sums",
    "target_counts",
    "target_sums",
)

PROHIBITED_INPUT_KEYS = frozenset(
    {
        "q_pi",
        "v_pi",
        "true_q",
        "true_v",
        "true_value",
        "oracle",
        "oracle_audit",
        "cluster",
        "clusters",
        "cluster_assignment",
        "cluster_labels",
        "prototype",
        "prototypes",
        "p_proto",
        "r_proto",
        "transition",
        "transitions",
        "reward_tensor",
        "occupancy",
        "mu_state",
        "mu_pair",
        "p_pi",
        "p_pair",
        "exact_return",
        "exact_value",
        "stationary_distribution",
    }
)


class KernelInputError(ValueError):
    """Ordered structural/numeric rejection at the pure input boundary."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def _as_float_matrix(name: str, value: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise KernelInputError(
            REASON_SHAPE, f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        if np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float64)
        else:
            raise KernelInputError(REASON_SHAPE, f"{name} must be a float matrix")
    return array.astype(np.float64, copy=False)


def _as_count_matrix(name: str, value: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise KernelInputError(
            REASON_SHAPE, f"{name} must have shape {shape}, got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise KernelInputError(REASON_SHAPE, f"{name} must have an integer dtype")
    return array.astype(np.int64, copy=False)


def validate_observable_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate the declared observable record inputs.

    Record-level failures raise KernelInputError with one of the first three
    ordered reasons; no route output may be produced from invalid inputs.
    """
    if not isinstance(inputs, Mapping):
        raise KernelInputError(REASON_SHAPE, "inputs must be a mapping")
    keys = set(str(key) for key in inputs)
    prohibited = sorted(keys & PROHIBITED_INPUT_KEYS)
    if prohibited:
        raise KernelInputError(
            REASON_SHAPE, f"prohibited oracle input keys: {prohibited}"
        )
    unknown = sorted(keys - set(OBSERVABLE_KEYS))
    if unknown:
        raise KernelInputError(REASON_SHAPE, f"unknown input keys: {unknown}")
    missing = [key for key in OBSERVABLE_KEYS if key not in keys]
    if missing:
        raise KernelInputError(REASON_SHAPE, f"missing input keys: {missing}")

    n_states = inputs["n_states"]
    n_actions = inputs["n_actions"]
    if (
        not isinstance(n_states, (int, np.integer))
        or not isinstance(n_actions, (int, np.integer))
        or int(n_states) < 2
        or int(n_actions) < 3
    ):
        raise KernelInputError(
            REASON_SHAPE, "n_states >= 2 and n_actions >= 3 are required"
        )
    n_states = int(n_states)
    n_actions = int(n_actions)
    gamma = float(inputs["gamma"])
    reward_bound = float(inputs["reward_bound"])
    if not math.isfinite(gamma) or not 0.0 < gamma < 1.0:
        raise KernelInputError(REASON_SHAPE, "gamma must lie in (0, 1)")
    if not math.isfinite(reward_bound) or reward_bound <= 0.0:
        raise KernelInputError(REASON_SHAPE, "reward_bound must be positive")

    shape = (n_states, n_actions)
    policy = _as_float_matrix("policy", inputs["policy"], shape)
    if not np.all(np.isfinite(policy)) or np.any(policy < 0.0):
        raise KernelInputError(REASON_SHAPE, "policy entries must be nonnegative")
    if not np.allclose(policy.sum(axis=1), 1.0, rtol=0.0, atol=1e-9):
        raise KernelInputError(REASON_SHAPE, "policy rows must sum to one")

    value = np.asarray(inputs["value_estimate"])
    if value.shape != (n_states,):
        raise KernelInputError(REASON_SHAPE, "value_estimate has wrong shape")

    signature_counts = _as_count_matrix(
        "signature_counts", inputs["signature_counts"], shape
    )
    target_counts = _as_count_matrix("target_counts", inputs["target_counts"], shape)
    signature_sums = _as_float_matrix(
        "signature_sums", inputs["signature_sums"], shape
    )
    target_sums = _as_float_matrix("target_sums", inputs["target_sums"], shape)

    for name, counts, sums in (
        ("signature", signature_counts, signature_sums),
        ("target", target_counts, target_sums),
    ):
        if np.any(counts < 0):
            raise KernelInputError(REASON_COUNT, f"{name} counts are negative")
        if np.any(~np.isfinite(sums)):
            raise KernelInputError(REASON_COUNT, f"{name} sums are nonfinite")
        if np.any(sums[counts == 0] != 0.0):
            raise KernelInputError(
                REASON_COUNT, f"{name} sums are nonzero at zero counts"
            )

    value = value.astype(np.float64, copy=False)
    if not np.all(np.isfinite(value)):
        raise KernelInputError(REASON_VALUE, "value_estimate is nonfinite")

    return {
        "n_states": n_states,
        "n_actions": n_actions,
        "gamma": gamma,
        "reward_bound": reward_bound,
        "value_scale": reward_bound / (1.0 - gamma),
        "policy": policy,
        "value_estimate": value,
        "signature_counts": signature_counts,
        "signature_sums": signature_sums,
        "target_counts": target_counts,
        "target_sums": target_sums,
    }


def signature_means(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    means = np.full(sums.shape, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means


def _common_counts(observed: np.ndarray, action: int) -> np.ndarray:
    others = observed[:, [b for b in range(observed.shape[1]) if b != action]]
    return (others.astype(np.int64)[:, None, :] & others.astype(np.int64)[None, :, :]).sum(
        axis=2
    )


def _distances(
    q_sig: np.ndarray, common: np.ndarray, action: int, value_scale: float
) -> np.ndarray:
    """Frozen normalized distances; NaN where the pair is ineligible."""
    n_states, n_actions = q_sig.shape
    distance = np.full((n_states, n_states), np.nan, dtype=np.float64)
    others = [b for b in range(n_actions) if b != action]
    for s in range(n_states):
        for sp in range(n_states):
            if s == sp:
                distance[s, sp] = 0.0
                continue
            if common[s, sp] < 2:
                continue
            shared = [
                b
                for b in others
                if math.isfinite(q_sig[s, b]) and math.isfinite(q_sig[sp, b])
            ]
            sq = float(
                np.mean(
                    [((q_sig[s, b] - q_sig[sp, b]) / (2.0 * value_scale)) ** 2 for b in shared]
                )
            )
            distance[s, sp] = math.sqrt(sq)
    return distance


def _median_bandwidth(distance: np.ndarray) -> float | None:
    n_states = distance.shape[0]
    eligible = [
        distance[s, sp]
        for s in range(n_states)
        for sp in range(s + 1, n_states)
        if math.isfinite(distance[s, sp]) and distance[s, sp] > 0.0
    ]
    if not eligible:
        return None
    median = float(np.median(np.asarray(sorted(eligible), dtype=np.float64)))
    if not math.isfinite(median) or median <= 0.0:
        return None
    return median


def _gaussian_weights(distance: np.ndarray, bandwidth: float | None) -> np.ndarray:
    n_states = distance.shape[0]
    weights = np.zeros((n_states, n_states), dtype=np.float64)
    np.fill_diagonal(weights, 1.0)
    if bandwidth is None:
        return weights
    factor = 2.0 * bandwidth * bandwidth
    for s in range(n_states):
        for sp in range(n_states):
            if s == sp or not math.isfinite(distance[s, sp]):
                continue
            weights[s, sp] = math.exp(-(distance[s, sp] ** 2) / factor)
    return weights


def _anchor_distances(
    q_sig: np.ndarray, counts: np.ndarray, action: int, value_scale: float
) -> np.ndarray:
    n_states = q_sig.shape[0]
    distance = np.full((n_states, n_states), np.nan, dtype=np.float64)
    np.fill_diagonal(distance, 0.0)
    available = counts[:, action] > 0
    for s in range(n_states):
        if not available[s]:
            distance[s, s] = np.nan
            continue
        for sp in range(n_states):
            if s == sp or not available[sp]:
                continue
            distance[s, sp] = abs(q_sig[s, action] - q_sig[sp, action]) / (
                2.0 * value_scale
            )
    return distance


def _kernel_pair(
    weights_row: np.ndarray,
    sums_col: np.ndarray,
    counts_col: np.ndarray,
) -> tuple[float | None, float, float | None, str | None]:
    denominator = float(np.sum(weights_row * counts_col))
    if not math.isfinite(denominator) or denominator <= 0.0:
        return None, denominator, None, REASON_DENOMINATOR
    numerator = float(np.sum(weights_row * sums_col))
    ess_denominator = float(np.sum(weights_row * weights_row * counts_col))
    if not math.isfinite(ess_denominator) or ess_denominator <= 0.0:
        return None, denominator, None, REASON_DENOMINATOR
    ess = denominator * denominator / ess_denominator
    estimate = numerator / denominator
    if not math.isfinite(estimate):
        return None, denominator, ess, REASON_ESTIMATE
    return estimate, denominator, ess, None


def _to_jsonable(matrix: np.ndarray) -> list[list[float | None]]:
    return [
        [None if not math.isfinite(value) else float(value) for value in row]
        for row in np.asarray(matrix, dtype=np.float64)
    ]


def evaluate_all_routes(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate all four frozen routes on one observable record.

    Raises KernelInputError for record-level ordered reasons 1-3; ordinary
    pair-level abstentions are reported per pair with ordered reasons 4-8.
    """
    data = validate_observable_inputs(inputs)
    n_states = data["n_states"]
    n_actions = data["n_actions"]
    value_scale = data["value_scale"]
    signature_counts = data["signature_counts"]
    signature_sums = data["signature_sums"]
    target_counts = data["target_counts"]
    target_sums = data["target_sums"]
    observed = signature_counts > 0
    q_sig = signature_means(signature_sums, signature_counts)
    target_totals = target_counts.sum(axis=0)

    routes: dict[str, dict[str, Any]] = {}
    primary_bandwidth: list[float | None] = []
    primary_common: dict[str, list[list[int]]] = {}
    primary_distances: dict[str, list[list[float | None]]] = {}
    primary_weights: dict[str, list[list[float]]] = {}
    anchor_bandwidth: list[float | None] = []
    anchor_distances: dict[str, list[list[float | None]]] = {}
    anchor_weights: dict[str, list[list[float]]] = {}

    for route in ROUTES:
        routes[route] = {
            "estimates": [[None] * n_actions for _ in range(n_states)],
            "reasons": [[None] * n_actions for _ in range(n_states)],
            "denominators": [[None] * n_actions for _ in range(n_states)],
            "ess": [[None] * n_actions for _ in range(n_states)],
        }

    for action in range(n_actions):
        common = _common_counts(observed, action)
        distance = _distances(q_sig, common, action, value_scale)
        bandwidth = _median_bandwidth(distance)
        weights = _gaussian_weights(distance, bandwidth)
        anchor_distance = _anchor_distances(q_sig, signature_counts, action, value_scale)
        anchor_band = _median_bandwidth(anchor_distance)
        anchor_w = _gaussian_weights(anchor_distance, anchor_band)
        primary_bandwidth.append(bandwidth)
        primary_common[str(action)] = common.astype(int).tolist()
        primary_distances[str(action)] = _to_jsonable(distance)
        primary_weights[str(action)] = _to_jsonable(weights)
        anchor_bandwidth.append(anchor_band)
        anchor_distances[str(action)] = _to_jsonable(anchor_distance)
        anchor_weights[str(action)] = _to_jsonable(anchor_w)

        sums_col = target_sums[:, action]
        counts_col = target_counts[:, action]

        for state in range(n_states):
            count = int(counts_col[state])

            local = routes[ROUTE_LOCAL]
            if count == 0:
                local["reasons"][state][action] = REASON_TARGET
            else:
                estimate = float(sums_col[state]) / count
                if not math.isfinite(estimate):
                    local["reasons"][state][action] = REASON_ESTIMATE
                else:
                    local["estimates"][state][action] = estimate
                    local["denominators"][state][action] = float(count)

            pool = routes[ROUTE_POOL]
            total = int(target_totals[action])
            if total == 0:
                pool["reasons"][state][action] = REASON_TARGET
            else:
                estimate = float(np.sum(sums_col)) / total
                if not math.isfinite(estimate):
                    pool["reasons"][state][action] = REASON_ESTIMATE
                else:
                    pool["estimates"][state][action] = estimate
                    pool["denominators"][state][action] = float(total)

            anchor = routes[ROUTE_ANCHOR]
            anchor_reason: str | None = None
            if signature_counts[state, action] == 0:
                anchor_reason = REASON_SUPPORT
            elif anchor_band is None:
                anchor_reason = REASON_BANDWIDTH
            elif count == 0:
                anchor_reason = REASON_TARGET
            if anchor_reason is not None:
                anchor["reasons"][state][action] = anchor_reason
            else:
                estimate, denominator, ess, failure = _kernel_pair(
                    anchor_w[state], sums_col, counts_col
                )
                if failure is not None:
                    anchor["reasons"][state][action] = failure
                    anchor["denominators"][state][action] = denominator
                else:
                    anchor["estimates"][state][action] = estimate
                    anchor["denominators"][state][action] = denominator
                    anchor["ess"][state][action] = ess

            primary = routes[ROUTE_PRIMARY]
            has_eligible_other = bool(
                np.any(
                    (common[state] >= 2) & (np.arange(n_states) != state)
                )
            )
            primary_reason: str | None = None
            if count == 0 and not has_eligible_other:
                primary_reason = REASON_SUPPORT
            elif bandwidth is None:
                primary_reason = REASON_BANDWIDTH
            elif int(target_totals[action]) == 0:
                primary_reason = REASON_TARGET
            if primary_reason is not None:
                primary["reasons"][state][action] = primary_reason
            else:
                estimate, denominator, ess, failure = _kernel_pair(
                    weights[state], sums_col, counts_col
                )
                if failure is not None:
                    primary["reasons"][state][action] = failure
                    primary["denominators"][state][action] = (
                        denominator if math.isfinite(denominator) else None
                    )
                else:
                    primary["estimates"][state][action] = estimate
                    primary["denominators"][state][action] = denominator
                    primary["ess"][state][action] = ess

    return {
        "value_scale": float(value_scale),
        "primary": {
            "bandwidth": primary_bandwidth,
            "common_counts": primary_common,
            "distances": primary_distances,
            "weights": primary_weights,
        },
        "anchor": {
            "bandwidth": anchor_bandwidth,
            "distances": anchor_distances,
            "weights": anchor_weights,
        },
        "routes": routes,
    }


def greedy_with_floor_policy(
    estimates: np.ndarray, base_policy: np.ndarray, pi_min: float
) -> tuple[np.ndarray, list[str]]:
    """Frozen greedy-with-floor diagnostic policy; descriptive only.

    States with any nonfinite route estimate retain the original policy row
    and are marked route_incomplete_for_policy.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    base_policy = np.asarray(base_policy, dtype=np.float64)
    n_states, n_actions = base_policy.shape
    updated = base_policy.copy()
    statuses: list[str] = []
    for state in range(n_states):
        row = estimates[state]
        if row.shape != (n_actions,) or not np.all(np.isfinite(row)):
            statuses.append(REASON_POLICY)
            continue
        greedy = int(np.argmax(row))
        new_row = np.full(n_actions, pi_min, dtype=np.float64)
        new_row[greedy] = 1.0 - (n_actions - 1) * pi_min
        updated[state] = new_row
        statuses.append("updated")
    return updated, statuses


def vfirst_value_from_aggregates(
    state_counts: np.ndarray,
    reward_sums: np.ndarray,
    transition_counts: np.ndarray,
    gamma: float,
    alpha: float,
    iterations: int,
    value_limit: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Replicate the unchanged exact V-first estimator from stream aggregates.

    The aggregate statistics must be derived from the value trajectory with
    the same bincount/add.at operations as iterative_state_evaluation; the
    update order then matches that estimator bitwise.
    """
    state_counts = np.asarray(state_counts, dtype=np.int64)
    reward_sums = np.asarray(reward_sums, dtype=np.float64)
    transition_counts = np.asarray(transition_counts, dtype=np.float64)
    n_states = state_counts.shape[0]
    values = np.zeros(n_states, dtype=np.float64)
    visited = state_counts > 0
    iterations_used = 0
    for iteration in range(iterations):
        residual_sums = (
            reward_sums
            + gamma * (transition_counts @ values)
            - state_counts * values
        )
        means = np.zeros(n_states, dtype=np.float64)
        means[visited] = residual_sums[visited] / state_counts[visited]
        update = alpha * means[visited]
        values[visited] += update
        values = np.clip(values, -value_limit, value_limit)
        iterations_used = iteration + 1
        if float(np.max(np.abs(update))) < 1e-13:
            break
    diagnostics = {
        "iterations_used": iterations_used,
        "missing_states": int(np.sum(~visited)),
    }
    return values, diagnostics


def aggregate_y_sums(
    reward_sums: np.ndarray,
    successor_counts: np.ndarray,
    value_estimate: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Y-sum per pair: sum over visits of R + gamma * V_hat(S')."""
    return np.asarray(reward_sums, dtype=np.float64) + gamma * (
        np.asarray(successor_counts, dtype=np.float64)
        @ np.asarray(value_estimate, dtype=np.float64)
    )
