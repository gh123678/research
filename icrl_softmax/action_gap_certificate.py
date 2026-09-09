"""Pure observable action-gap certificates and one frozen safe policy update.

The module contains no sampling, file I/O, plotting, or oracle computation.  It
uses only declared constants, observed counts/estimates, and already-emitted
FP-TU-001 bounds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


FAILURE_REASONS = (
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
_FAILURE_RANK = {reason: index for index, reason in enumerate(FAILURE_REASONS)}
_EXPECTED_ESTIMATES = {
    "vfirst_exact",
    "vfirst_softmax",
    "direct_exact",
    "direct_softmax",
}
_EXPECTED_STATE_BOUNDS = {"exact", "softmax"}
_EXPECTED_GLOBAL_BOUNDS = _EXPECTED_ESTIMATES
_FORBIDDEN_INPUT_KEY_TOKENS = (
    "oracle",
    "truth",
    "true_",
    "occupancy",
    "realized_error",
    "exact_return",
)


def strict_json_ready(value: Any) -> Any:
    """Convert recursively to strict-JSON values; nonfinite fields become null."""
    if isinstance(value, Mapping):
        return {str(key): strict_json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return strict_json_ready(value.tolist())
    if isinstance(value, (list, tuple)):
        return [strict_json_ready(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def canonical_reasons(reasons: Sequence[str]) -> list[str]:
    """Deduplicate reasons and impose the frozen FP-ADV-001 order."""
    return sorted(
        {str(reason) for reason in reasons},
        key=lambda reason: (_FAILURE_RANK.get(reason, len(FAILURE_REASONS)), reason),
    )


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _probability_row(name: str, values: Sequence[float]) -> np.ndarray:
    row = np.asarray(values, dtype=np.float64)
    if row.ndim != 1 or row.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(row)) or np.any(row < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    if not math.isclose(float(np.sum(row)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return row


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    """Return ``0.5 * ||left-right||_1`` for validated probability rows."""
    p = _probability_row("left", left)
    q = _probability_row("right", right)
    if p.shape != q.shape:
        raise ValueError("probability rows must have the same shape")
    return float(0.5 * np.sum(np.abs(p - q)))


def empirical_successor_row(
    group_successor_counts: Sequence[int], group_count: int
) -> list[float]:
    """Normalize the exact-match observed successor counts for one pair."""
    count = int(group_count)
    values = np.asarray(group_successor_counts, dtype=np.float64)
    if count <= 0:
        raise ValueError("group_count must be positive")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("group_successor_counts must be a nonempty vector")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("successor counts must be finite and nonnegative")
    if not math.isclose(float(np.sum(values)), float(count), abs_tol=1e-12):
        raise ValueError("successor counts must sum to group_count")
    return (values / count).tolist()


def _stable_on_group_mass(count: int, horizon: int, beta: float) -> float:
    if count <= 0 or horizon < count:
        raise ValueError("group count must lie in [1, trajectory_length]")
    sharpness = _finite("beta", beta)
    off_count = horizon - count
    if off_count == 0:
        return 1.0
    logit = sharpness + math.log(count) - math.log(off_count)
    if logit >= 0.0:
        tail = math.exp(-logit) if logit < 746.0 else 0.0
        return 1.0 / (1.0 + tail)
    head = math.exp(logit) if logit > -746.0 else 0.0
    return head / (1.0 + head)


def softmax_effective_successor_row(
    group_successor_counts: Sequence[int],
    total_successor_counts: Sequence[int],
    *,
    group_count: int,
    trajectory_length: int,
    beta: float,
) -> dict[str, Any]:
    """Return stable one-hot-softmax mass and effective successor row."""
    count = int(group_count)
    horizon = int(trajectory_length)
    group = np.asarray(group_successor_counts, dtype=np.float64)
    total = np.asarray(total_successor_counts, dtype=np.float64)
    if group.ndim != 1 or group.size == 0 or group.shape != total.shape:
        raise ValueError("successor count rows must be nonempty and shape-matched")
    if not np.all(np.isfinite(group)) or not np.all(np.isfinite(total)):
        raise ValueError("successor counts must be finite")
    if np.any(group < 0.0) or np.any(total < group):
        raise ValueError("successor counts must be nonnegative and nested")
    if horizon <= 0 or count <= 0 or count > horizon:
        raise ValueError("invalid group count or trajectory length")
    if not math.isclose(float(np.sum(group)), float(count), abs_tol=1e-12):
        raise ValueError("group successor counts do not sum to group_count")
    if not math.isclose(float(np.sum(total)), float(horizon), abs_tol=1e-12):
        raise ValueError("total successor counts do not sum to trajectory_length")
    kappa = _stable_on_group_mass(count, horizon, beta)
    on_row = group / count
    off_count = horizon - count
    if off_count == 0:
        effective = on_row
    else:
        off_row = (total - group) / off_count
        effective = kappa * on_row + (1.0 - kappa) * off_row
    if (
        not math.isfinite(kappa)
        or not 0.0 <= kappa <= 1.0
        or not np.all(np.isfinite(effective))
        or np.any(effective < -1e-15)
        or not math.isclose(float(np.sum(effective)), 1.0, abs_tol=1e-12)
    ):
        raise ValueError("invalid finite-softmax attention mass or effective row")
    effective = np.maximum(effective, 0.0)
    effective /= float(np.sum(effective))
    return {"kappa": float(kappa), "row": effective.tolist()}


def exact_local_uncertainty(
    radius_a: float,
    radius_b: float,
    gamma: float,
    value_bound: float,
    row_a: Sequence[float],
    row_b: Sequence[float],
) -> float:
    """Frozen exact local V-first pair uncertainty."""
    first = _finite("radius_a", radius_a)
    second = _finite("radius_b", radius_b)
    discount = _finite("gamma", gamma)
    bound = _finite("value_bound", value_bound)
    if first < 0.0 or second < 0.0 or bound < 0.0 or not 0.0 < discount < 1.0:
        raise ValueError("invalid exact local uncertainty input")
    return first + second + 2.0 * discount * bound * total_variation(row_a, row_b)


def softmax_local_uncertainty(
    *,
    radius_a: float,
    radius_b: float,
    kappa_a: float,
    kappa_b: float,
    reward_bound: float,
    gamma: float,
    value_bound: float,
    row_a: Sequence[float],
    row_b: Sequence[float],
) -> float:
    """Frozen finite-softmax local V-first pair uncertainty."""
    first = _finite("radius_a", radius_a)
    second = _finite("radius_b", radius_b)
    mass_a = _finite("kappa_a", kappa_a)
    mass_b = _finite("kappa_b", kappa_b)
    reward = _finite("reward_bound", reward_bound)
    discount = _finite("gamma", gamma)
    bound = _finite("value_bound", value_bound)
    if (
        first < 0.0
        or second < 0.0
        or reward < 0.0
        or bound < 0.0
        or not 0.0 < discount < 1.0
        or not 0.0 <= mass_a <= 1.0
        or not 0.0 <= mass_b <= 1.0
    ):
        raise ValueError("invalid softmax local uncertainty input")
    value_limit = reward / (1.0 - discount)
    component_a = mass_a * first + 2.0 * value_limit * (1.0 - mass_a)
    component_b = mass_b * second + 2.0 * value_limit * (1.0 - mass_b)
    return (
        component_a
        + component_b
        + 2.0 * discount * bound * total_variation(row_a, row_b)
    )


def global_uncertainty(q_bound: float) -> float:
    """Pairwise uncertainty induced by a complete-Q sup-norm bound."""
    bound = _finite("q_bound", q_bound)
    if bound < 0.0:
        raise ValueError("q_bound must be nonnegative")
    return 2.0 * bound


def _reject_forbidden_keys(name: str, value: Any) -> None:
    if not isinstance(value, Mapping):
        return
    for key, item in value.items():
        normalized = str(key).lower()
        if any(token in normalized for token in _FORBIDDEN_INPUT_KEY_TOKENS):
            raise ValueError(f"{name} contains prohibited key {key!r}")
        _reject_forbidden_keys(name, item)


def _pruned_bound(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    emitted = bool(value.get("finite_bound_emitted", False))
    total_raw = value.get("total_bound")
    total = None if total_raw is None else _finite(f"{name}.total_bound", total_raw)
    if total is not None and total < 0.0:
        raise ValueError(f"{name}.total_bound must be nonnegative")
    if emitted and total is None:
        raise ValueError(f"{name} cannot emit without a total bound")
    reasons_raw = value.get("failure_reasons", [])
    if not isinstance(reasons_raw, Sequence) or isinstance(reasons_raw, (str, bytes)):
        raise ValueError(f"{name}.failure_reasons must be a sequence")
    return {
        "finite_bound_emitted": emitted,
        "total_bound": total,
        "failure_reasons": [str(reason) for reason in reasons_raw],
    }


def _comparison(
    *,
    route_name: str,
    route_scope: str,
    matching: str,
    state: int,
    receiver: int,
    donor: int,
    n_actions: int,
    q_values: np.ndarray,
    policy: np.ndarray,
    counts: np.ndarray,
    exact_rows: list[list[float] | None],
    soft_rows: list[list[float] | None],
    kappas: list[float | None],
    recovery_radii: list[float | None],
    state_bound: dict[str, Any] | None,
    global_bound: dict[str, Any] | None,
    algorithm_mode: str,
    diverged: bool,
    reward_bound: float,
    gamma: float,
    pi_min: float,
    transfer_fraction: float,
) -> dict[str, Any]:
    receiver_group = state * n_actions + receiver
    donor_group = state * n_actions + donor
    reasons: list[str] = []
    if algorithm_mode != "fixed_policy_synchronous":
        reasons.append("algorithm_mode_mismatch")
    if diverged:
        reasons.append("divergence_guard_triggered")
    if route_scope == "local":
        if state_bound is None or not state_bound["finite_bound_emitted"]:
            reasons.append("state_certificate_not_emitted")
    elif global_bound is None or not global_bound["finite_bound_emitted"]:
        reasons.append("state_certificate_not_emitted")
    if counts[receiver_group] <= 0 or counts[donor_group] <= 0:
        reasons.append("candidate_pair_unvisited")

    q_gap = float(q_values[state, receiver] - q_values[state, donor])
    uncertainty: float | None = None
    components: dict[str, Any] = {}
    if route_scope == "local":
        radius_a = recovery_radii[receiver_group]
        radius_b = recovery_radii[donor_group]
        if radius_a is None or radius_b is None:
            reasons.append("recovery_radius_unavailable")
        if matching == "softmax":
            kappa_a = kappas[receiver_group]
            kappa_b = kappas[donor_group]
            if kappa_a is None or kappa_b is None:
                reasons.append("attention_mass_invalid")
            row_a = soft_rows[receiver_group]
            row_b = soft_rows[donor_group]
        else:
            kappa_a = kappa_b = None
            row_a = exact_rows[receiver_group]
            row_b = exact_rows[donor_group]
        if row_a is None or row_b is None:
            reasons.append("effective_transition_row_invalid")
        if (
            state_bound is not None
            and state_bound["finite_bound_emitted"]
            and state_bound["total_bound"] is not None
            and radius_a is not None
            and radius_b is not None
            and row_a is not None
            and row_b is not None
            and (matching == "exact" or (kappa_a is not None and kappa_b is not None))
        ):
            if matching == "exact":
                uncertainty = exact_local_uncertainty(
                    radius_a,
                    radius_b,
                    gamma,
                    float(state_bound["total_bound"]),
                    row_a,
                    row_b,
                )
                components = {
                    "radius_receiver": radius_a,
                    "radius_donor": radius_b,
                    "value_total_bound": state_bound["total_bound"],
                    "transition_tv": total_variation(row_a, row_b),
                }
            else:
                uncertainty = softmax_local_uncertainty(
                    radius_a=radius_a,
                    radius_b=radius_b,
                    kappa_a=float(kappa_a),
                    kappa_b=float(kappa_b),
                    reward_bound=reward_bound,
                    gamma=gamma,
                    value_bound=float(state_bound["total_bound"]),
                    row_a=row_a,
                    row_b=row_b,
                )
                value_limit = reward_bound / (1.0 - gamma)
                components = {
                    "radius_receiver": radius_a,
                    "radius_donor": radius_b,
                    "kappa_receiver": kappa_a,
                    "kappa_donor": kappa_b,
                    "contamination_receiver": 2.0 * value_limit * (1.0 - float(kappa_a)),
                    "contamination_donor": 2.0 * value_limit * (1.0 - float(kappa_b)),
                    "value_total_bound": state_bound["total_bound"],
                    "transition_tv": total_variation(row_a, row_b),
                }
    elif global_bound is not None and global_bound["finite_bound_emitted"]:
        total = global_bound["total_bound"]
        if total is not None:
            uncertainty = global_uncertainty(float(total))
            components = {"q_total_bound": total}

    lcb = None if uncertainty is None else q_gap - uncertainty
    if uncertainty is not None and (
        not math.isfinite(uncertainty) or not math.isfinite(q_gap) or not math.isfinite(lcb)
    ):
        reasons.append("numerical_nonfinite")
        lcb = None
    elif lcb is not None and lcb <= 0.0:
        reasons.append("gap_lcb_nonpositive")
    donor_mass = float(policy[state, donor])
    transferable = transfer_fraction * (donor_mass - pi_min)
    if donor_mass <= pi_min or transferable <= 0.0:
        reasons.append("no_transferable_mass")
        transferable = 0.0
    ordered = canonical_reasons(reasons)
    eligible = not ordered and lcb is not None and lcb > 0.0 and transferable > 0.0
    return {
        "route": route_name,
        "state": state,
        "receiver": receiver,
        "donor": donor,
        "receiver_pair_count": int(counts[receiver_group]),
        "donor_pair_count": int(counts[donor_group]),
        "estimated_gap": q_gap,
        "uncertainty": uncertainty,
        "gap_lcb": lcb,
        "components": components,
        "donor_policy_mass": donor_mass,
        "transferable_mass": float(transferable),
        "eligible": bool(eligible),
        "status": "eligible" if eligible else "abstained",
        "failure_reasons": ordered,
    }


def _build_route(
    *,
    route_name: str,
    estimate_key: str,
    route_scope: str,
    matching: str,
    policy: np.ndarray,
    q_values: np.ndarray,
    counts: np.ndarray,
    exact_rows: list[list[float] | None],
    soft_rows: list[list[float] | None],
    kappas: list[float | None],
    recovery_radii: list[float | None],
    state_bound: dict[str, Any] | None,
    global_bound: dict[str, Any] | None,
    algorithm_mode: str,
    diverged: bool,
    reward_bound: float,
    gamma: float,
    pi_min: float,
    transfer_fraction: float,
) -> dict[str, Any]:
    n_states, n_actions = policy.shape
    updated = policy.copy()
    receivers: list[int] = []
    state_results: list[dict[str, Any]] = []
    bellman_lcbs = np.zeros(n_states, dtype=np.float64)
    transferred_total = 0.0
    eligible_total = 0
    route_reasons: list[str] = []
    for state in range(n_states):
        receiver = int(np.argmax(q_values[state]))
        receivers.append(receiver)
        comparisons: list[dict[str, Any]] = []
        for donor in range(n_actions):
            if donor == receiver:
                continue
            item = _comparison(
                route_name=route_name,
                route_scope=route_scope,
                matching=matching,
                state=state,
                receiver=receiver,
                donor=donor,
                n_actions=n_actions,
                q_values=q_values,
                policy=policy,
                counts=counts,
                exact_rows=exact_rows,
                soft_rows=soft_rows,
                kappas=kappas,
                recovery_radii=recovery_radii,
                state_bound=state_bound,
                global_bound=global_bound,
                algorithm_mode=algorithm_mode,
                diverged=diverged,
                reward_bound=reward_bound,
                gamma=gamma,
                pi_min=pi_min,
                transfer_fraction=transfer_fraction,
            )
            comparisons.append(item)
            route_reasons.extend(item["failure_reasons"])
            if item["eligible"]:
                transfer = float(item["transferable_mass"])
                updated[state, donor] -= transfer
                updated[state, receiver] += transfer
                bellman_lcbs[state] += transfer * float(item["gap_lcb"])
                transferred_total += transfer
                eligible_total += 1
        state_eligible = sum(bool(item["eligible"]) for item in comparisons)
        if state_eligible:
            # Make the conservation identity explicit only for an actual update.
            nonreceiver_sum = float(np.sum(np.delete(updated[state], receiver)))
            updated[state, receiver] = 1.0 - nonreceiver_sum
        else:
            # The frozen contract requires exact identity when no donor is eligible.
            updated[state] = policy[state]
        state_reasons = canonical_reasons(
            [reason for item in comparisons for reason in item["failure_reasons"]]
        )
        if not comparisons:
            state_reasons = ["no_transferable_mass"]
            route_reasons.append("no_transferable_mass")
        state_results.append(
            {
                "state": state,
                "receiver": receiver,
                "comparisons": comparisons,
                "eligible_donor_count": state_eligible,
                "transferred_mass": float(
                    sum(float(item["transferable_mass"]) for item in comparisons if item["eligible"])
                ),
                "bellman_lcb": float(bellman_lcbs[state]),
                "status": "safe_update_emitted" if state_eligible else "abstained",
                "failure_reasons": state_reasons,
            }
        )

    row_errors = np.abs(np.sum(updated, axis=1) - 1.0)
    if (
        not np.all(np.isfinite(updated))
        or np.any(updated < -1e-12)
        or np.any(updated < pi_min - 1e-12)
        or float(np.max(row_errors)) > 1e-12
        or np.any(bellman_lcbs < -1e-15)
    ):
        raise AssertionError("constructed policy violates the frozen update contract")
    return {
        "route": route_name,
        "estimate_source": estimate_key,
        "scope": route_scope,
        "matching": matching,
        "receiver_by_state": receivers,
        "state_results": state_results,
        "policy_plus": updated.tolist(),
        "eligible_donor_count": eligible_total,
        "updated_state_count": sum(
            result["eligible_donor_count"] > 0 for result in state_results
        ),
        "total_transferred_mass": float(transferred_total),
        "bellman_lcb_by_state": bellman_lcbs.tolist(),
        "minimum_policy_mass": float(np.min(updated)),
        "maximum_row_sum_error": float(np.max(row_errors)),
        "status": "safe_update_emitted" if eligible_total else "abstained",
        "failure_reasons": canonical_reasons(route_reasons),
        "inherited_bound_failure_reasons": (
            list(state_bound["failure_reasons"])
            if route_scope == "local" and state_bound is not None
            else list(global_bound["failure_reasons"])
            if global_bound is not None
            else []
        ),
    }


def _dominance(
    local: dict[str, Any], global_route: dict[str, Any]
) -> dict[str, Any]:
    comparisons = 0
    differences: list[float] = []
    for local_state, global_state in zip(
        local["state_results"], global_route["state_results"], strict=True
    ):
        if local_state["receiver"] != global_state["receiver"]:
            raise AssertionError("matching local and global receivers differ")
        global_by_donor = {
            item["donor"]: item for item in global_state["comparisons"]
        }
        for local_item in local_state["comparisons"]:
            global_item = global_by_donor[local_item["donor"]]
            if local_item["uncertainty"] is None or global_item["uncertainty"] is None:
                continue
            comparisons += 1
            differences.append(
                float(local_item["uncertainty"])
                - float(global_item["uncertainty"])
            )
    return {
        "checked_comparisons": comparisons,
        "all_local_penalties_nonincreasing": all(value <= 1e-12 for value in differences),
        "maximum_local_minus_global": max(differences) if differences else None,
        "mean_global_minus_local": (
            None if not differences else -float(np.mean(differences))
        ),
    }


def build_action_gap_certificate(
    *,
    policy: Sequence[Sequence[float]],
    pair_counts: Sequence[int],
    pair_successor_counts: Sequence[Sequence[int]],
    q_estimates: Mapping[str, Sequence[Sequence[float]]],
    state_value_bounds: Mapping[str, Mapping[str, Any]],
    global_q_bounds: Mapping[str, Mapping[str, Any]],
    recovery_radii: Sequence[float | None],
    trajectory_length: int,
    reward_bound: float,
    gamma: float,
    beta: float,
    pi_min: float,
    transfer_fraction: float = 0.5,
    algorithm_mode: str = "fixed_policy_synchronous",
    divergence_guards: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Build six observable routes and their single deterministic policy update."""
    for name, value in (
        ("q_estimates", q_estimates),
        ("state_value_bounds", state_value_bounds),
        ("global_q_bounds", global_q_bounds),
    ):
        _reject_forbidden_keys(name, value)
    if set(q_estimates) != _EXPECTED_ESTIMATES:
        raise ValueError("q_estimates has an unexpected route set")
    if set(state_value_bounds) != _EXPECTED_STATE_BOUNDS:
        raise ValueError("state_value_bounds must contain exact and softmax")
    if set(global_q_bounds) != _EXPECTED_GLOBAL_BOUNDS:
        raise ValueError("global_q_bounds has an unexpected route set")

    current_policy = np.asarray(policy, dtype=np.float64)
    if current_policy.ndim != 2 or min(current_policy.shape) <= 0:
        raise ValueError("policy must be a nonempty matrix")
    n_states, n_actions = current_policy.shape
    if n_actions < 2 or not np.all(np.isfinite(current_policy)):
        raise ValueError("policy must have at least two actions and finite entries")
    floor = _finite("pi_min", pi_min)
    if floor <= 0.0 or floor >= 1.0 / n_actions:
        raise ValueError("pi_min must lie in (0,1/n_actions)")
    if np.any(current_policy < floor - 1e-12):
        raise ValueError("policy violates pi_min")
    if not np.allclose(np.sum(current_policy, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("policy rows must sum to one")

    horizon = int(trajectory_length)
    if horizon <= 0 or float(horizon) != float(trajectory_length):
        raise ValueError("trajectory_length must be a positive integer")
    discount = _finite("gamma", gamma)
    reward = _finite("reward_bound", reward_bound)
    sharpness = _finite("beta", beta)
    fraction = _finite("transfer_fraction", transfer_fraction)
    if not 0.0 < discount < 1.0 or reward < 0.0:
        raise ValueError("invalid gamma or reward_bound")
    if fraction != 0.5:
        raise ValueError("FP-ADV-001 freezes transfer_fraction at 0.5")

    counts_raw = np.asarray(pair_counts)
    if counts_raw.ndim != 1 or counts_raw.size != n_states * n_actions:
        raise ValueError("pair_counts shape does not match the policy")
    if not np.issubdtype(counts_raw.dtype, np.integer):
        raise ValueError("pair_counts must preserve an integer source dtype")
    counts = counts_raw.astype(np.int64)
    if np.any(counts < 0) or int(np.sum(counts)) != horizon:
        raise ValueError("pair_counts must be nonnegative and sum to trajectory_length")
    successors_raw = np.asarray(pair_successor_counts)
    if successors_raw.shape != (n_states * n_actions, n_states):
        raise ValueError("pair_successor_counts has the wrong shape")
    if not np.issubdtype(successors_raw.dtype, np.integer):
        raise ValueError(
            "pair_successor_counts must preserve an integer source dtype"
        )
    successors = successors_raw.astype(np.int64)
    if np.any(successors < 0) or not np.array_equal(np.sum(successors, axis=1), counts):
        raise ValueError("successor rows must be nonnegative and match pair_counts")
    total_successors = np.sum(successors, axis=0)
    if int(np.sum(total_successors)) != horizon:
        raise ValueError("successor counts do not sum to trajectory_length")

    estimates: dict[str, np.ndarray] = {}
    for name, values in q_estimates.items():
        array = np.asarray(values, dtype=np.float64)
        if array.shape != current_policy.shape or not np.all(np.isfinite(array)):
            raise ValueError(f"q_estimates[{name!r}] must be finite and policy-shaped")
        estimates[name] = array
    state_bounds = {
        name: _pruned_bound(f"state_value_bounds.{name}", value)
        for name, value in state_value_bounds.items()
    }
    global_bounds = {
        name: _pruned_bound(f"global_q_bounds.{name}", value)
        for name, value in global_q_bounds.items()
    }
    radii_values = list(recovery_radii)
    if len(radii_values) != counts.size:
        raise ValueError("recovery_radii length must match pair_counts")
    radii: list[float | None] = []
    for index, value in enumerate(radii_values):
        if value is None:
            radii.append(None)
        else:
            radius = _finite(f"recovery_radii[{index}]", value)
            if radius < 0.0:
                raise ValueError("recovery radii must be nonnegative")
            radii.append(radius)

    exact_rows: list[list[float] | None] = []
    soft_rows: list[list[float] | None] = []
    kappas: list[float | None] = []
    for index, count in enumerate(counts):
        if count <= 0:
            exact_rows.append(None)
            soft_rows.append(None)
            kappas.append(None)
            continue
        exact_rows.append(empirical_successor_row(successors[index], int(count)))
        attention = softmax_effective_successor_row(
            successors[index],
            total_successors,
            group_count=int(count),
            trajectory_length=horizon,
            beta=sharpness,
        )
        soft_rows.append(attention["row"])
        kappas.append(float(attention["kappa"]))

    divergence = {"direct_exact": False, "direct_softmax": False}
    if divergence_guards is not None:
        if set(divergence_guards) != set(divergence):
            raise ValueError("divergence_guards has an unexpected route set")
        divergence = {name: bool(value) for name, value in divergence_guards.items()}

    route_specs = (
        ("vfirst_local_exact", "vfirst_exact", "local", "exact"),
        ("vfirst_local_softmax", "vfirst_softmax", "local", "softmax"),
        ("vfirst_global_exact", "vfirst_exact", "global", "exact"),
        ("vfirst_global_softmax", "vfirst_softmax", "global", "softmax"),
        ("direct_global_exact", "direct_exact", "global", "exact"),
        ("direct_global_softmax", "direct_softmax", "global", "softmax"),
    )
    routes: dict[str, dict[str, Any]] = {}
    for route_name, estimate_key, scope, matching in route_specs:
        routes[route_name] = _build_route(
            route_name=route_name,
            estimate_key=estimate_key,
            route_scope=scope,
            matching=matching,
            policy=current_policy,
            q_values=estimates[estimate_key],
            counts=counts,
            exact_rows=exact_rows,
            soft_rows=soft_rows,
            kappas=kappas,
            recovery_radii=radii,
            state_bound=state_bounds[matching] if scope == "local" else None,
            global_bound=global_bounds[estimate_key] if scope == "global" else None,
            algorithm_mode=str(algorithm_mode),
            diverged=bool(divergence.get(estimate_key, False)),
            reward_bound=reward,
            gamma=discount,
            pi_min=floor,
            transfer_fraction=fraction,
        )

    output = {
        "task_id": "FP-ADV-001",
        "probability_statement": (
            "P(EmitUpdate and (any used ordering is false or any value decreases)) <= delta"
        ),
        "certificate_inputs": {
            "observed": {
                "policy": current_policy,
                "pair_counts": counts,
                "pair_successor_counts": successors,
                "q_estimates": estimates,
                "recovery_radii": radii,
                "exact_successor_rows": exact_rows,
                "softmax_effective_successor_rows": soft_rows,
                "softmax_kappa_by_group": kappas,
            },
            "declared": {
                "trajectory_length": horizon,
                "reward_bound": reward,
                "value_bound": reward / (1.0 - discount),
                "gamma": discount,
                "beta": sharpness,
                "pi_min": floor,
                "transfer_fraction": fraction,
            },
            "inherited": {
                "state_value_bounds": state_bounds,
                "global_q_bounds": global_bounds,
            },
            "algorithm": {
                "mode": str(algorithm_mode),
                "divergence_guards": divergence,
            },
        },
        "formulas": {
            "exact_local": "r_a+r_b+2*gamma*E_V*TV(Pbar_a,Pbar_b)",
            "softmax_component": "kappa_g*r_g+2*B*(1-kappa_g)",
            "softmax_local": "C_a+C_b+2*gamma*E_V_beta*TV(Pbar_a_beta,Pbar_b_beta)",
            "global": "2*E_Q",
            "transfer": "0.5*(pi_donor-pi_min)",
        },
        "routes": routes,
        "local_global_dominance": {
            "exact": _dominance(
                routes["vfirst_local_exact"], routes["vfirst_global_exact"]
            ),
            "softmax": _dominance(
                routes["vfirst_local_softmax"], routes["vfirst_global_softmax"]
            ),
        },
    }
    ready = strict_json_ready(output)
    # The output must already be finite except for intentionally unavailable nulls.
    return ready
