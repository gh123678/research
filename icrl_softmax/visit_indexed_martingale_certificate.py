"""Pure visit-indexed certificates for frozen fixed-policy evaluation.

The module consumes only observed visit counts, declared bounds, public
hyperparameters, and algorithm metadata.  It deliberately has no sampling,
file-writing, plotting, true-model, occupancy, or spectral dependency.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from fixed_policy_finite_sample_certificate import strict_json_ready as strict_json_ready


STATUS_SELECTIVE_HIGH_PROBABILITY = "selective_high_probability_certified"
STATUS_NOT_CERTIFIED = "not_certified"

_FAILURE_ORDER = (
    "algorithm_mode_mismatch",
    "divergence_guard_triggered",
    "state_support_missing",
    "pair_support_missing",
    "state_kernel_margin_nonpositive",
    "pair_kernel_margin_nonpositive",
    "numerical_nonfinite",
)
_FAILURE_RANK = {reason: index for index, reason in enumerate(_FAILURE_ORDER)}


def _finite_number(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_integer(name: str, value: int) -> int:
    integer = int(value)
    if isinstance(value, bool) or integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _nonnegative_integer(name: str, value: int) -> int:
    integer = int(value)
    if isinstance(value, bool) or integer != value or integer < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return integer


def _validated_counts(
    name: str,
    counts: Sequence[int],
    horizon: int,
) -> list[int]:
    values = [_nonnegative_integer(f"{name}[{index}]", value) for index, value in enumerate(counts)]
    if not values:
        raise ValueError(f"{name} must contain at least one group")
    if sum(values) != horizon:
        raise ValueError(f"{name} must sum to trajectory_length")
    return values


def _canonical_reasons(reasons: Sequence[str]) -> list[str]:
    return sorted(
        set(reasons),
        key=lambda reason: (_FAILURE_RANK.get(reason, len(_FAILURE_ORDER)), reason),
    )


def simultaneous_hoeffding_radius(
    count: int,
    *,
    reward_bound: float,
    gamma: float,
    n_groups: int,
    horizon: int,
    delta: float,
) -> float:
    """Return the preallocated two-sided radius for one group and visit count.

    Every residual has conditional support width at most ``2B``, where
    ``B = reward_bound / (1-gamma)``.  A two-sided tail receives
    ``delta / (n_groups*horizon)`` before observing any counts.
    """
    visits = _positive_integer("count", count)
    length = _positive_integer("horizon", horizon)
    groups = _positive_integer("n_groups", n_groups)
    if visits > length:
        raise ValueError("count cannot exceed horizon")
    reward = _finite_number("reward_bound", reward_bound)
    discount = _finite_number("gamma", gamma)
    confidence = _finite_number("delta", delta)
    if reward < 0.0:
        raise ValueError("reward_bound must be nonnegative")
    if not 0.0 < discount < 1.0:
        raise ValueError("gamma must lie in (0,1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("delta must lie in (0,1)")
    value_bound = reward / (1.0 - discount)
    log_term = math.log(2.0 * groups * length / confidence)
    radius = value_bound * math.sqrt(2.0 * log_term / visits)
    if not math.isfinite(radius):
        raise ValueError("Hoeffding radius is nonfinite")
    return radius


def empirical_one_hot_diagonals(
    counts: Sequence[int],
    *,
    horizon: int,
    beta: float,
) -> list[float | None]:
    """Return observed one-hot softmax self-mass for every empirical group."""
    length = _positive_integer("horizon", horizon)
    count_values = _validated_counts("counts", counts, length)
    sharpness = _finite_number("beta", beta)
    diagonals: list[float | None] = []
    for count in count_values:
        if count == 0:
            diagonals.append(None)
            continue
        if count == length:
            diagonals.append(1.0)
            continue
        logit = sharpness + math.log(count) - math.log(length - count)
        if logit >= 0.0:
            diagonal = 1.0 / (1.0 + math.exp(-logit))
        else:
            exponential = math.exp(logit)
            diagonal = exponential / (1.0 + exponential)
        diagonals.append(diagonal)
    return diagonals


def _family_summary(
    counts: Sequence[int],
    *,
    reward_bound: float,
    gamma: float,
    n_groups: int,
    horizon: int,
    delta: float,
) -> dict[str, Any]:
    radii = [
        None
        if count == 0
        else simultaneous_hoeffding_radius(
            count,
            reward_bound=reward_bound,
            gamma=gamma,
            n_groups=n_groups,
            horizon=horizon,
            delta=delta,
        )
        for count in counts
    ]
    visited_radii = [float(radius) for radius in radii if radius is not None]
    full_support = all(count > 0 for count in counts)
    return {
        "counts": list(counts),
        "missing_groups": int(sum(count == 0 for count in counts)),
        "full_support": full_support,
        "min_count": int(min(counts)),
        "min_visited_count": int(min(count for count in counts if count > 0)),
        "radius_by_group": radii,
        "max_radius": max(visited_radii) if full_support else None,
        "max_visited_radius": max(visited_radii),
    }


def _minimum_diagonal(diagonals: Sequence[float | None]) -> float | None:
    if any(value is None for value in diagonals):
        return None
    values = [float(value) for value in diagonals if value is not None]
    return min(values)


def _uniform_terms(
    *,
    rho: float,
    iterations: int,
    initial_error: float,
    residual_radius: float,
    margin: float,
) -> tuple[float, float, float] | None:
    optimization = rho**iterations * initial_error
    one_minus_power = 0.0 if iterations == 0 else -math.expm1(iterations * math.log(rho))
    statistical = one_minus_power * residual_radius / (2.0 * margin)
    total = optimization + statistical
    if not all(math.isfinite(value) for value in (optimization, statistical, total)):
        return None
    return optimization, statistical, total


def _uniform_route_bound(
    *,
    route: str,
    matching: str,
    support_missing_reason: str,
    margin_failure_reason: str,
    full_support: bool,
    residual_radius: float | None,
    diagonal: float | None,
    beta: float,
    gamma: float,
    alpha: float,
    iterations: int,
    value_bound: float,
    base_reasons: Sequence[str],
    diverged: bool,
) -> dict[str, Any]:
    reasons = list(base_reasons)
    if diverged:
        reasons.append("divergence_guard_triggered")
    if not full_support or residual_radius is None:
        reasons.append(support_missing_reason)

    if matching == "exact":
        route_diagonal = 1.0 if full_support else None
        margin = (1.0 - gamma) / 2.0 if full_support else None
        route_beta: float | None = None
    elif matching == "softmax":
        route_diagonal = diagonal
        route_beta = beta
        margin = (
            None
            if route_diagonal is None
            else route_diagonal - (1.0 + gamma) / 2.0
        )
        if margin is not None and margin <= 0.0:
            reasons.append(margin_failure_reason)
    else:
        raise ValueError("matching must be exact or softmax")

    rho: float | None = None
    optimization: float | None = None
    statistical: float | None = None
    total: float | None = None
    ordered = _canonical_reasons(reasons)
    if not ordered and margin is not None and residual_radius is not None:
        rho = 1.0 - 2.0 * alpha * margin
        if not 0.0 < rho <= 1.0 or not math.isfinite(rho):
            reasons.append("numerical_nonfinite")
        else:
            terms = _uniform_terms(
                rho=rho,
                iterations=iterations,
                initial_error=value_bound,
                residual_radius=residual_radius,
                margin=margin,
            )
            if terms is None:
                reasons.append("numerical_nonfinite")
            else:
                optimization, statistical, total = terms

    ordered = _canonical_reasons(reasons)
    if ordered:
        rho = optimization = statistical = total = None
    emitted = not ordered
    return {
        "route": route,
        "matching": matching,
        "beta": route_beta,
        "kernel_diagonal_lower_bound": route_diagonal,
        "margin": margin,
        "rho": rho,
        "iterations": iterations,
        "initial_error": value_bound,
        "residual_radius": residual_radius,
        "optimization_term": optimization,
        "statistical_term": statistical,
        "total_bound": total,
        "finite_bound_emitted": bool(emitted and total is not None),
        "improves_over_zero_initialization": bool(
            emitted and total is not None and total < value_bound
        ),
        "selective_high_probability_certified": emitted,
        "status": (
            STATUS_SELECTIVE_HIGH_PROBABILITY if emitted else STATUS_NOT_CERTIFIED
        ),
        "failure_reasons": ordered,
    }


def _vfirst_bound(
    *,
    route: str,
    matching: str,
    state_bound: dict[str, Any],
    pair_full_support: bool,
    recovery_radius: float | None,
    pair_diagonal: float | None,
    beta: float,
    gamma: float,
    value_bound: float,
    base_reasons: Sequence[str],
) -> dict[str, Any]:
    reasons = list(base_reasons)
    reasons.extend(str(reason) for reason in state_bound["failure_reasons"])
    if not pair_full_support or recovery_radius is None:
        reasons.append("pair_support_missing")

    if matching == "exact":
        diagonal = 1.0 if pair_full_support else None
        route_beta: float | None = None
        leakage = 0.0 if pair_full_support else None
    elif matching == "softmax":
        diagonal = pair_diagonal
        route_beta = beta
        leakage = (
            None if diagonal is None else 2.0 * value_bound * (1.0 - diagonal)
        )
    else:
        raise ValueError("matching must be exact or softmax")

    propagation: float | None = None
    total: float | None = None
    ordered = _canonical_reasons(reasons)
    state_total = state_bound.get("total_bound")
    if not ordered and state_total is not None and recovery_radius is not None and leakage is not None:
        propagation = gamma * float(state_total)
        total = propagation + recovery_radius + leakage
        if not all(math.isfinite(value) for value in (propagation, total)):
            reasons.append("numerical_nonfinite")

    ordered = _canonical_reasons(reasons)
    if ordered:
        propagation = total = None
    emitted = not ordered
    return {
        "route": route,
        "matching": matching,
        "beta": route_beta,
        "recovery_kernel_diagonal_lower_bound": diagonal,
        "value_total_bound": state_total,
        "value_propagation": propagation,
        "fixed_recovery": recovery_radius,
        "softmax_leakage": leakage,
        "total_bound": total,
        "finite_bound_emitted": bool(emitted and total is not None),
        "improves_over_zero_initialization": bool(
            emitted and total is not None and total < value_bound
        ),
        "selective_high_probability_certified": emitted,
        "status": (
            STATUS_SELECTIVE_HIGH_PROBABILITY if emitted else STATUS_NOT_CERTIFIED
        ),
        "failure_reasons": ordered,
    }


def build_visit_indexed_certificate(
    *,
    state_counts: Sequence[int],
    pair_counts: Sequence[int],
    trajectory_length: int,
    reward_bound: float,
    gamma: float,
    alpha: float,
    beta: float,
    delta: float,
    direct_exact_iterations: int,
    direct_softmax_iterations: int,
    state_exact_iterations: int,
    state_softmax_iterations: int,
    direct_exact_diverged: bool = False,
    direct_softmax_diverged: bool = False,
    fixed_context: bool = True,
    synchronous_update: bool = True,
    algorithm_mode: str = "fixed_policy_synchronous",
    iteration_cap: int | None = None,
) -> dict[str, Any]:
    """Build all mandatory visit-indexed route certificates.

    The interface intentionally excludes true values, residuals, transition
    laws, stationary occupancies, and spectral diagnostics.
    """
    horizon = _positive_integer("trajectory_length", trajectory_length)
    reward = _finite_number("reward_bound", reward_bound)
    discount = _finite_number("gamma", gamma)
    step_size = _finite_number("alpha", alpha)
    sharpness = _finite_number("beta", beta)
    confidence = _finite_number("delta", delta)
    if reward < 0.0:
        raise ValueError("reward_bound must be nonnegative")
    if not 0.0 < discount < 1.0:
        raise ValueError("gamma must lie in (0,1)")
    if not 0.0 < step_size <= 1.0:
        raise ValueError("alpha must lie in (0,1]")
    if not 0.0 < confidence < 1.0:
        raise ValueError("delta must lie in (0,1)")

    state_count_values = _validated_counts("state_counts", state_counts, horizon)
    pair_count_values = _validated_counts("pair_counts", pair_counts, horizon)
    n_states = len(state_count_values)
    n_pairs = len(pair_count_values)
    if n_pairs % n_states != 0:
        raise ValueError("pair_counts length must be a multiple of state_counts length")
    n_actions = n_pairs // n_states
    state_counts_from_pairs = [
        sum(pair_count_values[state * n_actions : (state + 1) * n_actions])
        for state in range(n_states)
    ]
    if state_counts_from_pairs != state_count_values:
        raise ValueError("state_counts must agree with state-aggregated pair_counts")
    if not isinstance(fixed_context, bool) or not isinstance(
        synchronous_update, bool
    ):
        raise ValueError("algorithm mode flags must be boolean")
    if not isinstance(algorithm_mode, str):
        raise ValueError("algorithm_mode must be a string")
    actual_iterations = {
        "direct_exact": _nonnegative_integer(
            "direct_exact_iterations", direct_exact_iterations
        ),
        "direct_softmax": _nonnegative_integer(
            "direct_softmax_iterations", direct_softmax_iterations
        ),
        "state_exact": _nonnegative_integer(
            "state_exact_iterations", state_exact_iterations
        ),
        "state_softmax": _nonnegative_integer(
            "state_softmax_iterations", state_softmax_iterations
        ),
    }
    if iteration_cap is None:
        cap = max(actual_iterations.values())
    else:
        cap = _nonnegative_integer("iteration_cap", iteration_cap)
        if any(value > cap for value in actual_iterations.values()):
            raise ValueError("actual iterations cannot exceed iteration_cap")
    if not isinstance(direct_exact_diverged, bool) or not isinstance(
        direct_softmax_diverged, bool
    ):
        raise ValueError("divergence flags must be boolean")

    n_groups = n_states + 2 * n_pairs
    value_bound = reward / (1.0 - discount)
    log_term = math.log(2.0 * n_groups * horizon / confidence)
    event = {
        "method": "finite_horizon_visit_indexed_hoeffding",
        "probability_statement": "P(Emit and certified error bound is violated) <= delta",
        "delta": confidence,
        "horizon": horizon,
        "n_states": n_states,
        "n_pairs": n_pairs,
        "n_groups": n_groups,
        "group_accounting": "m state Bellman + d pair Bellman + d recovery",
        "value_bound": value_bound,
        "conditional_interval_width": 2.0 * value_bound,
        "absolute_residual_bound": 2.0 * value_bound,
        "log_term": log_term,
        "risk_allocation": {
            "allocation_time": "before observing trajectory counts",
            "one_sided_failure_probability": confidence / (2.0 * n_groups * horizon),
            "two_sided_group_count_failure_probability": confidence
            / (n_groups * horizon),
            "group_count_events": n_groups * horizon,
            "total_failure_probability": confidence,
            "route_level_resplit": False,
        },
        "state_bellman": _family_summary(
            state_count_values,
            reward_bound=reward,
            gamma=discount,
            n_groups=n_groups,
            horizon=horizon,
            delta=confidence,
        ),
        "pair_bellman": _family_summary(
            pair_count_values,
            reward_bound=reward,
            gamma=discount,
            n_groups=n_groups,
            horizon=horizon,
            delta=confidence,
        ),
        "recovery": _family_summary(
            pair_count_values,
            reward_bound=reward,
            gamma=discount,
            n_groups=n_groups,
            horizon=horizon,
            delta=confidence,
        ),
    }

    state_diagonals = empirical_one_hot_diagonals(
        state_count_values, horizon=horizon, beta=sharpness
    )
    pair_diagonals = empirical_one_hot_diagonals(
        pair_count_values, horizon=horizon, beta=sharpness
    )
    state_diagonal = _minimum_diagonal(state_diagonals)
    pair_diagonal = _minimum_diagonal(pair_diagonals)
    base_reasons: list[str] = []
    if (
        not fixed_context
        or not synchronous_update
        or algorithm_mode != "fixed_policy_synchronous"
    ):
        base_reasons.append("algorithm_mode_mismatch")

    state_exact = _uniform_route_bound(
        route="state_exact",
        matching="exact",
        support_missing_reason="state_support_missing",
        margin_failure_reason="state_kernel_margin_nonpositive",
        full_support=bool(event["state_bellman"]["full_support"]),
        residual_radius=event["state_bellman"]["max_radius"],
        diagonal=None,
        beta=sharpness,
        gamma=discount,
        alpha=step_size,
        iterations=actual_iterations["state_exact"],
        value_bound=value_bound,
        base_reasons=base_reasons,
        diverged=False,
    )
    state_softmax = _uniform_route_bound(
        route="state_softmax",
        matching="softmax",
        support_missing_reason="state_support_missing",
        margin_failure_reason="state_kernel_margin_nonpositive",
        full_support=bool(event["state_bellman"]["full_support"]),
        residual_radius=event["state_bellman"]["max_radius"],
        diagonal=state_diagonal,
        beta=sharpness,
        gamma=discount,
        alpha=step_size,
        iterations=actual_iterations["state_softmax"],
        value_bound=value_bound,
        base_reasons=base_reasons,
        diverged=False,
    )
    direct_exact = _uniform_route_bound(
        route="direct_exact",
        matching="exact",
        support_missing_reason="pair_support_missing",
        margin_failure_reason="pair_kernel_margin_nonpositive",
        full_support=bool(event["pair_bellman"]["full_support"]),
        residual_radius=event["pair_bellman"]["max_radius"],
        diagonal=None,
        beta=sharpness,
        gamma=discount,
        alpha=step_size,
        iterations=actual_iterations["direct_exact"],
        value_bound=value_bound,
        base_reasons=base_reasons,
        diverged=direct_exact_diverged,
    )
    direct_softmax = _uniform_route_bound(
        route="direct_softmax",
        matching="softmax",
        support_missing_reason="pair_support_missing",
        margin_failure_reason="pair_kernel_margin_nonpositive",
        full_support=bool(event["pair_bellman"]["full_support"]),
        residual_radius=event["pair_bellman"]["max_radius"],
        diagonal=pair_diagonal,
        beta=sharpness,
        gamma=discount,
        alpha=step_size,
        iterations=actual_iterations["direct_softmax"],
        value_bound=value_bound,
        base_reasons=base_reasons,
        diverged=direct_softmax_diverged,
    )
    vfirst_exact = _vfirst_bound(
        route="vfirst_nosplit_exact",
        matching="exact",
        state_bound=state_exact,
        pair_full_support=bool(event["recovery"]["full_support"]),
        recovery_radius=event["recovery"]["max_radius"],
        pair_diagonal=None,
        beta=sharpness,
        gamma=discount,
        value_bound=value_bound,
        base_reasons=base_reasons,
    )
    vfirst_softmax = _vfirst_bound(
        route="vfirst_nosplit_softmax",
        matching="softmax",
        state_bound=state_softmax,
        pair_full_support=bool(event["recovery"]["full_support"]),
        recovery_radius=event["recovery"]["max_radius"],
        pair_diagonal=pair_diagonal,
        beta=sharpness,
        gamma=discount,
        value_bound=value_bound,
        base_reasons=base_reasons,
    )

    return {
        "certificate_inputs": {
            "observed": {
                "state_counts": state_count_values,
                "pair_counts": pair_count_values,
                "state_softmax_diagonal_by_group": state_diagonals,
                "pair_softmax_diagonal_by_group": pair_diagonals,
            },
            "declared": {
                "reward_bound": reward,
                "value_bound": value_bound,
                "gamma": discount,
                "alpha": step_size,
                "beta": sharpness,
                "delta": confidence,
                "trajectory_length": horizon,
                "nontriviality_rule": "emitted total_bound < value_bound",
            },
            "algorithm": {
                "mode": algorithm_mode,
                "fixed_context": bool(fixed_context),
                "synchronous_update": bool(synchronous_update),
                "zero_initialization": True,
                "iteration_cap": cap,
                "iterations_used": actual_iterations,
                "direct_exact_diverged": direct_exact_diverged,
                "direct_softmax_diverged": direct_softmax_diverged,
            },
        },
        "event": event,
        "state_value": {
            "exact": state_exact,
            "softmax": state_softmax,
        },
        "routes": {
            "direct_exact": direct_exact,
            "direct_softmax": direct_softmax,
            "vfirst_nosplit_exact": vfirst_exact,
            "vfirst_nosplit_softmax": vfirst_softmax,
        },
        "optional_variance_adaptive": {
            "status": "not_implemented_no_observable_variance_proxy",
            "selected": False,
            "delta_allocated": 0.0,
            "affects_mandatory_status": False,
            "reason": (
                "counts alone do not supply the predictable conditional variance "
                "or a proved empirical substitute required by a Freedman-style bound"
            ),
        },
    }
