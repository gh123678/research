"""Pure time-uniform mixture certificates for frozen fixed-policy evaluation.

The module consumes only observed visit counts, declared bounds, and public
hyperparameters.  It replaces the count-wise Hoeffding union radius of the
verified visit-indexed certificate by a preregistered 15-component geometric
cosh mixture of exponential supermartingales, with the analytic line
stitching used only as an audit and deterministic inversion bracket.  It has
no sampling, file-writing, plotting, true-model, occupancy, or spectral
dependency.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from fixed_policy_finite_sample_certificate import strict_json_ready as strict_json_ready
from visit_indexed_martingale_certificate import (
    empirical_one_hot_diagonals,
    simultaneous_hoeffding_radius,
)


STATUS_SELECTIVE_HIGH_PROBABILITY = "selective_high_probability_certified"
STATUS_NOT_CERTIFIED = "not_certified"

MIXTURE_GRID_SIZE = 15
INVERSION_TOL = 1e-12
INVERSION_MAX_ITERATIONS = 200

_FAILURE_ORDER = (
    "algorithm_mode_mismatch",
    "divergence_guard_triggered",
    "state_support_missing",
    "pair_support_missing",
    "state_kernel_margin_nonpositive",
    "pair_kernel_margin_nonpositive",
    "mixture_inversion_unbracketed",
    "mixture_inversion_not_converged",
    "mixture_root_not_conservative",
    "numerical_nonfinite",
)
_FAILURE_RANK = {reason: index for index, reason in enumerate(_FAILURE_ORDER)}


class MixtureInversionError(RuntimeError):
    """Deterministic numerical-inversion failure with an ordered reason."""

    def __init__(self, reason: str, message: str) -> None:
        if reason not in _FAILURE_RANK:
            raise ValueError(f"unknown inversion failure reason {reason}")
        super().__init__(message)
        self.reason = reason


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


_GRID_CACHE: dict[tuple[int, float], dict[str, Any]] = {}


def mixture_grid(n_groups: int, delta: float) -> dict[str, Any]:
    """Build and validate the frozen geometric grid, weights, and rates."""
    groups = _positive_integer("n_groups", n_groups)
    confidence = _finite_number("delta", delta)
    if not 0.0 < confidence < 1.0:
        raise ValueError("delta must lie in (0,1)")
    key = (groups, confidence)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    count_grid = [2**j for j in range(MIXTURE_GRID_SIZE)]
    raw = [(j + 1) ** -2 for j in range(MIXTURE_GRID_SIZE)]
    norm = math.fsum(raw)
    weights = [value / norm for value in raw]
    if abs(math.fsum(weights) - 1.0) > 1e-15:
        raise ValueError("mixture weights do not sum to one within 1e-15")
    log_terms = [math.log(2.0 * groups / (confidence * w)) for w in weights]
    rates = [math.sqrt(2.0 * log_term / k) for log_term, k in zip(log_terms, count_grid, strict=True)]
    if any(not math.isfinite(rate) or rate <= 0.0 for rate in rates):
        raise ValueError("mixture rates must be positive and finite")
    grid = {
        "n_groups": groups,
        "delta": confidence,
        "grid_size": MIXTURE_GRID_SIZE,
        "count_grid": count_grid,
        "weights": weights,
        "log_terms": log_terms,
        "rates": rates,
        "log_weights": [math.log(w) for w in weights],
        "half_squared_rates": [0.5 * a * a for a in rates],
    }
    _GRID_CACHE[key] = grid
    return grid


def log_cosh(x: float) -> float:
    """Stable log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log 2."""
    value = _finite_number("x", x)
    magnitude = abs(value)
    return magnitude + math.log1p(math.exp(-2.0 * magnitude)) - math.log(2.0)


def log_mixture(count: int, q: float, *, grid: dict[str, Any]) -> float:
    """Stable log M(k, q) over the frozen fifteen mixture components."""
    visits = _positive_integer("count", count)
    level = _finite_number("q", q)
    terms = [
        log_w - half_a2 * visits + log_cosh(rate * level)
        for log_w, half_a2, rate in zip(
            grid["log_weights"], grid["half_squared_rates"], grid["rates"], strict=True
        )
    ]
    peak = max(terms)
    result = peak + math.log(math.fsum(math.exp(term - peak) for term in terms))
    if not math.isfinite(result):
        raise ValueError("log mixture is nonfinite")
    return result


def stitch_boundary(count: int, *, grid: dict[str, Any]) -> float:
    """Analytic line-stitching boundary q_stitch(k) = min_j q_j(k)."""
    visits = _positive_integer("count", count)
    boundary = min(
        (log_term + half_a2 * visits) / rate
        for log_term, half_a2, rate in zip(
            grid["log_terms"], grid["half_squared_rates"], grid["rates"], strict=True
        )
    )
    if not math.isfinite(boundary) or boundary <= 0.0:
        raise ValueError("stitch boundary must be positive and finite")
    return boundary


def mixture_root(
    count: int,
    *,
    n_groups: int,
    delta: float,
    tol: float = INVERSION_TOL,
    max_iterations: int = INVERSION_MAX_ITERATIONS,
    grid: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Conservative upper endpoint of the unique root of M(k,q) = G/delta."""
    visits = _positive_integer("count", count)
    tolerance = _finite_number("tol", tol)
    if tolerance <= 0.0:
        raise ValueError("tol must be positive")
    iteration_cap = _positive_integer("max_iterations", max_iterations)
    if grid is None:
        grid = mixture_grid(n_groups, delta)
    target = math.log(float(grid["n_groups"]) / float(grid["delta"]))
    q_lo = 0.0
    q_hi = stitch_boundary(visits, grid=grid)
    log_lo = log_mixture(visits, q_lo, grid=grid)
    log_hi = log_mixture(visits, q_hi, grid=grid)
    if not (math.isfinite(log_lo) and math.isfinite(log_hi)):
        raise MixtureInversionError(
            "numerical_nonfinite", f"log mixture nonfinite at count {visits}"
        )
    if not log_lo < target:
        raise MixtureInversionError(
            "mixture_inversion_unbracketed",
            f"log M({visits}, 0) = {log_lo} is not below target {target}",
        )
    if not target <= log_hi:
        raise MixtureInversionError(
            "mixture_inversion_unbracketed",
            f"log M({visits}, q_stitch) = {log_hi} is below target {target}",
        )
    iterations = 0
    while True:
        width_ok = q_hi - q_lo <= tolerance * (1.0 + q_hi)
        residual = log_hi - target
        residual_ok = 0.0 <= residual <= tolerance * (1.0 + abs(target))
        if width_ok and residual_ok:
            break
        iterations += 1
        if iterations > iteration_cap:
            raise MixtureInversionError(
                "mixture_inversion_not_converged",
                f"bisection did not converge at count {visits} within {iteration_cap}",
            )
        mid = 0.5 * (q_lo + q_hi)
        if not q_lo < mid < q_hi:
            raise MixtureInversionError(
                "mixture_inversion_not_converged",
                f"bisection stagnated at count {visits}",
            )
        log_mid = log_mixture(visits, mid, grid=grid)
        if not math.isfinite(log_mid):
            raise MixtureInversionError(
                "numerical_nonfinite", f"log mixture nonfinite at count {visits}"
            )
        if log_mid >= target:
            q_hi, log_hi = mid, log_mid
        else:
            q_lo, log_lo = mid, log_mid
    if log_mixture(visits, q_hi, grid=grid) < target:
        raise MixtureInversionError(
            "mixture_root_not_conservative",
            f"returned endpoint below threshold at count {visits}",
        )
    return {
        "count": visits,
        "q_mix": q_hi,
        "q_stitch": stitch_boundary(visits, grid=grid),
        "target_log_threshold": target,
        "log_mixture_at_root": log_hi,
        "iterations": iterations,
        "bracket_width": q_hi - q_lo,
    }


def mixture_radius(
    count: int,
    *,
    reward_bound: float,
    gamma: float,
    n_groups: int,
    delta: float,
) -> float:
    """Time-uniform residual-mean radius r_mix(k) = B q_mix(k) / k."""
    visits = _positive_integer("count", count)
    reward = _finite_number("reward_bound", reward_bound)
    discount = _finite_number("gamma", gamma)
    if reward <= 0.0:
        raise ValueError("reward_bound must be positive so that B > 0")
    if not 0.0 < discount < 1.0:
        raise ValueError("gamma must lie in (0,1)")
    value_bound = reward / (1.0 - discount)
    result = mixture_root(visits, n_groups=n_groups, delta=delta)
    radius = value_bound * result["q_mix"] / visits
    stitch = value_bound * result["q_stitch"] / visits
    if not math.isfinite(radius):
        raise MixtureInversionError(
            "numerical_nonfinite", f"mixture radius nonfinite at count {visits}"
        )
    if radius > stitch * (1.0 + 1e-15) + 1e-300:
        raise MixtureInversionError(
            "mixture_root_not_conservative",
            f"mixture radius exceeds stitch audit at count {visits}",
        )
    return radius


def stitch_radius(
    count: int,
    *,
    reward_bound: float,
    gamma: float,
    n_groups: int,
    delta: float,
) -> float:
    """Audit-only stitched radius r_stitch(k) = B q_stitch(k) / k."""
    visits = _positive_integer("count", count)
    reward = _finite_number("reward_bound", reward_bound)
    discount = _finite_number("gamma", gamma)
    if reward <= 0.0:
        raise ValueError("reward_bound must be positive so that B > 0")
    if not 0.0 < discount < 1.0:
        raise ValueError("gamma must lie in (0,1)")
    grid = mixture_grid(n_groups, delta)
    value_bound = reward / (1.0 - discount)
    return value_bound * stitch_boundary(visits, grid=grid) / visits


def _family_summary(
    counts: Sequence[int],
    *,
    reward_bound: float,
    gamma: float,
    n_groups: int,
    horizon: int,
    delta: float,
) -> dict[str, Any]:
    mixture_radii: list[float | None] = []
    stitch_radii: list[float | None] = []
    old_radii: list[float | None] = []
    inversion_reasons: list[str] = []
    for count in counts:
        if count == 0:
            mixture_radii.append(None)
            stitch_radii.append(None)
            old_radii.append(None)
            continue
        try:
            mixture_value = mixture_radius(
                count,
                reward_bound=reward_bound,
                gamma=gamma,
                n_groups=n_groups,
                delta=delta,
            )
        except MixtureInversionError as error:
            mixture_value = None
            inversion_reasons.append(error.reason)
        mixture_radii.append(mixture_value)
        stitch_radii.append(
            stitch_radius(
                count,
                reward_bound=reward_bound,
                gamma=gamma,
                n_groups=n_groups,
                delta=delta,
            )
        )
        old_radii.append(
            simultaneous_hoeffding_radius(
                count,
                reward_bound=reward_bound,
                gamma=gamma,
                n_groups=n_groups,
                horizon=horizon,
                delta=delta,
            )
        )
    visited_mixture = [r for r in mixture_radii if r is not None]
    visited_stitch = [r for r in stitch_radii if r is not None]
    visited_old = [r for r in old_radii if r is not None]
    support_complete = all(count > 0 for count in counts)
    full_support = support_complete and not inversion_reasons
    paired = [
        (mix, stitch, old)
        for mix, stitch, old in zip(mixture_radii, stitch_radii, old_radii, strict=True)
        if mix is not None and stitch is not None and old is not None
    ]
    return {
        "counts": list(counts),
        "missing_groups": int(sum(count == 0 for count in counts)),
        "support_complete": support_complete,
        "full_support": full_support,
        "min_count": int(min(counts)),
        "min_visited_count": int(min(count for count in counts if count > 0)),
        "mixture_radius_by_group": mixture_radii,
        "stitch_radius_by_group": stitch_radii,
        "old_radius_by_group": old_radii,
        "max_radius": max(visited_mixture) if full_support else None,
        "max_visited_radius": max(visited_mixture) if visited_mixture else None,
        "max_stitch_radius": max(visited_stitch) if visited_stitch else None,
        "max_old_radius": max(visited_old) if visited_old else None,
        "max_radius_ratio_vs_old": (
            max(mix / old for mix, _, old in paired) if paired else None
        ),
        "max_radius_ratio_vs_stitch": (
            max(mix / stitch for mix, stitch, _ in paired) if paired else None
        ),
        "max_radius_reduction_vs_old": (
            max(old - mix for mix, _, old in paired) if paired else None
        ),
        "inversion_failure_reasons": _canonical_reasons(inversion_reasons),
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
    support_complete: bool,
    residual_radius: float | None,
    stitch_residual_radius: float | None,
    old_residual_radius: float | None,
    inversion_reasons: Sequence[str],
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
    reasons.extend(str(reason) for reason in inversion_reasons)
    if diverged:
        reasons.append("divergence_guard_triggered")
    if not support_complete:
        reasons.append(support_missing_reason)

    if matching == "exact":
        route_diagonal = 1.0 if support_complete else None
        margin = (1.0 - gamma) / 2.0 if support_complete else None
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
    old_total: float | None = None
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
            if terms is None or old_residual_radius is None:
                reasons.append("numerical_nonfinite")
            else:
                optimization, statistical, total = terms
                old_terms = _uniform_terms(
                    rho=rho,
                    iterations=iterations,
                    initial_error=value_bound,
                    residual_radius=old_residual_radius,
                    margin=margin,
                )
                if old_terms is None:
                    reasons.append("numerical_nonfinite")
                else:
                    old_total = old_terms[2]

    ordered = _canonical_reasons(reasons)
    if ordered:
        rho = optimization = statistical = total = old_total = None
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
        "residual_radius": residual_radius if emitted else None,
        "stitch_residual_radius": stitch_residual_radius if emitted else None,
        "old_residual_radius": old_residual_radius if emitted else None,
        "radius_ratio_vs_old": (
            residual_radius / old_residual_radius
            if emitted and residual_radius is not None and old_residual_radius
            else None
        ),
        "radius_reduction_vs_old": (
            old_residual_radius - residual_radius
            if emitted and residual_radius is not None and old_residual_radius is not None
            else None
        ),
        "radius_ratio_vs_stitch": (
            residual_radius / stitch_residual_radius
            if emitted and residual_radius is not None and stitch_residual_radius
            else None
        ),
        "optimization_term": optimization,
        "statistical_term": statistical,
        "total_bound": total,
        "old_total_bound": old_total,
        "total_bound_ratio_vs_old": (
            total / old_total if emitted and total is not None and old_total else None
        ),
        "total_bound_reduction_vs_old": (
            old_total - total
            if emitted and total is not None and old_total is not None
            else None
        ),
        "finite_bound_emitted": bool(emitted and total is not None),
        "improves_over_zero_initialization": bool(
            emitted and total is not None and total < value_bound
        ),
        "below_two_value_bound": bool(
            emitted and total is not None and total < 2.0 * value_bound
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
    pair_support_complete: bool,
    recovery_radius: float | None,
    stitch_recovery_radius: float | None,
    old_recovery_radius: float | None,
    recovery_inversion_reasons: Sequence[str],
    pair_diagonal: float | None,
    beta: float,
    gamma: float,
    value_bound: float,
    base_reasons: Sequence[str],
) -> dict[str, Any]:
    reasons = list(base_reasons)
    reasons.extend(str(reason) for reason in state_bound["failure_reasons"])
    reasons.extend(str(reason) for reason in recovery_inversion_reasons)
    if not pair_support_complete:
        reasons.append("pair_support_missing")

    if matching == "exact":
        diagonal = 1.0 if pair_support_complete else None
        route_beta: float | None = None
        leakage = 0.0 if pair_support_complete else None
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
    old_propagation: float | None = None
    old_total: float | None = None
    ordered = _canonical_reasons(reasons)
    state_total = state_bound.get("total_bound")
    old_state_total = state_bound.get("old_total_bound")
    if not ordered and state_total is not None and recovery_radius is not None and leakage is not None:
        propagation = gamma * float(state_total)
        total = propagation + recovery_radius + leakage
        if not all(math.isfinite(value) for value in (propagation, total)):
            reasons.append("numerical_nonfinite")
        elif old_state_total is not None and old_recovery_radius is not None:
            old_propagation = gamma * float(old_state_total)
            old_total = old_propagation + old_recovery_radius + leakage
            if not all(math.isfinite(value) for value in (old_propagation, old_total)):
                reasons.append("numerical_nonfinite")

    ordered = _canonical_reasons(reasons)
    if ordered:
        propagation = total = old_propagation = old_total = None
    emitted = not ordered
    return {
        "route": route,
        "matching": matching,
        "beta": route_beta,
        "recovery_kernel_diagonal_lower_bound": diagonal,
        "value_total_bound": state_total,
        "old_value_total_bound": old_state_total,
        "value_propagation": propagation,
        "old_value_propagation": old_propagation,
        "fixed_recovery": recovery_radius if emitted else None,
        "stitch_fixed_recovery": stitch_recovery_radius if emitted else None,
        "old_fixed_recovery": old_recovery_radius if emitted else None,
        "radius_ratio_vs_old": (
            recovery_radius / old_recovery_radius
            if emitted and recovery_radius is not None and old_recovery_radius
            else None
        ),
        "radius_reduction_vs_old": (
            old_recovery_radius - recovery_radius
            if emitted and recovery_radius is not None and old_recovery_radius is not None
            else None
        ),
        "radius_ratio_vs_stitch": (
            recovery_radius / stitch_recovery_radius
            if emitted and recovery_radius is not None and stitch_recovery_radius
            else None
        ),
        "softmax_leakage": leakage,
        "total_bound": total,
        "old_total_bound": old_total,
        "total_bound_ratio_vs_old": (
            total / old_total if emitted and total is not None and old_total else None
        ),
        "total_bound_reduction_vs_old": (
            old_total - total
            if emitted and total is not None and old_total is not None
            else None
        ),
        "finite_bound_emitted": bool(emitted and total is not None),
        "improves_over_zero_initialization": bool(
            emitted and total is not None and total < value_bound
        ),
        "below_two_value_bound": bool(
            emitted and total is not None and total < 2.0 * value_bound
        ),
        "selective_high_probability_certified": emitted,
        "status": (
            STATUS_SELECTIVE_HIGH_PROBABILITY if emitted else STATUS_NOT_CERTIFIED
        ),
        "failure_reasons": ordered,
    }


def build_time_uniform_certificate(
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
    """Build all mandatory time-uniform mixture route certificates.

    The interface intentionally excludes true values, residuals, transition
    laws, stationary occupancies, and spectral diagnostics.
    """
    horizon = _positive_integer("trajectory_length", trajectory_length)
    reward = _finite_number("reward_bound", reward_bound)
    discount = _finite_number("gamma", gamma)
    step_size = _finite_number("alpha", alpha)
    sharpness = _finite_number("beta", beta)
    confidence = _finite_number("delta", delta)
    if reward <= 0.0:
        raise ValueError("reward_bound must be positive so that B > 0")
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
    grid = mixture_grid(n_groups, confidence)
    state_family = _family_summary(
        state_count_values,
        reward_bound=reward,
        gamma=discount,
        n_groups=n_groups,
        horizon=horizon,
        delta=confidence,
    )
    pair_family = _family_summary(
        pair_count_values,
        reward_bound=reward,
        gamma=discount,
        n_groups=n_groups,
        horizon=horizon,
        delta=confidence,
    )
    recovery_family = _family_summary(
        pair_count_values,
        reward_bound=reward,
        gamma=discount,
        n_groups=n_groups,
        horizon=horizon,
        delta=confidence,
    )
    event = {
        "method": "time_uniform_geometric_cosh_mixture",
        "probability_statement": "P(Emit and certified error bound is violated) <= delta",
        "delta": confidence,
        "horizon": horizon,
        "n_states": n_states,
        "n_pairs": n_pairs,
        "n_groups": n_groups,
        "group_accounting": "m state Bellman + d pair Bellman + d recovery",
        "value_bound": value_bound,
        "conditional_interval_width": 2.0 * value_bound,
        "mixture": {
            "grid_size": grid["grid_size"],
            "count_grid": grid["count_grid"],
            "weights": grid["weights"],
            "log_terms": grid["log_terms"],
            "rates": grid["rates"],
            "weight_formula": "w_j = (j+1)^(-2) / sum_{l=0}^{14} (l+1)^(-2)",
            "rate_formula": "a_j = sqrt(2 L_j / k_j), L_j = log(2 G / (delta w_j)), k_j = 2^j",
            "target_log_threshold": math.log(n_groups / confidence),
            "inversion_tolerance": INVERSION_TOL,
            "inversion_max_iterations": INVERSION_MAX_ITERATIONS,
        },
        "risk_allocation": {
            "allocation_time": "before observing trajectory counts",
            "per_group_failure_probability": confidence / n_groups,
            "groups": n_groups,
            "total_failure_probability": confidence,
            "two_sided_control": "one cosh mixture per group controls both signs",
            "line_stitching_role": "audit-only upper bracket; never selected",
            "route_level_resplit": False,
        },
        "state_bellman": state_family,
        "pair_bellman": pair_family,
        "recovery": recovery_family,
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
        support_complete=bool(state_family["support_complete"]),
        residual_radius=state_family["max_radius"],
        stitch_residual_radius=state_family["max_stitch_radius"],
        old_residual_radius=state_family["max_old_radius"],
        inversion_reasons=state_family["inversion_failure_reasons"],
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
        support_complete=bool(state_family["support_complete"]),
        residual_radius=state_family["max_radius"],
        stitch_residual_radius=state_family["max_stitch_radius"],
        old_residual_radius=state_family["max_old_radius"],
        inversion_reasons=state_family["inversion_failure_reasons"],
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
        support_complete=bool(pair_family["support_complete"]),
        residual_radius=pair_family["max_radius"],
        stitch_residual_radius=pair_family["max_stitch_radius"],
        old_residual_radius=pair_family["max_old_radius"],
        inversion_reasons=pair_family["inversion_failure_reasons"],
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
        support_complete=bool(pair_family["support_complete"]),
        residual_radius=pair_family["max_radius"],
        stitch_residual_radius=pair_family["max_stitch_radius"],
        old_residual_radius=pair_family["max_old_radius"],
        inversion_reasons=pair_family["inversion_failure_reasons"],
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
        pair_support_complete=bool(recovery_family["support_complete"]),
        recovery_radius=recovery_family["max_radius"],
        stitch_recovery_radius=recovery_family["max_stitch_radius"],
        old_recovery_radius=recovery_family["max_old_radius"],
        recovery_inversion_reasons=recovery_family["inversion_failure_reasons"],
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
        pair_support_complete=bool(recovery_family["support_complete"]),
        recovery_radius=recovery_family["max_radius"],
        stitch_recovery_radius=recovery_family["max_stitch_radius"],
        old_recovery_radius=recovery_family["max_old_radius"],
        recovery_inversion_reasons=recovery_family["inversion_failure_reasons"],
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
    }
