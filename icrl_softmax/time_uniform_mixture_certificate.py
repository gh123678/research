"""Pure time-uniform mixture certificates for FP-TU-001.

Only observed visit counts, declared bounds, public hyperparameters, and
algorithm metadata enter this module.  Sampling and truth-based diagnostics
belong in the evaluator's separate oracle audit.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any

from fixed_policy_finite_sample_certificate import strict_json_ready as strict_json_ready
from visit_indexed_martingale_certificate import build_visit_indexed_certificate


DEFAULT_COMPONENTS = 15
DEFAULT_MAX_COUNT = 16384
DEFAULT_TOLERANCE = 1e-12
DEFAULT_MAX_ITERATIONS = 200

STATUS_SELECTIVE_HIGH_PROBABILITY = "selective_high_probability_certified"
STATUS_NOT_CERTIFIED = "not_certified"
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
    """Numerical failure carrying the frozen ordered non-emission reason."""

    def __init__(self, reason: str, message: str) -> None:
        if reason not in _FAILURE_RANK:
            raise ValueError(f"unknown inversion failure reason {reason}")
        super().__init__(message)
        self.reason = reason


def _canonical_reasons(reasons: Sequence[str]) -> list[str]:
    return sorted(
        set(reasons),
        key=lambda reason: (_FAILURE_RANK.get(reason, len(_FAILURE_ORDER)), reason),
    )


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_integer(name: str, value: int) -> int:
    integer = int(value)
    if isinstance(value, bool) or integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def log_cosh(value: float) -> float:
    """Return log(cosh(value)) without overflow."""
    number = _finite("value", value)
    magnitude = abs(number)
    result = magnitude + math.log1p(math.exp(-2.0 * magnitude)) - math.log(2.0)
    if not math.isfinite(result):
        raise ValueError("logcosh is nonfinite")
    return result


def logsumexp(values: Sequence[float]) -> float:
    """Return the stable logarithm of a nonempty exponential sum."""
    numbers = [_finite(f"values[{index}]", value) for index, value in enumerate(values)]
    if not numbers:
        raise ValueError("values must be nonempty")
    maximum = max(numbers)
    result = maximum + math.log(sum(math.exp(value - maximum) for value in numbers))
    if not math.isfinite(result):
        raise ValueError("logsumexp is nonfinite")
    return result


def build_mixture_grid(
    *, n_groups: int, delta: float, components: int = DEFAULT_COMPONENTS
) -> list[dict[str, float | int]]:
    """Construct the frozen geometric grid, normalized weights, and rates."""
    groups = _positive_integer("n_groups", n_groups)
    component_count = _positive_integer("components", components)
    if component_count != DEFAULT_COMPONENTS:
        raise ValueError(f"components must equal {DEFAULT_COMPONENTS}")
    confidence = _finite("delta", delta)
    if not 0.0 < confidence < 1.0:
        raise ValueError("delta must lie in (0,1)")
    normalizer = sum((index + 1) ** -2 for index in range(component_count))
    grid: list[dict[str, float | int]] = []
    for index in range(component_count):
        target_count = 2**index
        weight = (index + 1) ** -2 / normalizer
        log_level = math.log(2.0 * groups / (confidence * weight))
        rate = math.sqrt(2.0 * log_level / target_count)
        grid.append(
            {
                "index": index,
                "target_count": target_count,
                "weight": weight,
                "log_level": log_level,
                "rate": rate,
            }
        )
    if abs(sum(float(row["weight"]) for row in grid) - 1.0) > 1e-15:
        raise ValueError("mixture weights do not sum to one")
    return grid


def _validated_grid(
    grid: Sequence[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    rows = [dict(row) for row in grid]
    if len(rows) != DEFAULT_COMPONENTS:
        raise ValueError(f"grid must contain exactly {DEFAULT_COMPONENTS} components")
    for index, row in enumerate(rows):
        if int(row.get("index", -1)) != index:
            raise ValueError("grid indices are not canonical")
        if int(row.get("target_count", 0)) != 2**index:
            raise ValueError("grid target counts are not canonical")
        for key in ("weight", "log_level", "rate"):
            if _finite(f"grid[{index}].{key}", float(row[key])) <= 0.0:
                raise ValueError(f"grid[{index}].{key} must be positive")
    if abs(sum(float(row["weight"]) for row in rows) - 1.0) > 1e-15:
        raise ValueError("mixture weights do not sum to one")
    return rows


def _require_frozen_grid(
    rows: Sequence[dict[str, float | int]], *, n_groups: int, delta: float
) -> None:
    expected = build_mixture_grid(n_groups=n_groups, delta=delta)
    for index, (actual, frozen) in enumerate(zip(rows, expected, strict=True)):
        for key in ("weight", "log_level", "rate"):
            if not math.isclose(
                float(actual[key]), float(frozen[key]), rel_tol=1e-15, abs_tol=1e-15
            ):
                raise ValueError(f"grid[{index}].{key} differs from the frozen formula")


def log_mixture(
    count: int,
    boundary: float,
    grid: Sequence[dict[str, float | int]],
) -> float:
    """Evaluate log M(k,q) in a stable form."""
    visits = int(count)
    if isinstance(count, bool) or visits != count or visits < 0:
        raise ValueError("count must be a nonnegative integer")
    q_value = _finite("boundary", boundary)
    rows = _validated_grid(grid)
    terms = []
    for row in rows:
        weight = float(row["weight"])
        rate = float(row["rate"])
        terms.append(
            math.log(weight)
            - rate * rate * visits / 2.0
            + log_cosh(rate * q_value)
        )
    return logsumexp(terms)


def stitch_boundary(
    count: int, grid: Sequence[dict[str, float | int]]
) -> float:
    """Return the analytic audit boundary min_j q_j(k)."""
    visits = _positive_integer("count", count)
    rows = _validated_grid(grid)
    boundary = min(
        (float(row["log_level"]) + float(row["rate"]) ** 2 * visits / 2.0)
        / float(row["rate"])
        for row in rows
    )
    if not math.isfinite(boundary) or boundary <= 0.0:
        raise ValueError("stitch boundary is invalid")
    return boundary


def solve_mixture_boundary(
    count: int,
    grid: Sequence[dict[str, float | int]],
    *,
    n_groups: int,
    delta: float,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> dict[str, float | int]:
    """Conservatively invert the frozen mixture with a maintained bracket."""
    visits = _positive_integer("count", count)
    groups = _positive_integer("n_groups", n_groups)
    confidence = _finite("delta", delta)
    tol = _finite("tolerance", tolerance)
    iterations_cap = _positive_integer("max_iterations", max_iterations)
    if not 0.0 < confidence < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    if iterations_cap > DEFAULT_MAX_ITERATIONS:
        raise ValueError(f"max_iterations cannot exceed {DEFAULT_MAX_ITERATIONS}")
    rows = _validated_grid(grid)
    _require_frozen_grid(rows, n_groups=groups, delta=confidence)
    target = math.log(groups / confidence)
    lower = 0.0
    upper = stitch_boundary(visits, rows)
    lower_value = log_mixture(visits, lower, rows)
    upper_value = log_mixture(visits, upper, rows)
    if not all(math.isfinite(value) for value in (lower_value, upper_value)):
        raise MixtureInversionError(
            "numerical_nonfinite", "mixture inversion bracket is nonfinite"
        )
    if not lower_value < target <= upper_value:
        raise MixtureInversionError(
            "mixture_inversion_unbracketed", "mixture inversion bracket is invalid"
        )

    used = 0
    for used in range(1, iterations_cap + 1):
        midpoint = (lower + upper) / 2.0
        midpoint_value = log_mixture(visits, midpoint, rows)
        if midpoint_value >= target:
            upper = midpoint
            upper_value = midpoint_value
        else:
            lower = midpoint
            lower_value = midpoint_value
        width_ok = upper - lower <= tol * (1.0 + upper)
        residual = upper_value - target
        residual_ok = 0.0 <= residual <= tol * (1.0 + abs(target))
        if width_ok and residual_ok:
            break
    else:
        raise MixtureInversionError(
            "mixture_inversion_not_converged", "mixture inversion did not converge"
        )

    if not lower_value < target <= upper_value:
        raise MixtureInversionError(
            "mixture_root_not_conservative",
            "mixture inversion lost its conservative bracket",
        )
    return {
        "boundary": upper,
        "lower_boundary": lower,
        "bracket_width": upper - lower,
        "upper_gap": upper_value - target,
        "target_log_level": target,
        "iterations": used,
        "tolerance": tol,
    }


def mixture_radius(
    count: int,
    *,
    reward_bound: float,
    gamma: float,
    n_groups: int,
    delta: float,
) -> float:
    """Return B*q_mix(k)/k for the frozen mixture."""
    visits = _positive_integer("count", count)
    reward = _finite("reward_bound", reward_bound)
    discount = _finite("gamma", gamma)
    if reward <= 0.0:
        raise ValueError("reward_bound must be positive")
    if not 0.0 < discount < 1.0:
        raise ValueError("gamma must lie in (0,1)")
    grid = build_mixture_grid(n_groups=n_groups, delta=delta)
    solution = solve_mixture_boundary(
        visits, grid, n_groups=n_groups, delta=delta
    )
    return reward / (1.0 - discount) * float(solution["boundary"]) / visits


def old_visit_indexed_radius(
    count: int,
    *,
    reward_bound: float,
    gamma: float,
    n_groups: int,
    horizon: int,
    delta: float,
) -> float:
    """Return the exact legacy per-record radius for an audit comparison."""
    visits = _positive_integer("count", count)
    length = _positive_integer("horizon", horizon)
    groups = _positive_integer("n_groups", n_groups)
    if visits > length:
        raise ValueError("count cannot exceed horizon")
    reward = _finite("reward_bound", reward_bound)
    discount = _finite("gamma", gamma)
    confidence = _finite("delta", delta)
    if reward <= 0.0 or not 0.0 < discount < 1.0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid radius parameters")
    value_bound = reward / (1.0 - discount)
    return value_bound * math.sqrt(2.0 * math.log(2.0 * groups * length / confidence) / visits)


def _family_summary(
    counts: Sequence[int],
    *,
    reward_bound: float,
    gamma: float,
    n_groups: int,
    horizon: int,
    delta: float,
    grid: Sequence[dict[str, float | int]],
    cache: dict[int, dict[str, float | int]],
) -> dict[str, Any]:
    radii: list[float | None] = []
    stitch_radii: list[float | None] = []
    old_radii: list[float | None] = []
    ratios: list[float | None] = []
    inversion_reasons: list[str] = []
    value_bound = reward_bound / (1.0 - gamma)
    for count in counts:
        if count == 0:
            radii.append(None)
            stitch_radii.append(None)
            old_radii.append(None)
            ratios.append(None)
            continue
        try:
            if count not in cache:
                cache[count] = solve_mixture_boundary(
                    count, grid, n_groups=n_groups, delta=delta
                )
            q_mix = float(cache[count]["boundary"])
            new_radius: float | None = value_bound * q_mix / count
        except MixtureInversionError as error:
            new_radius = None
            inversion_reasons.append(error.reason)
        audit_radius = value_bound * stitch_boundary(count, grid) / count
        old_radius = old_visit_indexed_radius(
            count,
            reward_bound=reward_bound,
            gamma=gamma,
            n_groups=n_groups,
            horizon=horizon,
            delta=delta,
        )
        radii.append(new_radius)
        stitch_radii.append(audit_radius)
        old_radii.append(old_radius)
        ratios.append(None if new_radius is None else new_radius / old_radius)
    visited = [float(value) for value in radii if value is not None]
    full_support = all(count > 0 for count in counts) and not inversion_reasons
    result = {
        "counts": list(counts),
        "missing_groups": sum(count == 0 for count in counts),
        "full_support": full_support,
        "min_count": min(counts),
        "min_visited_count": min(count for count in counts if count > 0),
        "radius_by_group": radii,
        "stitch_radius_by_group": stitch_radii,
        "legacy_radius_by_group": old_radii,
        "mixture_to_legacy_ratio_by_group": ratios,
        "max_radius": max(visited) if full_support else None,
        "max_visited_radius": max(visited) if visited else None,
    }
    if inversion_reasons:
        result["inversion_failure_reasons"] = _canonical_reasons(inversion_reasons)
    return result


def _replace_uniform_radius(
    route: dict[str, Any],
    residual_radius: float | None,
    inversion_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    result = copy.deepcopy(route)
    result["residual_radius"] = residual_radius
    if inversion_reasons:
        result.update(
            {
                "rho": None,
                "residual_radius": None,
                "optimization_term": None,
                "statistical_term": None,
                "total_bound": None,
                "finite_bound_emitted": False,
                "improves_over_zero_initialization": False,
                "selective_high_probability_certified": False,
                "status": STATUS_NOT_CERTIFIED,
                "failure_reasons": _canonical_reasons(
                    [*result["failure_reasons"], *inversion_reasons]
                ),
            }
        )
        return result
    if not bool(result["selective_high_probability_certified"]):
        return result
    if residual_radius is None:
        raise ValueError("emitted route cannot have a missing residual radius")
    rho = float(result["rho"])
    iterations = int(result["iterations"])
    initial_error = float(result["initial_error"])
    margin = float(result["margin"])
    optimization = rho**iterations * initial_error
    one_minus_power = 0.0 if iterations == 0 else -math.expm1(iterations * math.log(rho))
    statistical = one_minus_power * residual_radius / (2.0 * margin)
    total = optimization + statistical
    if not all(math.isfinite(value) for value in (optimization, statistical, total)):
        raise ValueError("numerical_nonfinite")
    result.update(
        {
            "optimization_term": optimization,
            "statistical_term": statistical,
            "total_bound": total,
            "finite_bound_emitted": True,
            "improves_over_zero_initialization": total < initial_error,
        }
    )
    return result


def _replace_vfirst_radius(
    route: dict[str, Any],
    state_bound: dict[str, Any],
    recovery_radius: float | None,
    *,
    gamma: float,
    inversion_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    result = copy.deepcopy(route)
    result["value_total_bound"] = state_bound["total_bound"]
    result["fixed_recovery"] = recovery_radius
    new_reasons = _canonical_reasons(
        [
            *result["failure_reasons"],
            *state_bound["failure_reasons"],
            *inversion_reasons,
        ]
    )
    if new_reasons != result["failure_reasons"]:
        result.update(
            {
                "value_propagation": None,
                "fixed_recovery": None,
                "total_bound": None,
                "finite_bound_emitted": False,
                "improves_over_zero_initialization": False,
                "selective_high_probability_certified": False,
                "status": STATUS_NOT_CERTIFIED,
                "failure_reasons": new_reasons,
            }
        )
        return result
    if not bool(result["selective_high_probability_certified"]):
        return result
    if state_bound["total_bound"] is None or recovery_radius is None:
        raise ValueError("emitted V-first route cannot have a missing radius")
    propagation = gamma * float(state_bound["total_bound"])
    total = propagation + recovery_radius + float(result["softmax_leakage"])
    if not all(math.isfinite(value) for value in (propagation, total)):
        raise ValueError("numerical_nonfinite")
    result.update(
        {
            "value_propagation": propagation,
            "total_bound": total,
            "finite_bound_emitted": True,
            "improves_over_zero_initialization": total < float(result["initial_error"])
            if "initial_error" in result
            else total < float(state_bound["initial_error"]),
        }
    )
    return result


def build_time_uniform_certificate(**arguments: Any) -> dict[str, Any]:
    """Build all frozen routes while preserving legacy emission decisions."""
    old = build_visit_indexed_certificate(**arguments)
    inputs = old["certificate_inputs"]
    observed = inputs["observed"]
    declared = inputs["declared"]
    event_old = old["event"]
    n_groups = int(event_old["n_groups"])
    delta = float(declared["delta"])
    reward_bound = float(declared["reward_bound"])
    gamma = float(declared["gamma"])
    horizon = int(declared["trajectory_length"])
    grid = build_mixture_grid(n_groups=n_groups, delta=delta)
    cache: dict[int, dict[str, float | int]] = {}
    state_family = _family_summary(
        observed["state_counts"],
        reward_bound=reward_bound,
        gamma=gamma,
        n_groups=n_groups,
        horizon=horizon,
        delta=delta,
        grid=grid,
        cache=cache,
    )
    pair_family = _family_summary(
        observed["pair_counts"],
        reward_bound=reward_bound,
        gamma=gamma,
        n_groups=n_groups,
        horizon=horizon,
        delta=delta,
        grid=grid,
        cache=cache,
    )
    state_inversion = state_family.get("inversion_failure_reasons", [])
    pair_inversion = pair_family.get("inversion_failure_reasons", [])
    state_exact = _replace_uniform_radius(
        old["state_value"]["exact"], state_family["max_radius"], state_inversion
    )
    state_softmax = _replace_uniform_radius(
        old["state_value"]["softmax"], state_family["max_radius"], state_inversion
    )
    direct_exact = _replace_uniform_radius(
        old["routes"]["direct_exact"], pair_family["max_radius"], pair_inversion
    )
    direct_softmax = _replace_uniform_radius(
        old["routes"]["direct_softmax"], pair_family["max_radius"], pair_inversion
    )
    vfirst_exact = _replace_vfirst_radius(
        old["routes"]["vfirst_nosplit_exact"],
        state_exact,
        pair_family["max_radius"],
        gamma=gamma,
        inversion_reasons=pair_inversion,
    )
    vfirst_softmax = _replace_vfirst_radius(
        old["routes"]["vfirst_nosplit_softmax"],
        state_softmax,
        pair_family["max_radius"],
        gamma=gamma,
        inversion_reasons=pair_inversion,
    )
    event = {
        "method": "finite_geometric_time_uniform_cosh_mixture",
        "probability_statement": event_old["probability_statement"],
        "delta": delta,
        "horizon": horizon,
        "n_states": event_old["n_states"],
        "n_pairs": event_old["n_pairs"],
        "n_groups": n_groups,
        "group_accounting": event_old["group_accounting"],
        "value_bound": event_old["value_bound"],
        "conditional_interval_width": event_old["conditional_interval_width"],
        "absolute_residual_bound": event_old["absolute_residual_bound"],
        "mixture": {
            "components": DEFAULT_COMPONENTS,
            "grid": grid,
            "threshold": n_groups / delta,
            "target_log_level": math.log(n_groups / delta),
            "tolerance": DEFAULT_TOLERANCE,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
            "reported_boundary": "mixture_root",
            "stitch_role": "audit_and_upper_bracket_only",
        },
        "risk_allocation": {
            "allocation_time": "before observing trajectory counts",
            "per_group_failure_probability": delta / n_groups,
            "groups": n_groups,
            "total_failure_probability": delta,
            "route_level_resplit": False,
            "posthoc_minimum_selected": False,
        },
        "state_bellman": state_family,
        "pair_bellman": pair_family,
        "recovery": copy.deepcopy(pair_family),
    }
    return {
        "certificate_inputs": copy.deepcopy(inputs),
        "event": event,
        "state_value": {"exact": state_exact, "softmax": state_softmax},
        "routes": {
            "direct_exact": direct_exact,
            "direct_softmax": direct_softmax,
            "vfirst_nosplit_exact": vfirst_exact,
            "vfirst_nosplit_softmax": vfirst_softmax,
        },
        "optional_variance_adaptive": {
            "status": "feasibility_gap_no_uniform_observable_variance_certificate",
            "selected": False,
            "delta_allocated": 0.0,
            "affects_mandatory_status": False,
            "reason": (
                "no proved observable transition confidence optimization uniformly "
                "covers every V in [-B,B]^m, including unseen successors"
            ),
        },
    }
