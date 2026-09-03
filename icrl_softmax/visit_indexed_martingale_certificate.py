"""Visit-indexed martingale certificates for frozen fixed-policy trajectories.

Pure validation, risk allocation, radius, emission, and route-composition
functions.  This module has no sampling, plotting, file-writing, or
true-model dependency: certificate inputs are observed counts, observed
empirical kernel diagonals, the declared reward bound, and public
hyperparameters only.  See
``docs/research_branches/FP-MART-001/claude/theory.md`` for the proof record.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


STATUS_SELECTIVE = "selective_high_probability_certified"
STATUS_NOT_EMITTED = "not_emitted"

_FAILURE_ORDER = (
    "risk_budget_invalid",
    "algorithm_mode_mismatch",
    "divergence_guard_triggered",
    "missing_support",
    "state_kernel_margin_nonpositive",
    "pair_kernel_margin_nonpositive",
    "variance_adaptive_unavailable",
    "numerical_nonfinite",
)
_FAILURE_RANK = {reason: rank for rank, reason in enumerate(_FAILURE_ORDER)}


def _ordered(reasons: Sequence[str]) -> list[str]:
    unique = set(str(reason) for reason in reasons)
    return sorted(
        unique,
        key=lambda reason: (_FAILURE_RANK.get(reason, len(_FAILURE_ORDER)), reason),
    )


def _status(reasons: Sequence[str]) -> str:
    return STATUS_NOT_EMITTED if reasons else STATUS_SELECTIVE


def _validate_structure(
    gamma: float,
    alpha: float,
    iterations_used: int,
) -> tuple[float, float, int]:
    gamma_value = float(gamma)
    alpha_value = float(alpha)
    iterations = int(iterations_used)
    if not math.isfinite(gamma_value) or not 0.0 < gamma_value < 1.0:
        raise ValueError("gamma must lie in (0,1)")
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value <= 1.0:
        raise ValueError("alpha must lie in (0,1]")
    if iterations != iterations_used or iterations < 0:
        raise ValueError("iterations_used must be a nonnegative integer")
    return gamma_value, alpha_value, iterations


def _one_minus_power(rho: float, iterations: int) -> float:
    if iterations == 0:
        return 0.0
    if rho == 0.0:
        return 1.0
    return -math.expm1(iterations * math.log(rho))


def hoeffding_radius(
    count: int,
    *,
    n_groups: int,
    trajectory_length: int,
    delta: float,
    value_bound: float,
) -> float:
    """Simultaneous visit-indexed radius ``B*sqrt(2*log(2*G*n/delta)/k)``.

    ``count`` is any deterministic visit count in ``[1, n]``; the risk
    allocation over ``G`` groups and ``n`` counts is fixed before seeing data,
    so substituting an observed (random) count needs no reallocation.
    """
    k = int(count)
    groups = int(n_groups)
    length = int(trajectory_length)
    confidence = float(delta)
    bound = float(value_bound)
    if k != count or not 1 <= k <= length:
        raise ValueError("count must be an integer in [1, trajectory_length]")
    if groups != n_groups or groups <= 0:
        raise ValueError("n_groups must be a positive integer")
    if length != trajectory_length or length <= 0:
        raise ValueError("trajectory_length must be a positive integer")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if not math.isfinite(bound) or bound < 0.0:
        raise ValueError("value_bound must be nonnegative and finite")
    return bound * math.sqrt(
        2.0 * math.log(2.0 * groups * length / confidence) / k
    )


def support_summary(counts: Sequence[int]) -> dict[str, Any]:
    """Summarize observed visit counts for one residual family."""
    observed = [int(count) for count in counts]
    if not observed or any(count < 0 for count in observed):
        raise ValueError("counts must be a nonempty list of nonnegative integers")
    missing = sum(1 for count in observed if count == 0)
    return {
        "min_count": min(observed),
        "missing": missing,
        "full_support": missing == 0,
    }


def shared_visit_event(
    *,
    trajectory_length: int,
    n_states: int,
    n_pairs: int,
    state_counts: Sequence[int],
    pair_counts: Sequence[int],
    delta: float,
    value_bound: float,
    fixed_context: bool = True,
    synchronous_update: bool = True,
) -> dict[str, Any]:
    """Build the simultaneous visit-indexed event from observed counts only."""
    length = int(trajectory_length)
    states = int(n_states)
    pairs = int(n_pairs)
    if length != trajectory_length or length <= 0:
        raise ValueError("trajectory_length must be a positive integer")
    if states != n_states or states <= 0:
        raise ValueError("n_states must be a positive integer")
    if pairs != n_pairs or pairs <= 0:
        raise ValueError("n_pairs must be a positive integer")
    state_observed = [int(count) for count in state_counts]
    pair_observed = [int(count) for count in pair_counts]
    if len(state_observed) != states or len(pair_observed) != pairs:
        raise ValueError("count vectors must match n_states and n_pairs")
    if any(count < 0 for count in state_observed + pair_observed):
        raise ValueError("counts must be nonnegative")

    groups = states + 2 * pairs
    confidence = float(delta)
    bound = float(value_bound)
    reasons: list[str] = []
    risk_valid = math.isfinite(confidence) and 0.0 < confidence < 1.0
    if not risk_valid:
        reasons.append("risk_budget_invalid")
    if not fixed_context or not synchronous_update:
        reasons.append("algorithm_mode_mismatch")

    state_support = support_summary(state_observed)
    pair_support = support_summary(pair_observed)
    log_factor: float | None = None
    state_radius: float | None = None
    pair_radius: float | None = None
    if risk_valid:
        log_factor = math.log(2.0 * groups * length / confidence)
        if not math.isfinite(bound) or bound < 0.0:
            reasons.append("numerical_nonfinite")
        else:
            if state_support["full_support"]:
                state_radius = bound * math.sqrt(
                    2.0 * log_factor / state_support["min_count"]
                )
            if pair_support["full_support"]:
                pair_radius = bound * math.sqrt(
                    2.0 * log_factor / pair_support["min_count"]
                )
            for radius in (state_radius, pair_radius):
                if radius is not None and not math.isfinite(radius):
                    reasons.append("numerical_nonfinite")
                    state_radius = None
                    pair_radius = None
                    break

    ordered = _ordered(reasons)
    return {
        "trajectory_length": length,
        "n_states": states,
        "n_pairs": pairs,
        "n_groups": groups,
        "delta": confidence,
        "value_bound": bound,
        "log_factor": log_factor,
        "state_counts": state_observed,
        "pair_counts": pair_observed,
        "state_min_count": state_support["min_count"],
        "pair_min_count": pair_support["min_count"],
        "state_missing": state_support["missing"],
        "pair_missing": pair_support["missing"],
        "state_full_support": state_support["full_support"],
        "pair_full_support": pair_support["full_support"],
        "state_radius": state_radius,
        "pair_radius": pair_radius,
        "assumptions": {
            "fixed_context": bool(fixed_context),
            "synchronous_update": bool(synchronous_update),
            "zero_initialization": True,
        },
        "event_valid": not ordered,
        "failure_reasons": ordered,
    }


def _route_terms(
    rho: float,
    iterations: int,
    initial_error: float,
    residual_radius: float,
    margin: float,
) -> tuple[float, float, float] | None:
    optimization = (rho**iterations) * initial_error
    statistical = (
        _one_minus_power(rho, iterations) * residual_radius / (2.0 * margin)
    )
    total = optimization + statistical
    if not all(math.isfinite(term) for term in (optimization, statistical, total)):
        return None
    return optimization, statistical, total


def _kernel_margin(
    matching: str,
    kernel_diagonal_min: float | None,
    beta: float | None,
    gamma: float,
    margin_failure: str,
    reasons: list[str],
) -> tuple[float | None, float | None]:
    if matching == "exact":
        return 1.0, (1.0 - gamma) / 2.0
    if beta is None:
        raise ValueError("softmax matching requires beta")
    beta_value = float(beta)
    if not math.isfinite(beta_value):
        raise ValueError("beta must be finite")
    if kernel_diagonal_min is None:
        reasons.append("algorithm_mode_mismatch")
        return None, None
    diagonal = float(kernel_diagonal_min)
    if not math.isfinite(diagonal):
        reasons.append("numerical_nonfinite")
        return None, None
    if not 0.0 <= diagonal <= 1.0:
        raise ValueError("kernel_diagonal_min must lie in [0,1]")
    margin = diagonal - (1.0 + gamma) / 2.0
    if margin <= 0.0:
        reasons.append(margin_failure)
        return diagonal, None
    return diagonal, margin


def _uniform_route_certificate(
    route: str,
    matching: str,
    event_reasons: list[str],
    full_support: bool,
    residual_radius: float | None,
    initial_error_bound: float,
    margin_failure: str,
    *,
    gamma: float,
    alpha: float,
    iterations_used: int,
    beta: float | None,
    kernel_diagonal_min: float | None,
    divergence_guard_triggered: bool,
) -> dict[str, Any]:
    gamma_value, alpha_value, iterations = _validate_structure(
        gamma, alpha, iterations_used
    )
    if matching not in {"exact", "softmax"}:
        raise ValueError("matching must be exact or softmax")

    reasons = list(event_reasons)
    if divergence_guard_triggered:
        reasons.append("divergence_guard_triggered")
    if not full_support:
        reasons.append("missing_support")

    diagonal: float | None = None
    margin: float | None = None
    rho: float | None = None
    optimization: float | None = None
    statistical: float | None = None
    total: float | None = None

    diagonal, margin = _kernel_margin(
        matching,
        kernel_diagonal_min,
        beta,
        gamma_value,
        margin_failure,
        reasons,
    )
    if not _ordered(reasons) and margin is not None:
        rho = 1.0 - 2.0 * alpha_value * margin
        if not 0.0 <= rho < 1.0:
            reasons.append("numerical_nonfinite")
            rho = None
        else:
            terms = _route_terms(
                rho,
                iterations,
                initial_error_bound,
                float(residual_radius),
                margin,
            )
            if terms is None:
                reasons.append("numerical_nonfinite")
                rho = None
            else:
                optimization, statistical, total = terms

    ordered = _ordered(reasons)
    if ordered:
        optimization = statistical = total = None
        rho = None
    return {
        "route": route,
        "matching": matching,
        "beta": None if matching == "exact" else float(beta),
        "kernel_diagonal_min": diagonal,
        "margin": margin,
        "rho": rho,
        "iterations": iterations,
        "initial_error_bound": initial_error_bound,
        "residual_radius": residual_radius,
        "optimization_term": optimization,
        "statistical_term": statistical,
        "total_bound": total,
        "status": _status(ordered),
        "failure_reasons": ordered,
    }


def direct_q_certificate(
    event: dict[str, Any],
    *,
    matching: str,
    gamma: float,
    alpha: float,
    iterations_used: int,
    beta: float | None = None,
    kernel_diagonal_min: float | None = None,
    divergence_guard_triggered: bool = False,
) -> dict[str, Any]:
    """Direct-Q all-layer certificate from the visit-indexed pair radius."""
    return _uniform_route_certificate(
        f"direct_{matching}",
        matching,
        list(event.get("failure_reasons", [])),
        bool(event["pair_full_support"]),
        event["pair_radius"],
        float(event["value_bound"]),
        "pair_kernel_margin_nonpositive",
        gamma=gamma,
        alpha=alpha,
        iterations_used=iterations_used,
        beta=beta,
        kernel_diagonal_min=kernel_diagonal_min,
        divergence_guard_triggered=divergence_guard_triggered,
    )


def state_value_certificate(
    event: dict[str, Any],
    *,
    matching: str,
    gamma: float,
    alpha: float,
    iterations_used: int,
    beta: float | None = None,
    kernel_diagonal_min: float | None = None,
) -> dict[str, Any]:
    """State-value all-layer certificate from the visit-indexed state radius."""
    return _uniform_route_certificate(
        f"state_{matching}",
        matching,
        list(event.get("failure_reasons", [])),
        bool(event["state_full_support"]),
        event["state_radius"],
        float(event["value_bound"]),
        "state_kernel_margin_nonpositive",
        gamma=gamma,
        alpha=alpha,
        iterations_used=iterations_used,
        beta=beta,
        kernel_diagonal_min=kernel_diagonal_min,
        divergence_guard_triggered=False,
    )


def vfirst_nosplit_certificate(
    event: dict[str, Any],
    state_value: dict[str, Any],
    *,
    recovery_matching: str,
    gamma: float,
    beta: float | None = None,
    recovery_diagonal_min: float | None = None,
) -> dict[str, Any]:
    """Compose the state stage, ghost recovery radius, and softmax leakage."""
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or not 0.0 < gamma_value < 1.0:
        raise ValueError("gamma must lie in (0,1)")
    if recovery_matching not in {"exact", "softmax"}:
        raise ValueError("recovery_matching must be exact or softmax")
    bound = float(event["value_bound"])

    reasons = list(event.get("failure_reasons", []))
    reasons.extend(str(reason) for reason in state_value.get("failure_reasons", []))
    if not bool(event["pair_full_support"]):
        reasons.append("missing_support")

    diagonal: float | None = None
    leakage: float | None = None
    propagation: float | None = None
    total: float | None = None

    if recovery_matching == "exact":
        diagonal = 1.0
        leakage = 0.0
    else:
        if beta is None:
            raise ValueError("softmax recovery requires beta")
        if recovery_diagonal_min is None:
            reasons.append("algorithm_mode_mismatch")
        else:
            diagonal = float(recovery_diagonal_min)
            if not math.isfinite(diagonal):
                reasons.append("numerical_nonfinite")
                diagonal = None
            elif not 0.0 <= diagonal <= 1.0:
                raise ValueError("recovery_diagonal_min must lie in [0,1]")
            else:
                leakage = 2.0 * bound * (1.0 - diagonal)

    value_total = state_value.get("total_bound")
    recovery_radius = event["pair_radius"]
    if not _ordered(reasons):
        if value_total is None or recovery_radius is None:
            reasons.append("numerical_nonfinite")
        else:
            propagation = gamma_value * float(value_total)
            total = propagation + float(recovery_radius) + float(leakage)
            if not all(
                math.isfinite(term) for term in (propagation, float(leakage), total)
            ):
                reasons.append("numerical_nonfinite")
                propagation = leakage = total = None

    ordered = _ordered(reasons)
    if ordered:
        propagation = leakage = total = None
    return {
        "route": f"vfirst_nosplit_{recovery_matching}",
        "matching": recovery_matching,
        "beta": None if recovery_matching == "exact" else float(beta),
        "recovery_kernel_diagonal_min": diagonal,
        "value_total_bound": value_total,
        "value_propagation": propagation,
        "recovery_radius": recovery_radius,
        "softmax_leakage": leakage,
        "total_bound": total,
        "status": _status(ordered),
        "failure_reasons": ordered,
    }


def variance_adaptive_certificate(delta_share: float) -> dict[str, Any]:
    """Optional variance-adaptive constituent.

    The conditional variance of the fixed-target residuals is not observable
    from the trajectory without true values or the true kernel, so no
    oracle-free Freedman/empirical-Bernstein radius exists in this contract.
    The constituent is deterministically unavailable; it is never combined
    with the Hoeffding radius by a post-hoc minimum.
    """
    share = float(delta_share)
    reasons: list[str] = []
    if not math.isfinite(share) or not 0.0 < share < 1.0:
        reasons.append("risk_budget_invalid")
    reasons.append("variance_adaptive_unavailable")
    return {
        "route": "variance_adaptive",
        "preallocated_delta_share": share if math.isfinite(share) else None,
        "radius": None,
        "status": "unavailable",
        "failure_reasons": _ordered(reasons),
        "explanation": (
            "fixed-target conditional variance is not observable without "
            "oracle quantities; Popoviciu recovers the Hoeffding radius"
        ),
    }
