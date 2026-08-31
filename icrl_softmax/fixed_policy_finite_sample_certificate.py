"""Pure finite-sample certificates for frozen-context policy evaluation.

This module deliberately has no sampling, plotting, or file-writing side
effects.  It turns exact-model Markov diagnostics into a shared concentration
event and then composes route-specific deterministic recurrences on that event.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


STATUS_HIGH_PROBABILITY = "high_probability_certified"
STATUS_PATHWISE = "pathwise_bound_verified"
STATUS_DIAGNOSTIC = "diagnostic_only"
STATUS_NOT_CERTIFIED = "not_certified"

_FAILURE_ORDER = (
    "numerical_support_truncated",
    "spectral_condition_failed",
    "algorithm_mode_mismatch",
    "divergence_guard_triggered",
    "state_coverage_failed",
    "pair_coverage_failed",
    "state_kernel_margin_nonpositive",
    "pair_kernel_margin_nonpositive",
    "stationary_start_failed",
    "reward_model_mismatch",
    "numerical_nonfinite",
)
_FAILURE_RANK = {reason: rank for rank, reason in enumerate(_FAILURE_ORDER)}


def strict_json_ready(value: Any) -> Any:
    """Recursively convert values to strict-JSON-safe Python objects.

    Non-finite floating-point values are unavailable numerical fields and are
    represented by ``None``.  Callers should still attach a failure reason when
    a theorem computation unexpectedly becomes non-finite.
    """
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


def _canonical_reasons(reasons: Sequence[str]) -> list[str]:
    unique = set(reasons)
    return sorted(
        unique,
        key=lambda reason: (_FAILURE_RANK.get(reason, len(_FAILURE_ORDER)), reason),
    )


def _status(reasons: Sequence[str], evidence_level: str) -> str:
    if reasons:
        return STATUS_NOT_CERTIFIED
    mapping = {
        "high_probability": STATUS_HIGH_PROBABILITY,
        "pathwise": STATUS_PATHWISE,
        "diagnostic": STATUS_DIAGNOSTIC,
    }
    if evidence_level not in mapping:
        raise ValueError(
            "evidence_level must be high_probability, pathwise, or diagnostic"
        )
    return mapping[evidence_level]


def _finite_number(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_common_bound_inputs(
    gamma: float,
    alpha: float,
    iterations_used: int,
    initial_error: float,
) -> tuple[float, float, int, float]:
    gamma_value = _finite_number("gamma", gamma)
    alpha_value = _finite_number("alpha", alpha)
    initial = _finite_number("initial_error", initial_error)
    iterations = int(iterations_used)
    if not 0.0 < gamma_value < 1.0:
        raise ValueError("gamma must lie in (0,1)")
    if not 0.0 < alpha_value <= 1.0:
        raise ValueError("alpha must lie in (0,1]")
    if iterations != iterations_used or iterations < 0:
        raise ValueError("iterations_used must be a nonnegative integer")
    if initial < 0.0:
        raise ValueError("initial_error must be nonnegative")
    return gamma_value, alpha_value, iterations, initial


def _one_minus_power(rho: float, iterations: int) -> float:
    if iterations == 0:
        return 0.0
    if rho == 0.0:
        return 1.0
    return -math.expm1(iterations * math.log(rho))


def right_hoeffding_inflation(lambda_right: float) -> float:
    """Return the stationary right-Hoeffding variance inflation.

    The right spectral value is the nontrivial top eigenvalue of the additive
    reversiblization.  Negative values receive no inflation.
    """
    spectral_value = _finite_number("lambda_right", lambda_right)
    if spectral_value < -1.0 - 1e-12 or spectral_value >= 1.0:
        raise ValueError("lambda_right must lie in [-1,1)")
    positive = max(spectral_value, 0.0)
    return (1.0 + positive) / (1.0 - positive)


def shared_event_radii(
    *,
    trajectory_length: int,
    n_states: int,
    n_pairs: int,
    delta: float,
    value_bound: float,
    state_stationary_min: float,
    pair_stationary_min: float,
    state_right_inflation: float,
    pair_right_inflation: float,
    edge_right_inflation: float,
    numerical_support_truncated: bool = False,
    stationary_start: bool = True,
    edge_determined_reward: bool = True,
    fixed_context: bool = True,
    synchronous_update: bool = True,
) -> dict[str, Any]:
    """Build the one-event radii shared by all fixed-policy routes."""
    length = int(trajectory_length)
    states = int(n_states)
    pairs = int(n_pairs)
    confidence = _finite_number("delta", delta)
    bound = _finite_number("value_bound", value_bound)
    mu_state = _finite_number("state_stationary_min", state_stationary_min)
    mu_pair = _finite_number("pair_stationary_min", pair_stationary_min)
    if length != trajectory_length or length <= 0:
        raise ValueError("trajectory_length must be a positive integer")
    if states != n_states or states <= 0:
        raise ValueError("n_states must be a positive integer")
    if pairs != n_pairs or pairs <= 0:
        raise ValueError("n_pairs must be a positive integer")
    if not 0.0 < confidence < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if bound < 0.0:
        raise ValueError("value_bound must be nonnegative")
    if not 0.0 < mu_state <= 1.0 or not 0.0 < mu_pair <= 1.0:
        raise ValueError("stationary minima must lie in (0,1]")

    inflations = (
        float(state_right_inflation),
        float(pair_right_inflation),
        float(edge_right_inflation),
    )
    spectral_ok = all(
        math.isfinite(inflation) and inflation >= 1.0
        for inflation in inflations
    )
    n_functions = 2 * states + 3 * pairs
    log_factor = math.log(2.0 * n_functions) - math.log(confidence)
    reasons: list[str] = []
    if numerical_support_truncated:
        reasons.append("numerical_support_truncated")
    if not spectral_ok:
        reasons.append("spectral_condition_failed")
    if not stationary_start:
        reasons.append("stationary_start_failed")
    if not edge_determined_reward:
        reasons.append("reward_model_mismatch")
    if not fixed_context or not synchronous_update:
        reasons.append("algorithm_mode_mismatch")

    b_state: float | None = None
    b_pair: float | None = None
    edge_radius: float | None = None
    u_state: float | None = None
    u_pair: float | None = None
    epsilon_state: float | None = None
    epsilon_pair: float | None = None
    if spectral_ok:
        b_state = math.sqrt(inflations[0] * log_factor / (2.0 * length))
        b_pair = math.sqrt(inflations[1] * log_factor / (2.0 * length))
        edge_radius = bound * math.sqrt(
            2.0 * inflations[2] * log_factor / length
        )
        u_state = mu_state - b_state
        u_pair = mu_pair - b_pair
        if u_state <= 0.0:
            reasons.append("state_coverage_failed")
        else:
            epsilon_state = edge_radius / u_state
        if u_pair <= 0.0:
            reasons.append("pair_coverage_failed")
        else:
            epsilon_pair = edge_radius / u_pair
        finite_outputs = (
            b_state,
            b_pair,
            edge_radius,
            u_state,
            u_pair,
            epsilon_state if epsilon_state is not None else 0.0,
            epsilon_pair if epsilon_pair is not None else 0.0,
        )
        if not all(math.isfinite(number) for number in finite_outputs):
            reasons.append("numerical_nonfinite")
            b_state = b_pair = edge_radius = None
            u_state = u_pair = None
            epsilon_state = epsilon_pair = None

    ordered_reasons = _canonical_reasons(reasons)
    return {
        "assumptions": {
            "stationary_start": bool(stationary_start),
            "fixed_context": bool(fixed_context),
            "edge_determined_reward": bool(edge_determined_reward),
            "synchronous_update": bool(synchronous_update),
            "full_target_support": bool(
                u_state is not None
                and u_state > 0.0
                and u_pair is not None
                and u_pair > 0.0
            ),
        },
        "delta": confidence,
        "M": n_functions,
        "B": bound,
        "L_delta": log_factor,
        "omega_S": inflations[0] if math.isfinite(inflations[0]) else None,
        "omega_X": inflations[1] if math.isfinite(inflations[1]) else None,
        "omega_E": inflations[2] if math.isfinite(inflations[2]) else None,
        "mu_S_min": mu_state,
        "mu_X_min": mu_pair,
        "b_S": b_state,
        "b_X": b_pair,
        "e_E": edge_radius,
        "u_S": u_state,
        "u_X": u_pair,
        "epsilon_S": epsilon_state,
        "epsilon_X": epsilon_pair,
        "state_coverage_certified": bool(u_state is not None and u_state > 0.0),
        "pair_coverage_certified": bool(u_pair is not None and u_pair > 0.0),
        "spectral_condition_certified": spectral_ok,
        "numerical_support_truncated": bool(numerical_support_truncated),
        "status": _status(ordered_reasons, "high_probability"),
        "failure_reasons": ordered_reasons,
    }


def one_hot_diagonal_lower_bound(
    occupancy_lower_bound: float | None,
    beta: float,
) -> float | None:
    """Evaluate m_beta(u) stably, returning None outside u in (0,1]."""
    if occupancy_lower_bound is None:
        return None
    occupancy = float(occupancy_lower_bound)
    if not math.isfinite(occupancy):
        return None
    if occupancy <= 0.0:
        return None
    if occupancy > 1.0 + 1e-12:
        raise ValueError("occupancy lower bound cannot exceed one")
    occupancy = min(occupancy, 1.0)
    sharpness = _finite_number("beta", beta)
    if occupancy == 1.0:
        return 1.0
    logit = sharpness + math.log(occupancy) - math.log1p(-occupancy)
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def _event_reasons(
    event: Mapping[str, Any],
    relevant_coverage: str,
) -> list[str]:
    retained = {
        "numerical_support_truncated",
        "spectral_condition_failed",
        "stationary_start_failed",
        "reward_model_mismatch",
        "algorithm_mode_mismatch",
        "numerical_nonfinite",
        relevant_coverage,
    }
    return [
        str(reason)
        for reason in event.get("failure_reasons", [])
        if reason in retained
    ]


def _uniform_bound_terms(
    rho: float,
    iterations: int,
    initial_error: float,
    residual_radius: float,
    margin: float,
) -> tuple[float, float, float] | None:
    optimization = (rho ** iterations) * initial_error
    statistical = (
        _one_minus_power(rho, iterations)
        * residual_radius
        / (2.0 * margin)
    )
    total = optimization + statistical
    if not all(math.isfinite(number) for number in (optimization, statistical, total)):
        return None
    return optimization, statistical, total


def direct_q_uniform_bound(
    shared_event: Mapping[str, Any],
    *,
    matching: str,
    gamma: float,
    alpha: float,
    iterations_used: int,
    initial_error: float,
    beta: float | None = None,
    missing_pairs: int = 0,
    full_target_support: bool = True,
    fixed_context: bool = True,
    synchronous_update: bool = True,
    divergence_guard_triggered: bool = False,
    evidence_level: str = "high_probability",
    residual_bound: float | None = None,
    kernel_diagonal_lower_bound: float | None = None,
) -> dict[str, Any]:
    """Return the all-layer Direct-Q bound on a frozen empirical operator."""
    gamma_value, alpha_value, iterations, initial = _validate_common_bound_inputs(
        gamma, alpha, iterations_used, initial_error
    )
    if matching not in {"exact", "softmax"}:
        raise ValueError("matching must be exact or softmax")
    missing = int(missing_pairs)
    if missing != missing_pairs or missing < 0:
        raise ValueError("missing_pairs must be a nonnegative integer")

    reasons = (
        _event_reasons(shared_event, "pair_coverage_failed")
        if evidence_level == "high_probability"
        else []
    )
    if not fixed_context or not synchronous_update:
        reasons.append("algorithm_mode_mismatch")
    if divergence_guard_triggered:
        reasons.append("divergence_guard_triggered")
    if missing > 0 or not full_target_support:
        reasons.append("pair_coverage_failed")

    occupancy = shared_event.get("u_X")
    residual = (
        shared_event.get("epsilon_X")
        if evidence_level == "high_probability"
        else residual_bound
    )
    if evidence_level == "high_probability" and (
        occupancy is None or float(occupancy) <= 0.0 or residual is None
    ):
        reasons.append("pair_coverage_failed")
    if evidence_level != "high_probability" and residual is None:
        reasons.append(
            "pair_coverage_failed"
            if missing > 0 or not full_target_support
            else "algorithm_mode_mismatch"
        )
    if residual is not None:
        residual = _finite_number("residual_bound", residual)
        if residual < 0.0:
            raise ValueError("residual_bound must be nonnegative")

    diagonal: float | None = None
    margin: float | None = None
    rho: float | None = None
    optimization: float | None = None
    statistical: float | None = None
    total: float | None = None
    route_beta: float | None = None

    blockers = _canonical_reasons(reasons)
    if not blockers:
        if matching == "exact":
            diagonal = 1.0
            margin = (1.0 - gamma_value) / 2.0
        else:
            if kernel_diagonal_lower_bound is not None:
                diagonal = _finite_number(
                    "kernel_diagonal_lower_bound", kernel_diagonal_lower_bound
                )
                if not 0.0 <= diagonal <= 1.0:
                    raise ValueError(
                        "kernel_diagonal_lower_bound must lie in [0,1]"
                    )
                route_beta = None if beta is None else _finite_number("beta", beta)
                margin = diagonal - (1.0 + gamma_value) / 2.0
                if margin <= 0.0:
                    reasons.append("pair_kernel_margin_nonpositive")
            elif beta is None:
                reasons.append("algorithm_mode_mismatch")
            else:
                route_beta = _finite_number("beta", beta)
                diagonal = one_hot_diagonal_lower_bound(
                    None if occupancy is None else float(occupancy), route_beta
                )
                if diagonal is None:
                    reasons.append("pair_coverage_failed")
                else:
                    margin = diagonal - (1.0 + gamma_value) / 2.0
                    if margin <= 0.0:
                        reasons.append("pair_kernel_margin_nonpositive")

    blockers = _canonical_reasons(reasons)
    if not blockers and margin is not None:
        rho = 1.0 - 2.0 * alpha_value * margin
        if not 0.0 <= rho < 1.0:
            reasons.append("numerical_nonfinite")
            rho = None
            margin = None
            diagonal = None
        else:
            terms = _uniform_bound_terms(
                rho, iterations, initial, float(residual), margin
            )
            if terms is None:
                reasons.append("numerical_nonfinite")
                rho = None
                margin = None
                diagonal = None
            else:
                optimization, statistical, total = terms

    ordered_reasons = _canonical_reasons(reasons)
    if ordered_reasons:
        optimization = statistical = total = None
        if "pair_coverage_failed" in ordered_reasons:
            diagonal = margin = rho = None
    return {
        "route": f"direct_{matching}",
        "matching": matching,
        "beta": route_beta,
        "kernel_diagonal_lower_bound": diagonal,
        "margin": margin,
        "rho": rho,
        "rho_Q": rho,
        "iterations": iterations,
        "initial_error": initial,
        "residual_radius": None if residual is None else float(residual),
        "optimization_term": optimization,
        "statistical_term": statistical,
        "total_bound": total,
        "coverage_certified": "pair_coverage_failed" not in ordered_reasons,
        "kernel_margin_certified": bool(
            not ordered_reasons and margin is not None and margin > 0.0
        ),
        "status": _status(ordered_reasons, evidence_level),
        "failure_reasons": ordered_reasons,
    }


def state_value_uniform_bound(
    shared_event: Mapping[str, Any],
    *,
    matching: str,
    gamma: float,
    alpha: float,
    iterations_used: int,
    initial_error: float,
    beta: float | None = None,
    missing_states: int = 0,
    full_target_support: bool = True,
    fixed_context: bool = True,
    synchronous_update: bool = True,
    evidence_level: str = "high_probability",
    residual_bound: float | None = None,
    kernel_diagonal_lower_bound: float | None = None,
) -> dict[str, Any]:
    """Return the all-layer state-value bound, including clipping metadata."""
    gamma_value, alpha_value, iterations, initial = _validate_common_bound_inputs(
        gamma, alpha, iterations_used, initial_error
    )
    if matching not in {"exact", "softmax"}:
        raise ValueError("matching must be exact or softmax")
    missing = int(missing_states)
    if missing != missing_states or missing < 0:
        raise ValueError("missing_states must be a nonnegative integer")

    reasons = (
        _event_reasons(shared_event, "state_coverage_failed")
        if evidence_level == "high_probability"
        else []
    )
    if not fixed_context or not synchronous_update:
        reasons.append("algorithm_mode_mismatch")
    if missing > 0 or not full_target_support:
        reasons.append("state_coverage_failed")
    occupancy = shared_event.get("u_S")
    residual = (
        shared_event.get("epsilon_S")
        if evidence_level == "high_probability"
        else residual_bound
    )
    if evidence_level == "high_probability" and (
        occupancy is None or float(occupancy) <= 0.0 or residual is None
    ):
        reasons.append("state_coverage_failed")
    if evidence_level != "high_probability" and residual is None:
        reasons.append(
            "state_coverage_failed"
            if missing > 0 or not full_target_support
            else "algorithm_mode_mismatch"
        )
    if residual is not None:
        residual = _finite_number("residual_bound", residual)
        if residual < 0.0:
            raise ValueError("residual_bound must be nonnegative")

    diagonal: float | None = None
    margin: float | None = None
    rho: float | None = None
    optimization: float | None = None
    statistical: float | None = None
    total: float | None = None
    route_beta: float | None = None

    blockers = _canonical_reasons(reasons)
    if not blockers:
        if matching == "exact":
            diagonal = 1.0
            margin = (1.0 - gamma_value) / 2.0
        else:
            if kernel_diagonal_lower_bound is not None:
                diagonal = _finite_number(
                    "kernel_diagonal_lower_bound", kernel_diagonal_lower_bound
                )
                if not 0.0 <= diagonal <= 1.0:
                    raise ValueError(
                        "kernel_diagonal_lower_bound must lie in [0,1]"
                    )
                route_beta = None if beta is None else _finite_number("beta", beta)
                margin = diagonal - (1.0 + gamma_value) / 2.0
                if margin <= 0.0:
                    reasons.append("state_kernel_margin_nonpositive")
            elif beta is None:
                reasons.append("algorithm_mode_mismatch")
            else:
                route_beta = _finite_number("beta", beta)
                diagonal = one_hot_diagonal_lower_bound(
                    None if occupancy is None else float(occupancy), route_beta
                )
                if diagonal is None:
                    reasons.append("state_coverage_failed")
                else:
                    margin = diagonal - (1.0 + gamma_value) / 2.0
                    if margin <= 0.0:
                        reasons.append("state_kernel_margin_nonpositive")

    blockers = _canonical_reasons(reasons)
    if not blockers and margin is not None:
        rho = 1.0 - 2.0 * alpha_value * margin
        if not 0.0 <= rho < 1.0:
            reasons.append("numerical_nonfinite")
            rho = None
            margin = None
            diagonal = None
        else:
            terms = _uniform_bound_terms(
                rho, iterations, initial, float(residual), margin
            )
            if terms is None:
                reasons.append("numerical_nonfinite")
                rho = None
                margin = None
                diagonal = None
            else:
                optimization, statistical, total = terms

    ordered_reasons = _canonical_reasons(reasons)
    if ordered_reasons:
        optimization = statistical = total = None
        if "state_coverage_failed" in ordered_reasons:
            diagonal = margin = rho = None
    return {
        "route": f"state_{matching}",
        "matching": matching,
        "beta": route_beta,
        "kernel_diagonal_lower_bound": diagonal,
        "margin": margin,
        "rho": rho,
        "rho_V": rho,
        "iterations": iterations,
        "initial_error": initial,
        "residual_radius": None if residual is None else float(residual),
        "optimization_term": optimization,
        "statistical_term": statistical,
        "total_bound": total,
        "coverage_certified": "state_coverage_failed" not in ordered_reasons,
        "kernel_margin_certified": bool(
            not ordered_reasons and margin is not None and margin > 0.0
        ),
        "clipping_nonexpansive": True,
        "status": _status(ordered_reasons, evidence_level),
        "failure_reasons": ordered_reasons,
    }


def vfirst_nosplit_bound(
    shared_event: Mapping[str, Any],
    state_value_bound: Mapping[str, Any],
    *,
    recovery_matching: str,
    gamma: float,
    value_bound: float,
    beta: float | None = None,
    missing_pairs: int = 0,
    full_target_support: bool = True,
    algorithm_mode: str = "vfirst_nosplit",
    evidence_level: str = "high_probability",
    fixed_recovery_bound: float | None = None,
    recovery_diagonal_lower_bound: float | None = None,
) -> dict[str, Any]:
    """Compose same-sample value propagation, ghost recovery, and leakage."""
    gamma_value = _finite_number("gamma", gamma)
    bound = _finite_number("value_bound", value_bound)
    if not 0.0 < gamma_value < 1.0:
        raise ValueError("gamma must lie in (0,1)")
    if bound < 0.0:
        raise ValueError("value_bound must be nonnegative")
    if recovery_matching not in {"exact", "softmax"}:
        raise ValueError("recovery_matching must be exact or softmax")
    missing = int(missing_pairs)
    if missing != missing_pairs or missing < 0:
        raise ValueError("missing_pairs must be a nonnegative integer")

    reasons = (
        _event_reasons(shared_event, "pair_coverage_failed")
        if evidence_level == "high_probability"
        else []
    )
    reasons.extend(str(reason) for reason in state_value_bound.get("failure_reasons", []))
    if algorithm_mode != "vfirst_nosplit":
        reasons.append("algorithm_mode_mismatch")
    if missing > 0 or not full_target_support:
        reasons.append("pair_coverage_failed")

    occupancy = shared_event.get("u_X")
    if evidence_level == "high_probability" and (
        occupancy is None or float(occupancy) <= 0.0
    ):
        reasons.append("pair_coverage_failed")
    recovery_radius = (
        shared_event.get("epsilon_X")
        if fixed_recovery_bound is None and evidence_level == "high_probability"
        else fixed_recovery_bound
    )
    if recovery_radius is None:
        reasons.append(
            "pair_coverage_failed"
            if evidence_level == "high_probability"
            or missing > 0
            or not full_target_support
            else "algorithm_mode_mismatch"
        )
    else:
        recovery_radius = _finite_number("fixed_recovery_bound", recovery_radius)
        if recovery_radius < 0.0:
            raise ValueError("fixed_recovery_bound must be nonnegative")

    value_total = state_value_bound.get("total_bound")
    if value_total is None and not state_value_bound.get("failure_reasons"):
        reasons.append("numerical_nonfinite")
    route_beta: float | None = None
    diagonal: float | None = None
    leakage: float | None = None
    propagation: float | None = None
    total: float | None = None

    if not _canonical_reasons(reasons):
        if recovery_matching == "exact":
            diagonal = 1.0
            leakage = 0.0
        else:
            if beta is None and recovery_diagonal_lower_bound is None:
                reasons.append("algorithm_mode_mismatch")
            else:
                if beta is not None:
                    route_beta = _finite_number("beta", beta)
                if recovery_diagonal_lower_bound is not None:
                    diagonal = _finite_number(
                        "recovery_diagonal_lower_bound",
                        recovery_diagonal_lower_bound,
                    )
                    if not 0.0 <= diagonal <= 1.0:
                        raise ValueError(
                            "recovery_diagonal_lower_bound must lie in [0,1]"
                        )
                else:
                    diagonal = one_hot_diagonal_lower_bound(
                        None if occupancy is None else float(occupancy),
                        float(route_beta),
                    )
                if diagonal is None:
                    reasons.append("pair_coverage_failed")
                else:
                    leakage = 2.0 * bound * (1.0 - diagonal)

    if not _canonical_reasons(reasons):
        propagation = gamma_value * float(value_total)
        total = propagation + float(recovery_radius) + float(leakage)
        if not all(math.isfinite(number) for number in (propagation, total)):
            reasons.append("numerical_nonfinite")
            propagation = leakage = total = None

    ordered_reasons = _canonical_reasons(reasons)
    if ordered_reasons:
        propagation = leakage = total = None
        if "pair_coverage_failed" in ordered_reasons:
            diagonal = None
    return {
        "route": f"vfirst_nosplit_{recovery_matching}",
        "matching": recovery_matching,
        "beta": route_beta,
        "recovery_kernel_diagonal_lower_bound": diagonal,
        "value_total_bound": None if value_total is None else float(value_total),
        "value_propagation": propagation,
        "fixed_recovery": (
            None if recovery_radius is None else float(recovery_radius)
        ),
        "softmax_leakage": leakage,
        "total_bound": total,
        "consistent_at_fixed_beta": recovery_matching == "exact",
        "status": _status(ordered_reasons, evidence_level),
        "failure_reasons": ordered_reasons,
    }


def _group_means(
    residuals: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, list[float | None], float | None, float | None]:
    counts = np.bincount(groups, minlength=n_groups).astype(np.int64)
    sums = np.bincount(groups, weights=residuals, minlength=n_groups)
    means: list[float | None] = []
    for group in range(n_groups):
        means.append(
            None if counts[group] == 0 else float(sums[group] / counts[group])
        )
    visited = [abs(mean) for mean in means if mean is not None]
    visited_sup = None if not visited else float(max(visited))
    full_sup = visited_sup if np.all(counts > 0) else None
    return counts, means, full_sup, visited_sup


def observed_ghost_residuals(
    *,
    current_states: np.ndarray,
    current_pairs: np.ndarray,
    next_states: np.ndarray,
    next_pairs: np.ndarray,
    rewards: np.ndarray,
    true_value: np.ndarray,
    true_q: np.ndarray,
    gamma: float,
    n_states: int,
    n_pairs: int,
) -> dict[str, Any]:
    """Compute fixed-target empirical residuals for synthetic diagnostics."""
    current_state_array = np.asarray(current_states, dtype=np.int64)
    current_pair_array = np.asarray(current_pairs, dtype=np.int64)
    next_state_array = np.asarray(next_states, dtype=np.int64)
    next_pair_array = np.asarray(next_pairs, dtype=np.int64)
    reward_array = np.asarray(rewards, dtype=np.float64)
    value = np.asarray(true_value, dtype=np.float64)
    q_value = np.asarray(true_q, dtype=np.float64).reshape(-1)
    gamma_value = _finite_number("gamma", gamma)
    states = int(n_states)
    pairs = int(n_pairs)
    arrays = (
        current_state_array,
        current_pair_array,
        next_state_array,
        next_pair_array,
        reward_array,
    )
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("trajectory arrays must be one-dimensional")
    if len({array.size for array in arrays}) != 1 or reward_array.size == 0:
        raise ValueError("trajectory arrays must have the same positive length")
    if value.shape != (states,) or q_value.shape != (pairs,):
        raise ValueError("true_value and true_q must match n_states and n_pairs")
    if not np.all(np.isfinite(reward_array)):
        raise ValueError("rewards must be finite")
    if not np.all(np.isfinite(value)) or not np.all(np.isfinite(q_value)):
        raise ValueError("true targets must be finite")
    if (
        np.min(current_state_array) < 0
        or np.max(current_state_array) >= states
        or np.min(next_state_array) < 0
        or np.max(next_state_array) >= states
        or np.min(current_pair_array) < 0
        or np.max(current_pair_array) >= pairs
        or np.min(next_pair_array) < 0
        or np.max(next_pair_array) >= pairs
    ):
        raise ValueError("state or pair index lies outside its declared support")

    state_residuals = (
        reward_array
        + gamma_value * value[next_state_array]
        - value[current_state_array]
    )
    pair_residuals = (
        reward_array
        + gamma_value * q_value[next_pair_array]
        - q_value[current_pair_array]
    )
    recovery_residuals = (
        reward_array
        + gamma_value * value[next_state_array]
        - q_value[current_pair_array]
    )
    state_counts, state_means, state_sup, state_visited_sup = _group_means(
        state_residuals, current_state_array, states
    )
    pair_counts, pair_means, pair_sup, pair_visited_sup = _group_means(
        pair_residuals, current_pair_array, pairs
    )
    _, recovery_means, recovery_sup, recovery_visited_sup = _group_means(
        recovery_residuals, current_pair_array, pairs
    )
    return {
        "state_counts": state_counts,
        "pair_counts": pair_counts,
        "state_residual_means": state_means,
        "pair_residual_means": pair_means,
        "recovery_residual_means": recovery_means,
        "state_residual_sup": state_sup,
        "pair_residual_sup": pair_sup,
        "recovery_residual_sup": recovery_sup,
        "state_residual_visited_sup": state_visited_sup,
        "pair_residual_visited_sup": pair_visited_sup,
        "recovery_residual_visited_sup": recovery_visited_sup,
        "state_full_support": bool(np.all(state_counts > 0)),
        "pair_full_support": bool(np.all(pair_counts > 0)),
        "missing_states": int(np.sum(state_counts == 0)),
        "missing_pairs": int(np.sum(pair_counts == 0)),
    }
