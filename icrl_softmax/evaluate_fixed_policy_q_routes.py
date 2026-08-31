"""Matched-budget fixed-policy comparison of Direct-Q and V-first routes.

The two main routes share every sampled MDP, fixed policy, start state, and
trajectory. Direct-Q uses the full trajectory. V-first spends the first half
on state-value evaluation and the second half on one-step Q recovery. The
no-split and oracle-V variants are diagnostics, not fairness baselines.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from crossfit_vfirst import crossfit_vfirst_estimate
from fixed_policy_finite_sample_certificate import (
    direct_q_uniform_bound,
    observed_ghost_residuals,
    shared_event_radii,
    state_value_uniform_bound,
    strict_json_ready,
    vfirst_nosplit_bound,
)
from markov_coverage_certificate import (
    build_markov_base_certificate,
    build_markov_coverage_certificate,
)
from mdps import rollout, sample_mdp
from verify_fixed_policy_q_routes import policy_quantities


ROUTE_ORDER = (
    "direct_exact",
    "direct_softmax",
    "vfirst_split_exact",
    "vfirst_split_softmax",
    "vfirst_nosplit_exact",
    "vfirst_nosplit_softmax",
    "vfirst_crossfit_exact",
    "vfirst_crossfit_softmax",
    "vfirst_crossfit_gap_exact",
    "vfirst_crossfit_gap_softmax",
    "vfirst_oracle_exact",
)

ROUTE_LABELS = {
    "direct_exact": "Direct-Q exact",
    "direct_softmax": "Direct-Q softmax",
    "vfirst_split_exact": "V-first split exact",
    "vfirst_split_softmax": "V-first split softmax",
    "vfirst_nosplit_exact": "V-first no-split",
    "vfirst_nosplit_softmax": "V-first no-split softmax",
    "vfirst_crossfit_exact": "V-first cross-fit",
    "vfirst_crossfit_softmax": "V-first cross-fit softmax",
    "vfirst_crossfit_gap_exact": "V-first gap cross-fit",
    "vfirst_crossfit_gap_softmax": "V-first gap cross-fit softmax",
    "vfirst_oracle_exact": "V-first oracle-V",
}


def json_ready(value: Any) -> Any:
    return strict_json_ready(value)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def make_mdp(
    n_states: int,
    n_actions: int,
    gamma: float,
    mixing: float,
    gap_bonus: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sample an ergodic MDP with controllable self-loop persistence."""
    mdp = sample_mdp(n_states, n_actions, gamma, rng=rng)
    random_transition = np.asarray(mdp["P"], dtype=np.float64)
    random_transition /= random_transition.sum(axis=2, keepdims=True)
    sticky = np.zeros_like(random_transition)
    for state in range(n_states):
        sticky[state, :, state] = 1.0
    mdp["P"] = (1.0 - mixing) * sticky + mixing * random_transition

    reward = np.asarray(mdp["R"], dtype=np.float64)
    reward[:, 0, :] += gap_bonus
    mdp["R"] = reward
    mdp["p0"] = np.asarray(mdp["p0"], dtype=np.float64)
    return mdp


def make_policy(
    n_states: int,
    n_actions: int,
    pi_min: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if pi_min <= 0.0 or pi_min >= 1.0 / n_actions:
        raise ValueError("pi_min must lie in (0, 1 / n_actions)")
    policy = np.full((n_states, n_actions), pi_min, dtype=np.float64)
    preferred = rng.integers(0, n_actions, size=n_states)
    policy[np.arange(n_states), preferred] = 1.0 - (n_actions - 1) * pi_min
    return policy


def grouped_exact_mean(
    values: np.ndarray, groups: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(groups, minlength=n_groups).astype(np.int64)
    sums = np.bincount(groups, weights=values, minlength=n_groups)
    means = np.zeros(n_groups, dtype=np.float64)
    visited = counts > 0
    means[visited] = sums[visited] / counts[visited]
    return means, counts


def grouped_softmax_mean(
    values: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-hot softmax mean without materializing a token-query matrix."""
    counts = np.bincount(groups, minlength=n_groups).astype(np.int64)
    sums = np.bincount(groups, weights=values, minlength=n_groups)
    means = np.zeros(n_groups, dtype=np.float64)
    diagonal_mass = np.zeros(n_groups, dtype=np.float64)
    visited = counts > 0
    sharp = float(np.exp(beta))
    denominators = counts[visited] * sharp + values.size - counts[visited]
    means[visited] = (
        sharp * sums[visited] + (float(values.sum()) - sums[visited])
    ) / denominators
    diagonal_mass[visited] = counts[visited] * sharp / denominators
    return means, counts, diagonal_mass


def iterative_pair_evaluation(
    current_pairs: np.ndarray,
    next_pairs: np.ndarray,
    rewards: np.ndarray,
    n_pairs: int,
    gamma: float,
    alpha: float,
    iterations: int,
    beta: float | None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    q_values = np.zeros(n_pairs, dtype=np.float64)
    counts = np.bincount(current_pairs, minlength=n_pairs).astype(np.int64)
    reward_sums = np.bincount(
        current_pairs, weights=rewards, minlength=n_pairs
    ).astype(np.float64)
    transition_counts = np.zeros((n_pairs, n_pairs), dtype=np.float64)
    np.add.at(transition_counts, (current_pairs, next_pairs), 1.0)
    visited = counts > 0
    diagonal_mass = np.where(visited, 1.0, 0.0)
    if beta is not None:
        sharp = float(np.exp(beta))
        denominators = counts[visited] * sharp + rewards.size - counts[visited]
        diagonal_mass[visited] = counts[visited] * sharp / denominators
    diverged = False
    iterations_used = 0

    for iteration in range(iterations):
        residual_sums = (
            reward_sums
            + gamma * (transition_counts @ q_values)
            - counts * q_values
        )
        if beta is None:
            means = np.zeros(n_pairs, dtype=np.float64)
            means[visited] = residual_sums[visited] / counts[visited]
        else:
            means = np.zeros(n_pairs, dtype=np.float64)
            total_sum = float(np.sum(residual_sums))
            means[visited] = (
                sharp * residual_sums[visited]
                + total_sum
                - residual_sums[visited]
            ) / denominators
        update = alpha * means[visited]
        q_values[visited] += update
        iterations_used = iteration + 1
        if not np.all(np.isfinite(q_values)) or float(np.max(np.abs(q_values))) > 1e9:
            diverged = True
            q_values = np.nan_to_num(q_values, nan=1e9, posinf=1e9, neginf=-1e9)
            q_values = np.clip(q_values, -1e9, 1e9)
            break
        if float(np.max(np.abs(update))) < 1e-13:
            break

    residuals = rewards + gamma * q_values[next_pairs] - q_values[current_pairs]
    exact_means, _ = grouped_exact_mean(residuals, current_pairs, n_pairs)
    if beta is None:
        soft_means = exact_means
        last_write_bound = 0.0
        last_write_error = 0.0
    else:
        soft_means, _, diagonal_mass = grouped_softmax_mean(
            residuals, current_pairs, n_pairs, beta
        )
        residual_range = float(np.max(residuals) - np.min(residuals))
        per_pair_bound = alpha * (1.0 - diagonal_mass) * residual_range
        per_pair_error = alpha * np.abs(soft_means - exact_means)
        last_write_bound = float(np.max(per_pair_bound[visited]))
        last_write_error = float(np.max(per_pair_error[visited]))
        if last_write_error > last_write_bound + 1e-10:
            raise AssertionError("finite pair-kernel leakage bound failed")

    diagnostics: dict[str, float | int | bool] = {
        "diverged": diverged,
        "iterations_used": iterations_used,
        "missing_pairs": int(np.sum(~visited)),
        "min_pair_count": int(np.min(counts)),
        "min_visited_pair_count": int(np.min(counts[visited])),
        "kernel_diagonal_min": float(np.min(diagonal_mass[visited])),
        "last_write_error": last_write_error,
        "last_write_bound": last_write_bound,
        "last_write_bound_slack": last_write_bound - last_write_error,
    }
    return q_values, diagnostics


def iterative_state_evaluation(
    states: np.ndarray,
    rewards: np.ndarray,
    n_states: int,
    gamma: float,
    alpha: float,
    iterations: int,
    beta: float | None,
    value_limit: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    current = states[:-1]
    following = states[1:]
    values = np.zeros(n_states, dtype=np.float64)
    counts = np.bincount(current, minlength=n_states).astype(np.int64)
    reward_sums = np.bincount(
        current, weights=rewards, minlength=n_states
    ).astype(np.float64)
    transition_counts = np.zeros((n_states, n_states), dtype=np.float64)
    np.add.at(transition_counts, (current, following), 1.0)
    visited = counts > 0
    diagonal_mass = np.where(visited, 1.0, 0.0)
    if beta is not None:
        sharp = float(np.exp(beta))
        denominators = counts[visited] * sharp + rewards.size - counts[visited]
        diagonal_mass[visited] = counts[visited] * sharp / denominators
    iterations_used = 0

    for iteration in range(iterations):
        residual_sums = (
            reward_sums
            + gamma * (transition_counts @ values)
            - counts * values
        )
        if beta is None:
            means = np.zeros(n_states, dtype=np.float64)
            means[visited] = residual_sums[visited] / counts[visited]
        else:
            means = np.zeros(n_states, dtype=np.float64)
            total_sum = float(np.sum(residual_sums))
            means[visited] = (
                sharp * residual_sums[visited]
                + total_sum
                - residual_sums[visited]
            ) / denominators
        update = alpha * means[visited]
        values[visited] += update
        values = np.clip(values, -value_limit, value_limit)
        iterations_used = iteration + 1
        if float(np.max(np.abs(update))) < 1e-13:
            break

    diagnostics: dict[str, float | int] = {
        "iterations_used": iterations_used,
        "missing_states": int(np.sum(~visited)),
        "min_state_count": int(np.min(counts)),
        "kernel_diagonal_min": float(np.min(diagonal_mass[visited])),
    }
    return values, diagnostics


def recover_q(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    rewards: np.ndarray,
    values: np.ndarray,
    n_states: int,
    n_actions: int,
    gamma: float,
    beta: float | None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    pairs = states * n_actions + actions
    targets = rewards + gamma * values[next_states]
    n_pairs = n_states * n_actions
    if beta is None:
        estimates, counts = grouped_exact_mean(targets, pairs, n_pairs)
        diagonal_mass = np.where(counts > 0, 1.0, 0.0)
    else:
        estimates, counts, diagonal_mass = grouped_softmax_mean(
            targets, pairs, n_pairs, beta
        )
    visited = counts > 0
    diagnostics: dict[str, float | int] = {
        "missing_pairs": int(np.sum(~visited)),
        "min_pair_count": int(np.min(counts)),
        "min_visited_pair_count": int(np.min(counts[visited])),
        "kernel_diagonal_min": float(np.min(diagonal_mass[visited])),
    }
    return estimates.reshape(n_states, n_actions), diagnostics


def q_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    error = np.abs(estimate - truth)
    result = dict(diagnostics)
    result.update(
        {
            "q_sup_error": float(np.max(error)),
            "q_mean_pair_error": float(np.mean(error)),
            "greedy_action_accuracy": float(
                np.mean(np.argmax(estimate, axis=1) == np.argmax(truth, axis=1))
            ),
        }
    )
    return result


def attach_finite_sample_certificate(
    metrics: dict[str, Any],
    high_probability: dict[str, Any],
    pathwise: dict[str, Any],
) -> None:
    """Attach theorem evidence without changing the route estimate itself."""
    actual_error = float(metrics["q_sup_error"])
    high_bound = high_probability.get("total_bound")
    pathwise_bound = pathwise.get("total_bound")
    high_slack = (
        None if high_bound is None else float(high_bound) - actual_error
    )
    pathwise_slack = (
        None if pathwise_bound is None else float(pathwise_bound) - actual_error
    )
    pathwise_verified = bool(
        pathwise.get("status") == "pathwise_bound_verified"
        and pathwise_slack is not None
        and pathwise_slack >= -1e-9
    )
    if (
        pathwise.get("status") == "pathwise_bound_verified"
        and not pathwise_verified
    ):
        raise AssertionError("observed pathwise finite-sample bound failed")

    if high_probability.get("status") == "high_probability_certified":
        status = "high_probability_certified"
    elif pathwise_verified:
        status = "pathwise_bound_verified"
    else:
        status = "not_certified"
    failure_reasons = list(high_probability.get("failure_reasons", []))
    if not pathwise_verified:
        for reason in pathwise.get("failure_reasons", []):
            if reason not in failure_reasons:
                failure_reasons.append(reason)

    metrics.update(
        {
            "certificate_status": status,
            "certificate_failure_reasons": failure_reasons,
            "high_probability_certified": bool(
                high_probability.get("status")
                == "high_probability_certified"
            ),
            "pathwise_bound_verified": pathwise_verified,
            "finite_sample_total_bound": high_bound,
            "finite_sample_bound_slack": high_slack,
            "finite_sample_bound_to_error_ratio": (
                None
                if high_bound is None or actual_error <= 0.0
                else float(high_bound) / actual_error
            ),
            "pathwise_total_bound": pathwise_bound,
            "pathwise_bound_slack": pathwise_slack,
            "finite_sample_certificate": {
                "high_probability": high_probability,
                "pathwise": pathwise,
            },
        }
    )


def vfirst_route(
    mdp: dict[str, Any],
    exact: dict[str, np.ndarray],
    value_estimate: np.ndarray,
    value_diagnostics: dict[str, Any],
    recovery_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    beta: float | None,
) -> dict[str, Any]:
    n_states = int(mdp["nS"])
    n_actions = int(mdp["nA"])
    gamma = float(mdp["gamma"])
    rec_states, rec_actions, rec_next_states, rec_rewards = recovery_data
    q_estimate, recovery_diagnostics = recover_q(
        rec_states,
        rec_actions,
        rec_next_states,
        rec_rewards,
        value_estimate,
        n_states,
        n_actions,
        gamma,
        beta,
    )
    transition = np.asarray(mdp["P"], dtype=np.float64)
    transition /= transition.sum(axis=2, keepdims=True)
    q_from_value = exact["reward_sa"] + gamma * np.einsum(
        "san,n->sa", transition, value_estimate
    )
    value_error = float(np.max(np.abs(value_estimate - exact["v_pi"])))
    recovery_error = float(np.max(np.abs(q_estimate - q_from_value)))
    composed_bound = gamma * value_error + recovery_error
    result = q_metrics(q_estimate, exact["q_pi"], recovery_diagnostics)
    result.update(
        {f"value_{key}": value for key, value in value_diagnostics.items()}
    )
    result.update(
        {
            "v_sup_error": value_error,
            "population_recovery_error": float(
                np.max(np.abs(q_from_value - exact["q_pi"]))
            ),
            "recovery_only_sup_error": recovery_error,
            "composed_bound": composed_bound,
            "composed_bound_slack": composed_bound - result["q_sup_error"],
        }
    )
    if result["composed_bound_slack"] < -1e-10:
        raise AssertionError("V-first composed error bound failed")
    return result


def crossfit_route_metrics(
    crossfit_result: dict[str, Any],
    exact: dict[str, np.ndarray],
    certificate: dict[str, Any],
    theory_status: str,
    uses_certificate_gap: bool,
) -> dict[str, Any]:
    diagnostics = dict(crossfit_result["diagnostics"])
    gap_diagnostics = certificate["gap"]
    diagnostics.update(
        {
            "theory_status": theory_status,
            "gap_raw": gap_diagnostics["gap_raw"],
            "gap_capped": (
                gap_diagnostics["gap_capped"]
                if uses_certificate_gap
                else False
            ),
            "dependency_tv_at_gap": (
                gap_diagnostics["dependency_tv_at_gap"]
                if uses_certificate_gap
                else 1.0
            ),
            "coverage_certified": certificate["kernel_coverage"][
                "coverage_certified"
            ],
            "kernel_certified": certificate["kernel_coverage"][
                "kernel_certified"
            ],
            "conditional_certificate_satisfied": bool(
                uses_certificate_gap
                and not gap_diagnostics["gap_capped"]
                and certificate["kernel_coverage"]["coverage_certified"]
                and certificate["kernel_coverage"]["kernel_certified"]
            ),
        }
    )
    return q_metrics(
        crossfit_result["q_estimate"], exact["q_pi"], diagnostics
    )


def evaluate_prefix(
    mdp: dict[str, Any],
    exact: dict[str, np.ndarray],
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    trajectory_length: int,
    beta: float,
    alpha: float,
    iterations: int,
    certificate: dict[str, Any],
    certificate_delta: float = 0.05,
) -> dict[str, dict[str, Any]]:
    n_states = int(mdp["nS"])
    n_actions = int(mdp["nA"])
    n_pairs = n_states * n_actions
    gamma = float(mdp["gamma"])
    pair_current = states[:trajectory_length] * n_actions + actions[:trajectory_length]
    pair_following = (
        states[1 : trajectory_length + 1] * n_actions
        + actions[1 : trajectory_length + 1]
    )
    transition_rewards = rewards[1 : trajectory_length + 1].astype(np.float64)
    reward_limit = float(np.max(np.abs(np.asarray(mdp["R"]))))
    value_limit = reward_limit / (1.0 - gamma)
    edge_support = certificate.get("edge_support", {})
    shared_event = shared_event_radii(
        trajectory_length=trajectory_length,
        n_states=n_states,
        n_pairs=n_pairs,
        delta=certificate_delta,
        value_bound=value_limit,
        state_stationary_min=float(np.min(exact["mu_state"])),
        pair_stationary_min=float(np.min(exact["mu_pair"])),
        state_right_inflation=certificate["state"][
            "right_hoeffding_inflation"
        ],
        pair_right_inflation=certificate["pair"][
            "right_hoeffding_inflation"
        ],
        edge_right_inflation=certificate["edge"][
            "right_hoeffding_inflation"
        ],
        numerical_support_truncated=bool(
            edge_support.get("numerical_support_truncated", False)
        ),
    )
    observed_ghost = observed_ghost_residuals(
        current_states=states[:trajectory_length],
        current_pairs=pair_current,
        next_states=states[1 : trajectory_length + 1],
        next_pairs=pair_following,
        rewards=transition_rewards,
        true_value=exact["v_pi"],
        true_q=exact["q_pi"],
        gamma=gamma,
        n_states=n_states,
        n_pairs=n_pairs,
    )
    certificate["finite_sample"] = {
        "assumptions": shared_event["assumptions"],
        "shared_event": shared_event,
        "observed_ghost_residuals": strict_json_ready(observed_ghost),
    }

    direct_exact, direct_exact_diag = iterative_pair_evaluation(
        pair_current,
        pair_following,
        transition_rewards,
        n_pairs,
        gamma,
        alpha,
        iterations,
        beta=None,
    )
    direct_softmax, direct_softmax_diag = iterative_pair_evaluation(
        pair_current,
        pair_following,
        transition_rewards,
        n_pairs,
        gamma,
        alpha,
        iterations,
        beta=beta,
    )

    transition = np.asarray(mdp["P"], dtype=np.float64)
    transition /= transition.sum(axis=2, keepdims=True)
    half = trajectory_length // 2
    first_states = states[: half + 1]
    first_rewards = rewards[1 : half + 1].astype(np.float64)
    split_recovery = (
        states[half:trajectory_length],
        actions[half:trajectory_length],
        states[half + 1 : trajectory_length + 1],
        rewards[half + 1 : trajectory_length + 1].astype(np.float64),
    )
    all_recovery = (
        states[:trajectory_length],
        actions[:trajectory_length],
        states[1 : trajectory_length + 1],
        transition_rewards,
    )

    v_split_exact, v_split_exact_diag = iterative_state_evaluation(
        first_states,
        first_rewards,
        n_states,
        gamma,
        alpha,
        iterations,
        beta=None,
        value_limit=value_limit,
    )
    v_split_soft, v_split_soft_diag = iterative_state_evaluation(
        first_states,
        first_rewards,
        n_states,
        gamma,
        alpha,
        iterations,
        beta=beta,
        value_limit=value_limit,
    )
    v_all_exact, v_all_exact_diag = iterative_state_evaluation(
        states[: trajectory_length + 1],
        transition_rewards,
        n_states,
        gamma,
        alpha,
        iterations,
        beta=None,
        value_limit=value_limit,
    )
    v_all_soft, v_all_soft_diag = iterative_state_evaluation(
        states[: trajectory_length + 1],
        transition_rewards,
        n_states,
        gamma,
        alpha,
        iterations,
        beta=beta,
        value_limit=value_limit,
    )
    prefix_states = states[: trajectory_length + 1]
    prefix_actions = actions[: trajectory_length + 1]
    prefix_rewards = rewards[: trajectory_length + 1]
    gap_used = int(certificate["gap"]["gap_used"])
    crossfit_exact = crossfit_vfirst_estimate(
        prefix_states,
        prefix_actions,
        prefix_rewards,
        transition,
        exact["reward_sa"],
        gamma,
        alpha,
        iterations,
        value_limit,
        gap_each_side=0,
        beta=None,
        true_value=exact["v_pi"],
        true_q=exact["q_pi"],
    )
    crossfit_softmax = crossfit_vfirst_estimate(
        prefix_states,
        prefix_actions,
        prefix_rewards,
        transition,
        exact["reward_sa"],
        gamma,
        alpha,
        iterations,
        value_limit,
        gap_each_side=0,
        beta=beta,
        true_value=exact["v_pi"],
        true_q=exact["q_pi"],
    )
    crossfit_gap_exact = crossfit_vfirst_estimate(
        prefix_states,
        prefix_actions,
        prefix_rewards,
        transition,
        exact["reward_sa"],
        gamma,
        alpha,
        iterations,
        value_limit,
        gap_each_side=gap_used,
        beta=None,
        true_value=exact["v_pi"],
        true_q=exact["q_pi"],
    )
    crossfit_gap_softmax = crossfit_vfirst_estimate(
        prefix_states,
        prefix_actions,
        prefix_rewards,
        transition,
        exact["reward_sa"],
        gamma,
        alpha,
        iterations,
        value_limit,
        gap_each_side=gap_used,
        beta=beta,
        true_value=exact["v_pi"],
        true_q=exact["q_pi"],
    )

    routes = {
        "direct_exact": q_metrics(
            direct_exact.reshape(n_states, n_actions),
            exact["q_pi"],
            direct_exact_diag,
        ),
        "direct_softmax": q_metrics(
            direct_softmax.reshape(n_states, n_actions),
            exact["q_pi"],
            direct_softmax_diag,
        ),
        "vfirst_split_exact": vfirst_route(
            mdp,
            exact,
            v_split_exact,
            v_split_exact_diag,
            split_recovery,
            beta=None,
        ),
        "vfirst_split_softmax": vfirst_route(
            mdp,
            exact,
            v_split_soft,
            v_split_soft_diag,
            split_recovery,
            beta=beta,
        ),
        "vfirst_nosplit_exact": vfirst_route(
            mdp,
            exact,
            v_all_exact,
            v_all_exact_diag,
            all_recovery,
            beta=None,
        ),
        "vfirst_nosplit_softmax": vfirst_route(
            mdp,
            exact,
            v_all_soft,
            v_all_soft_diag,
            all_recovery,
            beta=beta,
        ),
        "vfirst_crossfit_exact": crossfit_route_metrics(
            crossfit_exact,
            exact,
            certificate,
            theory_status="empirical_crossfit_without_gap",
            uses_certificate_gap=False,
        ),
        "vfirst_crossfit_softmax": crossfit_route_metrics(
            crossfit_softmax,
            exact,
            certificate,
            theory_status="empirical_crossfit_without_gap",
            uses_certificate_gap=False,
        ),
        "vfirst_crossfit_gap_exact": crossfit_route_metrics(
            crossfit_gap_exact,
            exact,
            certificate,
            theory_status="blocked_crossfit_dependency_diagnostic",
            uses_certificate_gap=True,
        ),
        "vfirst_crossfit_gap_softmax": crossfit_route_metrics(
            crossfit_gap_softmax,
            exact,
            certificate,
            theory_status="blocked_crossfit_dependency_diagnostic",
            uses_certificate_gap=True,
        ),
        "vfirst_oracle_exact": vfirst_route(
            mdp,
            exact,
            exact["v_pi"],
            {"missing_states": 0, "min_state_count": 0, "kernel_diagonal_min": 1.0},
            split_recovery,
            beta=None,
        ),
    }

    initial_q_error = float(np.max(np.abs(exact["q_pi"])))
    initial_v_error = float(np.max(np.abs(exact["v_pi"])))
    missing_pairs = int(observed_ghost["missing_pairs"])
    missing_states = int(observed_ghost["missing_states"])

    direct_exact_hp = direct_q_uniform_bound(
        shared_event,
        matching="exact",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(direct_exact_diag["iterations_used"]),
        initial_error=initial_q_error,
        missing_pairs=missing_pairs,
        divergence_guard_triggered=bool(direct_exact_diag["diverged"]),
    )
    direct_softmax_hp = direct_q_uniform_bound(
        shared_event,
        matching="softmax",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(direct_softmax_diag["iterations_used"]),
        initial_error=initial_q_error,
        beta=beta,
        missing_pairs=missing_pairs,
        divergence_guard_triggered=bool(direct_softmax_diag["diverged"]),
    )
    direct_exact_path = direct_q_uniform_bound(
        shared_event,
        matching="exact",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(direct_exact_diag["iterations_used"]),
        initial_error=initial_q_error,
        missing_pairs=missing_pairs,
        evidence_level="pathwise",
        residual_bound=observed_ghost["pair_residual_sup"],
        divergence_guard_triggered=bool(direct_exact_diag["diverged"]),
    )
    direct_softmax_path = direct_q_uniform_bound(
        shared_event,
        matching="softmax",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(direct_softmax_diag["iterations_used"]),
        initial_error=initial_q_error,
        beta=beta,
        missing_pairs=missing_pairs,
        evidence_level="pathwise",
        residual_bound=observed_ghost["pair_residual_sup"],
        kernel_diagonal_lower_bound=float(
            direct_softmax_diag["kernel_diagonal_min"]
        ),
        divergence_guard_triggered=bool(direct_softmax_diag["diverged"]),
    )

    state_exact_hp = state_value_uniform_bound(
        shared_event,
        matching="exact",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(v_all_exact_diag["iterations_used"]),
        initial_error=initial_v_error,
        missing_states=missing_states,
    )
    state_softmax_hp = state_value_uniform_bound(
        shared_event,
        matching="softmax",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(v_all_soft_diag["iterations_used"]),
        initial_error=initial_v_error,
        beta=beta,
        missing_states=missing_states,
    )
    state_exact_path = state_value_uniform_bound(
        shared_event,
        matching="exact",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(v_all_exact_diag["iterations_used"]),
        initial_error=initial_v_error,
        missing_states=missing_states,
        evidence_level="pathwise",
        residual_bound=observed_ghost["state_residual_sup"],
    )
    state_softmax_path = state_value_uniform_bound(
        shared_event,
        matching="softmax",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(v_all_soft_diag["iterations_used"]),
        initial_error=initial_v_error,
        beta=beta,
        missing_states=missing_states,
        evidence_level="pathwise",
        residual_bound=observed_ghost["state_residual_sup"],
        kernel_diagonal_lower_bound=float(
            v_all_soft_diag["kernel_diagonal_min"]
        ),
    )

    vfirst_exact_hp = vfirst_nosplit_bound(
        shared_event,
        state_exact_hp,
        recovery_matching="exact",
        gamma=gamma,
        value_bound=value_limit,
        missing_pairs=missing_pairs,
    )
    vfirst_softmax_hp = vfirst_nosplit_bound(
        shared_event,
        state_softmax_hp,
        recovery_matching="softmax",
        gamma=gamma,
        value_bound=value_limit,
        beta=beta,
        missing_pairs=missing_pairs,
    )
    vfirst_exact_path = vfirst_nosplit_bound(
        shared_event,
        state_exact_path,
        recovery_matching="exact",
        gamma=gamma,
        value_bound=value_limit,
        missing_pairs=missing_pairs,
        evidence_level="pathwise",
        fixed_recovery_bound=observed_ghost["recovery_residual_sup"],
    )
    vfirst_softmax_path = vfirst_nosplit_bound(
        shared_event,
        state_softmax_path,
        recovery_matching="softmax",
        gamma=gamma,
        value_bound=value_limit,
        beta=beta,
        missing_pairs=missing_pairs,
        evidence_level="pathwise",
        fixed_recovery_bound=observed_ghost["recovery_residual_sup"],
        recovery_diagonal_lower_bound=float(
            routes["vfirst_nosplit_softmax"]["kernel_diagonal_min"]
        ),
    )

    attach_finite_sample_certificate(
        routes["direct_exact"], direct_exact_hp, direct_exact_path
    )
    attach_finite_sample_certificate(
        routes["direct_softmax"], direct_softmax_hp, direct_softmax_path
    )
    attach_finite_sample_certificate(
        routes["vfirst_nosplit_exact"], vfirst_exact_hp, vfirst_exact_path
    )
    attach_finite_sample_certificate(
        routes["vfirst_nosplit_softmax"],
        vfirst_softmax_hp,
        vfirst_softmax_path,
    )
    routes["vfirst_nosplit_exact"]["finite_sample_certificate"][
        "state_value"
    ] = {
        "high_probability": state_exact_hp,
        "pathwise": state_exact_path,
    }
    routes["vfirst_nosplit_softmax"]["finite_sample_certificate"][
        "state_value"
    ] = {
        "high_probability": state_softmax_hp,
        "pathwise": state_softmax_path,
    }
    certificate["finite_sample"]["state_value"] = {
        "exact": {
            "high_probability": state_exact_hp,
            "pathwise": state_exact_path,
        },
        "softmax": {
            "high_probability": state_softmax_hp,
            "pathwise": state_softmax_path,
        },
    }

    kernel_certificate = certificate["kernel_coverage"]
    direct_common = {
        "recovery_transitions_used": trajectory_length,
        "discarded_gap_transitions": 0,
        "coverage_certified": kernel_certificate["coverage_certified"],
        "kernel_certified": kernel_certificate["kernel_certified"],
        "population_kernel_threshold_slack": kernel_certificate[
            "population_kernel_threshold_slack"
        ],
        "empirical_kernel_threshold_slack": kernel_certificate[
            "empirical_kernel_threshold_slack"
        ],
        "theory_status": "conditional_primitive_markov_certificate",
    }
    routes["direct_exact"].update(direct_common)
    routes["direct_softmax"].update(direct_common)
    split_used = trajectory_length - half
    for route in ("vfirst_split_exact", "vfirst_split_softmax"):
        routes[route].update(
            {
                "recovery_transitions_used": split_used,
                "discarded_gap_transitions": 0,
                "theory_status": "sample_split_markov_evaluation",
            }
        )
    routes["vfirst_nosplit_exact"].update(
        {
            "recovery_transitions_used": trajectory_length,
            "discarded_gap_transitions": 0,
            # Retained for byte-compatible route regression; the nested
            # finite_sample_certificate carries the corrected theorem status.
            "theory_status": "same_sample_plugin_diagnostic",
        }
    )
    routes["vfirst_nosplit_softmax"].update(
        {
            "recovery_transitions_used": trajectory_length,
            "discarded_gap_transitions": 0,
            "theory_status": "same_sample_fixed_target_theorem",
        }
    )
    routes["vfirst_oracle_exact"].update(
        {
            "recovery_transitions_used": split_used,
            "discarded_gap_transitions": 0,
            "theory_status": "oracle_value_recovery_diagnostic",
        }
    )
    return routes


def summarize(task_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def mean_confidence_interval(values: list[float]) -> tuple[float, float]:
        array = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(array))
        if array.size < 2:
            return mean, mean
        half_width = float(
            student_t.ppf(0.975, array.size - 1)
            * np.std(array, ddof=1)
            / np.sqrt(array.size)
        )
        return mean - half_width, mean + half_width

    def optional_mean(
        metrics: list[dict[str, Any]], key: str
    ) -> float | None:
        values = [
            item[key]
            for item in metrics
            if key in item and item[key] is not None
        ]
        return None if not values else float(np.mean(values))

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for task in task_results:
        for route, metrics in task["routes"].items():
            key = (
                task["trajectory_length"],
                task["n_actions"],
                task["pi_min"],
                task["beta"],
                task["mixing"],
                task["gap_bonus"],
                route,
            )
            grouped[key].append(metrics)

    rows: list[dict[str, Any]] = []
    for key, metrics_list in sorted(grouped.items(), key=lambda item: item[0]):
        length, n_actions, pi_min, beta, mixing, gap_bonus, route = key
        q_errors = [item["q_sup_error"] for item in metrics_list]
        ci_low, ci_high = mean_confidence_interval(q_errors)
        theorem_metrics = [
            item for item in metrics_list if "finite_sample_certificate" in item
        ]
        failure_counts: dict[str, int] = defaultdict(int)
        for item in theorem_metrics:
            for reason in item.get("certificate_failure_reasons", []):
                failure_counts[str(reason)] += 1
        failure_rates = {
            reason: count / len(theorem_metrics)
            for reason, count in sorted(failure_counts.items())
        }
        rows.append(
            {
                "trajectory_length": length,
                "n_actions": n_actions,
                "pi_min": pi_min,
                "beta": beta,
                "mixing": mixing,
                "gap_bonus": gap_bonus,
                "route": route,
                "tasks": len(metrics_list),
                "q_sup_error_mean": float(np.mean(q_errors)),
                "q_sup_error_median": float(
                    np.median(q_errors)
                ),
                "q_sup_error_ci95_low": ci_low,
                "q_sup_error_ci95_high": ci_high,
                "q_mean_pair_error_mean": float(
                    np.mean([item["q_mean_pair_error"] for item in metrics_list])
                ),
                "greedy_action_accuracy_mean": float(
                    np.mean([item["greedy_action_accuracy"] for item in metrics_list])
                ),
                "missing_pairs_mean": float(
                    np.mean([item["missing_pairs"] for item in metrics_list])
                ),
                "worst_min_pair_count": int(
                    np.min([item["min_pair_count"] for item in metrics_list])
                ),
                "kernel_diagonal_min_mean": float(
                    np.mean([item["kernel_diagonal_min"] for item in metrics_list])
                ),
                "recovery_transitions_used_mean": optional_mean(
                    metrics_list, "recovery_transitions_used"
                ),
                "discarded_gap_transitions_mean": optional_mean(
                    metrics_list, "discarded_gap_transitions"
                ),
                "gap_capped_rate": optional_mean(
                    metrics_list, "gap_capped"
                ),
                "coverage_certified_rate": optional_mean(
                    metrics_list, "coverage_certified"
                ),
                "conditional_certificate_satisfied_rate": optional_mean(
                    metrics_list, "conditional_certificate_satisfied"
                ),
                "high_probability_certificate_rate": optional_mean(
                    theorem_metrics, "high_probability_certified"
                ),
                "pathwise_bound_verified_rate": optional_mean(
                    theorem_metrics, "pathwise_bound_verified"
                ),
                "finite_sample_total_bound_mean": optional_mean(
                    theorem_metrics, "finite_sample_total_bound"
                ),
                "finite_sample_bound_to_error_ratio_mean": optional_mean(
                    theorem_metrics, "finite_sample_bound_to_error_ratio"
                ),
                "pathwise_total_bound_mean": optional_mean(
                    theorem_metrics, "pathwise_total_bound"
                ),
                "pathwise_bound_slack_mean": optional_mean(
                    theorem_metrics, "pathwise_bound_slack"
                ),
                "certificate_failure_reason_rates": failure_rates,
            }
        )
    return rows


def plot_results(task_results: list[dict[str, Any]], output_dir: Path) -> None:
    by_route_length: dict[tuple[str, int], list[float]] = defaultdict(list)
    for task in task_results:
        for route, metrics in task["routes"].items():
            by_route_length[(route, task["trajectory_length"])].append(
                metrics["q_sup_error"]
            )

    lengths = sorted({task["trajectory_length"] for task in task_results})
    fig, axis = plt.subplots(figsize=(9.0, 5.5))
    for route in ROUTE_ORDER:
        means = [np.mean(by_route_length[(route, length)]) for length in lengths]
        axis.plot(lengths, means, marker="o", label=ROUTE_LABELS[route])
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Total trajectory transitions")
    axis.set_ylabel("Mean Q sup-norm error")
    axis.set_title("Fixed-policy Q estimation under matched total budgets")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "q_sup_error.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 5.5))
    for route, marker in (
        ("direct_exact", "o"),
        ("vfirst_split_exact", "s"),
    ):
        counts = []
        errors = []
        for task in task_results:
            metrics = task["routes"][route]
            counts.append(max(metrics["min_pair_count"], 0.5))
            errors.append(metrics["q_sup_error"])
        axis.scatter(counts, errors, alpha=0.45, marker=marker, label=ROUTE_LABELS[route])
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Minimum state-action count (0 shown at 0.5)")
    axis.set_ylabel("Q sup-norm error")
    axis.set_title("Pair coverage is the shared bottleneck")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_vs_error.png", dpi=180)
    plt.close(fig)

    comparison_routes = (
        "direct_exact",
        "vfirst_split_exact",
        "vfirst_nosplit_exact",
        "vfirst_nosplit_softmax",
        "vfirst_crossfit_exact",
        "vfirst_crossfit_gap_exact",
    )
    fig, axis = plt.subplots(figsize=(9.0, 5.5))
    for route in comparison_routes:
        means = [
            np.mean(by_route_length[(route, length)])
            for length in lengths
        ]
        axis.plot(lengths, means, marker="o", label=ROUTE_LABELS[route])
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Total trajectory transitions")
    axis.set_ylabel("Mean Q sup-norm error")
    axis.set_title("Direct-Q versus split, no-split, and cross-fit V-first")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "crossfit_error.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.0, 5.5))
    for route in comparison_routes:
        used = []
        for length in lengths:
            values = [
                task["routes"][route]["recovery_transitions_used"]
                for task in task_results
                if task["trajectory_length"] == length
            ]
            used.append(np.mean(values))
        axis.plot(lengths, used, marker="o", label=ROUTE_LABELS[route])
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("Total trajectory transitions")
    axis.set_ylabel("Mean transitions used for Q recovery")
    axis.set_title("Recovery budget retained by each route")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "recovery_budget.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.0, 5.5))
    for capped, color, label in (
        (False, "tab:blue", "gap sufficient"),
        (True, "tab:red", "gap capped"),
    ):
        selected = [
            task
            for task in task_results
            if bool(task["certificate"]["gap"]["gap_capped"]) is capped
        ]
        if selected:
            axis.scatter(
                [
                    max(
                        task["certificate"]["gap"]["dependency_tv_at_gap"],
                        1e-16,
                    )
                    for task in selected
                ],
                [
                    task["routes"]["vfirst_crossfit_gap_exact"][
                        "q_sup_error"
                    ]
                    for task in selected
                ],
                alpha=0.5,
                color=color,
                label=label,
            )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Worst forward/reverse TV dependency at used gap")
    axis.set_ylabel("Gap cross-fit Q sup-norm error")
    axis.set_title("Dependency certificate versus cross-fit error")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "certificate_vs_error.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--trajectory-lengths", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    parser.add_argument("--n-states", type=int, default=6)
    parser.add_argument("--n-actions", type=int, nargs="+")
    parser.add_argument("--pi-mins", type=float, nargs="+")
    parser.add_argument("--betas", type=float, nargs="+")
    parser.add_argument("--mixing", type=float, nargs="+")
    parser.add_argument("--gap-bonuses", type=float, nargs="+")
    parser.add_argument("--gamma", type=float, default=0.70)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--certificate-delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fixed_policy_q_routes"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_actions_grid = args.n_actions or ([2, 4] if args.quick else [2, 4, 8])
    pi_min_grid = args.pi_mins or ([0.05] if args.quick else [0.02, 0.08])
    beta_grid = args.betas or ([6.0, 10.0] if args.quick else [4.0, 8.0, 12.0])
    mixing_grid = args.mixing or ([0.08, 0.50] if args.quick else [0.03, 0.15, 0.60])
    gap_grid = args.gap_bonuses or ([0.0, 0.50] if args.quick else [0.0, 0.25, 0.75])
    iterations = args.iterations or (120 if args.quick else 240)
    if args.tasks <= 0 or min(args.trajectory_lengths) < 4:
        raise ValueError("tasks must be positive and trajectory lengths at least 4")
    if not 0.0 < args.gamma < 1.0 or not 0.0 < args.alpha <= 1.0:
        raise ValueError("gamma and alpha must lie in (0,1), with alpha allowing 1")
    if not 0.0 < args.certificate_delta < 1.0:
        raise ValueError("certificate_delta must lie in (0,1)")
    for n_actions in n_actions_grid:
        for pi_min in pi_min_grid:
            if not 0.0 < pi_min < 1.0 / n_actions:
                raise ValueError(f"pi_min={pi_min} invalid for n_actions={n_actions}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "tasks_per_cell": args.tasks,
        "trajectory_lengths": sorted(set(args.trajectory_lengths)),
        "n_states": args.n_states,
        "n_actions": n_actions_grid,
        "pi_mins": pi_min_grid,
        "betas": beta_grid,
        "mixing": mixing_grid,
        "gap_bonuses": gap_grid,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "iterations": iterations,
        "seed": args.seed,
        "quick": args.quick,
        "main_comparison": [
            "direct_exact",
            "direct_softmax",
            "vfirst_split_exact",
            "vfirst_split_softmax",
            "vfirst_crossfit_exact",
            "vfirst_crossfit_softmax",
            "vfirst_crossfit_gap_exact",
            "vfirst_crossfit_gap_softmax",
        ],
        "diagnostic_ablations": [
            "vfirst_nosplit_exact",
            "vfirst_nosplit_softmax",
            "vfirst_oracle_exact",
        ],
        "certificate_delta": args.certificate_delta,
    }
    write_json(output_dir / "config.json", config)

    seed_sequence = np.random.SeedSequence(args.seed)
    total_cells = (
        len(n_actions_grid)
        * len(pi_min_grid)
        * len(mixing_grid)
        * len(gap_grid)
        * args.tasks
    )
    child_seeds = iter(seed_sequence.spawn(total_cells))
    task_results: list[dict[str, Any]] = []
    maximum_length = max(args.trajectory_lengths)

    for n_actions in n_actions_grid:
        for pi_min in pi_min_grid:
            for mixing in mixing_grid:
                for gap_bonus in gap_grid:
                    for task_index in range(args.tasks):
                        child_seed = next(child_seeds)
                        rng = np.random.default_rng(child_seed)
                        mdp = make_mdp(
                            args.n_states,
                            n_actions,
                            args.gamma,
                            mixing,
                            gap_bonus,
                            rng,
                        )
                        policy = make_policy(args.n_states, n_actions, pi_min, rng)
                        exact = policy_quantities(mdp, policy)
                        base_certificate = build_markov_base_certificate(
                            exact["p_pi"], exact["p_pair"]
                        )
                        start = int(rng.choice(args.n_states, p=exact["mu_state"]))
                        states, actions, rewards = rollout(
                            mdp,
                            policy,
                            start=start,
                            n=maximum_length,
                            rng=rng,
                        )
                        sorted_q = np.sort(exact["q_pi"], axis=1)
                        action_gaps = sorted_q[:, -1] - sorted_q[:, -2]

                        for length in sorted(set(args.trajectory_lengths)):
                            for beta in beta_grid:
                                prefix_pairs = (
                                    states[:length] * n_actions
                                    + actions[:length]
                                )
                                pair_counts = np.bincount(
                                    prefix_pairs,
                                    minlength=args.n_states * n_actions,
                                )
                                certificate = (
                                    build_markov_coverage_certificate(
                                        exact["p_pi"],
                                        exact["p_pair"],
                                        pair_counts,
                                        length,
                                        beta,
                                        args.gamma,
                                        delta=args.certificate_delta,
                                        base_certificate=base_certificate,
                                    )
                                )
                                routes = evaluate_prefix(
                                    mdp,
                                    exact,
                                    states,
                                    actions,
                                    rewards,
                                    length,
                                    beta,
                                    args.alpha,
                                    iterations,
                                    certificate,
                                    certificate_delta=args.certificate_delta,
                                )
                                task_results.append(
                                    {
                                        "task_index": task_index,
                                        "seed_entropy": child_seed.entropy,
                                        "spawn_key": list(child_seed.spawn_key),
                                        "trajectory_length": length,
                                        "n_states": args.n_states,
                                        "n_actions": n_actions,
                                        "pi_min": pi_min,
                                        "beta": beta,
                                        "mixing": mixing,
                                        "gap_bonus": gap_bonus,
                                        "true_pair_occupancy_min": float(np.min(exact["mu_pair"])),
                                        "true_action_gap_min": float(np.min(action_gaps)),
                                        "true_action_gap_mean": float(np.mean(action_gaps)),
                                        "certificate": certificate,
                                        "routes": routes,
                                    }
                                )

    summary = summarize(task_results)
    write_json(output_dir / "task_results.json", task_results)
    write_json(output_dir / "summary.json", summary)
    plot_results(task_results, output_dir)

    overall = defaultdict(list)
    for task in task_results:
        for route, metrics in task["routes"].items():
            overall[route].append(metrics["q_sup_error"])
    print(f"wrote {len(task_results)} matched comparisons to {output_dir}")
    for route in ROUTE_ORDER:
        values = np.asarray(overall[route])
        print(
            f"{ROUTE_LABELS[route]:24s} "
            f"mean={values.mean():.4f} median={np.median(values):.4f} "
            f"max={values.max():.4f}"
        )
    print("PASS fixed-policy matched-budget experiment")


if __name__ == "__main__":
    main()
