"""Two-fold V-first cross-fitting for a single fixed-policy trajectory."""

from __future__ import annotations

from typing import Any

import numpy as np


def crossfit_blocks(
    trajectory_length: int, gap_each_side: int
) -> dict[str, int]:
    if trajectory_length < 4:
        raise ValueError("trajectory_length must be at least 4")
    if gap_each_side < 0 or 2 * gap_each_side >= trajectory_length - 1:
        raise ValueError("gap leaves insufficient data for two folds")
    usable = trajectory_length - 2 * gap_each_side
    fold_a_length = usable // 2
    fold_b_length = usable - fold_a_length
    fold_a_stop = fold_a_length
    fold_b_start = fold_a_stop + 2 * gap_each_side
    return {
        "fold_a_start": 0,
        "fold_a_stop": fold_a_stop,
        "fold_b_start": fold_b_start,
        "fold_b_stop": trajectory_length,
        "fold_a_transitions": fold_a_length,
        "fold_b_transitions": fold_b_length,
        "recovery_transitions_used": usable,
        "discarded_gap_transitions": 2 * gap_each_side,
        "gap_each_side": gap_each_side,
    }


def _grouped_exact_mean(
    values: np.ndarray, groups: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(groups, minlength=n_groups).astype(np.int64)
    sums = np.bincount(groups, weights=values, minlength=n_groups)
    means = np.zeros(n_groups, dtype=np.float64)
    visited = counts > 0
    means[visited] = sums[visited] / counts[visited]
    diagonal = np.where(visited, 1.0, 0.0)
    return means, counts, diagonal


def _grouped_softmax_mean(
    values: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(groups, minlength=n_groups).astype(np.int64)
    sums = np.bincount(groups, weights=values, minlength=n_groups)
    means = np.zeros(n_groups, dtype=np.float64)
    diagonal = np.zeros(n_groups, dtype=np.float64)
    visited = counts > 0
    sharp = float(np.exp(beta))
    denominators = counts[visited] * sharp + values.size - counts[visited]
    means[visited] = (
        sharp * sums[visited] + float(values.sum()) - sums[visited]
    ) / denominators
    diagonal[visited] = counts[visited] * sharp / denominators
    return means, counts, diagonal


def _estimate_state_value(
    states: np.ndarray,
    rewards: np.ndarray,
    start: int,
    stop: int,
    n_states: int,
    gamma: float,
    alpha: float,
    iterations: int,
    beta: float | None,
    value_limit: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    current = states[start:stop]
    following = states[start + 1 : stop + 1]
    transition_rewards = rewards[start + 1 : stop + 1].astype(np.float64)
    if current.size == 0:
        raise ValueError("a cross-fit value fold cannot be empty")
    counts = np.bincount(current, minlength=n_states).astype(np.int64)
    reward_sums = np.bincount(
        current, weights=transition_rewards, minlength=n_states
    ).astype(np.float64)
    transition_counts = np.zeros((n_states, n_states), dtype=np.float64)
    np.add.at(transition_counts, (current, following), 1.0)
    visited = counts > 0
    value = np.zeros(n_states, dtype=np.float64)
    diagonal = np.where(visited, 1.0, 0.0)
    if beta is not None:
        sharp = float(np.exp(beta))
        denominators = counts[visited] * sharp + current.size - counts[visited]
        diagonal[visited] = counts[visited] * sharp / denominators
    iterations_used = 0
    for iteration in range(iterations):
        residual_sums = (
            reward_sums
            + gamma * (transition_counts @ value)
            - counts * value
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
        value[visited] += update
        value = np.clip(value, -value_limit, value_limit)
        iterations_used = iteration + 1
        if float(np.max(np.abs(update))) < 1e-13:
            break
    diagnostics = {
        "iterations_used": iterations_used,
        "missing_states": int(np.sum(~visited)),
        "min_state_count": int(np.min(counts)),
        "min_visited_state_count": int(np.min(counts[visited])),
        "value_kernel_diagonal_min": float(np.min(diagonal[visited])),
    }
    return value, diagnostics


def build_crossfit_targets(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    value_a: np.ndarray,
    value_b: np.ndarray,
    n_actions: int,
    gamma: float,
    blocks: dict[str, int],
) -> dict[str, np.ndarray]:
    """Build targets with the opposite fold's value estimate.

    Fold A recovery uses value_b, while fold B recovery uses value_a.
    """
    a_start = blocks["fold_a_start"]
    a_stop = blocks["fold_a_stop"]
    b_start = blocks["fold_b_start"]
    b_stop = blocks["fold_b_stop"]
    a_indices = np.arange(a_start, a_stop, dtype=np.int64)
    b_indices = np.arange(b_start, b_stop, dtype=np.int64)
    targets_a = (
        rewards[a_indices + 1].astype(np.float64)
        + gamma * value_b[states[a_indices + 1]]
    )
    targets_b = (
        rewards[b_indices + 1].astype(np.float64)
        + gamma * value_a[states[b_indices + 1]]
    )
    pairs_a = states[a_indices] * n_actions + actions[a_indices]
    pairs_b = states[b_indices] * n_actions + actions[b_indices]
    return {
        "targets": np.concatenate([targets_a, targets_b]),
        "pairs": np.concatenate([pairs_a, pairs_b]).astype(np.int64),
        "targets_a_using_value_b": targets_a,
        "targets_b_using_value_a": targets_b,
        "pairs_a": pairs_a.astype(np.int64),
        "pairs_b": pairs_b.astype(np.int64),
        "transition_indices_a": a_indices,
        "transition_indices_b": b_indices,
    }


def crossfit_population_reference(
    reward_sa: np.ndarray,
    transition: np.ndarray,
    value_a: np.ndarray,
    value_b: np.ndarray,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Return the count-weighted population target for opposite-fold values."""
    q_a_using_b = reward_sa + gamma * np.einsum(
        "san,n->sa", transition, value_b
    )
    q_b_using_a = reward_sa + gamma * np.einsum(
        "san,n->sa", transition, value_a
    )
    flat_a = q_a_using_b.reshape(-1)
    flat_b = q_b_using_a.reshape(-1)
    total = counts_a + counts_b
    reference = 0.5 * (flat_a + flat_b)
    visited = total > 0
    reference[visited] = (
        counts_a[visited] * flat_a[visited]
        + counts_b[visited] * flat_b[visited]
    ) / total[visited]
    return reference.reshape(reward_sa.shape)


def crossfit_vfirst_estimate(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    transition: np.ndarray,
    reward_sa: np.ndarray,
    gamma: float,
    alpha: float,
    iterations: int,
    value_limit: float,
    gap_each_side: int,
    beta: float | None = None,
    true_value: np.ndarray | None = None,
    true_q: np.ndarray | None = None,
) -> dict[str, Any]:
    n_states, n_actions = reward_sa.shape
    trajectory_length = len(states) - 1
    if len(actions) != len(states) or len(rewards) != len(states):
        raise ValueError("states, actions, and rewards must align")
    blocks = crossfit_blocks(trajectory_length, gap_each_side)
    value_a, diagnostics_a = _estimate_state_value(
        states,
        rewards,
        blocks["fold_a_start"],
        blocks["fold_a_stop"],
        n_states,
        gamma,
        alpha,
        iterations,
        beta,
        value_limit,
    )
    value_b, diagnostics_b = _estimate_state_value(
        states,
        rewards,
        blocks["fold_b_start"],
        blocks["fold_b_stop"],
        n_states,
        gamma,
        alpha,
        iterations,
        beta,
        value_limit,
    )
    target_data = build_crossfit_targets(
        states,
        actions,
        rewards,
        value_a,
        value_b,
        n_actions,
        gamma,
        blocks,
    )
    n_pairs = n_states * n_actions
    if beta is None:
        q_flat, counts, diagonal = _grouped_exact_mean(
            target_data["targets"], target_data["pairs"], n_pairs
        )
    else:
        q_flat, counts, diagonal = _grouped_softmax_mean(
            target_data["targets"], target_data["pairs"], n_pairs, beta
        )
    counts_a = np.bincount(target_data["pairs_a"], minlength=n_pairs)
    counts_b = np.bincount(target_data["pairs_b"], minlength=n_pairs)
    if not np.array_equal(counts, counts_a + counts_b):
        raise AssertionError("cross-fit pair-count aggregation failed")
    reference = crossfit_population_reference(
        reward_sa,
        transition,
        value_a,
        value_b,
        counts_a,
        counts_b,
        gamma,
    )
    estimate = q_flat.reshape(n_states, n_actions)
    recovery_error = float(np.max(np.abs(estimate - reference)))
    visited = counts > 0
    diagnostics: dict[str, Any] = {
        **blocks,
        "fold_a_counts": counts_a,
        "fold_b_counts": counts_b,
        "combined_counts": counts,
        "missing_pairs": int(np.sum(~visited)),
        "min_pair_count": int(np.min(counts)),
        "min_visited_pair_count": int(np.min(counts[visited])),
        "kernel_diagonal_min": float(np.min(diagonal[visited])),
        "fold_a_missing_states": diagnostics_a["missing_states"],
        "fold_b_missing_states": diagnostics_b["missing_states"],
        "fold_a_min_state_count": diagnostics_a["min_state_count"],
        "fold_b_min_state_count": diagnostics_b["min_state_count"],
        "fold_a_value_kernel_diagonal_min": diagnostics_a[
            "value_kernel_diagonal_min"
        ],
        "fold_b_value_kernel_diagonal_min": diagnostics_b[
            "value_kernel_diagonal_min"
        ],
        "recovery_only_sup_error": recovery_error,
    }
    if true_value is not None:
        if true_q is None:
            raise ValueError("true_q is required when true_value is supplied")
        fold_a_error = float(np.max(np.abs(value_a - true_value)))
        fold_b_error = float(np.max(np.abs(value_b - true_value)))
        max_value_error = max(fold_a_error, fold_b_error)
        population_error = float(np.max(np.abs(reference - true_q)))
        diagnostics.update(
            {
                "fold_a_v_sup_error": fold_a_error,
                "fold_b_v_sup_error": fold_b_error,
                "max_fold_v_sup_error": max_value_error,
                "population_mixture_error": population_error,
                "population_mixture_bound": gamma * max_value_error,
            }
        )
        if population_error > gamma * max_value_error + 1e-10:
            raise AssertionError("cross-fit population mixture bound failed")
        total_error = float(np.max(np.abs(estimate - true_q)))
        composed_bound = gamma * max_value_error + recovery_error
        diagnostics.update(
            {
                "q_sup_error": total_error,
                "composed_bound": composed_bound,
                "composed_bound_slack": composed_bound - total_error,
            }
        )
        if total_error > composed_bound + 1e-10:
            raise AssertionError("cross-fit composed error bound failed")
    return {
        "q_estimate": estimate,
        "value_a": value_a,
        "value_b": value_b,
        "population_reference": reference,
        "target_data": target_data,
        "diagnostics": diagnostics,
    }
