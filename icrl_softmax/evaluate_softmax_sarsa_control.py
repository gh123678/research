"""Matched sampled-SARSA control with a standard normalized softmax write-back.

The experiment answers one question: can a standard softmax state-action kernel
support policy improvement when its signed values are sampled SARSA residuals?

For a trajectory batch, every transition is a context token with value

    delta_t = r_{t+1} + gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t).

One-hot state-action features form the matching kernel.  In the identity-kernel
limit, attention is uniform over all context occurrences of a queried pair and
zero elsewhere, so the update is exactly the tabular batch-SARSA convention

    Q(x) <- Q(x) + alpha * mean_{t: x_t=x} delta_t.

The finite-sharpness softmax learner and a shared-trajectory identity-limit
comparator use the same sampled trajectories and step sizes.  The comparator
isolates write-back error; it is not an independent on-policy baseline once its
policy differs from the learner's.  Policies are evaluated exactly from the
finite MDP rather than with noisy evaluation rollouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t as student_t

from mdps import rollout, sample_mdp
from model import KernelizedSoftmaxQTD


def epsilon_greedy_policy(q_values: np.ndarray, epsilon: float) -> np.ndarray:
    """Return the epsilon-greedy policy induced by a tabular action-value array."""
    if q_values.ndim != 2:
        raise ValueError("q_values must have shape (n_states, n_actions)")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    n_states, n_actions = q_values.shape
    policy = np.full((n_states, n_actions), epsilon / n_actions, dtype=np.float64)
    greedy_actions = np.argmax(q_values, axis=1)
    policy[np.arange(n_states), greedy_actions] += 1.0 - epsilon
    return policy


def exact_discounted_return(
    mdp: dict[str, Any], policy: np.ndarray, gamma: float
) -> float:
    """Compute p0^T (I - gamma P_pi)^(-1) r_pi exactly for a finite MDP."""
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    initial = np.asarray(mdp["p0"], dtype=np.float64)
    if policy.shape != (mdp["nS"], mdp["nA"]):
        raise ValueError("policy shape does not match the MDP")
    policy_transition = np.einsum("sa,san->sn", policy, transition)
    expected_sa_reward = np.sum(transition * reward, axis=-1)
    policy_reward = np.sum(policy * expected_sa_reward, axis=-1)
    system = np.eye(mdp["nS"], dtype=np.float64) - gamma * policy_transition
    value = np.linalg.solve(system, policy_reward)
    return float(initial @ value)


def policy_return(
    mdp: dict[str, Any], q_values: np.ndarray, epsilon: float, gamma: float
) -> float:
    return exact_discounted_return(
        mdp, epsilon_greedy_policy(q_values, epsilon), gamma
    )


def greedy_return(mdp: dict[str, Any], q_values: np.ndarray, gamma: float) -> float:
    return policy_return(mdp, q_values, epsilon=0.0, gamma=gamma)


def solve_optimal_return(mdp: dict[str, Any], gamma: float) -> float:
    """Return the exact discounted return of an optimal deterministic policy."""
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    expected_sa_reward = np.sum(transition * reward, axis=-1)
    value = np.zeros(mdp["nS"], dtype=np.float64)
    for _ in range(20_000):
        q_values = expected_sa_reward + gamma * np.einsum(
            "san,n->sa", transition, value
        )
        updated = np.max(q_values, axis=-1)
        if float(np.max(np.abs(updated - value))) < 1e-13:
            value = updated
            break
        value = updated
    else:
        raise RuntimeError("optimal value iteration did not converge")
    q_values = expected_sa_reward + gamma * np.einsum(
        "san,n->sa", transition, value
    )
    return greedy_return(mdp, q_values, gamma)


def sampled_sarsa_residuals(
    q_values: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return state-action indices and sampled SARSA residuals for a trajectory."""
    if len(states) != len(actions) or len(states) != len(rewards):
        raise ValueError("states, actions, and rewards must have equal trajectory length")
    if len(states) < 2:
        raise ValueError("a trajectory must contain at least one transition")
    n_actions = q_values.shape[1]
    pair_indices = states[:-1] * n_actions + actions[:-1]
    current = q_values[states[:-1], actions[:-1]]
    following = q_values[states[1:], actions[1:]]
    residuals = rewards[1:] + gamma * following - current
    return pair_indices.astype(np.int64), residuals.astype(np.float64)


def exact_batch_sarsa_write(
    q_values: np.ndarray,
    pair_indices: np.ndarray,
    residuals: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the exact identity-kernel batch SARSA update.

    The update convention averages all residuals belonging to the same visited
    state-action pair and leaves unvisited pairs unchanged.
    """
    flat = q_values.reshape(-1).copy()
    unique, inverse = np.unique(pair_indices, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    sums = np.bincount(inverse, weights=residuals, minlength=len(unique))
    means = sums / counts
    flat[unique] += alpha * means
    return flat.reshape(q_values.shape), unique, means


def softmax_batch_sarsa_write(
    q_values: np.ndarray,
    pair_indices: np.ndarray,
    residuals: np.ndarray,
    alpha: float,
    kernel_beta: float,
    state_action_eye: torch.Tensor,
) -> tuple[np.ndarray, dict[str, float]]:
    """Write sampled SARSA residuals with ``KernelizedSoftmaxQTD``."""
    unique = np.unique(pair_indices)
    pair_tensor = torch.as_tensor(pair_indices, dtype=torch.long)
    unique_tensor = torch.as_tensor(unique, dtype=torch.long)
    q_flat = torch.as_tensor(q_values.reshape(-1), dtype=torch.float64)
    td_values = torch.as_tensor(residuals, dtype=torch.float64)
    context_features = state_action_eye[pair_tensor]
    query_features = state_action_eye[unique_tensor]
    operator = KernelizedSoftmaxQTD(alpha=alpha, kernel_beta=kernel_beta)
    selected, update, attention = operator(
        q_flat[unique_tensor], td_values, context_features, query_features
    )

    updated = q_flat.clone()
    updated[unique_tensor] = selected

    match = pair_tensor[:, None] == unique_tensor[None, :]
    matching_mass = (attention * match.to(attention.dtype)).sum(dim=0)
    leakage = 1.0 - matching_mass
    diagnostics = {
        "max_kernel_leakage": float(torch.max(leakage)),
        "mean_kernel_leakage": float(torch.mean(leakage)),
        "max_abs_write": float(torch.max(torch.abs(update))),
        "attention_column_error": float(
            torch.max(
                torch.abs(
                    attention.sum(dim=0)
                    - torch.ones(attention.shape[1], dtype=attention.dtype)
                )
            )
        ),
    }
    return updated.numpy().reshape(q_values.shape), diagnostics


def formula_correspondence_check(finite_beta: float) -> dict[str, float]:
    """Numerically verify the exact identity limit and finite-softmax bound."""
    rng = np.random.default_rng(2026082401)
    n_states, n_actions, batch_size = 4, 3, 47
    n_pairs = n_states * n_actions
    alpha, gamma = 0.37, 0.61
    q_values = rng.normal(size=(n_states, n_actions))

    # Ensure every pair appears, then add repeated sampled occurrences.
    pair_indices = np.concatenate(
        [np.arange(n_pairs), rng.integers(0, n_pairs, size=batch_size - n_pairs)]
    )
    rng.shuffle(pair_indices)
    states = pair_indices // n_actions
    actions = pair_indices % n_actions
    next_states = rng.integers(0, n_states, size=batch_size)
    next_actions = rng.integers(0, n_actions, size=batch_size)
    rewards = rng.uniform(-1.0, 1.0, size=batch_size)
    residuals = (
        rewards
        + gamma * q_values[next_states, next_actions]
        - q_values[states, actions]
    )
    teacher, unique, means = exact_batch_sarsa_write(
        q_values, pair_indices, residuals, alpha
    )

    # The ideal attention matrix has 1/count(x) on matching occurrences.
    ideal_attention = np.zeros((batch_size, len(unique)), dtype=np.float64)
    for column, pair in enumerate(unique):
        matching = pair_indices == pair
        ideal_attention[matching, column] = 1.0 / np.sum(matching)
    ideal_selected = q_values.reshape(-1)[unique] + alpha * (
        residuals[:, None] * ideal_attention
    ).sum(axis=0)
    ideal_error = float(
        np.max(np.abs(ideal_selected - teacher.reshape(-1)[unique]))
    )

    eye = torch.eye(n_pairs, dtype=torch.float64)
    identity_limit, identity_diag = softmax_batch_sarsa_write(
        q_values,
        pair_indices,
        residuals,
        alpha,
        kernel_beta=1000.0,
        state_action_eye=eye,
    )
    identity_limit_error = float(np.max(np.abs(identity_limit - teacher)))

    finite, finite_diag = softmax_batch_sarsa_write(
        q_values,
        pair_indices,
        residuals,
        alpha,
        kernel_beta=finite_beta,
        state_action_eye=eye,
    )
    finite_error = float(np.max(np.abs(finite - teacher)))
    residual_range = float(np.max(residuals) - np.min(residuals))
    finite_bound = alpha * finite_diag["max_kernel_leakage"] * residual_range

    if ideal_error > 1e-13:
        raise AssertionError(f"ideal identity correspondence failed: {ideal_error}")
    if identity_limit_error > 1e-13:
        raise AssertionError(
            f"sharp-softmax identity correspondence failed: {identity_limit_error}"
        )
    if finite_error > finite_bound + 1e-12:
        raise AssertionError(
            f"finite-softmax error {finite_error} exceeds bound {finite_bound}"
        )
    if not np.all(np.isfinite(means)):
        raise AssertionError("non-finite batch means")

    return {
        "ideal_identity_max_abs_error": ideal_error,
        "sharp_softmax_beta": 1000.0,
        "sharp_softmax_max_abs_error": identity_limit_error,
        "sharp_softmax_max_kernel_leakage": identity_diag["max_kernel_leakage"],
        "finite_softmax_beta": finite_beta,
        "finite_softmax_max_abs_error": finite_error,
        "finite_softmax_error_bound": finite_bound,
        "finite_softmax_max_kernel_leakage": finite_diag["max_kernel_leakage"],
        "attention_column_error": max(
            identity_diag["attention_column_error"],
            finite_diag["attention_column_error"],
        ),
    }


def learning_rate(initial_alpha: float, decay: float, update: int) -> float:
    return initial_alpha / math.sqrt(1.0 + decay * update)


def evaluation_record(
    task: int,
    update: int,
    mdp: dict[str, Any],
    q_softmax: np.ndarray,
    q_comparator: np.ndarray,
    epsilon: float,
    gamma: float,
    max_leakage: float,
    one_step_identity_discrepancy: float,
    one_step_error_bound: float,
) -> dict[str, float | int]:
    soft_return = policy_return(mdp, q_softmax, epsilon, gamma)
    comparator_return = policy_return(mdp, q_comparator, epsilon, gamma)
    greedy_soft = greedy_return(mdp, q_softmax, gamma)
    greedy_comparator = greedy_return(mdp, q_comparator, gamma)
    policy_agreement = float(
        np.mean(np.argmax(q_softmax, axis=1) == np.argmax(q_comparator, axis=1))
    )
    return {
        "task": task,
        "update": update,
        "softmax_policy_return": soft_return,
        "comparator_policy_return": comparator_return,
        "softmax_greedy_return": greedy_soft,
        "comparator_greedy_return": greedy_comparator,
        "cumulative_comparator_q_inf_discrepancy": float(
            np.max(np.abs(q_softmax - q_comparator))
        ),
        "comparator_return_abs_discrepancy": abs(soft_return - comparator_return),
        "greedy_policy_agreement": policy_agreement,
        "max_kernel_leakage": max_leakage,
        "one_step_identity_discrepancy": one_step_identity_discrepancy,
        "one_step_error_bound": one_step_error_bound,
    }


def run_task(args: argparse.Namespace, task: int) -> tuple[dict[str, Any], list[dict]]:
    """Run one matched finite-MDP task."""
    seed_sequence = np.random.SeedSequence([args.seed, task])
    mdp_seed, init_seed, trajectory_seed = seed_sequence.spawn(3)
    mdp_rng = np.random.default_rng(mdp_seed)
    init_rng = np.random.default_rng(init_seed)
    trajectory_rng = np.random.default_rng(trajectory_seed)
    mdp = sample_mdp(args.nS, args.nA, args.gamma, mdp_rng)
    q_initial = init_rng.normal(
        loc=0.0, scale=args.initial_q_scale, size=(args.nS, args.nA)
    ).astype(np.float64)
    q_softmax = q_initial.copy()
    q_comparator = q_initial.copy()
    eye = torch.eye(args.nS * args.nA, dtype=torch.float64)

    history: list[dict] = [
        evaluation_record(
            task,
            0,
            mdp,
            q_softmax,
            q_comparator,
            args.epsilon,
            args.gamma,
            max_leakage=0.0,
            one_step_identity_discrepancy=0.0,
            one_step_error_bound=0.0,
        )
    ]
    observed_max_leakage = 0.0
    leakage_sum = 0.0
    agreement_sum = 0.0
    one_step_discrepancy_sum = 0.0
    max_one_step_discrepancy = 0.0
    max_one_step_bound = 0.0

    for update in range(1, args.updates + 1):
        behavior = epsilon_greedy_policy(q_softmax, args.epsilon)
        start = int(trajectory_rng.choice(args.nS, p=mdp["p0"]))
        states, actions, rewards = rollout(
            mdp, behavior, start, args.batch_size, trajectory_rng
        )
        pair_indices, softmax_residuals = sampled_sarsa_residuals(
            q_softmax, states, actions, rewards, args.gamma
        )
        comparator_pairs, comparator_residuals = sampled_sarsa_residuals(
            q_comparator, states, actions, rewards, args.gamma
        )
        if not np.array_equal(pair_indices, comparator_pairs):
            raise AssertionError("matched learners received different state-action tokens")

        alpha = learning_rate(args.alpha, args.alpha_decay, update - 1)
        identity_step, _, _ = exact_batch_sarsa_write(
            q_softmax, pair_indices, softmax_residuals, alpha
        )
        q_softmax_updated, diagnostics = softmax_batch_sarsa_write(
            q_softmax,
            pair_indices,
            softmax_residuals,
            alpha,
            args.kernel_beta,
            eye,
        )
        one_step_discrepancy = float(
            np.max(np.abs(q_softmax_updated - identity_step))
        )
        one_step_bound = (
            alpha
            * diagnostics["max_kernel_leakage"]
            * float(np.ptp(softmax_residuals))
        )
        if one_step_discrepancy > one_step_bound + 1e-12:
            raise AssertionError(
                "finite-softmax write-back exceeds its identity-limit error bound"
            )
        q_softmax = q_softmax_updated
        q_comparator, _, _ = exact_batch_sarsa_write(
            q_comparator, comparator_pairs, comparator_residuals, alpha
        )
        observed_max_leakage = max(
            observed_max_leakage, diagnostics["max_kernel_leakage"]
        )
        leakage_sum += diagnostics["mean_kernel_leakage"]
        one_step_discrepancy_sum += one_step_discrepancy
        max_one_step_discrepancy = max(
            max_one_step_discrepancy, one_step_discrepancy
        )
        max_one_step_bound = max(max_one_step_bound, one_step_bound)
        agreement_sum += float(
            np.mean(
                np.argmax(q_softmax, axis=1) == np.argmax(q_comparator, axis=1)
            )
        )
        if diagnostics["attention_column_error"] > 1e-12:
            raise AssertionError("softmax attention columns do not sum to one")

        if update % args.eval_every == 0 or update == args.updates:
            history.append(
                evaluation_record(
                    task,
                    update,
                    mdp,
                    q_softmax,
                    q_comparator,
                    args.epsilon,
                    args.gamma,
                    diagnostics["max_kernel_leakage"],
                    one_step_discrepancy,
                    one_step_bound,
                )
            )

    initial = history[0]
    final = history[-1]
    task_result = {
        "task": task,
        "task_seed_entropy": [args.seed, task],
        "initial_return": initial["softmax_policy_return"],
        "final_return": final["softmax_policy_return"],
        "improvement": final["softmax_policy_return"]
        - initial["softmax_policy_return"],
        "comparator_initial_return": initial["comparator_policy_return"],
        "comparator_final_return": final["comparator_policy_return"],
        "comparator_improvement": final["comparator_policy_return"]
        - initial["comparator_policy_return"],
        "initial_greedy_return": initial["softmax_greedy_return"],
        "final_greedy_return": final["softmax_greedy_return"],
        "greedy_improvement": final["softmax_greedy_return"]
        - initial["softmax_greedy_return"],
        "optimal_return": solve_optimal_return(mdp, args.gamma),
        "final_cumulative_comparator_q_inf_discrepancy": final[
            "cumulative_comparator_q_inf_discrepancy"
        ],
        "final_comparator_return_abs_discrepancy": final[
            "comparator_return_abs_discrepancy"
        ],
        "mean_one_step_identity_discrepancy": one_step_discrepancy_sum
        / args.updates,
        "max_one_step_identity_discrepancy": max_one_step_discrepancy,
        "max_one_step_error_bound": max_one_step_bound,
        "final_greedy_policy_agreement": final["greedy_policy_agreement"],
        "mean_greedy_policy_agreement": agreement_sum / args.updates,
        "max_kernel_leakage": observed_max_leakage,
        "mean_kernel_leakage": leakage_sum / args.updates,
    }
    evaluated_returns = np.asarray(
        [row["softmax_policy_return"] for row in history], dtype=np.float64
    )
    task_result["decreasing_evaluation_intervals"] = int(
        np.sum(np.diff(evaluated_returns) < -1e-12)
    )
    task_result["evaluation_intervals"] = len(evaluated_returns) - 1
    return task_result, history


def sample_mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return mean, std


def aggregate_results(task_results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "initial_return",
        "final_return",
        "improvement",
        "comparator_initial_return",
        "comparator_final_return",
        "comparator_improvement",
        "initial_greedy_return",
        "final_greedy_return",
        "greedy_improvement",
        "optimal_return",
        "final_cumulative_comparator_q_inf_discrepancy",
        "final_comparator_return_abs_discrepancy",
        "mean_one_step_identity_discrepancy",
        "max_one_step_identity_discrepancy",
        "max_one_step_error_bound",
        "final_greedy_policy_agreement",
        "mean_greedy_policy_agreement",
        "max_kernel_leakage",
        "mean_kernel_leakage",
    ]
    summary: dict[str, Any] = {"n_tasks": len(task_results)}
    for field in fields:
        mean, std = sample_mean_std([float(row[field]) for row in task_results])
        summary[f"{field}_mean"] = mean
        summary[f"{field}_std"] = std
    summary["improved_task_fraction"] = float(
        np.mean([row["improvement"] > 0.0 for row in task_results])
    )
    summary["comparator_improved_task_fraction"] = float(
        np.mean([row["comparator_improvement"] > 0.0 for row in task_results])
    )
    summary["total_decreasing_evaluation_intervals"] = int(
        sum(row["decreasing_evaluation_intervals"] for row in task_results)
    )
    summary["total_evaluation_intervals"] = int(
        sum(row["evaluation_intervals"] for row in task_results)
    )
    summary["tasks_with_any_decrease_fraction"] = float(
        np.mean([row["decreasing_evaluation_intervals"] > 0 for row in task_results])
    )
    improvements = np.asarray(
        [row["improvement"] for row in task_results], dtype=np.float64
    )
    if len(improvements) > 1:
        standard_error = float(np.std(improvements, ddof=1) / math.sqrt(len(improvements)))
    else:
        standard_error = 0.0
    critical = (
        float(student_t.ppf(0.975, df=len(improvements) - 1))
        if len(improvements) > 1
        else 0.0
    )
    summary["improvement_95ci_method"] = "two-sided Student-t interval"
    summary["improvement_95ci_low"] = float(
        np.mean(improvements) - critical * standard_error
    )
    summary["improvement_95ci_high"] = float(
        np.mean(improvements) + critical * standard_error
    )
    return summary


def method_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    common = {"n_tasks": summary["n_tasks"]}
    return [
        {
            "method": "standard-softmax sampled-SARSA write-back",
            **common,
            "initial_return_mean": summary["initial_return_mean"],
            "initial_return_std": summary["initial_return_std"],
            "final_return_mean": summary["final_return_mean"],
            "final_return_std": summary["final_return_std"],
            "improvement_mean": summary["improvement_mean"],
            "improvement_std": summary["improvement_std"],
            "improved_task_fraction": summary["improved_task_fraction"],
            "comparator_q_inf_discrepancy_mean": summary[
                "final_cumulative_comparator_q_inf_discrepancy_mean"
            ],
            "comparator_q_inf_discrepancy_std": summary[
                "final_cumulative_comparator_q_inf_discrepancy_std"
            ],
        },
        {
            "method": "shared-trajectory identity-limit comparator",
            **common,
            "initial_return_mean": summary["comparator_initial_return_mean"],
            "initial_return_std": summary["comparator_initial_return_std"],
            "final_return_mean": summary["comparator_final_return_mean"],
            "final_return_std": summary["comparator_final_return_std"],
            "improvement_mean": summary["comparator_improvement_mean"],
            "improvement_std": summary["comparator_improvement_std"],
            "improved_task_fraction": summary["comparator_improved_task_fraction"],
            "comparator_q_inf_discrepancy_mean": 0.0,
            "comparator_q_inf_discrepancy_std": 0.0,
        },
    ]


def make_figure(
    histories: dict[str, list[dict]],
    task_results: list[dict[str, Any]],
    outdir: Path,
) -> None:
    """Create a publication-ready policy-improvement figure."""
    task_keys = sorted(histories, key=lambda key: int(key.split("_")[-1]))
    updates = np.asarray([row["update"] for row in histories[task_keys[0]]])
    soft = np.asarray(
        [
            [row["softmax_policy_return"] for row in histories[key]]
            for key in task_keys
        ],
        dtype=np.float64,
    )
    comparator = np.asarray(
        [
            [row["comparator_policy_return"] for row in histories[key]]
            for key in task_keys
        ],
        dtype=np.float64,
    )
    soft_change = soft - soft[:, :1]
    comparator_change = comparator - comparator[:, :1]

    def mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        if values.shape[0] > 1:
            critical = float(student_t.ppf(0.975, df=values.shape[0] - 1))
            half = (
                critical
                * values.std(axis=0, ddof=1)
                / math.sqrt(values.shape[0])
            )
        else:
            half = np.zeros_like(mean)
        return mean, half

    soft_mean, soft_ci = mean_ci(soft_change)
    comparator_mean, comparator_ci = mean_ci(comparator_change)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {"softmax": "#0072B2", "comparator": "#D55E00"}
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.85))

    axes[0].plot(
        updates,
        soft_mean,
        color=colors["softmax"],
        linewidth=1.8,
        label="softmax SARSA write-back",
    )
    axes[0].fill_between(
        updates,
        soft_mean - soft_ci,
        soft_mean + soft_ci,
        color=colors["softmax"],
        alpha=0.18,
        linewidth=0,
    )
    axes[0].plot(
        updates,
        comparator_mean,
        color=colors["comparator"],
        linestyle="--",
        linewidth=1.35,
        label="identity-limit comparator",
    )
    axes[0].fill_between(
        updates,
        comparator_mean - comparator_ci,
        comparator_mean + comparator_ci,
        color=colors["comparator"],
        alpha=0.12,
        linewidth=0,
    )
    axes[0].axhline(0.0, color="0.45", linewidth=0.8)
    axes[0].set_xlabel("control update")
    axes[0].set_ylabel(r"policy-return change $J(\pi_t)-J(\pi_0)$")
    axes[0].set_title("(a) Closed-loop return gain")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(axis="y", color="0.9", linewidth=0.6)

    ordered = sorted(task_results, key=lambda row: row["improvement"])
    task_axis = np.arange(len(ordered))
    improvement = np.asarray([row["improvement"] for row in ordered])
    comparator_improvement = np.asarray(
        [row["comparator_improvement"] for row in ordered]
    )
    axes[1].axhline(0.0, color="0.45", linewidth=0.8)
    axes[1].bar(
        task_axis,
        improvement,
        width=0.72,
        color=colors["softmax"],
        alpha=0.8,
        label="softmax",
    )
    axes[1].scatter(
        task_axis,
        comparator_improvement,
        s=11,
        marker="o",
        facecolors="white",
        edgecolors=colors["comparator"],
        linewidths=0.8,
        zorder=3,
        label="identity comparator",
    )
    axes[1].set_xlabel("matched task (sorted)")
    axes[1].set_ylabel(r"final improvement $J(\pi_T)-J(\pi_0)$")
    axes[1].set_title("(b) Task-wise final improvement")
    axes[1].set_xticks([])
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(axis="y", color="0.9", linewidth=0.6)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(outdir / "softmax_sarsa_policy_improvement.pdf", bbox_inches="tight")
    fig.savefig(
        outdir / "softmax_sarsa_policy_improvement.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_outputs(
    args: argparse.Namespace,
    formula_check: dict[str, float],
    task_results: list[dict[str, Any]],
    histories: dict[str, list[dict]],
    summary: dict[str, Any],
) -> None:
    args.outdir.mkdir(parents=True, exist_ok=True)
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    configuration["update_convention"] = (
        "Q(x) <- Q(x) + alpha_t * mean_{k: x_k=x} "
        "[r_k + gamma Q(s'_k,a'_k) - Q(x_k)]"
    )
    configuration["behavior_policy"] = (
        "epsilon-greedy from the current standard-softmax learner Q"
    )
    configuration["evaluation"] = "exact p0^T (I - gamma P_pi)^(-1) r_pi"

    with open(args.outdir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(configuration, handle, indent=2)
    with open(args.outdir / "verification.json", "w", encoding="utf-8") as handle:
        json.dump(formula_check, handle, indent=2)
    with open(args.outdir / "task_results.json", "w", encoding="utf-8") as handle:
        json.dump(task_results, handle, indent=2)
    with open(args.outdir / "histories.json", "w", encoding="utf-8") as handle:
        json.dump(histories, handle, indent=2)
    with open(args.outdir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with open(
        args.outdir / "task_results.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_results[0].keys()))
        writer.writeheader()
        writer.writerows(task_results)

    rows = method_summary_rows(summary)
    with open(
        args.outdir / "summary.csv", "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standard-softmax sampled-SARSA policy improvement"
    )
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--nS", type=int, default=9)
    parser.add_argument("--nA", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--alpha-decay", type=float, default=0.02)
    parser.add_argument("--kernel-beta", type=float, default=12.0)
    parser.add_argument("--initial-q-scale", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/softmax_sarsa_control"),
    )
    args = parser.parse_args()
    if args.tasks <= 0 or args.updates <= 0 or args.batch_size <= 0:
        parser.error("tasks, updates, and batch-size must be positive")
    if args.nS <= 0 or args.nA <= 1:
        parser.error("nS must be positive and nA must exceed one")
    if not 0.0 <= args.gamma < 1.0:
        parser.error("gamma must lie in [0, 1)")
    if not 0.0 <= args.epsilon <= 1.0:
        parser.error("epsilon must lie in [0, 1]")
    if args.alpha <= 0.0 or args.alpha_decay < 0.0 or args.kernel_beta <= 0.0:
        parser.error("alpha and kernel-beta must be positive; decay must be nonnegative")
    if args.eval_every <= 0:
        parser.error("eval-every must be positive")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    formula_check = formula_correspondence_check(args.kernel_beta)
    task_results: list[dict[str, Any]] = []
    histories: dict[str, list[dict]] = {}
    for task in range(args.tasks):
        result, history = run_task(args, task)
        task_results.append(result)
        histories[f"task_{task}"] = history

    summary = aggregate_results(task_results)
    write_outputs(args, formula_check, task_results, histories, summary)
    make_figure(histories, task_results, args.outdir)

    print("FORMULA CORRESPONDENCE")
    print(
        "identity error="
        f"{formula_check['sharp_softmax_max_abs_error']:.3e}; "
        "finite error/bound="
        f"{formula_check['finite_softmax_max_abs_error']:.3e}/"
        f"{formula_check['finite_softmax_error_bound']:.3e}"
    )
    print("\nPOLICY IMPROVEMENT (mean +/- sample std)")
    print(
        f"softmax: {summary['initial_return_mean']:.6f} +/- "
        f"{summary['initial_return_std']:.6f} -> "
        f"{summary['final_return_mean']:.6f} +/- "
        f"{summary['final_return_std']:.6f}; change="
        f"{summary['improvement_mean']:.6f} +/- "
        f"{summary['improvement_std']:.6f}"
    )
    print(
        f"comparator: {summary['comparator_initial_return_mean']:.6f} +/- "
        f"{summary['comparator_initial_return_std']:.6f} -> "
        f"{summary['comparator_final_return_mean']:.6f} +/- "
        f"{summary['comparator_final_return_std']:.6f}; change="
        f"{summary['comparator_improvement_mean']:.6f} +/- "
        f"{summary['comparator_improvement_std']:.6f}"
    )
    print(
        f"improved tasks={summary['improved_task_fraction']:.3f}; "
        "95% CI for mean change=["
        f"{summary['improvement_95ci_low']:.6f}, "
        f"{summary['improvement_95ci_high']:.6f}]"
    )
    print(
        "cumulative comparator Q-inf discrepancy="
        f"{summary['final_cumulative_comparator_q_inf_discrepancy_mean']:.3e} +/- "
        f"{summary['final_cumulative_comparator_q_inf_discrepancy_std']:.3e}; "
        "policy agreement="
        f"{summary['final_greedy_policy_agreement_mean']:.6f}"
    )
    print(f"saved to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
