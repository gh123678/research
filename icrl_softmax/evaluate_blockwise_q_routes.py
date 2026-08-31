"""Blockwise policy improvement using Direct-Q and V-first estimators.

Each block freezes its route-specific policy, samples a fresh on-policy
trajectory, estimates Q^pi from zero, and then applies the same exploratory
greedy improvement rule. Routes receive equal transition budgets per block.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from evaluate_fixed_policy_q_routes import (
    iterative_pair_evaluation,
    iterative_state_evaluation,
    make_mdp,
    make_policy,
    q_metrics,
    recover_q,
    write_json,
)
from mdps import rollout
from verify_fixed_policy_q_routes import policy_quantities


ROUTES = (
    "direct_exact",
    "direct_softmax",
    "vfirst_exact",
    "vfirst_softmax",
)

ROUTE_LABELS = {
    "direct_exact": "Direct-Q exact",
    "direct_softmax": "Direct-Q softmax",
    "vfirst_exact": "V-first exact",
    "vfirst_softmax": "V-first softmax",
}


def exploratory_greedy_policy(q_values: np.ndarray, pi_min: float) -> np.ndarray:
    n_states, n_actions = q_values.shape
    policy = np.full((n_states, n_actions), pi_min, dtype=np.float64)
    greedy = np.argmax(q_values, axis=1)
    policy[np.arange(n_states), greedy] = 1.0 - (n_actions - 1) * pi_min
    return policy


def exact_return(mdp: dict[str, Any], policy: np.ndarray) -> float:
    transition = np.asarray(mdp["P"], dtype=np.float64)
    transition /= transition.sum(axis=2, keepdims=True)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    reward_sa = np.sum(transition * reward, axis=2)
    p_pi = np.einsum("sa,san->sn", policy, transition)
    r_pi = np.sum(policy * reward_sa, axis=1)
    value = np.linalg.solve(
        np.eye(int(mdp["nS"])) - float(mdp["gamma"]) * p_pi,
        r_pi,
    )
    initial = np.asarray(mdp["p0"], dtype=np.float64)
    initial /= initial.sum()
    return float(initial @ value)


def estimate_route(
    route: str,
    mdp: dict[str, Any],
    exact: dict[str, np.ndarray],
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    beta: float,
    alpha: float,
    iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_states = int(mdp["nS"])
    n_actions = int(mdp["nA"])
    n_pairs = n_states * n_actions
    gamma = float(mdp["gamma"])
    block_length = len(states) - 1
    finite = route.endswith("softmax")
    route_beta = beta if finite else None

    if route.startswith("direct"):
        current_pairs = states[:-1] * n_actions + actions[:-1]
        next_pairs = states[1:] * n_actions + actions[1:]
        q_flat, diagnostics = iterative_pair_evaluation(
            current_pairs,
            next_pairs,
            rewards[1:].astype(np.float64),
            n_pairs,
            gamma,
            alpha,
            iterations,
            beta=route_beta,
        )
        estimate = q_flat.reshape(n_states, n_actions)
        return estimate, q_metrics(estimate, exact["q_pi"], diagnostics)

    half = block_length // 2
    reward_limit = float(np.max(np.abs(np.asarray(mdp["R"]))))
    value_limit = reward_limit / (1.0 - gamma)
    value, value_diagnostics = iterative_state_evaluation(
        states[: half + 1],
        rewards[1 : half + 1].astype(np.float64),
        n_states,
        gamma,
        alpha,
        iterations,
        beta=route_beta,
        value_limit=value_limit,
    )
    estimate, recovery_diagnostics = recover_q(
        states[half:-1],
        actions[half:-1],
        states[half + 1 :],
        rewards[half + 1 :].astype(np.float64),
        value,
        n_states,
        n_actions,
        gamma,
        beta=route_beta,
    )
    metrics = q_metrics(estimate, exact["q_pi"], recovery_diagnostics)
    transition = np.asarray(mdp["P"], dtype=np.float64)
    transition /= transition.sum(axis=2, keepdims=True)
    q_from_value = exact["reward_sa"] + gamma * np.einsum(
        "san,n->sa", transition, value
    )
    value_error = float(np.max(np.abs(value - exact["v_pi"])))
    recovery_error = float(np.max(np.abs(estimate - q_from_value)))
    composed_bound = gamma * value_error + recovery_error
    metrics.update(
        {f"value_{key}": item for key, item in value_diagnostics.items()}
    )
    metrics.update(
        {
            "v_sup_error": value_error,
            "recovery_only_sup_error": recovery_error,
            "composed_bound": composed_bound,
            "composed_bound_slack": composed_bound - metrics["q_sup_error"],
        }
    )
    if metrics["composed_bound_slack"] < -1e-10:
        raise AssertionError("blockwise V-first composed error bound failed")
    return estimate, metrics


def improvement_floor(
    current_return: float,
    q_error: float,
    mdp: dict[str, Any],
    pi_min: float,
) -> float:
    """Conservative floor for epsilon-uniform greedy improvement."""
    gamma = float(mdp["gamma"])
    epsilon = int(mdp["nA"]) * pi_min
    reward_limit = float(np.max(np.abs(np.asarray(mdp["R"]))))
    value_limit = reward_limit / (1.0 - gamma)
    possible_drop = (
        2.0 * (1.0 - epsilon) * q_error + 2.0 * epsilon * value_limit
    ) / (1.0 - gamma)
    return current_return - possible_drop


def run_task(
    task_index: int,
    task_seed: np.random.SeedSequence,
    args: argparse.Namespace,
) -> dict[str, Any]:
    children = task_seed.spawn(args.blocks + 1)
    mdp_rng = np.random.default_rng(children[0])
    mdp = make_mdp(
        args.n_states,
        args.n_actions,
        args.gamma,
        args.mixing,
        args.gap_bonus,
        mdp_rng,
    )
    initial_policy = make_policy(
        args.n_states, args.n_actions, args.pi_min, mdp_rng
    )
    policies = {route: initial_policy.copy() for route in ROUTES}
    route_results: dict[str, dict[str, Any]] = {
        route: {"blocks": []} for route in ROUTES
    }

    for block in range(args.blocks):
        block_seed = children[block + 1]
        for route in ROUTES:
            policy = policies[route]
            exact = policy_quantities(mdp, policy)
            rng = np.random.default_rng(block_seed)
            start = int(rng.choice(args.n_states, p=exact["mu_state"]))
            states, actions, rewards = rollout(
                mdp,
                policy,
                start=start,
                n=args.block_length,
                rng=rng,
            )
            estimate, metrics = estimate_route(
                route,
                mdp,
                exact,
                states,
                actions,
                rewards,
                args.beta,
                args.alpha,
                args.iterations,
            )
            next_policy = exploratory_greedy_policy(estimate, args.pi_min)
            current_return = exact_return(mdp, policy)
            next_return = exact_return(mdp, next_policy)
            guaranteed_floor = improvement_floor(
                current_return, metrics["q_sup_error"], mdp, args.pi_min
            )
            true_q = exact["q_pi"]
            sorted_q = np.sort(true_q, axis=1)
            action_gaps = sorted_q[:, -1] - sorted_q[:, -2]
            old_greedy = np.argmax(policy, axis=1)
            estimated_greedy = np.argmax(estimate, axis=1)
            true_greedy = np.argmax(true_q, axis=1)

            block_result = dict(metrics)
            block_result.update(
                {
                    "block": block,
                    "policy_return": current_return,
                    "next_policy_return": next_return,
                    "return_change": next_return - current_return,
                    "nonmonotone": bool(next_return < current_return - 1e-12),
                    "guaranteed_return_floor": guaranteed_floor,
                    "improvement_floor_holds": bool(
                        next_return >= guaranteed_floor - 1e-10
                    ),
                    "policy_change_fraction": float(
                        np.mean(estimated_greedy != old_greedy)
                    ),
                    "wrong_greedy_fraction": float(
                        np.mean(estimated_greedy != true_greedy)
                    ),
                    "state_occupancy_min": float(np.min(exact["mu_state"])),
                    "pair_occupancy_min": float(np.min(exact["mu_pair"])),
                    "action_gap_min": float(np.min(action_gaps)),
                    "action_gap_mean": float(np.mean(action_gaps)),
                    "old_greedy_actions": old_greedy.tolist(),
                    "estimated_greedy_actions": estimated_greedy.tolist(),
                    "true_greedy_actions": true_greedy.tolist(),
                }
            )
            if not block_result["improvement_floor_holds"]:
                raise AssertionError("approximate improvement floor failed")
            route_results[route]["blocks"].append(block_result)
            policies[route] = next_policy

    for route in ROUTES:
        blocks = route_results[route]["blocks"]
        route_results[route].update(
            {
                "initial_return": blocks[0]["policy_return"],
                "final_return": blocks[-1]["next_policy_return"],
                "return_gain": (
                    blocks[-1]["next_policy_return"]
                    - blocks[0]["policy_return"]
                ),
                "nonmonotone_blocks": int(
                    np.sum([item["nonmonotone"] for item in blocks])
                ),
                "cumulative_q_sup_error": float(
                    np.sum([item["q_sup_error"] for item in blocks])
                ),
            }
        )

    return {
        "task_index": task_index,
        "spawn_key": list(task_seed.spawn_key),
        "routes": route_results,
    }


def summarize(task_results: list[dict[str, Any]], blocks: int) -> dict[str, Any]:
    block_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for route in ROUTES:
        task_routes = [task["routes"][route] for task in task_results]
        route_rows.append(
            {
                "route": route,
                "tasks": len(task_routes),
                "initial_return_mean": float(
                    np.mean([item["initial_return"] for item in task_routes])
                ),
                "final_return_mean": float(
                    np.mean([item["final_return"] for item in task_routes])
                ),
                "return_gain_mean": float(
                    np.mean([item["return_gain"] for item in task_routes])
                ),
                "nonmonotone_blocks_mean": float(
                    np.mean([item["nonmonotone_blocks"] for item in task_routes])
                ),
                "cumulative_q_sup_error_mean": float(
                    np.mean([item["cumulative_q_sup_error"] for item in task_routes])
                ),
            }
        )
        for block in range(blocks):
            items = [item["blocks"][block] for item in task_routes]
            block_rows.append(
                {
                    "route": route,
                    "block": block,
                    "policy_return_mean": float(
                        np.mean([item["policy_return"] for item in items])
                    ),
                    "next_policy_return_mean": float(
                        np.mean([item["next_policy_return"] for item in items])
                    ),
                    "return_change_mean": float(
                        np.mean([item["return_change"] for item in items])
                    ),
                    "nonmonotone_rate": float(
                        np.mean([item["nonmonotone"] for item in items])
                    ),
                    "q_sup_error_mean": float(
                        np.mean([item["q_sup_error"] for item in items])
                    ),
                    "greedy_action_accuracy_mean": float(
                        np.mean([item["greedy_action_accuracy"] for item in items])
                    ),
                    "wrong_greedy_fraction_mean": float(
                        np.mean([item["wrong_greedy_fraction"] for item in items])
                    ),
                    "missing_pairs_mean": float(
                        np.mean([item["missing_pairs"] for item in items])
                    ),
                    "pair_occupancy_min_mean": float(
                        np.mean([item["pair_occupancy_min"] for item in items])
                    ),
                    "kernel_diagonal_min_mean": float(
                        np.mean([item["kernel_diagonal_min"] for item in items])
                    ),
                }
            )
    return {"route_summary": route_rows, "block_summary": block_rows}


def plot_results(
    task_results: list[dict[str, Any]], blocks: int, output_dir: Path
) -> None:
    fig, axis = plt.subplots(figsize=(9.0, 5.5))
    for route in ROUTES:
        series = []
        for block in range(blocks + 1):
            if block == 0:
                values = [
                    task["routes"][route]["blocks"][0]["policy_return"]
                    for task in task_results
                ]
            else:
                values = [
                    task["routes"][route]["blocks"][block - 1][
                        "next_policy_return"
                    ]
                    for task in task_results
                ]
            series.append(np.mean(values))
        axis.plot(range(blocks + 1), series, marker="o", label=ROUTE_LABELS[route])
    axis.set_xlabel("Policy-improvement block")
    axis.set_ylabel("Mean exact discounted return")
    axis.set_title("Blockwise control with equal per-block trajectory budgets")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "return_by_block.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.0, 5.5))
    for route in ROUTES:
        errors = [
            np.mean(
                [
                    task["routes"][route]["blocks"][block]["q_sup_error"]
                    for task in task_results
                ]
            )
            for block in range(blocks)
        ]
        axis.plot(range(blocks), errors, marker="o", label=ROUTE_LABELS[route])
    axis.set_xlabel("Policy-evaluation block")
    axis.set_ylabel("Mean Q sup-norm error")
    axis.set_title("Evaluation error under changing policy coverage")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "q_error_by_block.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--block-length", type=int, default=1024)
    parser.add_argument("--n-states", type=int, default=6)
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument("--pi-min", type=float, default=0.05)
    parser.add_argument("--mixing", type=float, default=0.15)
    parser.add_argument("--gap-bonus", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.70)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--beta", type=float, default=8.0)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/blockwise_q_routes")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tasks <= 0 or args.blocks <= 0 or args.block_length < 4:
        raise ValueError("tasks, blocks, and block length must be positive")
    if not 0.0 < args.pi_min < 1.0 / args.n_actions:
        raise ValueError("pi_min must lie in (0, 1 / n_actions)")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "tasks": args.tasks,
        "blocks": args.blocks,
        "block_length": args.block_length,
        "n_states": args.n_states,
        "n_actions": args.n_actions,
        "pi_min": args.pi_min,
        "mixing": args.mixing,
        "gap_bonus": args.gap_bonus,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "beta": args.beta,
        "iterations": args.iterations,
        "seed": args.seed,
        "fresh_data_each_block": True,
        "equal_transition_budget_per_route": True,
        "common_random_numbers_per_block": True,
    }
    write_json(output_dir / "config.json", config)

    task_seeds = np.random.SeedSequence(args.seed).spawn(args.tasks)
    task_results = [
        run_task(index, task_seed, args)
        for index, task_seed in enumerate(task_seeds)
    ]
    summary = summarize(task_results, args.blocks)
    write_json(output_dir / "task_results.json", task_results)
    write_json(output_dir / "summary.json", summary)
    plot_results(task_results, args.blocks, output_dir)

    print(f"wrote {args.tasks} blockwise tasks to {output_dir}")
    for row in summary["route_summary"]:
        print(
            f"{ROUTE_LABELS[row['route']]:22s} "
            f"gain={row['return_gain_mean']:+.4f} "
            f"final={row['final_return_mean']:.4f} "
            f"nonmono={row['nonmonotone_blocks_mean']:.2f}/{args.blocks}"
        )
    print("PASS blockwise Direct-Q and V-first control comparison")


if __name__ == "__main__":
    main()
