"""Paired evaluator for the frozen FP-KERN-001 kernel feasibility matrix.

Each record samples one MDP and fixed policy, derives three independent
stationary-start trajectories (value, signature, target) from deterministic
nonoverlapping substreams, builds the unchanged exact V-first value estimate
from the value stream, evaluates the four frozen routes from observable
aggregates only, and attaches exact truth and hidden structure under a
separate ``oracle_audit`` namespace after route outputs are frozen.

The evaluator refuses a nonempty output directory, runs the inherited and new
verifiers as preflight, and writes strict JSON config, task_results, summary,
environment, commands, and checks artifacts.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from evaluate_fixed_policy_q_routes import iterative_state_evaluation, make_policy
from fixed_policy_finite_sample_certificate import strict_json_ready
from kernel_generalization_mdps import make_current_mdp, make_hidden_cluster_mdp
from kernel_state_generalization import (
    ROUTES,
    KernelInputError,
    aggregate_y_sums,
    evaluate_all_routes,
    greedy_with_floor_policy,
)
from mdps import rollout
from verify_fixed_policy_q_routes import policy_quantities


TASK_ID = "FP-KERN-001"
ACTIVATION_COMMIT = "2308c372eb47ce7c181f98caee952121f4e47644"
COMMON_EXECUTION_START = "28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c"
FAMILIES = ("current_unstructured", "hidden_cluster")
PREFLIGHT_SCRIPTS = (
    "verify_fixed_policy_q_routes.py",
    "verify_finite_sample_theorems.py",
    "verify_visit_indexed_martingale_certificate.py",
    "verify_time_uniform_mixture_certificate.py",
    "verify_kernel_state_generalization.py",
)
FORMAL_PROTOCOL = {
    "tasks": 15,
    "families": list(FAMILIES),
    "trajectory_lengths": [256, 1024, 4096, 16384],
    "mixing": [0.08, 0.50],
    "gap_bonuses": [0.0, 0.50],
    "n_states": 6,
    "n_actions": 4,
    "pi_min": 0.05,
    "gamma": 0.70,
    "alpha": 0.65,
    "iterations": 160,
    "seed": 20260909,
    "records": 480,
}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            strict_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )


def run_preflight_checks(project_dir: Path) -> str:
    sections: list[str] = []
    for script in PREFLIGHT_SCRIPTS:
        command = [sys.executable, "-B", script]
        completed = subprocess.run(
            command, cwd=project_dir, capture_output=True, text=True, check=False
        )
        sections.extend(
            [
                f"COMMAND: {subprocess.list2cmdline(command)}",
                f"EXIT_CODE: {completed.returncode}",
                "STDOUT:",
                completed.stdout.rstrip(),
                "STDERR:",
                completed.stderr.rstrip(),
                "",
            ]
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"preflight verifier failed: {script}\n" + "\n".join(sections)
            )
    return "\n".join(sections).rstrip() + "\n"


def stream_aggregates(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_states: int,
    n_actions: int,
) -> dict[str, np.ndarray]:
    """Sufficient statistics of one stream: pair counts, reward and Y inputs."""
    current = states[:-1]
    pairs = current * n_actions + actions[:-1]
    transition_rewards = rewards[1:].astype(np.float64)
    pair_counts = np.bincount(pairs, minlength=n_states * n_actions).reshape(
        n_states, n_actions
    )
    reward_sums = np.bincount(
        pairs, weights=transition_rewards, minlength=n_states * n_actions
    ).reshape(n_states, n_actions)
    successor_counts = np.zeros((n_states * n_actions, n_states), dtype=np.float64)
    np.add.at(successor_counts, (pairs, states[1:]), 1.0)
    successor_counts = successor_counts.reshape(n_states, n_actions, n_states)
    return {
        "pair_counts": pair_counts.astype(np.int64),
        "reward_sums": reward_sums,
        "successor_counts": successor_counts,
    }


def value_aggregates(
    states: np.ndarray, rewards: np.ndarray, n_states: int
) -> dict[str, np.ndarray]:
    current = states[:-1]
    transition_rewards = rewards[1:].astype(np.float64)
    state_counts = np.bincount(current, minlength=n_states).astype(np.int64)
    reward_sums = np.bincount(current, weights=transition_rewards, minlength=n_states)
    transition_counts = np.zeros((n_states, n_states), dtype=np.float64)
    np.add.at(transition_counts, (current, states[1:]), 1.0)
    return {
        "state_counts": state_counts,
        "reward_sums": reward_sums,
        "transition_counts": transition_counts,
    }


def build_record(
    *,
    family: str,
    task_index: int,
    record_index: int,
    task_seed: np.random.SeedSequence,
    length_seed: np.random.SeedSequence,
    trajectory_length: int,
    mixing: float,
    gap_bonus: float,
    n_states: int,
    n_actions: int,
    pi_min: float,
    gamma: float,
    alpha: float,
    iterations: int,
) -> dict[str, Any]:
    reward_bound = 1.0 + float(gap_bonus)
    value_limit = reward_bound / (1.0 - gamma)
    generation_rng = np.random.default_rng(task_seed)
    if family == "current_unstructured":
        mdp = make_current_mdp(n_states, n_actions, gamma, mixing, gap_bonus, generation_rng)
        hidden: dict[str, Any] | None = None
    elif family == "hidden_cluster":
        mdp, hidden = make_hidden_cluster_mdp(
            n_states, n_actions, gamma, mixing, gap_bonus, generation_rng
        )
    else:
        raise ValueError(f"unknown family {family}")
    policy = make_policy(n_states, n_actions, pi_min, generation_rng)
    exact = policy_quantities(mdp, policy)

    stream_seeds = length_seed.spawn(3)
    streams: dict[str, dict[str, np.ndarray]] = {}
    states_by_name: dict[str, np.ndarray] = {}
    rewards_by_name: dict[str, np.ndarray] = {}
    for name, stream_seed in zip(("value", "signature", "target"), stream_seeds, strict=True):
        stream_rng = np.random.default_rng(stream_seed)
        start = int(stream_rng.choice(n_states, p=exact["mu_state"]))
        states, actions, rewards = rollout(
            mdp, policy, start=start, n=trajectory_length, rng=stream_rng
        )
        states_by_name[name] = states
        rewards_by_name[name] = rewards
        if name != "value":
            streams[name] = stream_aggregates(
                states, actions, rewards, n_states, n_actions
            )

    value_estimate, value_diagnostics = iterative_state_evaluation(
        states_by_name["value"],
        rewards_by_name["value"][1:].astype(np.float64),
        n_states,
        gamma,
        alpha,
        iterations,
        beta=None,
        value_limit=value_limit,
    )
    signature_y = aggregate_y_sums(
        streams["signature"]["reward_sums"],
        streams["signature"]["successor_counts"],
        value_estimate,
        gamma,
    )
    target_y = aggregate_y_sums(
        streams["target"]["reward_sums"],
        streams["target"]["successor_counts"],
        value_estimate,
        gamma,
    )
    observable_inputs = {
        "n_states": n_states,
        "n_actions": n_actions,
        "gamma": gamma,
        "reward_bound": reward_bound,
        "policy": policy,
        "value_estimate": value_estimate,
        "signature_counts": streams["signature"]["pair_counts"],
        "signature_sums": signature_y,
        "target_counts": streams["target"]["pair_counts"],
        "target_sums": target_y,
    }
    record_failure: str | None = None
    try:
        route_outputs = evaluate_all_routes(observable_inputs)
    except KernelInputError as error:
        record_failure = error.reason
        route_outputs = {
            "value_scale": reward_bound / (1.0 - gamma),
            "primary": {"bandwidth": [None] * n_actions},
            "anchor": {"bandwidth": [None] * n_actions},
            "routes": {
                route: {
                    "estimates": [[None] * n_actions for _ in range(n_states)],
                    "reasons": [[error.reason] * n_actions for _ in range(n_states)],
                    "denominators": [[None] * n_actions for _ in range(n_states)],
                    "ess": [[None] * n_actions for _ in range(n_states)],
                }
                for route in ROUTES
            },
        }

    diagnostic_policy: dict[str, Any] = {}
    for route in ROUTES:
        estimates = np.asarray(
            [
                [np.nan if value is None else value for value in row]
                for row in route_outputs["routes"][route]["estimates"]
            ],
            dtype=np.float64,
        )
        updated, statuses = greedy_with_floor_policy(estimates, policy, pi_min)
        row_sums = updated.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-9):
            raise AssertionError("diagnostic policy rows left the simplex")
        diagnostic_policy[route] = {
            "statuses": statuses,
            "updated_states": int(sum(status == "updated" for status in statuses)),
        }

    oracle_audit: dict[str, Any] = {
        "purpose": "truth-based audit only; never an estimator or kernel input",
        "q_pi": exact["q_pi"],
        "v_pi": exact["v_pi"],
        "mu_state": exact["mu_state"],
        "transition": np.asarray(mdp["P"], dtype=np.float64),
        "reward": np.asarray(mdp["R"], dtype=np.float64),
        "p0": np.asarray(mdp["p0"], dtype=np.float64),
        "cluster_labels": (
            None if hidden is None else np.asarray(hidden["cluster_labels"], dtype=np.int64)
        ),
    }
    return {
        "task_id": TASK_ID,
        "record_index": record_index,
        "task_index": task_index,
        "family": family,
        "trajectory_length": trajectory_length,
        "mixing": float(mixing),
        "gap_bonus": float(gap_bonus),
        "n_states": n_states,
        "n_actions": n_actions,
        "pi_min": float(pi_min),
        "gamma": float(gamma),
        "alpha": float(alpha),
        "iterations": iterations,
        "seed_provenance": {
            "task_entropy": task_seed.entropy,
            "task_spawn_key": list(task_seed.spawn_key),
            "length_spawn_key": list(length_seed.spawn_key),
            "stream_spawn_keys": {
                name: list(seed.spawn_key)
                for name, seed in zip(("value", "signature", "target"), stream_seeds, strict=True)
            },
        },
        "observable_inputs": {
            "policy": policy,
            "reward_bound": reward_bound,
            "value_estimate": value_estimate,
            "value_iterations_used": int(value_diagnostics["iterations_used"]),
            "value_aggregates": value_aggregates(
                states_by_name["value"], rewards_by_name["value"], n_states
            ),
            "signature_aggregates": streams["signature"],
            "target_aggregates": streams["target"],
            "signature_counts": streams["signature"]["pair_counts"],
            "signature_sums": signature_y,
            "target_counts": streams["target"]["pair_counts"],
            "target_sums": target_y,
        },
        "record_level_failure": record_failure,
        "route_outputs": route_outputs,
        "diagnostic_policy": diagnostic_policy,
        "oracle_audit": oracle_audit,
    }


def count_bin(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 4:
        return "1-4"
    if count <= 16:
        return "5-16"
    return "17+"


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["family"],
            int(record["trajectory_length"]),
            float(record["mixing"]),
            float(record["gap_bonus"]),
        )
        grouped[key].append(record)
    rows: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1], item[2], item[3])):
        family, length, mixing, gap_bonus = key
        selected = grouped[key]
        for route in ROUTES:
            total_pairs = 0
            finite_pairs = 0
            zero_pairs = 0
            zero_finite = 0
            bin_pairs: dict[str, int] = defaultdict(int)
            for record in selected:
                counts = np.asarray(
                    record["observable_inputs"]["target_counts"], dtype=np.int64
                )
                estimates = record["route_outputs"]["routes"][route]["estimates"]
                for s in range(record["n_states"]):
                    for a in range(record["n_actions"]):
                        total_pairs += 1
                        emitted = estimates[s][a] is not None
                        finite_pairs += int(emitted)
                        bin_label = count_bin(int(counts[s, a]))
                        bin_pairs[bin_label] += 1
                        if counts[s, a] == 0:
                            zero_pairs += 1
                            zero_finite += int(emitted)
            rows.append(
                {
                    "family": family,
                    "trajectory_length": length,
                    "mixing": mixing,
                    "gap_bonus": gap_bonus,
                    "route": route,
                    "records": len(selected),
                    "total_pairs": total_pairs,
                    "finite_estimates": finite_pairs,
                    "finite_rate": finite_pairs / total_pairs,
                    "zero_count_pairs": zero_pairs,
                    "zero_count_finite": zero_finite,
                    "count_bin_pairs": dict(sorted(bin_pairs.items())),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=1)
    parser.add_argument(
        "--families", type=str, nargs="+", default=list(FAMILIES), choices=FAMILIES
    )
    parser.add_argument(
        "--trajectory-lengths", type=int, nargs="+", default=[256, 1024]
    )
    parser.add_argument("--n-states", type=int, default=6)
    parser.add_argument("--n-actions", type=int, default=4)
    parser.add_argument("--pi-min", type=float, default=0.05)
    parser.add_argument("--mixing", type=float, nargs="+", default=[0.08, 0.50])
    parser.add_argument("--gap-bonuses", type=float, nargs="+", default=[0.0, 0.50])
    parser.add_argument("--gamma", type=float, default=0.70)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/FP-KERN-001/claude_smoke"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.tasks <= 0:
        raise ValueError("tasks must be positive")
    if min(args.trajectory_lengths) < 4:
        raise ValueError("trajectory lengths must be at least 4")
    if not 0.0 < args.gamma < 1.0 or not 0.0 < args.alpha <= 1.0:
        raise ValueError("gamma and alpha must lie in (0,1), with alpha allowing 1")
    if not 0.0 < args.pi_min < 1.0 / args.n_actions:
        raise ValueError("pi_min must lie in (0, 1 / n_actions)")
    if args.n_states != 6 or args.n_actions != 4:
        raise ValueError("the frozen task fixes 6 states and 4 actions")
    if any(not 0.0 <= mixing <= 1.0 for mixing in args.mixing):
        raise ValueError("mixing values must lie in [0, 1]")
    if any(gap < 0.0 for gap in args.gap_bonuses):
        raise ValueError("gap bonuses must be nonnegative")
    if args.formal:
        observed = {
            "tasks": args.tasks,
            "families": sorted(args.families),
            "trajectory_lengths": sorted(set(args.trajectory_lengths)),
            "mixing": sorted(args.mixing),
            "gap_bonuses": sorted(args.gap_bonuses),
            "n_states": args.n_states,
            "n_actions": args.n_actions,
            "pi_min": args.pi_min,
            "gamma": args.gamma,
            "alpha": args.alpha,
            "iterations": args.iterations,
            "seed": args.seed,
        }
        expected = dict(FORMAL_PROTOCOL)
        expected.pop("records")
        expected["families"] = sorted(expected["families"])
        if observed != expected:
            raise ValueError(
                f"formal mode requires the exact frozen matrix: {observed} != {expected}"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    project_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output directory {output_dir} is not empty; refusing to overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    command = subprocess.list2cmdline(
        [sys.executable, "-B", Path(__file__).name, *sys.argv[1:]]
    )
    (output_dir / "commands.log").write_text(
        f"WORKDIR: {project_dir}\nEVALUATION: {command}\n", encoding="utf-8"
    )
    checks_log = run_preflight_checks(project_dir)
    (output_dir / "checks.log").write_text(checks_log, encoding="utf-8")

    families = list(args.families)
    lengths = sorted(set(args.trajectory_lengths))
    mixing_grid = sorted(args.mixing)
    gap_grid = sorted(args.gap_bonuses)
    config = {
        "task_id": TASK_ID,
        "mode": "formal" if args.formal else "smoke",
        "tasks_per_cell": args.tasks,
        "families": families,
        "trajectory_lengths": lengths,
        "n_states": args.n_states,
        "n_actions": args.n_actions,
        "pi_min": args.pi_min,
        "mixing": mixing_grid,
        "gap_bonuses": gap_grid,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "iterations": args.iterations,
        "seed": args.seed,
        "routes": list(ROUTES),
        "reward_bound_rule": "1 + gap_bonus",
        "value_scale_rule": "reward_bound / (1 - gamma)",
        "structure_strength": {"prototype": 0.90, "independent": 0.10},
        "streams": ["value", "signature", "target"],
        "activation_commit": ACTIVATION_COMMIT,
        "common_execution_start": COMMON_EXECUTION_START,
    }
    write_json(output_dir / "config.json", config)

    total_task_cells = len(families) * len(mixing_grid) * len(gap_grid) * args.tasks
    seed_sequence = np.random.SeedSequence(args.seed)
    task_seeds = iter(seed_sequence.spawn(total_task_cells))
    records: list[dict[str, Any]] = []
    record_index = 0
    for family in families:
        for mixing in mixing_grid:
            for gap_bonus in gap_grid:
                for task_index in range(args.tasks):
                    task_seed = next(task_seeds)
                    length_seeds = task_seed.spawn(len(lengths))
                    for length, length_seed in zip(lengths, length_seeds, strict=True):
                        records.append(
                            build_record(
                                family=family,
                                task_index=task_index,
                                record_index=record_index,
                                task_seed=task_seed,
                                length_seed=length_seed,
                                trajectory_length=length,
                                mixing=mixing,
                                gap_bonus=gap_bonus,
                                n_states=args.n_states,
                                n_actions=args.n_actions,
                                pi_min=args.pi_min,
                                gamma=args.gamma,
                                alpha=args.alpha,
                                iterations=args.iterations,
                            )
                        )
                        record_index += 1

    expected_records = (
        total_task_cells * len(lengths)
    )
    if len(records) != expected_records:
        raise AssertionError("record count differs from the frozen matrix")
    if args.formal and len(records) != FORMAL_PROTOCOL["records"]:
        raise AssertionError("formal run must produce exactly 480 records")

    summary = summarize(records)
    write_json(output_dir / "task_results.json", records)
    write_json(output_dir / "summary.json", summary)

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "working_directory": str(project_dir),
        "script": str(Path(__file__).resolve()),
        "git_head_at_run": git_head,
        "task_id": TASK_ID,
        "activation_commit": ACTIVATION_COMMIT,
        "common_execution_start": COMMON_EXECUTION_START,
    }
    write_json(output_dir / "environment.json", environment)

    failures = sum(1 for record in records if record["record_level_failure"] is not None)
    print(f"wrote {len(records)} records to {output_dir}")
    print(f"record-level failures: {failures}")
    print(f"PASS {TASK_ID} evaluation ({'formal' if args.formal else 'smoke'})")


if __name__ == "__main__":
    main()
