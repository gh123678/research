"""Frozen paired evaluator for FP-KERN-001."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_fixed_policy_q_routes import (
    grouped_exact_mean,
    iterative_state_evaluation,
    make_policy,
)
from kernel_generalization_mdps import make_environment
from kernel_state_generalization import ROUTE_NAMES, build_kernel_routes
from mdps import rollout
from verify_fixed_policy_q_routes import policy_quantities


TASK_ID = "FP-KERN-001"
TASK_VERSION = "1.0"
COMMON_ROUTE_BASELINE = "28c4ae0f68ca51c7c9a0fd981159e85b7742dd4c"
FAMILIES = ("current_unstructured", "hidden_cluster")
FORMAL_LENGTHS = (256, 1024, 4096, 16384)
SMOKE_LENGTHS = (256, 1024)
MIXING_GRID = (0.08, 0.50)
GAP_GRID = (0.0, 0.50)
ALPHA = 0.65


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError("nonfinite float cannot be serialized")
        return converted
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _stream_components(
    seed: int,
    family_index: int,
    trajectory_length: int,
    mixing_index: int,
    gap_index: int,
    task_index: int,
) -> list[int]:
    return [seed, family_index, trajectory_length, mixing_index, gap_index, task_index]


def _stream_generators(components: list[int]) -> list[np.random.Generator]:
    sequence = np.random.SeedSequence(components)
    return [np.random.default_rng(child) for child in sequence.spawn(5)]


def _trajectory(
    mdp: dict[str, Any],
    policy: np.ndarray,
    stationary: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = int(rng.choice(int(mdp["nS"]), p=stationary))
    states, actions, rewards = rollout(mdp, policy, start, length, rng=rng)
    return states, actions, rewards


def _pair_aggregates(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    value: np.ndarray,
    gamma: float,
    n_states: int,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_states = states[:-1]
    current_actions = actions[:-1]
    next_states = states[1:]
    targets = rewards[1:] + gamma * value[next_states]
    pairs = current_states * n_actions + current_actions
    q_values, counts = grouped_exact_mean(targets, pairs, n_states * n_actions)
    sums = np.bincount(
        pairs, weights=targets, minlength=n_states * n_actions
    ).astype(np.float64)
    return (
        q_values.reshape(n_states, n_actions),
        counts.reshape(n_states, n_actions),
        sums.reshape(n_states, n_actions),
    )


def _optional_error(estimate: list[list[float | None]], truth: np.ndarray) -> list[list[float | None]]:
    output: list[list[float | None]] = []
    for state, row in enumerate(estimate):
        output.append(
            [
                None if value is None else float(value - truth[state, action])
                for action, value in enumerate(row)
            ]
        )
    return output


def _oracle_route_audit(
    mdp: dict[str, Any],
    old_value: np.ndarray,
    old_return: float,
    truth_q: np.ndarray,
    route: dict[str, Any],
) -> dict[str, Any]:
    diagnostic_policy = np.asarray(
        route["diagnostic_policy"]["policy"], dtype=np.float64
    )
    quantities = policy_quantities(mdp, diagnostic_policy)
    new_value = np.asarray(quantities["v_pi"], dtype=np.float64)
    new_return = float(np.asarray(mdp["p0"], dtype=np.float64) @ new_value)
    return {
        "q_error": _optional_error(route["estimate"], truth_q),
        "diagnostic_value": new_value.tolist(),
        "componentwise_value_change": (new_value - old_value).tolist(),
        "diagnostic_return": new_return,
        "return_change": new_return - old_return,
    }


def build_record(
    *,
    record_index: int,
    seed: int,
    family: str,
    family_index: int,
    trajectory_length: int,
    mixing: float,
    mixing_index: int,
    gap_bonus: float,
    gap_index: int,
    task_index: int,
    n_states: int,
    n_actions: int,
    pi_min: float,
    gamma: float,
    alpha: float,
    iterations: int,
) -> dict[str, Any]:
    components = _stream_components(
        seed,
        family_index,
        trajectory_length,
        mixing_index,
        gap_index,
        task_index,
    )
    mdp_rng, policy_rng, value_rng, signature_rng, target_rng = _stream_generators(
        components
    )
    mdp, hidden_audit = make_environment(
        family,
        n_states,
        n_actions,
        gamma,
        mixing,
        gap_bonus,
        mdp_rng,
    )
    policy = make_policy(n_states, n_actions, pi_min, policy_rng)
    truth = policy_quantities(mdp, policy)
    stationary = np.asarray(truth["mu_state"], dtype=np.float64)
    value_states, _, value_rewards = _trajectory(
        mdp, policy, stationary, trajectory_length, value_rng
    )
    signature_states, signature_actions, signature_rewards = _trajectory(
        mdp, policy, stationary, trajectory_length, signature_rng
    )
    target_states, target_actions, target_rewards = _trajectory(
        mdp, policy, stationary, trajectory_length, target_rng
    )
    reward_bound = 1.0 + gap_bonus
    value_bound = reward_bound / (1.0 - gamma)
    value_hat, value_diagnostics = iterative_state_evaluation(
        value_states,
        value_rewards[1:],
        n_states,
        gamma,
        alpha,
        iterations,
        None,
        value_bound,
    )
    signature_q, signature_counts, _ = _pair_aggregates(
        signature_states,
        signature_actions,
        signature_rewards,
        value_hat,
        gamma,
        n_states,
        n_actions,
    )
    _, target_counts, target_sums = _pair_aggregates(
        target_states,
        target_actions,
        target_rewards,
        value_hat,
        gamma,
        n_states,
        n_actions,
    )
    observable_inputs: dict[str, Any] = {
        "signature_q": signature_q,
        "signature_counts": signature_counts,
        "target_sums": target_sums,
        "target_counts": target_counts,
        "current_policy": policy,
        "pi_min": pi_min,
        "value_bound": value_bound,
    }
    kernel_output = build_kernel_routes(observable_inputs)

    true_v = np.asarray(truth["v_pi"], dtype=np.float64)
    true_q = np.asarray(truth["q_pi"], dtype=np.float64)
    old_return = float(np.asarray(mdp["p0"], dtype=np.float64) @ true_v)
    route_audit = {
        route_name: _oracle_route_audit(
            mdp,
            true_v,
            old_return,
            true_q,
            kernel_output["routes"][route_name],
        )
        for route_name in ROUTE_NAMES
    }
    oracle_audit = {
        "true_v": true_v.tolist(),
        "true_q": true_q.tolist(),
        "old_return": old_return,
        "routes": route_audit,
        "generator": hidden_audit,
    }
    return {
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "record_index": record_index,
        "environment_family": family,
        "trajectory_length": trajectory_length,
        "mixing": mixing,
        "gap_bonus": gap_bonus,
        "task_index": task_index,
        "seed_components": components,
        "value_estimate": value_hat.tolist(),
        "value_diagnostics": value_diagnostics,
        "observable_inputs": _json_ready(observable_inputs),
        "kernel_generalization": kernel_output,
        "oracle_audit": oracle_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--trajectory-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--mixing", type=float, nargs="+", required=True)
    parser.add_argument("--gap-bonuses", type=float, nargs="+", required=True)
    parser.add_argument("--n-states", type=int, required=True)
    parser.add_argument("--n-actions", type=int, required=True)
    parser.add_argument("--pi-min", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    expected_lengths = SMOKE_LENGTHS if args.mode == "smoke" else FORMAL_LENGTHS
    expected_tasks = 1 if args.mode == "smoke" else 15
    if args.tasks != expected_tasks:
        raise ValueError(f"{args.mode} tasks must equal {expected_tasks}")
    if tuple(args.families) != FAMILIES:
        raise ValueError(f"families must equal {FAMILIES}")
    if tuple(args.trajectory_lengths) != expected_lengths:
        raise ValueError(f"trajectory lengths must equal {expected_lengths}")
    if tuple(args.mixing) != MIXING_GRID or tuple(args.gap_bonuses) != GAP_GRID:
        raise ValueError("mixing and gap grids differ from the frozen contract")
    if (args.n_states, args.n_actions) != (6, 4):
        raise ValueError("state/action dimensions differ from the frozen contract")
    if args.pi_min != 0.05 or args.gamma != 0.70 or args.alpha != ALPHA:
        raise ValueError("pi_min, gamma, or alpha differs from the frozen contract")
    if args.iterations != 160 or args.seed != 20260909:
        raise ValueError("iterations or seed differs from the frozen contract")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output_dir = args.output_dir.resolve()
    _prepare_output(output_dir)
    config = {
        "task_id": TASK_ID,
        "task_version": TASK_VERSION,
        "mode": args.mode,
        "common_route_baseline": COMMON_ROUTE_BASELINE,
        "families": list(FAMILIES),
        "tasks": args.tasks,
        "trajectory_lengths": list(args.trajectory_lengths),
        "mixing": list(args.mixing),
        "gap_bonuses": list(args.gap_bonuses),
        "n_states": args.n_states,
        "n_actions": args.n_actions,
        "pi_min": args.pi_min,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "iterations": args.iterations,
        "seed": args.seed,
        "routes": list(ROUTE_NAMES),
        "count_bins": ["0", "1-4", "5-16", "17+"],
        "prototype_weight": 0.90,
        "independent_weight": 0.10,
    }
    records: list[dict[str, Any]] = []
    record_index = 0
    for family_index, family in enumerate(args.families):
        for trajectory_length in args.trajectory_lengths:
            for mixing_index, mixing in enumerate(args.mixing):
                for gap_index, gap_bonus in enumerate(args.gap_bonuses):
                    for task_index in range(args.tasks):
                        records.append(
                            build_record(
                                record_index=record_index,
                                seed=args.seed,
                                family=family,
                                family_index=family_index,
                                trajectory_length=trajectory_length,
                                mixing=mixing,
                                mixing_index=mixing_index,
                                gap_bonus=gap_bonus,
                                gap_index=gap_index,
                                task_index=task_index,
                                n_states=args.n_states,
                                n_actions=args.n_actions,
                                pi_min=args.pi_min,
                                gamma=args.gamma,
                                alpha=args.alpha,
                                iterations=args.iterations,
                            )
                        )
                        record_index += 1
    expected = len(FAMILIES) * len(args.trajectory_lengths) * 2 * 2 * args.tasks
    if len(records) != expected:
        raise AssertionError("record count mismatch")

    _write_json_atomic(output_dir / "config.json", config)
    _write_json_atomic(output_dir / "task_results.json", records)
    _write_json_atomic(
        output_dir / "summary.json", {"status": "pending_strict_analyzer"}
    )
    _write_json_atomic(
        output_dir / "analysis.json", {"status": "pending_strict_analyzer"}
    )
    _write_json_atomic(
        output_dir / "environment.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "script": str(Path(__file__).resolve()),
            "common_route_baseline": COMMON_ROUTE_BASELINE,
        },
    )
    command = subprocess.list2cmdline([sys.executable, "-B", *sys.argv])
    (output_dir / "commands.log").write_text(
        f"WORKDIR: {Path.cwd()}\nEVALUATION: {command}\n", encoding="utf-8"
    )
    (output_dir / "checks.log").write_text(
        f"PASS evaluator generated {len(records)} records\n", encoding="utf-8"
    )
    hashes = {
        name: sha256_file(output_dir / name)
        for name in ("config.json", "task_results.json", "environment.json")
    }
    print(f"PASS {TASK_ID} {args.mode} evaluation with {len(records)} records")
    print(json.dumps(hashes, sort_keys=True))


if __name__ == "__main__":
    main()

