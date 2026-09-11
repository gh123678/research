"""Frozen formal evaluator for FP-ESARSA-001 (Claude main route).

Runs the three frozen Expected SARSA routes over the shared frozen matrix,
builds the held-out residual certificate, applies the relative-softmax
decision rule, and audits exact truth only inside ``oracle_audit``.

Frozen protocol (task sheet ``docs/research_tasks/FP-ESARSA-001.md``):
30 tasks per cell, 6 states, 4 actions, lengths 256/1024/4096/16384,
contiguous half split, mixing 0.08 and 0.5, gap bonuses 0 and 0.5,
pi_min 0.05, R_star 1.5, gamma 0.70, alpha 0.65, 160 layers,
zeta=xi=tau=8, eta grid descending (1.0, 0.5, 0.2, 0.1, 0.05),
delta 0.05, 15 mixture components, seed 20260829, 480 matched records.

The MDP generator, fixed-policy generator, stationary-start rule, trajectory
sampler, task seed spawning, and truth computation are inherited unchanged
from ``evaluate_fixed_policy_q_routes.py`` / ``evaluate_visit_indexed_certificates.py``;
the generator identity is regression-checked against the frozen FP-TU-001
baseline record by record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import fixed_policy_expected_sarsa as es
from evaluate_fixed_policy_q_routes import (
    json_ready,
    make_mdp,
    make_policy,
    write_json,
)
from mdps import rollout
from verify_fixed_policy_q_routes import one_hot_population_kernel, policy_quantities

TASK_ID = "FP-ESARSA-001"
ROUTES = ("expected_exact", "expected_finite", "sampled_exact")

FROZEN_N_STATES = 6
FROZEN_N_ACTIONS = 4
FROZEN_PI_MIN = 0.05
FROZEN_LENGTHS = (256, 1024, 4096, 16384)
FROZEN_MIXING = (0.08, 0.5)
FROZEN_GAP = (0.0, 0.5)
FROZEN_GAMMA = 0.70
FROZEN_ALPHA = 0.65
FROZEN_LAYERS = 160
FROZEN_ZETA = 8.0
FROZEN_XI = 8.0
FROZEN_TAU = 8.0
FROZEN_DELTA = 0.05
FROZEN_SEED = 20260829
FROZEN_TASKS = 30
FROZEN_REWARD_BOUND = 1.5
FROZEN_VALUE_BOUND = FROZEN_REWARD_BOUND / (1.0 - FROZEN_GAMMA)
IDENTITY_TOLERANCE = 1e-12
AUDIT_TOLERANCE = 1e-9
PREFLIGHT_SCRIPT = "verify_fixed_policy_expected_sarsa.py"


# --------------------------------------------------------------------------- #
# Memoised mixture boundary (semantics-neutral speed-up)
# --------------------------------------------------------------------------- #
_MIXTURE_CACHE: dict[int, dict[str, float]] = {}
_ORIGINAL_SOLVE = es.solve_mixture_boundary


def _memoised_solve(count, grid, *, n_groups, delta, tolerance):
    key = int(count)
    if key not in _MIXTURE_CACHE:
        _MIXTURE_CACHE[key] = _ORIGINAL_SOLVE(
            count, grid, n_groups=n_groups, delta=delta, tolerance=tolerance
        )
    return _MIXTURE_CACHE[key]


es.solve_mixture_boundary = _memoised_solve


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_preflight(project_dir: Path) -> str:
    command = [sys.executable, "-B", PREFLIGHT_SCRIPT]
    completed = subprocess.run(
        command, cwd=project_dir, capture_output=True, text=True, check=False
    )
    log = "\n".join(
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
        raise RuntimeError("preflight verifier failed:\n" + log)
    return log


def load_baseline_identity(baseline_dir: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    path = baseline_dir / "task_results.json"
    if not path.exists():
        raise RuntimeError(f"frozen FP-TU-001 baseline missing: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 480:
        raise RuntimeError("frozen FP-TU-001 baseline must contain 480 records")
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            float(record["gap_bonus"]),
            float(record["mixing"]),
            int(record["task_index"]),
            int(record["trajectory_length"]),
        )
        indexed[key] = record
    if len(indexed) != 480:
        raise RuntimeError("frozen baseline identity keys are not unique")
    return indexed


def empirical_route_kernel(counts: np.ndarray, sharpness: float | None) -> np.ndarray:
    """Row-stochastic writeback kernel from training pair counts.

    ``sharpness=None`` is the exact route (one-hot writeback, identity kernel);
    a finite sharpness uses the frozen one-hot population form with the
    empirical occupancy ``N_x / m`` instead of the oracle occupancy.
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    total = float(counts.sum())
    occupancy = counts / total if total > 0.0 else counts
    if sharpness is None:
        return np.eye(counts.size, dtype=np.float64)
    return one_hot_population_kernel(occupancy, float(sharpness))


def split_trajectory(states, actions, rewards, n_states, n_actions, length: int) -> dict:
    """Contiguous half split: first half trains, second half is held out."""
    half = length // 2
    if half < 1 or length - half < 1:
        raise ValueError("trajectory length must yield nonempty train and held-out halves")
    train = {
        "states": states[:half],
        "actions": actions[:half],
        "rewards": rewards[1 : half + 1].astype(np.float64),
        "next_states": states[1 : half + 1],
    }
    heldout = {
        "states": states[half:length],
        "actions": actions[half:length],
        "rewards": rewards[half + 1 : length + 1].astype(np.float64),
        "next_states": states[half + 1 : length + 1],
    }
    # Structural isolation: the two index windows are disjoint and contiguous.
    train_idx = set(range(0, half))
    held_idx = set(range(half, length))
    if train_idx & held_idx or len(train_idx) + len(held_idx) != length:
        raise AssertionError("train/held-out split is not a disjoint contiguous partition")
    return {"train": train, "heldout": heldout, "half": half}


def run_route(route: str, policy, train, *, length: int) -> dict:
    q0 = np.zeros((FROZEN_N_STATES, FROZEN_N_ACTIONS), dtype=np.float64)
    common = dict(
        q0=q0,
        policy=policy,
        states=train["states"],
        actions=train["actions"],
        rewards=train["rewards"],
        next_states=train["next_states"],
        gamma=FROZEN_GAMMA,
        alpha=FROZEN_ALPHA,
        layers=FROZEN_LAYERS,
        value_bound=FROZEN_VALUE_BOUND,
    )
    if route == "expected_exact":
        result = es.run_expected_exact(**common)
    elif route == "expected_finite":
        result = es.run_expected_finite(
            **common, zeta=FROZEN_ZETA, xi=FROZEN_XI, tau=FROZEN_TAU
        )
    elif route == "sampled_exact":
        result = es.run_sampled_exact(**common, next_actions=train["next_actions"])
    else:
        raise ValueError(f"unknown route {route}")
    return result


def certificate_for(route: str, q_hat, policy, heldout) -> dict:
    return es.build_residual_certificate(
        q_hat=q_hat,
        policy=policy,
        states=heldout["states"],
        actions=heldout["actions"],
        rewards=heldout["rewards"],
        next_states=heldout["next_states"],
        reward_bound=FROZEN_REWARD_BOUND,
        gamma=FROZEN_GAMMA,
        delta=FROZEN_DELTA,
    )


def improvement_for(policy, q_hat, certificate) -> dict:
    if certificate["status"] != "certificate_emitted":
        return {
            "status": "not_attempted",
            "failure_reasons": [],
            "eta_selected": None,
            "lb_by_state": [0.0] * FROZEN_N_STATES,
            "policy_plus": None,
            "changed": False,
        }
    decision = es.decide_policy_update(policy, q_hat, certificate["e_q"])
    plus = np.asarray(decision["policy_plus"], dtype=np.float64)
    changed = bool(np.max(np.abs(plus - policy)) > 1e-12)
    return {
        "status": decision["status"],
        "failure_reasons": list(decision["failure_reasons"]),
        "eta_selected": decision["eta_selected"],
        "lb_by_state": [float(value) for value in np.asarray(decision["lb_by_state"]).reshape(-1)],
        "policy_plus": plus if decision["status"] == "safe_update_emitted" else None,
        "changed": changed,
    }


def route_failure_reasons(certificate: dict, improvement: dict) -> list[str]:
    reasons = list(certificate["failure_reasons"]) + list(improvement["failure_reasons"])
    unique = sorted(
        set(reasons),
        key=lambda reason: (es._REASON_RANK.get(reason, len(es._REASON_ORDER)), reason),
    )
    return unique


def oracle_route_audit(
    route: str, q_hat, policy, certificate, improvement, exact
) -> dict:
    q_pi = np.asarray(exact["q_pi"], dtype=np.float64)
    v_pi = np.asarray(exact["v_pi"], dtype=np.float64)
    actual_error = float(np.max(np.abs(q_hat - q_pi)))
    emitted = certificate["status"] == "certificate_emitted"
    bound = certificate["e_q"]
    violation = (
        None
        if not emitted or bound is None
        else bool(actual_error > float(bound) + AUDIT_TOLERANCE)
    )
    means = np.asarray(certificate["residual_means"], dtype=np.float64)
    radii = np.asarray(certificate["radii"], dtype=np.float64)
    supplied = radii > 0.0
    residual_excess = float(np.max(np.abs(means[supplied]) - radii[supplied])) if np.any(supplied) else None
    residual_violation = (
        None if residual_excess is None else bool(residual_excess > AUDIT_TOLERANCE)
    )
    value_decrease = None
    value_min_delta = None
    bellman_improvement_min = None
    if improvement["status"] == "safe_update_emitted" and improvement["policy_plus"] is not None:
        new_policy = np.asarray(improvement["policy_plus"], dtype=np.float64)
        new_exact = policy_quantities(_CURRENT_MDP, new_policy)
        new_v = np.asarray(new_exact["v_pi"], dtype=np.float64)
        delta_v = new_v - v_pi
        value_min_delta = float(np.min(delta_v))
        value_decrease = bool(value_min_delta < -AUDIT_TOLERANCE)
        p = np.asarray(_CURRENT_MDP["P"], dtype=np.float64)
        reward = np.asarray(_CURRENT_MDP["R"], dtype=np.float64)
        reward_sa = np.sum(p * reward, axis=2)
        reward_new = np.sum(new_policy * reward_sa, axis=1)
        p_pi_new = np.einsum("sa,san->sn", new_policy, p)
        t_new = reward_new + FROZEN_GAMMA * (p_pi_new @ v_pi)
        bellman_improvement_min = float(np.min(t_new - v_pi))
    return {
        "route": route,
        "certificate_emitted": emitted,
        "actual_q_sup_error": actual_error,
        "certified_e_q": None if bound is None else float(bound),
        "bound_slack": None if bound is None else float(bound) - actual_error,
        "certificate_violation": violation,
        "residual_event_max_excess": residual_excess,
        "residual_event_violation": residual_violation,
        "update_emitted": improvement["status"] == "safe_update_emitted",
        "oracle_value_decrease": value_decrease,
        "oracle_value_min_delta": value_min_delta,
        "oracle_bellman_improvement_min": bellman_improvement_min,
    }


_CURRENT_MDP: dict | None = None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--trajectory-lengths", type=int, nargs="+", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/FP-ESARSA-001/claude"),
    )
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--label", type=str, default="")
    return parser.parse_args()


def resolve_matrix(args: argparse.Namespace) -> tuple[int, tuple[int, ...]]:
    if args.mode == "formal":
        tasks = FROZEN_TASKS
        lengths = FROZEN_LENGTHS
        if args.tasks is not None and args.tasks != FROZEN_TASKS:
            raise SystemExit("formal mode forbids changing the frozen 30 tasks per cell")
        if args.trajectory_lengths is not None and tuple(sorted(args.trajectory_lengths)) != FROZEN_LENGTHS:
            raise SystemExit("formal mode forbids changing the frozen lengths")
        return tasks, lengths
    tasks = args.tasks if args.tasks is not None else 2
    lengths = tuple(sorted(set(args.trajectory_lengths))) if args.trajectory_lengths else (256, 1024)
    if tasks <= 0 or tasks > FROZEN_TASKS:
        raise SystemExit("smoke tasks must lie in (0, 30]")
    if min(lengths) < 4 or any(length not in FROZEN_LENGTHS for length in lengths):
        raise SystemExit("smoke lengths must be a subset of the frozen lengths")
    return tasks, lengths


def default_baseline_dir(project_dir: Path) -> Path:
    # project_dir = <repo>/icrl_softmax/results/FP-ESARSA-001/claude_worktree/icrl_softmax
    results_root = project_dir.parents[2]
    return results_root / "FP-TU-001" / "claude"


def main() -> None:
    global _CURRENT_MDP
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    tasks, lengths = resolve_matrix(args)
    maximum_length = max(lengths)

    baseline_dir = (args.baseline_dir or default_baseline_dir(project_dir)).resolve()
    baseline = load_baseline_identity(baseline_dir)

    output_dir = args.output_dir.resolve()
    if (output_dir / "task_results.json").exists() and not args.allow_overwrite:
        raise SystemExit(
            f"refusing to overwrite existing results in {output_dir}; "
            "use a fresh directory or --allow-overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    checks_log = run_preflight(project_dir)
    (output_dir / "checks.log").write_text(checks_log, encoding="utf-8")
    command = subprocess.list2cmdline([sys.executable, "-B", Path(__file__).name, *sys.argv[1:]])
    (output_dir / "commands.log").write_text(
        f"WORKDIR: {project_dir}\nMODE: {args.mode}\nLABEL: {args.label}\nEVALUATION: {command}\n",
        encoding="utf-8",
    )

    config = {
        "task_id": TASK_ID,
        "mode": args.mode,
        "label": args.label,
        "tasks_per_cell": tasks,
        "trajectory_lengths": list(lengths),
        "n_states": FROZEN_N_STATES,
        "n_actions": FROZEN_N_ACTIONS,
        "pi_min": FROZEN_PI_MIN,
        "mixing": list(FROZEN_MIXING),
        "gap_bonuses": list(FROZEN_GAP),
        "gamma": FROZEN_GAMMA,
        "alpha": FROZEN_ALPHA,
        "layers": FROZEN_LAYERS,
        "zeta": FROZEN_ZETA,
        "xi": FROZEN_XI,
        "tau": FROZEN_TAU,
        "eta_grid": list(es.ETA_GRID),
        "certificate_delta": FROZEN_DELTA,
        "mixture_components": 15,
        "reward_bound": FROZEN_REWARD_BOUND,
        "value_bound": FROZEN_VALUE_BOUND,
        "seed": FROZEN_SEED,
        "routes": list(ROUTES),
        "algorithm_mode": es.ALGORITHM_MODE,
        "split": "contiguous_first_half_training_second_half_heldout",
        "ordered_non_emission_reasons": list(es._REASON_ORDER),
        "frozen_baseline_identity": {
            "directory": str(baseline_dir),
            "record_count": len(baseline),
        },
    }
    write_json(output_dir / "config.json", config)

    seed_sequence = np.random.SeedSequence(FROZEN_SEED)
    total_seed_cells = (
        len([FROZEN_N_ACTIONS])
        * len([FROZEN_PI_MIN])
        * len(FROZEN_MIXING)
        * len(FROZEN_GAP)
        * FROZEN_TASKS
    )
    if total_seed_cells != 120:
        raise AssertionError("frozen seed schedule must spawn exactly 120 cells")
    child_seeds = iter(seed_sequence.spawn(total_seed_cells))

    task_results: list[dict[str, Any]] = []
    identity_mismatches: list[dict[str, Any]] = []

    for n_actions in (FROZEN_N_ACTIONS,):
        for pi_min in (FROZEN_PI_MIN,):
            for mixing in FROZEN_MIXING:
                for gap_bonus in FROZEN_GAP:
                    reward_bound = FROZEN_REWARD_BOUND
                    cell_seeds = [next(child_seeds) for _ in range(FROZEN_TASKS)]
                    for task_index in range(tasks):
                        child_seed = cell_seeds[task_index]
                        rng = np.random.default_rng(child_seed)
                        mdp = make_mdp(
                            FROZEN_N_STATES, n_actions, FROZEN_GAMMA, mixing, gap_bonus, rng
                        )
                        policy = make_policy(FROZEN_N_STATES, n_actions, pi_min, rng)
                        exact = policy_quantities(mdp, policy)
                        _CURRENT_MDP = mdp
                        start = int(rng.choice(FROZEN_N_STATES, p=exact["mu_state"]))
                        states, actions, rewards = rollout(
                            mdp, policy, start=start, n=maximum_length, rng=rng
                        )
                        sorted_q = np.sort(exact["q_pi"], axis=1)
                        action_gaps = sorted_q[:, -1] - sorted_q[:, -2]
                        actual_reward_abs_max = float(np.max(np.abs(np.asarray(mdp["R"]))))

                        for length in lengths:
                            split = split_trajectory(
                                states, actions, rewards,
                                FROZEN_N_STATES, n_actions, length,
                            )
                            train = dict(split["train"])
                            train["next_actions"] = actions[1 : split["half"] + 1]
                            heldout = split["heldout"]

                            routes: dict[str, Any] = {}
                            for route in ROUTES:
                                result = run_route(route, policy, train, length=length)
                                q_hat = np.asarray(result["q_hat"], dtype=np.float64)
                                certificate = certificate_for(route, q_hat, policy, heldout)
                                improvement = improvement_for(policy, q_hat, certificate)
                                counts = np.asarray(result["train_pair_counts"], dtype=np.int64)
                                sharpness = None if route in ("expected_exact", "sampled_exact") else FROZEN_TAU
                                kernel = empirical_route_kernel(counts, sharpness)
                                diagonal_min = float(np.diag(kernel).min())
                                routes[route] = {
                                    "status": certificate["status"],
                                    "failure_reasons": route_failure_reasons(certificate, improvement),
                                    "diverged": bool(result["diverged"]),
                                    "layers_used": int(len(result["layer_snapshots"])),
                                    "q_hat": q_hat.reshape(-1),
                                    "q_hat_shape": [FROZEN_N_STATES, FROZEN_N_ACTIONS],
                                    "train_pair_counts": counts,
                                    "certificate": {
                                        "e_q": certificate["e_q"],
                                        "epsilon_res": certificate["epsilon_res"],
                                        "residual_means": certificate["residual_means"].reshape(-1),
                                        "radii": certificate["radii"].reshape(-1),
                                        "n_groups": certificate["n_groups"],
                                    },
                                    "contraction": {
                                        "writeback": "identity" if sharpness is None else "one_hot_population",
                                        "sharpness": sharpness,
                                        "diagonal_min": diagonal_min,
                                        "premise_satisfied": bool(
                                            es.contraction_premise_satisfied(kernel, FROZEN_GAMMA)
                                        ),
                                    },
                                    "improvement": improvement,
                                }

                            oracle_routes = {
                                route: oracle_route_audit(
                                    route,
                                    np.asarray(routes[route]["q_hat"], dtype=np.float64).reshape(
                                        FROZEN_N_STATES, FROZEN_N_ACTIONS
                                    ),
                                    policy,
                                    {
                                        "status": routes[route]["status"],
                                        "failure_reasons": routes[route]["failure_reasons"],
                                        "e_q": routes[route]["certificate"]["e_q"],
                                        "residual_means": np.asarray(
                                            routes[route]["certificate"]["residual_means"]
                                        ),
                                        "radii": np.asarray(routes[route]["certificate"]["radii"]),
                                    },
                                    routes[route]["improvement"],
                                    exact,
                                )
                                for route in ROUTES
                            }

                            identity_key = (float(gap_bonus), float(mixing), int(task_index), int(length))
                            base = baseline.get(identity_key)
                            if base is None:
                                raise RuntimeError(f"baseline missing identity key {identity_key}")
                            generator_identity = {
                                "seed_entropy": int(child_seed.entropy),
                                "spawn_key": list(child_seed.spawn_key),
                                "true_pair_occupancy_min": float(np.min(exact["mu_pair"])),
                                "true_action_gap_min": float(np.min(action_gaps)),
                                "true_action_gap_mean": float(np.mean(action_gaps)),
                            }
                            for field in ("seed_entropy", "spawn_key"):
                                if json_ready(generator_identity[field]) != json_ready(base[field]):
                                    identity_mismatches.append(
                                        {
                                            "key": list(identity_key),
                                            "field": field,
                                            "expected": json_ready(base[field]),
                                            "observed": json_ready(generator_identity[field]),
                                        }
                                    )
                            for field in (
                                "true_pair_occupancy_min",
                                "true_action_gap_min",
                                "true_action_gap_mean",
                            ):
                                expected = float(base[field])
                                observed = float(generator_identity[field])
                                if abs(expected - observed) > IDENTITY_TOLERANCE * (
                                    1.0 + max(abs(expected), abs(observed))
                                ):
                                    identity_mismatches.append(
                                        {
                                            "key": list(identity_key),
                                            "field": field,
                                            "expected": expected,
                                            "observed": observed,
                                        }
                                    )

                            exact_q = np.asarray(routes["expected_exact"]["q_hat"]).reshape(
                                FROZEN_N_STATES, FROZEN_N_ACTIONS
                            )
                            finite_q = np.asarray(routes["expected_finite"]["q_hat"]).reshape(
                                FROZEN_N_STATES, FROZEN_N_ACTIONS
                            )
                            sampled_q = np.asarray(routes["sampled_exact"]["q_hat"]).reshape(
                                FROZEN_N_STATES, FROZEN_N_ACTIONS
                            )
                            oracle_audit = {
                                "purpose": "empirical truth-based audit only; not proof or certificate input",
                                "generator_identity": generator_identity,
                                "reward_bound": {
                                    "actual_reward_abs_max": actual_reward_abs_max,
                                    "declared_reward_abs_bound": FROZEN_REWARD_BOUND,
                                    "satisfied": bool(
                                        actual_reward_abs_max <= FROZEN_REWARD_BOUND + 1e-12
                                    ),
                                },
                                "routes": oracle_routes,
                                "any_certificate_violation": any(
                                    audit["certificate_violation"] is True
                                    for audit in oracle_routes.values()
                                ),
                                "any_residual_event_violation": any(
                                    audit["residual_event_violation"] is True
                                    for audit in oracle_routes.values()
                                ),
                                "any_value_decrease": any(
                                    audit["oracle_value_decrease"] is True
                                    for audit in oracle_routes.values()
                                ),
                                "any_contraction_premise_failure": any(
                                    routes[route]["contraction"]["premise_satisfied"] is False
                                    for route in ROUTES
                                ),
                            }

                            task_results.append(
                                {
                                    "task_index": int(task_index),
                                    "identity": {
                                        "seed_entropy": int(child_seed.entropy),
                                        "spawn_key": list(child_seed.spawn_key),
                                    },
                                    "trajectory_length": int(length),
                                    "n_states": FROZEN_N_STATES,
                                    "n_actions": int(n_actions),
                                    "pi_min": float(pi_min),
                                    "mixing": float(mixing),
                                    "gap_bonus": float(gap_bonus),
                                    "train_count": int(split["half"]),
                                    "heldout_count": int(length - split["half"]),
                                    "reward_bound": float(reward_bound),
                                    "gamma": FROZEN_GAMMA,
                                    "alpha": FROZEN_ALPHA,
                                    "layers": FROZEN_LAYERS,
                                    "cross_route": {
                                        "expected_exact_vs_sampled_exact_q_sup_gap": float(
                                            np.max(np.abs(exact_q - sampled_q))
                                        ),
                                        "expected_exact_vs_expected_finite_q_sup_gap": float(
                                            np.max(np.abs(exact_q - finite_q))
                                        ),
                                    },
                                    "routes": routes,
                                    "oracle_audit": oracle_audit,
                                }
                            )

    if len(task_results) != len(lengths) * len(FROZEN_MIXING) * len(FROZEN_GAP) * tasks:
        raise AssertionError("record count does not match the matrix")

    write_json(output_dir / "task_results.json", task_results)

    regression = {
        "status": "pass" if not identity_mismatches else "fail",
        "baseline_directory": str(baseline_dir),
        "baseline_task_results_sha256": sha256_file(baseline_dir / "task_results.json"),
        "compared_records": len(task_results),
        "identity_fields": [
            "seed_entropy",
            "spawn_key",
            "true_pair_occupancy_min",
            "true_action_gap_min",
            "true_action_gap_mean",
        ],
        "tolerance": IDENTITY_TOLERANCE,
        "mismatch_count": len(identity_mismatches),
        "mismatches": identity_mismatches[:50],
    }
    write_json(output_dir / "regression.json", regression)
    if identity_mismatches:
        raise RuntimeError(
            f"generator identity regression failed with {len(identity_mismatches)} mismatches"
        )

    environment = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "working_directory": str(project_dir),
        "script": str(Path(__file__).resolve()),
        "task_id": TASK_ID,
        "mode": args.mode,
        "seed_schedule_cells": total_seed_cells,
        "mixture_boundary_cache_entries": len(_MIXTURE_CACHE),
    }
    write_json(output_dir / "environment.json", environment)

    emissions = Counter()
    for record in task_results:
        for route in ROUTES:
            if record["routes"][route]["improvement"]["status"] == "safe_update_emitted":
                emissions[route] += 1
    certificates = Counter(
        route
        for record in task_results
        for route in ROUTES
        if record["routes"][route]["status"] == "certificate_emitted"
    )
    print(f"wrote {len(task_results)} matched records to {output_dir}")
    for route in ROUTES:
        print(
            f"{route:18s} certificates={certificates[route]}/{len(task_results)} "
            f"safe_updates={emissions[route]}/{len(task_results)}"
        )
    print(f"identity regression: {len(identity_mismatches)} mismatches")
    print("PASS fixed-policy Expected SARSA evaluation")


if __name__ == "__main__":
    main()
