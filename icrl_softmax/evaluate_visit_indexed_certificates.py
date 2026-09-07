"""Evaluate selective visit-indexed certificates on the frozen policy matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy

from evaluate_fixed_policy_q_routes import (
    ROUTE_ORDER,
    evaluate_prefix,
    make_mdp,
    make_policy,
    summarize,
    write_json,
)
from markov_coverage_certificate import (
    build_markov_base_certificate,
    build_markov_coverage_certificate,
)
from mdps import rollout
from verify_fixed_policy_q_routes import policy_quantities
from visit_indexed_martingale_certificate import (
    STATUS_SELECTIVE_HIGH_PROBABILITY,
    build_visit_indexed_certificate,
)


ACTIVATION_COMMIT = "c8ec7e5c3165930663e26a07f98b38cc9ec186ad"
AUDIT_TOLERANCE = 1e-9
FROZEN_SEED_TASKS_PER_CELL = 30
THEOREM_ROUTES = (
    "direct_exact",
    "direct_softmax",
    "vfirst_nosplit_exact",
    "vfirst_nosplit_softmax",
)
BASELINE_HASHES = {
    "config.json": "a2276eae06ba8689864cac2ad3d3d0e92b069014b2046d0180ee172bb8296ea0",
    "task_results.json": "c84329bd6b9fcd495789f2067f250ead28fec807193d1b69ae1038e9cc3b2f25",
    "summary.json": "49846c0825c859df28ff773164352d19f6cb86943c484ab411bac7f69434b5ae",
}
PREFLIGHT_SCRIPTS = (
    "verify_finite_sample_theorems.py",
    "verify_fixed_policy_q_routes.py",
    "verify_crossfit_markov_certificate.py",
    "verify_end_to_end_sarsa.py",
    "verify_visit_indexed_martingale_certificate.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)).replace("\\", "/"): sha256_file(path)
        for path in sorted(item for item in directory.rglob("*") if item.is_file())
    }


def load_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def verify_frozen_baseline(baseline_dir: Path) -> dict[str, Any]:
    observed = {
        name: sha256_file(baseline_dir / name) for name in BASELINE_HASHES
    }
    if observed != BASELINE_HASHES:
        raise RuntimeError(
            f"frozen baseline hash mismatch: expected {BASELINE_HASHES}, got {observed}"
        )
    tasks = load_strict_json(baseline_dir / "task_results.json")
    if not isinstance(tasks, list) or len(tasks) != 480:
        raise RuntimeError("frozen baseline task_results.json must contain 480 records")
    return {
        "canonical_directory": str(baseline_dir),
        "required_hashes": observed,
        "record_count": len(tasks),
        "complete_inventory": directory_hashes(baseline_dir),
    }


def run_preflight_checks(project_dir: Path) -> str:
    sections: list[str] = []
    for script in PREFLIGHT_SCRIPTS:
        command = [sys.executable, "-B", script]
        completed = subprocess.run(
            command,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=False,
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


def declared_reward_bound(gap_bonus: float) -> float:
    bonus = float(gap_bonus)
    if not math_is_finite(bonus) or bonus < 0.0:
        raise ValueError("gap_bonus must be finite and nonnegative")
    return 1.0 + bonus


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


def _residual_family_audit(
    means: list[float | None],
    family: dict[str, Any],
) -> dict[str, Any]:
    radii = family["radius_by_group"]
    if len(means) != len(radii):
        raise AssertionError("oracle residual and radius dimensions differ")
    violations: list[dict[str, Any]] = []
    observed_abs: list[float] = []
    excesses: list[float] = []
    for group, (mean, radius) in enumerate(zip(means, radii, strict=True)):
        if mean is None:
            if radius is not None:
                raise AssertionError("missing oracle mean has a nonmissing radius")
            continue
        if radius is None:
            raise AssertionError("observed oracle mean has a missing radius")
        absolute = abs(float(mean))
        excess = absolute - float(radius)
        observed_abs.append(absolute)
        excesses.append(excess)
        if excess > AUDIT_TOLERANCE:
            violations.append(
                {
                    "group": group,
                    "absolute_residual_mean": absolute,
                    "radius": float(radius),
                    "excess": excess,
                }
            )
    return {
        "observed_group_count": len(observed_abs),
        "max_absolute_residual_mean": max(observed_abs),
        "max_excess_over_radius": max(excesses),
        "violation_count": len(violations),
        "violations": violations,
        "all_observed_groups_within_radius": not violations,
        "tolerance": AUDIT_TOLERANCE,
    }


def build_oracle_audit(
    *,
    visit_certificate: dict[str, Any],
    legacy_certificate: dict[str, Any],
    routes: dict[str, dict[str, Any]],
    actual_reward_abs_max: float,
    declared_reward_abs_bound: float,
) -> dict[str, Any]:
    ghost = legacy_certificate["finite_sample"]["observed_ghost_residuals"]
    event = visit_certificate["event"]
    residuals = {
        "state_bellman": _residual_family_audit(
            list(ghost["state_residual_means"]), event["state_bellman"]
        ),
        "pair_bellman": _residual_family_audit(
            list(ghost["pair_residual_means"]), event["pair_bellman"]
        ),
        "recovery": _residual_family_audit(
            list(ghost["recovery_residual_means"]), event["recovery"]
        ),
    }
    route_audits: dict[str, Any] = {}
    for route in THEOREM_ROUTES:
        certificate = visit_certificate["routes"][route]
        emitted = certificate["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY
        actual_error = float(routes[route]["q_sup_error"])
        bound = certificate["total_bound"]
        violation = (
            None
            if not emitted or bound is None
            else actual_error > float(bound) + AUDIT_TOLERANCE
        )
        route_audits[route] = {
            "emitted": emitted,
            "actual_q_sup_error": actual_error,
            "certified_total_bound": bound,
            "bound_slack": None if bound is None else float(bound) - actual_error,
            "violation": violation,
            "tolerance": AUDIT_TOLERANCE,
        }
    return {
        "purpose": "empirical truth-based audit only; not proof or certificate input",
        "reward_bound": {
            "actual_reward_abs_max": float(actual_reward_abs_max),
            "declared_reward_abs_bound": float(declared_reward_abs_bound),
            "satisfied": bool(
                actual_reward_abs_max <= declared_reward_abs_bound + 1e-12
            ),
        },
        "fixed_target_residuals": residuals,
        "routes": route_audits,
        "any_residual_event_violation": any(
            family["violation_count"] > 0 for family in residuals.values()
        ),
        "any_emitted_route_violation": any(
            audit["violation"] is True for audit in route_audits.values()
        ),
    }


def _summary_group_key(record: dict[str, Any], route: str) -> tuple[Any, ...]:
    return (
        int(record["trajectory_length"]),
        int(record["n_actions"]),
        float(record["pi_min"]),
        float(record["beta"]),
        float(record["mixing"]),
        float(record["gap_bonus"]),
        route,
    )


def augment_summary(
    legacy_summary: list[dict[str, Any]],
    task_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for task in task_results:
        for route in THEOREM_ROUTES:
            grouped[_summary_group_key(task, route)].append(task)

    augmented: list[dict[str, Any]] = []
    for legacy_row in legacy_summary:
        row = dict(legacy_row)
        route = str(row["route"])
        if route not in THEOREM_ROUTES:
            augmented.append(row)
            continue
        key = (
            int(row["trajectory_length"]),
            int(row["n_actions"]),
            float(row["pi_min"]),
            float(row["beta"]),
            float(row["mixing"]),
            float(row["gap_bonus"]),
            route,
        )
        selected = grouped[key]
        if not selected:
            raise AssertionError(f"missing visit-indexed summary group {key}")
        certificates = [
            task["visit_indexed_certificate"]["routes"][route]
            for task in selected
        ]
        audits = [
            task["visit_indexed_certificate"]["oracle_audit"]["routes"][route]
            for task in selected
        ]
        emitted = [
            item["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY
            for item in certificates
        ]
        finite_bounds = [
            float(item["total_bound"])
            for item in certificates
            if item["total_bound"] is not None
        ]
        reasons = Counter(
            reason for item in certificates for reason in item["failure_reasons"]
        )
        audited = [audit for audit in audits if audit["violation"] is not None]
        violations = [audit for audit in audited if audit["violation"] is True]
        row["visit_indexed_certificate"] = {
            "tasks": len(selected),
            "pair_full_support_rate": float(
                np.mean(
                    [
                        task["visit_indexed_certificate"]["event"]["pair_bellman"][
                            "full_support"
                        ]
                        for task in selected
                    ]
                )
            ),
            "state_full_support_rate": float(
                np.mean(
                    [
                        task["visit_indexed_certificate"]["event"]["state_bellman"][
                            "full_support"
                        ]
                        for task in selected
                    ]
                )
            ),
            "emission_rate": float(np.mean(emitted)),
            "finite_bound_rate": float(
                np.mean([item["finite_bound_emitted"] for item in certificates])
            ),
            "improves_over_zero_initialization_rate": float(
                np.mean(
                    [
                        item["improves_over_zero_initialization"]
                        for item in certificates
                    ]
                )
            ),
            "emitted_total_bound_mean": (
                None if not finite_bounds else float(np.mean(finite_bounds))
            ),
            "failure_reason_rates": {
                reason: count / len(selected) for reason, count in sorted(reasons.items())
            },
            "oracle_audited_emission_count": len(audited),
            "oracle_violation_count": len(violations),
            "oracle_violation_rate_among_emitted": (
                None if not audited else len(violations) / len(audited)
            ),
        }
        augmented.append(row)
    return augmented


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=30)
    parser.add_argument(
        "--trajectory-lengths", type=int, nargs="+", default=[256, 1024, 4096, 16384]
    )
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
        default=Path("results/FP-MART-001/codex"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    baseline_dir = (project_dir / "results/fixed_policy_finite_sample_certificates").resolve()
    baseline_before = verify_frozen_baseline(baseline_dir)
    checks_log = run_preflight_checks(project_dir)

    n_actions_grid = args.n_actions or ([2, 4] if args.quick else [2, 4, 8])
    pi_min_grid = args.pi_mins or ([0.05] if args.quick else [0.02, 0.08])
    beta_grid = args.betas or ([6.0, 10.0] if args.quick else [4.0, 8.0, 12.0])
    mixing_grid = args.mixing or ([0.08, 0.50] if args.quick else [0.03, 0.15, 0.60])
    gap_grid = args.gap_bonuses or ([0.0, 0.50] if args.quick else [0.0, 0.25, 0.75])
    iterations = args.iterations or (120 if args.quick else 240)
    if args.tasks <= 0 or min(args.trajectory_lengths) < 4:
        raise ValueError("tasks must be positive and trajectory lengths at least 4")
    if args.tasks > FROZEN_SEED_TASKS_PER_CELL:
        raise ValueError("tasks cannot exceed the frozen 30-seed schedule per cell")
    if not 0.0 < args.gamma < 1.0 or not 0.0 < args.alpha <= 1.0:
        raise ValueError("gamma and alpha must lie in (0,1), with alpha allowing 1")
    if not 0.0 < args.certificate_delta < 1.0:
        raise ValueError("certificate_delta must lie in (0,1)")
    for n_actions in n_actions_grid:
        for pi_min in pi_min_grid:
            if not 0.0 < pi_min < 1.0 / n_actions:
                raise ValueError(f"pi_min={pi_min} invalid for n_actions={n_actions}")
    if any(float(gap) < 0.0 for gap in gap_grid):
        raise ValueError("gap bonuses must be nonnegative for the declared reward rule")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = subprocess.list2cmdline(
        [sys.executable, "-B", Path(__file__).name, *sys.argv[1:]]
    )
    (output_dir / "commands.log").write_text(
        f"WORKDIR: {project_dir}\nEVALUATION: {command}\n",
        encoding="utf-8",
    )
    (output_dir / "checks.log").write_text(checks_log, encoding="utf-8")

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
        "visit_indexed_certificate": {
            "method": "finite_horizon_visit_indexed_hoeffding",
            "group_count": args.n_states + 2 * args.n_states * max(n_actions_grid),
            "declared_reward_bound_rule": "1 + gap_bonus",
            "nontriviality_rule": "emitted total_bound < B",
            "oracle_audit_tolerance": AUDIT_TOLERANCE,
            "probability_statement": "P(Emit and certified error bound is violated) <= delta",
            "mandatory_delta_allocation": "full delta over G*n two-sided events",
            "optional_variance_adaptive": "not_implemented_no_observable_variance_proxy",
            "common_activation_commit": ACTIVATION_COMMIT,
            "seed_schedule_tasks_per_cell": FROZEN_SEED_TASKS_PER_CELL,
        },
    }
    write_json(output_dir / "config.json", config)

    seed_sequence = np.random.SeedSequence(args.seed)
    total_seed_cells = (
        len(n_actions_grid)
        * len(pi_min_grid)
        * len(mixing_grid)
        * len(gap_grid)
        * FROZEN_SEED_TASKS_PER_CELL
    )
    child_seeds = iter(seed_sequence.spawn(total_seed_cells))
    task_results: list[dict[str, Any]] = []
    maximum_length = max(args.trajectory_lengths)

    for n_actions in n_actions_grid:
        for pi_min in pi_min_grid:
            for mixing in mixing_grid:
                for gap_bonus in gap_grid:
                    reward_bound = declared_reward_bound(gap_bonus)
                    cell_seeds = [
                        next(child_seeds)
                        for _ in range(FROZEN_SEED_TASKS_PER_CELL)
                    ]
                    for task_index in range(args.tasks):
                        child_seed = cell_seeds[task_index]
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
                                current_states = states[:length]
                                current_pairs = (
                                    current_states * n_actions + actions[:length]
                                )
                                state_counts = np.bincount(
                                    current_states, minlength=args.n_states
                                ).astype(np.int64)
                                pair_counts = np.bincount(
                                    current_pairs,
                                    minlength=args.n_states * n_actions,
                                ).astype(np.int64)
                                certificate = build_markov_coverage_certificate(
                                    exact["p_pi"],
                                    exact["p_pair"],
                                    pair_counts,
                                    length,
                                    beta,
                                    args.gamma,
                                    delta=args.certificate_delta,
                                    base_certificate=base_certificate,
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
                                visit = build_visit_indexed_certificate(
                                    state_counts=state_counts,
                                    pair_counts=pair_counts,
                                    trajectory_length=length,
                                    reward_bound=reward_bound,
                                    gamma=args.gamma,
                                    alpha=args.alpha,
                                    beta=beta,
                                    delta=args.certificate_delta,
                                    direct_exact_iterations=int(
                                        routes["direct_exact"]["iterations_used"]
                                    ),
                                    direct_softmax_iterations=int(
                                        routes["direct_softmax"]["iterations_used"]
                                    ),
                                    state_exact_iterations=int(
                                        routes["vfirst_nosplit_exact"][
                                            "value_iterations_used"
                                        ]
                                    ),
                                    state_softmax_iterations=int(
                                        routes["vfirst_nosplit_softmax"][
                                            "value_iterations_used"
                                        ]
                                    ),
                                    direct_exact_diverged=bool(
                                        routes["direct_exact"]["diverged"]
                                    ),
                                    direct_softmax_diverged=bool(
                                        routes["direct_softmax"]["diverged"]
                                    ),
                                    iteration_cap=iterations,
                                )
                                visit["oracle_audit"] = build_oracle_audit(
                                    visit_certificate=visit,
                                    legacy_certificate=certificate,
                                    routes=routes,
                                    actual_reward_abs_max=float(
                                        np.max(np.abs(np.asarray(mdp["R"])))
                                    ),
                                    declared_reward_abs_bound=reward_bound,
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
                                        "true_pair_occupancy_min": float(
                                            np.min(exact["mu_pair"])
                                        ),
                                        "true_action_gap_min": float(
                                            np.min(action_gaps)
                                        ),
                                        "true_action_gap_mean": float(
                                            np.mean(action_gaps)
                                        ),
                                        "certificate": certificate,
                                        "routes": routes,
                                        "visit_indexed_certificate": visit,
                                    }
                                )

    legacy_summary = summarize(task_results)
    summary = augment_summary(legacy_summary, task_results)
    write_json(output_dir / "task_results.json", task_results)
    write_json(output_dir / "summary.json", summary)

    baseline_after = verify_frozen_baseline(baseline_dir)
    if baseline_before["complete_inventory"] != baseline_after["complete_inventory"]:
        raise RuntimeError("read-only baseline directory changed during evaluation")
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
        "matplotlib_version": matplotlib.__version__,
        "working_directory": str(project_dir),
        "script": str(Path(__file__).resolve()),
        "git_head_at_run": git_head,
        "common_activation_commit": ACTIVATION_COMMIT,
        "frozen_baseline": baseline_after,
    }
    write_json(output_dir / "environment.json", environment)

    emissions = Counter()
    for task in task_results:
        for route in THEOREM_ROUTES:
            item = task["visit_indexed_certificate"]["routes"][route]
            if item["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY:
                emissions[route] += 1
    print(f"wrote {len(task_results)} matched comparisons to {output_dir}")
    for route in THEOREM_ROUTES:
        print(f"{route:26s} emitted={emissions[route]}/{len(task_results)}")
    if tuple(routes) != ROUTE_ORDER:
        raise AssertionError("legacy route order changed")
    print("PASS visit-indexed certificate evaluation")


if __name__ == "__main__":
    main()
