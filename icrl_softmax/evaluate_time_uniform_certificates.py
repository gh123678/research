"""Evaluate time-uniform mixture certificates on the frozen policy matrix.

Additive route over the frozen FP-MART-001 protocol: every legacy leaf is
reproduced exactly and all new data live under ``time_uniform_certificate``.
"""

from __future__ import annotations

import argparse
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
from evaluate_visit_indexed_certificates import (
    AUDIT_TOLERANCE,
    FROZEN_SEED_TASKS_PER_CELL,
    build_oracle_audit,
    declared_reward_bound,
    directory_hashes,
    load_strict_json,
    sha256_file,
)
from markov_coverage_certificate import (
    build_markov_base_certificate,
    build_markov_coverage_certificate,
)
from mdps import rollout
from time_uniform_mixture_certificate import (
    STATUS_SELECTIVE_HIGH_PROBABILITY,
    build_time_uniform_certificate,
)
from verify_fixed_policy_q_routes import policy_quantities
from visit_indexed_martingale_certificate import build_visit_indexed_certificate


ACTIVATION_COMMIT = "0ce18b4676f70ca0556496e804aa65563efa63da"
LEGACY_ACTIVATION_COMMIT = "c8ec7e5c3165930663e26a07f98b38cc9ec186ad"
THEOREM_ROUTES = (
    "direct_exact",
    "direct_softmax",
    "vfirst_nosplit_exact",
    "vfirst_nosplit_softmax",
)
BASELINE_DIR = Path(
    r"C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex"
)
BASELINE_HASHES = {
    "config.json": "bcc377b422711302163b976d5d5ece389e1e3ee8540d37817a8719bf9ec3bf8a",
    "task_results.json": "929e2f65689af850b65f000ee6675a8a87b3506c2e2a9600c07d28cf183c6d52",
    "summary.json": "fa13619755b627b9dcff281dc5e4bc9b4d0262b9a6592f2982012b53df2b34ed",
}
PREFLIGHT_SCRIPTS = (
    "verify_finite_sample_theorems.py",
    "verify_fixed_policy_q_routes.py",
    "verify_crossfit_markov_certificate.py",
    "verify_end_to_end_sarsa.py",
    "verify_visit_indexed_martingale_certificate.py",
    "verify_time_uniform_mixture_certificate.py",
)


def verify_frozen_baseline(baseline_dir: Path) -> dict[str, Any]:
    observed = {name: sha256_file(baseline_dir / name) for name in BASELINE_HASHES}
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


def _residual_family_audit(
    means: list[float | None],
    radii: list[float | None],
) -> dict[str, Any]:
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


def build_time_uniform_oracle_audit(
    *,
    uniform_certificate: dict[str, Any],
    legacy_certificate: dict[str, Any],
    routes: dict[str, dict[str, Any]],
    actual_reward_abs_max: float,
    declared_reward_abs_bound: float,
) -> dict[str, Any]:
    ghost = legacy_certificate["finite_sample"]["observed_ghost_residuals"]
    event = uniform_certificate["event"]
    residuals = {
        family: _residual_family_audit(
            list(ghost[f"{key}_residual_means"]),
            list(event[family]["mixture_radius_by_group"]),
        )
        for family, key in (
            ("state_bellman", "state"),
            ("pair_bellman", "pair"),
            ("recovery", "recovery"),
        )
    }
    stitch_residuals = {
        family: _residual_family_audit(
            list(ghost[f"{key}_residual_means"]),
            list(event[family]["stitch_radius_by_group"]),
        )
        for family, key in (
            ("state_bellman", "state"),
            ("pair_bellman", "pair"),
            ("recovery", "recovery"),
        )
    }
    route_audits: dict[str, Any] = {}
    for route in THEOREM_ROUTES:
        certificate = uniform_certificate["routes"][route]
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
        "stitch_fixed_target_residuals": stitch_residuals,
        "routes": route_audits,
        "any_residual_event_violation": any(
            family["violation_count"] > 0 for family in residuals.values()
        ),
        "any_stitch_residual_event_violation": any(
            family["violation_count"] > 0 for family in stitch_residuals.values()
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
            raise AssertionError(f"missing time-uniform summary group {key}")
        certificates = [
            task["time_uniform_certificate"]["routes"][route] for task in selected
        ]
        legacy_certificates = [
            task["visit_indexed_certificate"]["routes"][route] for task in selected
        ]
        audits = [
            task["time_uniform_certificate"]["oracle_audit"]["routes"][route]
            for task in selected
        ]
        emitted = [
            item["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY
            for item in certificates
        ]
        legacy_emitted = [
            item["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY
            for item in legacy_certificates
        ]
        finite_bounds = [
            float(item["total_bound"])
            for item in certificates
            if item["total_bound"] is not None
        ]
        bound_reductions = [
            float(item["total_bound_reduction_vs_old"])
            for item in certificates
            if item["total_bound_reduction_vs_old"] is not None
        ]
        radius_reductions = [
            float(item["radius_reduction_vs_old"])
            for item in certificates
            if item["radius_reduction_vs_old"] is not None
        ]
        nonincreasing = [
            float(new["total_bound"]) <= float(old["total_bound"]) + 1e-12
            for new, old in zip(certificates, legacy_certificates, strict=True)
            if new["total_bound"] is not None and old["total_bound"] is not None
        ]
        reasons = Counter(
            reason for item in certificates for reason in item["failure_reasons"]
        )
        audited = [audit for audit in audits if audit["violation"] is not None]
        violations = [audit for audit in audited if audit["violation"] is True]
        row["time_uniform_certificate"] = {
            "tasks": len(selected),
            "pair_full_support_rate": float(
                np.mean(
                    [
                        task["time_uniform_certificate"]["event"]["pair_bellman"][
                            "full_support"
                        ]
                        for task in selected
                    ]
                )
            ),
            "state_full_support_rate": float(
                np.mean(
                    [
                        task["time_uniform_certificate"]["event"]["state_bellman"][
                            "full_support"
                        ]
                        for task in selected
                    ]
                )
            ),
            "emission_rate": float(np.mean(emitted)),
            "emission_decisions_match_legacy_rate": float(
                np.mean([new == old for new, old in zip(emitted, legacy_emitted, strict=True)])
            ),
            "emitted_bound_nonincreasing_rate": (
                float(np.mean(nonincreasing)) if nonincreasing else None
            ),
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
            "below_two_value_bound_rate": float(
                np.mean([item["below_two_value_bound"] for item in certificates])
            ),
            "emitted_total_bound_mean": (
                None if not finite_bounds else float(np.mean(finite_bounds))
            ),
            "emitted_total_bound_reduction_vs_old_mean": (
                None if not bound_reductions else float(np.mean(bound_reductions))
            ),
            "residual_radius_reduction_vs_old_mean": (
                None if not radius_reductions else float(np.mean(radius_reductions))
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
    parser.add_argument("--n-actions", type=int, nargs="+", default=[4])
    parser.add_argument("--pi-mins", type=float, nargs="+", default=[0.05])
    parser.add_argument("--betas", type=float, nargs="+", default=[8.0])
    parser.add_argument("--mixing", type=float, nargs="+", default=[0.08, 0.50])
    parser.add_argument("--gap-bonuses", type=float, nargs="+", default=[0.0, 0.50])
    parser.add_argument("--gamma", type=float, default=0.70)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--certificate-delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/FP-TU-001/claude"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    baseline_dir = args.baseline_dir.resolve()
    baseline_before = verify_frozen_baseline(baseline_dir)
    checks_log = run_preflight_checks(project_dir)

    n_actions_grid = list(args.n_actions)
    pi_min_grid = list(args.pi_mins)
    beta_grid = list(args.betas)
    mixing_grid = list(args.mixing)
    gap_grid = list(args.gap_bonuses)
    iterations = int(args.iterations)
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
        "quick": False,
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
            "common_activation_commit": LEGACY_ACTIVATION_COMMIT,
            "seed_schedule_tasks_per_cell": FROZEN_SEED_TASKS_PER_CELL,
        },
        "time_uniform_certificate": {
            "method": "time_uniform_geometric_cosh_mixture",
            "group_count_rule": "m + 2d",
            "mixture_grid_size": 15,
            "mixture_count_grid": "k_j = 2^j for j = 0..14",
            "mixture_weight_formula": "w_j = (j+1)^(-2) / sum_{l=0}^{14} (l+1)^(-2)",
            "mixture_rate_formula": "a_j = sqrt(2 L_j / k_j), L_j = log(2 G / (delta w_j))",
            "declared_reward_bound_rule": "1 + gap_bonus",
            "nontriviality_rule": "emitted total_bound < B",
            "oracle_audit_tolerance": AUDIT_TOLERANCE,
            "probability_statement": "P(Emit and certified error bound is violated) <= delta",
            "mandatory_delta_allocation": "full delta over G groups, delta/G each",
            "line_stitching_role": "audit-only upper bracket; never selected",
            "inversion_tolerance": 1e-12,
            "inversion_max_iterations": 200,
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
                                uniform = build_time_uniform_certificate(
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
                                for route in THEOREM_ROUTES:
                                    new_route = uniform["routes"][route]
                                    old_route = visit["routes"][route]
                                    new_emitted = (
                                        new_route["status"]
                                        == STATUS_SELECTIVE_HIGH_PROBABILITY
                                    )
                                    old_emitted = (
                                        old_route["status"]
                                        == STATUS_SELECTIVE_HIGH_PROBABILITY
                                    )
                                    if new_emitted != old_emitted:
                                        raise AssertionError(
                                            f"emission mismatch on route {route}"
                                        )
                                    if new_emitted and not (
                                        float(new_route["total_bound"])
                                        <= float(old_route["total_bound"]) + 1e-12
                                    ):
                                        raise AssertionError(
                                            f"mixture total bound exceeds legacy on {route}"
                                        )
                                uniform["oracle_audit"] = (
                                    build_time_uniform_oracle_audit(
                                        uniform_certificate=uniform,
                                        legacy_certificate=certificate,
                                        routes=routes,
                                        actual_reward_abs_max=float(
                                            np.max(np.abs(np.asarray(mdp["R"])))
                                        ),
                                        declared_reward_abs_bound=reward_bound,
                                    )
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
                                        "time_uniform_certificate": uniform,
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
            item = task["time_uniform_certificate"]["routes"][route]
            if item["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY:
                emissions[route] += 1
    print(f"wrote {len(task_results)} matched comparisons to {output_dir}")
    for route in THEOREM_ROUTES:
        print(f"{route:26s} emitted={emissions[route]}/{len(task_results)}")
    if tuple(routes) != ROUTE_ORDER:
        raise AssertionError("legacy route order changed")
    print("PASS time-uniform certificate evaluation")


if __name__ == "__main__":
    main()
