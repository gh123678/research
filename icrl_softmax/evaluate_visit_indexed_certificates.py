"""Evaluate visit-indexed martingale certificates on the frozen FP-MART-001 matrix.

Reuses the frozen MDP, policy, rollout, estimator, and route functions from
``evaluate_fixed_policy_q_routes.py`` without modifying them, reproduces every
legacy field, and adds one ``visit_indexed_certificate`` namespace per record.
Certificate construction uses only observed counts, observed empirical kernel
diagonals, the declared reward bound, and public hyperparameters; true values
appear only inside ``visit_indexed_certificate.oracle_audit``.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy

import visit_indexed_martingale_certificate as vimc
from evaluate_fixed_policy_q_routes import (
    evaluate_prefix,
    make_mdp,
    make_policy,
    write_json,
)
from markov_coverage_certificate import (
    build_markov_base_certificate,
    build_markov_coverage_certificate,
)
from mdps import rollout
from verify_fixed_policy_q_routes import policy_quantities


THEOREM_ROUTES = (
    "direct_exact",
    "direct_softmax",
    "vfirst_nosplit_exact",
    "vfirst_nosplit_softmax",
)

MAIN_COMPARISON = [
    "direct_exact",
    "direct_softmax",
    "vfirst_split_exact",
    "vfirst_split_softmax",
    "vfirst_crossfit_exact",
    "vfirst_crossfit_softmax",
    "vfirst_crossfit_gap_exact",
    "vfirst_crossfit_gap_softmax",
]
DIAGNOSTIC_ABLATIONS = [
    "vfirst_nosplit_exact",
    "vfirst_nosplit_softmax",
    "vfirst_oracle_exact",
]

FROZEN_TASKS_PER_CELL = 30


def git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _audit_route(
    certificate: dict[str, Any], true_error: float | None
) -> dict[str, Any]:
    bound = certificate.get("total_bound")
    emitted = certificate.get("status") == vimc.STATUS_SELECTIVE
    slack = None if bound is None or true_error is None else bound - true_error
    violated = (
        None if slack is None else bool(slack < -1e-9)
    )
    return {
        "emitted": bool(emitted),
        "true_sup_error": true_error,
        "total_bound": bound,
        "bound_slack": slack,
        "bound_violated": violated,
    }


def _per_group_residual_audit(
    event: dict[str, Any], ghost: dict[str, Any]
) -> dict[str, Any]:
    """Audit each visited residual group against its own visit-indexed radius."""
    delta = float(event["delta"])
    bound = float(event["value_bound"])
    length = int(event["trajectory_length"])
    groups = int(event["n_groups"])
    state_counts = [int(count) for count in event["state_counts"]]
    pair_counts = [int(count) for count in event["pair_counts"]]

    def _radius(count: int) -> float:
        return vimc.hoeffding_radius(
            count,
            n_groups=groups,
            trajectory_length=length,
            delta=delta,
            value_bound=bound,
        )

    violation_count = 0
    visited_groups = 0
    worst_margin: float | None = None
    worst_group: str | None = None
    per_family: dict[str, int] = {}
    families = (
        ("state", state_counts, ghost.get("state_residual_means")),
        ("pair", pair_counts, ghost.get("pair_residual_means")),
        ("recovery", pair_counts, ghost.get("recovery_residual_means")),
    )
    for family, counts, means in families:
        if means is None:
            continue
        mean_values = list(means)
        family_violations = 0
        for index, count in enumerate(counts):
            if count <= 0:
                continue
            visited_groups += 1
            mean = mean_values[index] if index < len(mean_values) else None
            radius = _radius(count)
            if mean is None:
                family_violations += 1
                continue
            margin = radius - abs(float(mean))
            if margin < 0.0:
                family_violations += 1
            if worst_margin is None or margin < worst_margin:
                worst_margin = margin
                worst_group = f"{family}[{index}]"
        per_family[family] = family_violations
        violation_count += family_violations
    return {
        "per_family_violations": per_family,
        "violation_count": violation_count,
        "visited_groups": visited_groups,
        "worst_margin": worst_margin,
        "worst_group": worst_group,
    }


def build_visit_indexed_namespace(
    *,
    trajectory_length: int,
    n_states: int,
    n_pairs: int,
    state_counts: np.ndarray,
    pair_counts: np.ndarray,
    delta: float,
    declared_reward_bound: float,
    beta: float,
    gamma: float,
    alpha: float,
    routes: dict[str, dict[str, Any]],
    ghost: dict[str, Any],
    true_reward_abs_max: float,
) -> dict[str, Any]:
    """Attach the new certificate namespace; inputs are observed-only."""
    event = vimc.shared_visit_event(
        trajectory_length=trajectory_length,
        n_states=n_states,
        n_pairs=n_pairs,
        state_counts=[int(count) for count in state_counts],
        pair_counts=[int(count) for count in pair_counts],
        delta=delta,
        declared_reward_bound=declared_reward_bound,
        gamma=gamma,
    )
    direct_exact = vimc.direct_q_certificate(
        event,
        matching="exact",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(routes["direct_exact"]["iterations_used"]),
        divergence_guard_triggered=bool(routes["direct_exact"]["diverged"]),
    )
    direct_softmax = vimc.direct_q_certificate(
        event,
        matching="softmax",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(routes["direct_softmax"]["iterations_used"]),
        beta=beta,
        kernel_diagonal_min=float(routes["direct_softmax"]["kernel_diagonal_min"]),
        divergence_guard_triggered=bool(routes["direct_softmax"]["diverged"]),
    )
    state_exact = vimc.state_value_certificate(
        event,
        matching="exact",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(routes["vfirst_nosplit_exact"]["value_iterations_used"]),
    )
    state_softmax = vimc.state_value_certificate(
        event,
        matching="softmax",
        gamma=gamma,
        alpha=alpha,
        iterations_used=int(
            routes["vfirst_nosplit_softmax"]["value_iterations_used"]
        ),
        beta=beta,
        kernel_diagonal_min=float(
            routes["vfirst_nosplit_softmax"]["value_kernel_diagonal_min"]
        ),
    )
    vfirst_exact = vimc.vfirst_nosplit_certificate(
        event,
        state_exact,
        recovery_matching="exact",
        gamma=gamma,
    )
    vfirst_softmax = vimc.vfirst_nosplit_certificate(
        event,
        state_softmax,
        recovery_matching="softmax",
        gamma=gamma,
        beta=beta,
        recovery_diagonal_min=float(
            routes["vfirst_nosplit_softmax"]["kernel_diagonal_min"]
        ),
    )
    adaptive = vimc.variance_adaptive_certificate(delta / 2.0)

    radius_pairs = event["pair_radius"]
    radius_states = event["state_radius"]

    def radius_margin(realized: float | None, radius: float | None) -> float | None:
        if realized is None or radius is None:
            return None
        return float(radius) - float(realized)

    oracle_audit = {
        "note": (
            "truth-based audit only; never an input to certificate construction"
        ),
        "reward_declaration": {
            "declared_reward_bound": float(declared_reward_bound),
            "true_reward_abs_max": float(true_reward_abs_max),
            "declaration_verified": bool(
                float(true_reward_abs_max) <= float(declared_reward_bound)
            ),
        },
        "routes": {
            "direct_exact": _audit_route(
                direct_exact, float(routes["direct_exact"]["q_sup_error"])
            ),
            "direct_softmax": _audit_route(
                direct_softmax, float(routes["direct_softmax"]["q_sup_error"])
            ),
            "vfirst_nosplit_exact": _audit_route(
                vfirst_exact, float(routes["vfirst_nosplit_exact"]["q_sup_error"])
            ),
            "vfirst_nosplit_softmax": _audit_route(
                vfirst_softmax,
                float(routes["vfirst_nosplit_softmax"]["q_sup_error"]),
            ),
        },
        "state_stage": {
            "exact": _audit_route(
                state_exact, float(routes["vfirst_nosplit_exact"]["v_sup_error"])
            ),
            "softmax": _audit_route(
                state_softmax,
                float(routes["vfirst_nosplit_softmax"]["v_sup_error"]),
            ),
        },
        "realized_residual_sup": {
            "state": ghost.get("state_residual_sup"),
            "pair": ghost.get("pair_residual_sup"),
            "recovery": ghost.get("recovery_residual_sup"),
        },
        "per_group_residual_audit": _per_group_residual_audit(event, ghost),
        "radius_margin": {
            "state": radius_margin(ghost.get("state_residual_sup"), radius_states),
            "pair": radius_margin(ghost.get("pair_residual_sup"), radius_pairs),
            "recovery": radius_margin(
                ghost.get("recovery_residual_sup"), radius_pairs
            ),
        },
    }

    certificate = {
        "risk": {
            "delta": float(delta),
            "n_groups": int(event["n_groups"]),
            "trajectory_length": int(trajectory_length),
            "log_factor": event["log_factor"],
            "radius_form": "B*sqrt(2*log(2*G*n/delta)/k)",
            "probability_semantics": "P(Emit and bound violated) <= delta",
            "declared_reward_rule": "declared before sampling; B = declared/(1-gamma)",
            "value_bound_derivation": event["value_bound_derivation"],
        },
        "event": event,
        "routes": {
            "direct_exact": direct_exact,
            "direct_softmax": direct_softmax,
            "state_exact": state_exact,
            "state_softmax": state_softmax,
            "vfirst_nosplit_exact": vfirst_exact,
            "vfirst_nosplit_softmax": vfirst_softmax,
        },
        "variance_adaptive": adaptive,
    }
    return {**certificate, "oracle_audit": oracle_audit}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument("--trajectory-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--n-states", type=int, required=True)
    parser.add_argument("--n-actions", type=int, nargs="+", required=True)
    parser.add_argument("--pi-mins", type=float, nargs="+", required=True)
    parser.add_argument("--betas", type=float, nargs="+", required=True)
    parser.add_argument("--mixing", type=float, nargs="+", required=True)
    parser.add_argument("--gap-bonuses", type=float, nargs="+", required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--certificate-delta", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def canonical_command(args: argparse.Namespace) -> str:
    lengths = " ".join(str(item) for item in args.trajectory_lengths)
    n_actions = " ".join(str(item) for item in args.n_actions)
    pi_mins = " ".join(str(item) for item in args.pi_mins)
    betas = " ".join(str(item) for item in args.betas)
    mixing = " ".join(str(item) for item in args.mixing)
    gaps = " ".join(str(item) for item in args.gap_bonuses)
    return (
        "python -B evaluate_visit_indexed_certificates.py "
        f"--tasks {args.tasks} "
        f"--trajectory-lengths {lengths} "
        f"--n-states {args.n_states} "
        f"--n-actions {n_actions} "
        f"--pi-mins {pi_mins} "
        f"--betas {betas} "
        f"--mixing {mixing} "
        f"--gap-bonuses {gaps} "
        f"--gamma {args.gamma} "
        f"--alpha {args.alpha} "
        f"--iterations {args.iterations} "
        f"--certificate-delta {args.certificate_delta} "
        f"--seed {args.seed} "
        f"--output-dir {args.output_dir}"
    )


def main() -> None:
    args = parse_args()
    n_actions_grid = args.n_actions
    pi_min_grid = args.pi_mins
    beta_grid = args.betas
    mixing_grid = args.mixing
    gap_grid = args.gap_bonuses
    if args.tasks <= 0 or args.tasks > FROZEN_TASKS_PER_CELL:
        raise ValueError(
            f"tasks must lie in [1,{FROZEN_TASKS_PER_CELL}] so the frozen "
            "per-cell seed stride is preserved"
        )
    if min(args.trajectory_lengths) < 4:
        raise ValueError("trajectory lengths must be at least 4")
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
        "iterations": args.iterations,
        "seed": args.seed,
        "quick": False,
        "main_comparison": MAIN_COMPARISON,
        "diagnostic_ablations": DIAGNOSTIC_ABLATIONS,
        "certificate_delta": args.certificate_delta,
    }
    write_json(output_dir / "config.json", config)
    environment = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "git_head": git_revision(),
        "cwd": str(Path.cwd()),
    }
    write_json(output_dir / "environment.json", environment)

    seed_sequence = np.random.SeedSequence(args.seed)
    total_cells = (
        len(n_actions_grid)
        * len(pi_min_grid)
        * len(mixing_grid)
        * len(gap_grid)
        * FROZEN_TASKS_PER_CELL
    )
    child_seeds = iter(seed_sequence.spawn(total_cells))
    task_results: list[dict[str, Any]] = []
    maximum_length = max(args.trajectory_lengths)

    for n_actions in n_actions_grid:
        for pi_min in pi_min_grid:
            for mixing in mixing_grid:
                for gap_bonus in gap_grid:
                    cell_seeds = [
                        next(child_seeds) for _ in range(FROZEN_TASKS_PER_CELL)
                    ]
                    declared_reward_bound = 1.0 + float(gap_bonus)
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
                                    args.iterations,
                                    certificate,
                                    certificate_delta=args.certificate_delta,
                                )
                                true_reward_abs_max = float(
                                    np.max(np.abs(np.asarray(mdp["R"])))
                                )
                                state_counts = np.bincount(
                                    states[:length], minlength=args.n_states
                                )
                                ghost = certificate["finite_sample"][
                                    "observed_ghost_residuals"
                                ]
                                visit_indexed = build_visit_indexed_namespace(
                                    trajectory_length=length,
                                    n_states=args.n_states,
                                    n_pairs=args.n_states * n_actions,
                                    state_counts=state_counts,
                                    pair_counts=pair_counts,
                                    delta=args.certificate_delta,
                                    declared_reward_bound=declared_reward_bound,
                                    beta=beta,
                                    gamma=args.gamma,
                                    alpha=args.alpha,
                                    routes=routes,
                                    ghost=ghost,
                                    true_reward_abs_max=true_reward_abs_max,
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
                                        "visit_indexed_certificate": visit_indexed,
                                    }
                                )

    write_json(output_dir / "task_results.json", task_results)
    (output_dir / "commands.log").write_text(
        canonical_command(args) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(task_results)} matched comparisons to {output_dir}")
    print("PASS visit-indexed certificate evaluation")


if __name__ == "__main__":
    main()
