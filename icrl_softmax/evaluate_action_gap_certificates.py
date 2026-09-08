"""Paired evaluator for the frozen FP-ADV-001 action-gap certificates.

Reconstructs the frozen FP-TU-001 480-record protocol exactly (same seeds,
MDPs, policies, trajectories, estimators, and certificates), then attaches the
new ``action_gap_certificate`` namespace computed by the pure module.  All
truth-based diagnostics live only inside ``action_gap_certificate.oracle_audit``
and never enter certificate inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import action_gap_certificate as agc
from evaluate_fixed_policy_q_routes import (
    evaluate_prefix,
    iterative_pair_evaluation,
    iterative_state_evaluation,
    make_mdp,
    make_policy,
    recover_q,
    summarize,
    write_json,
)
from evaluate_time_uniform_certificates import augment_summary as tu_augment_summary
from evaluate_visit_indexed_certificates import (
    augment_summary as vi_augment_summary,
    build_oracle_audit,
    declared_reward_bound,
)
from markov_coverage_certificate import (
    build_markov_base_certificate,
    build_markov_coverage_certificate,
)
from mdps import rollout
from time_uniform_mixture_certificate import build_time_uniform_certificate
from verify_fixed_policy_q_routes import policy_quantities
from visit_indexed_martingale_certificate import build_visit_indexed_certificate


TASK_ID = "FP-ADV-001"
ACTIVATION_COMMIT = "10a9a94e24ec92a59e7c756f9af6ce07b2f30e59"
COMMON_ROUTE_COMMIT = "4078f6911cbfb4654205772685f49896e4e8cad2"
SCIENTIFIC_BASELINE = "c579047950dfabb2600020cd2e53dd24b3e39c84"
MART_ACTIVATION_COMMIT = "c8ec7e5c3165930663e26a07f98b38cc9ec186ad"
TU_ACTIVATION_COMMIT = "0ce18b4676f70ca0556496e804aa65563efa63da"
FROZEN_SEED_TASKS_PER_CELL = 30
AUDIT_TOLERANCE = 1e-9

FPTU_BASELINE_DIR = Path(
    r"C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-TU-001\codex"
)
FPTU_BASELINE_HASHES = {
    "config.json": "43dcb96b0f6f95e76f1c0b484d6375e3727dbb5609b16a8952a69e2ac0dddf3a",
    "task_results.json": "0e5eab39bf49894832c5ebcdd6f7b70c453fff6b9600b234617889f8f9fa79be",
    "summary.json": "565fc4d261a938d13350984bb97242e79fba42f014d11f807a18942517f4444f",
}
FPMART_BASELINE_DIR = Path(
    r"C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex"
)
FPMART_BASELINE_HASHES = {
    "config.json": "bcc377b422711302163b976d5d5ece389e1e3ee8540d37817a8719bf9ec3bf8a",
    "task_results.json": "929e2f65689af850b65f000ee6675a8a87b3506c2e2a9600c07d28cf183c6d52",
    "summary.json": "fa13619755b627b9dcff281dc5e4bc9b4d0262b9a6592f2982012b53df2b34ed",
}

PROBABILITY_STATEMENT = (
    "P(EmitUpdate and (any used action ordering is false "
    "or exists s: V^{pi_plus}(s) < V^pi(s))) <= delta"
)

LOCAL_ROUTES = ("vfirst_local_exact", "vfirst_local_softmax")
GLOBAL_ROUTES = (
    "vfirst_global_exact",
    "vfirst_global_softmax",
    "direct_global_exact",
    "direct_global_softmax",
)
ROUTE_NAMES = (*LOCAL_ROUTES, *GLOBAL_ROUTES)
SUMMARY_ROUTE_MAP = {
    "vfirst_nosplit_exact": ("vfirst_local_exact", "vfirst_global_exact"),
    "vfirst_nosplit_softmax": ("vfirst_local_softmax", "vfirst_global_softmax"),
    "direct_exact": ("direct_global_exact",),
    "direct_softmax": ("direct_global_softmax",),
}

PREFLIGHT_SCRIPTS = (
    "verify_finite_sample_theorems.py",
    "verify_fixed_policy_q_routes.py",
    "verify_crossfit_markov_certificate.py",
    "verify_end_to_end_sarsa.py",
    "verify_visit_indexed_martingale_certificate.py",
    "verify_time_uniform_mixture_certificate.py",
    "verify_action_gap_certificate.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_baseline(directory: Path, hashes: dict[str, str], label: str) -> dict[str, Any]:
    observed = {name: sha256_file(directory / name) for name in hashes}
    if observed != hashes:
        raise RuntimeError(f"frozen {label} baseline mismatch: {observed}")
    records = json.loads((directory / "task_results.json").read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 480:
        raise RuntimeError(f"frozen {label} baseline must contain 480 records")
    return {"directory": str(directory), "hashes": observed, "record_count": len(records)}


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
            raise RuntimeError(f"preflight verifier failed: {script}\n" + "\n".join(sections))
    return "\n".join(sections).rstrip() + "\n"


def reconstruct_route_estimates(
    mdp: dict[str, Any],
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    trajectory_length: int,
    beta: float,
    alpha: float,
    iterations: int,
) -> dict[str, np.ndarray]:
    """Rebuild the exact four theorem-route estimates from the trajectory."""
    n_states = int(mdp["nS"])
    n_actions = int(mdp["nA"])
    n_pairs = n_states * n_actions
    gamma = float(mdp["gamma"])
    transition_rewards = rewards[1 : trajectory_length + 1].astype(np.float64)
    reward_limit = float(np.max(np.abs(np.asarray(mdp["R"]))))
    value_limit = reward_limit / (1.0 - gamma)
    pair_current = states[:trajectory_length] * n_actions + actions[:trajectory_length]
    pair_following = (
        states[1 : trajectory_length + 1] * n_actions
        + actions[1 : trajectory_length + 1]
    )
    direct_exact, _ = iterative_pair_evaluation(
        pair_current, pair_following, transition_rewards, n_pairs,
        gamma, alpha, iterations, beta=None,
    )
    direct_softmax, _ = iterative_pair_evaluation(
        pair_current, pair_following, transition_rewards, n_pairs,
        gamma, alpha, iterations, beta=beta,
    )
    v_exact, _ = iterative_state_evaluation(
        states[: trajectory_length + 1], transition_rewards, n_states,
        gamma, alpha, iterations, beta=None, value_limit=value_limit,
    )
    v_softmax, _ = iterative_state_evaluation(
        states[: trajectory_length + 1], transition_rewards, n_states,
        gamma, alpha, iterations, beta=beta, value_limit=value_limit,
    )
    q_vfirst_exact, _ = recover_q(
        states[:trajectory_length], actions[:trajectory_length],
        states[1 : trajectory_length + 1], transition_rewards,
        v_exact, n_states, n_actions, gamma, beta=None,
    )
    q_vfirst_softmax, _ = recover_q(
        states[:trajectory_length], actions[:trajectory_length],
        states[1 : trajectory_length + 1], transition_rewards,
        v_softmax, n_states, n_actions, gamma, beta=beta,
    )
    return {
        "direct_exact": direct_exact.reshape(n_states, n_actions),
        "direct_softmax": direct_softmax.reshape(n_states, n_actions),
        "vfirst_nosplit_exact": q_vfirst_exact,
        "vfirst_nosplit_softmax": q_vfirst_softmax,
        "v_exact": v_exact,
        "v_softmax": v_softmax,
    }


def assert_estimate_equivalence(
    estimates: dict[str, np.ndarray],
    routes: dict[str, dict[str, Any]],
    exact: dict[str, np.ndarray],
) -> None:
    """Prove the reconstructed estimates equal the serialized route metrics."""
    for route in (
        "direct_exact",
        "direct_softmax",
        "vfirst_nosplit_exact",
        "vfirst_nosplit_softmax",
    ):
        error = float(np.max(np.abs(estimates[route] - exact["q_pi"])))
        serialized = float(routes[route]["q_sup_error"])
        if not math.isclose(error, serialized, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"estimate reconstruction mismatch for {route}")
    for key, route in (("v_exact", "vfirst_nosplit_exact"), ("v_softmax", "vfirst_nosplit_softmax")):
        error = float(np.max(np.abs(estimates[key] - exact["v_pi"])))
        serialized = float(routes[route]["v_sup_error"])
        if not math.isclose(error, serialized, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"value reconstruction mismatch for {route}")


def build_oracle_audit_namespace(
    mdp: dict[str, Any],
    exact: dict[str, np.ndarray],
    route_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Truth-based diagnostics; structurally separate from certificate inputs."""
    q_pi = np.asarray(exact["q_pi"]).reshape(int(mdp["nS"]), int(mdp["nA"]))
    v_pi = np.asarray(exact["v_pi"], dtype=np.float64)
    p0 = np.asarray(mdp["p0"], dtype=np.float64)
    audits: dict[str, Any] = {}
    any_false_ordering = False
    any_bound_violation = False
    any_bellman_violation = False
    any_value_decrease = False
    for route, record in route_records.items():
        donor_truth: list[dict[str, Any]] = []
        receiver_true_greedy = 0
        for state_entry in record["states"]:
            state = int(state_entry["state"])
            receiver = int(state_entry["receiver"])
            if math.isclose(
                float(q_pi[state, receiver]), float(np.max(q_pi[state])),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                receiver_true_greedy += 1
            for donor in state_entry["donors"]:
                if not donor["eligible"]:
                    continue
                true_gap = float(q_pi[state, receiver] - q_pi[state, donor["action"]])
                lcb = float(donor["lcb"])
                ordering_false = true_gap <= 0.0
                bound_violation = true_gap < lcb - AUDIT_TOLERANCE
                any_false_ordering |= ordering_false
                any_bound_violation |= bound_violation
                donor_truth.append(
                    {
                        "state": state,
                        "donor": int(donor["action"]),
                        "lcb": lcb,
                        "true_gap": true_gap,
                        "ordering_false": ordering_false,
                        "bound_violation": bound_violation,
                    }
                )
        audit: dict[str, Any] = {
            "update_emitted": bool(record["update_emitted"]),
            "receiver_true_greedy_states": receiver_true_greedy,
            "state_count": len(record["states"]),
            "emitted_donor_truth": donor_truth,
            "false_ordering_count": sum(item["ordering_false"] for item in donor_truth),
            "bound_violation_count": sum(item["bound_violation"] for item in donor_truth),
            "min_bellman_improvement": None,
            "min_value_improvement": None,
            "return_change": None,
            "bellman_violation": False,
            "value_decrease": False,
        }
        if record["update_emitted"]:
            pi_plus = np.asarray(record["policy_plus"], dtype=np.float64)
            bellman = np.array(
                [float(pi_plus[s] @ q_pi[s] - v_pi[s]) for s in range(q_pi.shape[0])]
            )
            exact_plus = policy_quantities(mdp, pi_plus)
            value_diff = np.asarray(exact_plus["v_pi"], dtype=np.float64) - v_pi
            audit["min_bellman_improvement"] = float(bellman.min())
            audit["min_value_improvement"] = float(value_diff.min())
            audit["return_change"] = float(p0 @ value_diff)
            audit["bellman_violation"] = bool(bellman.min() < -AUDIT_TOLERANCE)
            audit["value_decrease"] = bool(value_diff.min() < -AUDIT_TOLERANCE)
            any_bellman_violation |= audit["bellman_violation"]
            any_value_decrease |= audit["value_decrease"]
        audits[route] = audit
    return {
        "purpose": "empirical truth-based audit only; not proof or certificate input",
        "tolerance": AUDIT_TOLERANCE,
        "routes": audits,
        "any_false_ordering": any_false_ordering,
        "any_bound_violation": any_bound_violation,
        "any_bellman_violation": any_bellman_violation,
        "any_value_decrease": any_value_decrease,
    }


def build_action_gap_namespace(
    *,
    estimates: dict[str, np.ndarray],
    tu_certificate: dict[str, Any],
    policy: np.ndarray,
    pair_counts: np.ndarray,
    successor_counts: np.ndarray,
    total_successor_counts: np.ndarray,
    trajectory_length: int,
    beta: float,
    gamma: float,
    reward_bound: float,
    pi_min: float,
    theta: float,
) -> dict[str, Any]:
    """Call the pure module for all six routes on observable inputs only."""
    algorithm = tu_certificate["certificate_inputs"]["algorithm"]
    mode_ok = bool(
        algorithm["mode"] == "fixed_policy_synchronous"
        and algorithm["fixed_context"]
        and algorithm["synchronous_update"]
    )
    radii = tu_certificate["event"]["recovery"]["radius_by_group"]
    state_exact_bound = tu_certificate["state_value"]["exact"]["total_bound"]
    state_softmax_bound = tu_certificate["state_value"]["softmax"]["total_bound"]
    route_bounds = {
        route: tu_certificate["routes"][route]["total_bound"]
        for route in (
            "direct_exact",
            "direct_softmax",
            "vfirst_nosplit_exact",
            "vfirst_nosplit_softmax",
        )
    }
    common = {
        "policy": policy,
        "pi_min": pi_min,
        "theta": theta,
        "gamma": gamma,
        "beta": beta,
        "trajectory_length": trajectory_length,
        "reward_bound": reward_bound,
        "pair_counts": pair_counts.tolist(),
        "algorithm_mode_ok": mode_ok,
    }
    local_data = {
        "successor_counts": successor_counts.astype(np.int64).tolist(),
        "total_successor_counts": total_successor_counts.astype(np.int64).tolist(),
        "recovery_radii": radii,
    }
    route_records = {
        "vfirst_local_exact": agc.evaluate_route_update(
            family="vfirst", matching="exact", scope="local",
            q_estimate=estimates["vfirst_nosplit_exact"],
            state_value_bound=state_exact_bound, diverged=False,
            **common, **local_data,
        ),
        "vfirst_local_softmax": agc.evaluate_route_update(
            family="vfirst", matching="softmax", scope="local",
            q_estimate=estimates["vfirst_nosplit_softmax"],
            state_value_bound=state_softmax_bound, diverged=False,
            **common, **local_data,
        ),
        "vfirst_global_exact": agc.evaluate_route_update(
            family="vfirst", matching="exact", scope="global",
            q_estimate=estimates["vfirst_nosplit_exact"],
            state_value_bound=state_exact_bound,
            global_q_bound=route_bounds["vfirst_nosplit_exact"], diverged=False,
            **common,
        ),
        "vfirst_global_softmax": agc.evaluate_route_update(
            family="vfirst", matching="softmax", scope="global",
            q_estimate=estimates["vfirst_nosplit_softmax"],
            state_value_bound=state_softmax_bound,
            global_q_bound=route_bounds["vfirst_nosplit_softmax"], diverged=False,
            **common,
        ),
        "direct_global_exact": agc.evaluate_route_update(
            family="direct", matching="exact", scope="global",
            q_estimate=estimates["direct_exact"],
            global_q_bound=route_bounds["direct_exact"],
            diverged=bool(algorithm["direct_exact_diverged"]),
            **common,
        ),
        "direct_global_softmax": agc.evaluate_route_update(
            family="direct", matching="softmax", scope="global",
            q_estimate=estimates["direct_softmax"],
            global_q_bound=route_bounds["direct_softmax"],
            diverged=bool(algorithm["direct_softmax_diverged"]),
            **common,
        ),
    }
    dominance = {
        "exact": agc.check_local_global_dominance(
            route_records["vfirst_local_exact"], route_records["vfirst_global_exact"]
        ),
        "softmax": agc.check_local_global_dominance(
            route_records["vfirst_local_softmax"],
            route_records["vfirst_global_softmax"],
        ),
        "weak_update_exact": agc.check_weak_update_dominance(
            route_records["vfirst_local_exact"], route_records["vfirst_global_exact"]
        ),
        "weak_update_softmax": agc.check_weak_update_dominance(
            route_records["vfirst_local_softmax"],
            route_records["vfirst_global_softmax"],
        ),
    }
    for label in ("exact", "softmax"):
        result = dominance[label]
        if result["available"] and result["satisfied"] is False:
            raise AssertionError(f"local/global dominance failed for {label}")
        weak = dominance[f"weak_update_{label}"]
        if weak["available"] and weak["satisfied"] is False:
            raise AssertionError(f"weak update dominance failed for {label}")
    return {
        "task_id": TASK_ID,
        "probability_statement": PROBABILITY_STATEMENT,
        "delta": float(tu_certificate["certificate_inputs"]["declared"]["delta"]),
        "transfer_fraction": theta,
        "routes": route_records,
        "dominance": dominance,
    }


def augment_summary_action_gap(
    summary: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach per-row action-gap metrics for the four theorem routes."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        for route in SUMMARY_ROUTE_MAP:
            key = (
                int(record["trajectory_length"]),
                int(record["n_actions"]),
                float(record["pi_min"]),
                float(record["beta"]),
                float(record["mixing"]),
                float(record["gap_bonus"]),
                route,
            )
            grouped.setdefault(key, []).append(record)
    output: list[dict[str, Any]] = []
    for row in summary:
        route_name = str(row["route"])
        if route_name not in SUMMARY_ROUTE_MAP:
            output.append(row)
            continue
        key = (
            int(row["trajectory_length"]),
            int(row["n_actions"]),
            float(row["pi_min"]),
            float(row["beta"]),
            float(row["mixing"]),
            float(row["gap_bonus"]),
            route_name,
        )
        selected = grouped[key]
        route_stats: dict[str, Any] = {}
        for new_route in SUMMARY_ROUTE_MAP[route_name]:
            entries = [item["action_gap_certificate"]["routes"][new_route] for item in selected]
            audits = [
                item["action_gap_certificate"]["oracle_audit"]["routes"][new_route]
                for item in selected
            ]
            emitted = [entry for entry in entries if entry["update_emitted"]]
            reason_counts: dict[str, int] = {}
            for entry in entries:
                for reason in entry["reasons"]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            bellman_mins = [
                audit["min_bellman_improvement"]
                for audit in audits
                if audit["min_bellman_improvement"] is not None
            ]
            route_stats[new_route] = {
                "records": len(entries),
                "update_emission_count": len(emitted),
                "update_emission_rate": len(emitted) / len(entries),
                "eligible_donor_mean": float(
                    np.mean([entry["eligible_donor_count"] for entry in entries])
                ),
                "changed_state_mean": float(
                    np.mean([entry["changed_state_count"] for entry in entries])
                ),
                "transferred_mass_mean": float(
                    np.mean([entry["total_transferred_mass"] for entry in entries])
                ),
                "abstention_reason_counts": dict(sorted(reason_counts.items())),
                "oracle_false_ordering_count": sum(
                    audit["false_ordering_count"] for audit in audits
                ),
                "oracle_bound_violation_count": sum(
                    audit["bound_violation_count"] for audit in audits
                ),
                "oracle_value_decrease_count": sum(
                    audit["value_decrease"] for audit in audits
                ),
                "oracle_min_bellman_improvement": (
                    None if not bellman_mins else float(np.min(bellman_mins))
                ),
                "oracle_return_change_mean": float(
                    np.mean(
                        [
                            audit["return_change"]
                            for audit in audits
                            if audit["return_change"] is not None
                        ]
                    )
                ) if any(audit["return_change"] is not None for audit in audits) else None,
            }
        row = dict(row)
        row["action_gap_certificate"] = {"routes": route_stats}
        output.append(row)
    return output


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
    parser.add_argument("--transfer-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/FP-ADV-001/claude/smoke")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    if args.transfer_fraction != 0.5:
        raise ValueError("the transfer fraction is frozen at 0.5")
    n_actions_grid = args.n_actions or [4]
    pi_min_grid = args.pi_mins or [0.05]
    beta_grid = args.betas or [8.0]
    mixing_grid = args.mixing or [0.08, 0.50]
    gap_grid = args.gap_bonuses or [0.0, 0.50]
    iterations = args.iterations or 160
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

    fptu_before = verify_baseline(FPTU_BASELINE_DIR, FPTU_BASELINE_HASHES, "FP-TU-001")
    fpmart_before = verify_baseline(FPMART_BASELINE_DIR, FPMART_BASELINE_HASHES, "FP-MART-001")
    checks_log = run_preflight_checks(project_dir)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = subprocess.list2cmdline(
        [sys.executable, "-B", Path(__file__).name, *sys.argv[1:]]
    )
    (output_dir / "commands.log").write_text(
        f"WORKDIR: {project_dir}\nEVALUATION: {command}\n", encoding="utf-8"
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
            "common_activation_commit": MART_ACTIVATION_COMMIT,
            "seed_schedule_tasks_per_cell": FROZEN_SEED_TASKS_PER_CELL,
        },
        "time_uniform_certificate": {
            "task_id": "FP-TU-001",
            "method": "finite_geometric_time_uniform_cosh_mixture",
            "components": 15,
            "target_counts": [2**index for index in range(15)],
            "reported_boundary": "mixture_root",
            "stitch_role": "audit_and_upper_bracket_only",
            "tolerance": 1e-12,
            "max_iterations": 200,
            "common_activation_commit": TU_ACTIVATION_COMMIT,
            "frozen_regression_baseline": fpmart_before,
        },
        "action_gap_certificate": {
            "task_id": TASK_ID,
            "probability_statement": PROBABILITY_STATEMENT,
            "transfer_fraction": args.transfer_fraction,
            "routes": list(ROUTE_NAMES),
            "local_routes": list(LOCAL_ROUTES),
            "control_routes": list(GLOBAL_ROUTES),
            "reason_order": list(agc.REASON_ORDER),
            "receiver_rule": "argmax_a q_hat(s,a), smallest-index tie break",
            "donor_rule": "both counts positive, applicable LCB > 0, pi(b|s) > pi_min",
            "common_activation_commit": ACTIVATION_COMMIT,
            "common_route_execution_start_commit": COMMON_ROUTE_COMMIT,
            "scientific_baseline": SCIENTIFIC_BASELINE,
            "frozen_regression_baseline": fptu_before,
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
                        next(child_seeds) for _ in range(FROZEN_SEED_TASKS_PER_CELL)
                    ]
                    for task_index in range(args.tasks):
                        child_seed = cell_seeds[task_index]
                        rng = np.random.default_rng(child_seed)
                        mdp = make_mdp(
                            args.n_states, n_actions, args.gamma, mixing, gap_bonus, rng
                        )
                        policy = make_policy(args.n_states, n_actions, pi_min, rng)
                        exact = policy_quantities(mdp, policy)
                        base_certificate = build_markov_base_certificate(
                            exact["p_pi"], exact["p_pair"]
                        )
                        start = int(rng.choice(args.n_states, p=exact["mu_state"]))
                        states, actions, rewards = rollout(
                            mdp, policy, start=start, n=maximum_length, rng=rng
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
                                    mdp, exact, states, actions, rewards,
                                    length, beta, args.alpha, iterations,
                                    certificate, args.certificate_delta,
                                )
                                certificate_arguments = {
                                    "state_counts": state_counts,
                                    "pair_counts": pair_counts,
                                    "trajectory_length": length,
                                    "reward_bound": reward_bound,
                                    "gamma": args.gamma,
                                    "alpha": args.alpha,
                                    "beta": beta,
                                    "delta": args.certificate_delta,
                                    "direct_exact_iterations": int(
                                        routes["direct_exact"]["iterations_used"]
                                    ),
                                    "direct_softmax_iterations": int(
                                        routes["direct_softmax"]["iterations_used"]
                                    ),
                                    "state_exact_iterations": int(
                                        routes["vfirst_nosplit_exact"]["value_iterations_used"]
                                    ),
                                    "state_softmax_iterations": int(
                                        routes["vfirst_nosplit_softmax"]["value_iterations_used"]
                                    ),
                                    "direct_exact_diverged": bool(
                                        routes["direct_exact"]["diverged"]
                                    ),
                                    "direct_softmax_diverged": bool(
                                        routes["direct_softmax"]["diverged"]
                                    ),
                                    "iteration_cap": iterations,
                                }
                                visit = build_visit_indexed_certificate(
                                    **certificate_arguments
                                )
                                time_uniform = build_time_uniform_certificate(
                                    **certificate_arguments
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
                                time_uniform["oracle_audit"] = build_oracle_audit(
                                    visit_certificate=time_uniform,
                                    legacy_certificate=certificate,
                                    routes=routes,
                                    actual_reward_abs_max=float(
                                        np.max(np.abs(np.asarray(mdp["R"])))
                                    ),
                                    declared_reward_abs_bound=reward_bound,
                                )

                                estimates = reconstruct_route_estimates(
                                    mdp, states, actions, rewards,
                                    length, beta, args.alpha, iterations,
                                )
                                assert_estimate_equivalence(estimates, routes, exact)
                                successor_counts = np.zeros(
                                    (args.n_states * n_actions, args.n_states)
                                )
                                np.add.at(
                                    successor_counts,
                                    (current_pairs, states[1 : length + 1]),
                                    1.0,
                                )
                                total_successor = np.bincount(
                                    states[1 : length + 1], minlength=args.n_states
                                )
                                action_gap = build_action_gap_namespace(
                                    estimates=estimates,
                                    tu_certificate=time_uniform,
                                    policy=policy,
                                    pair_counts=pair_counts,
                                    successor_counts=successor_counts,
                                    total_successor_counts=total_successor,
                                    trajectory_length=length,
                                    beta=beta,
                                    gamma=args.gamma,
                                    reward_bound=reward_bound,
                                    pi_min=pi_min,
                                    theta=args.transfer_fraction,
                                )
                                action_gap["oracle_audit"] = build_oracle_audit_namespace(
                                    mdp, exact, action_gap["routes"]
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
                                        "true_action_gap_min": float(np.min(action_gaps)),
                                        "true_action_gap_mean": float(
                                            np.mean(action_gaps)
                                        ),
                                        "certificate": certificate,
                                        "routes": routes,
                                        "visit_indexed_certificate": visit,
                                        "time_uniform_certificate": time_uniform,
                                        "action_gap_certificate": action_gap,
                                    }
                                )

    legacy_summary = summarize(task_results)
    vi_summary = vi_augment_summary(legacy_summary, task_results)
    tu_summary = tu_augment_summary(vi_summary, task_results)
    summary = augment_summary_action_gap(tu_summary, task_results)
    write_json(output_dir / "task_results.json", task_results)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "regression.json",
        {
            "status": "pending_analyzer",
            "baseline": fptu_before,
            "generated_records": len(task_results),
        },
    )

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
        "common_activation_commit": ACTIVATION_COMMIT,
        "common_route_execution_start_commit": COMMON_ROUTE_COMMIT,
        "task_id": TASK_ID,
        "frozen_baselines": {"FP-TU-001": fptu_before, "FP-MART-001": fpmart_before},
    }
    write_json(output_dir / "environment.json", environment)

    fptu_after = verify_baseline(FPTU_BASELINE_DIR, FPTU_BASELINE_HASHES, "FP-TU-001")
    fpmart_after = verify_baseline(FPMART_BASELINE_DIR, FPMART_BASELINE_HASHES, "FP-MART-001")
    if fptu_before != fptu_after or fpmart_before != fpmart_after:
        raise RuntimeError("frozen baseline changed during evaluation")

    emissions = {route: 0 for route in ROUTE_NAMES}
    for record in task_results:
        for route in ROUTE_NAMES:
            if record["action_gap_certificate"]["routes"][route]["update_emitted"]:
                emissions[route] += 1
    print(f"wrote {len(task_results)} matched comparisons to {output_dir}")
    for route in ROUTE_NAMES:
        print(f"{route:24s} updates={emissions[route]}/{len(task_results)}")
    print(f"PASS {TASK_ID} paired evaluation")


if __name__ == "__main__":
    main()
