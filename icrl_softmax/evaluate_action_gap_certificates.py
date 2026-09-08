"""Paired FP-ADV-001 evaluator preserving the frozen FP-TU-001 output."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_fixed_policy_q_routes as fixed
import evaluate_time_uniform_certificates as time_evaluator
import evaluate_visit_indexed_certificates as legacy
from action_gap_certificate import build_action_gap_certificate


TASK_ID = "FP-ADV-001"
SCIENTIFIC_ACTIVATION_COMMIT = "10a9a94e24ec92a59e7c756f9af6ce07b2f30e59"
ROUTE_BASELINE_COMMIT = "4078f6911cbfb4654205772685f49896e4e8cad2"
BASELINE_HASHES = {
    "config.json": "43dcb96b0f6f95e76f1c0b484d6375e3727dbb5609b16a8952a69e2ac0dddf3a",
    "task_results.json": "0e5eab39bf49894832c5ebcdd6f7b70c453fff6b9600b234617889f8f9fa79be",
    "summary.json": "565fc4d261a938d13350984bb97242e79fba42f014d11f807a18942517f4444f",
}
ACTION_ROUTES = (
    "vfirst_local_exact",
    "vfirst_local_softmax",
    "vfirst_global_exact",
    "vfirst_global_softmax",
    "direct_global_exact",
    "direct_global_softmax",
)
SUMMARY_ROUTE_MAP = {
    "vfirst_nosplit_exact": ("vfirst_local_exact", "vfirst_global_exact"),
    "vfirst_nosplit_softmax": ("vfirst_local_softmax", "vfirst_global_softmax"),
    "direct_exact": ("direct_global_exact",),
    "direct_softmax": ("direct_global_softmax",),
}
AUDIT_TOLERANCE = 1e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_load(path: Path) -> Any:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {path}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
    )


def verify_frozen_time_uniform_baseline(path: Path) -> dict[str, Any]:
    observed = {name: sha256_file(path / name) for name in BASELINE_HASHES}
    if observed != BASELINE_HASHES:
        raise RuntimeError(f"frozen FP-TU-001 baseline mismatch: {observed}")
    records = _strict_load(path / "task_results.json")
    if not isinstance(records, list) or len(records) != 480:
        raise RuntimeError("frozen FP-TU-001 baseline must contain 480 records")
    return {"directory": str(path), "hashes": observed, "record_count": 480}


def _compare_legacy(left: Any, right: Any, path: str = "root") -> dict[str, Any]:
    result = {
        "numeric_compared": 0,
        "nonnumeric_compared": 0,
        "numeric_mismatches": [],
        "nonnumeric_mismatches": [],
    }

    def visit(first: Any, second: Any, location: str) -> None:
        if isinstance(first, dict) and isinstance(second, dict):
            if set(first) != set(second):
                result["nonnumeric_mismatches"].append(
                    {
                        "path": location,
                        "left_keys": sorted(first),
                        "right_keys": sorted(second),
                    }
                )
                return
            for key in first:
                visit(first[key], second[key], f"{location}.{key}")
            return
        if isinstance(first, list) and isinstance(second, list):
            if len(first) != len(second):
                result["nonnumeric_mismatches"].append(
                    {"path": location, "left_length": len(first), "right_length": len(second)}
                )
                return
            for index, (item_left, item_right) in enumerate(zip(first, second, strict=True)):
                visit(item_left, item_right, f"{location}[{index}]")
            return
        numeric_left = isinstance(first, (int, float)) and not isinstance(first, bool)
        numeric_right = isinstance(second, (int, float)) and not isinstance(second, bool)
        if numeric_left and numeric_right:
            result["numeric_compared"] += 1
            if not math.isclose(
                float(first), float(second), rel_tol=1e-12, abs_tol=1e-12
            ):
                result["numeric_mismatches"].append(
                    {"path": location, "generated": first, "baseline": second}
                )
            return
        result["nonnumeric_compared"] += 1
        if type(first) is not type(second) or first != second:
            result["nonnumeric_mismatches"].append(
                {"path": location, "generated": first, "baseline": second}
            )

    visit(left, right, path)
    result["status"] = (
        "PASS"
        if not result["numeric_mismatches"] and not result["nonnumeric_mismatches"]
        else "FAIL"
    )
    return result


def _reconstruct_estimates(
    mdp: dict[str, Any],
    exact: dict[str, np.ndarray],
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    trajectory_length: int,
    beta: float,
    alpha: float,
    iterations: int,
    policy: np.ndarray,
) -> dict[str, Any]:
    n_states = int(mdp["nS"])
    n_actions = int(mdp["nA"])
    n_pairs = n_states * n_actions
    gamma = float(mdp["gamma"])
    current_states = states[:trajectory_length]
    next_states = states[1 : trajectory_length + 1]
    current_actions = actions[:trajectory_length]
    current_pairs = current_states * n_actions + current_actions
    next_pairs = next_states * n_actions + actions[1 : trajectory_length + 1]
    transition_rewards = rewards[1 : trajectory_length + 1].astype(np.float64)
    reward_limit = float(np.max(np.abs(np.asarray(mdp["R"]))))
    value_limit = reward_limit / (1.0 - gamma)

    direct_exact, direct_exact_diag = fixed.iterative_pair_evaluation(
        current_pairs,
        next_pairs,
        transition_rewards,
        n_pairs,
        gamma,
        alpha,
        iterations,
        beta=None,
    )
    direct_softmax, direct_softmax_diag = fixed.iterative_pair_evaluation(
        current_pairs,
        next_pairs,
        transition_rewards,
        n_pairs,
        gamma,
        alpha,
        iterations,
        beta=beta,
    )
    value_exact, _ = fixed.iterative_state_evaluation(
        states[: trajectory_length + 1],
        transition_rewards,
        n_states,
        gamma,
        alpha,
        iterations,
        beta=None,
        value_limit=value_limit,
    )
    value_softmax, _ = fixed.iterative_state_evaluation(
        states[: trajectory_length + 1],
        transition_rewards,
        n_states,
        gamma,
        alpha,
        iterations,
        beta=beta,
        value_limit=value_limit,
    )
    vfirst_exact, _ = fixed.recover_q(
        current_states,
        current_actions,
        next_states,
        transition_rewards,
        value_exact,
        n_states,
        n_actions,
        gamma,
        beta=None,
    )
    vfirst_softmax, _ = fixed.recover_q(
        current_states,
        current_actions,
        next_states,
        transition_rewards,
        value_softmax,
        n_states,
        n_actions,
        gamma,
        beta=beta,
    )
    pair_counts = np.bincount(current_pairs, minlength=n_pairs).astype(np.int64)
    pair_successors = np.zeros((n_pairs, n_states), dtype=np.int64)
    np.add.at(pair_successors, (current_pairs, next_states), 1)
    return {
        "policy": policy.copy(),
        "pair_counts": pair_counts,
        "pair_successor_counts": pair_successors,
        "q_estimates": {
            "vfirst_exact": vfirst_exact,
            "vfirst_softmax": vfirst_softmax,
            "direct_exact": direct_exact.reshape(n_states, n_actions),
            "direct_softmax": direct_softmax.reshape(n_states, n_actions),
        },
        "divergence_guards": {
            "direct_exact": bool(direct_exact_diag["diverged"]),
            "direct_softmax": bool(direct_softmax_diag["diverged"]),
        },
        "mdp": copy.deepcopy(mdp),
        "true_q": np.asarray(exact["q_pi"], dtype=np.float64).copy(),
        "true_v": np.asarray(exact["v_pi"], dtype=np.float64).copy(),
    }


def _prune_bound(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "finite_bound_emitted": bool(route["finite_bound_emitted"]),
        "total_bound": route["total_bound"],
        "failure_reasons": list(route["failure_reasons"]),
    }


def build_oracle_audit(
    certificate: dict[str, Any], capture: dict[str, Any]
) -> dict[str, Any]:
    old_policy = np.asarray(capture["policy"], dtype=np.float64)
    true_q = np.asarray(capture["true_q"], dtype=np.float64)
    true_v = np.asarray(capture["true_v"], dtype=np.float64)
    mdp = capture["mdp"]
    p0 = np.asarray(mdp["p0"], dtype=np.float64)
    route_audits: dict[str, dict[str, Any]] = {}
    for route_name, route in certificate["routes"].items():
        used: list[dict[str, Any]] = []
        for state_result in route["state_results"]:
            for comparison in state_result["comparisons"]:
                if not comparison["eligible"]:
                    continue
                state = int(comparison["state"])
                receiver = int(comparison["receiver"])
                donor = int(comparison["donor"])
                true_gap = float(true_q[state, receiver] - true_q[state, donor])
                used.append(
                    {
                        "state": state,
                        "receiver": receiver,
                        "donor": donor,
                        "gap_lcb": comparison["gap_lcb"],
                        "true_gap": true_gap,
                        "false_ordering": true_gap <= -AUDIT_TOLERANCE,
                    }
                )
        new_policy = np.asarray(route["policy_plus"], dtype=np.float64)
        bellman_change = np.sum((new_policy - old_policy) * true_q, axis=1)
        lcb = np.asarray(route["bellman_lcb_by_state"], dtype=np.float64)
        new_exact = fixed.policy_quantities(mdp, new_policy)
        new_v = np.asarray(new_exact["v_pi"], dtype=np.float64)
        value_change = new_v - true_v
        old_return = float(p0 @ true_v)
        new_return = float(p0 @ new_v)
        route_audits[route_name] = {
            "used_orderings": used,
            "used_ordering_count": len(used),
            "false_ordering_count": sum(item["false_ordering"] for item in used),
            "bellman_change_by_state": bellman_change.tolist(),
            "bellman_lcb_by_state": lcb.tolist(),
            "minimum_bellman_change": float(np.min(bellman_change)),
            "bellman_bound_violation_count": int(
                np.sum(bellman_change < lcb - AUDIT_TOLERANCE)
            ),
            "value_change_by_state": value_change.tolist(),
            "minimum_value_change": float(np.min(value_change)),
            "value_decrease_count": int(np.sum(value_change < -AUDIT_TOLERANCE)),
            "old_exact_return": old_return,
            "new_exact_return": new_return,
            "exact_return_change": new_return - old_return,
            "return_decrease": new_return < old_return - AUDIT_TOLERANCE,
        }
    return {
        "separation": (
            "computed after the pure certificate; no field in this audit was passed "
            "to build_action_gap_certificate"
        ),
        "tolerance": AUDIT_TOLERANCE,
        "routes": route_audits,
        "any_false_ordering": any(
            audit["false_ordering_count"] > 0 for audit in route_audits.values()
        ),
        "any_bellman_bound_violation": any(
            audit["bellman_bound_violation_count"] > 0
            for audit in route_audits.values()
        ),
        "any_value_decrease": any(
            audit["value_decrease_count"] > 0 for audit in route_audits.values()
        ),
        "any_return_decrease": any(
            audit["return_decrease"] for audit in route_audits.values()
        ),
    }


def _summary_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(record["trajectory_length"]),
        int(record["n_actions"]),
        float(record["pi_min"]),
        float(record["beta"]),
        float(record["mixing"]),
        float(record["gap_bonus"]),
    )


def _record_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(record["task_index"]),
        tuple(record["spawn_key"]),
        int(record["trajectory_length"]),
        int(record["n_states"]),
        int(record["n_actions"]),
        float(record["pi_min"]),
        float(record["beta"]),
        float(record["mixing"]),
        float(record["gap_bonus"]),
    )


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def augment_summary(
    baseline_summary: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_summary_key(record)].append(record)
    output: list[dict[str, Any]] = []
    for baseline_row in baseline_summary:
        row = copy.deepcopy(baseline_row)
        selected_routes = SUMMARY_ROUTE_MAP.get(str(row["route"]))
        if selected_routes is None:
            output.append(row)
            continue
        selected = grouped[
            (
                int(row["trajectory_length"]),
                int(row["n_actions"]),
                float(row["pi_min"]),
                float(row["beta"]),
                float(row["mixing"]),
                float(row["gap_bonus"]),
            )
        ]
        action_summary: dict[str, Any] = {}
        for route_name in selected_routes:
            routes = [item["action_gap_certificate"]["routes"][route_name] for item in selected]
            audits = [
                item["action_gap_certificate"]["oracle_audit"]["routes"][route_name]
                for item in selected
            ]
            action_summary[route_name] = {
                "records": len(selected),
                "update_emission_count": sum(
                    route["status"] == "safe_update_emitted" for route in routes
                ),
                "update_emission_rate": float(
                    np.mean([route["status"] == "safe_update_emitted" for route in routes])
                ),
                "updated_state_count": sum(int(route["updated_state_count"]) for route in routes),
                "eligible_donor_count": sum(int(route["eligible_donor_count"]) for route in routes),
                "mean_transferred_mass": _mean(
                    [float(route["total_transferred_mass"]) for route in routes]
                ),
                "mean_positive_bellman_lcb": _mean(
                    [
                        float(value)
                        for route in routes
                        for value in route["bellman_lcb_by_state"]
                        if float(value) > 0.0
                    ]
                ),
                "false_ordering_count": sum(
                    int(audit["false_ordering_count"]) for audit in audits
                ),
                "bellman_bound_violation_count": sum(
                    int(audit["bellman_bound_violation_count"]) for audit in audits
                ),
                "value_decrease_count": sum(
                    int(audit["value_decrease_count"]) for audit in audits
                ),
                "return_decrease_count": sum(bool(audit["return_decrease"]) for audit in audits),
            }
        row["action_gap_certificate"] = action_summary
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
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/FP-ADV-001/codex")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if float(args.transfer_fraction) != 0.5:
        raise ValueError("FP-ADV-001 freezes transfer_fraction at 0.5")
    project_dir = Path(__file__).resolve().parent
    result_dir = args.output_dir.resolve()
    baseline_dir = (project_dir / "results/FP-TU-001/codex").resolve()
    baseline_identity = verify_frozen_time_uniform_baseline(baseline_dir)
    baseline_config = _strict_load(baseline_dir / "config.json")
    baseline_records = _strict_load(baseline_dir / "task_results.json")
    baseline_summary = _strict_load(baseline_dir / "summary.json")

    captures: list[dict[str, Any]] = []
    current_policy: np.ndarray | None = None
    original_parse_args = legacy.parse_args
    original_evaluate_prefix = legacy.evaluate_prefix
    original_make_policy = legacy.make_policy
    original_preflight = legacy.PREFLIGHT_SCRIPTS

    def capture_policy(*policy_args: Any, **policy_kwargs: Any) -> np.ndarray:
        nonlocal current_policy
        policy = original_make_policy(*policy_args, **policy_kwargs)
        current_policy = np.asarray(policy, dtype=np.float64).copy()
        return policy

    def capture_evaluate_prefix(
        mdp: dict[str, Any],
        exact: dict[str, np.ndarray],
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        trajectory_length: int,
        beta: float,
        alpha: float,
        iterations: int,
        certificate: dict[str, Any],
        certificate_delta: float = 0.05,
    ) -> dict[str, dict[str, Any]]:
        if current_policy is None:
            raise RuntimeError("policy capture missing before prefix evaluation")
        routes = original_evaluate_prefix(
            mdp,
            exact,
            states,
            actions,
            rewards,
            trajectory_length,
            beta,
            alpha,
            iterations,
            certificate,
            certificate_delta=certificate_delta,
        )
        capture = _reconstruct_estimates(
            mdp,
            exact,
            states,
            actions,
            rewards,
            trajectory_length,
            beta,
            alpha,
            iterations,
            current_policy,
        )
        for estimate_key, route_key in (
            ("vfirst_exact", "vfirst_nosplit_exact"),
            ("vfirst_softmax", "vfirst_nosplit_softmax"),
            ("direct_exact", "direct_exact"),
            ("direct_softmax", "direct_softmax"),
        ):
            reconstructed = fixed.q_metrics(
                np.asarray(capture["q_estimates"][estimate_key]),
                np.asarray(capture["true_q"]),
                {},
            )
            for metric in ("q_sup_error", "q_mean_pair_error", "greedy_action_accuracy"):
                if not math.isclose(
                    float(reconstructed[metric]),
                    float(routes[route_key][metric]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise AssertionError(
                        f"reconstructed {estimate_key} changed legacy metric {metric}"
                    )
        captures.append(capture)
        return routes

    legacy.parse_args = lambda: args
    legacy.evaluate_prefix = capture_evaluate_prefix
    legacy.make_policy = capture_policy
    legacy.PREFLIGHT_SCRIPTS = (*original_preflight, "verify_action_gap_certificate.py")
    try:
        time_evaluator.main()
    finally:
        legacy.parse_args = original_parse_args
        legacy.evaluate_prefix = original_evaluate_prefix
        legacy.make_policy = original_make_policy
        legacy.PREFLIGHT_SCRIPTS = original_preflight

    generated_config = _strict_load(result_dir / "config.json")
    generated_records = _strict_load(result_dir / "task_results.json")
    generated_summary = _strict_load(result_dir / "summary.json")
    if len(generated_records) != len(captures):
        raise RuntimeError("captured estimate count differs from generated record count")
    if len(generated_records) == 480:
        preservation = {
            "config": _compare_legacy(generated_config, baseline_config, "config"),
            "task_results": _compare_legacy(
                generated_records, baseline_records, "task_results"
            ),
            "summary": _compare_legacy(generated_summary, baseline_summary, "summary"),
        }
    else:
        baseline_by_identity = {
            _record_identity(record): record for record in baseline_records
        }
        matched_baseline = []
        for record in generated_records:
            identity = _record_identity(record)
            if identity not in baseline_by_identity:
                raise RuntimeError(f"smoke record is absent from frozen baseline: {identity}")
            matched_baseline.append(baseline_by_identity[identity])
        preservation = {
            "config": {
                "status": "PASS",
                "scope": "smoke_configuration_is_a_declared_subset",
            },
            "task_results": _compare_legacy(
                generated_records, matched_baseline, "task_results"
            ),
            "summary": {
                "status": "PASS",
                "scope": "smoke_summary_is_not_compared_to_30-task_aggregates",
            },
        }
    if any(section["status"] != "PASS" for section in preservation.values()):
        raise RuntimeError(f"generated FP-TU leaves differ from frozen baseline: {preservation}")

    for record, capture in zip(generated_records, captures, strict=True):
        time_certificate = record["time_uniform_certificate"]
        action = build_action_gap_certificate(
            policy=capture["policy"],
            pair_counts=capture["pair_counts"],
            pair_successor_counts=capture["pair_successor_counts"],
            q_estimates=capture["q_estimates"],
            state_value_bounds={
                "exact": _prune_bound(time_certificate["state_value"]["exact"]),
                "softmax": _prune_bound(time_certificate["state_value"]["softmax"]),
            },
            global_q_bounds={
                "vfirst_exact": _prune_bound(
                    time_certificate["routes"]["vfirst_nosplit_exact"]
                ),
                "vfirst_softmax": _prune_bound(
                    time_certificate["routes"]["vfirst_nosplit_softmax"]
                ),
                "direct_exact": _prune_bound(
                    time_certificate["routes"]["direct_exact"]
                ),
                "direct_softmax": _prune_bound(
                    time_certificate["routes"]["direct_softmax"]
                ),
            },
            recovery_radii=time_certificate["event"]["recovery"]["radius_by_group"],
            trajectory_length=int(record["trajectory_length"]),
            reward_bound=1.0 + float(record["gap_bonus"]),
            gamma=float(args.gamma),
            beta=float(record["beta"]),
            pi_min=float(record["pi_min"]),
            transfer_fraction=float(args.transfer_fraction),
            algorithm_mode="fixed_policy_synchronous",
            divergence_guards=capture["divergence_guards"],
        )
        action["oracle_audit"] = build_oracle_audit(action, capture)
        record["action_gap_certificate"] = action

    generated_config["action_gap_certificate"] = {
        "task_id": TASK_ID,
        "scientific_activation_commit": SCIENTIFIC_ACTIVATION_COMMIT,
        "common_route_execution_start_commit": ROUTE_BASELINE_COMMIT,
        "primary_routes": ["vfirst_local_exact", "vfirst_local_softmax"],
        "control_routes": [
            "vfirst_global_exact",
            "vfirst_global_softmax",
            "direct_global_exact",
            "direct_global_softmax",
        ],
        "transfer_fraction": float(args.transfer_fraction),
        "probability_statement": (
            "P(EmitUpdate and (any used ordering is false or any value decreases)) <= delta"
        ),
        "frozen_regression_baseline": baseline_identity,
    }
    summary = augment_summary(generated_summary, generated_records)
    fixed.write_json(result_dir / "config.json", generated_config)
    fixed.write_json(result_dir / "task_results.json", generated_records)
    fixed.write_json(result_dir / "summary.json", summary)
    fixed.write_json(
        result_dir / "regression.json",
        {
            "status": "PASS",
            "baseline": baseline_identity,
            "generated_records": len(generated_records),
            "legacy_preservation": preservation,
            "analyzer_status": "pending_analyzer",
        },
    )

    command = subprocess.list2cmdline(
        [sys.executable, "-B", Path(__file__).name, *sys.argv[1:]]
    )
    (result_dir / "commands.log").write_text(
        f"WORKDIR: {project_dir}\nEVALUATION: {command}\n", encoding="utf-8"
    )
    environment = _strict_load(result_dir / "environment.json")
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    environment.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "script": str(Path(__file__).resolve()),
            "git_head_at_run": git_head,
            "task_id": TASK_ID,
            "scientific_activation_commit": SCIENTIFIC_ACTIVATION_COMMIT,
            "common_route_execution_start_commit": ROUTE_BASELINE_COMMIT,
            "frozen_regression_baseline": baseline_identity,
        }
    )
    fixed.write_json(result_dir / "environment.json", environment)
    baseline_after = verify_frozen_time_uniform_baseline(baseline_dir)
    if baseline_after != baseline_identity:
        raise RuntimeError("frozen FP-TU-001 baseline changed during evaluation")
    if len(generated_records) == 480 and not args.quick:
        print(f"PASS {TASK_ID} frozen paired evaluation with 480 records")
    else:
        print(f"PASS {TASK_ID} smoke paired evaluation with {len(generated_records)} records")


if __name__ == "__main__":
    main()
