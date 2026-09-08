"""Strict regression, formula, policy, and oracle audit for FP-ADV-001."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from action_gap_certificate import build_action_gap_certificate, strict_json_ready
from evaluate_action_gap_certificates import (
    ACTION_ROUTES,
    BASELINE_HASHES,
    TASK_ID,
    _compare_legacy,
    _record_identity,
    _strict_load,
    augment_summary,
    verify_frozen_time_uniform_baseline,
)


CORE_ARTIFACTS = {
    "config.json",
    "task_results.json",
    "summary.json",
    "regression.json",
    "environment.json",
    "commands.log",
    "checks.log",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strip_action_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_action_namespace(item)
            for key, item in value.items()
            if key != "action_gap_certificate"
        }
    if isinstance(value, list):
        return [_strip_action_namespace(item) for item in value]
    return value


def _contains_forbidden_oracle_input(value: Any, path: str = "root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if any(
                token in normalized
                for token in (
                    "oracle",
                    "truth",
                    "true_",
                    "occupancy",
                    "realized_error",
                    "exact_return",
                )
            ):
                findings.append(f"{path}.{key}")
            findings.extend(_contains_forbidden_oracle_input(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(_contains_forbidden_oracle_input(item, f"{path}[{index}]"))
    return findings


def _rebuild_certificate(stored: dict[str, Any]) -> dict[str, Any]:
    inputs = stored["certificate_inputs"]
    observed = inputs["observed"]
    declared = inputs["declared"]
    inherited = inputs["inherited"]
    algorithm = inputs["algorithm"]
    return build_action_gap_certificate(
        policy=observed["policy"],
        pair_counts=observed["pair_counts"],
        pair_successor_counts=observed["pair_successor_counts"],
        q_estimates=observed["q_estimates"],
        state_value_bounds=inherited["state_value_bounds"],
        global_q_bounds=inherited["global_q_bounds"],
        recovery_radii=observed["recovery_radii"],
        trajectory_length=int(declared["trajectory_length"]),
        reward_bound=float(declared["reward_bound"]),
        gamma=float(declared["gamma"]),
        beta=float(declared["beta"]),
        pi_min=float(declared["pi_min"]),
        transfer_fraction=float(declared["transfer_fraction"]),
        algorithm_mode=str(algorithm["mode"]),
        divergence_guards=algorithm["divergence_guards"],
    )


def _formal_configuration(config: dict[str, Any], records: list[dict[str, Any]]) -> bool:
    return bool(
        len(records) == 480
        and config["tasks_per_cell"] == 30
        and config["trajectory_lengths"] == [256, 1024, 4096, 16384]
        and config["n_states"] == 6
        and config["n_actions"] == [4]
        and config["pi_mins"] == [0.05]
        and config["betas"] == [8.0]
        and config["mixing"] == [0.08, 0.5]
        and config["gap_bonuses"] == [0.0, 0.5]
        and math.isclose(float(config["gamma"]), 0.70)
        and math.isclose(float(config["alpha"]), 0.65)
        and config["iterations"] == 160
        and math.isclose(float(config["certificate_delta"]), 0.05)
        and config["seed"] == 20260829
        and config["quick"] is False
        and math.isclose(
            float(config["action_gap_certificate"]["transfer_fraction"]), 0.5
        )
    )


def analyze(result_dir: Path, *, allow_smoke: bool) -> dict[str, Any]:
    project_dir = Path(__file__).resolve().parent
    baseline_dir = (project_dir / "results/FP-TU-001/codex").resolve()
    baseline_identity = verify_frozen_time_uniform_baseline(baseline_dir)
    config = _strict_load(result_dir / "config.json")
    records = _strict_load(result_dir / "task_results.json")
    summary = _strict_load(result_dir / "summary.json")
    environment = _strict_load(result_dir / "environment.json")
    if not isinstance(records, list) or not records:
        raise RuntimeError("task_results.json must contain records")
    formal = _formal_configuration(config, records)
    if not formal and not allow_smoke:
        raise RuntimeError("result is not the frozen 480-record formal matrix")
    if config["action_gap_certificate"]["task_id"] != TASK_ID:
        raise RuntimeError("action-gap task identity mismatch")
    if environment.get("task_id") != TASK_ID:
        raise RuntimeError("environment task identity mismatch")
    if config["action_gap_certificate"]["frozen_regression_baseline"] != baseline_identity:
        raise RuntimeError("serialized frozen baseline identity mismatch")

    baseline_config = _strict_load(baseline_dir / "config.json")
    baseline_records = _strict_load(baseline_dir / "task_results.json")
    baseline_summary = _strict_load(baseline_dir / "summary.json")
    stripped_config = _strip_action_namespace(config)
    stripped_records = _strip_action_namespace(records)
    stripped_summary = _strip_action_namespace(summary)
    if formal:
        preservation = {
            "config": _compare_legacy(stripped_config, baseline_config, "config"),
            "task_results": _compare_legacy(
                stripped_records, baseline_records, "task_results"
            ),
            "summary": _compare_legacy(stripped_summary, baseline_summary, "summary"),
        }
    else:
        baseline_by_identity = {
            _record_identity(record): record for record in baseline_records
        }
        matched = []
        for record in stripped_records:
            identity = _record_identity(record)
            if identity not in baseline_by_identity:
                raise RuntimeError(f"smoke record absent from frozen baseline: {identity}")
            matched.append(baseline_by_identity[identity])
        preservation = {
            "config": {"status": "PASS", "scope": "declared_smoke_subset"},
            "task_results": _compare_legacy(
                stripped_records, matched, "task_results"
            ),
            "summary": {"status": "PASS", "scope": "smoke_aggregate"},
        }
    if any(section["status"] != "PASS" for section in preservation.values()):
        raise AssertionError(f"legacy regression failed: {preservation}")

    reconstruction_mismatches: list[dict[str, Any]] = []
    forbidden_inputs: list[dict[str, Any]] = []
    route_counts = Counter()
    reason_counts = Counter()
    oracle_counts = Counter()
    dominance_violations = 0
    decision_dominance_violations = 0
    policy_violations = 0
    compared_numeric = 0
    compared_nonnumeric = 0
    exact_softmax_same_updates = 0
    exact_softmax_comparisons = 0
    for record_index, record in enumerate(records):
        stored = record["action_gap_certificate"]
        forbidden = _contains_forbidden_oracle_input(
            stored["certificate_inputs"],
            f"task_results[{record_index}].action_gap_certificate.certificate_inputs",
        )
        if forbidden:
            forbidden_inputs.append({"record": record_index, "paths": forbidden})
        rebuilt = _rebuild_certificate(stored)
        stored_pure = {key: value for key, value in stored.items() if key != "oracle_audit"}
        comparison = _compare_legacy(
            rebuilt,
            stored_pure,
            f"task_results[{record_index}].action_gap_certificate",
        )
        compared_numeric += int(comparison["numeric_compared"])
        compared_nonnumeric += int(comparison["nonnumeric_compared"])
        if comparison["status"] != "PASS":
            reconstruction_mismatches.append(
                {
                    "record": record_index,
                    "numeric": comparison["numeric_mismatches"][:20],
                    "nonnumeric": comparison["nonnumeric_mismatches"][:20],
                }
            )
        routes = stored["routes"]
        if tuple(routes) != ACTION_ROUTES:
            raise AssertionError("action route order or identity changed")
        for route_name, route in routes.items():
            route_counts[f"{route_name}.records"] += 1
            route_counts[f"{route_name}.emitted"] += route["status"] == "safe_update_emitted"
            route_counts[f"{route_name}.states"] += int(route["updated_state_count"])
            route_counts[f"{route_name}.donors"] += int(route["eligible_donor_count"])
            for reason in route["failure_reasons"]:
                reason_counts[f"{route_name}.{reason}"] += 1
            policy = np.asarray(route["policy_plus"], dtype=np.float64)
            declared = stored["certificate_inputs"]["declared"]
            if (
                not np.all(np.isfinite(policy))
                or np.any(policy < float(declared["pi_min"]) - 1e-12)
                or not np.allclose(np.sum(policy, axis=1), 1.0, rtol=0.0, atol=1e-12)
                or np.any(np.asarray(route["bellman_lcb_by_state"]) < -1e-15)
            ):
                policy_violations += 1
        for matching in ("exact", "softmax"):
            dominance = stored["local_global_dominance"][matching]
            if dominance["all_local_penalties_nonincreasing"] is not True:
                dominance_violations += 1
            local_name = f"vfirst_local_{matching}"
            global_name = f"vfirst_global_{matching}"
            local_donors = {
                (item["state"], item["donor"])
                for state in routes[local_name]["state_results"]
                for item in state["comparisons"]
                if item["eligible"]
            }
            global_donors = {
                (item["state"], item["donor"])
                for state in routes[global_name]["state_results"]
                for item in state["comparisons"]
                if item["eligible"]
            }
            if not global_donors.issubset(local_donors):
                decision_dominance_violations += 1
        exact_softmax_comparisons += 1
        if routes["vfirst_local_exact"]["policy_plus"] == routes["vfirst_local_softmax"]["policy_plus"]:
            exact_softmax_same_updates += 1

        audit = stored["oracle_audit"]
        for route_name, route_audit in audit["routes"].items():
            oracle_counts[f"{route_name}.used_orderings"] += int(
                route_audit["used_ordering_count"]
            )
            oracle_counts[f"{route_name}.false_orderings"] += int(
                route_audit["false_ordering_count"]
            )
            oracle_counts[f"{route_name}.bellman_bound_violations"] += int(
                route_audit["bellman_bound_violation_count"]
            )
            oracle_counts[f"{route_name}.value_decreases"] += int(
                route_audit["value_decrease_count"]
            )
            oracle_counts[f"{route_name}.return_decreases"] += bool(
                route_audit["return_decrease"]
            )

    if reconstruction_mismatches:
        raise AssertionError(f"serialized formula reconstruction failed: {reconstruction_mismatches[:3]}")
    if forbidden_inputs:
        raise AssertionError(f"oracle data leaked into pure inputs: {forbidden_inputs[:3]}")
    if policy_violations:
        raise AssertionError(f"invalid policy outputs: {policy_violations}")
    if dominance_violations or decision_dominance_violations:
        raise AssertionError(
            "local/global dominance failed: "
            f"penalties={dominance_violations}, decisions={decision_dominance_violations}"
        )

    core_present = {path.name for path in result_dir.iterdir() if path.is_file()}
    if core_present != CORE_ARTIFACTS:
        raise AssertionError(
            f"formal result directory must contain exactly seven core files: {sorted(core_present)}"
        )
    regression = {
        "status": "PASS",
        "task_id": TASK_ID,
        "scope": "formal_480" if formal else "smoke_subset",
        "record_count": len(records),
        "frozen_baseline": baseline_identity,
        "frozen_hashes_match": baseline_identity["hashes"] == BASELINE_HASHES,
        "legacy_preservation": preservation,
        "formula_reconstruction": {
            "status": "PASS",
            "numeric_leaves_compared": compared_numeric,
            "nonnumeric_leaves_compared": compared_nonnumeric,
            "mismatch_count": 0,
        },
        "oracle_input_separation": {"status": "PASS", "finding_count": 0},
        "policy_validation": {"status": "PASS", "violation_count": 0},
        "local_global_penalty_dominance": {
            "status": "PASS",
            "violation_count": 0,
        },
        "local_global_decision_dominance": {
            "status": "PASS",
            "violation_count": 0,
        },
        "route_counts": dict(sorted(route_counts.items())),
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "oracle_audit_counts": dict(sorted(oracle_counts.items())),
        "exact_softmax_update_agreement": {
            "same_policy_count": exact_softmax_same_updates,
            "record_count": exact_softmax_comparisons,
            "rate": exact_softmax_same_updates / exact_softmax_comparisons,
        },
    }
    return strict_json_ready(regression)


def repair_no_update_policy_identity(result_dir: Path) -> int:
    """Correct the first formal serialization without rerunning its matrix.

    The initial serializer normalized the selected receiver even when a state
    had no eligible donor, causing at most 6.94e-17 representation drift.  This
    deterministic repair uses only serialized observable inputs.  Every formal
    route abstained, so its exact oracle consequence is the identity policy.
    """
    project_dir = Path(__file__).resolve().parent
    records = _strict_load(result_dir / "task_results.json")
    baseline_summary = _strict_load(
        project_dir / "results/FP-TU-001/codex/summary.json"
    )
    repaired = 0
    for record in records:
        certificate = record["action_gap_certificate"]
        original = certificate["certificate_inputs"]["observed"]["policy"]
        n_states = len(original)
        for route_name, route in certificate["routes"].items():
            if int(route["eligible_donor_count"]) != 0:
                continue
            if route["policy_plus"] != original:
                repaired += 1
            route["policy_plus"] = [list(row) for row in original]
            route["minimum_policy_mass"] = min(
                float(value) for row in original for value in row
            )
            route["maximum_row_sum_error"] = max(
                abs(sum(float(value) for value in row) - 1.0) for row in original
            )
            audit = certificate["oracle_audit"]["routes"][route_name]
            if int(audit["used_ordering_count"]) != 0:
                raise RuntimeError("cannot identity-repair a route with a used ordering")
            zeros = [0.0] * n_states
            audit["bellman_change_by_state"] = zeros
            audit["bellman_lcb_by_state"] = zeros
            audit["minimum_bellman_change"] = 0.0
            audit["bellman_bound_violation_count"] = 0
            audit["value_change_by_state"] = zeros
            audit["minimum_value_change"] = 0.0
            audit["value_decrease_count"] = 0
            audit["new_exact_return"] = audit["old_exact_return"]
            audit["exact_return_change"] = 0.0
            audit["return_decrease"] = False
        route_audits = certificate["oracle_audit"]["routes"].values()
        certificate["oracle_audit"]["any_false_ordering"] = any(
            int(audit["false_ordering_count"]) > 0 for audit in route_audits
        )
        route_audits = certificate["oracle_audit"]["routes"].values()
        certificate["oracle_audit"]["any_bellman_bound_violation"] = any(
            int(audit["bellman_bound_violation_count"]) > 0 for audit in route_audits
        )
        route_audits = certificate["oracle_audit"]["routes"].values()
        certificate["oracle_audit"]["any_value_decrease"] = any(
            int(audit["value_decrease_count"]) > 0 for audit in route_audits
        )
        route_audits = certificate["oracle_audit"]["routes"].values()
        certificate["oracle_audit"]["any_return_decrease"] = any(
            bool(audit["return_decrease"]) for audit in route_audits
        )
    summary = augment_summary(baseline_summary, records)
    (result_dir / "task_results.json").write_text(
        json.dumps(strict_json_ready(records), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (result_dir / "summary.json").write_text(
        json.dumps(strict_json_ready(summary), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    environment = _strict_load(result_dir / "environment.json")
    environment["post_formal_representation_repair"] = {
        "reason": "no-donor policy identity drift at most 6.94e-17",
        "routes_repaired": repaired,
        "formal_matrix_rerun": False,
    }
    (result_dir / "environment.json").write_text(
        json.dumps(strict_json_ready(environment), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return repaired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path, default=Path("results/FP-ADV-001/codex")
    )
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--repair-no-update-identity", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    if args.repair_no_update_identity:
        repaired = repair_no_update_policy_identity(result_dir)
        print(
            f"REPAIRED {repaired} no-update policy serializations without rerunning "
            "the formal matrix"
        )
    regression = analyze(result_dir, allow_smoke=bool(args.allow_smoke))
    (result_dir / "regression.json").write_text(
        json.dumps(regression, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    command = subprocess.list2cmdline(
        [sys.executable, "-B", Path(__file__).name, *sys.argv[1:]]
    )
    with (result_dir / "commands.log").open("a", encoding="utf-8") as handle:
        handle.write(f"ANALYSIS: {command}\n")
    with (result_dir / "checks.log").open("a", encoding="utf-8") as handle:
        handle.write(
            f"PASS {TASK_ID} strict analyzer; scope={regression['scope']}; "
            f"records={regression['record_count']}\n"
        )
    print(
        f"PASS {TASK_ID} strict analyzer with {regression['record_count']} records "
        f"({regression['scope']})"
    )


if __name__ == "__main__":
    main()
