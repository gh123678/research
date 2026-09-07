"""Strict regression and audit analysis for visit-indexed certificates."""

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

from analyze_fixed_policy_finite_sample_certificates import (
    compare_value,
    load_strict_json,
    record_key,
)
from fixed_policy_finite_sample_certificate import strict_json_ready
from visit_indexed_martingale_certificate import (
    STATUS_NOT_CERTIFIED,
    STATUS_SELECTIVE_HIGH_PROBABILITY,
)


BASELINE_HASHES = {
    "config.json": "a2276eae06ba8689864cac2ad3d3d0e92b069014b2046d0180ee172bb8296ea0",
    "task_results.json": "c84329bd6b9fcd495789f2067f250ead28fec807193d1b69ae1038e9cc3b2f25",
    "summary.json": "49846c0825c859df28ff773164352d19f6cb86943c484ab411bac7f69434b5ae",
}
THEOREM_ROUTES = (
    "direct_exact",
    "direct_softmax",
    "vfirst_nosplit_exact",
    "vfirst_nosplit_softmax",
)
EXPECTED_EXACT_EMISSION = {256: 0.025, 1024: 0.85, 4096: 1.0, 16384: 1.0}
BANNED_NEW_KEYS = {
    "occupancy",
    "transition_matrix",
    "spectral_gap",
    "q_pi",
    "v_pi",
    "true_residual",
    "true_initial_error",
    "oracle_audit",
}
SUMMARY_KEY_FIELDS = (
    "trajectory_length",
    "n_actions",
    "pi_min",
    "beta",
    "mixing",
    "gap_bonus",
    "route",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_records(
    records: list[dict[str, Any]], name: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates: list[tuple[Any, ...]] = []
    for record in records:
        key = record_key(record)
        if key in indexed:
            duplicates.append(key)
        indexed[key] = record
    if duplicates:
        raise AssertionError(f"{name} contains duplicate record keys: {duplicates[:5]}")
    return indexed


def regression_report(
    old_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    *,
    require_exact_keys: bool,
) -> dict[str, Any]:
    old_by_key = unique_records(old_records, "baseline task_results")
    new_by_key = unique_records(new_records, "new task_results")
    old_keys = set(old_by_key)
    new_keys = set(new_by_key)
    mismatches: list[dict[str, Any]] = []
    numeric_differences: list[float] = []
    if not new_keys <= old_keys:
        mismatches.append(
            {
                "path": "task_keys",
                "baseline_only_count": len(old_keys - new_keys),
                "new_only_count": len(new_keys - old_keys),
            }
        )
    if require_exact_keys and old_keys != new_keys:
        mismatches.append(
            {
                "path": "formal_task_keys",
                "baseline_only_count": len(old_keys - new_keys),
                "new_only_count": len(new_keys - old_keys),
            }
        )
    for key in sorted(old_keys & new_keys):
        compare_value(
            old_by_key[key],
            new_by_key[key],
            f"task_results[{key}]",
            mismatches,
            numeric_differences,
        )
    return {
        "passed": not mismatches,
        "baseline_record_count": len(old_records),
        "new_record_count": len(new_records),
        "compared_record_count": len(old_keys & new_keys),
        "exact_key_set_required": require_exact_keys,
        "max_common_numeric_abs_difference": (
            0.0 if not numeric_differences else max(numeric_differences)
        ),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def object_regression(old: Any, new: Any, path: str) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    numeric_differences: list[float] = []
    compare_value(old, new, path, mismatches, numeric_differences)
    return {
        "passed": not mismatches,
        "max_common_numeric_abs_difference": (
            0.0 if not numeric_differences else max(numeric_differences)
        ),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def recursive_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = {str(key).lower() for key in value}
        for child in value.values():
            result.update(recursive_keys(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(recursive_keys(child))
        return result
    return set()


def summary_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in SUMMARY_KEY_FIELDS)


def namespace_audit(
    old_config: dict[str, Any],
    new_config: dict[str, Any],
    old_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
    old_summary: list[dict[str, Any]],
    new_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []
    expected_namespace = {"visit_indexed_certificate"}
    config_additions = set(new_config) - set(old_config)
    if config_additions != expected_namespace:
        problems.append(
            {
                "object": "config",
                "problem": "new fields outside required namespace",
                "observed_additions": sorted(config_additions),
                "expected_additions": sorted(expected_namespace),
            }
        )

    old_by_key = unique_records(old_records, "baseline namespace records")
    new_by_key = unique_records(new_records, "new namespace records")
    for key in sorted(set(old_by_key) & set(new_by_key)):
        additions = set(new_by_key[key]) - set(old_by_key[key])
        if additions != expected_namespace:
            problems.append(
                {
                    "object": "task_result",
                    "record": key,
                    "problem": "new fields outside required namespace",
                    "observed_additions": sorted(additions),
                    "expected_additions": sorted(expected_namespace),
                }
            )

    old_summary_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in old_summary:
        key = summary_key(row)
        if key in old_summary_by_key:
            raise AssertionError(f"baseline summary contains duplicate key {key}")
        old_summary_by_key[key] = row
    new_summary_keys: set[tuple[Any, ...]] = set()
    for row in new_summary:
        key = summary_key(row)
        if key in new_summary_keys:
            raise AssertionError(f"new summary contains duplicate key {key}")
        new_summary_keys.add(key)
        old_row = old_summary_by_key.get(key)
        if old_row is None:
            problems.append(
                {"object": "summary", "record": key, "problem": "unknown summary key"}
            )
            continue
        expected = expected_namespace if str(row["route"]) in THEOREM_ROUTES else set()
        additions = set(row) - set(old_row)
        if additions != expected:
            problems.append(
                {
                    "object": "summary",
                    "record": key,
                    "problem": "new fields outside required namespace",
                    "observed_additions": sorted(additions),
                    "expected_additions": sorted(expected),
                }
            )
    return {
        "passed": not problems,
        "problem_count": len(problems),
        "problems": problems,
    }


def schema_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []
    for record in records:
        key = record_key(record)
        if "visit_indexed_certificate" not in record:
            problems.append({"record": key, "problem": "missing namespace"})
            continue
        visit = record["visit_indexed_certificate"]
        required = {
            "certificate_inputs",
            "event",
            "state_value",
            "routes",
            "optional_variance_adaptive",
            "oracle_audit",
        }
        if not required <= set(visit):
            problems.append(
                {
                    "record": key,
                    "problem": "missing namespace fields",
                    "missing": sorted(required - set(visit)),
                }
            )
            continue
        theorem_only = {
            field: value
            for field, value in visit.items()
            if field != "oracle_audit"
        }
        theorem_keys = recursive_keys(theorem_only)
        forbidden = sorted(
            key
            for key in theorem_keys
            if any(token in key for token in BANNED_NEW_KEYS)
        )
        if forbidden:
            problems.append(
                {"record": key, "problem": "oracle key in theorem object", "keys": forbidden}
            )

        event = visit["event"]
        state_counts = list(event["state_bellman"]["counts"])
        pair_counts = list(event["pair_bellman"]["counts"])
        horizon = int(record["trajectory_length"])
        n_states = int(record["n_states"])
        n_actions = int(record["n_actions"])
        if len(state_counts) != n_states or len(pair_counts) != n_states * n_actions:
            problems.append({"record": key, "problem": "count dimension mismatch"})
        if sum(state_counts) != horizon or sum(pair_counts) != horizon:
            problems.append({"record": key, "problem": "count total mismatch"})
        if len(pair_counts) == n_states * n_actions:
            state_from_pairs = [
                sum(pair_counts[state * n_actions : (state + 1) * n_actions])
                for state in range(n_states)
            ]
            if state_from_pairs != state_counts:
                problems.append({"record": key, "problem": "state/pair counts disagree"})
        if int(event["n_groups"]) != n_states + 2 * n_states * n_actions:
            problems.append({"record": key, "problem": "G != m+2d"})
        risk = event["risk_allocation"]
        if not math.isclose(
            float(risk["total_failure_probability"]),
            float(event["delta"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            problems.append({"record": key, "problem": "risk total differs from delta"})
        pair_radii = event["pair_bellman"]["radius_by_group"]
        recovery_radii = event["recovery"]["radius_by_group"]
        if pair_radii != recovery_radii:
            problems.append({"record": key, "problem": "pair/recovery radii differ"})

        allowed_statuses = {
            STATUS_SELECTIVE_HIGH_PROBABILITY,
            STATUS_NOT_CERTIFIED,
        }
        for route in THEOREM_ROUTES:
            certificate = visit["routes"].get(route)
            if certificate is None:
                problems.append({"record": key, "problem": f"missing route {route}"})
                continue
            status = certificate["status"]
            if status not in allowed_statuses:
                problems.append({"record": key, "problem": f"invalid status {status}"})
            emitted = status == STATUS_SELECTIVE_HIGH_PROBABILITY
            if emitted != bool(certificate["selective_high_probability_certified"]):
                problems.append({"record": key, "problem": f"status flag mismatch {route}"})
            if emitted and certificate["failure_reasons"]:
                problems.append({"record": key, "problem": f"emitted route has reasons {route}"})
            if not emitted and certificate["total_bound"] is not None:
                problems.append({"record": key, "problem": f"rejected route has bound {route}"})
            audit_violation = visit["oracle_audit"]["routes"][route]["violation"]
            if not emitted and audit_violation is not None:
                problems.append(
                    {"record": key, "problem": f"nonemission audit not null {route}"}
                )
        optional = visit["optional_variance_adaptive"]
        if optional["selected"] or optional["affects_mandatory_status"]:
            problems.append({"record": key, "problem": "optional route affects mandatory"})
        if not visit["oracle_audit"]["reward_bound"]["satisfied"]:
            problems.append({"record": key, "problem": "declared reward bound violated"})
    return {
        "passed": not problems,
        "records_checked": len(records),
        "problem_count": len(problems),
        "problems": problems,
    }


def route_and_length_summary(
    records: list[dict[str, Any]], route: str, length: int
) -> dict[str, Any]:
    selected = [record for record in records if int(record["trajectory_length"]) == length]
    certificates = [record["visit_indexed_certificate"]["routes"][route] for record in selected]
    emitted = [item["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY for item in certificates]
    finite = [float(item["total_bound"]) for item in certificates if item["total_bound"] is not None]
    reasons = Counter(reason for item in certificates for reason in item["failure_reasons"])
    audits = [
        record["visit_indexed_certificate"]["oracle_audit"]["routes"][route]
        for record in selected
    ]
    audited = [item for item in audits if item["violation"] is not None]
    violations = [item for item in audited if item["violation"] is True]
    return {
        "trajectory_length": length,
        "route": route,
        "records": len(selected),
        "pair_full_support_rate": float(
            np.mean(
                [
                    record["visit_indexed_certificate"]["event"]["pair_bellman"][
                        "full_support"
                    ]
                    for record in selected
                ]
            )
        ),
        "state_full_support_rate": float(
            np.mean(
                [
                    record["visit_indexed_certificate"]["event"]["state_bellman"][
                        "full_support"
                    ]
                    for record in selected
                ]
            )
        ),
        "emission_rate": float(np.mean(emitted)),
        "finite_bound_rate": float(
            np.mean([item["finite_bound_emitted"] for item in certificates])
        ),
        "improves_over_zero_initialization_rate": float(
            np.mean(
                [item["improves_over_zero_initialization"] for item in certificates]
            )
        ),
        "emitted_bound_mean": None if not finite else float(np.mean(finite)),
        "emitted_bound_median": None if not finite else float(np.median(finite)),
        "failure_reason_counts": dict(sorted(reasons.items())),
        "oracle_audited_emission_count": len(audited),
        "oracle_violation_count": len(violations),
        "oracle_violation_rate_among_emitted": (
            None if not audited else len(violations) / len(audited)
        ),
    }


def collect_violations(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for record in records:
        identity = {
            "task_index": record["task_index"],
            "spawn_key": record["spawn_key"],
            "trajectory_length": record["trajectory_length"],
            "n_states": record["n_states"],
            "n_actions": record["n_actions"],
            "pi_min": record["pi_min"],
            "beta": record["beta"],
            "mixing": record["mixing"],
            "gap_bonus": record["gap_bonus"],
        }
        audit = record["visit_indexed_certificate"]["oracle_audit"]
        for family, family_audit in audit["fixed_target_residuals"].items():
            for violation in family_audit["violations"]:
                violations.append(
                    {**identity, "kind": "residual_event", "family": family, **violation}
                )
        for route, route_audit in audit["routes"].items():
            if route_audit["violation"] is True:
                violations.append(
                    {**identity, "kind": "route_bound", "route": route, **route_audit}
                )
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("results/fixed_policy_finite_sample_certificates"),
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/FP-MART-001/codex"),
    )
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = args.baseline_dir.resolve()
    result_dir = args.result_dir.resolve()
    observed_hashes = {
        name: sha256_file(baseline_dir / name) for name in BASELINE_HASHES
    }
    if observed_hashes != BASELINE_HASHES:
        raise AssertionError("frozen baseline hashes differ")

    old_config = load_strict_json(baseline_dir / "config.json")
    old_records = load_strict_json(baseline_dir / "task_results.json")
    old_summary = load_strict_json(baseline_dir / "summary.json")
    new_config = load_strict_json(result_dir / "config.json")
    new_records = load_strict_json(result_dir / "task_results.json")
    new_summary = load_strict_json(result_dir / "summary.json")
    expected_records = (
        int(new_config["tasks_per_cell"])
        * len(new_config["trajectory_lengths"])
        * len(new_config["n_actions"])
        * len(new_config["pi_mins"])
        * len(new_config["betas"])
        * len(new_config["mixing"])
        * len(new_config["gap_bonuses"])
    )
    if len(new_records) != expected_records:
        raise AssertionError(
            f"record count {len(new_records)} differs from configured {expected_records}"
        )
    if args.mode == "formal" and len(new_records) != 480:
        raise AssertionError("formal result must contain exactly 480 records")

    records_regression = regression_report(
        old_records,
        new_records,
        require_exact_keys=args.mode == "formal",
    )
    config_regression = (
        object_regression(old_config, new_config, "config")
        if args.mode == "formal"
        else {"passed": True, "skipped": "smoke matrix intentionally changes tasks/lengths"}
    )
    summary_regression = (
        object_regression(old_summary, new_summary, "summary")
        if args.mode == "formal"
        else {"passed": True, "skipped": "smoke aggregates use two tasks"}
    )
    namespaces = namespace_audit(
        old_config,
        new_config,
        old_records,
        new_records,
        old_summary,
        new_summary,
    )
    schema = schema_audit(new_records)
    lengths = sorted({int(record["trajectory_length"]) for record in new_records})
    route_rows = [
        route_and_length_summary(new_records, route, length)
        for length in lengths
        for route in THEOREM_ROUTES
    ]
    exact_by_length = []
    for length in lengths:
        exact_row = next(
            row
            for row in route_rows
            if row["trajectory_length"] == length and row["route"] == "direct_exact"
        )
        expected = EXPECTED_EXACT_EMISSION.get(length)
        exact_by_length.append(
            {
                "trajectory_length": length,
                "observed_emission_rate": exact_row["emission_rate"],
                "preregistered_expected_rate": expected,
                "deviation": (
                    None if expected is None else exact_row["emission_rate"] - expected
                ),
                "attribution": (
                    "observed_pair_support"
                    if expected is not None
                    else "no_preregistered_comparator"
                ),
            }
        )
    violations = collect_violations(new_records)
    high_length_exact_ok = True
    if args.mode == "formal":
        for length in (4096, 16384):
            for route in ("direct_exact", "vfirst_nosplit_exact"):
                row = next(
                    item
                    for item in route_rows
                    if item["trajectory_length"] == length and item["route"] == route
                )
                high_length_exact_ok &= row["emission_rate"] == 1.0

    passed = bool(
        records_regression["passed"]
        and config_regression["passed"]
        and summary_regression["passed"]
        and namespaces["passed"]
        and schema["passed"]
        and high_length_exact_ok
    )
    regression = {
        "mode": args.mode,
        "passed": passed,
        "baseline_integrity": {
            "canonical_directory": str(baseline_dir),
            "hashes": observed_hashes,
            "record_count": len(old_records),
        },
        "new_result_directory": str(result_dir),
        "configured_record_count": expected_records,
        "actual_record_count": len(new_records),
        "complete_legacy_record_regression": records_regression,
        "config_regression": config_regression,
        "summary_regression": summary_regression,
        "namespace_audit": namespaces,
        "new_schema_audit": schema,
        "high_length_exact_emission_requirement_passed": high_length_exact_ok,
        "route_by_length": route_rows,
        "exact_emission_deviation": exact_by_length,
        "empirical_audit_violation_count": len(violations),
        "empirical_audit_violations": violations,
        "interpretation": (
            "Empirical audits are diagnostics only; selective validity follows from "
            "the proof, not observed coverage."
        ),
    }
    (result_dir / "regression.json").write_text(
        json.dumps(
            strict_json_ready(regression),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    command = subprocess.list2cmdline(
        [sys.executable, "-B", Path(__file__).name, *sys.argv[1:]]
    )
    with (result_dir / "commands.log").open("a", encoding="utf-8") as handle:
        handle.write(f"ANALYSIS: {command}\n")
    with (result_dir / "checks.log").open("a", encoding="utf-8") as handle:
        handle.write(f"ANALYZER_MODE: {args.mode}\n")
        handle.write(f"ANALYZER_PASSED: {passed}\n")
        handle.write(f"LEGACY_MISMATCH_COUNT: {records_regression['mismatch_count']}\n")
        handle.write(f"SCHEMA_PROBLEM_COUNT: {schema['problem_count']}\n")
        handle.write(f"NAMESPACE_PROBLEM_COUNT: {namespaces['problem_count']}\n")
        handle.write(f"EMPIRICAL_AUDIT_VIOLATION_COUNT: {len(violations)}\n")

    print(json.dumps(strict_json_ready(regression), ensure_ascii=False, indent=2))
    if not passed:
        raise AssertionError("visit-indexed regression or schema audit failed")
    print("PASS visit-indexed certificate analysis")


if __name__ == "__main__":
    main()
