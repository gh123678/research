"""Analyze visit-indexed martingale certificate results (Claude route).

Loads strict JSON, regresses all legacy leaves against the frozen baseline,
and summarizes the ``visit_indexed_certificate`` namespace: missing support,
emission, and bound-nontriviality as separate rates; rejection reasons by
length and route; oracle-audit violations with their configurations; and the
expected-rate deviation categories pre-registered before the formal run:

- A: missing observed support;
- B: nonpositive pair kernel margin;
- C: nonpositive state kernel margin;
- D: divergence guard or nonfinite arithmetic;
- E: any other reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_EXACT_EMISSION = {256: 0.025, 1024: 0.85, 4096: 1.0, 16384: 1.0}
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
FORMAL_RECORD_COUNT = 480


def load_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def record_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(task["task_index"]),
        tuple(task["spawn_key"]),
        int(task["trajectory_length"]),
        int(task["n_states"]),
        int(task["n_actions"]),
        float(task["pi_min"]),
        float(task["beta"]),
        float(task["mixing"]),
        float(task["gap_bonus"]),
    )


def verify_baseline_hashes(baseline_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"files": {}, "all_match": True}
    for name, expected in BASELINE_HASHES.items():
        digest = hashlib.sha256((baseline_dir / name).read_bytes()).hexdigest()
        result["files"][name] = {"sha256": digest, "match": digest == expected}
        result["all_match"] = result["all_match"] and digest == expected
    return result


def compare_legacy(
    old: Any,
    new: Any,
    path: str,
    mismatches: list[dict[str, Any]],
    numeric_differences: list[float],
) -> None:
    """Walk the baseline (old) structure; every legacy leaf must match in new."""
    if isinstance(old, dict):
        if not isinstance(new, dict):
            mismatches.append({"path": path, "old": old, "new": new})
            return
        for key, value in old.items():
            if key not in new:
                mismatches.append(
                    {"path": f"{path}.{key}", "old": value, "new": "<missing>"}
                )
            else:
                compare_legacy(
                    value, new[key], f"{path}.{key}", mismatches, numeric_differences
                )
        return
    if isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            mismatches.append({"path": path, "old": old, "new": new})
            return
        for index, (old_item, new_item) in enumerate(zip(old, new)):
            compare_legacy(
                old_item,
                new_item,
                f"{path}[{index}]",
                mismatches,
                numeric_differences,
            )
        return
    if (
        isinstance(old, (int, float))
        and not isinstance(old, bool)
        and isinstance(new, (int, float))
        and not isinstance(new, bool)
    ):
        difference = abs(float(old) - float(new))
        numeric_differences.append(difference)
        if not math.isclose(float(old), float(new), rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append({"path": path, "old": old, "new": new})
        return
    if old != new:
        mismatches.append({"path": path, "old": old, "new": new})


def legacy_regression(
    old_tasks: list[dict[str, Any]], new_tasks: list[dict[str, Any]], smoke: bool
) -> dict[str, Any]:
    old_by_key = {record_key(task): task for task in old_tasks}
    new_by_key = {record_key(task): task for task in new_tasks}
    mismatches: list[dict[str, Any]] = []
    numeric_differences: list[float] = []
    common = old_by_key.keys() & new_by_key.keys()
    if not smoke and old_by_key.keys() != new_by_key.keys():
        mismatches.append(
            {
                "path": "task_keys",
                "old_only_count": len(old_by_key.keys() - new_by_key.keys()),
                "new_only_count": len(new_by_key.keys() - old_by_key.keys()),
            }
        )
    if not smoke and not new_by_key.keys() <= old_by_key.keys():
        mismatches.append(
            {
                "path": "task_keys",
                "new_only_count": len(new_by_key.keys() - old_by_key.keys()),
            }
        )
    if smoke and not common:
        mismatches.append(
            {"path": "task_keys", "error": "smoke run shares no aligned record"}
        )
    for key in sorted(common):
        compare_legacy(
            old_by_key[key],
            new_by_key[key],
            str(key),
            mismatches,
            numeric_differences,
        )
    return {
        "smoke": bool(smoke),
        "compared_records": len(common),
        "baseline_records": len(old_tasks),
        "new_records": len(new_tasks),
        "old_only_keys": len(old_by_key.keys() - new_by_key.keys()),
        "new_only_keys": len(new_by_key.keys() - old_by_key.keys()),
        "mismatch_count": len(mismatches),
        "max_common_numeric_abs_difference": (
            0.0 if not numeric_differences else max(numeric_differences)
        ),
        "first_mismatches": mismatches[:20],
        "passed": not mismatches,
    }


def deviation_category(reasons: list[str]) -> str:
    if "missing_support" in reasons:
        return "A_missing_support"
    if "pair_kernel_margin_nonpositive" in reasons:
        return "B_pair_kernel_margin"
    if "state_kernel_margin_nonpositive" in reasons:
        return "C_state_kernel_margin"
    if "divergence_guard_triggered" in reasons or "numerical_nonfinite" in reasons:
        return "D_divergence_or_nonfinite"
    return "E_other"


def route_summary(
    tasks: list[dict[str, Any]], route: str, length: int
) -> dict[str, Any]:
    selected = [
        task for task in tasks if int(task["trajectory_length"]) == length
    ]
    count = len(selected)
    emitted = 0
    nontrivial = 0
    emitted_bounds: list[float] = []
    failures: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    audit_violations = 0
    statuses: Counter[str] = Counter()
    for task in selected:
        vic = task["visit_indexed_certificate"]
        route_cert = vic["routes"][route]
        statuses[str(route_cert["status"])] += 1
        reasons = list(route_cert["failure_reasons"])
        failures.update(reasons)
        bound = route_cert["total_bound"]
        value_bound = float(vic["event"]["value_bound"])
        if route_cert["status"] == "selective_high_probability_certified":
            emitted += 1
            if bound is not None:
                emitted_bounds.append(float(bound))
                if float(bound) < 2.0 * value_bound:
                    nontrivial += 1
        else:
            categories[deviation_category(reasons)] += 1
        audit = vic["oracle_audit"]["routes"][route]
        if audit["bound_violated"]:
            audit_violations += 1
    emission_rate = emitted / count if count else None
    return {
        "route": route,
        "trajectory_length": length,
        "tasks": count,
        "emission_rate": emission_rate,
        "non_emission_rate": None if count == 0 else 1.0 - emitted / count,
        "nontrivial_rate_among_emitted": (
            None if emitted == 0 else nontrivial / emitted
        ),
        "nontrivial_rate_overall": None if count == 0 else nontrivial / count,
        "emitted_bound_mean": (
            None
            if not emitted_bounds
            else sum(emitted_bounds) / len(emitted_bounds)
        ),
        "status_counts": dict(sorted(statuses.items())),
        "failure_reason_rates": {
            reason: failures[reason] / count for reason in sorted(failures)
        },
        "non_emission_categories": {
            category: categories[category] for category in sorted(categories)
        },
        "audit_violation_count": audit_violations,
        "audit_violation_rate": None if count == 0 else audit_violations / count,
        "expected_exact_emission": EXPECTED_EXACT_EMISSION.get(length),
        "deviation_from_expected": (
            None
            if emission_rate is None
            else emission_rate - EXPECTED_EXACT_EMISSION[length]
        ),
    }


def support_summary(tasks: list[dict[str, Any]], length: int) -> dict[str, Any]:
    selected = [
        task for task in tasks if int(task["trajectory_length"]) == length
    ]
    pair_full = sum(
        1
        for task in selected
        if task["visit_indexed_certificate"]["event"]["pair_full_support"]
    )
    state_full = sum(
        1
        for task in selected
        if task["visit_indexed_certificate"]["event"]["state_full_support"]
    )
    count = len(selected)
    return {
        "trajectory_length": length,
        "tasks": count,
        "observed_full_pair_support_rate": pair_full / count if count else None,
        "observed_full_state_support_rate": state_full / count if count else None,
    }


def collect_audit_violations(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for task in tasks:
        audit = task["visit_indexed_certificate"]["oracle_audit"]
        for route, entry in audit["routes"].items():
            if entry["bound_violated"]:
                violations.append(
                    {
                        "task_index": task["task_index"],
                        "spawn_key": task["spawn_key"],
                        "trajectory_length": task["trajectory_length"],
                        "mixing": task["mixing"],
                        "gap_bonus": task["gap_bonus"],
                        "route": route,
                        "total_bound": entry["total_bound"],
                        "true_sup_error": entry["true_sup_error"],
                        "bound_slack": entry["bound_slack"],
                    }
                )
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = args.baseline_dir.resolve()
    new_dir = args.new_dir.resolve()
    hashes = verify_baseline_hashes(baseline_dir)
    if not hashes["all_match"]:
        raise AssertionError(f"frozen baseline hash mismatch: {hashes}")
    old_tasks = load_strict_json(baseline_dir / "task_results.json")
    if len(old_tasks) != FORMAL_RECORD_COUNT:
        raise AssertionError("baseline record count mismatch")
    new_tasks = load_strict_json(new_dir / "task_results.json")
    if not args.smoke and len(new_tasks) != FORMAL_RECORD_COUNT:
        raise AssertionError(
            f"formal run must have {FORMAL_RECORD_COUNT} records, "
            f"found {len(new_tasks)}"
        )

    regression = legacy_regression(old_tasks, new_tasks, args.smoke)
    (new_dir / "regression.json").write_text(
        json.dumps(regression, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    lengths = sorted({int(task["trajectory_length"]) for task in new_tasks})
    route_rows = [
        route_summary(new_tasks, route, length)
        for route in THEOREM_ROUTES
        for length in lengths
    ]
    adaptive_statuses: Counter[str] = Counter()
    for task in new_tasks:
        adaptive_statuses[
            str(task["visit_indexed_certificate"]["variance_adaptive"]["status"])
        ] += 1
    summary = {
        "mode": "smoke" if args.smoke else "formal",
        "records": len(new_tasks),
        "baseline_hash_check": hashes,
        "legacy_regression": regression,
        "support_by_length": [
            support_summary(new_tasks, length) for length in lengths
        ],
        "theorem_routes_by_length": route_rows,
        "variance_adaptive_status_counts": dict(sorted(adaptive_statuses.items())),
        "audit_violations": collect_audit_violations(new_tasks),
        "audit_note": (
            "empirical coverage is reported for audit only and is not a "
            "substitute for the theorem; selective claim is "
            "P(Emit and bound violated) <= delta"
        ),
    }
    (new_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(regression, ensure_ascii=False, indent=2))
    if not regression["passed"]:
        raise AssertionError("legacy regression failed")
    print("PASS visit-indexed certificate analysis")


if __name__ == "__main__":
    main()
