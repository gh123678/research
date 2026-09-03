"""Analyze visit-indexed martingale certificate results (Claude route).

Loads strict JSON (rejecting NaN/Infinity constants and duplicate keys),
regresses every legacy leaf (config, task records, and the 176 summary rows)
against the frozen baseline, and summarizes the
``visit_indexed_certificate`` namespace: missing support, emission, and both
deterministic descriptive thresholds without selecting between them:

- primary ``improves_over_zero_initialization := total_bound < B``;
- secondary ``below_two_B_range := total_bound < 2B``.

Validity/emission and numerical usefulness stay separate.  Rejection reasons
are attributed to the categories pre-registered before the formal run:

- A: missing observed support;
- B: nonpositive pair kernel margin;
- C: nonpositive state kernel margin;
- D: divergence guard or nonfinite arithmetic;
- E: any other reason.

A complete namespace/schema/oracle-provenance audit walks every new record:
new values may appear only below ``visit_indexed_certificate``, certificate
inputs must carry no oracle field, the value bound must match the declared
public rule ``B = (1 + gap_bonus) / (1 - gamma)`` exactly, and the
``oracle_audit`` block must be structurally separate.  Oracle-audit
violations (per-route bound slack and per-group residual margins) are
reported with their configurations; empirical coverage is never used as
proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_fixed_policy_q_routes import summarize as legacy_summarize


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
LEGACY_SUMMARY_ROWS = 176
REQUIRED_NAMESPACE_KEYS = {
    "risk",
    "event",
    "routes",
    "variance_adaptive",
    "oracle_audit",
}
REQUIRED_EVENT_KEYS = {
    "trajectory_length",
    "n_states",
    "n_pairs",
    "n_actions_per_state",
    "n_groups",
    "delta",
    "declared_reward_bound",
    "value_bound",
    "value_bound_derivation",
    "state_counts",
    "pair_counts",
    "state_full_support",
    "pair_full_support",
    "state_radius",
    "pair_radius",
    "event_valid",
    "failure_reasons",
}
REQUIRED_ROUTE_KEYS = {
    "route",
    "matching",
    "status",
    "failure_reasons",
    "total_bound",
}
ALLOWED_ROUTE_STATUS = {
    "selective_high_probability_certified",
    "not_emitted",
}
# Key fragments forbidden anywhere in the certificate namespace outside the
# structurally separate oracle_audit block.
FORBIDDEN_CERTIFICATE_KEY_FRAGMENTS = ("true", "oracle", "occupancy", "spectral")


class DuplicateKeyError(ValueError):
    """Raised when a JSON object repeats a key."""


def load_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                raise DuplicateKeyError(f"duplicate JSON key {key!r} in {path}")
            seen[key] = value
        return seen

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate,
    )


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


def summary_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["trajectory_length"]),
        int(row["n_actions"]),
        float(row["pi_min"]),
        float(row["beta"]),
        float(row["mixing"]),
        float(row["gap_bonus"]),
        str(row["route"]),
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
        for index, old_item in enumerate(old):
            compare_legacy(
                old_item,
                new[index],
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


def regression_report(
    old: Any,
    new: Any,
    label: str,
    key_align: bool = False,
    key_fn: Any = None,
) -> dict[str, Any]:
    """Compare baseline structure against new structure, reporting mismatches."""
    mismatches: list[dict[str, Any]] = []
    numeric_differences: list[float] = []
    if key_align:
        old_by_key = {key_fn(item): item for item in old}
        new_by_key = {key_fn(item): item for item in new}
        common = sorted(old_by_key.keys() & new_by_key.keys())
        if set(old_by_key) != set(new_by_key):
            mismatches.append(
                {
                    "path": f"{label}_keys",
                    "old_only": sorted(set(old_by_key) - set(new_by_key)),
                    "new_only": sorted(set(new_by_key) - set(old_by_key)),
                }
            )
        for key in common:
            compare_legacy(
                old_by_key[key],
                new_by_key[key],
                f"{label}[{key!r}]",
                mismatches,
                numeric_differences,
            )
        old_count, new_count = len(old), len(new)
    else:
        compare_legacy(old, new, label, mismatches, numeric_differences)
        old_count, new_count = None, None
    return {
        "label": label,
        "old_count": old_count,
        "new_count": new_count,
        "mismatch_count": len(mismatches),
        "max_common_numeric_abs_difference": (
            0.0 if not numeric_differences else max(numeric_differences)
        ),
        "first_mismatches": mismatches[:20],
        "passed": not mismatches,
    }


def legacy_regression(
    old_tasks: list[dict[str, Any]], new_tasks: list[dict[str, Any]], smoke: bool
) -> dict[str, Any]:
    old_by_key = {record_key(task): task for task in old_tasks}
    new_by_key = {record_key(task): task for task in new_tasks}
    if len(new_by_key) != len(new_tasks):
        duplicate_records = len(new_tasks) - len(new_by_key)
    else:
        duplicate_records = 0
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
    if smoke and not new_by_key.keys() <= old_by_key.keys():
        mismatches.append(
            {
                "path": "task_keys",
                "error": "smoke run must only contain baseline task keys",
                "new_only_keys": sorted(
                    new_by_key.keys() - old_by_key.keys(), key=repr
                )[:20],
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
    new_only = len(new_by_key.keys() - old_by_key.keys())
    return {
        "smoke": bool(smoke),
        "compared_records": len(common),
        "baseline_records": len(old_tasks),
        "new_records": len(new_tasks),
        "duplicate_record_keys": duplicate_records,
        "old_only_keys": len(old_by_key.keys() - new_by_key.keys()),
        "new_only_keys": new_only,
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
    improves = 0
    below_two_b = 0
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
                if float(bound) < value_bound:
                    improves += 1
                if float(bound) < 2.0 * value_bound:
                    below_two_b += 1
        else:
            categories[deviation_category(reasons)] += 1
        audit = vic["oracle_audit"]["routes"][route]
        if audit["bound_violated"]:
            audit_violations += 1
    emission_rate = emitted / count if count else None

    def _rate(numer: int, denom: int) -> float | None:
        return None if denom == 0 else numer / denom

    return {
        "route": route,
        "trajectory_length": length,
        "tasks": count,
        "emission_rate": emission_rate,
        "non_emission_rate": _rate(count - emitted, count),
        "improves_over_zero_initialization_rate_among_emitted": _rate(
            improves, emitted
        ),
        "improves_over_zero_initialization_rate_overall": _rate(improves, count),
        "below_two_B_range_rate_among_emitted": _rate(below_two_b, emitted),
        "below_two_B_range_rate_overall": _rate(below_two_b, count),
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
        "audit_violation_rate": _rate(audit_violations, count),
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


def collect_per_group_violations(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for task in tasks:
        audit = task["visit_indexed_certificate"]["oracle_audit"]
        per_group = audit.get("per_group_residual_audit", {})
        for family, entry in per_group.items():
            if not isinstance(entry, dict):
                continue
            if int(entry.get("violation_count", 0)) > 0:
                violations.append(
                    {
                        "task_index": task["task_index"],
                        "spawn_key": task["spawn_key"],
                        "trajectory_length": task["trajectory_length"],
                        "mixing": task["mixing"],
                        "gap_bonus": task["gap_bonus"],
                        "family": family,
                        "violation_count": entry["violation_count"],
                        "groups_visited": entry["groups_visited"],
                        "worst_margin": entry["worst_margin"],
                        "worst_group": entry["worst_group"],
                    }
                )
    return violations


def _walk_forbidden_keys(value: Any, path: str, violations: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_CERTIFICATE_KEY_FRAGMENTS):
                violations.append(f"{path}.{key}")
            _walk_forbidden_keys(item, f"{path}.{key}", violations)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_forbidden_keys(item, f"{path}[{index}]", violations)


def audit_record_schema(
    task: dict[str, Any], gamma: float
) -> list[dict[str, Any]]:
    """Complete namespace/schema/oracle-provenance audit for one record."""
    problems: list[dict[str, Any]] = []
    key = str(record_key(task))
    if "visit_indexed_certificate" not in task:
        return [
            {
                "record": key,
                "problem": "missing visit_indexed_certificate namespace",
            }
        ]
    vic = task["visit_indexed_certificate"]
    if not isinstance(vic, dict):
        return [{"record": key, "problem": "namespace is not an object"}]
    missing_keys = REQUIRED_NAMESPACE_KEYS - set(vic)
    if missing_keys:
        problems.append(
            {"record": key, "problem": "missing namespace keys", "keys": sorted(missing_keys)}
        )
    event = vic.get("event", {})
    if isinstance(event, dict):
        missing_event = REQUIRED_EVENT_KEYS - set(event)
        if missing_event:
            problems.append(
                {
                    "record": key,
                    "problem": "missing event keys",
                    "keys": sorted(missing_event),
                }
            )
        else:
            declared = float(event["declared_reward_bound"])
            expected_bound = declared / (1.0 - gamma)
            if not math.isclose(
                float(event["value_bound"]), expected_bound, rel_tol=1e-12, abs_tol=1e-12
            ):
                problems.append(
                    {
                        "record": key,
                        "problem": "value_bound does not follow the declared rule",
                        "declared_reward_bound": declared,
                        "value_bound": event["value_bound"],
                        "expected": expected_bound,
                    }
                )
            expected_declared = 1.0 + float(task["gap_bonus"])
            if not math.isclose(declared, expected_declared, rel_tol=1e-12, abs_tol=1e-12):
                problems.append(
                    {
                        "record": key,
                        "problem": "declared_reward_bound is not the public rule 1+gap_bonus",
                        "declared_reward_bound": declared,
                        "expected": expected_declared,
                    }
                )
            length = int(event["trajectory_length"])
            if length != int(task["trajectory_length"]):
                problems.append(
                    {"record": key, "problem": "event trajectory_length mismatch"}
                )
            state_counts = [int(c) for c in event["state_counts"]]
            pair_counts = [int(c) for c in event["pair_counts"]]
            if sum(state_counts) != length or sum(pair_counts) != length:
                problems.append(
                    {"record": key, "problem": "event counts do not sum to the horizon"}
                )
            actions = int(event["n_actions_per_state"])
            for state in range(int(event["n_states"])):
                aggregated = sum(
                    pair_counts[state * actions : (state + 1) * actions]
                )
                if aggregated != state_counts[state]:
                    problems.append(
                        {
                            "record": key,
                            "problem": "state counts disagree with aggregated pair counts",
                            "state": state,
                        }
                    )
                    break
    routes = vic.get("routes", {})
    if isinstance(routes, dict):
        for route_name in THEOREM_ROUTES:
            route_cert = routes.get(route_name)
            if not isinstance(route_cert, dict):
                problems.append(
                    {"record": key, "problem": f"missing route {route_name}"}
                )
                continue
            missing_route = REQUIRED_ROUTE_KEYS - set(route_cert)
            if missing_route:
                problems.append(
                    {
                        "record": key,
                        "problem": f"missing route keys for {route_name}",
                        "keys": sorted(missing_route),
                    }
                )
            if str(route_cert.get("status")) not in ALLOWED_ROUTE_STATUS:
                problems.append(
                    {
                        "record": key,
                        "problem": f"unexpected route status for {route_name}",
                        "status": route_cert.get("status"),
                    }
                )
    certificate_part = {
        key_: value
        for key_, value in vic.items()
        if key_ != "oracle_audit"
    }
    forbidden: list[str] = []
    _walk_forbidden_keys(certificate_part, "visit_indexed_certificate", forbidden)
    if forbidden:
        problems.append(
            {
                "record": key,
                "problem": "oracle-flavoured key outside oracle_audit",
                "keys": forbidden[:20],
            }
        )
    audit = vic.get("oracle_audit")
    if not isinstance(audit, dict):
        problems.append({"record": key, "problem": "oracle_audit is not an object"})
    return problems


def build_legacy_summary_rows(
    new_tasks: list[dict[str, Any]], route_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Recompute the 176 legacy summary rows and add per-row new metrics.

    Every legacy leaf is preserved exactly; new values appear only below the
    ``visit_indexed_certificate`` key of rows for the four theorem routes.
    """
    rows = legacy_summarize(new_tasks)
    new_by_key = {
        (row["trajectory_length"], row["route"]): row for row in route_rows
    }
    for row in rows:
        namespace = new_by_key.get((row["trajectory_length"], row["route"]))
        if namespace is not None:
            row["visit_indexed_certificate"] = {
                "emission_rate": namespace["emission_rate"],
                "non_emission_rate": namespace["non_emission_rate"],
                "improves_over_zero_initialization_rate_among_emitted": namespace[
                    "improves_over_zero_initialization_rate_among_emitted"
                ],
                "improves_over_zero_initialization_rate_overall": namespace[
                    "improves_over_zero_initialization_rate_overall"
                ],
                "below_two_B_range_rate_among_emitted": namespace[
                    "below_two_B_range_rate_among_emitted"
                ],
                "below_two_B_range_rate_overall": namespace[
                    "below_two_B_range_rate_overall"
                ],
                "audit_violation_count": namespace["audit_violation_count"],
            }
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke = bool(args.smoke)
    baseline_dir = args.baseline_dir.resolve()
    new_dir = args.new_dir.resolve()
    hashes = verify_baseline_hashes(baseline_dir)
    if not hashes["all_match"]:
        raise AssertionError(f"frozen baseline hash mismatch: {hashes}")
    old_config = load_strict_json(baseline_dir / "config.json")
    old_tasks = load_strict_json(baseline_dir / "task_results.json")
    old_summary = load_strict_json(baseline_dir / "summary.json")
    if len(old_tasks) != FORMAL_RECORD_COUNT:
        raise AssertionError("baseline record count mismatch")
    if len(old_summary) != LEGACY_SUMMARY_ROWS:
        raise AssertionError("baseline summary row count mismatch")
    new_config = load_strict_json(new_dir / "config.json")
    new_tasks = load_strict_json(new_dir / "task_results.json")
    if not smoke and len(new_tasks) != FORMAL_RECORD_COUNT:
        raise AssertionError(
            f"formal run must have {FORMAL_RECORD_COUNT} records, "
            f"found {len(new_tasks)}"
        )

    gamma = float(old_config["gamma"])

    config_regression = regression_report(old_config, new_config, "config")
    if smoke:
        # The smoke matrix legitimately requests fewer leading task seeds and
        # a shorter trajectory grid; every other config leaf must still match
        # the frozen baseline.
        SMOKE_CONFIG_EXCEPTIONS = {"config.tasks_per_cell", "config.trajectory_lengths"}
        tasks_leaf = next(
            (
                mismatch
                for mismatch in config_regression["first_mismatches"]
                if mismatch["path"] == "config.tasks_per_cell"
            ),
            None,
        )
        smoke_mismatches = [
            mismatch
            for mismatch in config_regression["first_mismatches"]
            if mismatch["path"] not in SMOKE_CONFIG_EXCEPTIONS
        ]
        config_regression = {
            **config_regression,
            "smoke_config_exceptions": sorted(SMOKE_CONFIG_EXCEPTIONS),
            "smoke_tasks_per_cell_exception": tasks_leaf is not None,
            "mismatch_count": len(smoke_mismatches),
            "first_mismatches": smoke_mismatches,
            "passed": not smoke_mismatches,
        }

    regression = legacy_regression(old_tasks, new_tasks, smoke)

    schema_problems: list[dict[str, Any]] = []
    for task in new_tasks:
        schema_problems.extend(audit_record_schema(task, gamma))
        if len(schema_problems) > 50:
            break

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
    audit_violations = collect_audit_violations(new_tasks)
    per_group_violations = collect_per_group_violations(new_tasks)
    summary_rows = build_legacy_summary_rows(new_tasks, route_rows)
    if smoke:
        summary_regression = {
            "label": "summary",
            "skipped": True,
            "reason": "the 176-row summary regression is a formal-mode gate; "
            "the smoke matrix intentionally uses fewer leading task seeds",
            "passed": True,
        }
    else:
        summary_regression = regression_report(
            old_summary,
            summary_rows,
            "summary",
            key_align=True,
            key_fn=summary_row_key,
        )

    namespace_summary = {
        "mode": "smoke" if smoke else "formal",
        "records": len(new_tasks),
        "baseline_hash_check": hashes,
        "legacy_config_regression": config_regression,
        "legacy_task_record_regression": regression,
        "legacy_summary_regression": summary_regression,
        "schema_oracle_provenance_audit": {
            "problem_count": len(schema_problems),
            "first_problems": schema_problems[:20],
            "passed": not schema_problems,
            "rules": [
                "new values only below visit_indexed_certificate",
                "no oracle-flavoured keys outside oracle_audit",
                "value_bound derived from the declared public rule",
                "original integer counts summing to the horizon",
                "state counts consistent with aggregated pair counts",
            ],
        },
        "support_by_length": [
            support_summary(new_tasks, length) for length in lengths
        ],
        "theorem_routes_by_length": route_rows,
        "variance_adaptive_status_counts": dict(sorted(adaptive_statuses.items())),
        "audit_violations": audit_violations,
        "per_group_residual_violations": per_group_violations,
        "nontriviality_definitions": {
            "primary_improves_over_zero_initialization": "total_bound < B",
            "secondary_below_two_B_range": "total_bound < 2B",
            "selection": "none; both deterministic descriptive thresholds "
            "are reported without selecting between them after seeing results",
        },
        "audit_note": (
            "empirical coverage is reported for audit only and is not a "
            "substitute for the theorem; selective claim is "
            "P(Emit and bound violated) <= delta"
        ),
    }
    summary = {
        "legacy_summary": summary_rows,
        "visit_indexed_certificate": namespace_summary,
    }
    (new_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (new_dir / "regression.json").write_text(
        json.dumps(
            {
                "baseline_hash_check": hashes,
                "legacy_config_regression": config_regression,
                "legacy_task_record_regression": regression,
                "legacy_summary_regression": summary_regression,
                "schema_oracle_provenance_audit": namespace_summary[
                    "schema_oracle_provenance_audit"
                ],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(regression, ensure_ascii=False, indent=2))
    checks = {
        "baseline_hash_check": hashes["all_match"],
        "legacy_config_regression": config_regression["passed"],
        "legacy_task_record_regression": regression["passed"],
        "legacy_summary_regression": summary_regression["passed"],
        "schema_oracle_provenance_audit": not schema_problems,
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        failed = [name for name, ok in checks.items() if not ok]
        raise AssertionError(f"analysis checks failed: {failed}")
    print("PASS visit-indexed certificate analysis")


if __name__ == "__main__":
    main()
