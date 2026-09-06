"""Strict regression and audit analysis for time-uniform mixture certificates."""

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

from analyze_fixed_policy_finite_sample_certificates import compare_value, record_key
from fixed_policy_finite_sample_certificate import strict_json_ready
from time_uniform_mixture_certificate import (
    STATUS_NOT_CERTIFIED,
    STATUS_SELECTIVE_HIGH_PROBABILITY,
    mixture_radius,
    stitch_radius,
)
from visit_indexed_martingale_certificate import simultaneous_hoeffding_radius


BASELINE_DIR = Path(
    r"C:\Users\Admin\Desktop\research\icrl_softmax\results\FP-MART-001\codex"
)
BASELINE_HASHES = {
    "config.json": "bcc377b422711302163b976d5d5ece389e1e3ee8540d37817a8719bf9ec3bf8a",
    "task_results.json": "929e2f65689af850b65f000ee6675a8a87b3506c2e2a9600c07d28cf183c6d52",
    "summary.json": "fa13619755b627b9dcff281dc5e4bc9b4d0262b9a6592f2982012b53df2b34ed",
}
THEOREM_ROUTES = (
    "direct_exact",
    "direct_softmax",
    "vfirst_nosplit_exact",
    "vfirst_nosplit_softmax",
)
FROZEN_LENGTHS = (256, 1024, 4096, 16384)
EXPECTED_NAMESPACE = {"time_uniform_certificate"}
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


def load_strict_json(path: Path) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=reject_constant,
    )


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
    config_additions = set(new_config) - set(old_config)
    if config_additions != EXPECTED_NAMESPACE:
        problems.append(
            {
                "object": "config",
                "problem": "new fields outside required namespace",
                "observed_additions": sorted(config_additions),
                "expected_additions": sorted(EXPECTED_NAMESPACE),
            }
        )

    old_by_key = unique_records(old_records, "baseline namespace records")
    new_by_key = unique_records(new_records, "new namespace records")
    for key in sorted(set(old_by_key) & set(new_by_key)):
        additions = set(new_by_key[key]) - set(old_by_key[key])
        if additions != EXPECTED_NAMESPACE:
            problems.append(
                {
                    "object": "task_result",
                    "record": key,
                    "problem": "new fields outside required namespace",
                    "observed_additions": sorted(additions),
                    "expected_additions": sorted(EXPECTED_NAMESPACE),
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
        expected = EXPECTED_NAMESPACE if str(row["route"]) in THEOREM_ROUTES else set()
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
        if "time_uniform_certificate" not in record:
            problems.append({"record": key, "problem": "missing namespace"})
            continue
        uniform = record["time_uniform_certificate"]
        required = {
            "certificate_inputs",
            "event",
            "state_value",
            "routes",
            "oracle_audit",
        }
        if not required <= set(uniform):
            problems.append(
                {
                    "record": key,
                    "problem": "missing namespace fields",
                    "missing": sorted(required - set(uniform)),
                }
            )
            continue
        theorem_only = {
            field: value for field, value in uniform.items() if field != "oracle_audit"
        }
        forbidden = sorted(
            key
            for key in recursive_keys(theorem_only)
            if any(token in key for token in BANNED_NEW_KEYS)
        )
        if forbidden:
            problems.append(
                {"record": key, "problem": "oracle key in theorem object", "keys": forbidden}
            )

        event = uniform["event"]
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
        n_groups = n_states + 2 * n_states * n_actions
        if int(event["n_groups"]) != n_groups:
            problems.append({"record": key, "problem": "G != m+2d"})
        mixture = event["mixture"]
        if int(mixture["grid_size"]) != 15:
            problems.append({"record": key, "problem": "mixture grid size != 15"})
        if list(mixture["count_grid"]) != [2**j for j in range(15)]:
            problems.append({"record": key, "problem": "mixture count grid changed"})
        weights = [float(value) for value in mixture["weights"]]
        raw = [(j + 1) ** -2 for j in range(15)]
        norm = math.fsum(raw)
        expected_weights = [value / norm for value in raw]
        if not all(
            math.isclose(observed, expected, rel_tol=1e-15, abs_tol=1e-15)
            for observed, expected in zip(weights, expected_weights, strict=True)
        ):
            problems.append({"record": key, "problem": "mixture weights changed"})
        expected_log_terms = [
            math.log(2.0 * n_groups / (float(event["delta"]) * w)) for w in expected_weights
        ]
        if not all(
            math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
            for observed, expected in zip(
                mixture["log_terms"], expected_log_terms, strict=True
            )
        ):
            problems.append({"record": key, "problem": "mixture log terms changed"})
        risk = event["risk_allocation"]
        if not math.isclose(
            float(risk["per_group_failure_probability"]),
            float(event["delta"]) / n_groups,
            rel_tol=1e-15,
            abs_tol=1e-15,
        ):
            problems.append({"record": key, "problem": "per-group risk differs from delta/G"})
        if not math.isclose(
            float(risk["total_failure_probability"]),
            float(event["delta"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            problems.append({"record": key, "problem": "risk total differs from delta"})
        if risk["route_level_resplit"] is not False:
            problems.append({"record": key, "problem": "route-level risk resplit detected"})
        pair_radii = event["pair_bellman"]["mixture_radius_by_group"]
        recovery_radii = event["recovery"]["mixture_radius_by_group"]
        if pair_radii != recovery_radii:
            problems.append({"record": key, "problem": "pair/recovery mixture radii differ"})

        allowed_statuses = {
            STATUS_SELECTIVE_HIGH_PROBABILITY,
            STATUS_NOT_CERTIFIED,
        }
        for route in THEOREM_ROUTES:
            certificate = uniform["routes"].get(route)
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
            audit_violation = uniform["oracle_audit"]["routes"][route]["violation"]
            if not emitted and audit_violation is not None:
                problems.append(
                    {"record": key, "problem": f"nonemission audit not null {route}"}
                )
        if not uniform["oracle_audit"]["reward_bound"]["satisfied"]:
            problems.append({"record": key, "problem": "declared reward bound violated"})
    return {
        "passed": not problems,
        "records_checked": len(records),
        "problem_count": len(problems),
        "problems": problems,
    }


def recompute_radius_checks(
    records: list[dict[str, Any]], *, gamma: float, delta: float
) -> dict[str, Any]:
    """Recompute every recorded count radius from the public module."""
    problems: list[dict[str, Any]] = []
    max_ratio_vs_old = 0.0
    strict_dominance_seen = False
    checked = 0
    for record in records:
        key = record_key(record)
        horizon = int(record["trajectory_length"])
        reward_bound = 1.0 + float(record["gap_bonus"])
        n_groups = int(record["n_states"]) + 2 * int(record["n_states"]) * int(
            record["n_actions"]
        )
        event = record["time_uniform_certificate"]["event"]
        for family_name in ("state_bellman", "pair_bellman", "recovery"):
            family = event[family_name]
            for group, (count, recorded, stitch_recorded, old_recorded) in enumerate(
                zip(
                    family["counts"],
                    family["mixture_radius_by_group"],
                    family["stitch_radius_by_group"],
                    family["old_radius_by_group"],
                    strict=True,
                )
            ):
                if int(count) == 0:
                    if recorded is not None or stitch_recorded is not None or old_recorded is not None:
                        problems.append(
                            {"record": key, "problem": f"unvisited group has radius in {family_name}[{group}]"}
                        )
                    continue
                checked += 1
                recomputed = mixture_radius(
                    int(count),
                    reward_bound=reward_bound,
                    gamma=gamma,
                    n_groups=n_groups,
                    delta=delta,
                )
                if not math.isclose(float(recorded), recomputed, rel_tol=1e-12, abs_tol=1e-12):
                    problems.append(
                        {"record": key, "problem": f"mixture radius mismatch {family_name}[{group}]"}
                    )
                recomputed_stitch = stitch_radius(
                    int(count),
                    reward_bound=reward_bound,
                    gamma=gamma,
                    n_groups=n_groups,
                    delta=delta,
                )
                if not math.isclose(
                    float(stitch_recorded), recomputed_stitch, rel_tol=1e-12, abs_tol=1e-12
                ):
                    problems.append(
                        {"record": key, "problem": f"stitch radius mismatch {family_name}[{group}]"}
                    )
                recomputed_old = simultaneous_hoeffding_radius(
                    int(count),
                    reward_bound=reward_bound,
                    gamma=gamma,
                    n_groups=n_groups,
                    horizon=horizon,
                    delta=delta,
                )
                if not math.isclose(
                    float(old_recorded), recomputed_old, rel_tol=1e-12, abs_tol=1e-12
                ):
                    problems.append(
                        {"record": key, "problem": f"old radius mismatch {family_name}[{group}]"}
                    )
                if not recomputed <= recomputed_stitch + 1e-15:
                    problems.append(
                        {"record": key, "problem": f"mixture exceeds stitch {family_name}[{group}]"}
                    )
                if not recomputed <= recomputed_old * (1.0 + 1e-12):
                    problems.append(
                        {"record": key, "problem": f"mixture exceeds old {family_name}[{group}]"}
                    )
                ratio = recomputed / recomputed_old
                max_ratio_vs_old = max(max_ratio_vs_old, ratio)
                if ratio < 1.0 - 1e-12:
                    strict_dominance_seen = True
        legacy_event = record["visit_indexed_certificate"]["event"]
        for family_name in ("state_bellman", "pair_bellman", "recovery"):
            if (
                event[family_name]["old_radius_by_group"]
                != legacy_event[family_name]["radius_by_group"]
            ):
                problems.append(
                    {"record": key, "problem": f"old radius drift vs legacy in {family_name}"}
                )
    return {
        "passed": not problems and strict_dominance_seen,
        "radii_checked": checked,
        "max_radius_ratio_vs_old": max_ratio_vs_old,
        "strict_dominance_seen": strict_dominance_seen,
        "problem_count": len(problems),
        "problems": problems,
    }


def exhaustive_count_radius_audit() -> dict[str, Any]:
    """Frozen per-record dominance and global n_max audit at G = 54, delta = 0.05."""
    n_groups, delta, gamma = 54, 0.05, 0.7
    reward = 1.0 - gamma  # B = 1; both radii scale linearly in B.
    per_length_max: dict[int, float] = {}
    strict_seen = False
    monotonic = True
    previous = math.inf
    for length in FROZEN_LENGTHS:
        log_term = math.log(2.0 * n_groups * length / delta)
        max_ratio = 0.0
        for count in range(1, length + 1):
            radius = mixture_radius(
                count, reward_bound=reward, gamma=gamma, n_groups=n_groups, delta=delta
            )
            old = reward / (1.0 - gamma) * math.sqrt(2.0 * log_term / count)
            ratio = radius / old
            if ratio > 1.0 + 1e-12:
                return {
                    "passed": False,
                    "failure": f"r_mix({count}) > r_old({count}; {length})",
                }
            max_ratio = max(max_ratio, ratio)
            if ratio < 1.0 - 1e-12:
                strict_seen = True
            if length == FROZEN_LENGTHS[-1]:
                if radius > previous * (1.0 + 1e-12):
                    monotonic = False
                previous = radius
        per_length_max[length] = max_ratio
    return {
        "passed": strict_seen and monotonic,
        "per_length_max_ratio_vs_old": {str(k): v for k, v in per_length_max.items()},
        "strict_dominance_seen": strict_seen,
        "monotone_nonincreasing_through_16384": monotonic,
        "global_n_max_audit_implied": (
            "r_old(k; n) increases in n, so the per-record checks imply the "
            "global n_max = 16384 design audit"
        ),
    }


def emission_and_bound_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    problems: list[dict[str, Any]] = []
    compared = 0
    total_reductions: list[float] = []
    for record in records:
        key = record_key(record)
        uniform = record["time_uniform_certificate"]["routes"]
        legacy = record["visit_indexed_certificate"]["routes"]
        for route in THEOREM_ROUTES:
            new_route = uniform[route]
            old_route = legacy[route]
            new_emitted = new_route["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY
            old_emitted = old_route["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY
            if new_emitted != old_emitted:
                problems.append(
                    {"record": key, "problem": f"emission decision changed on {route}"}
                )
                continue
            if new_emitted:
                compared += 1
                new_total = float(new_route["total_bound"])
                old_total = float(old_route["total_bound"])
                if not math.isclose(
                    float(new_route["old_total_bound"]), old_total,
                    rel_tol=1e-12, abs_tol=1e-12,
                ):
                    problems.append(
                        {"record": key, "problem": f"old total mismatch on {route}"}
                    )
                if new_total > old_total + 1e-12:
                    problems.append(
                        {"record": key, "problem": f"emitted bound increased on {route}"}
                    )
                total_reductions.append(old_total - new_total)
    return {
        "passed": not problems,
        "emitted_pairs_compared": compared,
        "min_total_bound_reduction": min(total_reductions) if total_reductions else None,
        "mean_total_bound_reduction": (
            float(np.mean(total_reductions)) if total_reductions else None
        ),
        "problem_count": len(problems),
        "problems": problems,
    }


def route_and_length_summary(
    records: list[dict[str, Any]], route: str, length: int
) -> dict[str, Any]:
    selected = [record for record in records if int(record["trajectory_length"]) == length]
    certificates = [
        record["time_uniform_certificate"]["routes"][route] for record in selected
    ]
    legacy_certificates = [
        record["visit_indexed_certificate"]["routes"][route] for record in selected
    ]
    emitted = [item["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY for item in certificates]
    finite = [
        float(item["total_bound"]) for item in certificates if item["total_bound"] is not None
    ]
    reductions = [
        float(item["total_bound_reduction_vs_old"])
        for item in certificates
        if item["total_bound_reduction_vs_old"] is not None
    ]
    radius_ratios = [
        float(item["radius_ratio_vs_old"])
        for item in certificates
        if item["radius_ratio_vs_old"] is not None
    ]
    reasons = Counter(reason for item in certificates for reason in item["failure_reasons"])
    audits = [
        record["time_uniform_certificate"]["oracle_audit"]["routes"][route]
        for record in selected
    ]
    audited = [item for item in audits if item["violation"] is not None]
    violations = [item for item in audited if item["violation"] is True]
    legacy_emitted = [
        item["status"] == STATUS_SELECTIVE_HIGH_PROBABILITY for item in legacy_certificates
    ]
    return {
        "trajectory_length": length,
        "route": route,
        "records": len(selected),
        "pair_full_support_rate": float(
            np.mean(
                [
                    record["time_uniform_certificate"]["event"]["pair_bellman"][
                        "full_support"
                    ]
                    for record in selected
                ]
            )
        ),
        "state_full_support_rate": float(
            np.mean(
                [
                    record["time_uniform_certificate"]["event"]["state_bellman"][
                        "full_support"
                    ]
                    for record in selected
                ]
            )
        ),
        "emission_rate": float(np.mean(emitted)),
        "emission_decisions_match_legacy_rate": float(
            np.mean([new == old for new, old in zip(emitted, legacy_emitted, strict=True)])
        ),
        "finite_bound_rate": float(
            np.mean([item["finite_bound_emitted"] for item in certificates])
        ),
        "improves_over_zero_initialization_rate": float(
            np.mean([item["improves_over_zero_initialization"] for item in certificates])
        ),
        "below_two_value_bound_rate": float(
            np.mean([item["below_two_value_bound"] for item in certificates])
        ),
        "emitted_bound_mean": None if not finite else float(np.mean(finite)),
        "emitted_bound_median": None if not finite else float(np.median(finite)),
        "emitted_total_bound_reduction_vs_old_mean": (
            None if not reductions else float(np.mean(reductions))
        ),
        "residual_radius_ratio_vs_old_mean": (
            None if not radius_ratios else float(np.mean(radius_ratios))
        ),
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
        audit = record["time_uniform_certificate"]["oracle_audit"]
        for kind, families in (
            ("residual_event", audit["fixed_target_residuals"]),
            ("stitch_residual_event", audit["stitch_fixed_target_residuals"]),
        ):
            for family, family_audit in families.items():
                for violation in family_audit["violations"]:
                    violations.append(
                        {**identity, "kind": kind, "family": family, **violation}
                    )
        for route, route_audit in audit["routes"].items():
            if route_audit["violation"] is True:
                violations.append(
                    {**identity, "kind": "route_bound", "route": route, **route_audit}
                )
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/FP-TU-001/claude"),
    )
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_dir = args.baseline_dir.resolve()
    result_dir = args.result_dir.resolve()
    observed_hashes = {name: sha256_file(baseline_dir / name) for name in BASELINE_HASHES}
    if observed_hashes != BASELINE_HASHES:
        raise AssertionError("frozen baseline hashes differ")

    old_config = load_strict_json(baseline_dir / "config.json")
    old_records = load_strict_json(baseline_dir / "task_results.json")
    old_summary = load_strict_json(baseline_dir / "summary.json")
    if len(old_records) != 480:
        raise AssertionError("frozen baseline must contain 480 records")
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
        else {"passed": True, "skipped": "smoke aggregates use a reduced task set"}
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
    radius_checks = recompute_radius_checks(
        new_records,
        gamma=float(new_config["gamma"]),
        delta=float(new_config["certificate_delta"]),
    )
    exhaustive = exhaustive_count_radius_audit()
    emission = emission_and_bound_audit(new_records)
    lengths = sorted({int(record["trajectory_length"]) for record in new_records})
    route_rows = [
        route_and_length_summary(new_records, route, length)
        for length in lengths
        for route in THEOREM_ROUTES
    ]
    violations = collect_violations(new_records)

    passed = bool(
        records_regression["passed"]
        and config_regression["passed"]
        and summary_regression["passed"]
        and namespaces["passed"]
        and schema["passed"]
        and radius_checks["passed"]
        and exhaustive["passed"]
        and emission["passed"]
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
        "recorded_radius_recomputation": radius_checks,
        "exhaustive_count_radius_audit": exhaustive,
        "emission_and_bound_audit": emission,
        "route_by_length": route_rows,
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
        handle.write(f"RADIUS_RECOMPUTE_PROBLEM_COUNT: {radius_checks['problem_count']}\n")
        handle.write(f"EMISSION_BOUND_PROBLEM_COUNT: {emission['problem_count']}\n")
        handle.write(f"EMPIRICAL_AUDIT_VIOLATION_COUNT: {len(violations)}\n")

    print(json.dumps(strict_json_ready(regression), ensure_ascii=False, indent=2))
    if not passed:
        raise AssertionError("time-uniform regression or schema audit failed")
    print("PASS time-uniform certificate analysis")


if __name__ == "__main__":
    main()
