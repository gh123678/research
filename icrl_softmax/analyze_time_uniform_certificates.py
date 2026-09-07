"""Strict regression and acceptance analysis for FP-TU-001 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_visit_indexed_certificates import write_json
from time_uniform_mixture_certificate import (
    DEFAULT_MAX_COUNT,
    build_mixture_grid,
    mixture_radius,
    old_visit_indexed_radius,
    solve_mixture_boundary,
    stitch_boundary,
)


BASELINE_HASHES = {
    "config.json": "bcc377b422711302163b976d5d5ece389e1e3ee8540d37817a8719bf9ec3bf8a",
    "task_results.json": "929e2f65689af850b65f000ee6675a8a87b3506c2e2a9600c07d28cf183c6d52",
    "summary.json": "fa13619755b627b9dcff281dc5e4bc9b4d0262b9a6592f2982012b53df2b34ed",
}
ROUTES = (
    "direct_exact",
    "direct_softmax",
    "vfirst_nosplit_exact",
    "vfirst_nosplit_softmax",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_strict_json(path: Path) -> Any:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite constant {value} in {path}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )


def _compare(expected: Any, observed: Any, path: str, mismatches: list[str]) -> None:
    if isinstance(expected, bool) or isinstance(observed, bool):
        if expected is not observed:
            mismatches.append(path)
        return
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if not math.isclose(float(expected), float(observed), rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append(path)
        return
    if type(expected) is not type(observed):
        mismatches.append(path)
        return
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            mismatches.append(path + ".keys")
            return
        for key in expected:
            _compare(expected[key], observed[key], f"{path}.{key}", mismatches)
        return
    if isinstance(expected, list):
        if len(expected) != len(observed):
            mismatches.append(path + ".length")
            return
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            _compare(left, right, f"{path}[{index}]", mismatches)
        return
    if expected != observed:
        mismatches.append(path)


def _without_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_namespace(item)
            for key, item in value.items()
            if key != "time_uniform_certificate"
        }
    if isinstance(value, list):
        return [_without_namespace(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir", type=Path, default=project / "results/FP-TU-001/codex"
    )
    parser.add_argument(
        "--baseline-dir", type=Path, default=project / "results/FP-MART-001/codex"
    )
    return parser.parse_args()


def exhaustive_radius_audit() -> dict[str, Any]:
    groups = 54
    delta = 0.05
    grid = build_mixture_grid(n_groups=groups, delta=delta)
    previous = math.inf
    maximum_ratios = {str(horizon): 0.0 for horizon in (256, 1024, 4096, 16384)}
    strict_count = 0
    max_stitch_ratio = 0.0
    max_iterations = 0
    for count in range(1, DEFAULT_MAX_COUNT + 1):
        solution = solve_mixture_boundary(count, grid, n_groups=groups, delta=delta)
        q_mix = float(solution["boundary"])
        q_stitch = stitch_boundary(count, grid)
        radius = mixture_radius(
            count,
            reward_bound=1.0,
            gamma=0.7,
            n_groups=groups,
            delta=delta,
        )
        if radius > previous * (1.0 + 2e-12):
            raise AssertionError(f"mixture radius increases at count {count}")
        previous = radius
        if q_mix > q_stitch * (1.0 + 2e-12):
            raise AssertionError(f"mixture root exceeds stitch at count {count}")
        max_stitch_ratio = max(max_stitch_ratio, q_mix / q_stitch)
        max_iterations = max(max_iterations, int(solution["iterations"]))
        for horizon in (256, 1024, 4096, 16384):
            if count > horizon:
                continue
            old = old_visit_indexed_radius(
                count,
                reward_bound=1.0,
                gamma=0.7,
                n_groups=groups,
                horizon=horizon,
                delta=delta,
            )
            ratio = radius / old
            maximum_ratios[str(horizon)] = max(maximum_ratios[str(horizon)], ratio)
            if ratio > 1.0 + 2e-12:
                raise AssertionError(f"radius dominance fails at n={horizon}, k={count}")
            strict_count += ratio < 1.0 - 1e-12
    if strict_count == 0:
        raise AssertionError("radius dominance is nowhere strict")
    return {
        "counts_checked": DEFAULT_MAX_COUNT,
        "maximum_mixture_to_old_ratio_by_horizon": maximum_ratios,
        "maximum_mixture_to_stitch_ratio": max_stitch_ratio,
        "maximum_bisection_iterations": max_iterations,
        "strict_comparisons": strict_count,
        "passed": True,
    }


def analyze_records(records: list[dict[str, Any]], formal: bool) -> dict[str, Any]:
    emissions: dict[str, Counter[int]] = {route: Counter() for route in ROUTES}
    failure_reasons: Counter[str] = Counter()
    ratios: list[float] = []
    reductions: dict[str, list[float]] = defaultdict(list)
    route_violations: list[dict[str, Any]] = []
    residual_violations: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if set(record) - set(_without_namespace(record)) != {"time_uniform_certificate"}:
            raise AssertionError(f"record {index} has an invalid additive namespace")
        old = record["visit_indexed_certificate"]
        new = record["time_uniform_certificate"]
        if "oracle_audit" in new["certificate_inputs"]:
            raise AssertionError(f"record {index} leaks oracle data into certificate inputs")
        for family_name in ("state_bellman", "pair_bellman", "recovery"):
            family = new["event"][family_name]
            for ratio in family["mixture_to_legacy_ratio_by_group"]:
                if ratio is not None:
                    ratio_value = float(ratio)
                    if ratio_value > 1.0 + 2e-12:
                        raise AssertionError(f"record {index} radius dominance failure")
                    ratios.append(ratio_value)
        for route in ROUTES:
            old_route = old["routes"][route]
            new_route = new["routes"][route]
            if (
                old_route["status"] != new_route["status"]
                or old_route["failure_reasons"] != new_route["failure_reasons"]
                or old_route["finite_bound_emitted"] != new_route["finite_bound_emitted"]
            ):
                raise AssertionError(f"record {index} changes emission for {route}")
            for reason in new_route["failure_reasons"]:
                failure_reasons[str(reason)] += 1
            if new_route["finite_bound_emitted"]:
                emissions[route][int(record["trajectory_length"])] += 1
                new_bound = float(new_route["total_bound"])
                old_bound = float(old_route["total_bound"])
                if new_bound > old_bound + 1e-12:
                    raise AssertionError(f"record {index} increases bound for {route}")
                reductions[route].append(old_bound - new_bound)
            audit = new["oracle_audit"]["routes"][route]
            if audit["violation"] is True:
                route_violations.append({"record": index, "route": route, **audit})
        for family_name, audit in new["oracle_audit"]["fixed_target_residuals"].items():
            for violation in audit["violations"]:
                residual_violations.append(
                    {"record": index, "family": family_name, **violation}
                )
    if formal:
        expected = {256: 3, 1024: 102, 4096: 120, 16384: 120}
        for route in ("direct_exact", "vfirst_nosplit_exact"):
            if dict(emissions[route]) != expected:
                raise AssertionError(f"exact emission mismatch for {route}: {emissions[route]}")
    return {
        "record_count": len(records),
        "formal_protocol": formal,
        "emissions_by_route_and_length": {
            route: dict(sorted(counter.items())) for route, counter in emissions.items()
        },
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "radius_ratio": {
            "minimum": min(ratios),
            "mean": sum(ratios) / len(ratios),
            "maximum": max(ratios),
        },
        "total_bound_reduction": {
            route: {
                "count": len(values),
                "minimum": min(values) if values else None,
                "mean": sum(values) / len(values) if values else None,
                "maximum": max(values) if values else None,
            }
            for route, values in reductions.items()
        },
        "oracle_audit": {
            "route_violation_count": len(route_violations),
            "route_violations": route_violations,
            "residual_violation_count": len(residual_violations),
            "residual_violations": residual_violations,
            "used_as_theorem_evidence": False,
        },
    }


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()
    observed_hashes = {name: sha256_file(baseline_dir / name) for name in BASELINE_HASHES}
    if observed_hashes != BASELINE_HASHES:
        raise RuntimeError(f"baseline hash mismatch: {observed_hashes}")
    baseline_config = load_strict_json(baseline_dir / "config.json")
    baseline_records = load_strict_json(baseline_dir / "task_results.json")
    baseline_summary = load_strict_json(baseline_dir / "summary.json")
    config = load_strict_json(result_dir / "config.json")
    records = load_strict_json(result_dir / "task_results.json")
    summary = load_strict_json(result_dir / "summary.json")
    formal = len(records) == 480
    mismatches: list[str] = []
    if formal:
        _compare(baseline_config, _without_namespace(config), "config", mismatches)
        _compare(baseline_records, _without_namespace(records), "records", mismatches)
        _compare(baseline_summary, _without_namespace(summary), "summary", mismatches)
        if mismatches:
            raise AssertionError(f"legacy regression has {len(mismatches)} mismatches: {mismatches[:10]}")
    elif not records:
        raise AssertionError("smoke result is empty")
    radius_audit = exhaustive_radius_audit()
    record_audit = analyze_records(records, formal)
    regression = {
        "status": "PASS",
        "baseline_hashes": observed_hashes,
        "baseline_record_count": len(baseline_records),
        "result_record_count": len(records),
        "formal_protocol": formal,
        "legacy_mismatch_count": len(mismatches),
        "numeric_tolerance": {"relative": 1e-12, "absolute": 1e-12},
        "radius_audit": radius_audit,
        "record_audit": record_audit,
    }
    write_json(result_dir / "regression.json", regression)
    write_json(result_dir / "summary.json", summary)
    checks = [
        "PASS strict JSON with duplicate and nonfinite rejection",
        f"PASS frozen baseline hashes and {len(baseline_records)} records",
        f"PASS legacy regression mismatches={len(mismatches)} formal={formal}",
        f"PASS exhaustive mixture count audit 1..{DEFAULT_MAX_COUNT}",
        "PASS emission preservation and nonincreasing emitted bounds",
        f"PASS oracle separation; route violations={record_audit['oracle_audit']['route_violation_count']}",
    ]
    with (result_dir / "checks.log").open("a", encoding="utf-8") as handle:
        handle.write("\n" + "\n".join(checks) + "\n")
    for name in ("config.json", "task_results.json", "summary.json", "regression.json", "environment.json"):
        load_strict_json(result_dir / name)
    hashes_after = {name: sha256_file(baseline_dir / name) for name in BASELINE_HASHES}
    if hashes_after != observed_hashes:
        raise RuntimeError("baseline changed during analysis")
    print(f"PASS FP-TU-001 analysis: records={len(records)}, formal={formal}")


if __name__ == "__main__":
    main()
