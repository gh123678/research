"""Additive paired evaluator for the frozen FP-TU-001 mixture certificate."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_visit_indexed_certificates as legacy
from time_uniform_mixture_certificate import build_time_uniform_certificate


TASK_ID = "FP-TU-001"
ACTIVATION_COMMIT = "0ce18b4676f70ca0556496e804aa65563efa63da"
BASELINE_HASHES = {
    "config.json": "bcc377b422711302163b976d5d5ece389e1e3ee8540d37817a8719bf9ec3bf8a",
    "task_results.json": "929e2f65689af850b65f000ee6675a8a87b3506c2e2a9600c07d28cf183c6d52",
    "summary.json": "fa13619755b627b9dcff281dc5e4bc9b4d0262b9a6592f2982012b53df2b34ed",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_task_baseline(path: Path) -> dict[str, Any]:
    observed = {name: sha256_file(path / name) for name in BASELINE_HASHES}
    if observed != BASELINE_HASHES:
        raise RuntimeError(f"frozen FP-MART-001 baseline mismatch: {observed}")
    records = json.loads((path / "task_results.json").read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 480:
        raise RuntimeError("frozen FP-MART-001 baseline must contain 480 records")
    return {"directory": str(path), "hashes": observed, "record_count": len(records)}


def _summary_key(record: dict[str, Any], route: str) -> tuple[Any, ...]:
    return (
        int(record["trajectory_length"]),
        int(record["n_actions"]),
        float(record["pi_min"]),
        float(record["beta"]),
        float(record["mixing"]),
        float(record["gap_bonus"]),
        route,
    )


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def augment_summary(
    old_summary: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for route in legacy.THEOREM_ROUTES:
            grouped[_summary_key(record, route)].append(record)
    output: list[dict[str, Any]] = []
    for old_row in old_summary:
        row = copy.deepcopy(old_row)
        route_name = str(row["route"])
        if route_name not in legacy.THEOREM_ROUTES:
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
        new_routes = [item["time_uniform_certificate"]["routes"][route_name] for item in selected]
        old_routes = [item["visit_indexed_certificate"]["routes"][route_name] for item in selected]
        emitted = [item for item in new_routes if item["finite_bound_emitted"]]
        paired = [
            (float(new["total_bound"]), float(old["total_bound"]))
            for new, old in zip(new_routes, old_routes, strict=True)
            if new["total_bound"] is not None and old["total_bound"] is not None
        ]
        family_name = "pair_bellman" if route_name.startswith("direct") else "recovery"
        ratios = []
        for item in selected:
            family = item["time_uniform_certificate"]["event"][family_name]
            ratios.extend(float(value) for value in family["mixture_to_legacy_ratio_by_group"] if value is not None)
        value_bound = float(selected[0]["time_uniform_certificate"]["event"]["value_bound"])
        row["time_uniform_certificate"] = {
            "records": len(selected),
            "emitted": len(emitted),
            "emission_rate": len(emitted) / len(selected),
            "mean_mixture_to_legacy_radius_ratio": _mean(ratios),
            "max_mixture_to_legacy_radius_ratio": max(ratios) if ratios else None,
            "mean_total_bound_reduction": _mean([old - new for new, old in paired]),
            "all_emitted_bounds_nonincreasing": all(new <= old + 1e-12 for new, old in paired),
            "useful_b_count": sum(float(item["total_bound"]) < value_bound for item in emitted),
            "useful_2b_count": sum(float(item["total_bound"]) < 2.0 * value_bound for item in emitted),
            "oracle_violation_count": sum(
                item["time_uniform_certificate"]["oracle_audit"]["routes"][route_name]["violation"] is True
                for item in selected
            ),
        }
        output.append(row)
    return output


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    args = legacy.parse_args()
    result_dir = args.output_dir.resolve()
    baseline_dir = (project_dir / "results/FP-MART-001/codex").resolve()
    baseline_before = verify_task_baseline(baseline_dir)

    new_certificates: list[dict[str, Any]] = []
    by_old_identity: dict[int, dict[str, Any]] = {}
    original_builder = legacy.build_visit_indexed_certificate
    original_audit = legacy.build_oracle_audit

    def paired_builder(**arguments: Any) -> dict[str, Any]:
        old_certificate = original_builder(**arguments)
        new_certificate = build_time_uniform_certificate(**arguments)
        new_certificates.append(new_certificate)
        by_old_identity[id(old_certificate)] = new_certificate
        return old_certificate

    def paired_audit(**arguments: Any) -> dict[str, Any]:
        old_certificate = arguments["visit_certificate"]
        new_certificate = by_old_identity[id(old_certificate)]
        new_arguments = dict(arguments)
        new_arguments["visit_certificate"] = new_certificate
        new_certificate["oracle_audit"] = original_audit(**new_arguments)
        return original_audit(**arguments)

    legacy.build_visit_indexed_certificate = paired_builder
    legacy.build_oracle_audit = paired_audit
    legacy.PREFLIGHT_SCRIPTS = (*legacy.PREFLIGHT_SCRIPTS, "verify_time_uniform_mixture_certificate.py")
    try:
        legacy.main()
    finally:
        legacy.build_visit_indexed_certificate = original_builder
        legacy.build_oracle_audit = original_audit

    records = json.loads((result_dir / "task_results.json").read_text(encoding="utf-8"))
    if len(records) != len(new_certificates):
        raise RuntimeError("paired certificate count differs from generated records")
    for record, certificate in zip(records, new_certificates, strict=True):
        if "time_uniform_certificate" in record:
            raise RuntimeError("new namespace already exists")
        record["time_uniform_certificate"] = certificate

    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    config["time_uniform_certificate"] = {
        "task_id": TASK_ID,
        "method": "finite_geometric_time_uniform_cosh_mixture",
        "components": 15,
        "target_counts": [2**index for index in range(15)],
        "reported_boundary": "mixture_root",
        "stitch_role": "audit_and_upper_bracket_only",
        "tolerance": 1e-12,
        "max_iterations": 200,
        "common_activation_commit": ACTIVATION_COMMIT,
        "frozen_regression_baseline": baseline_before,
    }
    old_summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    summary = augment_summary(old_summary, records)
    legacy.write_json(result_dir / "config.json", config)
    legacy.write_json(result_dir / "task_results.json", records)
    legacy.write_json(result_dir / "summary.json", summary)
    legacy.write_json(
        result_dir / "regression.json",
        {
            "status": "pending_analyzer",
            "baseline": baseline_before,
            "generated_records": len(records),
        },
    )

    command = subprocess.list2cmdline([sys.executable, "-B", Path(__file__).name, *sys.argv[1:]])
    (result_dir / "commands.log").write_text(
        f"WORKDIR: {project_dir}\nEVALUATION: {command}\n", encoding="utf-8"
    )
    environment = json.loads((result_dir / "environment.json").read_text(encoding="utf-8"))
    environment.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "script": str(Path(__file__).resolve()),
            "common_activation_commit": ACTIVATION_COMMIT,
            "task_id": TASK_ID,
            "frozen_regression_baseline": baseline_before,
        }
    )
    legacy.write_json(result_dir / "environment.json", environment)
    baseline_after = verify_task_baseline(baseline_dir)
    if baseline_before != baseline_after:
        raise RuntimeError("frozen FP-MART-001 baseline changed during evaluation")
    print(f"PASS {TASK_ID} paired evaluation with {len(records)} records")


if __name__ == "__main__":
    main()
