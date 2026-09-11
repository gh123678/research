"""FP-SCALE-002 model-free analyzer.

Reads only a sealed result bundle and recomputes every reported metric with a
strict JSON reload and duplicate-key detection. Writes ``summary.json`` and
``analysis.json``.

Usage:
    python -B analyze_fp_scale_002.py --result-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

TASK_ID = "FP-SCALE-002"
PRIMARY = ("variance_adaptive_exact", "variance_adaptive_finite")
CONTROL = "envelope_control_exact"


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key: {key}")
        seen.add(key)
        out[key] = value
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var / len(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()

    raw = (args.result_dir / "task_results.json").read_text(encoding="utf-8")
    bundle = json.loads(raw, object_pairs_hook=no_duplicate_keys)
    records = bundle["records"]

    attempted = 0
    emitted = 0
    nondegrading = 0
    strict = 0
    violations = 0
    control_emitted = 0
    control_violations = 0
    adaptive_e: list[float] = []
    control_e: list[float] = []
    ratios: list[float] = []
    realized: list[float] = []
    etas: dict[float, int] = {}
    reasons: dict[str, int] = {}
    gains: list[float] = []
    count_failures = 0
    per_record: list[dict[str, Any]] = []

    for record in records:
        min_count = record["min_cert_count_observed"]
        if min_count < 2 * record["min_half_count"]:
            count_failures += 1
        control_e_q = record["routes"][CONTROL]["e_q"]
        if control_e_q is not None:
            control_e.append(float(control_e_q))
        control_audit = record["routes"][CONTROL]["oracle_audit"]
        if control_audit["certificate_violation"]:
            control_violations += 1
        if record["routes"][CONTROL]["update_emitted"]:
            control_emitted += 1

        for name in PRIMARY:
            entry = record["routes"][name]
            audit = entry["oracle_audit"]
            attempted += 1
            realized.append(float(audit["realized_q_sup_error"]))
            if entry["e_q"] is not None:
                adaptive_e.append(float(entry["e_q"]))
                if control_e_q is not None and float(entry["e_q"]) > 0.0:
                    ratios.append(float(control_e_q) / float(entry["e_q"]))
            if audit["certificate_violation"]:
                violations += 1
            if entry["update_emitted"]:
                emitted += 1
                etas[float(entry["eta_selected"])] = (
                    etas.get(float(entry["eta_selected"]), 0) + 1
                )
                gains.append(float(audit["total_value_gain"]))
                if audit["componentwise_nondegrading"]:
                    nondegrading += 1
                if audit["total_value_gain"] > 0.0:
                    strict += 1
            else:
                for reason in entry["ordered_reasons"]:
                    reasons[reason] = reasons.get(reason, 0) + 1
        per_record.append(
            {
                "mixing": record["mixing"],
                "task_index": record["task_index"],
                "min_cert_count": min_count,
                "variance_adaptive_e_q": record["routes"][PRIMARY[0]]["e_q"],
                "envelope_control_e_q": control_e_q,
                "primary_emissions": sum(
                    1 for n in PRIMARY if record["routes"][n]["update_emitted"]
                ),
            }
        )

    summary = {
        "task_id": TASK_ID,
        "mode": bundle["mode"],
        "label": bundle["label"],
        "record_count": len(records),
        "primary_route_records": attempted,
        "primary_emissions": emitted,
        "primary_emission_rate": emitted / attempted if attempted else 0.0,
        "componentwise_nondegrading": nondegrading,
        "strict_improvements": strict,
        "certificate_violations": violations,
        "control_emissions": control_emitted,
        "control_certificate_violations": control_violations,
        "count_rule_failures": count_failures,
        "eta_selected_counts": {str(k): v for k, v in sorted(etas.items())},
        "abstention_reason_counts": reasons,
        "mean_e_q_adaptive": mean(adaptive_e),
        "mean_e_q_control": mean(control_e),
        "mean_control_over_adaptive_ratio": mean(ratios),
        "min_control_over_adaptive_ratio": min(ratios) if ratios else None,
        "max_control_over_adaptive_ratio": max(ratios) if ratios else None,
        "control_worse_in_every_record": bool(ratios and min(ratios) > 1.0),
        "mean_realized_q_sup_error": mean(realized),
        "mean_value_gain_among_emitted": mean(gains),
        "h3_threshold": 12,
        "h3_passed": bool(emitted >= 12),
        "h4_passed": bool(emitted > 0 and strict > 0 and nondegrading == emitted),
        "h5_passed": bool(ratios and min(ratios) > 1.0),
        "h2_passed": bool(violations == 0 and control_violations == 0),
        "per_record": per_record,
    }
    analysis = {
        "task_id": TASK_ID,
        "reconstruction": "recomputed from task_results.json with strict duplicate-key detection",
        "ratios_control_over_adaptive": ratios,
        "adaptive_e_q_values": adaptive_e,
        "control_e_q_values": control_e,
        "realized_errors": realized,
        "artifacts": {
            name: sha256_file(args.result_dir / name)
            for name in ("task_results.json", "config.json", "environment.json")
            if (args.result_dir / name).exists()
        },
    }

    if args.write_results:
        (args.result_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        (args.result_dir / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
        )

    print(
        f"{TASK_ID} analysis: records={summary['record_count']} "
        f"primary_emissions={emitted}/{attempted} "
        f"violations={violations} "
        f"control_worse_everywhere={summary['control_worse_in_every_record']} "
        f"H3={'PASS' if summary['h3_passed'] else 'FAIL'} "
        f"H4={'PASS' if summary['h4_passed'] else 'FAIL'} "
        f"H5={'PASS' if summary['h5_passed'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
