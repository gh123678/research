"""FP-ITER4-001 analyzer: per-level attrition and gain across four steps.

Evaluates H1--H9, including the three pre-registered step-4 predictions against
the sealed FP-ITER3-001 step-1..3 baseline.

Usage:
    python -B analyze_fp_iter4_001.py --result-dir <dir> [--write-results]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TASK_ID = "FP-ITER4-001"
PRIMARY = ("expected_exact", "expected_finite")
ITER3_TASK_BUNDLE = (
    Path(__file__).resolve().parent
    / "results"
    / "FP-ITER3-001"
    / "claude"
    / "formal"
    / "task_results.json"
)


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key: {key}")
        seen.add(key)
        out[key] = value
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()

    bundle = json.loads(
        (args.result_dir / "task_results.json").read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicates,
    )
    records = bundle["records"]
    max_steps = int(bundle["max_steps"])
    blocks = [
        (record, route, record["routes"][route])
        for record in records
        for route in PRIMARY
    ]
    steps = [(r, route, b, s) for r, route, b in blocks for s in b["steps"]]

    per_level: dict[int, dict[str, Any]] = {}
    for level in range(1, max_steps + 1):
        gains = [
            s["oracle_audit"]["total_value_gain"]
            for _, _, _, s in steps
            if s["step"] == level and s["update_emitted"]
        ]
        per_level[level] = {
            "emissions": len(gains),
            "mean_gain": (sum(gains) / len(gains)) if gains else None,
            "min_gain": min(gains) if gains else None,
            "max_gain": max(gains) if gains else None,
        }

    # H1/H2 against the sealed FP-ITER3-001 bundle.
    h1_compared = 0
    h1_mismatches: list[tuple] = []
    if ITER3_TASK_BUNDLE.exists():
        sealed = json.loads(ITER3_TASK_BUNDLE.read_text(encoding="utf-8"))
        sealed_by_key = {
            (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
        }
        for record, route, block in blocks:
            ref = sealed_by_key.get((float(record["mixing"]), int(record["task_index"])))
            if ref is None:
                continue
            ref_steps = ref["routes"][route]["steps"]
            for index in range(min(3, len(block["steps"]), len(ref_steps))):
                h1_compared += 1
                a, b = block["steps"][index], ref_steps[index]
                if (
                    a["update_emitted"] != b["update_emitted"]
                    or a["eta_selected"] != b["eta_selected"]
                    or a["e_q"] != b["e_q"]
                ):
                    h1_mismatches.append(
                        (
                            record["mixing"],
                            record["task_index"],
                            route,
                            index + 1,
                        )
                    )
    sealed_three = None
    if ITER3_TASK_BUNDLE.exists():
        sealed_three = sum(
            1
            for r in json.loads(ITER3_TASK_BUNDLE.read_text(encoding="utf-8"))["records"]
            for route in PRIMARY
            if r["routes"][route]["emitted_steps"] >= 3
        )

    violations = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s["oracle_audit"].get("certificate_violation")
    ]
    nondegrading = [
        (r["mixing"], r["task_index"], route, s["step"], s["oracle_audit"].get("value_delta_vs_previous"))
        for r, route, _, s in steps
        if s["update_emitted"]
        and min(s["oracle_audit"]["value_delta_vs_previous"]) < -1e-12
    ]
    repro_failures = [
        (r["mixing"], r["task_index"], route)
        for r, route, b in blocks
        if b["step1_reproduction"].get("exact") is False
    ]

    reasons: dict[str, int] = {}
    for _, _, _, s in steps:
        if s["update_emitted"]:
            continue
        for reason in s["ordered_reasons"] or ["<none>"]:
            reasons[reason] = reasons.get(reason, 0) + 1

    n3 = per_level[3]["emissions"]
    n4 = per_level[4]["emissions"] if max_steps >= 4 else None
    mean3 = per_level[3]["mean_gain"]
    mean4 = per_level[4]["mean_gain"] if max_steps >= 4 else None
    min4 = per_level[4]["min_gain"] if max_steps >= 4 else None

    summary = {
        "task_id": TASK_ID,
        "label": bundle.get("label"),
        "record_count": len(records),
        "route_records": len(blocks),
        "max_steps": max_steps,
        "steps_executed": len(steps),
        "per_level": {str(k): v for k, v in per_level.items()},
        "step1_reproduction_failures": len(repro_failures),
        "certificate_violations": len(violations),
        "nondegrading_violations": len(nondegrading),
        "abstention_reasons": reasons,
        "H1_step_limit_monotone": {
            "compared_entries": h1_compared,
            "mismatches": len(h1_mismatches),
            "outcome": "PASS" if (h1_compared > 0 and not h1_mismatches) else "FAIL",
        },
        "H2_step3_reproduced": {
            "sealed_three_step_routes": sealed_three,
            "this_run_three_step_routes": sum(
                1 for _, _, b in blocks if b["emitted_steps"] >= 3
            ),
            "outcome": (
                "PASS"
                if sealed_three is not None
                and sum(1 for _, _, b in blocks if b["emitted_steps"] >= 3)
                == sealed_three
                else "FAIL"
            ),
        },
        "H3_fourth_step_certifiable": {
            "n4": n4,
            "outcome": "PASS" if (n4 or 0) > 0 else "FALSIFIED",
        },
        "H4_fourth_step_valid": {
            "all_gains_positive": bool(
                per_level.get(4, {}).get("min_gain") is not None
                and per_level[4]["min_gain"] > 0
            ),
            "outcome": (
                "PASS"
                if (per_level.get(4, {}).get("min_gain") or 0) > 0 and not nondegrading
                else "FALSIFIED"
            ),
        },
        "H5_monotone_value": bool(not nondegrading),
        "H6_no_certificate_violations": bool(not violations),
        "H7_attrition_prediction": {
            "prediction": "n4 < n3",
            "n3": n3,
            "n4": n4,
            "outcome": "PASS" if (n4 is not None and n4 < n3) else "FALSIFIED",
        },
        "H8_mean_gain_prediction": {
            "prediction": "mean_gain_step4 < mean_gain_step3",
            "mean_gain_step3": mean3,
            "mean_gain_step4": mean4,
            "outcome": (
                "PASS"
                if (mean3 is not None and mean4 is not None and mean4 < mean3)
                else "FALSIFIED"
            ),
        },
        "H9_non_vacuity_prediction": {
            "prediction": "min_gain_step4 > 0.05",
            "min_gain_step4": min4,
            "outcome": "PASS" if (min4 is not None and min4 > 0.05) else "FALSIFIED",
        },
        "nondegrading_violations_detail": nondegrading,
    }
    analysis = {
        "task_id": TASK_ID,
        "reconstruction": "recomputed from task_results.json with strict duplicate-key detection",
        "per_level_gains": {
            str(level): [
                s["oracle_audit"]["total_value_gain"]
                for _, _, _, s in steps
                if s["step"] == level and s["update_emitted"]
            ]
            for level in range(1, max_steps + 1)
        },
        "four_step_records": [
            {
                "mixing": r["mixing"],
                "task_index": r["task_index"],
                "route": route,
                "eta": [s["eta_selected"] for s in b["steps"]],
                "e_q": [s["e_q"] for s in b["steps"]],
                "min_lb": [s["min_lb"] for s in b["steps"]],
                "gain": [
                    s["oracle_audit"].get("total_value_gain")
                    for s in b["steps"]
                    if s["update_emitted"]
                ],
            }
            for r, route, b in blocks
            if b["emitted_steps"] >= 4
        ],
        "artifacts": {
            name: sha256(args.result_dir / name)
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

    print(f"{TASK_ID}: routes={len(blocks)} steps={len(steps)} max_steps={max_steps}")
    for level in range(1, max_steps + 1):
        row = per_level[level]
        print(
            f"  step {level}: emissions={row['emissions']:>3} "
            f"mean_gain={row['mean_gain']} min_gain={row['min_gain']}"
        )
    print(
        f"  emitted all {max_steps} steps="
        f"{sum(1 for _, _, b in blocks if b['emitted_steps'] >= max_steps)} "
        f"repro_failures={len(repro_failures)} "
        f"cert_violations={len(violations)} nondeg={len(nondegrading)}"
    )
    for key in (
        "H1_step_limit_monotone",
        "H2_step3_reproduced",
        "H3_fourth_step_certifiable",
        "H4_fourth_step_valid",
        "H7_attrition_prediction",
        "H8_mean_gain_prediction",
        "H9_non_vacuity_prediction",
    ):
        entry = summary[key]
        print(f"  {key}: {entry.get('outcome')}  {entry}")


if __name__ == "__main__":
    main()
