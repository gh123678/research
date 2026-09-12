"""FP-ITER3-001 analyzer: per-level attrition and gain across three steps.

Generalises the FP-ITER2-001 analyzer to any frozen horizon, and additionally
evaluates the two pre-registered predictions of FP-ITER3-001 (H6 attrition,
H7 filtering) against the sealed step-1/step-2 baseline.

Usage:
    python -B analyze_fp_iter3_001.py --result-dir <dir> [--write-results]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TASK_ID = "FP-ITER3-001"
PRIMARY = ("expected_exact", "expected_finite")
ITER2_SUMMARY = (
    Path(__file__).resolve().parent
    / "results"
    / "FP-ITER2-001"
    / "claude"
    / "formal"
    / "summary.json"
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

    # ---- per-level attrition and gains ------------------------------------
    emissions: dict[int, int] = {}
    gains: dict[int, list[float]] = {level: [] for level in range(1, max_steps + 1)}
    for level in range(1, max_steps + 1):
        emissions[level] = sum(
            1
            for _, _, b in blocks
            if len(b["steps"]) >= level and b["steps"][level - 1]["update_emitted"]
        )
    for _, _, block in blocks:
        for step in block["steps"]:
            if step["update_emitted"]:
                gains[step["step"]].append(step["oracle_audit"]["total_value_gain"])

    per_level = {
        level: {
            "emissions": emissions[level],
            "mean_gain": (sum(gains[level]) / len(gains[level])) if gains[level] else None,
            "min_gain": min(gains[level]) if gains[level] else None,
            "max_gain": max(gains[level]) if gains[level] else None,
        }
        for level in range(1, max_steps + 1)
    }

    # ---- violations --------------------------------------------------------
    violations = []
    nondegrading = []
    for record, route, block in blocks:
        for step in block["steps"]:
            audit = step["oracle_audit"]
            if audit.get("certificate_violation"):
                violations.append(
                    {
                        "mixing": record["mixing"],
                        "task_index": record["task_index"],
                        "route": route,
                        "step": step["step"],
                    }
                )
            if step["update_emitted"] and min(
                audit["value_delta_vs_previous"]
            ) < -1e-12:
                nondegrading.append(
                    {
                        "mixing": record["mixing"],
                        "task_index": record["task_index"],
                        "route": route,
                        "step": step["step"],
                        "delta": audit["value_delta_vs_previous"],
                    }
                )

    repro_failures = [
        (r["mixing"], r["task_index"], route)
        for r, route, b in blocks
        if b["step1_reproduction"].get("exact") is False
    ]

    reasons: dict[str, int] = {}
    for _, _, block in blocks:
        last = block["steps"][-1]
        if last["update_emitted"]:
            continue
        for reason in last["ordered_reasons"] or ["<none>"]:
            reasons[reason] = reasons.get(reason, 0) + 1

    # ---- sealed baseline and pre-registered predictions --------------------
    baseline = None
    if ITER2_SUMMARY.exists():
        baseline = json.loads(ITER2_SUMMARY.read_text(encoding="utf-8"))

    h1_h2_ok = bool(
        baseline
        and emissions.get(1) == baseline["step1_emissions"]
        and emissions.get(2) == baseline["emitted_two_steps"]
        and not repro_failures
    )
    n2 = emissions.get(2, 0)
    n3 = emissions.get(3, 0)
    min2 = baseline["min_gain_step2"] if baseline else None
    min3 = per_level.get(3, {}).get("min_gain")

    summary = {
        "task_id": TASK_ID,
        "label": bundle.get("label"),
        "record_count": len(records),
        "route_records": len(blocks),
        "max_steps": max_steps,
        "per_level": {str(k): v for k, v in per_level.items()},
        "emitted_all_steps": sum(
            1 for _, _, b in blocks if b["emitted_steps"] >= max_steps
        ),
        "step1_reproduction_failures": len(repro_failures),
        "certificate_violations": len(violations),
        "nondegrading_violations": len(nondegrading),
        "abstention_reasons": reasons,
        "baseline_iter2": (
            {
                "step1_emissions": baseline["step1_emissions"],
                "emitted_two_steps": baseline["emitted_two_steps"],
                "mean_gain_step2": baseline["mean_gain_step2"],
                "min_gain_step2": baseline["min_gain_step2"],
            }
            if baseline
            else None
        ),
        "H1_step_limit_monotone": h1_h2_ok,
        "H2_step2_reproduced": bool(
            baseline and emissions.get(2) == baseline["emitted_two_steps"]
        ),
        "H3_third_step_certifiable": bool(n3 > 0),
        "H4_third_step_valid": bool(
            gains.get(3) and all(g > 0 for g in gains[3]) and not nondegrading
        ),
        "H5_monotone_value": bool(not nondegrading),
        "H6_attrition_prediction": {
            "prediction": "n3 < n2",
            "n2": n2,
            "n3": n3,
            "outcome": "PASS" if n3 < n2 else "FALSIFIED",
        },
        "H7_filtering_prediction": {
            "prediction": "min_gain_step3 > min_gain_step2",
            "min_gain_step2": min2,
            "min_gain_step3": min3,
            "outcome": (
                "PASS"
                if (min2 is not None and min3 is not None and min3 > min2)
                else "FALSIFIED"
            ),
        },
        "H8_no_certificate_violations": bool(not violations),
        "violations": violations,
        "nondegrading_violations_detail": nondegrading,
    }
    analysis = {
        "task_id": TASK_ID,
        "reconstruction": "recomputed from task_results.json with strict duplicate-key detection",
        "per_level_gains": {str(k): v for k, v in gains.items()},
        "multi_step_records": [
            {
                "mixing": r["mixing"],
                "task_index": r["task_index"],
                "route": route,
                "emitted_steps": b["emitted_steps"],
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
            if b["emitted_steps"] >= 2
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

    print(f"{TASK_ID}: routes={len(blocks)} max_steps={max_steps}")
    for level in range(1, max_steps + 1):
        info = per_level[level]
        print(
            f"  step {level}: emissions={info['emissions']} "
            f"mean_gain={info['mean_gain']} min_gain={info['min_gain']}"
        )
    print(
        f"  emitted all {max_steps} steps={summary['emitted_all_steps']} "
        f"repro_failures={len(repro_failures)} "
        f"cert_violations={len(violations)} "
        f"nondegrading_violations={len(nondegrading)}"
    )
    print(
        "  H1={} H2={} H3={} H4={} H5={} H8={}".format(
            summary["H1_step_limit_monotone"],
            summary["H2_step2_reproduced"],
            summary["H3_third_step_certifiable"],
            summary["H4_third_step_valid"],
            summary["H5_monotone_value"],
            summary["H8_no_certificate_violations"],
        )
    )
    print(
        f"  H6 attrition  : predict {summary['H6_attrition_prediction']['prediction']} "
        f"-> n2={n2} n3={n3} = {summary['H6_attrition_prediction']['outcome']}"
    )
    print(
        f"  H7 filtering  : predict {summary['H7_filtering_prediction']['prediction']} "
        f"-> {min2} vs {min3} = {summary['H7_filtering_prediction']['outcome']}"
    )


if __name__ == "__main__":
    main()
