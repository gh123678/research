"""FP-ITER2-001 analyzer: attrition and monotonicity of the second certified step.

Usage:
    python -B analyze_fp_iter2_001.py --result-dir <dir> [--write-results]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TASK_ID = "FP-ITER2-001"
PRIMARY = ("expected_exact", "expected_finite")


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

    blocks = [
        (record, route, record["routes"][route])
        for record in records
        for route in PRIMARY
    ]

    two = [b for b in blocks if b[2]["emitted_steps"] >= 2]
    one = [b for b in blocks if b[2]["emitted_steps"] == 1]
    zero = [b for b in blocks if b[2]["emitted_steps"] == 0]

    repro_failures = [
        (r["mixing"], r["task_index"], route, block["step1_reproduction"])
        for r, route, block in blocks
        if block["step1_reproduction"].get("exact") is False
    ]
    repro_checked = sum(1 for _, _, b in blocks if b["step1_reproduction"])

    violations = []
    nondegrading = []
    gains_step1 = []
    gains_step2 = []
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
                        "e_q": step["e_q"],
                        "realized": audit["realized_q_sup_error_vs_current_target_pi"],
                    }
                )
            if not step["update_emitted"]:
                continue
            if not audit.get("componentwise_nondegrading", False):
                nondegrading.append(
                    {
                        "mixing": record["mixing"],
                        "task_index": record["task_index"],
                        "route": route,
                        "step": step["step"],
                        "delta": audit["value_delta_vs_previous"],
                    }
                )
            if step["step"] == 1:
                gains_step1.append(audit["total_value_gain"])
            else:
                gains_step2.append(audit["total_value_gain"])

    reasons: dict[str, int] = {}
    for _, _, block in blocks:
        if block["emitted_steps"] >= 2:
            continue
        for reason in block["steps"][-1]["ordered_reasons"] or ["<none>"]:
            reasons[reason] = reasons.get(reason, 0) + 1

    # H4: every two-step route must be componentwise non-decreasing at BOTH steps.
    h4_bad = []
    for record, route, block in two:
        running = None
        previous = None
        for step in block["steps"]:
            audit = step["oracle_audit"]
            delta = audit["value_delta_vs_previous"]
            if min(delta) < -1e-12:
                h4_bad.append((record["mixing"], record["task_index"], route, step["step"]))
            previous = running
            del previous

    mean_gain1 = sum(gains_step1) / len(gains_step1) if gains_step1 else None
    mean_gain2 = sum(gains_step2) / len(gains_step2) if gains_step2 else None

    summary = {
        "task_id": TASK_ID,
        "label": bundle.get("label"),
        "record_count": len(records),
        "route_records": len(blocks),
        "emitted_two_steps": len(two),
        "emitted_one_step": len(one),
        "emitted_zero_steps": len(zero),
        "step1_reproduction_checked": repro_checked,
        "step1_reproduction_failures": len(repro_failures),
        "certificate_violations": len(violations),
        "nondegrading_violations": len(nondegrading),
        "step1_emissions": len(gains_step1),
        "step2_emissions": len(gains_step2),
        "mean_gain_step1": mean_gain1,
        "mean_gain_step2": mean_gain2,
        "min_gain_step1": min(gains_step1) if gains_step1 else None,
        "min_gain_step2": min(gains_step2) if gains_step2 else None,
        "abstention_reasons": reasons,
        "H1_step1_reproduction": bool(repro_checked > 0 and not repro_failures),
        "H2_second_step_certifiable": bool(len(two) > 0),
        "H3_second_step_valid": bool(
            gains_step2 and all(g > 0 for g in gains_step2) and not nondegrading
        ),
        "H4_monotone_value": bool(not h4_bad),
        "H5_no_certificate_violations": bool(not violations),
        "H6_attrition_quantified": True,
        "violations": violations,
        "nondegrading_violations_detail": nondegrading,
        "reproduction_failures": repro_failures,
    }
    analysis = {
        "task_id": TASK_ID,
        "reconstruction": "recomputed from task_results.json with strict duplicate-key detection",
        "two_step_records": [
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
            for r, route, b in two
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

    print(
        f"{TASK_ID}: routes={len(blocks)} two_steps={len(two)} one_step={len(one)} "
        f"zero={len(zero)} repro_failures={len(repro_failures)} "
        f"cert_violations={len(violations)} nondegrading_violations={len(nondegrading)}"
    )
    print(
        f"  step1 emissions={len(gains_step1)} mean_gain={mean_gain1} "
        f"min_gain={min(gains_step1) if gains_step1 else None}"
    )
    print(
        f"  step2 emissions={len(gains_step2)} mean_gain={mean_gain2} "
        f"min_gain={min(gains_step2) if gains_step2 else None}"
    )
    print(
        "  H1={} H2={} H3={} H4={} H5={} H6={}".format(
            summary["H1_step1_reproduction"],
            summary["H2_second_step_certifiable"],
            summary["H3_second_step_valid"],
            summary["H4_monotone_value"],
            summary["H5_no_certificate_violations"],
            summary["H6_attrition_quantified"],
        )
    )


if __name__ == "__main__":
    main()
