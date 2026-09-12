"""FP-ATTN-ITER-001 analyzer: literal-network iteration vs the numpy route.

Usage:
    python -B analyze_fp_attn_iter_001.py --result-dir <dir> [--write-results]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TASK_ID = "FP-ATTN-ITER-001"
PRIMARY = ("expected_exact", "expected_finite")
ITER3_SUMMARY = (
    Path(__file__).resolve().parent
    / "results"
    / "FP-ITER3-001"
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
    atol = float(bundle["atol"])
    max_steps = int(bundle["max_steps"])
    blocks = [
        (record, route, record["routes"][route])
        for record in records
        for route in PRIMARY
    ]
    steps_all = [(r, route, b, s) for r, route, b in blocks for s in b["steps"]]

    # H1: step 1 anchored to FP-ATTN-001 (single-step) tolerances.
    step1_gaps = [s["q_hat_gap_vs_numpy"] for _, _, _, s in steps_all if s["step"] == 1]
    step1_flips = [
        (r["mixing"], r["task_index"], route)
        for r, route, _, s in steps_all
        if s["step"] == 1 and s["decision_flip_vs_numpy"]
    ]

    # H2: provenance.
    producers = {s["qhat_producer"] for _, _, _, s in steps_all}

    # H3: three-step routes driven by the network.
    three = [b for _, _, b in blocks if b["emitted_steps"] >= 3]
    emissions = {
        level: sum(
            1
            for _, _, b in blocks
            if len(b["steps"]) >= level and b["steps"][level - 1]["update_emitted"]
        )
        for level in range(1, max_steps + 1)
    }

    # H4: decision and eta agreement per step.
    comparisons = [
        (r, route, comp)
        for r, route, b in blocks
        for comp in b["reference_comparisons"]
    ]
    decision_disagreements = [c for c in comparisons if not c[2]["decision_match"]]
    eta_disagreements = [c for c in comparisons if not c[2]["eta_match"]]

    # H5: every flip listed individually with its boundary margin.
    flips = [
        {
            "mixing": r["mixing"],
            "task_index": r["task_index"],
            "route": route,
            "step": s["step"],
            "literal_status": s["status"],
            "literal_e_q": s["e_q"],
            "literal_min_lb": s["min_lb"],
            "numpy_status": s["numpy"]["status"],
            "numpy_e_q": s["numpy"]["e_q"],
            "numpy_min_lb": s["numpy"]["min_lb"],
            "q_hat_gap": s["q_hat_gap_vs_numpy"],
        }
        for r, route, _, s in steps_all
        if s["decision_flip_vs_numpy"]
    ]

    # H6: validity.
    cert_violations = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps_all
        if s["oracle_audit"].get("certificate_violation")
    ]
    nondegrading = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps_all
        if s["update_emitted"]
        and not s["oracle_audit"].get("componentwise_nondegrading", False)
    ]

    gaps = [s["q_hat_gap_vs_numpy"] for _, _, _, s in steps_all]
    gains = {
        level: [
            s["oracle_audit"]["total_value_gain"]
            for _, _, _, s in steps_all
            if s["step"] == level and s["update_emitted"]
        ]
        for level in range(1, max_steps + 1)
    }

    baseline = (
        json.loads(ITER3_SUMMARY.read_text(encoding="utf-8"))
        if ITER3_SUMMARY.exists()
        else None
    )

    reasons: dict[str, int] = {}
    for _, _, _, s in steps_all:
        if s["update_emitted"]:
            continue
        for reason in s["ordered_reasons"] or ["<none>"]:
            reasons[reason] = reasons.get(reason, 0) + 1

    summary = {
        "task_id": TASK_ID,
        "label": bundle.get("label"),
        "record_count": len(records),
        "route_records": len(blocks),
        "max_steps": max_steps,
        "atol": atol,
        "steps_executed": len(steps_all),
        "emissions_by_step": {str(k): v for k, v in emissions.items()},
        "network_three_step_routes": len(three),
        "max_q_hat_gap": max(gaps),
        "mean_q_hat_gap": sum(gaps) / len(gaps),
        "max_step1_q_hat_gap": max(step1_gaps),
        "decision_flips": len(flips),
        "eta_flips": len(eta_disagreements),
        "decision_disagreements": len(decision_disagreements),
        "step_comparisons": len(comparisons),
        "certificate_violations": len(cert_violations),
        "nondegrading_violations": len(nondegrading),
        "abstention_reasons": reasons,
        "mean_gain_by_step": {
            str(k): (sum(v) / len(v) if v else None) for k, v in gains.items()
        },
        "baseline_iter3": (
            {
                "per_level": baseline["per_level"],
                "emitted_all_steps": baseline["emitted_all_steps"],
            }
            if baseline
            else None
        ),
        "H1_step1_anchored": bool(max(step1_gaps) <= atol and not step1_flips),
        "H2_network_produced_qhat": bool(producers == {"literal_attention_network"}),
        "H3_three_steps_in_network": bool(len(three) > 0),
        "H4_decisions_agree_with_numpy": bool(
            not decision_disagreements and not eta_disagreements
        ),
        "H5_flips_reported": True,
        "H6_validity_under_network": bool(not cert_violations and not nondegrading),
        "flip_details": flips,
    }
    analysis = {
        "task_id": TASK_ID,
        "reconstruction": "recomputed from task_results.json with strict duplicate-key detection",
        "q_hat_gaps_by_step": {
            str(level): [
                s["q_hat_gap_vs_numpy"] for _, _, _, s in steps_all if s["step"] == level
            ]
            for level in range(1, max_steps + 1)
        },
        "three_step_records": [
            {
                "mixing": r["mixing"],
                "task_index": r["task_index"],
                "route": route,
                "q_gaps": [s["q_hat_gap_vs_numpy"] for s in b["steps"]],
                "eta": [s["eta_selected"] for s in b["steps"]],
                "gain": [
                    s["oracle_audit"].get("total_value_gain")
                    for s in b["steps"]
                    if s["update_emitted"]
                ],
            }
            for r, route, b in blocks
            if b["emitted_steps"] >= 3
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
        f"{TASK_ID}: routes={len(blocks)} steps={len(steps_all)} "
        f"emissions_by_step={summary['emissions_by_step']} "
        f"three_step={len(three)}"
    )
    print(
        f"  max|dQ|={max(gaps):.3e} (ATOL {atol:.0e}) step1_max={max(step1_gaps):.3e} "
        f"flips={len(flips)} eta_flips={len(eta_disagreements)} "
        f"cert_violations={len(cert_violations)} nondeg={len(nondegrading)}"
    )
    print(
        f"  decision agreement: {len(comparisons) - len(decision_disagreements)}"
        f"/{len(comparisons)} vs FP-ITER3-001 numpy"
    )
    print(
        "  H1={} H2={} H3={} H4={} H5={} H6={}".format(
            summary["H1_step1_anchored"],
            summary["H2_network_produced_qhat"],
            summary["H3_three_steps_in_network"],
            summary["H4_decisions_agree_with_numpy"],
            summary["H5_flips_reported"],
            summary["H6_validity_under_network"],
        )
    )
    if baseline:
        print(
            f"  baseline FP-ITER3-001 per-level emissions: "
            f"{[baseline['per_level'][str(k)]['emissions'] for k in range(1, max_steps + 1)]}"
        )
        print(
            f"  network per-level emissions             : "
            f"{[emissions[k] for k in range(1, max_steps + 1)]}"
        )


if __name__ == "__main__":
    main()
