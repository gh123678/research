"""FP-ATTN-ITER4-001 analyzer: four-step network iteration vs the numpy route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TASK_ID = "FP-ATTN-ITER4-001"
PRIMARY = ("expected_exact", "expected_finite")
ITER4_SUMMARY = (
    Path(__file__).resolve().parent
    / "results"
    / "FP-ITER4-001"
    / "claude"
    / "formal"
    / "summary.json"
)
ATTN_ITER_BUNDLE = (
    Path(__file__).resolve().parent
    / "results"
    / "FP-ATTN-ITER-001"
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
    atol = float(bundle["atol"])
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
        gaps = [s["q_hat_gap_vs_numpy"] for _, _, _, s in steps if s["step"] == level]
        per_level[level] = {
            "emissions": len(gains),
            "mean_gain": (sum(gains) / len(gains)) if gains else None,
            "min_gain": min(gains) if gains else None,
            "max_q_gap": max(gaps) if gaps else None,
        }

    # H1/H2 against the sealed network bundle.
    compared = 0
    h1_mismatches: list[tuple] = []
    if ATTN_ITER_BUNDLE.exists():
        sealed = json.loads(ATTN_ITER_BUNDLE.read_text(encoding="utf-8"))
        sealed_by_key = {
            (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
        }
        for record, route, block in blocks:
            ref = sealed_by_key.get((float(record["mixing"]), int(record["task_index"])))
            if ref is None:
                continue
            ref_steps = ref["routes"][route]["steps"]
            for index in range(min(3, len(block["steps"]), len(ref_steps))):
                compared += 1
                a, b = block["steps"][index], ref_steps[index]
                if (
                    a["update_emitted"] != b["update_emitted"]
                    or a["eta_selected"] != b["eta_selected"]
                    or a["e_q"] != b["e_q"]
                    or a["q_hat_gap_vs_numpy"] != b["q_hat_gap_vs_numpy"]
                ):
                    h1_mismatches.append(
                        (record["mixing"], record["task_index"], route, index + 1)
                    )
    sealed_three = None
    if ATTN_ITER_BUNDLE.exists():
        sealed_three = sum(
            1
            for r in json.loads(ATTN_ITER_BUNDLE.read_text(encoding="utf-8"))["records"]
            for route in PRIMARY
            if r["routes"][route]["emitted_steps"] >= 3
        )

    producers = {s["qhat_producer"] for _, _, _, s in steps}
    flips = [
        {
            "mixing": r["mixing"],
            "task_index": r["task_index"],
            "route": route,
            "step": s["step"],
            "literal_status": s["status"],
            "numpy_status": s["numpy"]["status"],
            "literal_e_q": s["e_q"],
            "numpy_e_q": s["numpy"]["e_q"],
            "literal_min_lb": s["min_lb"],
            "numpy_min_lb": s["numpy"]["min_lb"],
            "q_hat_gap": s["q_hat_gap_vs_numpy"],
        }
        for r, route, _, s in steps
        if s["decision_flip_vs_numpy"]
    ]
    eta_flips = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s["eta_flip_vs_numpy"]
    ]
    cert_violations = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s["oracle_audit"].get("certificate_violation")
    ]
    nondegrading = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s["update_emitted"]
        and not s["oracle_audit"].get("componentwise_nondegrading", False)
    ]
    comparisons = [
        (r, route, c) for r, route, b in blocks for c in b["reference_comparisons"]
    ]
    disagreements = [c for c in comparisons if not c[2]["decision_match"]]

    gaps = [s["q_hat_gap_vs_numpy"] for _, _, _, s in steps]
    reasons: dict[str, int] = {}
    for _, _, _, s in steps:
        if s["update_emitted"]:
            continue
        for reason in s["ordered_reasons"] or ["<none>"]:
            reasons[reason] = reasons.get(reason, 0) + 1

    baseline = (
        json.loads(ITER4_SUMMARY.read_text(encoding="utf-8"))
        if ITER4_SUMMARY.exists()
        else None
    )

    summary = {
        "task_id": TASK_ID,
        "label": bundle.get("label"),
        "record_count": len(records),
        "route_records": len(blocks),
        "max_steps": max_steps,
        "atol": atol,
        "steps_executed": len(steps),
        "per_level": {str(k): v for k, v in per_level.items()},
        "network_four_step_routes": sum(
            1 for _, _, b in blocks if b["emitted_steps"] >= max_steps
        ),
        "producers": sorted(producers),
        "max_q_hat_gap": max(gaps),
        "decision_flips": len(flips),
        "eta_flips": len(eta_flips),
        "decision_disagreements": len(disagreements),
        "step_comparisons": len(comparisons),
        "certificate_violations": len(cert_violations),
        "nondegrading_violations": len(nondegrading),
        "abstention_reasons": reasons,
        "H1_horizon_inert": {
            "compared_entries": compared,
            "mismatches": len(h1_mismatches),
            "outcome": "PASS" if (compared > 0 and not h1_mismatches) else "FAIL",
        },
        "H2_three_step_reproduced": {
            "sealed": sealed_three,
            "this_run": sum(1 for _, _, b in blocks if b["emitted_steps"] >= 3),
            "outcome": (
                "PASS"
                if sealed_three is not None
                and sum(1 for _, _, b in blocks if b["emitted_steps"] >= 3) == sealed_three
                else "FAIL"
            ),
        },
        "H3_network_produced_qhat": bool(producers == {"literal_attention_network"}),
        "H4_fourth_step_in_network": bool(per_level.get(4, {}).get("emissions", 0) > 0),
        "H5_decisions_agree": bool(not disagreements and not eta_flips),
        "H6_no_accumulated_drift": {
            "max_step4_q_gap": per_level.get(4, {}).get("max_q_gap"),
            "atol": atol,
            "outcome": (
                "PASS"
                if (per_level.get(4, {}).get("max_q_gap") or 1e9) <= atol
                else "FALSIFIED"
            ),
        },
        "H7_flips_reported": True,
        "H8_validity_under_network": bool(not cert_violations and not nondegrading),
        "baseline_iter4_numpy": (
            {"per_level": baseline["per_level"]} if baseline else None
        ),
        "flip_details": flips,
    }
    analysis = {
        "task_id": TASK_ID,
        "reconstruction": "recomputed from task_results.json with strict duplicate-key detection",
        "q_hat_gaps_by_step": {
            str(level): [
                s["q_hat_gap_vs_numpy"] for _, _, _, s in steps if s["step"] == level
            ]
            for level in range(1, max_steps + 1)
        },
        "four_step_records": [
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
            if b["emitted_steps"] >= max_steps
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
            f"mean_gain={row['mean_gain']} min_gain={row['min_gain']} "
            f"max|dQ|={row['max_q_gap']:.3e}"
        )
    print(f"  network four-step routes={summary['network_four_step_routes']}")
    print(
        f"  decision agreement {len(comparisons) - len(disagreements)}/{len(comparisons)} "
        f"flips={len(flips)} eta_flips={len(eta_flips)} "
        f"cert_violations={len(cert_violations)} nondeg={len(nondegrading)}"
    )
    for key in (
        "H1_horizon_inert",
        "H2_three_step_reproduced",
        "H3_network_produced_qhat",
        "H4_fourth_step_in_network",
        "H5_decisions_agree",
        "H6_no_accumulated_drift",
        "H8_validity_under_network",
    ):
        entry = summary[key]
        outcome = entry.get("outcome") if isinstance(entry, dict) else entry
        print(f"  {key}: {outcome}")
    if baseline:
        print(
            f"  numpy baseline (FP-ITER4-001) per-level emissions: "
            f"{[baseline['per_level'][str(k)]['emissions'] for k in range(1, 5)]}"
        )
        print(
            f"  network per-level emissions                       : "
            f"{[per_level[k]['emissions'] for k in range(1, 5)]}"
        )


if __name__ == "__main__":
    main()
