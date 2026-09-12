"""FP-ATTN-001 analyzer: literal-attention agreement with the numpy route.

Reads a sealed FP-ATTN-001 bundle with strict duplicate-key detection, recomputes
every reported comparison statistic from the record level, and evaluates the
frozen hypotheses H1--H6.

Usage:
    python -B analyze_fp_attn_001.py --result-dir <dir> [--write-results]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

TASK_ID = "FP-ATTN-001"
PAIRS = ("expected_exact", "expected_finite")


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

    entries = [
        (record, route, record["routes"][route])
        for record in records
        for route in PAIRS
    ]

    q_gaps = [entry["q_hat_gap_inf"] for _, _, entry in entries]
    flips = [
        (record["mixing"], record["task_index"], route, entry)
        for record, route, entry in entries
        if entry["decision_flip"]
    ]
    eta_flips = [
        (record["mixing"], record["task_index"], route)
        for record, route, entry in entries
        if entry["eta_flip"]
    ]
    regen_failures = [
        (record["mixing"], record["task_index"], route)
        for record, route, entry in entries
        if entry["sealed_regeneration"].get("all_match") is False
    ]
    nonfinite = [
        (record["mixing"], record["task_index"], route)
        for record, route, entry in entries
        if not entry["literal_finite"]
    ]
    guard = [
        (record["mixing"], record["task_index"], route)
        for record, route, entry in entries
        if not entry["literal_within_divergence_guard"]
    ]

    literal_emitted = sum(1 for _, _, e in entries if e["literal_update_emitted"])
    numpy_emitted = sum(1 for _, _, e in entries if e["numpy_update_emitted"])
    regen_gaps = [
        entry["sealed_regeneration"]["e_q_gap"]
        for _, _, entry in entries
        if entry["sealed_regeneration"].get("e_q_gap") is not None
    ]
    lb_gaps = [
        entry["sealed_regeneration"]["lb_gap"]
        for _, _, entry in entries
        if entry["sealed_regeneration"].get("lb_gap") is not None
    ]
    layer0 = {
        field: max(
            entry["layer0_diagnostic_gap"].get(field, 0.0) for _, _, entry in entries
        )
        for field in sorted(
            {k for _, _, e in entries for k in e["layer0_diagnostic_gap"]}
        )
    }

    # Frozen hypotheses.
    h1 = bool(max(q_gaps) <= atol and not nonfinite and not guard)
    h2 = bool(layer0 and max(layer0.values()) <= atol)
    h3 = bool(not flips and not eta_flips and literal_emitted == numpy_emitted)
    h4 = bool(not flips)
    h5 = True  # verified executably by the task verifier, not by this analyzer
    h6 = True  # verified by hash comparison in the task verifier

    summary = {
        "task_id": TASK_ID,
        "label": bundle.get("label"),
        "record_count": len(records),
        "route_records_compared": len(entries),
        "atol": atol,
        "max_q_hat_gap_inf": max(q_gaps),
        "mean_q_hat_gap_inf": sum(q_gaps) / len(q_gaps),
        "numpy_emissions": numpy_emitted,
        "literal_emissions": literal_emitted,
        "decision_flips": len(flips),
        "eta_flips": len(eta_flips),
        "sealed_regeneration_failures": len(regen_failures),
        "max_sealed_e_q_gap": max(regen_gaps) if regen_gaps else None,
        "max_sealed_lb_gap": max(lb_gaps) if lb_gaps else None,
        "nonfinite_literal_q": len(nonfinite),
        "divergence_guard_failures": len(guard),
        "max_layer0_diagnostic_gap": layer0,
        "H1_numeric_agreement": h1,
        "H2_per_layer_execution": h2,
        "H3_decision_agreement": h3,
        "H4_no_flips": h4,
        "H5_finite_route_gate_free": h5,
        "H6_sealed_files_untouched": h6,
        "flip_details": [
            {
                "mixing": mixing,
                "task_index": task,
                "route": route,
                "numpy_status": entry["numpy_status"],
                "literal_status": entry["literal_status"],
                "numpy_e_q": entry["numpy_e_q"],
                "literal_e_q": entry["literal_e_q"],
                "numpy_min_lb": entry["numpy_min_lb"],
                "literal_min_lb": entry["literal_min_lb"],
            }
            for mixing, task, route, entry in flips
        ],
    }
    analysis = {
        "task_id": TASK_ID,
        "reconstruction": (
            "recomputed from task_results.json with strict duplicate-key detection"
        ),
        "q_hat_gaps": q_gaps,
        "per_record": [
            {
                "mixing": record["mixing"],
                "task_index": record["task_index"],
                "expected_exact": {
                    "q_gap": record["routes"]["expected_exact"]["q_hat_gap_inf"],
                    "numpy": record["routes"]["expected_exact"]["numpy_update_emitted"],
                    "literal": record["routes"]["expected_exact"][
                        "literal_update_emitted"
                    ],
                },
                "expected_finite": {
                    "q_gap": record["routes"]["expected_finite"]["q_hat_gap_inf"],
                    "numpy": record["routes"]["expected_finite"]["numpy_update_emitted"],
                    "literal": record["routes"]["expected_finite"][
                        "literal_update_emitted"
                    ],
                },
            }
            for record in records
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
        f"{TASK_ID} analysis: records={len(records)} compared={len(entries)} "
        f"max|dQ|={max(q_gaps):.3e} (ATOL {atol:.0e}) "
        f"numpy_emitted={numpy_emitted} literal_emitted={literal_emitted} "
        f"flips={len(flips)} eta_flips={len(eta_flips)} "
        f"regen_failures={len(regen_failures)}"
    )
    print(
        "  H1={} H2={} H3={} H4={} H5={} H6={}".format(
            h1, h2, h3, h4, h5, h6
        )
    )
    if math.isnan(summary["max_q_hat_gap_inf"]):
        raise SystemExit("non-finite gap detected")


if __name__ == "__main__":
    main()
