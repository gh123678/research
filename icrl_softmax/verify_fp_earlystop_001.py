"""FP-EARLYSTOP-001 same-actor check: recompute the headline table from the
sealed task_results.json and confirm the safety invariants.

V1  per-arm emitted-step totals and step-1 emitting counts match summary.json;
V2  every emitted step has a complete oracle audit; certificate violations == 0;
    componentwise degradation == 0 (H4 -- the per-state arm's safety check);
V3  fraction_gap_closed is consistent with initial/final v sums and the sealed
    initial suboptimality for every record;
V4  trajectories stop after their first abstention (program invariant check);
V5  the paired headline numbers (C>A count, C>B count, rescued records) are
    recomputed here, not copied from the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.results / "task_results.json").read_text(encoding="utf-8"))
    summary = json.loads((args.results / "summary.json").read_text(encoding="utf-8"))
    arms = data["arms"]
    routes = data["routes"]
    failures: list[str] = []

    for arm in arms:
        total = step1 = 0
        for rec in data["records"]:
            for route in routes:
                out = rec["routes"][route][arm]
                flags = [bool(s["emitted"]) for s in out["steps"]]
                total += sum(flags)
                if flags and flags[0]:
                    step1 += 1
                if int(out["emitted_steps"]) != sum(flags):
                    failures.append(f"emitted_steps mismatch {rec['task_index']}/{route}/{arm}")
                if any(flags[i] for i in range(1, len(flags)) if not flags[i - 1]):
                    failures.append(f"emission after abstention {rec['task_index']}/{route}/{arm}")
                # V3
                init = float(out["initial_v_sum"])
                fin = float(out["final_v_sum"])
                gap0 = float(out["initial_suboptimality"])
                fc = float(out["fraction_gap_closed"])
                expect = (fin - init) / gap0 if gap0 > 0 else 1.0
                if abs(fc - expect) > 1e-9 * max(1.0, abs(expect)):
                    failures.append(f"fraction_gap_closed inconsistent {rec['task_index']}/{route}/{arm}")
                for s in out["steps"]:
                    if s["oracle_audit"]["certificate_violation"]:
                        failures.append(f"violation {rec['task_index']}/{route}/{arm} step {s['step']}")
                    if s["emitted"]:
                        if "total_value_gain" not in s["oracle_audit"]:
                            failures.append(f"incomplete audit {rec['task_index']}/{route}/{arm} step {s['step']}")
                        if not s["oracle_audit"].get("componentwise_nondegrading", False):
                            failures.append(f"DEGRADING {rec['task_index']}/{route}/{arm} step {s['step']}")
        if total != int(summary["per_arm"][arm]["emitted_steps"]):
            failures.append(f"arm {arm} total != summary")
        if step1 != int(summary["per_arm"][arm]["records_emitting_step1"]):
            failures.append(f"arm {arm} step1 != summary")

    # V5 paired headline numbers
    wins_ca = wins_cb = losses = rescued = zero_a = 0
    for rec in data["records"]:
        for route in routes:
            g = {
                arm: rec["routes"][route][arm]["final_v_sum"]
                - rec["routes"][route][arm]["initial_v_sum"]
                for arm in arms
            }
            if g["perstate_n16k"] > g["conj_n16k"] + 1e-12:
                wins_ca += 1
            if g["perstate_n16k"] > g["conj_n64k"] + 1e-12:
                wins_cb += 1
            if g["perstate_n16k"] < g["conj_n16k"] - 1e-12:
                losses += 1
            if rec["routes"][route]["conj_n16k"]["emitted_steps"] == 0:
                zero_a += 1
                if rec["routes"][route]["perstate_n16k"]["emitted_steps"] > 0:
                    rescued += 1

    out = {
        "results_dir": str(args.results),
        "V1_V2_V3_V4_failures": failures,
        "V5_paired": {
            "C_beats_A": wins_ca,
            "C_beats_B": wins_cb,
            "C_below_A": losses,
            "A_zero_emission_records": zero_a,
            "rescued_by_C": rescued,
        },
        "status": "PASS" if not failures else "FAIL",
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
