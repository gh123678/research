"""FP-XFAM-001 same-actor check: recompute the headline numbers per family.

X1  per-arm emitted totals / step-1 counts match summary.json;
X2  zero certificate violations and zero componentwise degradations (H3);
X3  fraction_gap_closed consistent with sealed v-sums;
X4  trajectories stop after first abstention;
X5  paired headline (perstate >= conj win/loss counts) recomputed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_dir(root: Path) -> dict:
    data = json.loads((root / "task_results.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    arms = data["arms"]
    routes = data["routes"]
    failures: list[str] = []
    wins = losses = 0
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
                    failures.append(f"count mismatch {rec['task_index']}/{route}/{arm}")
                if any(flags[i] for i in range(1, len(flags)) if not flags[i - 1]):
                    failures.append(f"emission after abstention {rec['task_index']}/{route}/{arm}")
                init, fin = float(out["initial_v_sum"]), float(out["final_v_sum"])
                gap0 = float(out["initial_suboptimality"])
                fc = float(out["fraction_gap_closed"])
                expect = (fin - init) / gap0 if gap0 > 0 else 1.0
                if abs(fc - expect) > 1e-9 * max(1.0, abs(expect)):
                    failures.append(f"fraction inconsistent {rec['task_index']}/{route}/{arm}")
                for s in out["steps"]:
                    if s["oracle_audit"]["certificate_violation"]:
                        failures.append(f"violation {rec['task_index']}/{route}/{arm}")
                    if s["emitted"] and not s["oracle_audit"].get(
                        "componentwise_nondegrading", False
                    ):
                        failures.append(f"DEGRADING {rec['task_index']}/{route}/{arm}")
        if total != int(summary["per_arm"][arm]["emitted_steps"]):
            failures.append(f"arm {arm} total != summary")
        if step1 != int(summary["per_arm"][arm]["records_emitting_step1"]):
            failures.append(f"arm {arm} step1 != summary")
    for rec in data["records"]:
        for route in routes:
            g = {
                arm: rec["routes"][route][arm]["final_v_sum"]
                - rec["routes"][route][arm]["initial_v_sum"]
                for arm in arms
            }
            if g["perstate"] > g["conj"] + 1e-12:
                wins += 1
            if g["perstate"] < g["conj"] - 1e-12:
                losses += 1
    return {
        "results_dir": str(root),
        "family": data.get("family"),
        "failures": failures,
        "paired_perstate_wins": wins,
        "paired_perstate_losses": losses,
        "records": data["record_count"],
        "status": "PASS" if not failures else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    outs = [check_dir(r) for r in args.results]
    print(json.dumps(outs, indent=2, sort_keys=True))
    return 0 if all(o["status"] == "PASS" for o in outs) else 1


if __name__ == "__main__":
    sys.exit(main())
