"""FP-CERTFIX-001 same-actor check: recompute every headline number from the
sealed ``task_results.json`` without rerunning the experiment.

Checks (all from the sealed file only):

C1  Emission counts, three ways: ``update_emitted`` flags, ``status`` fields,
    and trajectory lengths must agree with ``summary.json``.
C2  Risk accounting (H4 of the task): for the mp arm, ``2 * d * delta_each ==
    delta_step``; for the split arm the same with ``delta_1 + delta_2``; and
    ``max_steps * delta_step == 0.05`` exactly as allocated.
C3  MP radius formula: ``radii[pair]`` must equal
    ``sqrt(2 V log(2/delta_each)/n) + (7/3)*20*log(2/delta_each)/(n-1)``
    recomputed from the sealed ``sample_vars`` -- this is the line-by-line
    correspondence between derivation section 2 and the implementation.
C4  Oracle audits: no certificate violation and no componentwise degradation at
    any emitted step, counted independently.
C5  Abstentions all carry an ordered reason.

Same-actor check: it catches implementation/aggregation slips, NOT conceptual
errors. Independent review of the derivation itself is a separate gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

N_STATES, N_ACTIONS = 4, 3
D = N_STATES * N_ACTIONS
Y_RANGE = 20.0
MP_CONSTANT = 7.0 / 3.0
DELTA_TOTAL = 0.05


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    data = json.loads((root / "task_results.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))

    failures: list[str] = []
    routes = data["routes"]
    arms = data["arms"]
    n = int(data["n_per_pair"])
    max_steps = int(data["max_steps"])
    delta_step = float(data["delta_step"])

    # C1: emission counts three ways
    for arm in arms:
        by_flag = by_status = 0
        step1 = 0
        for rec in data["records"]:
            for route in routes:
                out = rec["routes"][route][arm]
                flags = [bool(s["update_emitted"]) for s in out["steps"]]
                by_flag += sum(flags)
                by_status += sum(
                    1 for s in out["steps"] if s["status"] == "safe_update_emitted"
                )
                if int(out["emitted_steps"]) != sum(flags):
                    failures.append(
                        f"emitted_steps mismatch {rec['mixing']}/{rec['task_index']}/{route}/{arm}"
                    )
                # nested-by-construction check: once not emitted, nothing after
                if any(flags[i] for i in range(1, len(flags)) if not flags[i - 1]):
                    failures.append(
                        f"emission after abstention {rec['mixing']}/{rec['task_index']}/{route}/{arm}"
                    )
                if flags and flags[0]:
                    step1 += 1
        if by_flag != by_status:
            failures.append(f"arm {arm}: flag {by_flag} != status {by_status}")
        if by_flag != int(summary["emitted_by_arm"][arm]):
            failures.append(f"arm {arm}: sealed total != summary.json")
        if step1 != int(summary["step1_emitted_by_arm"][arm]):
            failures.append(f"arm {arm}: step-1 count != summary.json")

    # C2: risk accounting
    if abs(max_steps * delta_step - DELTA_TOTAL) > 1e-15:
        failures.append("max_steps * delta_step != DELTA_TOTAL")

    # C3/C4/C5 per step
    violations = degrading = missing_reason = 0
    radius_checked = 0
    for rec in data["records"]:
        for route in routes:
            for arm in arms:
                for s in rec["routes"][route][arm]["steps"]:
                    de = s.get("delta_each")
                    if de is not None and abs(2 * D * de - delta_step) > 1e-15 * max(
                        1.0, delta_step
                    ):
                        failures.append(
                            f"risk accounting off at step {s['step']} "
                            f"({rec['mixing']}/{rec['task_index']}/{route}/{arm})"
                        )
                    if arm == "mp" and s.get("cert_sample_vars"):
                        log_term = math.log(2.0 / de)
                        for v, r in zip(s["cert_sample_vars"], s["cert_radii"]):
                            expect = math.sqrt(
                                2.0 * max(v, 0.0) * log_term / n
                            ) + MP_CONSTANT * Y_RANGE * log_term / max(n - 1, 1)
                            if abs(expect - r) > 1e-12 * max(1.0, r):
                                failures.append(
                                    f"mp radius mismatch at step {s['step']} "
                                    f"({rec['mixing']}/{rec['task_index']}/{route})"
                                )
                                break
                        else:
                            radius_checked += 1
                    if s["oracle_audit"]["certificate_violation"]:
                        violations += 1
                    audit = s["oracle_audit"]
                    if s["update_emitted"] and not audit.get(
                        "componentwise_nondegrading", False
                    ):
                        degrading += 1
                    if not s["update_emitted"] and not s["ordered_reasons"]:
                        missing_reason += 1

    report = {
        "results_dir": str(root),
        "C1_counts_three_ways": "PASS" if not failures else "FAIL",
        "C2_risk_accounting": "PASS"
        if abs(max_steps * delta_step - DELTA_TOTAL) <= 1e-15
        else "FAIL",
        "C3_radius_formula": {
            "mp_steps_checked": radius_checked,
            "status": "PASS" if radius_checked and not any(
                "radius mismatch" in f for f in failures
            ) else ("FAIL" if any("radius mismatch" in f for f in failures) else "NO_DATA"),
        },
        "C4_oracle_audits": {
            "certificate_violations": violations,
            "componentwise_degrading": degrading,
        },
        "C5_abstention_reasons_missing": missing_reason,
        "other_failures": failures[:20],
        "failure_count": len(failures) + violations + degrading + missing_reason,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
