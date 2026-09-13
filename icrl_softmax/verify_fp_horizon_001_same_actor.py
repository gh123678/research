"""FP-HORIZON-001 same-actor derived verification.

Re-derives validity, soundness, the population trajectory, the stopping step and the
cumulative value gain from the bundle; confirms the sealed corpus is untouched; and
replays the foundation checks.

By the user's instruction of 2026-09-11 the verification is not the focus of this
round; it is recorded for completeness. This is DERIVED verification by the same
actor that executed the task, NOT independent verification.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
RESULT = PROJECT / "results" / "FP-HORIZON-001" / "claude" / "formal"
ITER5_ENV = PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy" / "environment.json"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("frozen", "empirical_bernstein")
SCIENCE = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "model.py",
    "verify_variance_adaptive_certificate.py",
)
REPORT: list[str] = []
FAILURES = 0


def check(ok: bool, message: str) -> bool:
    global FAILURES
    if ok:
        REPORT.append(f"  PASS  {message}")
    else:
        FAILURES += 1
        REPORT.append(f"  FAIL  {message}")
    return bool(ok)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def rows_of(bundle: dict):
    return [
        (r["mixing"], r["task_index"], route, r["routes"][route])
        for r in bundle["records"]
        for route in PRIMARY
    ]


def level_set(rows, arm: str, level: int) -> set:
    return {
        (m, t, route)
        for m, t, route, block in rows
        for s in block[arm]["steps"]
        if s["step"] == level and s["update_emitted"]
    }


def main() -> None:
    REPORT.append("FP-HORIZON-001 same-actor derived verification")
    REPORT.append("=" * 92)
    REPORT.append(
        "NOTE: the user instructed that verification is not the focus of this\n"
        "round. These checks are recorded for completeness and are same-actor only."
    )

    bundle = load(RESULT / "task_results.json")
    summary = load(RESULT / "summary.json")
    horizon = int(bundle["max_steps"])
    rows = rows_of(bundle)

    REPORT.append("\n1. Frozen inputs")
    check(len(rows) == 48, f"48 route-records (found {len(rows)})")
    check(horizon == 32, f"horizon 32 (found {horizon})")
    check(
        bundle["task_id"] == "FP-HORIZON-001",
        f"the bundle carries this task's identity (found {bundle['task_id']})",
    )
    check(
        int(bundle["cert_chains"]) == 16384 * 8,
        f"certification is still 8x (found {bundle['cert_chains']} chains)",
    )

    REPORT.append("\n2. H1/H2 re-derived: validity and soundness")
    for arm in ARMS:
        steps = [
            (m, t, route, s)
            for m, t, route, block in rows
            for s in block[arm]["steps"]
        ]
        emitted = [x for x in steps if x[3]["update_emitted"]]
        check(
            all(
                x[3]["oracle_audit"].get("componentwise_nondegrading", False)
                for x in emitted
            ),
            f"{arm}: all {len(emitted)} emitted steps are componentwise non-degrading",
        )
        check(
            all(x[3]["oracle_audit"]["total_value_gain"] > 0.0 for x in emitted),
            f"{arm}: every emitted step has a strictly positive total gain",
        )
        check(
            not any(x[3]["oracle_audit"].get("certificate_violation") for x in emitted),
            f"{arm}: zero certificate violations over {len(emitted)} emitted steps",
        )
        check(
            all(x[3]["ordered_reasons"] for x in steps if not x[3]["update_emitted"]),
            f"{arm}: every abstention carries a frozen reason",
        )

    REPORT.append("\n3. The stopping claim re-derived")
    deepest = max(
        (b[arm]["emitted_steps"] for _, _, _, b in rows for arm in ARMS), default=0
    )
    check(
        deepest == max(summary["deepest_trajectory"].values()),
        f"the deepest trajectory {deepest} matches the summary",
    )
    last = max(
        (
            s["step"]
            for _, _, _, b in rows
            for arm in ARMS
            for s in b[arm]["steps"]
            if s["update_emitted"]
        ),
        default=0,
    )
    check(
        last == summary["last_emitting_step"],
        f"the last emitting step {last} matches the summary",
    )
    check(
        (last < horizon) == (summary["H3"] == "PASS"),
        f"the H3 verdict is consistent with the data (last step {last}, "
        f"horizon {horizon})",
    )
    if last < horizon:
        silent = horizon - last
        REPORT.append(
            f"  INFO  the iteration goes silent for the final {silent} step(s), so "
            "the stop is a property of the method and not of the budget."
        )
    else:
        REPORT.append(
            "  INFO  an emission occurs at the last step, so termination has NOT "
            "been demonstrated; the horizon is still binding."
        )

    REPORT.append("\n4. The population is re-derived")
    for arm in ARMS:
        seq = [len(level_set(rows, arm, lv)) for lv in range(1, horizon + 1)]
        check(
            seq == summary["emissions"][arm],
            f"{arm}: the emission sequence matches the summary",
        )
        rises = [
            (lv, seq[lv - 2], seq[lv - 1])
            for lv in range(2, horizon + 1)
            if seq[lv - 1] > seq[lv - 2]
        ]
        REPORT.append(
            f"  INFO  {arm}: {len(rises)} rises in the population"
            + (f", first at step {rises[0][0]}" if rises else "")
        )

    REPORT.append("\n5. The value gain is re-derived from the recorded values")
    # Independent of the analyzer: recompute the cumulative gain from start and
    # final per-state values rather than from the per-step gains.
    mismatches = []
    for mixing, task_index, route, block in rows:
        for arm in ARMS:
            start = block[arm]["start_policy_value"]
            final = block[arm]["final_policy_value"]
            delta = [f - s for s, f in zip(start, final)]
            if min(delta) < -1e-12:
                mismatches.append((mixing, task_index, route, arm, min(delta)))
    check(
        not mismatches,
        f"start-to-final per-state value never decreases on any record "
        f"({len(mismatches)} exceptions)",
    )
    for item in mismatches[:5]:
        REPORT.append(f"        decrease: {item}")
    totals = [
        sum(f - s for s, f in zip(b[arm]["start_policy_value"],
                                  b[arm]["final_policy_value"]))
        for _, _, _, b in rows
        for arm in ARMS
    ]
    # A trajectory that abstains at step 1 legitimately gains exactly zero, so the
    # property is non-negativity plus a correspondence: the zero-gain trajectories
    # must be exactly the ones that never emitted. An earlier version of this check
    # demanded a strictly positive gain everywhere and failed on that legitimate
    # case.
    check(
        all(t >= 0.0 for t in totals),
        f"no trajectory loses total value ({len(totals)} trajectories, "
        f"min {min(totals):.6f})",
    )
    zero_gain = sum(1 for t in totals if t == 0.0)
    never_emitted = sum(
        1
        for _, _, _, b in rows
        for arm in ARMS
        if b[arm]["emitted_steps"] == 0
    )
    check(
        zero_gain == never_emitted,
        f"the {zero_gain} zero-gain trajectories are exactly the {never_emitted} "
        f"that never emitted",
    )
    check(
        sum(1 for t in totals if t > 0.0) > 0,
        f"{sum(1 for t in totals if t > 0.0)} trajectories gain strictly positive "
        f"total value",
    )

    REPORT.append("\n6. Sealed corpus untouched")
    recorded = load(ITER5_ENV)["sealed_file_hashes"]
    for name in SCIENCE:
        if name in recorded:
            check(
                recorded[name] == sha256(PROJECT / name),
                f"SCIENCE {name} is byte-identical to the FP-ITER5-001 record",
            )

    REPORT.append("\n7. Analyzer determinism")
    before = (RESULT / "summary.json").read_bytes()
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "analyze_fp_horizon_001.py"),
         "--result-dir", str(RESULT)],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(
        completed.returncode == 0,
        f"analyze_fp_horizon_001.py replays with exit 0 (got {completed.returncode})",
    )
    check(
        before == (RESULT / "summary.json").read_bytes(),
        "re-running the analyzer reproduces summary.json byte-for-byte",
    )

    REPORT.append("\n8. Replay of the foundation checks")
    for script in (
        "verify_variance_adaptive_certificate.py",
        "verify_policy_quantities_by_solve.py",
    ):
        completed = subprocess.run(
            [sys.executable, "-B", str(PROJECT / script)],
            capture_output=True,
            text=True,
            cwd=PROJECT,
        )
        check(
            completed.returncode == 0,
            f"{script} replays with exit 0 (got {completed.returncode})",
        )

    REPORT.append("\n" + "=" * 92)
    REPORT.append("SUMMARY")
    REPORT.append(f"  emissions, frozen : {summary['emissions']['frozen']}")
    REPORT.append(f"  emissions, empB   : {summary['emissions']['empirical_bernstein']}")
    REPORT.append(f"  deepest trajectory: {summary['deepest_trajectory']}")
    REPORT.append(f"  last emitting step: {summary['last_emitting_step']}")
    REPORT.append(f"  H3 terminates     : {summary['H3']}")
    REPORT.append(f"  H5 decay past 12  : {summary['H5']}")
    REPORT.append(f"  H6 positive gains : {summary['H6']}")
    for arm in ARMS:
        v = summary["value"][arm]
        REPORT.append(
            f"  total value gain, {arm:<20}: mean {v['total_gain_mean']:.6f}, "
            f"worst state mean {v['worst_state_gain_mean']:.6f}"
        )
    REPORT.append("=" * 92)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round. The 8x batch is a fresh independent\n"
        "sample, not a superset of the sealed one."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT / "docs" / "research_branches" / "FP-HORIZON-001" / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
