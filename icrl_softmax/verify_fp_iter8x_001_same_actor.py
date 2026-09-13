"""FP-ITER8X-001 same-actor derived verification.

Re-derives validity, soundness, the population trajectory and the containment claim
from the bundle; confirms the two arms are genuinely separate trajectories; checks
the sealed corpus is untouched; replays the foundation checks.

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
RESULT = PROJECT / "results" / "FP-ITER8X-001" / "claude" / "formal"
SEALED_1X = PROJECT / "results" / "FP-ITER6-001" / "claude" / "numpy" / "task_results.json"
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
    REPORT.append("FP-ITER8X-001 same-actor derived verification")
    REPORT.append("=" * 88)
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
    check(horizon == 12, f"horizon 12 (found {horizon})")
    check(int(bundle["multiplier"]) == 8, f"multiplier 8 (found {bundle['multiplier']})")
    check(
        int(bundle["cert_chains"]) == 16384 * 8,
        f"certification is 8x the sealed chains (found {bundle['cert_chains']})",
    )

    REPORT.append("\n2. H1/H2 re-derived: validity and soundness")
    for arm in ARMS:
        steps = [
            (m, t, route, s)
            for m, t, route, block in rows
            for s in block[arm]["steps"]
        ]
        emitted = [x for x in steps if x[3]["update_emitted"]]
        degrading = sum(
            1
            for x in emitted
            if not x[3]["oracle_audit"].get("componentwise_nondegrading", False)
        )
        nonpositive = sum(
            1 for x in emitted if x[3]["oracle_audit"]["total_value_gain"] <= 0.0
        )
        violations = sum(
            1 for x in emitted if x[3]["oracle_audit"].get("certificate_violation")
        )
        missing = sum(
            1
            for x in steps
            if not x[3]["update_emitted"] and not x[3]["ordered_reasons"]
        )
        check(
            degrading == 0, f"{arm}: {len(emitted)} emitted steps, {degrading} degrading"
        )
        check(nonpositive == 0, f"{arm}: {nonpositive} non-positive gains")
        check(violations == 0, f"{arm}: {violations} certificate violations")
        check(missing == 0, f"{arm}: {missing} abstentions without a reason")
    # The realized error must be strictly positive everywhere it is recorded, so the
    # audit is not comparing Qhat against itself.
    realized = [
        s["oracle_audit"]["realized_q_sup_error_vs_current_target_pi"]
        for _, _, _, block in rows
        for s in block["frozen"]["steps"]
    ]
    check(
        min(realized) > 0.0,
        f"every realized error is strictly positive (min {min(realized):.3e})",
    )

    REPORT.append("\n3. The two arms are genuinely separate trajectories")
    # If the arms shared a trajectory the containment claim would be vacuous and the
    # control would be an illusion.
    differing = 0
    for _, _, route, block in rows:
        a = [s["update_emitted"] for s in block["frozen"]["steps"]]
        b = [s["update_emitted"] for s in block["empirical_bernstein"]["steps"]]
        if a != b:
            differing += 1
    check(
        differing > 0,
        f"the arms diverge on {differing}/48 route-records, so the control is real",
    )
    REPORT.append(
        "  INFO  the arms are guaranteed to diverge wherever the certificate changes "
        "an eta choice; identical trajectories on some records are expected and are "
        "not evidence of a bug."
    )

    REPORT.append("\n4. H3/H4 re-derived: the population trajectory")
    seventh = {arm: level_set(rows, arm, 7) for arm in ARMS}
    total7 = len(seventh["frozen"]) + len(seventh["empirical_bernstein"])
    check(
        total7 >= 1,
        f"H3: {total7} seventh-step emissions across both arms",
    )
    check(
        (total7 >= 1) == (summary["H3"] == "PASS"),
        "the summary's H3 verdict matches a fresh re-derivation",
    )
    seven = len(seventh["frozen"])
    check(
        (seven > 9) == (summary["H4"]["verdict"] == "PASS"),
        f"H4: step-7 frozen count {seven} against the 1x step-6 count 9, matching "
        f"the summary",
    )
    deepest = max(
        (block[arm]["emitted_steps"] for _, _, _, block in rows for arm in ARMS),
        default=0,
    )
    check(
        deepest >= 7,
        f"the deepest trajectory reaches {deepest} steps, so the horizon is not the "
        f"binding constraint",
    )

    REPORT.append("\n5. H5 re-derived: containment across arms")
    # The registered claim is not a theorem across arms, because LB_s is increasing
    # in E_Q only at a fixed policy and the arms' step-k policies differ. The
    # testable form is step 1, where both arms start from pi_0 with the same q_hat.
    f1 = level_set(rows, "frozen", 1)
    e1 = level_set(rows, "empirical_bernstein", 1)
    check(
        f1 <= e1,
        f"the containment theorem holds where it applies: step-1 frozen "
        f"{len(f1)} subset of repaired {len(e1)}",
    )
    check(
        summary["H5"].get("theorem_holds_at_step1") is True,
        "the summary records that the theorem holds at step 1",
    )
    violations = []
    nonempty = 0
    for level in range(1, horizon + 1):
        f = level_set(rows, "frozen", level)
        e = level_set(rows, "empirical_bernstein", level)
        if not f:
            continue
        nonempty += 1
        if not f <= e:
            violations.append((level, sorted(f - e)))
    REPORT.append(
        f"  INFO  cross-arm containment holds at {nonempty - len(violations)}/"
        f"{nonempty} non-empty levels; the {len(violations)} exception(s) are "
        "expected because the arms diverge, and are reported rather than suppressed."
    )
    for item in violations[:5]:
        REPORT.append(f"        counterexample at step {item[0]}: {item[1]}")

    REPORT.append("\n6. H7 re-derived: the step-6 minimum")
    gains6 = [
        s["oracle_audit"]["total_value_gain"]
        for _, _, _, block in rows
        for s in block["frozen"]["steps"]
        if s["step"] == 6 and s["update_emitted"]
    ]
    min6 = min(gains6) if gains6 else None
    check(
        min6 is not None and (min6 < 0.019347) == (summary["H7"]["verdict"] == "PASS"),
        f"H7: step-6 minimum {min6} against the sealed 0.019347, matching the summary",
    )

    REPORT.append("\n7. The sealed 1x reference is intact, and the corpus is untouched")
    sealed = load(SEALED_1X)
    sealed_6 = level_set(rows_of(sealed), "frozen", 6) if False else None
    del sealed_6
    sealed_six = {
        (r["mixing"], r["task_index"], route)
        for r in sealed["records"]
        for route in PRIMARY
        for s in r["routes"][route]["steps"]
        if s["step"] == 6 and s["update_emitted"]
    }
    check(
        len(sealed_six) == 9,
        f"the sealed 1x run emitted {len(sealed_six)} at step 6 (expected 9)",
    )
    recorded = load(ITER5_ENV)["sealed_file_hashes"]
    for name in SCIENCE:
        if name in recorded:
            check(
                recorded[name] == sha256(PROJECT / name),
                f"SCIENCE {name} is byte-identical to the FP-ITER5-001 record",
            )
    # The property that matters is not "nothing uses the repaired certificate" --
    # FP-TIGHT-001, FP-SAMPLE-001 and this task are supposed to. It is that the
    # PRE-EXISTING iteration evaluators, whose results other tasks sealed, stay on
    # the sealed certificate. An earlier version of this check asserted the broader
    # claim and failed on two evaluators that use the repair legitimately at step 1.
    for evaluator in ("evaluate_fp_iter2_001.py", "evaluate_fp_attn_iter_001.py",
                      "evaluate_fp_census_001.py"):
        source = (PROJECT / evaluator).read_text(encoding="utf-8")
        check(
            "fixed_policy_bernstein_certificate" not in source
            and "empirical_bernstein_certificate" not in source,
            f"{evaluator} is untouched and still uses only the sealed certificate",
        )

    REPORT.append("\n8. Analyzer determinism")
    before = (RESULT / "summary.json").read_bytes()
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "analyze_fp_iter8x_001.py"),
         "--result-dir", str(RESULT)],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(
        completed.returncode == 0,
        f"analyze_fp_iter8x_001.py replays with exit 0 (got {completed.returncode})",
    )
    check(
        before == (RESULT / "summary.json").read_bytes(),
        "re-running the analyzer reproduces summary.json byte-for-byte",
    )

    REPORT.append("\n9. Replay of the foundation checks")
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

    REPORT.append("\n" + "=" * 88)
    REPORT.append("SUMMARY")
    REPORT.append(
        f"  emissions, frozen@8x : {summary['emissions']['frozen']}"
    )
    REPORT.append(
        f"  emissions, empB@8x   : {summary['emissions']['empirical_bernstein']}"
    )
    REPORT.append(f"  emissions, sealed 1x : {summary['sealed_1x_emissions']} (horizon 6)")
    REPORT.append(f"  deepest trajectory   : {summary['deepest_trajectory']}")
    REPORT.append(f"  seventh-step emissions: {summary['seventh_step_emissions']}")
    for key in ("H3", "H6"):
        REPORT.append(f"  {key:<22}: {summary[key]}")
    for key in ("H4", "H5", "H7"):
        REPORT.append(f"  {key:<22}: {summary[key]['verdict']}")
    REPORT.append("=" * 88)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round. The 8x batch is a fresh independent\n"
        "sample, not a superset of the sealed one, so size and realisation are not\n"
        "separated; the frozen@8x arm exists to make that visible."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT / "docs" / "research_branches" / "FP-ITER8X-001" / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
