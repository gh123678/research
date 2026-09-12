"""FP-SAMPLE-001 same-actor derived verification.

Re-derives coverage and the revival counts from the bundles, re-runs the sampler
gate, confirms the sealed corpus is untouched, and replays the affected verifiers.

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
RESULT = PROJECT / "results" / "FP-SAMPLE-001" / "claude" / "formal"
TIGHT = PROJECT / "results" / "FP-TIGHT-001" / "claude" / "formal"
ITER5_ENV = PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy" / "environment.json"
PRIMARY = ("expected_exact", "expected_finite")
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


def main() -> None:
    REPORT.append("FP-SAMPLE-001 same-actor derived verification")
    REPORT.append("=" * 84)
    REPORT.append(
        "NOTE: the user instructed that verification is not the focus of this\n"
        "round. These checks are recorded for completeness and are same-actor only."
    )

    bundle = load(RESULT / "task_results.json")
    summary = load(RESULT / "summary.json")
    rungs = [int(r) for r in bundle["rungs"]]
    rows = rows_of(bundle)

    REPORT.append("\n1. Frozen inputs")
    check(len(rows) == 48, f"48 route-records (found {len(rows)})")
    check(rungs == [1, 2, 4, 8], f"the registered rungs (found {rungs})")
    items = {r: rows[0][3]["rungs"][str(r)]["items"] for r in rungs}
    check(
        all(items[r] == 1048576 * r for r in rungs),
        f"item counts scale exactly with the rung ({items})",
    )

    REPORT.append("\n2. Coverage re-derived, every rung, both arms")
    for arm in ("empirical_bernstein", "bernstein"):
        bad = 0
        for _, _, _, block in rows:
            realized = block["oracle_audit"]["realized_q_sup_error"]
            for rung in rungs:
                e_q = block["rungs"][str(rung)][arm]["e_q"]
                if e_q is None or float(e_q) < realized:
                    bad += 1
        check(
            bad == 0,
            f"{arm}: {48 * len(rungs)} certificate-fits, {bad} coverage violations",
        )

    REPORT.append("\n3. The revival counts, re-derived from the bundles")
    tight = load(TIGHT / "task_results.json")
    frozen_abstainers = {
        (m, t, route)
        for m, t, route, block in rows_of(tight)
        if not block["arms"]["frozen"]["emitted"]
    }
    check(
        len(frozen_abstainers) == 26,
        f"the FP-TIGHT-001 frozen baseline has 26 abstainers "
        f"(found {len(frozen_abstainers)})",
    )
    revived = {}
    for rung in rungs:
        emitters = {
            (m, t, route)
            for m, t, route, block in rows
            if block["rungs"][str(rung)]["empirical_bernstein"]["emitted"]
        }
        revived[rung] = len(frozen_abstainers & emitters)
    REPORT.append(f"  revived by rung: {revived}")
    check(
        [revived[r] for r in rungs]
        == [summary["revived_by_rung"][str(r)] for r in rungs],
        "the summary's revival counts match a fresh re-derivation",
    )
    monotone = all(
        b >= a for a, b in zip([revived[r] for r in rungs], [revived[r] for r in rungs][1:])
    )
    check(monotone, f"the revival counts are non-decreasing ({list(revived.values())})")
    # Eligibility must never be lost when E_Q shrinks: this is the H2 property from
    # FP-TIGHT-001, restated across rungs.
    lost = 0
    for _, _, _, block in rows:
        for a, b in zip(rungs, rungs[1:]):
            if (
                block["rungs"][str(a)]["empirical_bernstein"]["emitted"]
                and not block["rungs"][str(b)]["empirical_bernstein"]["emitted"]
            ):
                lost += 1
    check(lost == 0, f"no route-record loses eligibility as data grows ({lost} lost)")

    REPORT.append("\n4. The sampler gate re-runs")
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "fp_sample_vectorised_batch.py")],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(
        completed.returncode == 0,
        f"fp_sample_vectorised_batch.py exits 0 (got {completed.returncode})",
    )
    for line in completed.stdout.strip().splitlines()[-3:]:
        REPORT.append(f"  INFO  {line}")

    REPORT.append("\n5. Sealed corpus untouched")
    recorded = load(ITER5_ENV)["sealed_file_hashes"]
    for name in SCIENCE:
        if name in recorded:
            check(
                recorded[name] == sha256(PROJECT / name),
                f"SCIENCE {name} is byte-identical to the FP-ITER5-001 record",
            )
    own = load(RESULT / "environment.json")
    for name, digest in own["sealed_file_hashes"].items():
        raw = hashlib.sha256((PROJECT / name).read_bytes()).hexdigest()
        matches = digest in (raw, sha256(PROJECT / name))
        if name in SCIENCE:
            check(matches, f"SCIENCE {name} matches this task's own record")
    # The sampler must not be used where the sealed batch is required: the sealed
    # evaluators must not import it.
    for evaluator in ("evaluate_fp_iter2_001.py", "evaluate_fp_attn_iter_001.py",
                      "evaluate_fp_tight_001.py", "evaluate_fp_census_001.py"):
        source = (PROJECT / evaluator).read_text(encoding="utf-8")
        check(
            "fp_sample_vectorised_batch" not in source,
            f"{evaluator} does not import the vectorised sampler",
        )

    REPORT.append("\n6. Analyzer determinism")
    before = (RESULT / "summary.json").read_bytes()
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "analyze_fp_sample_001.py"),
         "--result-dir", str(RESULT), "--tight-dir", str(TIGHT)],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(
        completed.returncode == 0,
        f"analyze_fp_sample_001.py replays with exit 0 (got {completed.returncode})",
    )
    check(
        before == (RESULT / "summary.json").read_bytes(),
        "re-running the analyzer reproduces summary.json byte-for-byte",
    )

    REPORT.append("\n7. Replay of the affected sealed programs")
    # This task adds a sampler, an evaluator and an analyzer. It changes NO sealed
    # evaluator, so the deep horizon replays cannot be moved by it, and each of them
    # re-runs the same nested subtree -- listing them here would multiply the
    # runtime for zero extra coverage. What is kept is the pair that guards the
    # foundation everything else sits on, plus the hash checks in section 5, which
    # are what would actually catch a sealed module changing.
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

    REPORT.append("\n" + "=" * 84)
    REPORT.append("SUMMARY")
    for rung in rungs:
        REPORT.append(
            f"  {rung}x items={items[rung]:>9} "
            f"mean E_Q={summary['mean_e_q']['empirical_bernstein'][str(rung)]:.4f} "
            f"({summary['reduction_vs_ladder_1x']['empirical_bernstein'][str(rung)]:+.1%})"
            f"  revived={revived[rung]:>2}/26"
        )
    for key in ("H1_sampler_gate", "H4", "H6a", "H6c", "H7"):
        REPORT.append(f"  {key:<18}: {summary[key]}")
    for key in ("H3", "H5", "H6b"):
        REPORT.append(f"  {key:<18}: {summary[key]['verdict']}")
    REPORT.append("=" * 84)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round. Step 1 only; the ladder's rungs\n"
        "are fresh independent samples, not supersets of the sealed batch."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT / "docs" / "research_branches" / "FP-SAMPLE-001" / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
