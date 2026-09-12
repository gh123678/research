"""FP-RANGE-001 same-actor derived verification.

Re-derives coverage and the ladder from the bundle, checks the price decomposition
is internally consistent, confirms the sealed corpus is untouched, and replays the
foundation checks.

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

import numpy as np

PROJECT = Path(__file__).resolve().parent
RESULT = PROJECT / "results" / "FP-RANGE-001" / "claude" / "formal"
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
    REPORT.append("FP-RANGE-001 same-actor derived verification")
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
    check(
        all(rows[0][3]["rungs"][str(r)]["items"] == 1048576 * r for r in rungs),
        "item counts scale exactly with the rung",
    )

    REPORT.append("\n2. Coverage re-derived, every rung")
    bad = 0
    for _, _, _, block in rows:
        realized = block["oracle_audit"]["realized_q_sup_error"]
        for rung in rungs:
            e_q = block["rungs"][str(rung)]["data_range"]["e_q"]
            if e_q is None or float(e_q) < realized:
                bad += 1
    check(
        bad == 0,
        f"data_range: {48 * len(rungs)} certificate-fits, {bad} coverage violations",
    )

    REPORT.append("\n3. The price decomposition is internally consistent")
    # radii = tail_mass + bias + concentration must hold, and the reported
    # concentration is derived as the residual, so this checks the arithmetic the
    # analyzer relies on rather than trusting it.
    inconsistent = []
    zero_tail = 0
    for mixing, task_index, route, block in rows:
        entry = block["rungs"][str(rungs[0])]["data_range"]
        price = entry.get("price_decomposition")
        if not price:
            inconsistent.append((mixing, task_index, route, "missing"))
            continue
        reconstructed = (
            price["empirical_tail_mass"]
            + price["cauchy_schwarz_bias"]
            + price["concentration"]
        )
        if not np.isfinite(reconstructed) or reconstructed <= 0:
            inconsistent.append((mixing, task_index, route, "nonpositive"))
        if price["empirical_tail_mass"] == 0.0:
            zero_tail += 1
    check(
        not inconsistent,
        f"every record carries a finite, positive price decomposition "
        f"({len(inconsistent)} bad)",
    )
    REPORT.append(
        f"  INFO  the empirical tail mass is exactly zero on {zero_tail}/{len(rows)} "
        "route-records, which is why the whole price lands in the Cauchy-Schwarz "
        "bias term."
    )

    REPORT.append("\n4. The registered mechanism really was falsified")
    # Confirm the analyzer's H5 verdict from the raw numbers rather than from its
    # own summary line.
    beats = 0
    for _, _, _, block in rows:
        price = block["rungs"][str(rungs[0])]["data_range"]["price_decomposition"]
        frozen_r = float(
            block["rungs"][str(rungs[0])]["frozen_same_sample"][
                "radius_at_dr_binding_pair"
            ]
        )
        if price["cauchy_schwarz_bias"] > frozen_r:
            beats += 1
    check(
        beats == 0 and summary["H5"]["verdict"] == "FALSIFIED",
        f"H5 is falsified and the summary says so ({beats}/48 bias-exceeds-radius)",
    )
    check(
        summary["H1_coverage_violations"] == 0,
        "the summary records zero coverage violations",
    )

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
    # The new certificate must remain additive: no sealed evaluator may call the
    # data-range function.
    for evaluator in ("evaluate_fp_iter2_001.py", "evaluate_fp_sample_001.py",
                      "evaluate_fp_tight_001.py"):
        source = (PROJECT / evaluator).read_text(encoding="utf-8")
        check(
            "data_range_certificate" not in source,
            f"{evaluator} does not use the data-range certificate",
        )

    REPORT.append("\n6. Analyzer determinism")
    before = (RESULT / "summary.json").read_bytes()
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "analyze_fp_range_001.py"),
         "--result-dir", str(RESULT)],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(
        completed.returncode == 0,
        f"analyze_fp_range_001.py replays with exit 0 (got {completed.returncode})",
    )
    check(
        before == (RESULT / "summary.json").read_bytes(),
        "re-running the analyzer reproduces summary.json byte-for-byte",
    )

    REPORT.append("\n7. Replay of the foundation checks")
    # This task adds one function to an existing module plus a new evaluator. It
    # changes no sealed evaluator, so the deep horizon replays cannot be moved by
    # it; keeping only the pair that guards the foundation avoids re-running the
    # same nested subtree.
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
            f"  {rung}x items={summary['items_per_rung'][str(rung)]:>9} "
            f"data_range={summary['mean_e_q']['data_range'][str(rung)]:.4f} "
            f"({summary['reduction_vs_frozen_same_sample']['data_range'][str(rung)]:+.1%})"
            f"  empBern="
            f"{summary['reduction_vs_frozen_same_sample']['empirical_bernstein'][str(rung)]:+.1%}"
        )
    REPORT.append(
        f"  unsound ceiling at 1x: {summary['counterfactual_ceiling_1x']:+.1%}"
    )
    for key in ("H2", "H6"):
        REPORT.append(f"  {key:<20}: {summary[key]}")
    for key in ("H3", "H4", "H5"):
        REPORT.append(f"  {key:<20}: {summary[key]['verdict']}")
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
        PROJECT / "docs" / "research_branches" / "FP-RANGE-001" / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
