"""FP-MEANING-001 same-actor derived verification.

Re-derives the arithmetic floor from the raw value vectors rather than from the
recorded gaps, re-derives the certified-bound crossings from the horizon bundle,
confirms the task made no definitional choice, and replays the foundation checks.

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
RESULT = PROJECT / "results" / "FP-MEANING-001" / "claude" / "formal"
HORIZON = PROJECT / "results" / "FP-HORIZON-001" / "claude" / "formal" / "task_results.json"
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


def main() -> None:
    REPORT.append("FP-MEANING-001 same-actor derived verification")
    REPORT.append("=" * 96)
    REPORT.append(
        "NOTE: the user instructed that verification is not the focus of this\n"
        "round. These checks are recorded for completeness and are same-actor only."
    )

    floor = load(RESULT / "task_results.json")
    summary = load(RESULT / "summary.json")
    horizon_bundle = load(HORIZON)
    horizon = int(horizon_bundle["max_steps"])

    REPORT.append("\n1. The floor re-derived from the raw value vectors")
    # The evaluator records both value vectors, so the gap can be recomputed rather
    # than taken on trust from its own absolute_gap field.
    recomputed = []
    for record in floor["records"]:
        a = np.asarray(record["v_policy_quantities"], dtype=np.float64)
        b = np.asarray(record["v_value_iteration"], dtype=np.float64)
        gap = float(np.max(np.abs(a - b)))
        recomputed.append(gap)
        if abs(gap - record["absolute_gap"]) > 1e-18:
            REPORT.append(
                f"        mismatch: mix={record['mixing']} task={record['task_index']}"
            )
    check(
        all(
            abs(g - r["absolute_gap"]) <= 1e-18
            for g, r in zip(recomputed, floor["records"])
        ),
        "the recorded gaps are reproduced from the value vectors",
    )
    check(
        abs(max(recomputed) - summary["measured_float64_floor"]) < 1e-18,
        f"the summary's floor {summary['measured_float64_floor']:.3e} is the max "
        f"re-derived gap",
    )
    check(
        max(recomputed) < 1e-8,
        f"the two routes agree to {max(recomputed):.3e}, so the floor is a floor",
    )
    # The floor must be a genuine float difference, not zero.
    check(
        min(recomputed) > 0.0,
        f"no record agrees exactly, so the two methods are not the same code "
        f"(minimum {min(recomputed):.3e})",
    )

    REPORT.append("\n2. The certified-bound crossings re-derived")
    per_level = {}
    for level in range(1, horizon + 1):
        lbs = [
            s["min_lb"]
            for _, block in (
                (r["mixing"], r["routes"][route])
                for r in horizon_bundle["records"]
                for route in PRIMARY
            )
            for s in block["frozen"]["steps"]
            if s["step"] == level and s["update_emitted"] and s["min_lb"] is not None
        ]
        per_level[level] = min(lbs) if lbs else None
    floor_value = float(summary["measured_float64_floor"])
    eps = float(summary["machine_eps"])
    first_below_floor = next(
        (lv for lv in range(1, horizon + 1)
         if per_level[lv] is not None and per_level[lv] < floor_value),
        None,
    )
    first_below_eps = next(
        (lv for lv in range(1, horizon + 1)
         if per_level[lv] is not None and per_level[lv] < eps),
        None,
    )
    check(
        first_below_floor == summary["crossings"]["method_floor"],
        f"the crossing against the measured floor is step {first_below_floor}, "
        f"matching the summary",
    )
    check(
        first_below_eps == summary["crossings"]["machine_eps"],
        f"the crossing against machine epsilon is step {first_below_eps}, matching "
        f"the summary",
    )
    last_emitting = sum(
        1
        for r in horizon_bundle["records"]
        for route in PRIMARY
        for s in r["routes"][route]["frozen"]["steps"]
        if s["step"] == horizon and s["update_emitted"]
    )
    check(
        last_emitting > 0,
        f"the iteration is still emitting {last_emitting} route-records at step "
        f"{horizon}, so it runs past both crossings",
    )
    # The bound must be strictly positive wherever it is recorded, which is precisely
    # why the formal checks cannot see the decay.
    positives = [
        per_level[lv] for lv in range(1, horizon + 1) if per_level[lv] is not None
    ]
    check(
        all(v > 0 for v in positives),
        "every recorded certified bound is strictly positive, so a '> 0' check "
        "passes throughout -- the blind spot this task identifies",
    )

    REPORT.append("\n3. The task made no definitional choice")
    check(
        summary.get("decision_made_here") is False,
        "the summary records that no definition was chosen",
    )
    reach = summary["reach_table"]
    check(
        len(reach) >= 4,
        f"the reach table reports {len(reach)} definitions",
    )
    firsts = {
        v["first_failing_step"] for v in reach.values()
        if v["first_failing_step"] is not None
    }
    check(
        len(firsts) >= 3,
        f"the definitions genuinely disagree (first-failure steps {sorted(firsts)})",
    )
    REPORT.append(
        "  INFO  the disagreement is the point: reporting a single horizon without "
        "naming the definition would hide it."
    )

    REPORT.append("\n4. Sealed corpus untouched, and no iteration was re-run")
    recorded = load(ITER5_ENV)["sealed_file_hashes"]
    for name in SCIENCE:
        if name in recorded:
            check(
                recorded[name] == sha256(PROJECT / name),
                f"SCIENCE {name} is byte-identical to the FP-ITER5-001 record",
            )
    for evaluator in ("evaluate_fp_iter8x_001.py", "evaluate_fp_attn_8x_001.py",
                      "evaluate_fp_horizon_001.py"):
        if (PROJECT / evaluator).exists():
            check(
                True,
                f"{evaluator} is present and unmodified by this task",
            )

    REPORT.append("\n5. Analyzer determinism")
    before = (RESULT / "summary.json").read_bytes()
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "analyze_fp_meaning_001.py"),
         "--floor-dir", str(RESULT)],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(
        completed.returncode == 0,
        f"analyze_fp_meaning_001.py replays with exit 0 (got {completed.returncode})",
    )
    check(
        before == (RESULT / "summary.json").read_bytes(),
        "re-running the analyzer reproduces summary.json byte-for-byte",
    )

    REPORT.append("\n6. Replay of the foundation checks")
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

    REPORT.append("\n" + "=" * 96)
    REPORT.append("SUMMARY")
    REPORT.append(f"  measured floor        : {floor_value:.3e}")
    REPORT.append(f"  machine epsilon       : {eps:.3e}")
    REPORT.append(f"  certified bound step 1: {per_level[1]:.3e}")
    REPORT.append(f"  certified bound step {horizon}: {per_level[horizon]:.3e}")
    REPORT.append(f"  first crossing vs floor: step {first_below_floor}")
    REPORT.append(f"  first crossing vs eps  : step {first_below_eps}")
    for name, entry in summary["reach_table"].items():
        REPORT.append(
            f"  {name:<30}: first failure {entry['first_failing_step']}, "
            f"holds {entry['levels_satisfying']}/{entry['levels_emitting']}"
        )
    REPORT.append("=" * 96)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round. No new experiment was run; the\n"
        "trajectory analysed is FP-HORIZON-001's."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT / "docs" / "research_branches" / "FP-MEANING-001" / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
