"""FP-SHORT-001 same-actor derived verification.

Re-derives faithfulness, the inertness theorem, the trajectory lengths and the gate
diagnosis from the bundle; confirms the sealed corpus is untouched; and checks that the
extended grid did not leak into any protocol path.

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
RESULT = PROJECT / "results" / "FP-SHORT-001" / "claude" / "formal"
SEALED_8X = PROJECT / "results" / "FP-ITER8X-001" / "claude" / "formal" / "task_results.json"
ITER5_ENV = PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy" / "environment.json"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("frozen_grid", "extended_grid")
SCIENCE = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "model.py",
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
    REPORT.append("FP-SHORT-001 same-actor derived verification")
    REPORT.append("=" * 92)
    REPORT.append(
        "NOTE: the user instructed that verification is not the focus of this\n"
        "round. These checks are recorded for completeness and are same-actor only."
    )

    bundle = load(RESULT / "task_results.json")
    summary = load(RESULT / "summary.json")
    rows = rows_of(bundle)

    REPORT.append("\n1. Frozen inputs")
    check(len(rows) == 48, f"48 route-records (found {len(rows)})")
    check(
        bundle["frozen_grid"] == [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
        f"the frozen grid is the protocol grid (found {bundle['frozen_grid']})",
    )
    check(
        bundle["extended_grid"][: len(bundle["frozen_grid"])] == bundle["frozen_grid"],
        "the frozen grid is a descending PREFIX of the extended grid, which is what "
        "makes the inertness claim a theorem",
    )

    REPORT.append("\n2. H1 re-derived: the instrumented path is the frozen rule")
    steps_total = 0
    unfaithful = 0
    for _, _, _, block in rows:
        for s in block["frozen_grid"]["steps"]:
            steps_total += 1
            if s.get("faithful_to_sealed_rule") is not True:
                unfaithful += 1
    check(
        unfaithful == 0 and steps_total == summary["H1_unfaithful"] + steps_total,
        f"all {steps_total} steps faithful ({unfaithful} mismatches)",
    )
    sealed = load(SEALED_8X)
    by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
    }
    divergences = 0
    for m, t, route, block in rows:
        ref = by_key.get((m, t))
        if ref is None:
            continue
        ref_steps = ref["routes"][route]["frozen"]["steps"]
        for i, s in enumerate(block["frozen_grid"]["steps"]):
            if i >= len(ref_steps):
                break
            r = ref_steps[i]
            if (
                s["update_emitted"] != r["update_emitted"]
                or s["eta_selected"] != r["eta_selected"]
                or s["e_q"] != r["e_q"]
            ):
                divergences += 1
    check(
        divergences == 0,
        f"the frozen-grid arm reproduces the sealed FP-ITER8X-001 frozen arm "
        f"({divergences} divergences)",
    )

    REPORT.append("\n3. H2 re-derived: the extension is inert")
    changed = 0
    for _, _, _, block in rows:
        frozen = block["frozen_grid"]["steps"]
        extended = block["extended_grid"]["steps"]
        for i, s in enumerate(frozen):
            if i >= len(extended) or not s["update_emitted"]:
                continue
            e = extended[i]
            if not e["update_emitted"] or e["eta_selected"] != s["eta_selected"]:
                changed += 1
    check(changed == 0, f"no emitting decision changes ({changed} counterexamples)")

    REPORT.append("\n4. H5/H6 re-derived: the grid is not the constraint")
    frozen_total = sum(b["frozen_grid"]["emitted_steps"] for _, _, _, b in rows)
    extended_total = sum(b["extended_grid"]["emitted_steps"] for _, _, _, b in rows)
    check(
        frozen_total == extended_total,
        f"the two arms emit the same number of steps ({frozen_total} vs "
        f"{extended_total})",
    )
    lengths_differ = sum(
        1
        for _, _, _, b in rows
        if b["extended_grid"]["emitted_steps"] != b["frozen_grid"]["emitted_steps"]
    )
    check(lengths_differ == 0, f"no trajectory changes length ({lengths_differ})")
    check(
        summary["H5"]["verdict"] == "FALSIFIED"
        and summary["H6"]["verdict"] == "FALSIFIED",
        "the summary records both grid hypotheses as falsified",
    )

    REPORT.append("\n5. The gate diagnosis re-derived")
    ratios = []
    blocking = []
    for _, _, _, block in rows:
        steps = block["frozen_grid"]["steps"]
        if block["frozen_grid"]["emitted_steps"] > 5:
            continue
        diag = steps[-1].get("diagnosis")
        if not diag:
            continue
        if diag["e_q_over_h"] is not None:
            ratios.append(diag["e_q_over_h"])
        blocking.append(steps[-1]["blocking_states"])
    check(
        len(ratios) == summary["H3"]["total"],
        f"the short-trajectory count matches the summary ({len(ratios)})",
    )
    check(
        all(1.0 <= r for r in ratios),
        f"every short trajectory has E_Q/h >= 1, i.e. the margin really is the "
        f"binding quantity (min {min(ratios):.4f})",
    )
    check(
        all(b == 1 for b in blocking),
        f"exactly one state blocks in every short trajectory "
        f"(distribution {sorted(set(blocking))})",
    )
    check(
        summary["H4"]["verdict"] == "PASS" and summary["H3"]["verdict"] == "PASS",
        "the summary records H3 and H4 as passing",
    )

    REPORT.append("\n6. The extended grid did not leak into a protocol path")
    # The grid is frozen protocol; the diagnostic arm must not be adopted anywhere.
    for evaluator in ("evaluate_fp_iter2_001.py", "evaluate_fp_attn_iter_001.py",
                      "evaluate_fp_iter8x_001.py", "evaluate_fp_attn_8x_001.py"):
        source = (PROJECT / evaluator).read_text(encoding="utf-8")
        check(
            "EXTENDED_GRID" not in source,
            f"{evaluator} does not use the extended grid",
        )
    frozen_source = (PROJECT / "fixed_policy_expected_sarsa_scaled.py").read_text(
        encoding="utf-8"
    )
    check(
        "ETA_CANDIDATES = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)" in frozen_source,
        "the frozen eta grid in the sealed module is unchanged",
    )

    REPORT.append("\n7. Sealed corpus untouched")
    recorded = load(ITER5_ENV)["sealed_file_hashes"]
    for name in SCIENCE:
        if name in recorded:
            check(
                recorded[name] == sha256(PROJECT / name),
                f"SCIENCE {name} is byte-identical to the FP-ITER5-001 record",
            )

    REPORT.append("\n8. Analyzer determinism")
    before = (RESULT / "summary.json").read_bytes()
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "analyze_fp_short_001.py"),
         "--result-dir", str(RESULT), "--sealed-dir", str(SEALED_8X)],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(
        completed.returncode == 0,
        f"analyze_fp_short_001.py replays with exit 0 (got {completed.returncode})",
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

    REPORT.append("\n" + "=" * 92)
    REPORT.append("SUMMARY")
    REPORT.append(f"  short trajectories      : {len(ratios)}")
    REPORT.append(
        f"  E_Q/h median            : "
        f"{sorted(ratios)[len(ratios) // 2]:.4f} "
        f"(min {min(ratios):.4f}, max {max(ratios):.4f})"
    )
    REPORT.append(f"  blocking-state counts   : {sorted(set(blocking))}")
    REPORT.append(f"  emitted steps frozen    : {frozen_total}")
    REPORT.append(f"  emitted steps extended  : {extended_total}")
    for key in ("H1_unfaithful", "H2_changed_decisions"):
        REPORT.append(f"  {key:<24}: {summary[key]}")
    for key in ("H3", "H4", "H5", "H6"):
        REPORT.append(f"  {key:<24}: {summary[key]['verdict']}")
    REPORT.append("=" * 92)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round. The extended grid is a diagnostic,\n"
        "not a proposed protocol."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT / "docs" / "research_branches" / "FP-SHORT-001" / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
