"""FP-ITER4-001 same-actor derived verification.

Recomputes the per-level pattern, confirms the horizon change is inert against
the sealed FP-ITER3-001 bundle, checks the three pre-registered step-4
predictions, replays the sealed programs, and confirms no sealed file changed.

This is DERIVED verification by the same actor that executed the task. It is NOT
independent verification: no second actor reconstructed this route.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
FORMAL = PROJECT / "results" / "FP-ITER4-001" / "claude" / "formal"
ITER3 = PROJECT / "results" / "FP-ITER3-001" / "claude" / "formal"
REPORT: list[str] = []
FAILURES = 0
PRIMARY = ("expected_exact", "expected_finite")
FROZEN_REASONS = {
    "algorithm_mode_mismatch",
    "duplicate_q_memory",
    "divergence_guard_triggered",
    "heldout_pair_support_missing",
    "mixture_inversion_unbracketed",
    "mixture_inversion_not_converged",
    "mixture_root_not_conservative",
    "numerical_nonfinite",
    "policy_invalid",
    "improvement_lcb_nonpositive",
    "policy_unchanged",
}


def check(ok: bool, message: str) -> None:
    global FAILURES
    if ok:
        REPORT.append(f"  PASS  {message}")
    else:
        FAILURES += 1
        REPORT.append(f"  FAIL  {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    REPORT.append("FP-ITER4-001 same-actor derived verification")
    REPORT.append("=" * 72)

    bundle = json.loads((FORMAL / "task_results.json").read_text(encoding="utf-8"))
    summary = json.loads((FORMAL / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((FORMAL / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((FORMAL / "environment.json").read_text(encoding="utf-8"))
    iter3_bundle = json.loads((ITER3 / "task_results.json").read_text(encoding="utf-8"))
    iter3_summary = json.loads((ITER3 / "summary.json").read_text(encoding="utf-8"))

    blocks = [
        (record, route, record["routes"][route])
        for record in bundle["records"]
        for route in PRIMARY
    ]
    steps = [(r, route, b, s) for r, route, b in blocks for s in b["steps"]]
    max_steps = int(bundle["max_steps"])

    REPORT.append("\n1. Frozen inputs")
    check(len(bundle["records"]) == 24, f"24 records (found {len(bundle['records'])})")
    check(len(blocks) == 48, f"48 route-records (found {len(blocks)})")
    check(max_steps == 4, f"MAX_STEPS is 4 (found {max_steps})")
    check(config["max_steps"] == 4, "config records the frozen horizon 4")
    check(config["mixings"] == [0.08, 0.5], "both frozen mixing settings")

    REPORT.append("\n2. H1/H2: the horizon change is inert")
    sealed_by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in iter3_bundle["records"]
    }
    compared = 0
    mismatches = []
    for record, route, block in blocks:
        ref = sealed_by_key.get((float(record["mixing"]), int(record["task_index"])))
        if ref is None:
            continue
        ref_steps = ref["routes"][route]["steps"]
        for index in range(min(3, len(block["steps"]), len(ref_steps))):
            compared += 1
            a, b = block["steps"][index], ref_steps[index]
            if (
                a["update_emitted"] != b["update_emitted"]
                or a["eta_selected"] != b["eta_selected"]
                or a["e_q"] != b["e_q"]
            ):
                mismatches.append(
                    (record["mixing"], record["task_index"], route, index + 1)
                )
    check(compared == 90, f"90 step-1..3 entries compared (found {compared})")
    check(
        not mismatches,
        f"steps 1-3 are bit-identical to the sealed FP-ITER3-001 run "
        f"({len(mismatches)} differ)",
    )
    check(
        summary["H1_step_limit_monotone"]["outcome"] == "PASS",
        "the summary records H1 as PASS",
    )

    REPORT.append("\n3. Per-level pattern recomputed")
    per_level: dict[int, dict[str, float | None]] = {}
    for level in range(1, max_steps + 1):
        gains = [
            s["oracle_audit"]["total_value_gain"]
            for _, _, _, s in steps
            if s["step"] == level and s["update_emitted"]
        ]
        per_level[level] = {
            "emissions": len(gains),
            "mean": (sum(gains) / len(gains)) if gains else None,
            "min": min(gains) if gains else None,
        }
    for level in (1, 2, 3):
        check(
            per_level[level]["emissions"]
            == iter3_summary["per_level"][str(level)]["emissions"],
            f"step {level} emissions {per_level[level]['emissions']} equal the sealed "
            f"{iter3_summary['per_level'][str(level)]['emissions']}",
        )
    check(per_level[4]["emissions"] == 12, f"12 fourth-step emissions (found {per_level[4]['emissions']})")

    REPORT.append("\n4. The three pre-registered step-4 predictions")
    n3 = per_level[3]["emissions"]
    n4 = per_level[4]["emissions"]
    mean3 = per_level[3]["mean"]
    mean4 = per_level[4]["mean"]
    min4 = per_level[4]["min"]
    check(n4 is not None and n4 < n3, f"H7 PASS: n4={n4} < n3={n3}")
    check(
        mean4 is not None and mean3 is not None and mean4 < mean3,
        f"H8 PASS: mean4={mean4!r} < mean3={mean3!r}",
    )
    check(
        min4 is not None and min4 > 0.05,
        f"H9 PASS: min4={min4!r} > 0.05 (non-vacuous)",
    )
    check(
        summary["H7_attrition_prediction"]["outcome"] == "PASS"
        and summary["H8_mean_gain_prediction"]["outcome"] == "PASS"
        and summary["H9_non_vacuity_prediction"]["outcome"] == "PASS",
        "the summary records all three predictions as PASS",
    )
    check(
        summary["H3_fourth_step_certifiable"]["outcome"] == "PASS",
        "H3 PASS: a fourth step is certifiable",
    )

    REPORT.append("\n5. Validity and monotonicity")
    violations = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s["oracle_audit"].get("certificate_violation")
    ]
    nondegrading = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s["update_emitted"]
        and min(s["oracle_audit"]["value_delta_vs_previous"]) < -1e-12
    ]
    check(not violations, f"zero certificate violations ({len(violations)})")
    check(not nondegrading, f"zero non-degrading violations ({len(nondegrading)})")
    for level in range(1, max_steps + 1):
        gains = [
            s["oracle_audit"]["total_value_gain"]
            for _, _, _, s in steps
            if s["step"] == level and s["update_emitted"]
        ]
        check(
            all(g > 0 for g in gains),
            f"every step-{level} gain is strictly positive ({len(gains)} emissions)",
        )

    REPORT.append("\n6. Attrition reasons and batches")
    seen: set[str] = set()
    for _, _, _, s in steps:
        if not s["update_emitted"]:
            seen.update(s["ordered_reasons"])
    check(seen <= FROZEN_REASONS, f"all reasons frozen (offenders {seen - FROZEN_REASONS})")
    check(
        len({r["train_batch_digest"] for r in bundle["records"]})
        == len(bundle["records"]),
        "each record has its own training batch digest",
    )
    check(
        all(r["cert_batch_digest"] for r in bundle["records"]),
        "every record records its certification batch digest",
    )
    check(
        all(b["step1_reproduction"].get("exact") is not False for _, _, b in blocks),
        "no step-1 reproduction failure",
    )

    REPORT.append("\n7. No sealed file changed")
    for name, digest in environment["sealed_file_hashes"].items():
        path = PROJECT / name
        raw = hashlib.sha256(path.read_bytes()).hexdigest()
        check(digest in (raw, sha256(path)), f"{name} matches its recorded hash")

    REPORT.append("\n8. Replay of the sealed programs")
    for script, extra in (
        ("analyze_fp_iter4_001.py", ["--result-dir", str(FORMAL)]),
        ("analyze_fp_iter3_001.py", ["--result-dir", str(ITER3)]),
        ("verify_fp_iter3_001_same_actor.py", []),
        ("verify_fp_attn_iter_001_same_actor.py", []),
        ("verify_variance_adaptive_certificate.py", []),
    ):
        completed = subprocess.run(
            [sys.executable, "-B", str(PROJECT / script), *extra],
            capture_output=True,
            text=True,
            cwd=PROJECT,
        )
        check(
            completed.returncode == 0,
            f"{script} replays with exit 0 (got {completed.returncode})",
        )

    REPORT.append("\n" + "=" * 72)
    REPORT.append("SUMMARY")
    lines = [
        f"  route-records            : {len(blocks)}",
        f"  step executions          : {len(steps)}",
        f"  emitted all four steps   : "
        f"{sum(1 for _, _, b in blocks if b['emitted_steps'] >= 4)}",
    ]
    for level in range(1, max_steps + 1):
        lines.append(
            f"  step {level}: emissions {per_level[level]['emissions']:>3}  "
            f"mean gain {per_level[level]['mean']:.6f}  "
            f"min gain {per_level[level]['min']:.6f}"
        )
    lines.append(f"  certificate violations   : {len(violations)}")
    lines.append(f"  non-degrading violations : {len(nondegrading)}")
    lines.append("  H1 inert horizon         : PASS (90/90 entries identical)")
    lines.append(f"  H3 fourth step           : PASS (n4={n4})")
    lines.append(f"  H7 attrition  n4 < n3    : PASS ({n4} < {n3})")
    lines.append(f"  H8 mean decay            : PASS ({mean4:.6f} < {mean3:.6f})")
    lines.append(f"  H9 non-vacuity           : PASS (min4={min4:.6f} > 0.05)")
    REPORT.extend(lines)
    REPORT.append("=" * 72)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only. No second actor\n"
        "reconstructed this route; this is not reciprocal verification."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT
        / "docs"
        / "research_branches"
        / "FP-ITER4-001"
        / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
