"""FP-ITER3-001 same-actor derived verification.

Recomputes the per-level comparison, confirms the horizon change is inert,
checks no sealed file changed, and replays the sealed programs.

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
FORMAL = PROJECT / "results" / "FP-ITER3-001" / "claude" / "formal"
ITER2 = PROJECT / "results" / "FP-ITER2-001" / "claude" / "formal"
REPORT: list[str] = []
FAILURES = 0
PRIMARY = ("expected_exact", "expected_finite")


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
    REPORT.append("FP-ITER3-001 same-actor derived verification")
    REPORT.append("=" * 70)

    bundle = json.loads((FORMAL / "task_results.json").read_text(encoding="utf-8"))
    summary = json.loads((FORMAL / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((FORMAL / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((FORMAL / "environment.json").read_text(encoding="utf-8"))
    iter2 = json.loads((ITER2 / "summary.json").read_text(encoding="utf-8"))
    iter2_bundle = json.loads((ITER2 / "task_results.json").read_text(encoding="utf-8"))

    blocks = [
        (record, route, record["routes"][route])
        for record in bundle["records"]
        for route in PRIMARY
    ]
    max_steps = int(bundle["max_steps"])

    REPORT.append("\n1. Frozen inputs")
    check(len(bundle["records"]) == 24, f"24 records (found {len(bundle['records'])})")
    check(len(blocks) == 48, f"48 route-records (found {len(blocks)})")
    check(max_steps == 3, f"MAX_STEPS is 3 (found {max_steps})")
    check(config["max_steps"] == 3, "config records the frozen horizon 3")
    check(config["mixings"] == [0.08, 0.5], "both frozen mixing settings")

    REPORT.append("\n2. H1/H2: the horizon change is inert")
    per_level: dict[int, list[float]] = {k: [] for k in range(1, max_steps + 1)}
    emissions: dict[int, int] = {}
    for level in range(1, max_steps + 1):
        emissions[level] = sum(
            1
            for _, _, b in blocks
            if len(b["steps"]) >= level and b["steps"][level - 1]["update_emitted"]
        )
    for _, _, block in blocks:
        for step in block["steps"]:
            if step["update_emitted"]:
                per_level[step["step"]].append(step["oracle_audit"]["total_value_gain"])

    check(
        emissions[1] == iter2["step1_emissions"] == 22,
        f"step-1 emissions {emissions[1]} equals sealed {iter2['step1_emissions']}",
    )
    check(
        emissions[2] == iter2["emitted_two_steps"] == 20,
        f"step-2 emissions {emissions[2]} equals sealed {iter2['emitted_two_steps']}",
    )
    mean2 = sum(per_level[2]) / len(per_level[2])
    check(
        abs(mean2 - iter2["mean_gain_step2"]) < 1e-12,
        f"step-2 mean gain {mean2!r} equals sealed {iter2['mean_gain_step2']!r}",
    )
    check(
        abs(min(per_level[2]) - iter2["min_gain_step2"]) < 1e-12,
        f"step-2 minimum gain {min(per_level[2])!r} equals sealed "
        f"{iter2['min_gain_step2']!r}",
    )
    mean1 = sum(per_level[1]) / len(per_level[1])
    check(
        abs(mean1 - iter2["mean_gain_step1"]) < 1e-12,
        f"step-1 mean gain {mean1!r} equals sealed {iter2['mean_gain_step1']!r}",
    )

    # Bit-level comparison of the first two steps against the ITER2 bundle.
    iter2_by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in iter2_bundle["records"]
    }
    compared = 0
    expected_pairs = 0
    mismatched = []
    for record, route, block in blocks:
        sealed_record = iter2_by_key.get((float(record["mixing"]), int(record["task_index"])))
        if sealed_record is None:
            continue
        sealed_block = sealed_record["routes"][route]
        # A route-record only has as many comparable entries as the SEALED run
        # attained. 22 reached step 2 and 26 stopped after step 1, so the
        # expectation is 22*2 + 26*1 = 70 pairs, not 48*2.
        expected_pairs += min(2, len(sealed_block["steps"]))
        for index in range(min(2, len(block["steps"]), len(sealed_block["steps"]))):
            here = block["steps"][index]
            there = sealed_block["steps"][index]
            compared += 1
            if (
                here["update_emitted"] != there["update_emitted"]
                or here["eta_selected"] != there["eta_selected"]
                or here["e_q"] != there["e_q"]
            ):
                mismatched.append(
                    (record["mixing"], record["task_index"], route, index + 1)
                )
    check(
        compared == expected_pairs,
        f"compared every comparable step-1/step-2 entry "
        f"({compared} vs expected {expected_pairs})",
    )
    check(compared > 0, "at least one entry was compared")
    check(
        not mismatched,
        f"steps 1 and 2 are bit-identical to the sealed FP-ITER2-001 run "
        f"({len(mismatched)} differ)",
    )

    REPORT.append("\n3. Per-level attrition and validity")
    check(emissions[3] == 15, f"15 third-step emissions (found {emissions[3]})")
    check(emissions[3] < emissions[2], f"H6 PASS: n3={emissions[3]} < n2={emissions[2]}")
    check(
        abs(min(per_level[3]) - 0.37819659359698987) < 1e-12,
        f"step-3 minimum gain {min(per_level[3])!r}",
    )
    check(
        min(per_level[3]) < iter2["min_gain_step2"],
        f"H7 FALSIFIED as reported: min3={min(per_level[3]):.6f} < "
        f"min2={iter2['min_gain_step2']:.6f}",
    )
    check(
        all(g > 0 for g in per_level[3]),
        "every third-step gain is strictly positive",
    )
    check(
        summary["H7_filtering_prediction"]["outcome"] == "FALSIFIED",
        "the summary reports H7 as FALSIFIED rather than reinterpreting it",
    )

    REPORT.append("\n4. Violations recomputed")
    violations = 0
    nondegrading = 0
    for record, route, block in blocks:
        for step in block["steps"]:
            audit = step["oracle_audit"]
            if audit.get("certificate_violation"):
                violations += 1
            if step["update_emitted"] and min(audit["value_delta_vs_previous"]) < -1e-12:
                nondegrading += 1
    check(violations == 0, f"zero certificate violations ({violations})")
    check(nondegrading == 0, f"zero non-degrading violations ({nondegrading})")

    REPORT.append("\n5. Attrition reasons from the frozen list")
    frozen = {
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
    seen: set[str] = set()
    for _, _, block in blocks:
        for step in block["steps"]:
            if not step["update_emitted"]:
                seen.update(step["ordered_reasons"])
    check(seen <= frozen, f"all reasons frozen (offenders {seen - frozen})")

    REPORT.append("\n6. No sealed file changed")
    for name, digest in environment["sealed_file_hashes"].items():
        path = PROJECT / name
        raw = hashlib.sha256(path.read_bytes()).hexdigest()
        check(
            digest in (raw, sha256(path)),
            f"{name} matches its recorded hash",
        )

    REPORT.append("\n7. Replay of the sealed programs")
    for script, extra in (
        ("analyze_fp_iter3_001.py", ["--result-dir", str(FORMAL)]),
        ("analyze_fp_iter2_001.py", ["--result-dir", str(ITER2)]),
        ("verify_variance_adaptive_certificate.py", []),
        ("verify_fp_scale_002_same_actor.py", []),
        ("verify_fp_attn_001_same_actor.py", []),
        ("verify_fp_iter2_001_same_actor.py", []),
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

    REPORT.append("\n" + "=" * 70)
    REPORT.append("SUMMARY")
    REPORT.append(
        f"  route-records                : {len(blocks)}\n"
        f"  step-1 emissions             : {emissions[1]} (sealed 22)\n"
        f"  step-2 emissions             : {emissions[2]} (sealed 20)\n"
        f"  step-3 emissions             : {emissions[3]}\n"
        f"  emitted all three steps      : "
        f"{sum(1 for _, _, b in blocks if b['emitted_steps'] >= 3)}\n"
        f"  mean gain 1 / 2 / 3          : {mean1:.6f} / {mean2:.6f} / "
        f"{sum(per_level[3]) / len(per_level[3]):.6f}\n"
        f"  min  gain 1 / 2 / 3          : {min(per_level[1]):.6f} / "
        f"{min(per_level[2]):.6f} / {min(per_level[3]):.6f}\n"
        f"  certificate violations       : {violations}\n"
        f"  non-degrading violations     : {nondegrading}\n"
        f"  H6 attrition                 : PASS (n3 < n2)\n"
        f"  H7 filtering                 : FALSIFIED (min3 < min2)"
    )
    REPORT.append("=" * 70)
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
        / "FP-ITER3-001"
        / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
