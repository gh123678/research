"""FP-ITER2-001 same-actor derived verification.

Recomputes the comparison independently of the evaluator and analyzer, replays
the sealed programs, and confirms no sealed file changed.

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
sys.path.insert(0, str(PROJECT))

FORMAL = PROJECT / "results" / "FP-ITER2-001" / "claude" / "formal"
SEALED_ENV = (
    PROJECT / "results" / "FP-SCALE-002" / "claude" / "formal" / "environment.json"
)
SEALED_BUNDLE = (
    PROJECT / "results" / "FP-SCALE-002" / "claude" / "formal" / "task_results.json"
)
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
    REPORT.append("FP-ITER2-001 same-actor derived verification")
    REPORT.append("=" * 68)

    bundle = json.loads((FORMAL / "task_results.json").read_text(encoding="utf-8"))
    summary = json.loads((FORMAL / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((FORMAL / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((FORMAL / "environment.json").read_text(encoding="utf-8"))
    sealed = json.loads(SEALED_BUNDLE.read_text(encoding="utf-8"))
    sealed_emitted = sum(
        1
        for record in sealed["records"]
        for route in ("variance_adaptive_exact", "variance_adaptive_finite")
        if record["routes"][route]["update_emitted"]
    )

    blocks = [
        (record, route, record["routes"][route])
        for record in bundle["records"]
        for route in PRIMARY
    ]

    REPORT.append("\n1. Frozen inputs")
    check(len(bundle["records"]) == 24, f"24 records (found {len(bundle['records'])})")
    check(len(blocks) == 48, f"48 route-records (found {len(blocks)})")
    check(config["max_steps"] == 2, "MAX_STEPS is 2")
    check(config["mixings"] == [0.08, 0.5], "both frozen mixing settings")
    check(
        config["cert_chains"] == 16384 and config["cert_chain_length"] == 64,
        "FP-SCALE-002 certification constants (16384 x 64)",
    )

    REPORT.append("\n2. Step 1 reproduces the sealed FP-SCALE-002 result")
    repro = [b for _, _, b in blocks if b["step1_reproduction"]]
    repro_bad = [b for b in repro if b["step1_reproduction"].get("exact") is not True]
    check(len(repro) == 48, f"step-1 reproduction checked on all 48 (found {len(repro)})")
    check(not repro_bad, f"every step-1 decision, eta and E_Q matches sealed ({len(repro_bad)} bad)")
    step1_emitted = sum(1 for _, _, b in blocks if b["steps"][0]["update_emitted"])
    check(
        step1_emitted == sealed_emitted == 22,
        f"step-1 emissions {step1_emitted} equals the sealed {sealed_emitted}",
    )

    REPORT.append("\n3. Attrition and validity recomputed")
    two = [b for _, _, b in blocks if b["emitted_steps"] >= 2]
    one = [b for _, _, b in blocks if b["emitted_steps"] == 1]
    zero = [b for _, _, b in blocks if b["emitted_steps"] == 0]
    check(
        len(two) == summary["emitted_two_steps"] == 20,
        f"20 route-records emit twice (found {len(two)})",
    )
    check(len(one) == summary["emitted_one_step"], f"one-step count {len(one)} matches")
    check(len(zero) == summary["emitted_zero_steps"], f"zero-step count {len(zero)} matches")
    check(
        len(two) + len(one) == step1_emitted,
        "attrition accounting closes: two-step + one-step == step-1 emissions",
    )
    check(
        len(two) / step1_emitted > 0.5,
        f"step-2 survival {len(two)}/{step1_emitted} exceeds half",
    )

    violations = []
    nondegrading = []
    gains1 = []
    gains2 = []
    for record, route, block in blocks:
        for step in block["steps"]:
            audit = step["oracle_audit"]
            if audit.get("certificate_violation"):
                violations.append((record["mix"], None) if False else (
                    record["mixing"], record["task_index"], route, step["step"]
                ))
            if not step["update_emitted"]:
                continue
            delta = audit["value_delta_vs_previous"]
            if min(delta) < -1e-12:
                nondegrading.append(
                    (record["mixing"], record["task_index"], route, step["step"])
                )
            (gains1 if step["step"] == 1 else gains2).append(audit["total_value_gain"])

    check(not violations, f"zero certificate violations ({len(violations)} found)")
    check(
        not nondegrading,
        f"every emitted step is componentwise non-degrading ({len(nondegrading)} bad)",
    )
    check(len(gains1) == 22, f"22 step-1 gains recorded (found {len(gains1)})")
    check(len(gains2) == 20, f"20 step-2 gains recorded (found {len(gains2)})")
    check(all(g > 0 for g in gains2), "every step-2 gain is strictly positive")
    check(
        abs(sum(gains2) / len(gains2) - summary["mean_gain_step2"]) < 1e-12,
        f"mean step-2 gain {sum(gains2) / len(gains2):.6f} matches summary",
    )
    check(
        min(gains2) > 0 and abs(min(gains2) - summary["min_gain_step2"]) < 1e-12,
        f"minimum step-2 gain {min(gains2):.6f} matches summary",
    )

    REPORT.append("\n4. Attrition reasons are from the frozen list")
    frozen_reasons = {
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
        if block["emitted_steps"] >= 2:
            continue
        seen.update(block["steps"][-1]["ordered_reasons"])
    check(
        seen <= frozen_reasons,
        f"every abstention reason is in the frozen list (offenders: {seen - frozen_reasons})",
    )
    check(
        seen == {"improvement_lcb_nonpositive"},
        f"the only stopping reason observed is improvement_lcb_nonpositive (found {seen})",
    )

    REPORT.append("\n5. H5: no sealed file changed")
    recorded = environment["sealed_file_hashes"]
    for name, digest in recorded.items():
        check(sha256(PROJECT / name) == digest, f"{name} byte-identical")

    REPORT.append("\n6. Replay of the sealed programs")
    for script, extra in (
        ("analyze_fp_iter2_001.py", ["--result-dir", str(FORMAL)]),
        ("verify_variance_adaptive_certificate.py", []),
        ("verify_fp_scale_002_same_actor.py", []),
        ("verify_fp_attn_001_same_actor.py", []),
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

    REPORT.append("\n" + "=" * 68)
    REPORT.append("SUMMARY")
    REPORT.append(
        f"  route-records                 : {len(blocks)}\n"
        f"  step-1 emissions              : {step1_emitted} (sealed 22)\n"
        f"  step-1 reproduction failures  : {len(repro_bad)}\n"
        f"  emitted two steps             : {len(two)}\n"
        f"  emitted one step              : {len(one)}\n"
        f"  emitted zero steps            : {len(zero)}\n"
        f"  step-2 survival of step-1     : {len(two)}/{step1_emitted}\n"
        f"  mean gain step 1 / step 2     : {sum(gains1) / len(gains1):.6f} / "
        f"{sum(gains2) / len(gains2):.6f}\n"
        f"  min  gain step 1 / step 2     : {min(gains1):.6f} / {min(gains2):.6f}\n"
        f"  certificate violations        : {len(violations)}\n"
        f"  non-degrading violations      : {len(nondegrading)}"
    )
    REPORT.append("=" * 68)
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
        / "FP-ITER2-001"
        / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
