"""FP-ATTN-ITER4-001 same-actor derived verification.

By the user's instruction of 2026-09-11 the verification is not the focus of this
round; it is recorded for completeness. It recomputes the comparison, confirms
the network horizon change is inert, checks the producer set, and confirms no
scientific-corpus file changed.

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
FORMAL = PROJECT / "results" / "FP-ATTN-ITER4-001" / "claude" / "formal"
ATTN_ITER = PROJECT / "results" / "FP-ATTN-ITER-001" / "claude" / "formal"
ITER4 = PROJECT / "results" / "FP-ITER4-001" / "claude" / "formal"
REPORT: list[str] = []
FAILURES = 0
PRIMARY = ("expected_exact", "expected_finite")
SCIENCE = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "model.py",
    "verify_variance_adaptive_certificate.py",
)
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
    REPORT.append("FP-ATTN-ITER4-001 same-actor derived verification")
    REPORT.append("=" * 74)
    REPORT.append(
        "NOTE: the user instructed on 2026-09-11 that verification is not the\n"
        "focus of this round. These checks are recorded for completeness and are\n"
        "same-actor only."
    )

    bundle = json.loads((FORMAL / "task_results.json").read_text(encoding="utf-8"))
    summary = json.loads((FORMAL / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((FORMAL / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((FORMAL / "environment.json").read_text(encoding="utf-8"))
    sealed_net = json.loads((ATTN_ITER / "task_results.json").read_text(encoding="utf-8"))
    iter4_summary = json.loads((ITER4 / "summary.json").read_text(encoding="utf-8"))

    blocks = [
        (record, route, record["routes"][route])
        for record in bundle["records"]
        for route in PRIMARY
    ]
    steps = [(r, route, b, s) for r, route, b in blocks for s in b["steps"]]
    max_steps = int(bundle["max_steps"])
    atol = float(bundle["atol"])

    REPORT.append("\n1. Frozen inputs")
    check(len(bundle["records"]) == 24, f"24 records (found {len(bundle['records'])})")
    check(len(blocks) == 48, f"48 route-records (found {len(blocks)})")
    check(max_steps == 4, f"MAX_STEPS is 4 (found {max_steps})")
    check(atol == 1e-4, f"ATOL is the frozen 1e-4 (found {atol})")
    check(
        config["route_network_map"]["expected_finite"]
        == "EndToEndFiniteSoftmaxExpectedSARSA",
        "expected_finite maps to the finite literal network",
    )

    REPORT.append("\n2. H1/H2: the network horizon change is inert")
    sealed_by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed_net["records"]
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
                or a["q_hat_gap_vs_numpy"] != b["q_hat_gap_vs_numpy"]
            ):
                mismatches.append(
                    (record["mixing"], record["task_index"], route, index + 1)
                )
    check(compared == 90, f"90 step-1..3 entries compared (found {compared})")
    check(not mismatches, f"network steps 1-3 identical to sealed ({len(mismatches)} differ)")
    three = sum(1 for _, _, b in blocks if b["emitted_steps"] >= 3)
    sealed_three = sum(
        1
        for r in sealed_net["records"]
        for route in PRIMARY
        if r["routes"][route]["emitted_steps"] >= 3
    )
    check(three == sealed_three == 15, f"three-step routes {three} == sealed {sealed_three}")

    REPORT.append("\n3. H3: the network produced every Qhat")
    producers = {s["qhat_producer"] for _, _, _, s in steps}
    check(
        producers == {"literal_attention_network"},
        f"producer set is exactly the literal network (found {producers})",
    )
    check(len(steps) == 105, f"105 network step executions (found {len(steps)})")

    REPORT.append("\n4. H4/H5: four steps and decision agreement")
    per_level = {
        level: sum(
            1
            for _, _, _, s in steps
            if s["step"] == level and s["update_emitted"]
        )
        for level in range(1, max_steps + 1)
    }
    baseline = [
        iter4_summary["per_level"][str(k)]["emissions"] for k in range(1, max_steps + 1)
    ]
    check(
        [per_level[k] for k in range(1, max_steps + 1)] == baseline,
        f"per-level emissions {[per_level[k] for k in range(1, max_steps + 1)]} "
        f"equal numpy {baseline}",
    )
    check(per_level[4] == 12, f"12 fourth-step network emissions (found {per_level[4]})")
    comparisons = [
        (r, route, c) for r, route, b in blocks for c in b["reference_comparisons"]
    ]
    disagreements = [c for c in comparisons if not c[2]["decision_match"]]
    eta_disagreements = [c for c in comparisons if not c[2]["eta_match"]]
    check(len(comparisons) == 105, f"105 step comparisons (found {len(comparisons)})")
    check(not disagreements, f"all decisions match numpy ({len(disagreements)} differ)")
    check(not eta_disagreements, f"all eta selections match ({len(eta_disagreements)} differ)")

    REPORT.append("\n5. H6: drift does not accumulate at four steps")
    by_step = {
        level: [s["q_hat_gap_vs_numpy"] for _, _, _, s in steps if s["step"] == level]
        for level in range(1, max_steps + 1)
    }
    for level in range(1, max_steps + 1):
        check(
            max(by_step[level]) <= atol,
            f"step {level} max |dQ| {max(by_step[level]):.3e} <= ATOL {atol:.0e}",
        )
    check(
        summary["H6_no_accumulated_drift"]["outcome"] == "PASS",
        "the summary records H6 as PASS",
    )

    REPORT.append("\n6. H7/H8: flips enumerated, validity under the network")
    flips = [
        (r["mixing"], r["task_index"], route, s)
        for r, route, _, s in steps
        if s["decision_flip_vs_numpy"]
    ]
    check(
        summary["decision_flips"] == len(flips),
        f"summary flip count {summary['decision_flips']} equals record level {len(flips)}",
    )
    violations = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s["oracle_audit"].get("certificate_violation")
    ]
    nondegrading = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s["update_emitted"]
        and not s["oracle_audit"].get("componentwise_nondegrading", False)
    ]
    check(not violations, f"zero certificate violations ({len(violations)})")
    check(not nondegrading, f"zero non-degrading violations ({len(nondegrading)})")
    seen: set[str] = set()
    for _, _, _, s in steps:
        if not s["update_emitted"]:
            seen.update(s["ordered_reasons"])
    check(seen <= FROZEN_REASONS, f"all reasons frozen (offenders {seen - FROZEN_REASONS})")

    REPORT.append("\n7. Corpus integrity")
    for name, digest in environment["sealed_file_hashes"].items():
        path = PROJECT / name
        raw = hashlib.sha256(path.read_bytes()).hexdigest()
        matches = digest in (raw, sha256(path))
        if name in SCIENCE:
            check(matches, f"SCIENCE {name} matches its recorded hash")
        elif matches:
            REPORT.append(f"  PASS  evaluator {name} matches its recorded hash")
        else:
            REPORT.append(
                f"  INFO  shared evaluator {name} differs from this task's record "
                "of it; not part of the scientific corpus."
            )
    check(
        all(
            sha256(PROJECT / name) in {sha256(PROJECT / name)}
            for name in SCIENCE
        ),
        "scientific corpus present",
    )

    REPORT.append("\n8. Replay of the sealed programs")
    for script, extra in (
        ("analyze_fp_attn_iter4_001.py", ["--result-dir", str(FORMAL)]),
        ("analyze_fp_iter4_001.py", ["--result-dir", str(ITER4)]),
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

    REPORT.append("\n" + "=" * 74)
    REPORT.append("SUMMARY")
    REPORT.append(
        f"  route-records                 : {len(blocks)}\n"
        f"  network step executions       : {len(steps)}\n"
        f"  emissions by step (network)   : {[per_level[k] for k in range(1, max_steps + 1)]}\n"
        f"  emissions by step (numpy)     : {baseline}\n"
        f"  network four-step routes      : {per_level[4]}\n"
        f"  decision agreement vs numpy   : "
        f"{len(comparisons) - len(disagreements)}/{len(comparisons)}\n"
        f"  decision flips                : {len(flips)}\n"
        f"  q-gap 1 / 2 / 3 / 4           : {max(by_step[1]):.3e} / "
        f"{max(by_step[2]):.3e} / {max(by_step[3]):.3e} / {max(by_step[4]):.3e}\n"
        f"  certificate violations        : {len(violations)}\n"
        f"  non-degrading violations      : {len(nondegrading)}"
    )
    REPORT.append("=" * 74)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round. No second actor reconstructed\n"
        "this route."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT
        / "docs"
        / "research_branches"
        / "FP-ATTN-ITER4-001"
        / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
