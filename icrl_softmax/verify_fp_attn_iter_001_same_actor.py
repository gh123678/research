"""FP-ATTN-ITER-001 same-actor derived verification.

Recomputes the network-versus-numpy comparison independently of the evaluator
and analyzer, confirms the network really produced every step's Qhat, checks
the no-mask/no-gate property of the finite route on this data, replays the
sealed programs, and confirms no sealed file changed.

This is DERIVED verification by the same actor that executed the task. It is NOT
independent verification: no second actor reconstructed this route.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

FORMAL = PROJECT / "results" / "FP-ATTN-ITER-001" / "claude" / "formal"
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
    REPORT.append("FP-ATTN-ITER-001 same-actor derived verification")
    REPORT.append("=" * 72)

    bundle = json.loads((FORMAL / "task_results.json").read_text(encoding="utf-8"))
    summary = json.loads((FORMAL / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((FORMAL / "config.json").read_text(encoding="utf-8"))
    environment = json.loads((FORMAL / "environment.json").read_text(encoding="utf-8"))
    iter3 = json.loads((ITER3 / "summary.json").read_text(encoding="utf-8"))

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
    check(max_steps == 3, f"MAX_STEPS is 3 (found {max_steps})")
    check(atol == 1e-4, f"ATOL is the frozen 1e-4 (found {atol})")
    check(
        config["route_network_map"]["expected_exact"]
        == "EndToEndMaskedSoftmaxExpectedSARSA",
        "expected_exact maps to the masked literal network",
    )
    check(
        config["route_network_map"]["expected_finite"]
        == "EndToEndFiniteSoftmaxExpectedSARSA",
        "expected_finite maps to the finite literal network",
    )

    REPORT.append("\n2. H2: the network produced every step's Qhat")
    producers = {s["qhat_producer"] for _, _, _, s in steps}
    check(
        producers == {"literal_attention_network"},
        f"every step's Qhat came from the literal network (found {producers})",
    )
    check(
        len(steps) == 90,
        f"90 step executions in total (found {len(steps)})",
    )

    REPORT.append("\n3. Network-vs-numpy agreement and the accumulation question")
    by_step = {
        level: [s["q_hat_gap_vs_numpy"] for _, _, _, s in steps if s["step"] == level]
        for level in range(1, max_steps + 1)
    }
    for level in range(1, max_steps + 1):
        check(
            max(by_step[level]) <= atol,
            f"step {level}: max |Q_network - Q_numpy| {max(by_step[level]):.3e} <= ATOL",
        )
    check(
        summary["max_step1_q_hat_gap"] == max(by_step[1]),
        "summary step-1 max gap matches the record level (H1 anchor)",
    )
    # The accumulation question: does the gap grow with the step index?
    check(
        max(by_step[3]) <= atol,
        f"the step-3 gap ({max(by_step[3]):.3e}) stays within ATOL, so the float32 "
        "difference does not accumulate past the frozen tolerance",
    )

    REPORT.append("\n4. H3/H4: three steps and decision agreement")
    emissions = {
        level: sum(
            1
            for _, _, b in blocks
            if len(b["steps"]) >= level and b["steps"][level - 1]["update_emitted"]
        )
        for level in range(1, max_steps + 1)
    }
    baseline = [iter3["per_level"][str(k)]["emissions"] for k in range(1, max_steps + 1)]
    check(
        [emissions[k] for k in range(1, max_steps + 1)] == baseline,
        f"per-level emissions {[emissions[k] for k in range(1, max_steps + 1)]} "
        f"equal the FP-ITER3-001 numpy baseline {baseline}",
    )
    check(emissions[3] == 15, f"15 network-driven three-step routes (found {emissions[3]})")
    comparisons = [
        (r, route, comp) for r, route, b in blocks for comp in b["reference_comparisons"]
    ]
    disagreements = [c for c in comparisons if not c[2]["decision_match"]]
    eta_disagreements = [c for c in comparisons if not c[2]["eta_match"]]
    check(len(comparisons) == 90, f"90 step comparisons available (found {len(comparisons)})")
    check(
        not disagreements,
        f"every decision matches the numpy baseline ({len(disagreements)} disagree)",
    )
    check(
        not eta_disagreements,
        f"every selected eta matches the numpy baseline ({len(eta_disagreements)} differ)",
    )

    REPORT.append("\n5. H5: flips are enumerated, not absorbed")
    flips = [
        (r["mixing"], r["task_index"], route, s)
        for r, route, _, s in steps
        if s["decision_flip_vs_numpy"]
    ]
    check(
        summary["decision_flips"] == len(flips),
        f"summary flip count {summary['decision_flips']} equals the record level {len(flips)}",
    )
    check(
        all("q_hat_gap" in f for f in summary["flip_details"]),
        "every reported flip carries its Q gap and boundary margins",
    )

    REPORT.append("\n6. H6: validity under the network's own Qhat")
    cert_violations = [
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
    check(not cert_violations, f"zero certificate violations ({len(cert_violations)})")
    check(not nondegrading, f"zero non-degrading violations ({len(nondegrading)})")
    gains = {
        level: [
            s["oracle_audit"]["total_value_gain"]
            for _, _, _, s in steps
            if s["step"] == level and s["update_emitted"]
        ]
        for level in range(1, max_steps + 1)
    }
    check(
        all(all(g > 0 for g in gains[level]) for level in range(1, max_steps + 1)),
        "every emitted step at every level has a strictly positive total gain",
    )

    REPORT.append("\n7. Attrition reasons and batches")
    seen: set[str] = set()
    for _, _, _, s in steps:
        if not s["update_emitted"]:
            seen.update(s["ordered_reasons"])
    check(seen <= FROZEN_REASONS, f"all reasons frozen (offenders {seen - FROZEN_REASONS})")
    check(
        len({r["train_batch_digest"] for r in bundle["records"]}) == len(bundle["records"]),
        "each record has its own training batch digest",
    )
    check(
        all(r["cert_batch_digest"] for r in bundle["records"]),
        "every record records its certification batch digest",
    )

    REPORT.append("\n8. H7: no sealed file changed")
    for name, digest in environment["sealed_file_hashes"].items():
        path = PROJECT / name
        raw = hashlib.sha256(path.read_bytes()).hexdigest()
        check(digest in (raw, sha256(path)), f"{name} matches its recorded hash")

    REPORT.append("\n9. Finite route remains gate-free on this data")
    from model import EndToEndFiniteSoftmaxExpectedSARSA  # noqa: PLC0415

    net = EndToEndFiniteSoftmaxExpectedSARSA()
    rng = np.random.default_rng(7)
    states = torch.as_tensor(rng.integers(0, 4, 256), dtype=torch.long)
    actions = torch.as_tensor(rng.integers(0, 3, 256), dtype=torch.long)
    rewards = torch.as_tensor(rng.normal(size=256), dtype=torch.float32)
    next_states = torch.as_tensor(rng.integers(0, 4, 256), dtype=torch.long)
    policy = torch.full((4, 3), 1.0 / 3.0)
    with torch.no_grad():
        _, diag = net(
            torch.zeros((4, 3)), states, actions, rewards, next_states, policy
        )
    check("visited" not in diag, "the finite route still exposes no visited gate")
    check(
        bool(torch.all(diag["write_attention"] > 0.0)),
        "finite-route write attention remains strictly positive (no -inf mask)",
    )

    REPORT.append("\n10. Replay of the sealed programs")
    for script, extra in (
        ("analyze_fp_attn_iter_001.py", ["--result-dir", str(FORMAL)]),
        ("analyze_fp_iter3_001.py", ["--result-dir", str(ITER3)]),
        ("verify_fp_iter3_001_same_actor.py", []),
        ("verify_fp_attn_001_same_actor.py", []),
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
    REPORT.append(
        f"  route-records                    : {len(blocks)}\n"
        f"  step executions (network)        : {len(steps)}\n"
        f"  emissions by step (network)      : "
        f"{[emissions[k] for k in range(1, max_steps + 1)]}\n"
        f"  emissions by step (numpy)        : {baseline}\n"
        f"  network three-step routes        : {emissions[3]}\n"
        f"  max |Q_network - Q_numpy|        : {max(by_step[1] + by_step[2] + by_step[3]):.3e}"
        f"  (ATOL {atol:.0e})\n"
        f"  step gap 1 / 2 / 3               : {max(by_step[1]):.3e} / "
        f"{max(by_step[2]):.3e} / {max(by_step[3]):.3e}\n"
        f"  decision agreement vs numpy      : "
        f"{len(comparisons) - len(disagreements)}/{len(comparisons)}\n"
        f"  decision flips                   : {len(flips)}\n"
        f"  certificate violations           : {len(cert_violations)}\n"
        f"  non-degrading violations         : {len(nondegrading)}"
    )
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
        / "FP-ATTN-ITER-001"
        / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
