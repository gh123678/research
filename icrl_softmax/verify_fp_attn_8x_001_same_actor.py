"""FP-ATTN-8X-001 same-actor derived verification.

Proves structurally that the decisions are taken on the literal network's Qhat,
re-derives validity, soundness, path agreement and the agreement headroom from the
bundles, confirms the sealed corpus is untouched, and replays the foundation checks.

By the user's instruction of 2026-09-11 the verification is not the focus of this
round; it is recorded for completeness. This is DERIVED verification by the same
actor that executed the task, NOT independent verification.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
RESULT = PROJECT / "results" / "FP-ATTN-8X-001" / "claude" / "formal"
NUMPY_8X = PROJECT / "results" / "FP-ITER8X-001" / "claude" / "formal" / "task_results.json"
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


def provenance_is_structural() -> None:
    """The call sites that DRIVE the iteration must take the literal network's Qhat.

    The evaluator legitimately computes a numpy comparator too, so finding
    ``q_literal`` somewhere in the file proves nothing. The check is on the
    assignments whose results are bound to ``certificate`` and ``decision``.
    """
    tree = ast.parse(
        (PROJECT / "evaluate_fp_attn_8x_001.py").read_text(encoding="utf-8")
    )
    inspected = 0
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if targets not in (["certificate"], ["decision"]):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        inspected += 1
        names = {sub.id for sub in ast.walk(value) if isinstance(sub, ast.Name)}
        if "q_literal" not in names:
            offenders.append(f"{targets[0]} <- {sorted(names)}")
        if "q_numpy" in names:
            offenders.append(f"{targets[0]} <- q_numpy")
    check(inspected >= 2, f"found the driving call sites ({inspected})")
    check(
        not offenders,
        f"the certificate and the decision are taken on the literal network's Qhat "
        f"({offenders})",
    )
    source = (PROJECT / "evaluate_fp_attn_8x_001.py").read_text(encoding="utf-8")
    check(
        "cert_numpy = certificate_for(arm, q_numpy" in source
        and "decision_numpy = fs.improvement_for" in source,
        "the numpy comparator still exists, so the confinement check is not vacuous",
    )


def main() -> None:
    REPORT.append("FP-ATTN-8X-001 same-actor derived verification")
    REPORT.append("=" * 92)
    REPORT.append(
        "NOTE: the user instructed that verification is not the focus of this\n"
        "round. These checks are recorded for completeness and are same-actor only."
    )

    bundle = load(RESULT / "task_results.json")
    summary = load(RESULT / "summary.json")
    horizon = int(bundle["max_steps"])
    rows = rows_of(bundle)
    numpy_rows = rows_of(load(NUMPY_8X))

    REPORT.append("\n1. Frozen inputs")
    check(len(rows) == 48, f"48 route-records (found {len(rows)})")
    check(horizon == 12, f"horizon 12 (found {horizon})")
    check(
        int(bundle["cert_chains"]) == 16384 * 8,
        f"certification is 8x (found {bundle['cert_chains']} chains)",
    )
    check(
        float(bundle["atol"]) == 1e-4,
        f"ATOL is the frozen 1e-4 (found {bundle['atol']})",
    )
    REPORT.append(f"  INFO  torch {bundle.get('torch')}")

    REPORT.append("\n2. Provenance: structural, not asserted")
    provenance_is_structural()
    steps_all = [
        (m, t, route, s)
        for m, t, route, block in rows
        for arm in ARMS
        for s in block[arm]["steps"]
    ]
    producers = {s.get("qhat_producer") for _, _, _, s in steps_all}
    check(
        producers == {"literal_attention_network"},
        f"all {len(steps_all)} step entries record the literal network "
        f"(found {producers})",
    )
    smallest_gap = min(s["q_hat_gap_vs_numpy"] for _, _, _, s in steps_all)
    check(
        smallest_gap > 0.0,
        f"every step shows a nonzero float32 gap, so no numpy substitution "
        f"(minimum {smallest_gap:.3e})",
    )

    REPORT.append("\n3. Validity and soundness re-derived")
    for arm in ARMS:
        emitted = [
            s
            for _, _, _, block in rows
            for s in block[arm]["steps"]
            if s["update_emitted"]
        ]
        check(
            all(
                s["oracle_audit"].get("componentwise_nondegrading", False)
                for s in emitted
            ),
            f"{arm}: {len(emitted)} emitted steps, all componentwise non-degrading",
        )
        check(
            all(s["oracle_audit"]["total_value_gain"] > 0.0 for s in emitted),
            f"{arm}: every emitted step strictly improving",
        )
        check(
            not any(s["oracle_audit"].get("certificate_violation") for s in emitted),
            f"{arm}: zero certificate violations",
        )
    for arm in ARMS:
        missing = sum(
            1
            for _, _, _, block in rows
            for s in block[arm]["steps"]
            if not s["update_emitted"] and not s["ordered_reasons"]
        )
        check(missing == 0, f"{arm}: {missing} abstentions without a reason")

    REPORT.append("\n4. Path agreement re-derived against the numpy bundle")
    total = 0
    for arm in ARMS:
        for level in range(1, horizon + 1):
            net = level_set(rows, arm, level)
            num = level_set(numpy_rows, arm, level)
            total += len(net - num) + len(num - net)
    check(
        total == summary["H4"]["set_disagreements"],
        f"the set-disagreement count {total} matches the summary",
    )
    check(
        (total == 0) == (summary["H4"]["verdict"] == "PASS"),
        "the H4 verdict is consistent with the re-derived count",
    )
    eta_flips = sum(
        1
        for _, _, _, block in rows
        for arm in ARMS
        for s in block[arm]["steps"]
        if s.get("eta_flip_vs_numpy")
    )
    check(
        eta_flips == summary["H5"]["eta_flips"],
        f"the eta-flip count {eta_flips} matches the summary",
    )

    REPORT.append("\n5. Drift and headroom re-derived")
    worst = max(s["q_hat_gap_vs_numpy"] for _, _, _, s in steps_all)
    check(
        abs(worst - summary["H6"]["worst_gap"]) < 1e-18,
        f"the worst gap {worst:.3e} matches the summary and is "
        f"{'within' if worst <= 1e-4 else 'OUTSIDE'} ATOL",
    )
    headrooms = []
    for level in range(1, horizon + 1):
        gains = [
            s["oracle_audit"]["total_value_gain"]
            for _, _, _, block in rows
            for s in block["frozen"]["steps"]
            if s["step"] == level and s["update_emitted"]
        ]
        gaps = [
            s["q_hat_gap_vs_numpy"]
            for _, _, _, block in rows
            for s in block["frozen"]["steps"]
            if s["step"] == level
        ]
        if gains and gaps:
            headrooms.append(min(gains) / max(gaps))
    check(
        headrooms
        and abs(min(headrooms) - summary["H7"]["worst_headroom"]) < 1e-9,
        f"the worst headroom {min(headrooms):.1f}x matches the summary",
    )
    REPORT.append(
        "  INFO  headroom is the quantity that decides whether path agreement is a "
        "statement about the certificate or about float32 arithmetic."
    )

    REPORT.append("\n6. Sealed corpus untouched")
    recorded = load(ITER5_ENV)["sealed_file_hashes"]
    for name in SCIENCE:
        if name in recorded:
            check(
                recorded[name] == sha256(PROJECT / name),
                f"SCIENCE {name} is byte-identical to the FP-ITER5-001 record",
            )
    # The pre-existing SEALED iteration evaluators must stay on the sealed
    # certificate. evaluate_fp_iter8x_001.py is deliberately not in this list: it
    # runs both arms by design, and an earlier version of this check failed on it.
    for evaluator in ("evaluate_fp_iter2_001.py", "evaluate_fp_attn_iter_001.py",
                      "evaluate_fp_census_001.py"):
        source = (PROJECT / evaluator).read_text(encoding="utf-8")
        check(
            "fixed_policy_bernstein_certificate" not in source,
            f"{evaluator} still uses only the sealed certificate",
        )

    REPORT.append("\n7. Analyzer determinism")
    before = (RESULT / "summary.json").read_bytes()
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "analyze_fp_attn_8x_001.py"),
         "--result-dir", str(RESULT), "--numpy-dir", str(NUMPY_8X)],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(
        completed.returncode == 0,
        f"analyze_fp_attn_8x_001.py replays with exit 0 (got {completed.returncode})",
    )
    check(
        before == (RESULT / "summary.json").read_bytes(),
        "re-running the analyzer reproduces summary.json byte-for-byte",
    )

    REPORT.append("\n8. Replay of the foundation checks")
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
    for arm in ARMS:
        REPORT.append(
            f"  network {arm:<22}: {summary['emissions_network'][arm]}"
        )
        REPORT.append(
            f"  numpy   {arm:<22}: {summary['emissions_numpy'][arm]}"
        )
    REPORT.append(f"  deepest: network {summary['deepest_trajectory_network']}, "
                  f"numpy {summary['deepest_trajectory_numpy']}")
    REPORT.append(f"  H3 reach          : {summary['H3']}")
    REPORT.append(f"  H4 set agreement  : {summary['H4']['verdict']} "
                  f"({summary['H4']['set_disagreements']} disagreements)")
    REPORT.append(f"  H5 eta flips      : {summary['H5']['verdict']} "
                  f"({summary['H5']['eta_flips']})")
    REPORT.append(f"  H6 drift          : {summary['H6']['verdict']} "
                  f"({summary['H6']['worst_gap']:.3e})")
    REPORT.append(f"  H7 headroom       : {summary['H7']['verdict']} "
                  f"({summary['H7']['worst_headroom']:.1f}x)")
    REPORT.append("=" * 92)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round. The 8x batch is a fresh independent\n"
        "sample, not a superset of the sealed one."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT / "docs" / "research_branches" / "FP-ATTN-8X-001" / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
