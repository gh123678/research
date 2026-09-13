"""FP-GAP-001 same-actor derived verification.

Re-derives the reconstruction closure, the optimality of v*, the gap monotonicity and
the closed fraction from the bundles; checks the source bundle is unmodified; and
replays the foundation checks.

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
RESULT = PROJECT / "results" / "FP-GAP-001" / "claude" / "formal"
SOURCE = PROJECT / "results" / "FP-HORIZON-001" / "claude" / "formal" / "task_results.json"
ITER5_ENV = PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy" / "environment.json"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("frozen", "empirical_bernstein")
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


def main() -> None:
    REPORT.append("FP-GAP-001 same-actor derived verification")
    REPORT.append("=" * 92)
    REPORT.append(
        "NOTE: the user instructed that verification is not the focus of this\n"
        "round. These checks are recorded for completeness and are same-actor only."
    )

    bundle = load(RESULT / "task_results.json")
    summary = load(RESULT / "summary.json")

    REPORT.append("\n1. Frozen inputs")
    check(len(bundle["records"]) == 24, f"24 records (found {len(bundle['records'])})")
    check(
        bundle["source_bundle_sha256"]
        == hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "the source bundle's hash matches the one recorded at measurement time",
    )

    REPORT.append("\n2. H1 re-derived: every trajectory reconstructs")
    worst = 0.0
    count = 0
    for record in bundle["records"]:
        for route in PRIMARY:
            for arm in ARMS:
                traj = record["routes"][route][arm]
                count += 1
                worst = max(worst, traj["reconstruction_gap"])
    check(count == 96, f"96 trajectories inspected (found {count})")
    check(
        worst <= 1e-12,
        f"worst reconstruction discrepancy {worst:.3e} <= 1e-12",
    )
    check(
        abs(worst - summary["worst_reconstruction_gap"]) <= 1e-18,
        "the summary's worst discrepancy matches",
    )

    REPORT.append("\n3. H2 re-derived: v* is optimal and dominates")
    worst_res = 0.0
    violations = 0
    for record in bundle["records"]:
        worst_res = max(worst_res, float(record["bellman_residual_of_v_star"]))
        v_star = np.asarray(record["v_star"], dtype=np.float64)
        for route in PRIMARY:
            for arm in ARMS:
                v0 = np.asarray(
                    record["routes"][route][arm]["start_policy_value"],
                    dtype=np.float64,
                )
                if float(np.min(v_star - v0)) < -1e-9:
                    violations += 1
    check(
        worst_res <= 1e-12,
        f"v* satisfies the Bellman optimality equation (worst residual {worst_res:.3e})",
    )
    check(violations == 0, f"v* dominates v^pi_0 everywhere ({violations} violations)")

    REPORT.append("\n4. H3 re-derived: the gap is monotone")
    rises = 0
    for record in bundle["records"]:
        for route in PRIMARY:
            for arm in ARMS:
                seq = record["routes"][route][arm]["gap_trajectory"]
                rises += sum(
                    1 for k in range(1, len(seq)) if seq[k] > seq[k - 1] + 1e-12
                )
    check(rises == 0, f"no rise in any gap trajectory ({rises} rises)")

    REPORT.append("\n5. The headline number, re-derived independently")
    # Recompute the long-trajectory closure from the bundle rather than reading the
    # analyzer's summary.
    closed = []
    last_ratio = []
    for record in bundle["records"]:
        for route in PRIMARY:
            for arm in ARMS:
                traj = record["routes"][route][arm]
                seq = traj["gap_trajectory"]
                if len(seq) - 1 < 24 or seq[0] <= 0:
                    continue
                closed.append((seq[0] - seq[-1]) / seq[0])
                if seq[-1] > 0:
                    last_ratio.append((seq[-2] - seq[-1]) / seq[-1])
    mean_closed = sum(closed) / len(closed)
    median_ratio = sorted(last_ratio)[len(last_ratio) // 2]
    REPORT.append(
        f"  {len(closed)} long trajectories (>= 24 emitted steps): mean fraction of "
        f"the initial gap closed {mean_closed:.4%}"
    )
    REPORT.append(
        f"  median (last gain / remaining gap) {median_ratio:.3f} — the geometric "
        f"decay factor, which is what a converging process shows"
    )
    check(
        mean_closed > 0.99,
        f"long trajectories close more than 99% of the gap ({mean_closed:.4%})",
    )
    check(
        0.2 < median_ratio < 0.8,
        f"the decay factor is a stable interior value, not a vanishing one "
        f"({median_ratio:.3f})",
    )

    REPORT.append("\n6. The mis-specified metrics are recorded as such")
    check(
        summary["H4"]["verdict"] == "FALSIFIED"
        and summary["H5"]["verdict"] == "FALSIFIED"
        and summary["H6"]["verdict"] == "FALSIFIED",
        "the summary records H4, H5 and H6 as falsified rather than recasting them",
    )
    check(
        summary.get("length_breakdown_posthoc") is not None,
        "the length-conditioned breakdown is recorded alongside the pooled verdicts",
    )
    check(
        "relative-softmax family with pi_min = 0.15" in summary["policy_class_caveat"],
        "every floor statement carries the policy-class caveat",
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
        [sys.executable, "-B", str(PROJECT / "analyze_fp_gap_001.py"),
         "--result-dir", str(RESULT)],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    check(
        completed.returncode == 0,
        f"analyze_fp_gap_001.py replays with exit 0 (got {completed.returncode})",
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
    REPORT.append(f"  H1 reconstruction : {worst:.3e}")
    REPORT.append(f"  H2 v* optimal     : worst Bellman residual {worst_res:.3e}")
    REPORT.append(f"  H3 gap monotone   : {rises} rises")
    REPORT.append(
        f"  long trajectories : {len(closed)} closing {mean_closed:.4%} of the gap"
    )
    REPORT.append(f"  decay factor      : {median_ratio:.3f}")
    REPORT.append(
        f"  registered verdicts: H4 {summary['H4']['verdict']}, "
        f"H5 {summary['H5']['verdict']}, H6 {summary['H6']['verdict']}"
    )
    REPORT.append("=" * 92)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round. The floor is a joint property of the\n"
        "policy class and the iteration; this task does not decompose the two."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT / "docs" / "research_branches" / "FP-GAP-001" / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
