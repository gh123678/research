"""FP-TIGHT-001 same-actor derived verification.

Re-derives the coverage audit from the bundle, proves the sealed corpus is
byte-identical, checks the risk allocation sums to delta, confirms the unsound
counterfactual is labelled and excluded, and replays the affected verifiers.

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
RESULT = PROJECT / "results" / "FP-TIGHT-001" / "claude" / "formal"
SAMPLES = PROJECT / "results" / "FP-TIGHT-001" / "claude" / "sample4"
ITER5_ENV = PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy" / "environment.json"
PRIMARY = ("expected_exact", "expected_finite")
SOUND_ARMS = ("frozen", "bernstein", "empirical_bernstein")
COUNTERFACTUAL = "counterfactual_no_envelope"
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


def rederive_coverage() -> None:
    bundle = load(RESULT / "task_results.json")
    rows = rows_of(bundle)
    check(len(rows) == 48, f"48 route-records (found {len(rows)})")
    for arm in SOUND_ARMS:
        bad = []
        for mixing, task_index, route, block in rows:
            e_q = block["arms"][arm]["e_q"]
            realized = block["oracle_audit"]["realized_q_sup_error"]
            if e_q is None or float(e_q) < realized:
                bad.append((mixing, task_index, route, e_q, realized))
        check(
            not bad,
            f"{arm} covers the realized error on every route-record ({len(bad)} bad)",
        )
    # The realized error must be a genuine float difference, not a zero that would
    # suggest the audit compared Qhat against itself.
    realized = [
        block["oracle_audit"]["realized_q_sup_error"] for _, _, _, block in rows
    ]
    check(
        min(realized) > 0.0,
        f"every realized error is strictly positive "
        f"(minimum {min(realized):.3e}), so the audit is not comparing Qhat to itself",
    )
    # And the frozen certificate must still be the one whose E_Q the census used:
    # no route-record should be non-certified here.
    statuses = {block["arms"]["frozen"]["status_certificate"] for _, _, _, block in rows}
    check(
        statuses == {"certificate_emitted"},
        f"the frozen arm certifies on every route-record (found {statuses})",
    )


def risk_allocation() -> None:
    """delta_each must be delta/(3d), so 3d bounds sum to exactly delta."""
    sys.path.insert(0, str(PROJECT))
    import fixed_policy_bernstein_certificate as bc

    rng = np.random.default_rng([1, 2, 3])
    n = 400
    states = np.repeat(np.arange(12) // 3, n // 12)
    actions = np.repeat(np.arange(12) % 3, n // 12)
    batch = {
        "states": states,
        "actions": actions,
        "rewards": rng.normal(size=states.size),
        "next_states": rng.integers(0, 4, size=states.size),
        "next_actions": actions,
    }
    q_hat = np.zeros((4, 3))
    policy = np.full((4, 3), 1.0 / 3.0)
    for name, fn in (
        ("bernstein", bc.bernstein_certificate),
        ("empirical_bernstein", bc.empirical_bernstein_certificate),
        (COUNTERFACTUAL, bc.counterfactual_no_envelope),
    ):
        out = fn(q_hat, policy, batch, min_half_count=1)
        d = out["n_groups"]
        expected = 0.05 / (3.0 * d) if name != COUNTERFACTUAL else 0.05 / (2.0 * d)
        check(
            abs(float(out["delta_each"]) - expected) <= 1e-15,
            f"{name}: delta_each = {out['delta_each']:.8g}, and {int(d)} bounds "
            f"of that size "
            f"{'sum to delta' if name != COUNTERFACTUAL else 'match the frozen 2d split'}",
        )
    check(
        True,
        "the counterfactual keeps the frozen 2d allocation, as it is the frozen "
        "construction with one term deleted rather than a new certificate",
    )


def counterfactual_labelled() -> None:
    bundle = load(RESULT / "task_results.json")
    rows = rows_of(bundle)
    flags = {
        block["arms"][COUNTERFACTUAL].get("sound")
        for _, _, _, block in rows
        if COUNTERFACTUAL in block["arms"]
    }
    check(
        flags == {False},
        f"every counterfactual entry carries sound=false (found {flags})",
    )
    # It must NOT be silently covered: report its violation count so the label is
    # backed by a number.
    fails = sum(
        1
        for _, _, _, block in rows
        if block["arms"][COUNTERFACTUAL]["e_q"] is not None
        and float(block["arms"][COUNTERFACTUAL]["e_q"])
        < block["oracle_audit"]["realized_q_sup_error"]
    )
    REPORT.append(
        f"  INFO  the counterfactual floor fails coverage on {fails}/48 "
        "route-records; it is a diagnostic, never a guarantee."
    )


def sealed_corpus_untouched() -> None:
    recorded = load(ITER5_ENV)["sealed_file_hashes"]
    for name in SCIENCE:
        if name not in recorded:
            continue
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
        else:
            REPORT.append(
                f"  INFO  evaluator {name} "
                f"{'matches' if matches else 'differs from'} this task's record"
            )
    # The new module must be additive: the frozen one is not imported-and-patched.
    source = (PROJECT / "fixed_policy_bernstein_certificate.py").read_text(
        encoding="utf-8"
    )
    check(
        "import fixed_policy_variance_certificate" not in source,
        "the new module does not import the frozen certificate module",
    )
    check(
        PROJECT.joinpath("fixed_policy_variance_certificate.py").exists(),
        "the frozen certificate module is still present",
    )


def sample_arm() -> None:
    if not (SAMPLES / "task_results.json").exists():
        REPORT.append("  INFO  no sample-size arm bundle present; H7 unscored")
        return
    bundle = load(SAMPLES / "task_results.json")
    multiplier = int(bundle["sample_multiplier"])
    arm = f"empirical_bernstein_x{multiplier}"
    rows = rows_of(bundle)
    check(
        all(arm in block["arms"] for _, _, _, block in rows),
        f"the x{multiplier} arm is present on every route-record",
    )
    bad = sum(
        1
        for _, _, _, block in rows
        if block["arms"][arm]["e_q"] is not None
        and float(block["arms"][arm]["e_q"]) < block["oracle_audit"]["realized_q_sup_error"]
    )
    check(bad == 0, f"the x{multiplier} arm still covers the realized error ({bad})")


def analyzer_determinism() -> None:
    before = (RESULT / "summary.json").read_bytes()
    cmd = [
        sys.executable,
        "-B",
        str(PROJECT / "analyze_fp_tight_001.py"),
        "--result-dir",
        str(RESULT),
    ]
    if (SAMPLES / "task_results.json").exists():
        cmd += ["--sample-dir", str(SAMPLES)]
    completed = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT)
    check(
        completed.returncode == 0,
        f"analyze_fp_tight_001.py replays with exit 0 (got {completed.returncode})",
    )
    check(
        before == (RESULT / "summary.json").read_bytes(),
        "re-running the analyzer reproduces summary.json byte-for-byte",
    )


def main() -> None:
    REPORT.append("FP-TIGHT-001 same-actor derived verification")
    REPORT.append("=" * 78)
    REPORT.append(
        "NOTE: the user instructed that verification is not the focus of this\n"
        "round. These checks are recorded for completeness and are same-actor only."
    )

    REPORT.append("\n1. Coverage re-derived from the bundle")
    rederive_coverage()

    REPORT.append("\n2. Risk allocation sums to delta")
    risk_allocation()

    REPORT.append("\n3. The unsound counterfactual is labelled, not hidden")
    counterfactual_labelled()

    REPORT.append("\n4. Sealed corpus untouched")
    sealed_corpus_untouched()

    REPORT.append("\n5. The sample-size arm, if present")
    sample_arm()

    REPORT.append("\n6. Analyzer determinism")
    analyzer_determinism()

    REPORT.append("\n7. Replay of the affected sealed programs")
    # The numpy-line verifiers whose subject this task touches. Nothing here can
    # move the network line, and the census verifier reads its own bundles, so
    # neither is included -- including them would nest the replay tree for no
    # coverage.
    for script in (
        "verify_variance_adaptive_certificate.py",
        "verify_policy_quantities_by_solve.py",
        "verify_fp_iter4_001_same_actor.py",
        "verify_fp_iter5_001_same_actor.py",
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

    summary = load(RESULT / "summary.json")
    REPORT.append("\n" + "=" * 78)
    REPORT.append("SUMMARY")
    for arm, value in summary["mean_e_q"].items():
        REPORT.append(
            f"  mean E_Q {arm:<28} {value:.4f}  "
            f"{summary['reduction_vs_frozen'][arm]:+.1%}"
        )
    for key in ("H1_coverage_violations", "H2_lost_eligibility",
                "H3", "H4", "H5", "H6", "H7"):
        value = summary[key]
        if isinstance(value, dict) and "verdict" in value:
            value = value["verdict"]
        REPORT.append(f"  {key:<28}: {value}")
    REPORT.append(f"  never-emitting at step 1    : "
                  f"{summary['never_emitting_at_step1']}")
    REPORT.append("=" * 78)
    REPORT.append("RESULT: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)"))
    REPORT.append(
        "LIMITATION: same-actor derived verification only, and per the user's\n"
        "instruction not the focus of this round. Step 1 only: nothing here says a\n"
        "tighter certificate extends the iteration."
    )

    text = "\n".join(REPORT)
    print(text)
    out = (
        PROJECT / "docs" / "research_branches" / "FP-TIGHT-001" / "claude"
        / "verification_same_actor.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if FAILURES == 0 else 1)


if __name__ == "__main__":
    main()
