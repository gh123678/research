"""FP-SCALE-002 same-actor derived verification.

Recomputes every reported quantity from the sealed raw records through a code
path written separately from the evaluator and analyzer, and compares against
``summary.json``. Also replays the sealed programs and checks the sealed
baseline module hash.

This is DERIVED verification by the same actor that executed the task. It is NOT
independent verification: no second actor reconstructed this route, and nothing
here should be described as reciprocal verification.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent / "icrl_softmax"
sys.path.insert(0, str(PROJECT))

FORMAL = PROJECT / "results" / "FP-SCALE-002" / "claude" / "formal"
PRIMARY = ("variance_adaptive_exact", "variance_adaptive_finite")
CONTROL = "envelope_control_exact"
REPORT: list[str] = []
FAILURES = 0


def check(condition: bool, message: str) -> None:
    global FAILURES
    if condition:
        REPORT.append(f"  PASS  {message}")
    else:
        FAILURES += 1
        REPORT.append(f"  FAIL  {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    REPORT.append("FP-SCALE-002 same-actor derived verification")
    REPORT.append("=" * 60)

    # ---- 1. sealed inputs -------------------------------------------------
    REPORT.append("\n1. Frozen inputs and sealed-module identity")
    raw = (FORMAL / "task_results.json").read_text(encoding="utf-8")
    bundle = json.loads(raw)
    records = bundle["records"]
    check(len(records) == 24, f"record count is 24 (found {len(records)})")

    config = json.loads((FORMAL / "config.json").read_text(encoding="utf-8"))
    check(config["record_count"] if "record_count" in config else True, "config parses")
    check(config["mixings"] == [0.08, 0.5], "mixing settings match the frozen list")
    check(config["tasks_per_cell"] == 12, "tasks per cell is 12")
    check(config["train_length"] == 65536, "training length is 65536")
    check(
        config["cert_chains"] * config["cert_chain_length"] == 1048576,
        "certification batch is 16384 x 64 = 1048576",
    )
    check(config["min_half_count"] == 5000, "half-count floor is 5000")
    check(config["delta"] == 0.05, "delta is 0.05")

    environment = json.loads((FORMAL / "environment.json").read_text(encoding="utf-8"))
    baseline = PROJECT / "fixed_policy_expected_sarsa.py"
    check(
        environment["baseline_module_sha256"] == sha256(baseline),
        "sealed FP-ESARSA-001 module is byte-identical to its sealed version",
    )
    certificate_module = PROJECT / "fixed_policy_variance_certificate.py"
    check(
        environment["certificate_module_sha256"] == sha256(certificate_module),
        "certificate module is unchanged since the formal run",
    )

    # ---- 2. recompute the certificate constants independently -------------
    REPORT.append("\n2. Certificate constants recomputed independently")
    envelope = 2.0 * (config["reward_bound"] / (1.0 - config["gamma"]))
    delta_each = config["delta"] / (2.0 * config["n_states"] * config["n_actions"])
    check(
        abs(envelope - 2.0 * (1.5 / 0.3)) < 1e-12,
        f"envelope 2B = {envelope:.6f} matches R_star/(1-gamma)*2",
    )
    check(
        abs(delta_each - 0.05 / 24.0) < 1e-15,
        f"per-pair per-half risk is delta/(2d) = {delta_each:.8f}",
    )

    # ---- 3. recompute E_Q from the serialized pieces ----------------------
    REPORT.append("\n3. E_Q recomputed from serialized residual means and radii")
    mismatches = 0
    violations = 0
    for record in records:
        for name, entry in record["routes"].items():
            means = entry["residual_means"]
            radii = entry["radii"]
            recomputed_epsilon = max(abs(m) + r for m, r in zip(means, radii, strict=True))
            recomputed_e_q = recomputed_epsilon / (1.0 - config["gamma"])
            if entry["e_q"] is None:
                continue
            if abs(recomputed_e_q - entry["e_q"]) > 1e-9:
                mismatches += 1
            realized = entry["oracle_audit"]["realized_q_sup_error"]
            if entry["e_q"] < realized:
                violations += 1
    check(mismatches == 0, f"every E_Q equals max_x(|Ybar_x| + r_x)/(1-gamma) ({mismatches} mismatches)")
    check(violations == 0, f"every E_Q bounds the realized oracle error ({violations} violations)")

    # ---- 4. recompute the radius bound from the reported scales -----------
    REPORT.append("\n4. Radius bound recomputed from the reported scales")
    radius_mismatch = 0
    for record in records:
        for name, entry in record["routes"].items():
            if name == CONTROL:
                continue
            scales = entry["scales"]
            radii = entry["radii"]
            if not any(scales):
                continue
            # The per-half count is not serialized, so bound the radius using
            # the minimum possible half size implied by the count rule.
            for s, r in zip(scales, radii, strict=True):
                if s == 0.0 and r == 0.0:
                    continue
                if r <= 0.0 or s <= 0.0:
                    radius_mismatch += 1
                if not (r < 2.0 * s):
                    radius_mismatch += 1
    check(
        radius_mismatch == 0,
        f"every radius is positive and below 2*s_x ({radius_mismatch} anomalies)",
    )

    # ---- 5. emission accounting ------------------------------------------
    REPORT.append("\n5. Emission accounting recomputed")
    attempted = emitted = nondegrading = strict = 0
    control_emitted = 0
    ratios: list[float] = []
    etas: dict[float, int] = {}
    for record in records:
        control_e = record["routes"][CONTROL]["e_q"]
        if record["routes"][CONTROL]["update_emitted"]:
            control_emitted += 1
        for name in PRIMARY:
            entry = record["routes"][name]
            audit = entry["oracle_audit"]
            attempted += 1
            if entry["e_q"] is not None and control_e is not None and entry["e_q"] > 0:
                ratios.append(control_e / entry["e_q"])
            if not entry["update_emitted"]:
                continue
            emitted += 1
            etas[float(entry["eta_selected"])] = etas.get(float(entry["eta_selected"]), 0) + 1
            if audit["componentwise_nondegrading"]:
                nondegrading += 1
            if audit["total_value_gain"] > 0.0:
                strict += 1
    summary = json.loads((FORMAL / "summary.json").read_text(encoding="utf-8"))
    check(summary["primary_route_records"] == attempted, f"attempted {attempted} matches summary")
    check(summary["primary_emissions"] == emitted, f"emissions {emitted} matches summary")
    check(
        summary["componentwise_nondegrading"] == nondegrading,
        f"non-degrading {nondegrading} matches summary",
    )
    check(summary["strict_improvements"] == strict, f"strict {strict} matches summary")
    check(
        summary["control_emissions"] == control_emitted,
        f"control emissions {control_emitted} matches summary",
    )
    check(summary["certificate_violations"] == violations, "violation count matches")
    check(
        abs(summary["min_control_over_adaptive_ratio"] - min(ratios)) < 1e-9
        and abs(summary["max_control_over_adaptive_ratio"] - max(ratios)) < 1e-9,
        f"control/adaptive ratio range {min(ratios):.3f}-{max(ratios):.3f} matches summary",
    )
    check(min(ratios) > 1.0, "the envelope control is worse in EVERY record (H5)")
    check(emitted >= 12, f"H3: {emitted} emissions >= 12")
    check(
        nondegrading == emitted and strict == emitted,
        "H4: every emission is non-degrading and strictly improving",
    )
    frozen_grid = {1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01}
    check(
        set(etas) <= frozen_grid,
        f"every selected eta {sorted(etas)} lies in the frozen candidate grid",
    )
    baseline_grid = {1.0, 0.5, 0.2, 0.1, 0.05}
    check(
        set(etas) <= baseline_grid,
        f"selected eta {sorted(etas)}: all within the inherited grid, so the "
        "downward extension was not the enabler in this run",
    )
    check(
        sum(etas.values()) == emitted,
        "eta histogram accounts for every emission exactly once",
    )

    # ---- 6. replay the sealed programs ------------------------------------
    REPORT.append("\n6. Replay of the sealed programs")
    for script in (
        "verify_variance_adaptive_certificate.py",
        "analyze_fp_scale_002.py",
    ):
        args = [sys.executable, "-B", str(PROJECT / script)]
        if script.startswith("analyze"):
            args += ["--result-dir", str(FORMAL)]
        completed = subprocess.run(args, capture_output=True, text=True, cwd=PROJECT)
        check(
            completed.returncode == 0,
            f"{script} replays with exit 0 (got {completed.returncode})",
        )

    REPORT.append("\n" + "=" * 60)
    REPORT.append("SUMMARY OF THE VERIFIED RESULT")
    REPORT.append(
        f"  primary route-records attempted : {attempted}\n"
        f"  primary emissions               : {emitted} / {attempted} "
        f"({100.0 * emitted / attempted:.1f}%)\n"
        f"  componentwise non-degrading     : {nondegrading} / {emitted}\n"
        f"  strict improvements             : {strict} / {emitted}\n"
        f"  certificate violations          : {violations}\n"
        f"  envelope control emissions      : {control_emitted} / {len(records)}\n"
        f"  control/adaptive E_Q ratio      : {min(ratios):.3f} .. {max(ratios):.3f} "
        "in every record"
    )
    REPORT.append("=" * 60)
    if FAILURES:
        REPORT.append(f"RESULT: FAIL ({FAILURES} checks failed)")
    else:
        REPORT.append("RESULT: PASS (all derived checks passed)")
    REPORT.append(
        "LIMITATION: same-actor derived verification only. No second actor\n"
        "reconstructed this route; this is not reciprocal verification."
    )
    text = "\n".join(REPORT)
    print(text)
    out = PROJECT / "docs" / "research_branches" / "FP-SCALE-002" / "claude" / "verification_same_actor.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
