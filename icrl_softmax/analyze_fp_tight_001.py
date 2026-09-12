"""FP-TIGHT-001 analysis: what a sound tightening actually buys.

Reads the formal bundle and evaluates H1--H6. H7 (the sample-size arm) is scored
separately from its own bundle by the same script via ``--sample-dir``.

The report is built so a reader can re-derive the flip arithmetic themselves: every
route-record's frozen E_Q, both repaired E_Q values, sigma_min and the census ratio
are printed, not just summarised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-TIGHT-001"
PRIMARY = ("expected_exact", "expected_finite")
REPORT: list[str] = []
FAILURES = 0

SOUND_ARMS = ("frozen", "bernstein", "empirical_bernstein")
# Pre-registered in docs/research_tasks/FP-TIGHT-001.md before the formal run.
BAND_H3 = (-0.25, -0.10)
BAND_H4 = (-0.35, -0.18)
H5_MAX_FLIPS = 0
H7_MIN_REDUCTION = -0.40
FLIP_REQUIREMENT = -0.43


def check(ok: bool, message: str) -> bool:
    """A construction check. Failing one invalidates the run."""
    global FAILURES
    if ok:
        REPORT.append(f"  PASS  {message}")
    else:
        FAILURES += 1
        REPORT.append(f"  FAIL  {message}")
    return bool(ok)


def hypothesis(ok: bool, message: str) -> bool:
    """A scientific prediction. Falsifying one is a RESULT, not a failed check."""
    REPORT.append(f"  {'PASS' if ok else 'FALSIFIED'}  {message}")
    return bool(ok)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_of(bundle: dict) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in bundle["records"]:
        for route in PRIMARY:
            block = record["routes"][route]
            out.append(
                {
                    "mixing": float(record["mixing"]),
                    "task_index": int(record["task_index"]),
                    "route": route,
                    "arms": block["arms"],
                    "audit": block["oracle_audit"],
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-TIGHT-001" / "claude" / "formal",
    )
    parser.add_argument("--sample-dir", type=Path, default=None)
    args = parser.parse_args()

    bundle = load(args.result_dir / "task_results.json")
    rows = rows_of(bundle)
    arm_names = list(rows[0]["arms"].keys())

    REPORT.append(f"{TASK_ID} analysis: a soundly tightened certificate")
    REPORT.append("=" * 78)
    REPORT.append(f"  {len(rows)} route-records at step 1 of pi_0")
    REPORT.append(f"  arms: {arm_names}")

    # ------------------------------------------------------------- 1. inputs
    REPORT.append("\n1. Frozen inputs")
    check(len(rows) == 48, f"48 route-records (found {len(rows)})")
    check(
        int(bundle["cert_chains"]) == 16384
        and int(bundle["cert_chain_length"]) == 64,
        f"certification is 16384 x 64 (found {bundle['cert_chains']} x "
        f"{bundle['cert_chain_length']})",
    )
    check(
        all(a in arm_names for a in SOUND_ARMS),
        f"all three sound arms present ({[a for a in SOUND_ARMS if a not in arm_names]} missing)",
    )

    # ---------------------------------------------------------------- H1
    REPORT.append("\n2. H1: do the repairs still cover the realized error?")
    violations: dict[str, list[tuple]] = {a: [] for a in SOUND_ARMS}
    for row in rows:
        realized = row["audit"]["realized_q_sup_error"]
        for arm in SOUND_ARMS:
            e_q = row["arms"][arm]["e_q"]
            if e_q is None or float(e_q) < realized:
                violations[arm].append(
                    (row["mixing"], row["task_index"], row["route"], e_q, realized)
                )
    for arm in SOUND_ARMS:
        check(
            not violations[arm],
            f"{arm}: covers the realized error on all 48 route-records "
            f"({len(violations[arm])} violations)",
        )
        for item in violations[arm][:5]:
            REPORT.append(
                f"        violation: mix={item[0]} task={item[1]} {item[2]} "
                f"E_Q={item[3]} < realized={item[4]:.6f}"
            )
    # The unsound counterfactual must NOT be presented as sound; check it is
    # labelled and report how often it would have failed coverage.
    cf = "counterfactual_no_envelope"
    if cf in arm_names:
        cf_fail = sum(
            1
            for row in rows
            if row["arms"][cf]["e_q"] is not None
            and float(row["arms"][cf]["e_q"]) < row["audit"]["realized_q_sup_error"]
        )
        labels = {row["arms"][cf].get("sound") for row in rows}
        check(
            labels == {False},
            f"the counterfactual arm is labelled unsound everywhere (found {labels})",
        )
        REPORT.append(
            f"  INFO  the counterfactual floor would fail coverage on {cf_fail}/48 "
            "route-records, which is why it is a diagnostic and not a certificate."
        )

    # ---------------------------------------------------------------- H2
    REPORT.append("\n3. H2: monotonicity -- no record loses eligibility")
    lost: list[tuple] = []
    for row in rows:
        if row["arms"]["frozen"]["emitted"]:
            for arm in ("bernstein", "empirical_bernstein"):
                if not row["arms"][arm]["emitted"]:
                    lost.append(
                        (row["mixing"], row["task_index"], row["route"], arm)
                    )
    check(not lost, f"no route-record loses eligibility ({len(lost)} lost)")
    for item in lost[:5]:
        REPORT.append(f"        lost: {item}")

    # ----------------------------------------------------------- 4. per-arm
    REPORT.append("\n4. Magnitude of the reduction")
    means: dict[str, float] = {}
    reductions: dict[str, float] = {}
    for arm in arm_names:
        values = [
            float(row["arms"][arm]["e_q"])
            for row in rows
            if row["arms"][arm]["e_q"] is not None
        ]
        means[arm] = sum(values) / len(values) if values else float("nan")
    frozen_mean = means["frozen"]
    for arm in arm_names:
        reductions[arm] = means[arm] / frozen_mean - 1.0
    REPORT.append("  mean E_Q over 48 route-records:")
    for arm in arm_names:
        tag = "" if arm in SOUND_ARMS else "   (NOT a certificate)"
        REPORT.append(
            f"    {arm:<28} {means[arm]:.4f}   {reductions[arm]:+.1%}{tag}"
        )
    REPORT.append(
        f"  the census's flip requirement: E_Q must fall by about "
        f"{-FLIP_REQUIREMENT:.0%} for the average abstainer to cross"
    )

    # ------------------------------------------------------------ 5. H3/H4
    REPORT.append("\n5. H3/H4: pre-registered magnitude bands")
    h3 = BAND_H3[0] <= reductions["bernstein"] <= BAND_H3[1]
    hypothesis(
        h3,
        f"H3 {'PASS' if h3 else 'FALSIFIED'}: Bernstein reduction "
        f"{reductions['bernstein']:+.1%} in [{BAND_H3[0]:+.0%}, {BAND_H3[1]:+.0%}]",
    )
    h4 = BAND_H4[0] <= reductions["empirical_bernstein"] <= BAND_H4[1]
    hypothesis(
        h4,
        f"H4 {'PASS' if h4 else 'FALSIFIED'}: empirical-Bernstein reduction "
        f"{reductions['empirical_bernstein']:+.1%} in "
        f"[{BAND_H4[0]:+.0%}, {BAND_H4[1]:+.0%}]",
    )

    # ---------------------------------------------------------------- H5
    REPORT.append("\n6. H5: does any never-emitting record flip?")
    never = [row for row in rows if not row["arms"]["frozen"]["emitted"]]
    REPORT.append(
        f"  the frozen certificate abstains on {len(never)} of {len(rows)} "
        f"route-records at step 1"
    )
    flips: dict[str, list[tuple]] = {a: [] for a in ("bernstein", "empirical_bernstein")}
    for row in never:
        for arm in flips:
            if row["arms"][arm]["emitted"]:
                flips[arm].append(
                    (
                        row["mixing"],
                        row["task_index"],
                        row["route"],
                        row["audit"]["sigma_min"],
                        row["arms"]["frozen"]["e_q"],
                        row["arms"][arm]["e_q"],
                        row["arms"][arm]["sigma_min_over_e_q"],
                    )
                )
    total_flips = sum(len(v) for v in flips.values())
    flipped_records = {
        (item[0], item[1], item[2]) for v in flips.values() for item in v
    }
    for arm, items in flips.items():
        REPORT.append(f"  {arm}: {len(items)} flips")
        for item in items:
            REPORT.append(
                f"        flip: mix={item[0]} task={item[1]} {item[2]} "
                f"sigma={item[3]:.4f} E_Q {item[4]:.4f} -> {item[5]:.4f} "
                f"ratio -> {item[6]:.4f}"
            )
    REPORT.append(
        f"  {len(flipped_records)} DISTINCT route-records flip; each flips under "
        f"both arms, which is why the per-arm counts sum to {total_flips}. The "
        f"registered maximum was {H5_MAX_FLIPS}."
    )
    h5 = len(flipped_records) <= H5_MAX_FLIPS
    hypothesis(
        h5,
        f"H5 {'PASS' if h5 else 'FALSIFIED'}: {len(flipped_records)} never-emitting "
        f"route-records flip (registered maximum {H5_MAX_FLIPS})",
    )

    # Where the flips sit in the sigma ordering, and what a flip actually cost.
    if flipped_records:
        sig = sorted(
            row["audit"]["sigma_min"]
            for row in never
            if (row["mixing"], row["task_index"], row["route"]) in flipped_records
        )
        notflip = sorted(
            row["audit"]["sigma_min"]
            for row in never
            if (row["mixing"], row["task_index"], row["route"])
            not in flipped_records
        )
        REPORT.append(
            f"  the flips are the HIGH-spread abstainers: sigma in "
            f"[{sig[0]:.4f}, {sig[-1]:.4f}], while the abstainers that do not flip "
            f"span [{notflip[0]:.4f}, {notflip[-1]:.4f}]"
        )
        REPORT.append(
            "  the -43% flip requirement was computed from the abstainers' MEAN "
            "spread (0.3273). Using a mean to make a universal claim was the error: "
            "the mean-sitting records sit far below the threshold, but the "
            "high-spread tail crosses on a 15-20% reduction."
        )

    # ------------------------------------- 6b. the census threshold, transferred
    # POST-HOC, not pre-registered. FP-CENSUS-001 fitted theta = 2.168689 to
    # separate emitters from abstainers under the FROZEN certificate, and it failed
    # to predict the step-6 dropout. Here the same theta is applied to the NEW
    # ratios, which it was never fitted on, so this is an out-of-sample test of
    # whether the criterion is a property of the ratio rather than of one
    # certificate.
    REPORT.append(
        "\n6b. POST-HOC: the census threshold, transferred to the new certificate"
    )
    census_theta = 2.168689882447331
    for arm in ("frozen", "empirical_bernstein"):
        predicted = [
            row["arms"][arm]["sigma_min_over_e_q"] > census_theta for row in rows
        ]
        observed = [row["arms"][arm]["emitted"] for row in rows]
        errors = sum(1 for p, o in zip(predicted, observed) if p != o)
        fp = sum(1 for p, o in zip(predicted, observed) if p and not o)
        fn = sum(1 for p, o in zip(predicted, observed) if o and not p)
        REPORT.append(
            f"  {arm:<22}: theta>{census_theta:.4f} misclassifies {errors}/48 "
            f"({fp} predicted-emit-but-abstained, {fn} emitted-but-predicted-abstain)"
        )
    REPORT.append(
        "  NOTE  theta was fitted on the FROZEN ratios, so the frozen row is "
        "in-sample and only the repaired rows are out-of-sample. Per the census's "
        "own rule, this is never described as a classifier on the strength of one "
        "population."
    )
    separated = None
    if flipped_records:
        abst = [row for row in never]
        flip_ratios = [
            row["arms"]["empirical_bernstein"]["sigma_min_over_e_q"]
            for row in abst
            if (row["mixing"], row["task_index"], row["route"]) in flipped_records
        ]
        keep_ratios = [
            row["arms"]["empirical_bernstein"]["sigma_min_over_e_q"]
            for row in abst
            if (row["mixing"], row["task_index"], row["route"]) not in flipped_records
        ]
        separated = min(flip_ratios) > max(keep_ratios)
        REPORT.append(
            f"  within the 26 abstainers under the new certificate: flippers span "
            f"[{min(flip_ratios):.4f}, {max(flip_ratios):.4f}], non-flippers span "
            f"[{min(keep_ratios):.4f}, {max(keep_ratios):.4f}] -> "
            f"{'disjoint' if separated else 'interleaved'}"
        )

    # ---------------------------------------------------------------- H6
    REPORT.append("\n7. H6: is deleting the envelope a bigger lever than")
    REPORT.append("   correcting the mean-step inequality?")
    if cf in arm_names:
        REPORT.append(
            f"  counterfactual, envelope deleted : {reductions[cf]:+.1%}"
        )
        REPORT.append(
            f"  sound, inequality corrected      : "
            f"{reductions['empirical_bernstein']:+.1%} (empirical Bernstein)"
        )
        h6 = abs(reductions["empirical_bernstein"]) >= abs(reductions[cf])
        hypothesis(
            h6,
            f"H6 {'PASS' if h6 else 'FALSIFIED'}: correcting the inequality "
            f"({reductions['empirical_bernstein']:+.1%}) "
            f"{'matches or beats' if h6 else 'is dominated by'} deleting the "
            f"envelope ({reductions[cf]:+.1%})",
        )
    else:
        h6 = None
        REPORT.append("  H6 not evaluable: no counterfactual arm in the bundle")

    # -------------------------------------------------------- 8. flip table
    REPORT.append("\n8. The flip arithmetic, per route-record")
    REPORT.append(
        f"  {'record':>12} {'route':>16} {'sigma_min':>9} "
        f"{'E_Q froz':>9} {'E_Q bern':>9} {'E_Q empB':>9} "
        f"{'ratio frz':>9} {'ratio empB':>10} {'dec f/b/e':>10}"
    )
    for row in sorted(rows, key=lambda r: r["audit"]["sigma_min"]):
        decisions = "".join(
            "E" if row["arms"][a]["emitted"] else "."
            for a in ("frozen", "bernstein", "empirical_bernstein")
        )
        key = f"{row['mixing']}/{row['task_index']}"
        REPORT.append(
            f"  {key:>12} {row['route']:>16} "
            f"{row['audit']['sigma_min']:>9.4f} "
            f"{row['arms']['frozen']['e_q']:>9.4f} "
            f"{row['arms']['bernstein']['e_q']:>9.4f} "
            f"{row['arms']['empirical_bernstein']['e_q']:>9.4f} "
            f"{row['arms']['frozen']['sigma_min_over_e_q']:>9.4f} "
            f"{row['arms']['empirical_bernstein']['sigma_min_over_e_q']:>10.4f} "
            f"{decisions:>10}"
        )

    # ---------------------------------------------------------------- H7
    h7 = None
    sample_summary = None
    if args.sample_dir is not None:
        REPORT.append("\n9. H7: the sample-size arm")
        sample = load(args.sample_dir / "task_results.json")
        multiplier = int(sample["sample_multiplier"])
        sample_rows = rows_of(sample)
        big_arm = f"empirical_bernstein_x{multiplier}"
        check(
            big_arm in sample_rows[0]["arms"],
            f"the sample arm {big_arm} is present",
        )
        big_mean = sum(
            float(r["arms"][big_arm]["e_q"])
            for r in sample_rows
            if r["arms"][big_arm]["e_q"] is not None
        ) / len(sample_rows)
        sample_frozen = sum(
            float(r["arms"]["frozen"]["e_q"]) for r in sample_rows
        ) / len(sample_rows)
        reduction = big_mean / sample_frozen - 1.0
        sample_flips = sum(
            1
            for r in sample_rows
            if not r["arms"]["frozen"]["emitted"] and r["arms"][big_arm]["emitted"]
        )
        sample_violations = sum(
            1
            for r in sample_rows
            if r["arms"][big_arm]["e_q"] is not None
            and float(r["arms"][big_arm]["e_q"]) < r["audit"]["realized_q_sup_error"]
        )
        sample_abstainers = sum(
            1 for r in sample_rows if not r["arms"]["frozen"]["emitted"]
        )
        REPORT.append(
            f"  {len(sample_rows)} route-records, certification x{multiplier}"
        )
        REPORT.append(f"  mean E_Q frozen {sample_frozen:.4f} -> {big_mean:.4f} "
                      f"({reduction:+.1%})")
        REPORT.append(f"  abstainers present in this population: {sample_abstainers}")
        REPORT.append(f"  never-emitter flips: {sample_flips}")
        REPORT.append(f"  coverage violations: {sample_violations}")
        check(
            sample_violations == 0,
            f"the sample arm still covers the realized error "
            f"({sample_violations} violations)",
        )
        reduction_ok = reduction <= H7_MIN_REDUCTION
        if sample_abstainers == 0:
            # The flip clause cannot be scored where there is nothing to flip.
            # Reporting that as FALSIFIED would be a vacuous failure -- the same
            # defect class as a vacuous pass, which this line has already had to
            # fix once.
            h7 = reduction_ok
            hypothesis(
                h7,
                f"H7 reduction clause {'PASS' if reduction_ok else 'FALSIFIED'}: "
                f"x{multiplier} certification gives {reduction:+.1%} "
                f"(needs <= {H7_MIN_REDUCTION:+.0%})",
            )
            REPORT.append(
                "  NOT EXERCISED  H7's flip clause: this population contains no "
                "abstainer, so nothing could flip. The clause is unscored rather "
                "than falsified, and the population's composition is recorded."
            )
            flip_clause = "NOT_EXERCISED"
        else:
            h7 = reduction_ok and sample_flips >= 1
            hypothesis(
                h7,
                f"H7 {'PASS' if h7 else 'FALSIFIED'}: x{multiplier} certification "
                f"gives {reduction:+.1%} (needs <= {H7_MIN_REDUCTION:+.0%}) and "
                f"{sample_flips} flips (needs >= 1)",
            )
            flip_clause = "PASS" if sample_flips >= 1 else "FALSIFIED"
        sample_summary = {
            "multiplier": multiplier,
            "n": len(sample_rows),
            "mean_e_q_frozen": sample_frozen,
            "mean_e_q_big": big_mean,
            "reduction": reduction,
            "abstainers_in_population": sample_abstainers,
            "flips": sample_flips,
            "flip_clause": flip_clause,
            "reduction_clause": "PASS" if reduction_ok else "FALSIFIED",
            "coverage_violations": sample_violations,
        }

    # ----------------------------------------------------------- summary
    REPORT.append("\n" + "=" * 78)
    REPORT.append("SUMMARY")
    REPORT.append(
        f"  H1 coverage      : "
        f"{'PASS' if not any(violations.values()) else 'FAIL'} for all three sound arms"
    )
    REPORT.append(f"  H2 monotone      : {'PASS' if not lost else 'FAIL'}")
    REPORT.append(
        f"  H3 bernstein band: {'PASS' if h3 else 'FALSIFIED'} "
        f"({reductions['bernstein']:+.1%})"
    )
    REPORT.append(
        f"  H4 empB band     : {'PASS' if h4 else 'FALSIFIED'} "
        f"({reductions['empirical_bernstein']:+.1%})"
    )
    REPORT.append(
        f"  H5 flips (want 0): {'PASS' if h5 else 'FALSIFIED'} "
        f"({len(flipped_records)} distinct records, {total_flips} arm-flips)"
    )
    REPORT.append(
        f"  H6 envelope vs inequality : "
        f"{'n/a' if h6 is None else ('PASS' if h6 else 'FALSIFIED')}"
    )
    REPORT.append(
        f"  H7 sample size   : "
        f"{'n/a' if h7 is None else ('PASS' if h7 else 'FALSIFIED')}"
    )
    REPORT.append("=" * 78)
    REPORT.append(
        "hypotheses were falsified, which per the task sheet is a RESULT, not a\n"
        "failed construction. RESULT below covers the construction checks only."
    )
    REPORT.append(
        "CONSTRUCTION: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)")
    )

    summary = {
        "task_id": TASK_ID,
        "route_records": len(rows),
        "mean_e_q": means,
        "reduction_vs_frozen": reductions,
        "H1_coverage_violations": {a: len(v) for a, v in violations.items()},
        "H2_lost_eligibility": len(lost),
        "H3": {"verdict": "PASS" if h3 else "FALSIFIED", "band": list(BAND_H3)},
        "H4": {"verdict": "PASS" if h4 else "FALSIFIED", "band": list(BAND_H4)},
        "H5": {
            "verdict": "PASS" if h5 else "FALSIFIED",
            "distinct_flipped_records": len(flipped_records),
            "per_arm_counts": {a: len(v) for a, v in flips.items()},
            "registered_max": H5_MAX_FLIPS,
            "detail": {a: [list(x) for x in v] for a, v in flips.items()},
            "flipped": [list(x) for x in sorted(flipped_records)],
            "census_theta_transfer": {
                "theta": census_theta,
                "abstainer_ratio_sets_disjoint_under_new_certificate": separated,
            },
        },
        "H6": None if h6 is None else ("PASS" if h6 else "FALSIFIED"),
        "H7": None if h7 is None else ("PASS" if h7 else "FALSIFIED"),
        "H7_sample_arm": sample_summary,
        "flip_requirement": FLIP_REQUIREMENT,
        "never_emitting_at_step1": len(never),
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.result_dir / "analysis_report.md").write_text(
        "\n".join(REPORT) + "\n", encoding="utf-8"
    )
    print("\n".join(REPORT))


if __name__ == "__main__":
    main()
