"""FP-SAMPLE-001 analysis: the certification-size cost curve.

Reads the ladder bundle and the FP-TIGHT-001 formal bundle. The join matters: the
ladder computes only the two repaired certificates, so "the 26 abstainers" has to
come from FP-TIGHT-001's FROZEN arm. A prediction phrased against the frozen
baseline is therefore only measurable through that join, and the report says so
rather than quietly re-basing the claim onto the ladder's own 1x rung.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-SAMPLE-001"
PRIMARY = ("expected_exact", "expected_finite")
TIGHT = PROJECT / "results" / "FP-TIGHT-001" / "claude" / "formal"
REPORT: list[str] = []
FAILURES = 0

# Pre-registered in docs/research_tasks/FP-SAMPLE-001.md before the run.
GATE_TV = 0.02
GATE_M2_REL = 0.02
H3_TOLERANCE = 0.20
H3_REFERENCE = 0.1940  # FP-TIGHT-001 sealed 1x, empirical Bernstein
BAND_H5 = (-0.70, -0.50)
H6B_MIN = 16
FS_GAMMA = 0.70
# The census's step-1 emission threshold, which FP-TIGHT-001 validated
# out-of-sample at 0/48 on the repaired certificate.
CENSUS_THETA = 2.168689882447331


def check(ok: bool, message: str) -> bool:
    global FAILURES
    if ok:
        REPORT.append(f"  PASS  {message}")
    else:
        FAILURES += 1
        REPORT.append(f"  FAIL  {message}")
    return bool(ok)


def hypothesis(ok: bool, message: str) -> bool:
    REPORT.append(f"  {'PASS' if ok else 'FALSIFIED'}  {message}")
    return bool(ok)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_of(bundle: dict):
    return [
        (r["mixing"], r["task_index"], route, r["routes"][route])
        for r in bundle["records"]
        for route in PRIMARY
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-SAMPLE-001" / "claude" / "formal",
    )
    parser.add_argument("--tight-dir", type=Path, default=TIGHT)
    args = parser.parse_args()

    bundle = load(args.result_dir / "task_results.json")
    rungs = [int(r) for r in bundle["rungs"]]
    rows = rows_of(bundle)

    REPORT.append(f"{TASK_ID} analysis: the certification-size cost curve")
    REPORT.append("=" * 84)
    REPORT.append(f"  {len(rows)} route-records at step 1 of pi_0, rungs {rungs}")

    # ---------------------------------------------------------------- H1
    REPORT.append("\n1. H1: is the vectorised sampler unbiased? (mandatory gate)")
    completed = subprocess.run(
        [sys.executable, "-B", str(PROJECT / "fp_sample_vectorised_batch.py")],
        capture_output=True,
        text=True,
        cwd=PROJECT,
    )
    gate_ok = completed.returncode == 0
    for line in completed.stdout.strip().splitlines():
        REPORT.append(f"        {line}")
    check(
        gate_ok,
        f"sampler gate passes (TV < {GATE_TV}, E[Y^2] within {GATE_M2_REL:.0%}); "
        f"exit {completed.returncode}",
    )
    if not gate_ok:
        REPORT.append(
            "  ABORT  the ladder's numbers are void: a biased sampler shifts E_Q."
        )

    # ---------------------------------------------------------------- H2
    REPORT.append("\n2. H2: coverage at every rung, both arms")
    for arm in ("empirical_bernstein", "bernstein"):
        bad = []
        for mixing, task_index, route, block in rows:
            realized = block["oracle_audit"]["realized_q_sup_error"]
            for rung in rungs:
                e_q = block["rungs"][str(rung)][arm]["e_q"]
                if e_q is None or float(e_q) < realized:
                    bad.append((mixing, task_index, route, rung, e_q, realized))
        check(
            not bad,
            f"{arm}: covers the realized error at every rung "
            f"({len(bad)} violations over {len(rows) * len(rungs)} certificate-fits)",
        )
        for item in bad[:5]:
            REPORT.append(f"        violation: {item}")

    # -------------------------------------------------------- 3. cost curve
    REPORT.append("\n3. The cost curve")
    means: dict[str, dict[str, float]] = {"empirical_bernstein": {}, "bernstein": {}}
    for arm in means:
        for rung in rungs:
            values = [
                float(block["rungs"][str(rung)][arm]["e_q"]) for _, _, _, block in rows
            ]
            means[arm][rung] = sum(values) / len(values)
    items = {
        rung: int(rows[0][3]["rungs"][str(rung)]["items"]) for rung in rungs
    }
    REPORT.append(
        f"  {'rung':>6} {'items':>10} {'mean E_Q empB':>15} {'vs 1x':>9} "
        f"{'mean E_Q bern':>15} {'vs 1x':>9}"
    )
    reductions: dict[str, dict[int, float]] = {"empirical_bernstein": {}, "bernstein": {}}
    for rung in rungs:
        for arm in means:
            reductions[arm][rung] = means[arm][rung] / means[arm][rungs[0]] - 1.0
        REPORT.append(
            f"  {rung:>5}x {items[rung]:>10} {means['empirical_bernstein'][rung]:>15.4f} "
            f"{reductions['empirical_bernstein'][rung]:>+9.1%} "
            f"{means['bernstein'][rung]:>15.4f} {reductions['bernstein'][rung]:>+9.1%}"
        )
    REPORT.append(
        "  reductions are measured against the ladder's own 1x rung, on the same "
        "sampler, so size is the only thing that changes along the row."
    )

    # ------------------------------------------------------------- H3
    REPORT.append("\n4. H3: agreement with FP-TIGHT-001's sealed 1x")
    one_x = means["empirical_bernstein"][rungs[0]]
    rel = abs(one_x - H3_REFERENCE) / H3_REFERENCE
    REPORT.append(
        f"  ladder 1x mean E_Q {one_x:.4f} vs FP-TIGHT-001 sealed 1x "
        f"{H3_REFERENCE:.4f} -> {rel:.1%} apart"
    )
    h3 = rel <= H3_TOLERANCE
    hypothesis(
        h3,
        f"H3 {'PASS' if h3 else 'FALSIFIED'}: within {H3_TOLERANCE:.0%} "
        f"(a different sample from a different sampler, not a reproduction)",
    )

    # ------------------------------------------------------------- H4
    REPORT.append("\n5. H4: monotone in the rung")
    seq = [means["empirical_bernstein"][r] for r in rungs]
    h4 = all(b < a for a, b in zip(seq, seq[1:]))
    hypothesis(
        h4,
        f"H4 {'PASS' if h4 else 'FALSIFIED'}: mean E_Q {[round(v, 4) for v in seq]} "
        f"is strictly decreasing",
    )

    # ------------------------------------------------------------- H5
    REPORT.append("\n6. H5: magnitude at the 4x rung")
    four = reductions["empirical_bernstein"][4] if 4 in rungs else None
    h5 = four is not None and BAND_H5[0] <= four <= BAND_H5[1]
    hypothesis(
        h5,
        f"H5 {'PASS' if h5 else 'FALSIFIED'}: 4x reduction {four:+.1%} in "
        f"[{BAND_H5[0]:+.0%}, {BAND_H5[1]:+.0%}]",
    )

    # ------------------------------------------------------- 7. H6a/H6b/H6c
    REPORT.append("\n7. H6: do the abstainers revive?")
    tight = load(args.tight_dir / "task_results.json")
    frozen_abstainers = {
        (m, t, route)
        for m, t, route, block in rows_of(tight)
        if not block["arms"]["frozen"]["emitted"]
    }
    REPORT.append(
        f"  the FROZEN baseline (from FP-TIGHT-001): {len(frozen_abstainers)} "
        f"abstaining route-records of {len(rows)}"
    )
    ladder_1x_abstainers = {
        (m, t, route)
        for m, t, route, block in rows
        if not block["rungs"][str(rungs[0])]["empirical_bernstein"]["emitted"]
    }
    REPORT.append(
        f"  the ladder's own 1x abstainers (empirical Bernstein already applied): "
        f"{len(ladder_1x_abstainers)} — that is the baseline the ladder moves, and "
        f"it differs from the frozen one because the repair already flips records"
    )
    per_rung_emitters: dict[int, set] = {}
    for rung in rungs:
        per_rung_emitters[rung] = {
            (m, t, route)
            for m, t, route, block in rows
            if block["rungs"][str(rung)]["empirical_bernstein"]["emitted"]
        }
    REPORT.append(
        f"  {'rung':>6} {'emitting':>10} {'of the 26 frozen abstainers, emitting':>40}"
    )
    revived: dict[int, int] = {}
    for rung in rungs:
        count = len(frozen_abstainers & per_rung_emitters[rung])
        revived[rung] = count
        REPORT.append(
            f"  {rung:>5}x {len(per_rung_emitters[rung]):>10} {count:>40}"
        )
    ladder_flips = {
        rung: len(
            (frozen_abstainers & per_rung_emitters[rung])
            - (frozen_abstainers & per_rung_emitters[rungs[0]])
        )
        for rung in rungs
    }

    two_or_less = [r for r in rungs if r <= 2]
    h6a = any(len(ladder_1x_abstainers & per_rung_emitters[r]) >= 1 for r in two_or_less)
    hypothesis(
        h6a,
        f"H6a {'PASS' if h6a else 'FALSIFIED'}: at least one ladder-1x abstainer "
        f"emits at 2x or below",
    )
    h6b = revived[rungs[-1]] >= H6B_MIN
    hypothesis(
        h6b,
        f"H6b {'PASS' if h6b else 'FALSIFIED'}: {revived[rungs[-1]]} of the "
        f"{len(frozen_abstainers)} frozen abstainers emit at {rungs[-1]}x "
        f"(registered minimum {H6B_MIN})",
    )
    counts = [len(frozen_abstainers & per_rung_emitters[r]) for r in rungs]
    h6c = all(b >= a for a, b in zip(counts, counts[1:]))
    hypothesis(
        h6c,
        f"H6c {'PASS' if h6c else 'FALSIFIED'}: the revived count "
        f"{counts} is non-decreasing",
    )
    still_out = sorted(frozen_abstainers - per_rung_emitters[rungs[-1]])
    REPORT.append(
        f"  never revived even at {rungs[-1]}x ({len(still_out)}): {still_out}"
    )
    for m, t, route in still_out:
        block = next(
            b for mm, tt, rr, b in rows if (mm, tt, rr) == (m, t, route)
        )
        REPORT.append(
            f"        sigma_min {block['oracle_audit']['sigma_min']:.4f}  "
            f"E_Q {rungs[0]}x "
            f"{block['rungs'][str(rungs[0])]['empirical_bernstein']['e_q']:.4f} -> "
            f"{rungs[-1]}x "
            f"{block['rungs'][str(rungs[-1])]['empirical_bernstein']['e_q']:.4f}"
        )

    # ------------------------------------------------------------- H7
    REPORT.append("\n8. H7: does the curve flatten?")
    if len(rungs) >= 3:
        late = means["empirical_bernstein"][rungs[-1]] - means["empirical_bernstein"][
            rungs[-2]
        ]
        early = means["empirical_bernstein"][rungs[-2]] - means["empirical_bernstein"][
            rungs[-3]
        ]
        h7 = abs(late) < abs(early)
        hypothesis(
            h7,
            f"H7 {'PASS' if h7 else 'FALSIFIED'}: the drop from {rungs[-2]}x to "
            f"{rungs[-1]}x is {late:+.4f}, versus {early:+.4f} from {rungs[-3]}x to "
            f"{rungs[-2]}x",
        )
        REPORT.append(
            "  NOTE  E_Q flattens, but the REVIVAL count does not yet: "
            f"{[counts[i] for i in range(len(rungs))]}. Diminishing returns in the "
            "bound are not yet diminishing returns in the outcome."
        )
    else:
        h7 = None
        REPORT.append("  H7 not evaluable: fewer than three rungs")

    # --------------------------- 8b. what would the stragglers cost? (extrapolated)
    REPORT.append("\n8b. Cost to revive the stragglers (EXTRAPOLATED, not measured)")
    if len(rungs) >= 3 and still_out:
        # Fit PER RECORD: eps_res = a + b/sqrt(N), on that record's own four rungs.
        # An earlier draft fitted the POPULATION MEAN and applied it to individual
        # records, which is simply wrong -- a record's E_Q is not the population
        # mean -- and it produced the visible absurdity of predicting a crossing at
        # 6.3x for a record that had already been measured at 8x without crossing.
        # The per-record fit is the only version that can be checked against the
        # rung it is extrapolating from, so that check is reported.
        REPORT.append(
            f"  {'route-record':>28} {'sigma':>8} {'needs E_Q':>10} "
            f"{'fits at 8x?':>12} {'fitted floor':>13} {'N / 1x needed':>14}"
        )
        for m, t, route in still_out:
            block = next(
                bb for mm, tt, rr, bb in rows if (mm, tt, rr) == (m, t, route)
            )
            sigma = float(block["oracle_audit"]["sigma_min"])
            needed_e_q = sigma / CENSUS_THETA
            own = [
                float(block["rungs"][str(r)]["empirical_bernstein"]["e_q"])
                * (1.0 - FS_GAMMA)
                for r in rungs
            ]
            xs = [1.0 / math.sqrt(items[r]) for r in rungs]
            n2 = len(xs)
            sx, sy = sum(xs), sum(own)
            sxx = sum(x * x for x in xs)
            sxy = sum(x * y for x, y in zip(xs, own))
            denom = n2 * sxx - sx * sx
            b = (n2 * sxy - sx * sy) / denom
            a = (sy - b * sx) / n2
            # Does the fit reproduce the top rung it was fitted on? If not, reading
            # it beyond that range is not defensible and the row says so.
            top_residual = own[-1] - (a + b * xs[-1])
            consistent = abs(top_residual) < 0.15 * max(own[-1], 1e-12)
            needed_eps = needed_e_q * (1.0 - FS_GAMMA)
            if b <= 0 or needed_eps <= a:
                verdict = "unreachable under this fit"
            else:
                needed_n = (b / (needed_eps - a)) ** 2
                verdict = f"{needed_n / items[rungs[0]]:.1f}x"
            REPORT.append(
                f"  {f'{m}/{t} {route}':>28} {sigma:>8.4f} {needed_e_q:>10.4f} "
                f"{'yes' if consistent else 'NO':>12} {a:>13.4f} {verdict:>14}"
            )
        REPORT.append(
            "  'fits at 8x?' asks whether the two-parameter fit reproduces the very "
            "rung it was fitted on. A 'NO' means the shape assumption is already "
            "wrong inside the measured range, so the extrapolated cost for that "
            "record is not usable."
        )
        REPORT.append(
            "  the fitted floor is the part of eps_res that data does NOT shrink. "
            "Where it sits near the needed value, the large multiplier is the fit's "
            "way of saying the record is at its floor: buying data would not move "
            "it, and the honest reading is 'not reachable by sample size' rather "
            "than a precise 600x price tag."
        )
        REPORT.append(
            "  EXTRAPOLATION: at best this prices the stragglers; it does not "
            "predict that they will revive."
        )
    else:
        REPORT.append("  no stragglers at the top rung, or too few rungs to fit")

    # ----------------------------------------------------------- summary
    REPORT.append("\n" + "=" * 84)
    REPORT.append("SUMMARY")
    REPORT.append(
        f"  H1 sampler gate : {'PASS' if gate_ok else 'FAIL'}"
    )
    REPORT.append(f"  H2 coverage     : {'PASS' if FAILURES == 0 else 'see above'}")
    REPORT.append(f"  H3 1x agreement : {'PASS' if h3 else 'FALSIFIED'}")
    REPORT.append(f"  H4 monotone     : {'PASS' if h4 else 'FALSIFIED'}")
    REPORT.append(f"  H5 4x band      : {'PASS' if h5 else 'FALSIFIED'}")
    REPORT.append(f"  H6a cheap flip  : {'PASS' if h6a else 'FALSIFIED'}")
    REPORT.append(f"  H6b revival     : {'PASS' if h6b else 'FALSIFIED'}")
    REPORT.append(f"  H6c monotone    : {'PASS' if h6c else 'FALSIFIED'}")
    REPORT.append(
        f"  H7 flattening   : "
        f"{'n/a' if h7 is None else ('PASS' if h7 else 'FALSIFIED')}"
    )
    REPORT.append(
        "  revived of the 26 frozen abstainers: "
        + ", ".join(f"{r}x={revived[r]}" for r in rungs)
    )
    REPORT.append("=" * 84)
    REPORT.append(
        "hypotheses were falsified, which per the task sheet is a RESULT, not a\n"
        "failed construction. RESULT below covers the construction checks only."
    )
    REPORT.append(
        "CONSTRUCTION: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)")
    )

    summary = {
        "task_id": TASK_ID,
        "rungs": rungs,
        "items_per_rung": items,
        "mean_e_q": means,
        "reduction_vs_ladder_1x": reductions,
        "H1_sampler_gate": "PASS" if gate_ok else "FAIL",
        "H3": {"verdict": "PASS" if h3 else "FALSIFIED", "ladder_1x": one_x,
               "reference": H3_REFERENCE, "relative_gap": rel},
        "H4": "PASS" if h4 else "FALSIFIED",
        "H5": {"verdict": "PASS" if h5 else "FALSIFIED", "band": list(BAND_H5),
               "four_x_reduction": four},
        "H6a": "PASS" if h6a else "FALSIFIED",
        "H6b": {"verdict": "PASS" if h6b else "FALSIFIED",
                "revived_at_top_rung": revived[rungs[-1]],
                "frozen_abstainers": len(frozen_abstainers),
                "registered_minimum": H6B_MIN},
        "H6c": "PASS" if h6c else "FALSIFIED",
        "H7": None if h7 is None else ("PASS" if h7 else "FALSIFIED"),
        "revived_by_rung": {str(r): revived[r] for r in rungs},
        "ladder_flips_vs_ladder_1x": {str(r): ladder_flips[r] for r in rungs},
        "never_revived": [list(x) for x in still_out],
        "ladder_1x_abstainers": len(ladder_1x_abstainers),
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
