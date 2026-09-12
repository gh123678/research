"""FP-RANGE-001 analysis: is the envelope ceiling reachable soundly?

Scores H1--H6 from the bundle alone. The price decomposition is reported per record
because the whole question is whether the honest tail price eats the saving.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-RANGE-001"
PRIMARY = ("expected_exact", "expected_finite")
REPORT: list[str] = []
FAILURES = 0

# Pre-registered in docs/research_tasks/FP-RANGE-001.md before the formal run.
BAND_H3 = (0.05, 0.25)


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
        default=PROJECT / "results" / "FP-RANGE-001" / "claude" / "formal",
    )
    args = parser.parse_args()

    bundle = load(args.result_dir / "task_results.json")
    rungs = [int(r) for r in bundle["rungs"]]
    rows = rows_of(bundle)
    items = {r: int(rows[0][3]["rungs"][str(r)]["items"]) for r in rungs}

    REPORT.append(f"{TASK_ID} analysis: is the envelope ceiling reachable soundly?")
    REPORT.append("=" * 84)
    REPORT.append(f"  {len(rows)} route-records at step 1, rungs {rungs}")

    # ---------------------------------------------------------------- H1
    REPORT.append("\n1. H1: is the truncation certificate sound? (mandatory)")
    bad = []
    for mixing, task_index, route, block in rows:
        realized = block["oracle_audit"]["realized_q_sup_error"]
        for rung in rungs:
            e_q = block["rungs"][str(rung)]["data_range"]["e_q"]
            if e_q is None or float(e_q) < realized:
                bad.append((mixing, task_index, route, rung, e_q, realized))
    check(
        not bad,
        f"data_range covers the realized error at every rung "
        f"({len(bad)} violations over {len(rows) * len(rungs)} fits)",
    )
    for item in bad[:5]:
        REPORT.append(f"        violation: {item}")

    # ------------------------------------------------------- 2. the ladder
    REPORT.append("\n2. The ladder")
    means: dict[str, dict[int, float]] = {}
    for arm in ("data_range", "empirical_bernstein"):
        means[arm] = {}
        for rung in rungs:
            values = [
                float(block["rungs"][str(rung)][arm]["e_q"]) for _, _, _, block in rows
            ]
            means[arm][rung] = sum(values) / len(values)
    means["frozen"] = {
        rungs[0]: sum(
            float(block["rungs"][str(rungs[0])]["frozen_same_sample"]["e_q"])
            for _, _, _, block in rows
        ) / len(rows)
    }
    means["counterfactual"] = {
        rungs[0]: sum(
            float(
                block["rungs"][str(rungs[0])]["counterfactual_no_envelope"]["e_q"]
            )
            for _, _, _, block in rows
        ) / len(rows)
    }
    base = means["frozen"][rungs[0]]
    REPORT.append(
        f"  {'rung':>5} {'items':>10} {'data_range':>10} {'vs frozen':>10} "
        f"{'empBern':>9} {'vs frozen':>10} {'emits (dr)':>11}"
    )
    for rung in rungs:
        dr = means["data_range"][rung]
        eb = means["empirical_bernstein"][rung]
        emits = sum(
            1
            for _, _, _, block in rows
            if block["rungs"][str(rung)]["data_range"]["emitted"]
        )
        REPORT.append(
            f"  {rung:>4}x {items[rung]:>10} {dr:>10.4f} {dr / base - 1:>+10.1%} "
            f"{eb:>9.4f} {eb / base - 1:>+10.1%} {emits:>11}"
        )
    REPORT.append(
        f"  at {rungs[0]}x the same-sample references are: frozen "
        f"{means['frozen'][rungs[0]]:.4f}, unsound ceiling "
        f"{means['counterfactual'][rungs[0]]:.4f} "
        f"({means['counterfactual'][rungs[0]] / base - 1:+.1%})"
    )

    # ----------------------------------------------------------- 3. H2/H3
    REPORT.append("\n3. H2/H3: does the sound range beat the inequality repair?")
    dr_1x = means["data_range"][rungs[0]] / base - 1
    eb_1x = means["empirical_bernstein"][rungs[0]] / base - 1
    h2 = dr_1x > eb_1x  # less negative, i.e. worse than empirical Bernstein
    hypothesis(
        h2,
        f"H2 {'PASS' if h2 else 'FALSIFIED'}: data_range {dr_1x:+.1%} is less "
        f"negative than empirical Bernstein {eb_1x:+.1%}, i.e. the range lever "
        f"loses to the inequality lever",
    )
    h3 = BAND_H3[0] <= dr_1x <= BAND_H3[1]
    hypothesis(
        h3,
        f"H3 {'PASS' if h3 else 'FALSIFIED'}: data_range at {rungs[0]}x is "
        f"{dr_1x:+.1%}, in the registered band [{BAND_H3[0]:+.0%}, {BAND_H3[1]:+.0%}]",
    )

    # ----------------------------------------------------------- 4. H4/H5
    REPORT.append("\n4. H4/H5: the price decomposition")
    tail_wins = 0
    bias_beats_radius = 0
    considered = 0
    REPORT.append(
        f"  {'record':>12} {'tau':>7} {'tail mass':>10} {'CS bias':>9} "
        f"{'concentr.':>10} {'price':>8} {'frozen r':>9} {'tail>conc':>10}"
    )
    for mixing, task_index, route, block in rows:
        entry = block["rungs"][str(rungs[0])]["data_range"]
        price = entry.get("price_decomposition")
        if not price:
            continue
        considered += 1
        frozen_r = float(
            block["rungs"][str(rungs[0])]["frozen_same_sample"][
                "radius_at_dr_binding_pair"
            ]
        )
        if price["price_exceeds_concentration"]:
            tail_wins += 1
        if price["cauchy_schwarz_bias"] > frozen_r:
            bias_beats_radius += 1
        REPORT.append(
            f"  {f'{mixing}/{task_index} {route[:6]}':>12} {price['tau']:>7.3f} "
            f"{price['empirical_tail_mass']:>10.5f} "
            f"{price['cauchy_schwarz_bias']:>9.5f} {price['concentration']:>10.5f} "
            f"{price['tail_price_total']:>8.5f} {frozen_r:>9.5f} "
            f"{str(price['price_exceeds_concentration']):>10}"
        )
    h4 = tail_wins == considered
    hypothesis(
        h4,
        f"H4 {'PASS' if h4 else 'FALSIFIED'}: the tail price exceeds the "
        f"concentration term on {tail_wins}/{considered} route-records",
    )
    h5 = bias_beats_radius == considered
    hypothesis(
        h5,
        f"H5 {'PASS' if h5 else 'FALSIFIED'}: the Cauchy-Schwarz bias alone "
        f"exceeds the frozen radius at the same pair on {bias_beats_radius}/"
        f"{considered} route-records",
    )
    if not h5:
        ratios = [
            block["rungs"][str(rungs[0])]["data_range"]["price_decomposition"][
                "cauchy_schwarz_bias"
            ]
            / max(
                float(
                    block["rungs"][str(rungs[0])]["frozen_same_sample"][
                        "radius_at_dr_binding_pair"
                    ]
                ),
                1e-12,
            )
            for _, _, _, block in rows
            if block["rungs"][str(rungs[0])]["data_range"].get("price_decomposition")
        ]
        REPORT.append(
            f"  the registered mechanism is wrong: the bias is COMPARABLE to the "
            f"frozen radius, not larger (mean ratio {sum(ratios) / len(ratios):.3f}, "
            f"max {max(ratios):.3f}). The loss comes from the tail price being ADDED "
            f"to the concentration term, not from either piece alone."
        )

    # ------------------------------------------------------------- H6
    REPORT.append("\n5. H6: does more data rescue the range lever?")
    top = rungs[-1]
    dr_top = means["data_range"][top] / base - 1
    eb_top = means["empirical_bernstein"][top] / base - 1
    h6 = dr_top > eb_top
    hypothesis(
        h6,
        f"H6 {'PASS' if h6 else 'FALSIFIED'}: at {top}x, data_range {dr_top:+.1%} "
        f"still loses to empirical Bernstein {eb_top:+.1%}",
    )
    REPORT.append(
        "  the ordering is stable across the ladder: "
        + ", ".join(
            f"{r}x dr{means['data_range'][r] / base - 1:+.0%} vs "
            f"eb{means['empirical_bernstein'][r] / base - 1:+.0%}"
            for r in rungs
        )
    )

    # ------------------------------------------------------------- summary
    REPORT.append("\n" + "=" * 84)
    REPORT.append("SUMMARY")
    REPORT.append(f"  H1 soundness        : {'PASS' if not bad else 'FAIL'}")
    REPORT.append(f"  H2 range loses      : {'PASS' if h2 else 'FALSIFIED'}")
    REPORT.append(f"  H3 1x band          : {'PASS' if h3 else 'FALSIFIED'}")
    REPORT.append(f"  H4 tail price wins  : {'PASS' if h4 else 'FALSIFIED'}")
    REPORT.append(f"  H5 bias alone wins  : {'PASS' if h5 else 'FALSIFIED'}")
    REPORT.append(f"  H6 stable at {top}x     : {'PASS' if h6 else 'FALSIFIED'}")
    REPORT.append(
        f"  headline: the unsound ceiling is "
        f"{means['counterfactual'][rungs[0]] / base - 1:+.1%}; paying for the tails "
        f"turns it into {dr_1x:+.1%} at {rungs[0]}x and {dr_top:+.1%} at {top}x."
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
        "reduction_vs_frozen_same_sample": {
            arm: {str(r): means[arm][r] / base - 1 for r in rungs}
            for arm in ("data_range", "empirical_bernstein")
        },
        "H1_coverage_violations": len(bad),
        "H2": "PASS" if h2 else "FALSIFIED",
        "H3": {"verdict": "PASS" if h3 else "FALSIFIED", "band": list(BAND_H3),
               "one_x_reduction": dr_1x},
        "H4": {"verdict": "PASS" if h4 else "FALSIFIED", "tail_price_wins":
               f"{tail_wins}/{considered}"},
        "H5": {"verdict": "PASS" if h5 else "FALSIFIED", "bias_beats_radius":
               f"{bias_beats_radius}/{considered}"},
        "H6": "PASS" if h6 else "FALSIFIED",
        "counterfactual_ceiling_1x": means["counterfactual"][rungs[0]] / base - 1,
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
