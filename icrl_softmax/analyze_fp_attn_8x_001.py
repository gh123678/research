"""FP-ATTN-8X-001 analysis: does the network reproduce numpy at 8x over 12 steps?

Compares the network bundle against FP-ITER8X-001's numpy bundle step by step, in
both arms, and reports the agreement headroom (min emitted gain over max float32 gap)
so that agreement is a statement about the certificate rather than about arithmetic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-ATTN-8X-001"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("frozen", "empirical_bernstein")
NUMPY_8X = PROJECT / "results" / "FP-ITER8X-001" / "claude" / "formal" / "task_results.json"
REPORT: list[str] = []
FAILURES = 0

# Pre-registered in docs/research_tasks/FP-ATTN-8X-001.md before the run.
H7_MIN_HEADROOM = 20.0


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


def level_stats(rows, arm: str, level: int) -> dict:
    """Per-step stats. The gap field exists only on the network side, so it is read
    with a default -- the numpy reference bundle has no reason to record it."""
    emitters = {
        (m, t, route)
        for m, t, route, block in rows
        for s in block[arm]["steps"]
        if s["step"] == level and s["update_emitted"]
    }
    gains = [
        s["oracle_audit"]["total_value_gain"]
        for _, _, _, block in rows
        for s in block[arm]["steps"]
        if s["step"] == level and s["update_emitted"]
    ]
    gaps = [
        s["q_hat_gap_vs_numpy"]
        for _, _, _, block in rows
        for s in block[arm]["steps"]
        if s["step"] == level and "q_hat_gap_vs_numpy" in s
    ]
    return {
        "emitters": emitters,
        "count": len(emitters),
        "min_gain": min(gains) if gains else None,
        "mean_gain": (sum(gains) / len(gains)) if gains else None,
        "max_gap": max(gaps) if gaps else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=PROJECT / "results" / "FP-ATTN-8X-001" / "claude" / "formal",
    )
    parser.add_argument("--numpy-dir", type=Path, default=NUMPY_8X)
    args = parser.parse_args()

    bundle = load(args.result_dir / "task_results.json")
    horizon = int(bundle["max_steps"])
    atol = float(bundle["atol"])
    rows = rows_of(bundle)
    numpy_rows = rows_of(load(args.numpy_dir))

    REPORT.append(f"{TASK_ID} analysis: the literal network at 8x certification")
    REPORT.append("=" * 92)
    REPORT.append(
        f"  {len(rows)} route-records, horizon {horizon}, "
        f"certification {bundle['cert_chains'] * 64:,} items, arms {bundle['arms']}"
    )
    REPORT.append(f"  torch {bundle.get('torch')}, ATOL {atol:.0e}")

    # --------------------------------------------------------------- H1
    REPORT.append("\n1. H1: provenance (mandatory)")
    steps_all = [
        (m, t, route, s) for m, t, route, block in rows
        for arm in ARMS for s in block[arm]["steps"]
    ]
    producers = {s.get("qhat_producer") for _, _, _, s in steps_all}
    check(
        producers == {"literal_attention_network"},
        f"every one of {len(steps_all)} step entries records the literal network "
        f"(found {producers})",
    )
    # The gap must be a genuine float32 difference everywhere, not an exact zero that
    # would mean the network output had been replaced by numpy's.
    smallest_gap = min(s["q_hat_gap_vs_numpy"] for _, _, _, s in steps_all)
    check(
        smallest_gap > 0.0,
        f"every step shows a nonzero float32 gap, so no numpy substitution "
        f"(minimum {smallest_gap:.3e})",
    )

    # --------------------------------------------------------------- H2
    REPORT.append("\n2. H2: validity and soundness (mandatory)")
    for arm in ARMS:
        emitted = [
            s for _, _, _, block in rows for s in block[arm]["steps"]
            if s["update_emitted"]
        ]
        check(
            all(
                s["oracle_audit"].get("componentwise_nondegrading", False)
                for s in emitted
            ),
            f"{arm}: all {len(emitted)} emitted steps componentwise non-degrading",
        )
        check(
            all(s["oracle_audit"]["total_value_gain"] > 0.0 for s in emitted),
            f"{arm}: every emitted step strictly improving",
        )
        check(
            not any(s["oracle_audit"].get("certificate_violation") for s in emitted),
            f"{arm}: zero certificate violations",
        )

    # --------------------------------------------------------------- H3
    REPORT.append("\n3. H3: does the network reach twelve steps?")
    deepest = {
        arm: max((b[arm]["emitted_steps"] for _, _, _, b in rows), default=0)
        for arm in ARMS
    }
    np_deepest = {
        arm: max((b[arm]["emitted_steps"] for _, _, _, b in numpy_rows), default=0)
        for arm in ARMS
    }
    REPORT.append(f"  deepest trajectory — network {deepest}, numpy {np_deepest}")
    h3 = all(deepest[arm] >= horizon for arm in ARMS)
    hypothesis(
        h3,
        f"H3 {'PASS' if h3 else 'FALSIFIED'}: the network reaches the full horizon "
        f"on both arms",
    )

    # --------------------------------------------------------------- H4
    REPORT.append("\n4. H4: path agreement at every step")
    agreement: dict[str, dict[int, dict]] = {}
    for arm in ARMS:
        agreement[arm] = {}
        for level in range(1, horizon + 1):
            net = level_stats(rows, arm, level)["emitters"]
            num = level_stats(numpy_rows, arm, level)["emitters"]
            agreement[arm][level] = {
                "network": len(net),
                "numpy": len(num),
                "gained": sorted(net - num),
                "lost": sorted(num - net),
            }
    total_disagreements = sum(
        len(v["gained"]) + len(v["lost"])
        for arm in ARMS for v in agreement[arm].values()
    )
    REPORT.append(
        f"  {'step':>4} | {'frozen net':>10} {'numpy':>6} "
        f"| {'empB net':>9} {'numpy':>6} | disagreements"
    )
    for level in range(1, horizon + 1):
        f = agreement["frozen"][level]
        e = agreement["empirical_bernstein"][level]
        d = len(f["gained"]) + len(f["lost"]) + len(e["gained"]) + len(e["lost"])
        REPORT.append(
            f"  {level:>4} | {f['network']:>10} {f['numpy']:>6} "
            f"| {e['network']:>9} {e['numpy']:>6} | {d}"
        )
        for item in (f["gained"] + f["lost"])[:3]:
            REPORT.append(f"        frozen disagreement at step {level}: {item}")
    h4 = total_disagreements == 0
    hypothesis(
        h4,
        f"H4 {'PASS' if h4 else 'FALSIFIED'}: the emitting sets agree at every step "
        f"in both arms ({total_disagreements} set disagreements over "
        f"{2 * horizon} per-step comparisons)",
    )

    # --------------------------------------------------------------- H5
    REPORT.append("\n5. H5: decision and eta flips against the numpy comparator")
    flips = [
        (m, t, route, s["step"])
        for m, t, route, block in rows for arm in ARMS for s in block[arm]["steps"]
        if s.get("decision_flip_vs_numpy")
    ]
    eta_flips = [
        (m, t, route, s["step"])
        for m, t, route, block in rows for arm in ARMS for s in block[arm]["steps"]
        if s.get("eta_flip_vs_numpy")
    ]
    comparisons = len(steps_all)
    REPORT.append(f"  {comparisons} step entries inspected")
    for item in flips[:8]:
        REPORT.append(
            f"        decision flip: mix={item[0]} task={item[1]} {item[2]} "
            f"step={item[3]}"
        )
    for item in eta_flips[:8]:
        REPORT.append(
            f"        eta flip: mix={item[0]} task={item[1]} {item[2]} step={item[3]}"
        )
    check(
        not flips,
        f"zero decision flips in the same-process comparator ({len(flips)})",
    )
    h5 = not eta_flips
    hypothesis(
        h5,
        f"H5 {'PASS' if h5 else 'FALSIFIED'}: zero eta flips ({len(eta_flips)})",
    )

    # --------------------------------------------------------------- H6
    REPORT.append("\n6. H6: does the float32 gap stay inside ATOL?")
    gaps = [
        max(
            s["q_hat_gap_vs_numpy"]
            for _, _, _, block in rows for s in block[arm]["steps"]
            if s["step"] == level
        )
        for arm in ARMS for level in range(1, horizon + 1)
        if any(
            s["step"] == level for _, _, _, block in rows for s in block[arm]["steps"]
        )
    ]
    worst = max(gaps)
    h6 = worst <= atol
    hypothesis(
        h6,
        f"H6 {'PASS' if h6 else 'FALSIFIED'}: worst gap over all steps and arms "
        f"{worst:.3e} <= {atol:.0e}",
    )
    REPORT.append(
        "  worst gap by step (frozen): "
        + ", ".join(
            f"{max(s['q_hat_gap_vs_numpy'] for _, _, _, b in rows for s in b['frozen']['steps'] if s['step'] == lv):.2e}"
            for lv in range(1, horizon + 1)
        )
    )

    # --------------------------------------------------------------- H7
    REPORT.append("\n7. H7: agreement headroom per step")
    REPORT.append(
        f"  {'step':>4} | {'min gain (frozen)':>17} {'max gap':>10} {'headroom':>10}"
    )
    headrooms = []
    for level in range(1, horizon + 1):
        stats = level_stats(rows, "frozen", level)
        if stats["min_gain"] is None or not stats["max_gap"]:
            REPORT.append(f"  {level:>4} | {'nothing emitted':>17} {'-':>10} {'-':>10}")
            continue
        ratio = stats["min_gain"] / stats["max_gap"]
        headrooms.append(ratio)
        REPORT.append(
            f"  {level:>4} | {stats['min_gain']:>17.3e} {stats['max_gap']:>10.2e} "
            f"{ratio:>9.1f}x"
        )
    worst_headroom = min(headrooms) if headrooms else None
    h7 = worst_headroom is not None and worst_headroom >= H7_MIN_HEADROOM
    hypothesis(
        h7,
        f"H7 {'PASS' if h7 else 'FALSIFIED'}: the worst headroom is "
        f"{'n/a' if worst_headroom is None else f'{worst_headroom:.1f}x'} against the "
        f"registered floor {H7_MIN_HEADROOM:.0f}x",
    )
    REPORT.append(
        "  headroom = min emitted gain / max |dQ| at that step. When it approaches 1, "
        "the decision is being made by float32 arithmetic rather than by the "
        "certificate, and a flip is expected rather than surprising."
    )

    # ------------------------------- 7b. how far can the network be trusted?
    # POST-HOC projection, clearly labelled. The headroom falls from 64,000x at
    # step 1 to 40x at step 11 because the margins shrink geometrically while the
    # float32 gap stays flat. FP-HORIZON-001 ran the SAME iteration on numpy out to
    # 32 steps, so its minimum-gain sequence says where the headroom reaches 1 --
    # i.e. the step beyond which the network's arithmetic, not the certificate,
    # decides whether a record is eligible.
    REPORT.append("\n7b. How far can the network be trusted? (POST-HOC projection)")
    horizon_bundle = PROJECT / "results" / "FP-HORIZON-001" / "claude" / "formal"
    horizon_path = horizon_bundle / "task_results.json"
    crossing = None
    if horizon_path.exists():
        hrows = rows_of(load(horizon_path))
        typical_gap = worst
        crossing = None
        REPORT.append(
            f"  {'step':>4} {'numpy min gain':>15} {'headroom vs this run':>21} "
            f"{'decided by':>14}"
        )
        for lv in range(11, 33):
            gains = [
                s["oracle_audit"]["total_value_gain"]
                for _, _, _, block in hrows
                for s in block["frozen"]["steps"]
                if s["step"] == lv and s["update_emitted"]
            ]
            if not gains:
                continue
            mn = min(gains)
            ratio = mn / typical_gap
            if ratio >= 1.0 and crossing is None:
                decided = "certificate"
            elif ratio >= 1.0:
                decided = "certificate"
            else:
                decided = "arithmetic"
                if crossing is None:
                    crossing = lv
            REPORT.append(
                f"  {lv:>4} {mn:>15.3e} {ratio:>20.1f}x {decided:>14}"
            )
        REPORT.append(
            f"  the float32 gap is flat at ~{typical_gap:.1e} across twelve steps "
            f"(H6), so it is held fixed here while the numpy margin shrinks."
        )
        if crossing is not None:
            REPORT.append(
                f"  PROJECTION: the headroom reaches 1 near step {crossing}. Beyond "
                f"that the network cannot be relied on to reproduce numpy's "
                f"eligibility decisions, because the certified improvements are "
                f"smaller than the network's own arithmetic resolution."
            )
        else:
            REPORT.append(
                "  PROJECTION: the headroom stays above 1 out to step 32, so no "
                "arithmetic-decided step is projected within the measured range."
            )
        REPORT.append(
            "  this is a projection from the numpy margin sequence and a HELD-FIXED "
            "gap; it is not a measurement of the network beyond step 12."
        )
    else:
        REPORT.append("  the FP-HORIZON-001 bundle is absent, so nothing to project")

    # ------------------------------------------------------------- summary
    REPORT.append("\n" + "=" * 92)
    REPORT.append("SUMMARY")
    for arm in ARMS:
        REPORT.append(
            f"  emissions, network {arm:<20}: "
            f"{[level_stats(rows, arm, lv)['count'] for lv in range(1, horizon + 1)]}"
        )
        REPORT.append(
            f"  emissions, numpy   {arm:<20}: "
            f"{[level_stats(numpy_rows, arm, lv)['count'] for lv in range(1, horizon + 1)]}"
        )
    REPORT.append(f"  deepest trajectory : network {deepest}, numpy {np_deepest}")
    REPORT.append(f"  H3 reach           : {'PASS' if h3 else 'FALSIFIED'}")
    REPORT.append(f"  H4 set agreement   : {'PASS' if h4 else 'FALSIFIED'}")
    REPORT.append(f"  H5 eta flips       : {'PASS' if h5 else 'FALSIFIED'}")
    REPORT.append(f"  H6 drift           : {'PASS' if h6 else 'FALSIFIED'}")
    REPORT.append(f"  H7 headroom        : {'PASS' if h7 else 'FALSIFIED'}")
    REPORT.append("=" * 92)
    REPORT.append(
        "hypotheses were falsified, which per the task sheet is a RESULT, not a\n"
        "failed construction. RESULT below covers the construction checks only."
    )
    REPORT.append(
        "CONSTRUCTION: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)")
    )

    summary = {
        "task_id": TASK_ID,
        "horizon": horizon,
        "atol": atol,
        "emissions_network": {
            arm: [level_stats(rows, arm, lv)["count"] for lv in range(1, horizon + 1)]
            for arm in ARMS
        },
        "emissions_numpy": {
            arm: [
                level_stats(numpy_rows, arm, lv)["count"]
                for lv in range(1, horizon + 1)
            ]
            for arm in ARMS
        },
        "deepest_trajectory_network": deepest,
        "deepest_trajectory_numpy": np_deepest,
        "H1_producers": sorted(p for p in producers if p),
        "H1_smallest_gap": smallest_gap,
        "H3": "PASS" if h3 else "FALSIFIED",
        "H4": {
            "verdict": "PASS" if h4 else "FALSIFIED",
            "set_disagreements": total_disagreements,
            "detail": {
                arm: {
                    str(lv): {
                        "network": agreement[arm][lv]["network"],
                        "numpy": agreement[arm][lv]["numpy"],
                        "gained": [list(x) for x in agreement[arm][lv]["gained"]],
                        "lost": [list(x) for x in agreement[arm][lv]["lost"]],
                    }
                    for lv in range(1, horizon + 1)
                }
                for arm in ARMS
            },
        },
        "H5": {"verdict": "PASS" if h5 else "FALSIFIED", "eta_flips": len(eta_flips)},
        "H6": {"verdict": "PASS" if h6 else "FALSIFIED", "worst_gap": worst},
        "H7": {
            "verdict": "PASS" if h7 else "FALSIFIED",
            "worst_headroom": worst_headroom,
            "registered_floor": H7_MIN_HEADROOM,
        },
        "decision_flips_in_comparator": len(flips),
        "comparisons": comparisons,
        "trusted_through_step_posthoc": crossing,
        "trusted_note": (
            "POST-HOC projection: the first step at which the numpy minimum gain "
            "falls below this run's flat float32 gap, so the eligibility decision "
            "would be made by arithmetic rather than by the certificate. Projected "
            "from the FP-HORIZON-001 margin sequence with the gap held fixed; not a "
            "measurement of the network beyond step 12."
        ),
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
