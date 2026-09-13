"""FP-HORIZON-001 analysis: where does the certified iteration stop?

Scores H1--H6 and H8 from the bundle. The population trajectory, the deepest
trajectory per arm, the stopping step and the cumulative value gain are all reported,
because "how far" is only half the question and the other half is "how much".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-HORIZON-001"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("frozen", "empirical_bernstein")
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
        default=PROJECT / "results" / "FP-HORIZON-001" / "claude" / "formal",
    )
    args = parser.parse_args()

    bundle = load(args.result_dir / "task_results.json")
    horizon = int(bundle["max_steps"])
    rows = rows_of(bundle)

    REPORT.append(f"{TASK_ID} analysis: where does the certified iteration stop?")
    REPORT.append("=" * 92)
    REPORT.append(
        f"  {len(rows)} route-records, horizon {horizon}, "
        f"certification {bundle['cert_chains'] * 64:,} items, arms {bundle['arms']}"
    )

    # ------------------------------------------------------------- H1/H2
    REPORT.append("\n1. H1/H2: validity and soundness (mandatory)")
    for arm in ARMS:
        steps = [
            (m, t, route, s)
            for m, t, route, block in rows
            for s in block[arm]["steps"]
        ]
        emitted = [x for x in steps if x[3]["update_emitted"]]
        degrading = [
            x for x in emitted
            if not x[3]["oracle_audit"].get("componentwise_nondegrading", False)
        ]
        nonpositive = [
            x for x in emitted if x[3]["oracle_audit"]["total_value_gain"] <= 0.0
        ]
        violations = [
            x for x in emitted if x[3]["oracle_audit"].get("certificate_violation")
        ]
        missing = [
            x for x in steps
            if not x[3]["update_emitted"] and not x[3]["ordered_reasons"]
        ]
        check(
            not degrading,
            f"{arm}: {len(emitted)} emitted steps, {len(degrading)} componentwise "
            f"degrading",
        )
        check(not nonpositive, f"{arm}: {len(nonpositive)} non-positive total gains")
        check(not violations, f"{arm}: {len(violations)} certificate violations")
        check(not missing, f"{arm}: {len(missing)} abstentions without a reason")
        for item in (degrading + nonpositive + violations)[:5]:
            REPORT.append(
                f"        problem: mix={item[0]} task={item[1]} {item[2]} "
                f"step={item[3]['step']}"
            )

    # ---------------------------------------------------- 2. the trajectory
    REPORT.append("\n2. The emitting population")
    per_level: dict[str, dict[int, dict]] = {}
    for arm in ARMS:
        per_level[arm] = {}
        for level in range(1, horizon + 1):
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
            per_level[arm][level] = {
                "emitters": emitters,
                "count": len(emitters),
                "mean": (sum(gains) / len(gains)) if gains else None,
                "min": min(gains) if gains else None,
                "max": max(gains) if gains else None,
                "negative": sum(1 for g in gains if g <= 0.0),
            }

    header = (
        f"  {'step':>4} | {'frozen':>6} {'mean':>9} {'min':>10} "
        f"| {'empB':>5} {'mean':>9} {'min':>10}"
    )
    REPORT.append(header)
    last_nonzero = 0
    for level in range(1, horizon + 1):
        f = per_level["frozen"][level]
        e = per_level["empirical_bernstein"][level]
        if f["count"] or e["count"]:
            last_nonzero = level

        def fmt(v):
            return "  n/a  " if v is None else f"{v:.6f}"

        REPORT.append(
            f"  {level:>4} | {f['count']:>6} {fmt(f['mean']):>9} {fmt(f['min']):>10} "
            f"| {e['count']:>5} {fmt(e['mean']):>9} {fmt(e['min']):>10}"
        )
    REPORT.append(f"  last step with any emission: {last_nonzero}")

    deepest = {
        arm: max((b[arm]["emitted_steps"] for _, _, _, b in rows), default=0)
        for arm in ARMS
    }
    REPORT.append(
        f"  deepest trajectory: frozen {deepest['frozen']} steps, "
        f"empirical_bernstein {deepest['empirical_bernstein']} steps "
        f"(horizon {horizon})"
    )

    # ---------------------------------------------------------------- H3
    REPORT.append("\n3. H3: does the iteration terminate before the horizon?")
    h3 = deepest["frozen"] < horizon and deepest["empirical_bernstein"] < horizon
    hypothesis(
        h3,
        f"H3 {'PASS' if h3 else 'FALSIFIED'}: deepest trajectory "
        f"{max(deepest.values())} < horizon {horizon}",
    )
    if not h3:
        REPORT.append(
            "  the informative failure: at least one route-record is still emitting "
            "at the last step, so the certified iteration has NOT been shown to "
            "terminate. It keeps finding valid improvements on margins near "
            "numerical dust."
        )

    # ---------------------------------------------------------------- H4
    REPORT.append("\n4. H4: is the population non-increasing?")
    for arm in ARMS:
        seq = [per_level[arm][lv]["count"] for lv in range(1, horizon + 1)]
        rises = [
            (lv, seq[lv - 2], seq[lv - 1])
            for lv in range(2, horizon + 1)
            if seq[lv - 1] > seq[lv - 2]
        ]
        hypothesis(
            not rises,
            f"H4 {arm}: population non-increasing "
            f"({'no rises' if not rises else f'{len(rises)} rises, first at step {rises[0][0]}: {rises[0][1]} -> {rises[0][2]}'})",
        )

    # ---------------------------------------------------------------- H5
    REPORT.append("\n5. H5: does the mean gain keep decaying past step 12?")
    h5_results = {}
    for arm in ARMS:
        seq = [
            (lv, per_level[arm][lv]["mean"])
            for lv in range(1, horizon + 1)
            if per_level[arm][lv]["mean"] is not None
        ]
        beyond = [(b, y) for (a, x), (b, y) in zip(seq, seq[1:]) if b > 12]
        if not beyond:
            h5_results[arm] = None
            REPORT.append(f"  H5 {arm}: nothing emits beyond 12, so nothing to test")
            continue
        rises = [(b, x, y) for (a, x), (b, y) in zip(seq, seq[1:]) if b > 12 and y >= x]
        h5_results[arm] = not rises
        hypothesis(
            not rises,
            f"H5 {arm}: mean gain decays at every step past 12 "
            f"({'no rises' if not rises else f'{len(rises)} rises, first at step {rises[0][0]}: {rises[0][1]:.6f} -> {rises[0][2]:.6f}'})",
        )
    tested = [v for v in h5_results.values() if v is not None]
    h5 = bool(tested) and all(tested)

    # ---------------------------------------------------------------- H6
    REPORT.append("\n6. H6: is every emitted gain strictly positive?")
    for arm in ARMS:
        negatives = sum(
            per_level[arm][lv]["negative"] for lv in range(1, horizon + 1)
        )
        smallest = min(
            (
                per_level[arm][lv]["min"]
                for lv in range(1, horizon + 1)
                if per_level[arm][lv]["min"] is not None
            ),
            default=None,
        )
        hypothesis(
            negatives == 0,
            f"H6 {arm}: {negatives} non-positive emitted gains; smallest emitted "
            f"gain {'n/a' if smallest is None else f'{smallest:.3e}'}",
        )
    h6 = all(
        sum(per_level[arm][lv]["negative"] for lv in range(1, horizon + 1)) == 0
        for arm in ARMS
    )

    # ------------------------------------------------------------- H8
    REPORT.append("\n7. H8: what did the whole trajectory buy?")
    value_summary = {}
    for arm in ARMS:
        totals = []
        per_state_min = []
        for _, _, _, block in rows:
            start = block[arm]["start_policy_value"]
            final = block[arm]["final_policy_value"]
            delta = [f - s for s, f in zip(start, final)]
            totals.append(sum(delta))
            per_state_min.append(min(delta))
        value_summary[arm] = {
            "total_gain_mean": sum(totals) / len(totals),
            "total_gain_min": min(totals),
            "total_gain_max": max(totals),
            "worst_state_gain_min": min(per_state_min),
            "worst_state_gain_mean": sum(per_state_min) / len(per_state_min),
            "records_with_all_states_improved": sum(
                1 for v in per_state_min if v > 0.0
            ),
        }
        v = value_summary[arm]
        REPORT.append(
            f"  {arm}: total value gain per route-record "
            f"mean {v['total_gain_mean']:.6f}, range "
            f"[{v['total_gain_min']:.6f}, {v['total_gain_max']:.6f}]"
        )
        REPORT.append(
            f"      worst state per record: mean {v['worst_state_gain_mean']:.6f}, "
            f"min {v['worst_state_gain_min']:.6f}; "
            f"{v['records_with_all_states_improved']}/{len(rows)} records improved "
            f"every state"
        )

    # ------------------------------------------- 7b. meaningful reach (post hoc)
    # POST-HOC, not pre-registered. "Reaches 32" is only decision-relevant if the
    # gains are still worth having, and by step 32 the smallest emitted gain is
    # ~4e-7. This converts the reach into a statement at several gain floors, so a
    # reader can pick the floor they consider meaningful rather than accepting the
    # horizon as the answer.
    REPORT.append("\n7b. Meaningful reach at several gain floors (POST-HOC)")
    floors = (0.05, 0.01, 1e-3, 1e-4, 1e-6)
    REPORT.append(
        f"  {'floor':>9} | {'frozen: last step':>18} {'records':>8} "
        f"| {'empB: last step':>17} {'records':>8}"
    )
    reach: dict[str, dict[str, int]] = {arm: {} for arm in ARMS}
    for floor in floors:
        row = f"  {floor:>9.0e} |"
        for arm in ARMS:
            last = 0
            count = 0
            for lv in range(1, horizon + 1):
                gains = [
                    s["oracle_audit"]["total_value_gain"]
                    for _, _, _, block in rows
                    for s in block[arm]["steps"]
                    if s["step"] == lv and s["update_emitted"]
                ]
                if gains and (sum(gains) / len(gains)) >= floor:
                    last = lv
                    count = len(gains)
            reach[arm][f"{floor:.0e}"] = {"last_step": last, "records": count}
            row += f" {last:>18} {count:>8} |"
        REPORT.append(row)
    REPORT.append(
        "  'last step' is the last step whose MEAN gain is at or above the floor, "
        "and 'records' is how many emitted there."
    )
    REPORT.append(
        "  the choice of floor is a judgement this task does not make: a gain of "
        "4e-7 is certified, componentwise non-degrading and strictly positive, and "
        "whether that counts as 'policy improvement' is a definitional question the "
        "line has not settled."
    )

    # ------------------------------------------------------------- summary
    REPORT.append("\n" + "=" * 92)
    REPORT.append("SUMMARY")
    REPORT.append(
        f"  emissions, frozen : {[per_level['frozen'][lv]['count'] for lv in range(1, horizon + 1)]}"
    )
    REPORT.append(
        f"  emissions, empB   : "
        f"{[per_level['empirical_bernstein'][lv]['count'] for lv in range(1, horizon + 1)]}"
    )
    REPORT.append(f"  deepest trajectory: {deepest} (horizon {horizon})")
    REPORT.append(f"  last emitting step: {last_nonzero}")
    REPORT.append(f"  H3 terminates     : {'PASS' if h3 else 'FALSIFIED'}")
    REPORT.append(f"  H5 decay past 12  : {'PASS' if h5 else 'FALSIFIED'}")
    REPORT.append(f"  H6 positive gains : {'PASS' if h6 else 'FALSIFIED'}")
    for arm in ARMS:
        REPORT.append(
            f"  H8 total gain {arm:<20}: "
            f"{value_summary[arm]['total_gain_mean']:.6f}"
        )
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
        "emissions": {
            arm: [per_level[arm][lv]["count"] for lv in range(1, horizon + 1)]
            for arm in ARMS
        },
        "mean_gain": {
            arm: [per_level[arm][lv]["mean"] for lv in range(1, horizon + 1)]
            for arm in ARMS
        },
        "min_gain": {
            arm: [per_level[arm][lv]["min"] for lv in range(1, horizon + 1)]
            for arm in ARMS
        },
        "deepest_trajectory": deepest,
        "last_emitting_step": last_nonzero,
        "value": value_summary,
        "meaningful_reach_posthoc": reach,
        "H3": "PASS" if h3 else "FALSIFIED",
        "H5": "PASS" if h5 else "FALSIFIED",
        "H6": "PASS" if h6 else "FALSIFIED",
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
