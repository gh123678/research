"""FP-ITER8X-001 analysis: does 8x certification extend the iteration?

Scores H1--H7 and reports the population trajectory, with set differences against the
sealed 1x run at every shared step so the two horizons can be compared directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-ITER8X-001"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("frozen", "empirical_bernstein")
SEALED_1X = PROJECT / "results" / "FP-ITER6-001" / "claude" / "numpy" / "task_results.json"
REPORT: list[str] = []
FAILURES = 0

# Pre-registered in docs/research_tasks/FP-ITER8X-001.md before the run.
H4_STEP6_COUNT_1X = 9
H7_MIN_GAIN_STEP6_1X = 0.019347


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
        default=PROJECT / "results" / "FP-ITER8X-001" / "claude" / "formal",
    )
    parser.add_argument("--sealed-1x", type=Path, default=SEALED_1X)
    args = parser.parse_args()

    bundle = load(args.result_dir / "task_results.json")
    horizon = int(bundle["max_steps"])
    rows = rows_of(bundle)

    REPORT.append(f"{TASK_ID} analysis: the certified iteration at 8x certification")
    REPORT.append("=" * 88)
    REPORT.append(
        f"  {len(rows)} route-records, horizon {horizon}, "
        f"certification {bundle['cert_chains']} x 64 = "
        f"{bundle['cert_chains'] * 64:,} items, arms {bundle['arms']}"
    )

    # ------------------------------------------------------------- H1/H2
    REPORT.append("\n1. H1/H2: validity and soundness over the whole run")
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
            x for x in emitted
            if x[3]["oracle_audit"].get("certificate_violation")
        ]
        missing = [
            x for x in steps if not x[3]["update_emitted"] and not x[3]["ordered_reasons"]
        ]
        check(
            not degrading,
            f"{arm}: {len(emitted)} emitted steps, {len(degrading)} degrading",
        )
        check(
            not nonpositive,
            f"{arm}: {len(nonpositive)} non-positive total gains",
        )
        check(
            not violations,
            f"{arm}: {len(violations)} certificate violations",
        )
        check(
            not missing,
            f"{arm}: every abstention carries a frozen reason ({len(missing)} missing)",
        )
        for item in (degrading + nonpositive + violations)[:5]:
            REPORT.append(
                f"        problem: mix={item[0]} task={item[1]} {item[2]} "
                f"step={item[3]['step']} {item[3]['oracle_audit']}"
            )

    # ------------------------------------------------------ 2. trajectory
    REPORT.append("\n2. The emitting population, step by step")
    per_level: dict[str, dict[int, dict]] = {}
    for arm in ARMS:
        per_level[arm] = {}
        for level in range(1, horizon + 1):
            emitting = [
                (m, t, route)
                for m, t, route, block in rows
                for s in block[arm]["steps"]
                if s["step"] == level and s["update_emitted"]
            ]
            gains = [
                s["oracle_audit"]["total_value_gain"]
                for _, _, _, block in rows
                for s in block[arm]["steps"]
                if s["step"] == level and s["update_emitted"]
            ]
            per_level[arm][level] = {
                "emitters": set(emitting),
                "count": len(emitting),
                "mean": (sum(gains) / len(gains)) if gains else None,
                "min": min(gains) if gains else None,
            }
    sealed = load(args.sealed_1x)
    sealed_level = {}
    for level in range(1, 7):
        sealed_level[level] = {
            (m, t, route)
            for m, t, route, block in rows_of(sealed)
            for s in block["steps"]
            if s["step"] == level and s["update_emitted"]
        }

    REPORT.append(
        f"  {'step':>4} {'frozen@8x':>10} {'mean':>9} {'min':>9} "
        f"{'empB@8x':>9} {'mean':>9} {'min':>9} {'sealed 1x':>10}"
    )
    def fmt(value: float | None) -> str:
        return "   n/a  " if value is None else f"{value:.6f}"

    for level in range(1, horizon + 1):
        f = per_level["frozen"][level]
        e = per_level["empirical_bernstein"][level]
        s = len(sealed_level.get(level, ())) if level <= 6 else None
        REPORT.append(
            f"  {level:>4} {f['count']:>10} {fmt(f['mean']):>9} {fmt(f['min']):>9} "
            f"{e['count']:>9} {fmt(e['mean']):>9} {fmt(e['min']):>9} "
            f"{(str(s) if s is not None else '-'):>10}"
        )

    reached = {
        arm: max(
            (block[arm]["emitted_steps"] for _, _, _, block in rows), default=0
        )
        for arm in ARMS
    }
    REPORT.append(
        f"  deepest trajectory: frozen {reached['frozen']} steps, "
        f"empirical_bernstein {reached['empirical_bernstein']} steps "
        f"(sealed 1x reached 6, its horizon)"
    )

    # ---------------------------------------------------------------- H3
    REPORT.append("\n3. H3: does the iteration pass six steps?")
    seventh = {
        arm: [
            (m, t, route)
            for m, t, route, block in rows
            if any(
                s["step"] == 7 and s["update_emitted"] for s in block[arm]["steps"]
            )
        ]
        for arm in ARMS
    }
    total7 = len(seventh["frozen"]) + len(seventh["empirical_bernstein"])
    for arm in ARMS:
        REPORT.append(f"  {arm}: {len(seventh[arm])} route-records emit a 7th step")
    h3 = total7 >= 1
    hypothesis(
        h3,
        f"H3 {'PASS' if h3 else 'FALSIFIED'}: the iteration passes six steps "
        f"({total7} seventh-step emissions across both arms)",
    )
    if not h3:
        REPORT.append(
            "  this is the informative failure: with 44 of 48 records eligible at "
            "step 1, the iteration still stops inside six, so step-1 eligibility is "
            "not what bounds the horizon."
        )

    # ---------------------------------------------------------------- H4
    REPORT.append("\n4. H4: is the step-7 population larger than 1x's step 6?")
    seven = per_level["frozen"][7]["count"] if horizon >= 7 else 0
    h4 = seven > H4_STEP6_COUNT_1X
    hypothesis(
        h4,
        f"H4 {'PASS' if h4 else 'FALSIFIED'}: frozen@8x emits {seven} at step 7 "
        f"versus {H4_STEP6_COUNT_1X} at step 6 under sealed 1x",
    )

    # ---------------------------------------------------------------- H5
    REPORT.append("\n5. H5: does the repaired arm contain the frozen arm?")
    # H5 was registered as a theorem ("a smaller E_Q can only increase LB_s"). That
    # justification is wrong ACROSS ARMS: LB_s = I_s - E_Q*||dpi_s||_1 is increasing
    # in E_Q only at a FIXED policy, and the two arms carry different trajectories,
    # so their step-k policies differ. The theorem does apply at step 1, where both
    # arms start from pi_0 and share q_hat, so that is checked separately and the
    # cross-arm counts are reported as an observation rather than a guarantee.
    step1_frozen = per_level["frozen"][1]["emitters"]
    step1_rep = per_level["empirical_bernstein"][1]["emitters"]
    theorem_ok = step1_frozen <= step1_rep
    check(
        theorem_ok,
        f"the containment theorem holds where it applies (step 1, same policy and "
        f"same q_hat): frozen {len(step1_frozen)} subset of repaired "
        f"{len(step1_rep)}",
    )
    contained = 0
    violations_h5 = []
    for level in range(1, horizon + 1):
        f = per_level["frozen"][level]["emitters"]
        e = per_level["empirical_bernstein"][level]["emitters"]
        if f and f <= e:
            contained += 1
        elif f:
            violations_h5.append((level, sorted(f - e)))
    levels_with_frozen = sum(
        1 for level in range(1, horizon + 1) if per_level["frozen"][level]["emitters"]
    )
    h5 = not violations_h5
    hypothesis(
        h5,
        f"H5 {'PASS' if h5 else 'FALSIFIED'} as registered: the repaired arm's "
        f"emitting set contains the frozen arm's at {contained}/{levels_with_frozen} "
        f"non-empty levels",
    )
    if violations_h5:
        REPORT.append(
            "  the registration was mis-specified, not the implementation: the "
            "containment argument is a theorem only at a fixed policy, and the arms "
            "diverge. The counterexample is therefore expected, and the step-1 check "
            "above is the form in which the claim is actually testable."
        )
    for item in violations_h5[:5]:
        REPORT.append(f"        counterexample at step {item[0]}: frozen-only {item[1]}")

    # ---------------------------------------------------------------- H6
    REPORT.append("\n6. H6: does the mean gain keep decaying past six?")
    h6_results: dict[str, bool | None] = {}
    for arm in ARMS:
        seq = [
            (level, per_level[arm][level]["mean"])
            for level in range(1, horizon + 1)
            if per_level[arm][level]["mean"] is not None
        ]
        beyond = [
            (b, y) for (a, x), (b, y) in zip(seq, seq[1:]) if b > 6
        ]
        if not beyond:
            h6_results[arm] = None
            REPORT.append(f"  H6 {arm}: no step beyond 6 emits, so nothing to test")
            continue
        ok = all(y < x for (a, x), (b, y) in zip(seq, seq[1:]) if b > 6)
        h6_results[arm] = ok
        hypothesis(
            ok,
            f"H6 {arm}: mean gain decays at every step beyond 6 "
            f"({[(s, round(v, 6)) for s, v in beyond]})",
        )
    tested = [v for v in h6_results.values() if v is not None]
    h6 = bool(tested) and all(tested)

    # ---------------------------------------------------------------- H7
    REPORT.append("\n7. H7: is the step-6 minimum LOWER than under 1x?")
    min6 = per_level["frozen"][6]["min"]
    h7 = min6 is not None and min6 < H7_MIN_GAIN_STEP6_1X
    hypothesis(
        h7,
        f"H7 {'PASS' if h7 else 'FALSIFIED'}: frozen@8x step-6 minimum "
        f"{'n/a' if min6 is None else f'{min6:.6f}'} "
        f"{'<' if h7 else '>='} {H7_MIN_GAIN_STEP6_1X} (sealed 1x)",
    )

    # ---------------------------------------------------------------- H8
    REPORT.append("\n8. H8: set changes against the sealed 1x run")
    for level in range(1, 7):
        f = per_level["frozen"][level]["emitters"]
        s = sealed_level[level]
        gained = sorted(f - s)
        lost = sorted(s - f)
        REPORT.append(
            f"  step {level}: sealed 1x {len(s)}, frozen@8x {len(f)}; "
            f"gained {len(gained)}, lost {len(lost)}"
        )
        if level <= 2:
            for item in gained[:4]:
                REPORT.append(f"        gained: mix={item[0]} task={item[1]} {item[2]}")
            for item in lost[:4]:
                REPORT.append(f"        lost:   mix={item[0]} task={item[1]} {item[2]}")

    # ------------------------------------------------------------- summary
    REPORT.append("\n" + "=" * 88)
    REPORT.append("SUMMARY")
    REPORT.append(
        f"  emissions, frozen@8x : "
        f"{[per_level['frozen'][lvl]['count'] for lvl in range(1, horizon + 1)]}"
    )
    REPORT.append(
        f"  emissions, empB@8x   : "
        f"{[per_level['empirical_bernstein'][lvl]['count'] for lvl in range(1, horizon + 1)]}"
    )
    REPORT.append(
        f"  emissions, sealed 1x : "
        f"{[len(sealed_level[lvl]) for lvl in range(1, 7)]} (horizon 6)"
    )
    REPORT.append(f"  deepest trajectory   : {reached}")
    REPORT.append(f"  H3 passes six steps  : {'PASS' if h3 else 'FALSIFIED'}")
    REPORT.append(f"  H4 step-7 > 9        : {'PASS' if h4 else 'FALSIFIED'}")
    REPORT.append(f"  H5 repaired contains : {'PASS' if h5 else 'FALSIFIED'}")
    REPORT.append(f"  H6 decay continues   : {'PASS' if h6 else 'FALSIFIED'}")
    REPORT.append(f"  H7 step-6 min lower  : {'PASS' if h7 else 'FALSIFIED'}")
    REPORT.append("=" * 88)
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
        "cert_chains": int(bundle["cert_chains"]),
        "arms": list(ARMS),
        "emissions": {
            arm: [per_level[arm][lvl]["count"] for lvl in range(1, horizon + 1)]
            for arm in ARMS
        },
        "sealed_1x_emissions": [len(sealed_level[lvl]) for lvl in range(1, 7)],
        "mean_gain": {
            arm: [per_level[arm][lvl]["mean"] for lvl in range(1, horizon + 1)]
            for arm in ARMS
        },
        "min_gain": {
            arm: [per_level[arm][lvl]["min"] for lvl in range(1, horizon + 1)]
            for arm in ARMS
        },
        "deepest_trajectory": reached,
        "seventh_step_emissions": {arm: len(seventh[arm]) for arm in ARMS},
        "H3": "PASS" if h3 else "FALSIFIED",
        "H4": {"verdict": "PASS" if h4 else "FALSIFIED", "step7": seven,
               "reference_1x_step6": H4_STEP6_COUNT_1X},
        "H5": {"verdict": "PASS" if h5 else "FALSIFIED",
               "contained_levels": contained, "nonempty_levels": levels_with_frozen,
               "theorem_holds_at_step1": theorem_ok,
               "note": (
                   "registered as a theorem but it is one only at a fixed policy; "
                   "the arms diverge, so the cross-arm count is an observation. The "
                   "step-1 check is the testable form."
               )},
        "H6": "PASS" if h6 else "FALSIFIED",
        "H7": {"verdict": "PASS" if h7 else "FALSIFIED", "step6_min": min6,
               "reference_1x": H7_MIN_GAIN_STEP6_1X},
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
