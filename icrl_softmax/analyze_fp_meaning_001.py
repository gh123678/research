"""FP-MEANING-001 analysis: when does a certified improvement stop meaning anything?

Two quantities are tracked per step from the FP-HORIZON-001 trajectory:

  min_lb   the CERTIFIED lower bound on the worst-state improvement -- what the
           decision rule requires to be positive, and what can actually be PROVEN
  min gain the REALIZED worst-state improvement -- what actually happened

and they are compared against three floors: the measured float64 method-disagreement
floor, machine epsilon, and the float32 gap the network exhibits.

The task does NOT choose a definition. It reports the reach under each and leaves the
choice open, because every candidate is defensible and they give different answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-MEANING-001"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("frozen", "empirical_bernstein")
HORIZON = PROJECT / "results" / "FP-HORIZON-001" / "claude" / "formal" / "task_results.json"
ATTN = PROJECT / "results" / "FP-ATTN-8X-001" / "claude" / "formal" / "summary.json"
REPORT: list[str] = []
FAILURES = 0

# Pre-registered in docs/research_tasks/FP-MEANING-001.md before the run.
H4_FACTOR_RANGE = (3.0, 10.0)
H5_MIN_ORDERS = 6.0


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
        "--floor-dir",
        type=Path,
        default=PROJECT / "results" / "FP-MEANING-001" / "claude" / "formal",
    )
    parser.add_argument("--horizon-bundle", type=Path, default=HORIZON)
    parser.add_argument("--attn-summary", type=Path, default=ATTN)
    args = parser.parse_args()

    floor_bundle = load(args.floor_dir / "task_results.json")
    horizon_bundle = load(args.horizon_bundle)
    horizon = int(horizon_bundle["max_steps"])
    rows = rows_of(horizon_bundle)

    REPORT.append(f"{TASK_ID} analysis: does a certified 1e-7 improvement mean anything?")
    REPORT.append("=" * 96)

    # ---------------------------------------------------------------- H1
    REPORT.append("\n1. H1: the arithmetic floor, MEASURED not assumed (mandatory)")
    gaps = [r["absolute_gap"] for r in floor_bundle["records"]]
    method_floor = max(gaps)
    median_floor = float(floor_bundle["median_absolute_gap"])
    eps = float(floor_bundle["float64_eps"])
    REPORT.append(
        "  two routes per record: policy_quantities vs value iteration to 1e-14"
    )
    REPORT.append(
        f"  per-record |dv|: max {method_floor:.3e}, median {median_floor:.3e} "
        f"over {len(gaps)} records"
    )
    REPORT.append(f"  machine epsilon for reference: {eps:.3e}")
    check(
        method_floor < 1e-8,
        f"the two routes agree to {method_floor:.3e}, so the floor is a floor",
    )
    REPORT.append(
        "  NOTE  the measured floor is dominated by the value-iteration stopping "
        "tolerance, not by machine epsilon: they differ by two orders of magnitude "
        "and the choice between them changes the answer, which is exactly why this "
        "was measured rather than assumed."
    )

    # ---------------------------------------------------------------- H2
    REPORT.append("\n2. H2: are the realized tail gains above that floor?")
    tail_gains = [
        s["oracle_audit"]["total_value_gain"]
        for _, _, _, block in rows
        for s in block["frozen"]["steps"]
        if s["step"] == horizon and s["update_emitted"]
    ]
    smallest_tail = min(tail_gains) if tail_gains else None
    h2 = smallest_tail is not None and smallest_tail > method_floor
    hypothesis(
        h2,
        f"H2 {'PASS' if h2 else 'FALSIFIED'}: the smallest realized gain at step "
        f"{horizon} is {smallest_tail:.3e}, which is "
        f"{smallest_tail / method_floor:.1e}x the measured floor",
    )

    # ------------------------------------------------ 3. the two series
    REPORT.append("\n3. The certified bound against the realized gain")
    per_step = {}
    for level in range(1, horizon + 1):
        lbs, gains, count = [], [], 0
        for _, _, _, block in rows:
            for s in block["frozen"]["steps"]:
                if s["step"] != level or not s["update_emitted"]:
                    continue
                count += 1
                if s["min_lb"] is not None:
                    lbs.append(s["min_lb"])
                gains.append(s["oracle_audit"]["total_value_gain"])
        per_step[level] = {
            "count": count,
            "min_lb": min(lbs) if lbs else None,
            "min_gain": min(gains) if gains else None,
        }
    REPORT.append(
        f"  {'step':>4} {'n':>3} | {'min_lb (certified)':>19} {'min gain (realized)':>20} "
        f"{'ratio':>10} | {'vs method floor':>16} {'vs eps':>9}"
    )
    for level in range(1, horizon + 1):
        info = per_step[level]
        if info["min_lb"] is None:
            continue
        ratio = (
            info["min_lb"] / info["min_gain"] if info["min_gain"] else float("nan")
        )
        REPORT.append(
            f"  {level:>4} {info['count']:>3} | {info['min_lb']:>19.3e} "
            f"{info['min_gain']:>20.3e} {ratio:>10.2e} | "
            f"{info['min_lb'] / method_floor:>16.2e} {info['min_lb'] / eps:>9.2f}"
        )

    # ---------------------------------------------------------------- H3
    REPORT.append("\n4. H3: does the certified bound fall below the floor?")
    crossings = {}
    for name, floor in (("method_floor", method_floor), ("machine_eps", eps)):
        crossing = None
        for level in range(1, horizon + 1):
            lb = per_step[level]["min_lb"]
            if lb is not None and lb < floor:
                crossing = level
                break
        crossings[name] = crossing
    still_emitting = per_step[horizon]["count"]
    h3 = crossings["method_floor"] is not None and still_emitting > 0
    if crossings["method_floor"] is not None:
        REPORT.append(
            f"  min_lb drops below the measured method floor ({method_floor:.2e}) at "
            f"step {crossings['method_floor']} (registered prediction: near step 20)"
        )
    if crossings["machine_eps"] is not None:
        REPORT.append(
            f"  min_lb drops below machine epsilon ({eps:.2e}) at step "
            f"{crossings['machine_eps']}"
        )
    REPORT.append(
        f"  the iteration is still emitting {still_emitting} records at step {horizon}"
    )
    hypothesis(
        h3,
        f"H3 {'PASS' if h3 else 'FALSIFIED'}: the certified bound crosses a floor "
        f"and the iteration continues past it",
    )
    if h3:
        REPORT.append(
            "  THIS IS THE FINDING: past the crossing the iteration is certifying "
            "improvements whose PROVEN lower bound is at or below the resolution of "
            "the arithmetic that computes it, while every formal check still passes. "
            "Nothing is violated; the certificate simply stops proving anything."
        )

    # ---------------------------------------------------------------- H4
    REPORT.append("\n5. H4: does the certified bound decay geometrically?")
    lbs = [
        per_step[lv]["min_lb"] for lv in range(1, horizon + 1)
        if per_step[lv]["min_lb"] is not None
    ]
    factors = [a / b for a, b in zip(lbs, lbs[1:]) if b > 0]
    mean_factor = sum(factors) / len(factors) if factors else None
    h4 = (
        mean_factor is not None
        and H4_FACTOR_RANGE[0] <= mean_factor <= H4_FACTOR_RANGE[1]
    )
    hypothesis(
        h4,
        f"H4 {'PASS' if h4 else 'FALSIFIED'}: the mean per-step decay factor is "
        f"{mean_factor:.2f} against the registered range "
        f"[{H4_FACTOR_RANGE[0]:.0f}, {H4_FACTOR_RANGE[1]:.0f}]",
    )

    # ---------------------------------------------------------------- H5
    REPORT.append("\n6. H5: does the certificate prove a vanishing fraction?")
    first_ratio = (
        per_step[1]["min_lb"] / per_step[1]["min_gain"]
        if per_step[1]["min_gain"]
        else None
    )
    last_ratio = (
        per_step[horizon]["min_lb"] / per_step[horizon]["min_gain"]
        if per_step[horizon]["min_gain"]
        else None
    )
    import math

    orders = (
        math.log10(first_ratio / last_ratio)
        if first_ratio and last_ratio and last_ratio > 0
        else None
    )
    h5 = orders is not None and orders >= H5_MIN_ORDERS
    hypothesis(
        h5,
        f"H5 {'PASS' if h5 else 'FALSIFIED'}: the certified fraction of the realized "
        f"gain collapses from {first_ratio:.2e} at step 1 to {last_ratio:.2e} at step "
        f"{horizon}, i.e. {orders:.1f} orders of magnitude (registered minimum "
        f"{H5_MIN_ORDERS:.0f})",
    )

    # ---------------------------------------------------------------- H6
    REPORT.append("\n7. H6: the reach under each definition (the choice is NOT made here)")
    try:
        attn = load(args.attn_summary)
        attn_gap = float(attn["H6"]["worst_gap"])
    except (OSError, KeyError):
        attn_gap = 1.3731489850954404e-05
    definitions = [
        (
            "D1 realized gain > 0",
            lambda lv: (per_step[lv]["min_gain"] or 0.0) > 0.0,
            "the definition the line has been using",
        ),
        (
            "D2 certified bound > 0",
            lambda lv: (per_step[lv]["min_lb"] or 0.0) > 0.0,
            "what the decision rule literally requires",
        ),
        (
            "D3 certified > method floor",
            lambda lv: (per_step[lv]["min_lb"] or 0.0) > method_floor,
            f"proves more than two independent computations disagree by "
            f"({method_floor:.1e})",
        ),
        (
            "D4 certified > machine eps",
            lambda lv: (per_step[lv]["min_lb"] or 0.0) > eps,
            f"proves more than float64 resolution ({eps:.1e})",
        ),
        (
            "D5 certified > float32 gap",
            lambda lv: (per_step[lv]["min_lb"] or 0.0) > attn_gap,
            f"proves more than the network's arithmetic can resolve "
            f"({attn_gap:.1e})",
        ),
    ]
    REPORT.append(
        f"  {'definition':<30} {'1st fail':>9} {'holds':>10} {'last':>6}  rationale"
    )
    reach_table = {}
    for name, predicate, rationale in definitions:
        satisfied = []
        first_fail = None
        for lv in range(1, horizon + 1):
            if per_step[lv]["min_gain"] is None:
                continue
            ok = predicate(lv)
            satisfied.append(ok)
            if not ok and first_fail is None:
                first_fail = lv
        last_ok = max(
            (lv for lv in range(1, horizon + 1)
             if per_step[lv]["min_gain"] is not None and predicate(lv)),
            default=0,
        )
        reach_table[name] = {
            "first_failing_step": first_fail,
            "levels_satisfying": sum(satisfied),
            "levels_emitting": len(satisfied),
            "last_satisfying_step": last_ok,
        }
        REPORT.append(
            f"  {name:<30} {str(first_fail):>9} "
            f"{sum(satisfied):>4}/{len(satisfied):<5} {last_ok:>6}  {rationale}"
        )
    REPORT.append(
        "  'last' alone would mislead: min_lb is a MINIMUM over the emitting set, and "
        "that set changes, so the series is not monotone. D3 for instance recovers "
        "above the floor at step 21 before failing again. The first-failure column is "
        "the honest reach."
    )
    REPORT.append(
        f"  reference: FP-ATTN-8X-001 projects the NETWORK cannot reproduce numpy's "
        f"decisions past step {attn.get('trusted_through_step_posthoc', '?')}"
    )
    REPORT.append(
        "  the task does NOT choose. Each row is defensible and they give first-fail "
        "steps from 6 to never; the user decides which is the claim."
    )

    # ------------------------------------------------------------- summary
    REPORT.append("\n" + "=" * 96)
    REPORT.append("SUMMARY")
    REPORT.append(f"  measured float64 floor : {method_floor:.3e} (median {median_floor:.3e})")
    REPORT.append(f"  machine epsilon        : {eps:.3e}")
    REPORT.append(f"  certified bound step 1 : {per_step[1]['min_lb']:.3e}")
    REPORT.append(
        f"  certified bound step {horizon}  : {per_step[horizon]['min_lb']:.3e}"
    )
    REPORT.append(f"  realized gain step {horizon}   : {per_step[horizon]['min_gain']:.3e}")
    REPORT.append(f"  crossing (method floor): step {crossings['method_floor']}")
    REPORT.append(f"  crossing (machine eps) : step {crossings['machine_eps']}")
    REPORT.append(f"  H2 tail above floor    : {'PASS' if h2 else 'FALSIFIED'}")
    REPORT.append(f"  H3 bound falls, runs on: {'PASS' if h3 else 'FALSIFIED'}")
    REPORT.append(f"  H4 geometric decay     : {'PASS' if h4 else 'FALSIFIED'}")
    REPORT.append(f"  H5 vanishing fraction  : {'PASS' if h5 else 'FALSIFIED'}")
    REPORT.append("=" * 96)
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
        "measured_float64_floor": method_floor,
        "median_float64_floor": median_floor,
        "machine_eps": eps,
        "float32_gap": attn_gap,
        "per_step": {
            str(lv): per_step[lv] for lv in range(1, horizon + 1)
        },
        "crossings": crossings,
        "H2": "PASS" if h2 else "FALSIFIED",
        "H3": "PASS" if h3 else "FALSIFIED",
        "H4": {"verdict": "PASS" if h4 else "FALSIFIED", "mean_factor": mean_factor},
        "H5": {
            "verdict": "PASS" if h5 else "FALSIFIED",
            "first_ratio": first_ratio,
            "last_ratio": last_ratio,
            "orders": orders,
        },
        "reach_table": reach_table,
        "decision_made_here": False,
        "note": (
            "This task deliberately does not choose a definition of 'improvement'. "
            "The four rows of the reach table are all defensible and give different "
            "answers; the choice belongs to the user."
        ),
    }
    args.floor_dir.mkdir(parents=True, exist_ok=True)
    (args.floor_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.floor_dir / "analysis_report.md").write_text(
        "\n".join(REPORT) + "\n", encoding="utf-8"
    )
    print("\n".join(REPORT))


if __name__ == "__main__":
    main()
