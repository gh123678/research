"""FP-ITER6-001 analysis: the sixth certified step against the frozen prediction.

Reads the sixth-step numpy bundle, the sealed FP-ITER5-001 numpy bundle (for the
horizon-inertness proof and the scoring population) and the frozen
``prediction_step6.json`` written by FP-CENSUS-001 before this run.

Scores both pre-registered rules over the same population and reports each
hypothesis as PASS or FALSIFIED. No prediction is edited here, and a rule that
loses is reported as having lost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-ITER6-001"
PRIMARY = ("expected_exact", "expected_finite")
ITER5_NUMPY = (
    PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy" / "task_results.json"
)
CENSUS_DIR = PROJECT / "results" / "FP-CENSUS-001" / "claude" / "formal"
REPORT: list[str] = []
FAILURES = 0
LEVELS = 6


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
    """A scientific prediction. Falsifying one is a RESULT, not a failed check.

    Keeping this separate from ``check`` matters: a report that ends
    "FAIL (3 failed)" because three predictions were falsified would read as a
    broken construction, which is the opposite of what happened.
    """
    REPORT.append(f"  {'PASS' if ok else 'FALSIFIED'}  {message}")
    return bool(ok)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blocks(bundle: dict):
    return [
        (r, route, r["routes"][route])
        for r in bundle["records"]
        for route in PRIMARY
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=PROJECT / "results" / "FP-ITER6-001" / "claude" / "numpy",
    )
    parser.add_argument("--census-dir", type=Path, default=CENSUS_DIR)
    args = parser.parse_args()

    bundle_path = args.bundle_dir / "task_results.json"
    prediction_path = args.census_dir / "prediction_step6.json"
    bundle = load(bundle_path)
    sealed5 = load(ITER5_NUMPY)
    prediction = load(prediction_path)

    REPORT.append(f"{TASK_ID} analysis: the sixth certified step")
    REPORT.append("=" * 74)

    # --------------------------------------------------- 0. frozen prediction
    REPORT.append("\n0. The prediction was frozen before the run")
    bundle_mtime = bundle_path.stat().st_mtime
    prediction_mtime = prediction_path.stat().st_mtime
    check(
        prediction_mtime < bundle_mtime,
        f"prediction predates the sixth-step bundle "
        f"({datetime.fromtimestamp(prediction_mtime, timezone.utc).isoformat()} < "
        f"{datetime.fromtimestamp(bundle_mtime, timezone.utc).isoformat()})",
    )
    REPORT.append(f"  prediction sha256 = {sha256(prediction_path)}")
    REPORT.append(f"  prediction created_utc = {prediction['created_utc']}")
    theta = float(prediction["theta"])
    REPORT.append(
        f"  theta = {theta:.6f} (step-1 scan: "
        f"{prediction['theta_step1_misclassifications']}/"
        f"{prediction['theta_step1_population']} misclassified)"
    )

    # ------------------------------------------------------------ 1. inputs
    REPORT.append("\n1. Frozen inputs")
    check(
        len(bundle["records"]) == 24,
        f"24 records (found {len(bundle['records'])})",
    )
    check(
        int(bundle["max_steps"]) == LEVELS,
        f"MAX_STEPS {LEVELS} (found {bundle['max_steps']})",
    )
    check(
        int(sealed5["max_steps"]) == 5,
        f"the reference bundle is the five-step one (found {sealed5['max_steps']})",
    )

    steps = [
        (r, route, block, s)
        for r, route, block in blocks(bundle)
        for s in block["steps"]
    ]

    # ------------------------------------------------------- 2. H1 inertness
    REPORT.append("\n2. H1: the horizon change is inert")
    sealed5_by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed5["records"]
    }
    compared = 0
    expected = 0
    mismatches = []
    for record, route, block in blocks(bundle):
        reference = sealed5_by_key[(float(record["mixing"]), int(record["task_index"]))]
        reference_steps = reference["routes"][route]["steps"]
        # Every row the sealed five-step bundle contains must be reproduced. Note
        # this is NOT 105: that figure is levels 1--4 (48+22+20+15). Levels 1--5
        # hold 117 rows (48+22+20+15+12).
        expected += len(reference_steps)
        for index in range(min(len(reference_steps), len(block["steps"]))):
            compared += 1
            ours, theirs = block["steps"][index], reference_steps[index]
            if (
                ours["update_emitted"] != theirs["update_emitted"]
                or ours["eta_selected"] != theirs["eta_selected"]
                or ours["e_q"] != theirs["e_q"]
                or ours["ordered_reasons"] != theirs["ordered_reasons"]
            ):
                mismatches.append(
                    (record["mixing"], record["task_index"], route, index + 1)
                )
    check(
        compared == expected,
        f"every sealed row was compared ({compared} == {expected})",
    )
    check(
        compared == 117,
        f"117 sealed route-step rows across levels 1--5 (found {compared})",
    )
    check(
        not mismatches,
        f"H1 {'PASS' if not mismatches else 'FAIL'}: {compared} step entries "
        f"1--5 compared, {len(mismatches)} mismatches",
    )
    for item in mismatches[:10]:
        REPORT.append(f"        mismatch: {item}")

    # -------------------------------------------------- 3. per-level outcome
    def per_level(source_steps) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        for level in range(1, LEVELS + 1):
            gains = [
                (r["mixing"], r["task_index"], route, s["oracle_audit"]["total_value_gain"])
                for r, route, _, s in source_steps
                if s["step"] == level and s["update_emitted"]
            ]
            values = [g[3] for g in gains]
            out[level] = {
                "emissions": len(values),
                "mean": (sum(values) / len(values)) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "emitters": [(g[0], g[1], g[2]) for g in gains],
            }
        return out

    level = per_level(steps)
    REPORT.append("\n3. Per-level outcome")
    REPORT.append("  step  emissions   mean gain   minimum gain")
    for k in range(1, LEVELS + 1):
        info = level[k]
        mean = "n/a" if info["mean"] is None else f"{info['mean']:.6f}"
        low = "n/a" if info["min"] is None else f"{info['min']:.6f}"
        REPORT.append(f"  {k:>4}  {info['emissions']:>9}   {mean:>9}   {low:>12}")
    counts = [level[k]["emissions"] for k in range(1, LEVELS + 1)]
    deltas = [counts[i + 1] - counts[i] for i in range(len(counts) - 1)]
    REPORT.append(f"  emissions by step: {counts}")
    REPORT.append(f"  deltas           : {deltas}")

    # -------------------------------------------------------- 4. H2/H3/H4
    REPORT.append("\n4. H2/H3/H4: is the sixth step certifiable and valid?")
    h2 = level[6]["emissions"] >= 1
    check(h2, f"H2 {'PASS' if h2 else 'FALSIFIED'}: {level[6]['emissions']} "
              f"route-records emit a sixth step")

    six = [s for _, _, _, s in steps if s["step"] == 6 and s["update_emitted"]]
    degrading = [
        s for s in six
        if not s["oracle_audit"].get("componentwise_nondegrading", False)
    ]
    nonpositive = [
        s for s in six if s["oracle_audit"]["total_value_gain"] <= 0.0
    ]
    h3 = bool(six) and not degrading and not nonpositive
    check(h3, f"H3 {'PASS' if h3 else 'FALSIFIED'}: all {len(six)} emitted sixth "
              f"steps non-degrading and strictly improving "
              f"({len(degrading)} degrading, {len(nonpositive)} non-positive)")

    violations = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if s["oracle_audit"].get("certificate_violation")
    ]
    check(not violations, f"H4 PASS: zero certificate violations across all six "
                          f"steps ({len(violations)})")
    for item in violations[:10]:
        REPORT.append(f"        violation: {item}")

    missing_reasons = [
        (r["mixing"], r["task_index"], route, s["step"])
        for r, route, _, s in steps
        if not s["update_emitted"] and not s["ordered_reasons"]
    ]
    check(not missing_reasons, f"every non-emitting step carries its frozen "
                               f"ordered reason ({len(missing_reasons)} missing)")

    # ---------------------------------------------------- 5. score the rules
    REPORT.append("\n5. H5/H6: scoring the two pre-registered rules")
    # The step-5 RATIO is not a field of the sealed FP-ITER5-001 bundle; it was
    # measured by FP-CENSUS-001. The prediction file is the frozen bridge, and its
    # own rows are the authoritative source for the feature P2 uses.
    # Key by the exact three-tuple, then compare against the census bundle.
    predicted_ratio = {
        (float(r["mixing"]), int(r["task_index"]), r["route"]): float(
            r["ratio_step5"]
        )
        for r in prediction["predictions"]["P2_step1_ratio_threshold"]["rows"]
    }
    census = load(args.census_dir / "task_results.json")
    census_ratio = {}
    for record in census["records"]:
        for route in PRIMARY:
            block = record["routes"][route]
            if len(block["steps"]) >= 5:
                census_ratio[
                    (float(record["mixing"]), int(record["task_index"]), route)
                ] = float(block["steps"][4]["ratio"])
    check(
        len(predicted_ratio) == len(census_ratio)
        and all(
            abs(predicted_ratio[k] - census_ratio[k]) <= 1e-12
            for k in predicted_ratio
            if k in census_ratio
        ),
        f"the prediction's step-5 ratios reproduce the census bundle exactly "
        f"({len(predicted_ratio)} rows)",
    )

    population = []
    for record, route, block in blocks(sealed5):
        if len(block["steps"]) >= 5:
            key3 = (float(record["mixing"]), int(record["task_index"]), route)
            population.append(
                {
                    "key": (float(record["mixing"]), int(record["task_index"])),
                    "route": route,
                    "emitted_at_step5": bool(block["steps"][4]["update_emitted"]),
                    "ratio_step5": predicted_ratio[key3],
                }
            )
    expected_population = int(prediction["scoring_population_size"])
    check(
        len(population) == expected_population,
        f"scoring population matches the frozen prediction "
        f"({len(population)} == {expected_population})",
    )

    # Observed sixth-step outcome for each member of the population.
    six_by_key: dict[tuple[float, int, str], bool] = {}
    reached_six = set()
    for record, route, block in blocks(bundle):
        key = (float(record["mixing"]), int(record["task_index"]), route)
        if len(block["steps"]) >= 6:
            reached_six.add(key)
            six_by_key[key] = bool(block["steps"][5]["update_emitted"])
        else:
            six_by_key[key] = False

    population_keys = {(p["key"][0], p["key"][1], p["route"]) for p in population}
    not_reached = sorted(population_keys - reached_six)
    check(
        not not_reached,
        f"every member of the scoring population reached step 6 "
        f"({len(not_reached)} did not: {not_reached[:5]})",
    )

    # P1: the plateau carries on.
    p1_errors = []
    for p in population:
        key = (p["key"][0], p["key"][1], p["route"])
        predicted = p["emitted_at_step5"]
        observed = six_by_key.get(key, False)
        if predicted != observed:
            p1_errors.append(
                {
                    "mixing": p["key"][0],
                    "task_index": p["key"][1],
                    "route": p["route"],
                    "predicted": predicted,
                    "observed": observed,
                    "ratio_step5": p["ratio_step5"],
                }
            )
    p1_predicted_n = sum(1 for p in population if p["emitted_at_step5"])

    # P2: the frozen step-1 threshold applied to the step-5 ratio.
    p2_errors = []
    for p in population:
        key = (p["key"][0], p["key"][1], p["route"])
        predicted = bool(p["ratio_step5"] > theta)
        observed = six_by_key.get(key, False)
        if predicted != observed:
            p2_errors.append(
                {
                    "mixing": p["key"][0],
                    "task_index": p["key"][1],
                    "route": p["route"],
                    "predicted": predicted,
                    "observed": observed,
                    "ratio_step5": p["ratio_step5"],
                }
            )
    p2_predicted_n = sum(
        1 for p in population if p["ratio_step5"] > theta
    )

    # POST-HOC, NOT PRE-REGISTERED. The two rules above are the registered test
    # and are scored as such. This extra scan exists to answer the obvious
    # follow-up -- "could ANY threshold on the step-5 ratio have caught the
    # drop-outs?" -- and it is labelled everywhere as post-hoc so that it can
    # never be mistaken for the pre-registered rule. Per acceptance criterion 5
    # it is not described as a classifier if it misclassifies.
    posthoc_best: dict[str, Any] = {}
    if population:
        ordered = sorted({p["ratio_step5"] for p in population})
        candidates = [ordered[0] - 1.0]
        for a, b in zip(ordered, ordered[1:]):
            candidates.append((a + b) / 2.0)
        candidates.append(ordered[-1] + 1.0)
        scored = []
        for t in candidates:
            errors = sum(
                1
                for p in population
                if (p["ratio_step5"] > t) != six_by_key.get(
                    (p["key"][0], p["key"][1], p["route"]), False
                )
            )
            scored.append((errors, t))
        fewest = min(e for e, _ in scored)
        attaining = sorted(t for e, t in scored if e == fewest)
        posthoc_best = {
            "candidates": len(candidates),
            "population": len(population),
            "best_misclassifications": fewest,
            "best_threshold_interval": [attaining[0], attaining[-1]],
            "separates_exactly": fewest == 0,
            "min_ratio_of_survivors": min(
                p["ratio_step5"]
                for p in population
                if six_by_key.get((p["key"][0], p["key"][1], p["route"]), False)
            )
            if any(
                six_by_key.get((p["key"][0], p["key"][1], p["route"]), False)
                for p in population
            )
            else None,
            "ratio_range_of_dropouts": sorted(
                p["ratio_step5"]
                for p in population
                if not six_by_key.get((p["key"][0], p["key"][1], p["route"]), False)
            ),
        }

    REPORT.append(f"  population (reached step 5): {len(population)}")
    REPORT.append(
        f"  observed sixth-step emitters among them: "
        f"{sum(1 for k in population_keys if six_by_key.get(k, False))}"
    )
    REPORT.append(
        f"  P1 plateau rule : predicts {p1_predicted_n} emitters, "
        f"{len(p1_errors)} misclassified"
    )
    REPORT.append(
        f"  P2 ratio rule   : predicts {p2_predicted_n} emitters, "
        f"{len(p2_errors)} misclassified"
    )
    if posthoc_best:
        REPORT.append(
            "  POST-HOC, NOT PRE-REGISTERED: the best threshold on the step-5 "
            f"ratio still misclassifies {posthoc_best['best_misclassifications']}"
            f"/{posthoc_best['population']}"
        )
        REPORT.append(
            f"        lowest step-5 ratio among the SURVIVORS: "
            f"{posthoc_best['min_ratio_of_survivors']:.4f}; "
            f"step-5 ratios of the DROP-OUTS: "
            f"{[round(v, 4) for v in posthoc_best['ratio_range_of_dropouts']]}"
        )
        REPORT.append(
            "        this scan is a follow-up question, not the registered test, "
            "and is not used to revise H5 or H6."
        )
    for item in p1_errors:
        REPORT.append(
            f"        P1 error: mix={item['mixing']} task={item['task_index']} "
            f"{item['route']} predicted={item['predicted']} "
            f"observed={item['observed']} ratio5={item['ratio_step5']:.4f}"
        )
    for item in p2_errors:
        REPORT.append(
            f"        P2 error: mix={item['mixing']} task={item['task_index']} "
            f"{item['route']} predicted={item['predicted']} "
            f"observed={item['observed']} ratio5={item['ratio_step5']:.4f}"
        )

    h5 = len(p1_errors) == 0
    hypothesis(
        h5,
        f"H5 {'PASS' if h5 else 'FALSIFIED'}: P1 (the plateau carries on) "
        f"misclassifies {len(p1_errors)} of {len(population)}",
    )
    h6 = len(p2_errors) < len(p1_errors)
    hypothesis(
        h6,
        f"H6 {'PASS' if h6 else 'FALSIFIED'}: P2 "
        f"({len(p2_errors)} misclassified) is "
        f"{'strictly better than' if h6 else 'NOT strictly better than'} P1 "
        f"({len(p1_errors)} misclassified)",
    )

    # --------------------------------------------------------- 6. H7/H7b/H8/H9
    REPORT.append("\n6. H7/H7b/H8/H9: attrition, decay, non-vacuity")
    n6 = level[6]["emissions"]
    n5 = level[5]["emissions"]
    set5 = set(level[5]["emitters"])
    set6 = set(level[6]["emitters"])
    gained = sorted(set6 - set5)
    lost = sorted(set5 - set6)
    retained = sorted(set5 & set6)
    REPORT.append(
        f"  step-5 emitters {n5}, step-6 emitters {n6}; "
        f"retained {len(retained)}, lost {len(lost)}, gained {len(gained)}"
    )
    h7 = n6 < n5
    h7b = n6 == n5 and not gained and not lost
    check(not h7 or not h7b, "H7 and H7b are mutually exclusive as registered")
    REPORT.append(f"  H7  (n6 < n5)                : {'PASS' if h7 else 'FALSIFIED'}")
    REPORT.append(
        f"  H7b (n6 == n5, same set)     : {'PASS' if h7b else 'FALSIFIED'}"
    )

    mean6, mean5 = level[6]["mean"], level[5]["mean"]
    if mean6 is not None and mean5 is not None:
        h8 = mean6 < mean5
        hypothesis(h8, f"H8 {'PASS' if h8 else 'FALSIFIED'}: mean6={mean6:.6f} "
                        f"{'<' if h8 else '>='} mean5={mean5:.6f}")
    else:
        h8 = None
        REPORT.append("  H8 not evaluable: no sixth-step emission")

    min6 = level[6]["min"]
    if min6 is not None:
        h9 = min6 > 0.0 and min6 >= 0.05
        hypothesis(h9, f"H9 {'PASS' if h9 else 'FALSIFIED'}: min6={min6:.6f} "
                        f"(requires > 0 and >= 0.05)")
    else:
        h9 = None
        REPORT.append("  H9 not evaluable: no sixth-step emission")

    # -------------------------------------------------------- 7. H10 drift
    REPORT.append("\n7. H10: emitting-set membership, stated as a set difference")
    for item in lost:
        REPORT.append(f"        lost  : mix={item[0]} task={item[1]} {item[2]}")
    for item in gained:
        REPORT.append(f"        gained: mix={item[0]} task={item[1]} {item[2]}")
    if not lost and not gained:
        REPORT.append("        no membership change at all")

    REPORT.append("\n" + "=" * 74)
    REPORT.append("SUMMARY")
    REPORT.append(f"  H1  horizon inert           : "
                  f"{'PASS' if compared == expected and not mismatches else 'FAIL'} "
                  f"({compared}/{expected} rows)")
    REPORT.append(f"  H2  sixth step certifiable  : {'PASS' if h2 else 'FALSIFIED'}")
    REPORT.append(f"  H3  sixth step valid        : {'PASS' if h3 else 'FALSIFIED'}")
    REPORT.append(f"  H4  no certificate violation: {'PASS' if not violations else 'FAIL'}")
    REPORT.append(f"  H5  P1 plateau rule         : {'PASS' if h5 else 'FALSIFIED'} "
                  f"({len(p1_errors)} misclassified)")
    REPORT.append(f"  H6  P2 beats P1             : {'PASS' if h6 else 'FALSIFIED'} "
                  f"({len(p2_errors)} vs {len(p1_errors)})")
    REPORT.append(f"  H7  attrition (n6 < n5)     : {'PASS' if h7 else 'FALSIFIED'}")
    REPORT.append(f"  H7b plateau (same set)      : {'PASS' if h7b else 'FALSIFIED'}")
    REPORT.append(
        f"  H8  mean gain decays        : "
        f"{'PASS' if h8 else 'FALSIFIED'}" if h8 is not None else "  H8  n/a"
    )
    REPORT.append(
        f"  H9  non-vacuity floor 0.05  : "
        f"{'PASS' if h9 else 'FALSIFIED'}" if h9 is not None else "  H9  n/a"
    )
    REPORT.append("=" * 74)
    REPORT.append(
        "hypotheses were falsified, which per the task sheet is a RESULT, not a\n"
        "failed construction. RESULT below covers the construction checks only."
    )
    REPORT.append(
        "CONSTRUCTION: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)")
    )

    summary = {
        "task_id": TASK_ID,
        "prediction_sha256": sha256(prediction_path),
        "prediction_created_utc": prediction["created_utc"],
        "theta": theta,
        "emissions_by_step": counts,
        "emission_deltas": deltas,
        "mean_gain_by_step": [
            level[k]["mean"] for k in range(1, LEVELS + 1)
        ],
        "min_gain_by_step": [level[k]["min"] for k in range(1, LEVELS + 1)],
        "H1_horizon_inert": {
            "compared": compared,
            "expected": expected,
            "mismatches": len(mismatches),
        },
        "H2": "PASS" if h2 else "FALSIFIED",
        "H3": "PASS" if h3 else "FALSIFIED",
        "H4": "PASS" if not violations else "FAIL",
        "H5_prediction_P1": {
            "verdict": "PASS" if h5 else "FALSIFIED",
            "rule": prediction["predictions"]["P1_plateau_carries_on"]["rule"],
            "predicted_emitters": p1_predicted_n,
            "misclassifications": len(p1_errors),
            "errors": p1_errors,
        },
        "H6_prediction_P2": {
            "verdict": "PASS" if h6 else "FALSIFIED",
            "rule": prediction["predictions"]["P2_step1_ratio_threshold"]["rule"],
            "predicted_emitters": p2_predicted_n,
            "misclassifications": len(p2_errors),
            "errors": p2_errors,
        },
        "scoring_population": len(population),
        "posthoc_step5_ratio_scan": posthoc_best,
        "H7_attrition": "PASS" if h7 else "FALSIFIED",
        "H7b_plateau": "PASS" if h7b else "FALSIFIED",
        "H8_mean_decay": None if h8 is None else ("PASS" if h8 else "FALSIFIED"),
        "H9_non_vacuity": None if h9 is None else ("PASS" if h9 else "FALSIFIED"),
        "H10_set_change": {
            "retained": len(retained),
            "lost": [list(x) for x in lost],
            "gained": [list(x) for x in gained],
        },
    }
    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    (args.bundle_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.bundle_dir / "analysis_report.md").write_text(
        "\n".join(REPORT) + "\n", encoding="utf-8"
    )
    print("\n".join(REPORT))


if __name__ == "__main__":
    main()
