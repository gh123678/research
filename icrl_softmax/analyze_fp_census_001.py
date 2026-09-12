"""FP-CENSUS-001 analysis: does sigma_min / E_Q explain who emits?

Reads the formal census bundle and the sealed FP-ITER5-001 numpy bundle, evaluates
H0--H6, reports group distributions rather than only means, scans every threshold
for its misclassification count, and writes the frozen step-6 prediction.

Nothing here recomputes a certificate or a decision: the census bundle is the
input, and the sealed bundle is the only source of group labels, so the labels do
not depend on this task's own decisions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parent
TASK_ID = "FP-CENSUS-001"
PRIMARY = ("expected_exact", "expected_finite")
SEALED_ITER5 = (
    PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy" / "task_results.json"
)
REPORT: list[str] = []
FAILURES = 0


def check(ok: bool, message: str) -> bool:
    """A construction check. Failing one invalidates the census."""
    global FAILURES
    if ok:
        REPORT.append(f"  PASS  {message}")
    else:
        FAILURES += 1
        REPORT.append(f"  FAIL  {message}")
    return bool(ok)


def hypothesis(ok: bool, message: str) -> bool:
    """A scientific hypothesis. Falsifying one is a RESULT, not a failed check."""
    REPORT.append(f"  {'PASS' if ok else 'FALSIFIED'}  {message}")
    return bool(ok)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)

    def quantile(q: float) -> float:
        if n == 1:
            return ordered[0]
        position = q * (n - 1)
        low = int(position)
        high = min(low + 1, n - 1)
        frac = position - low
        return ordered[low] * (1.0 - frac) + ordered[high] * frac

    return {
        "n": n,
        "min": ordered[0],
        "q1": quantile(0.25),
        "median": quantile(0.5),
        "q3": quantile(0.75),
        "max": ordered[-1],
        "mean": sum(ordered) / n,
    }


def threshold_scan(
    ratios: list[float], labels: list[bool]
) -> dict[str, Any]:
    """Every candidate threshold and its misclassification count.

    The rule under test is "emit iff ratio > t". Candidate thresholds are the
    midpoints between consecutive distinct ratios plus one below the minimum and
    one above the maximum, so the scan covers every rule this feature can express.

    A scan with only one class present is reported as ``vacuous``: it cannot
    separate anything, and its zero (or maximal) count is not evidence.
    """
    if not ratios:
        return {"candidates": 0, "best": None, "vacuous": True}
    n_emitters = sum(1 for lab in labels if lab)
    n_abstainers = len(labels) - n_emitters
    ordered = sorted(set(ratios))
    candidates: list[float] = [ordered[0] - 1.0]
    for a, b in zip(ordered, ordered[1:]):
        candidates.append((a + b) / 2.0)
    candidates.append(ordered[-1] + 1.0)

    scored: list[tuple[int, float]] = []
    for t in candidates:
        errors = sum(
            1 for r, lab in zip(ratios, labels) if (r > t) != bool(lab)
        )
        scored.append((errors, t))
    fewest = min(e for e, _ in scored)
    attaining = sorted(t for e, t in scored if e == fewest)
    chosen = attaining[0]
    misclassified = [
        {"ratio": r, "emitted": bool(lab)}
        for r, lab in zip(ratios, labels)
        if (r > chosen) != bool(lab)
    ]
    return {
        "candidates": len(candidates),
        "population": len(ratios),
        "emitters": n_emitters,
        "abstainers": n_abstainers,
        "vacuous": n_emitters == 0 or n_abstainers == 0,
        "best_misclassifications": fewest,
        "best_thresholds": attaining[:8],
        "best_threshold_count": len(attaining),
        "best_threshold_interval": [attaining[0], attaining[-1]],
        "chosen_threshold": chosen,
        "chosen_misclassified": misclassified,
        "separates_exactly": fewest == 0 and n_emitters > 0 and n_abstainers > 0,
        "score_distribution": distribution([float(e) for e, _ in scored]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--census-dir",
        type=Path,
        default=PROJECT / "results" / "FP-CENSUS-001" / "claude" / "formal",
    )
    parser.add_argument(
        "--write-prediction",
        action="store_true",
        help="write prediction_step6.json; only for the formal run",
    )
    args = parser.parse_args()

    census = load(args.census_dir / "task_results.json")
    sealed = load(SEALED_ITER5)
    sealed_by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
    }

    # ---------------------------------------------------------------- rows
    rows: list[dict[str, Any]] = []
    for record in census["records"]:
        key = (float(record["mixing"]), int(record["task_index"]))
        reference = sealed_by_key[key]
        for route in PRIMARY:
            block = record["routes"][route]
            sealed_block = reference["routes"][route]
            rows.append(
                {
                    "key": key,
                    "mixing": key[0],
                    "task_index": key[1],
                    "route": route,
                    "steps": block["steps"],
                    "sealed_emitted_steps": int(sealed_block["emitted_steps"]),
                    "repro": block["sealed_reproduction"],
                }
            )

    REPORT.append(f"{TASK_ID} analysis")
    REPORT.append("=" * 74)

    # ------------------------------------------------------- 1. H0 restated
    REPORT.append("\n1. H0: the census reproduces the sealed classification")
    total_rows = len(rows)
    exact = [r for r in rows if r["repro"].get("exact")]
    bad = [r for r in rows if not r["repro"].get("exact")]
    for field in (
        "decision_mismatches",
        "reason_mismatches",
        "eta_mismatches",
        "e_q_mismatches",
    ):
        count = sum(int(r["repro"].get(field, 0)) for r in rows)
        check(count == 0, f"total {field} = {count}")
    row_count_ok = all(
        r["repro"]["census_step_count"] == r["repro"]["sealed_step_count"]
        for r in rows
    )
    check(row_count_ok, "census and sealed row counts agree on every route")
    check(
        len(exact) == total_rows,
        f"H0 exact on {len(exact)}/{total_rows} route-records",
    )
    for r in bad:
        REPORT.append(
            f"        mismatch: mixing={r['mixing']} task={r['task_index']} "
            f"route={r['route']} {r['repro']}"
        )

    # ------------------------------------------------------------ 2. groups
    def group_of(sealed_steps: int) -> str:
        if sealed_steps == 0:
            return "never"
        if sealed_steps >= 5:
            return "plateau"
        return "dropout"

    for r in rows:
        r["group"] = group_of(r["sealed_emitted_steps"])
        r["step1_ratio"] = r["steps"][0]["ratio"]
        r["step1_e_q"] = r["steps"][0]["e_q"]
        r["step1_sigma"] = r["steps"][0]["sigma_min"]

    REPORT.append("\n2. The three groups (labels from the sealed bundle)")
    groups = ("never", "dropout", "plateau")
    for name in groups:
        members = [r for r in rows if r["group"] == name]
        dist = distribution([r["step1_ratio"] for r in members])
        REPORT.append(
            f"  {name:<8} n={len(members):>2}  step-1 ratio  "
            f"min={dist.get('min', float('nan')):.4f} "
            f"q1={dist.get('q1', float('nan')):.4f} "
            f"med={dist.get('median', float('nan')):.4f} "
            f"q3={dist.get('q3', float('nan')):.4f} "
            f"max={dist.get('max', float('nan')):.4f} "
            f"mean={dist.get('mean', float('nan')):.4f}"
        )
    by_record: dict[tuple[float, int], list[str]] = {}
    for r in rows:
        by_record.setdefault(r["key"], []).append(r["group"])
    mixed = {k: v for k, v in by_record.items() if len(set(v)) > 1}
    REPORT.append(
        f"  route-level grouping over {len(rows)} route-records; "
        f"{len(mixed)} of {len(by_record)} records have routes in different groups"
    )

    # ---------------------------------------------------------------- H1
    REPORT.append("\n3. H1: step-1 separation")
    emitters = [r for r in rows if r["sealed_emitted_steps"] >= 1]
    abstainers = [r for r in rows if r["sealed_emitted_steps"] == 0]
    emit_dist = distribution([r["step1_ratio"] for r in emitters])
    abst_dist = distribution([r["step1_ratio"] for r in abstainers])
    REPORT.append(
        f"  emitted   n={emit_dist['n']:>2} mean={emit_dist['mean']:.4f} "
        f"range=[{emit_dist['min']:.4f}, {emit_dist['max']:.4f}]"
    )
    REPORT.append(
        f"  abstained n={abst_dist['n']:>2} mean={abst_dist['mean']:.4f} "
        f"range=[{abst_dist['min']:.4f}, {abst_dist['max']:.4f}]"
    )
    h1 = emit_dist["mean"] > abst_dist["mean"]
    hypothesis(h1, f"H1 {'PASS' if h1 else 'FALSIFIED'}: emitter mean ratio is "
              f"{'strictly higher' if h1 else 'NOT higher'}")

    # ---------------------------------------------------------------- H2
    REPORT.append("\n4. H2: is the ratio a classifier?")
    labels = [r["sealed_emitted_steps"] >= 1 for r in rows]
    ratios = [r["step1_ratio"] for r in rows]
    scan = threshold_scan(ratios, labels)
    REPORT.append(
        f"  step-1 scan: {scan['candidates']} candidate thresholds over "
        f"{scan['population']} records, {scan['emitters']} emitters"
    )
    REPORT.append(
        f"  fewest misclassifications = {scan['best_misclassifications']} "
        f"({scan['best_misclassifications']} of {scan['population']} = "
        f"{100.0 * scan['best_misclassifications'] / scan['population']:.1f}%)"
    )
    REPORT.append(
        f"  attained on {scan['best_threshold_count']} thresholds spanning "
        f"[{scan['best_threshold_interval'][0]:.4f}, "
        f"{scan['best_threshold_interval'][1]:.4f}]"
    )
    h2 = scan["separates_exactly"]
    hypothesis(
        h2,
        f"H2 {'PASS: a zero-misclassification threshold exists' if h2 else 'FALSIFIED: no threshold separates the groups'}",
    )
    REPORT.append(
        "  NOTE  per acceptance criterion 5 the ratio is NOT described as a "
        "classifier: the best attainable rule still misclassifies records."
        if not h2
        else "  NOTE  H2 attained a zero-misclassification threshold."
    )
    REPORT.append(
        f"  the chosen threshold {scan['chosen_threshold']:.4f} misclassifies "
        f"{len(scan['chosen_misclassified'])} record(s): "
        f"{scan['chosen_misclassified']}"
    )
    # Where the two groups overlap, stated as an interval.
    if emitters and abstainers:
        emit_min = min(r["step1_ratio"] for r in emitters)
        abst_max = max(r["step1_ratio"] for r in abstainers)
        REPORT.append(
            f"  overlap: lowest emitter ratio {emit_min:.4f}, highest abstainer "
            f"ratio {abst_max:.4f} -> the groups are "
            f"{'interleaved' if abst_max > emit_min else 'disjoint'}"
        )

    # ---------------------------------------------------- reasons by group
    REPORT.append("\n5b. The frozen ordered abstention reasons, by group")
    reason_by_group: dict[str, dict[str, int]] = {name: {} for name in groups}
    reason_by_step: dict[str, dict[str, int]] = {}
    for r in rows:
        for step in r["steps"]:
            if step["update_emitted"]:
                continue
            key = " | ".join(step["ordered_reasons"])
            reason_by_group[r["group"]][key] = (
                reason_by_group[r["group"]].get(key, 0) + 1
            )
            level_key = str(step["step"])
            reason_by_step.setdefault(level_key, {})
            reason_by_step[level_key][key] = (
                reason_by_step[level_key].get(key, 0) + 1
            )
    for name in groups:
        total = sum(reason_by_group[name].values())
        REPORT.append(f"  {name} ({total} abstaining rows):")
        for key, count in sorted(
            reason_by_group[name].items(), key=lambda kv: -kv[1]
        ):
            REPORT.append(f"        {count:>3}  {key}")
    REPORT.append("  by step level:")
    for level_key in sorted(reason_by_step, key=int):
        total = sum(reason_by_step[level_key].values())
        REPORT.append(f"    step {level_key} ({total} abstaining rows):")
        for key, count in sorted(
            reason_by_step[level_key].items(), key=lambda kv: -kv[1]
        ):
            REPORT.append(f"        {count:>3}  {key}")

    # ---------------------------------------------------------------- H3
    REPORT.append("\n6. H3: plateau vs drop-out")
    plateau = [r for r in rows if r["group"] == "plateau"]
    dropout = [r for r in rows if r["group"] == "dropout"]
    p_dist = distribution([r["step1_ratio"] for r in plateau])
    d_dist = distribution([r["step1_ratio"] for r in dropout])
    REPORT.append(
        f"  plateau n={p_dist['n']:>2} mean={p_dist['mean']:.4f} "
        f"range=[{p_dist['min']:.4f}, {p_dist['max']:.4f}]"
    )
    REPORT.append(
        f"  dropout n={d_dist['n']:>2} mean={d_dist['mean']:.4f} "
        f"range=[{d_dist['min']:.4f}, {d_dist['max']:.4f}]"
    )
    h3 = p_dist["mean"] > d_dist["mean"]
    hypothesis(h3, f"H3 {'PASS' if h3 else 'FALSIFIED'}: plateau mean step-1 ratio is "
              f"{'strictly higher' if h3 else 'NOT higher'} than drop-out")
    plateau_scan = threshold_scan(
        [r["step1_ratio"] for r in plateau + dropout],
        [r["group"] == "plateau" for r in plateau + dropout],
    )
    REPORT.append(
        f"  plateau/dropout scan: fewest misclassifications = "
        f"{plateau_scan['best_misclassifications']} of {plateau_scan['population']}"
        f" ({100.0 * plateau_scan['best_misclassifications'] / plateau_scan['population']:.1f}%)"
    )

    # ---------------------------------------------------------------- H4
    REPORT.append("\n7. H4: does iteration consume the margin?")
    decay_rows = []
    for r in rows:
        emitted_steps = [s for s in r["steps"] if s["update_emitted"]]
        if len(emitted_steps) < 2:
            continue
        first = r["steps"][0]["ratio"]
        last = emitted_steps[-1]["ratio"]
        decay_rows.append(
            {
                "key": r["key"],
                "route": r["route"],
                "group": r["group"],
                "steps": len(emitted_steps),
                "first": first,
                "last": last,
                "change": last - first,
                "fell": last < first,
            }
        )
    fell = [d for d in decay_rows if d["fell"]]
    REPORT.append(
        f"  trajectories with >= 2 emitted steps: {len(decay_rows)}; "
        f"ratio fell in {len(fell)} ({100.0 * len(fell) / max(len(decay_rows), 1):.1f}%)"
    )
    if decay_rows:
        changes = distribution([d["change"] for d in decay_rows])
        REPORT.append(
            f"  change (last emitted - step 1): mean={changes['mean']:.4f} "
            f"median={changes['median']:.4f} "
            f"range=[{changes['min']:.4f}, {changes['max']:.4f}]"
        )
        for d in sorted(decay_rows, key=lambda x: x["change"])[:5]:
            REPORT.append(
                f"        fell most: mix={d['key'][0]} task={d['key'][1]} "
                f"{d['route']} steps={d['steps']} "
                f"{d['first']:.4f} -> {d['last']:.4f}"
            )
        for d in sorted(decay_rows, key=lambda x: -x["change"])[:3]:
            if d["change"] > 0:
                REPORT.append(
                    f"        rose: mix={d['key'][0]} task={d['key'][1]} "
                    f"{d['route']} steps={d['steps']} "
                    f"{d['first']:.4f} -> {d['last']:.4f}"
                )
    h4 = bool(decay_rows) and len(fell) * 2 > len(decay_rows)
    hypothesis(h4, f"H4 {'PASS' if h4 else 'FALSIFIED'}: ratio is lower at the last "
              f"emitted step in a majority of trajectories")
    # The task sheet phrases H4 universally ("along each trajectory"), while the
    # frozen plan operationalises it as a majority claim. Both readings are
    # reported; the divergence is a defect in the task record and is not papered
    # over by quoting only the favourable one.
    h4_universal = bool(decay_rows) and len(fell) == len(decay_rows)
    REPORT.append(
        f"  reading as literally written (EVERY trajectory): "
        f"{'PASS' if h4_universal else 'FALSIFIED'} "
        f"({len(decay_rows) - len(fell)} of {len(decay_rows)} did not fall)"
    )
    REPORT.append(
        "  reading as operationalised in the frozen plan (a MAJORITY): "
        f"{'PASS' if h4 else 'FALSIFIED'} ({len(fell)}/{len(decay_rows)})"
    )

    # ---------------------------------------------------------------- H5
    REPORT.append("\n8. H5: which term binds for the abstainers?")
    e_q_abst = distribution([r["step1_e_q"] for r in abstainers])
    e_q_emit = distribution([r["step1_e_q"] for r in emitters])
    sig_abst = distribution([r["step1_sigma"] for r in abstainers])
    sig_emit = distribution([r["step1_sigma"] for r in emitters])
    REPORT.append(
        f"  E_Q       abstained mean={e_q_abst['mean']:.4f} "
        f"range=[{e_q_abst['min']:.4f}, {e_q_abst['max']:.4f}]"
    )
    REPORT.append(
        f"  E_Q       emitted   mean={e_q_emit['mean']:.4f} "
        f"range=[{e_q_emit['min']:.4f}, {e_q_emit['max']:.4f}]"
    )
    REPORT.append(
        f"  sigma_min abstained mean={sig_abst['mean']:.4f} "
        f"range=[{sig_abst['min']:.4f}, {sig_abst['max']:.4f}]"
    )
    REPORT.append(
        f"  sigma_min emitted   mean={sig_emit['mean']:.4f} "
        f"range=[{sig_emit['min']:.4f}, {sig_emit['max']:.4f}]"
    )
    REPORT.append(
        f"  mean E_Q ratio abstained/emitted = "
        f"{e_q_abst['mean'] / e_q_emit['mean']:.4f};  "
        f"mean sigma_min ratio = {sig_abst['mean'] / sig_emit['mean']:.4f}"
    )
    h5 = e_q_abst["mean"] > e_q_emit["mean"]
    hypothesis(h5, f"H5 {'PASS' if h5 else 'FALSIFIED'}: abstainers' mean E_Q is "
              f"{'larger' if h5 else 'NOT larger'}")
    e_q_scan = threshold_scan(
        [r["step1_e_q"] for r in rows], [not lab for lab in labels]
    )
    REPORT.append(
        f"  E_Q alone, scan for predicting abstention: fewest "
        f"misclassifications = {e_q_scan['best_misclassifications']} of "
        f"{e_q_scan['population']}"
    )

    # ---------------------------------------------------------------- H6
    REPORT.append("\n9. H6: does the spread move with the policy?")
    cv_rows = []
    for r in rows:
        if len(r["steps"]) < 2:
            continue
        sigmas = [s["sigma_min"] for s in r["steps"]]
        e_qs = [float(s["e_q"]) for s in r["steps"]]
        sig_mean = sum(sigmas) / len(sigmas)
        e_q_mean = sum(e_qs) / len(e_qs)
        if sig_mean <= 0 or e_q_mean <= 0:
            continue
        sig_sd = (sum((x - sig_mean) ** 2 for x in sigmas) / len(sigmas)) ** 0.5
        e_q_sd = (sum((x - e_q_mean) ** 2 for x in e_qs) / len(e_qs)) ** 0.5
        cv_rows.append(
            {
                "key": r["key"],
                "route": r["route"],
                "group": r["group"],
                "steps": len(r["steps"]),
                "cv_sigma": sig_sd / sig_mean,
                "cv_e_q": e_q_sd / e_q_mean,
                "sigma_range": max(sigmas) - min(sigmas),
                "e_q_range": max(e_qs) - min(e_qs),
            }
        )
    if cv_rows:
        cv_sig = distribution([c["cv_sigma"] for c in cv_rows])
        cv_e_q = distribution([c["cv_e_q"] for c in cv_rows])
        REPORT.append(
            f"  over {len(cv_rows)} trajectories with >= 2 rows: "
            f"mean CV(sigma_min) = {cv_sig['mean']:.4f}, "
            f"mean CV(E_Q) = {cv_e_q['mean']:.4f}"
        )
        REPORT.append(
            f"  mean absolute range: sigma_min = "
            f"{sum(c['sigma_range'] for c in cv_rows) / len(cv_rows):.4f}, "
            f"E_Q = {sum(c['e_q_range'] for c in cv_rows) / len(cv_rows):.4f}"
        )
        h6 = cv_sig["mean"] < cv_e_q["mean"]
        hypothesis(h6, f"H6 {'PASS' if h6 else 'FALSIFIED'}: sigma_min varies "
                  f"{'less' if h6 else 'MORE'} than E_Q along the trajectory")
    else:
        h6 = False
        hypothesis(False, "H6 not evaluable: no trajectory has two rows")

    # ------------------------------------------------- ratio by step level
    REPORT.append("\n10. Ratio and separation by step level")
    level_info: dict[str, Any] = {}
    for level in range(1, 6):
        present = [r for r in rows if len(r["steps"]) >= level]
        if not present:
            continue
        level_ratios = [r["steps"][level - 1]["ratio"] for r in present]
        level_labels = [
            bool(r["steps"][level - 1]["update_emitted"]) for r in present
        ]
        lv_scan = threshold_scan(level_ratios, level_labels)
        e_dist = distribution(
            [x for x, lab in zip(level_ratios, level_labels) if lab]
        )
        a_dist = distribution(
            [x for x, lab in zip(level_ratios, level_labels) if not lab]
        )
        level_info[str(level)] = {
            "population": len(present),
            "emitters": sum(1 for lab in level_labels if lab),
            "abstainers": sum(1 for lab in level_labels if not lab),
            "ratio_emitters": e_dist,
            "ratio_abstainers": a_dist,
            "scan": lv_scan,
        }
        if lv_scan["vacuous"]:
            REPORT.append(
                f"  step {level}: {len(present):>2} rows, "
                f"{lv_scan['emitters']} emitted, {lv_scan['abstainers']} abstained "
                f"-> NOT a separation test (only one class present)"
            )
            continue
        REPORT.append(
            f"  step {level}: {len(present):>2} rows, "
            f"{lv_scan['emitters']:>2} emitted, {lv_scan['abstainers']:>2} abstained; "
            f"best threshold misclassifies "
            f"{lv_scan['best_misclassifications']}/{lv_scan['population']}"
        )
        if a_dist["n"]:
            REPORT.append(
                f"        emitted ratio mean={e_dist['mean']:.4f} "
                f"[{e_dist['min']:.4f}, {e_dist['max']:.4f}] vs abstained "
                f"mean={a_dist['mean']:.4f} "
                f"[{a_dist['min']:.4f}, {a_dist['max']:.4f}]"
            )

    # ------------------------------------- 11. yardstick robustness (post hoc)
    # NOT a pre-registered hypothesis. The census CHOSE the within-state spread
    # sigma_min as its yardstick; that choice deserves a check against the obvious
    # alternative, and the bundle already records it per step as
    # oracle_audit.min_state_action_gap_true (the smallest top-1-minus-top-2 gap).
    # Reported post hoc so it can never be mistaken for a registered result, and it
    # does not revise H1--H6.
    REPORT.append("\n11. Yardstick robustness (POST-HOC, not pre-registered)")
    alt_scan = threshold_scan(
        [
            r["steps"][0]["oracle_audit"]["min_state_action_gap_true"]
            / float(r["steps"][0]["e_q"])
            for r in rows
        ],
        labels,
    )
    alt_raw = threshold_scan(
        [r["steps"][0]["oracle_audit"]["min_state_action_gap_true"] for r in rows],
        labels,
    )
    majority = min(sum(1 for lab in labels if lab), sum(1 for lab in labels if not lab))
    REPORT.append(
        f"  sigma_min / E_Q        : {scan['best_misclassifications']}/{scan['population']}"
    )
    REPORT.append(
        f"  top-2 gap / E_Q        : "
        f"{alt_scan['best_misclassifications']}/{alt_scan['population']}"
    )
    REPORT.append(
        f"  top-2 gap (raw)        : "
        f"{alt_raw['best_misclassifications']}/{alt_raw['population']}"
    )
    REPORT.append(
        f"  E_Q alone              : "
        f"{e_q_scan['best_misclassifications']}/{e_q_scan['population']}"
    )
    REPORT.append(
        f"  constant predictor     : {majority}/{len(labels)} "
        f"(always predict the majority class)"
    )
    REPORT.append(
        "  direction matters: the E_Q line is the best threshold for predicting "
        "ABSTENTION (the direction section 8 uses). Scanning E_Q in the opposite "
        "direction gives a strictly worse rule, and an earlier scratch version of "
        "this check quoted that worse number -- hence the explicit direction here."
    )
    if e_q_scan["best_misclassifications"] >= majority:
        REPORT.append(
            "  NOTE  even in its better direction, the best threshold on E_Q alone "
            "is no better than predicting the majority class. E_Q carries no "
            "information about WHO emits at step 1; the separation comes from the "
            "spread."
        )
    else:
        REPORT.append(
            f"  NOTE  E_Q alone is informative but weak: "
            f"{e_q_scan['best_misclassifications']}/{e_q_scan['population']} against "
            f"a constant-predictor baseline of {majority}/{len(labels)}, versus "
            f"{scan['best_misclassifications']}/{scan['population']} for the ratio. "
            "It is far from useless, and it is far from sufficient; the census's "
            "H5 passes on means without E_Q being the discriminating quantity."
        )
    REPORT.append(
        "  NOTE  this section changes no hypothesis; it is recorded because the "
        "yardstick was a choice and a worse choice would have changed the story."
    )

    # ------------------------------------------------------- H0 verdict line
    REPORT.append("\n" + "=" * 74)
    REPORT.append("SUMMARY")
    REPORT.append(
        f"  H0 sealed reproduction : "
        f"{'PASS' if len(exact) == total_rows else 'FAIL'} "
        f"({len(exact)}/{total_rows} exact)"
    )
    REPORT.append(f"  H1 step-1 separation   : {'PASS' if h1 else 'FALSIFIED'}")
    REPORT.append(
        f"  H2 separation strength : {'PASS' if h2 else 'FALSIFIED'} "
        f"({scan['best_misclassifications']}/{scan['population']} misclassified "
        f"at best)"
    )
    REPORT.append(f"  H3 plateau vs drop-out : {'PASS' if h3 else 'FALSIFIED'}")
    REPORT.append(f"  H4 ratio decay         : {'PASS' if h4 else 'FALSIFIED'}")
    REPORT.append(f"  H5 binding term        : {'PASS' if h5 else 'FALSIFIED'}")
    REPORT.append(f"  H6 spread is a record property : "
                  f"{'PASS' if h6 else 'FALSIFIED'}")
    REPORT.append("=" * 74)
    REPORT.append(
        "RESULT: the verdicts above are scientific; a FALSIFIED hypothesis is a\n"
        "finding, not a failed check. CONSTRUCTION below covers the checks only."
    )
    REPORT.append(
        "CONSTRUCTION: " + ("PASS" if FAILURES == 0 else f"FAIL ({FAILURES} failed)")
    )

    summary = {
        "task_id": TASK_ID,
        "route_records": total_rows,
        "groups": {
            name: {
                "n": sum(1 for r in rows if r["group"] == name),
                "step1_ratio": distribution(
                    [r["step1_ratio"] for r in rows if r["group"] == name]
                ),
            }
            for name in groups
        },
        "records_with_mixed_routes": len(mixed),
        "H0": {
            "exact": len(exact),
            "total": total_rows,
            "mismatches": {
                field: sum(int(r["repro"].get(field, 0)) for r in rows)
                for field in (
                    "decision_mismatches",
                    "reason_mismatches",
                    "eta_mismatches",
                    "e_q_mismatches",
                )
            },
        },
        "H1": {
            "verdict": "PASS" if h1 else "FALSIFIED",
            "emitters": emit_dist,
            "abstainers": abst_dist,
        },
        "H2": {"verdict": "PASS" if h2 else "FALSIFIED", "scan": scan},
        "abstention_reasons_by_group": reason_by_group,
        "abstention_reasons_by_step": reason_by_step,
        "H3": {
            "verdict": "PASS" if h3 else "FALSIFIED",
            "plateau": p_dist,
            "dropout": d_dist,
            "scan": plateau_scan,
        },
        "H4": {
            "verdict": "PASS" if h4 else "FALSIFIED",
            "verdict_as_literally_written": (
                "PASS" if h4_universal else "FALSIFIED"
            ),
            "trajectories": len(decay_rows),
            "fell": len(fell),
            "note": (
                "the task sheet says 'along each trajectory', the frozen plan "
                "operationalises it as a majority; both readings are recorded"
            ),
        },
        "H5": {
            "verdict": "PASS" if h5 else "FALSIFIED",
            "e_q_abstainers": e_q_abst,
            "e_q_emitters": e_q_emit,
            "sigma_abstainers": sig_abst,
            "sigma_emitters": sig_emit,
            "e_q_only_scan": e_q_scan,
        },
        "H6": {"verdict": "PASS" if h6 else "FALSIFIED"},
        "by_step_level": level_info,
        "yardstick_robustness_posthoc": {
            "note": (
                "POST-HOC robustness check on a choice the census made; not a "
                "pre-registered hypothesis and it revises nothing"
            ),
            "sigma_min_over_e_q": scan["best_misclassifications"],
            "top2_gap_over_e_q": alt_scan["best_misclassifications"],
            "top2_gap_raw": alt_raw["best_misclassifications"],
            "e_q_alone": e_q_scan["best_misclassifications"],
            "constant_predictor": majority,
            "population": len(labels),
            "e_q_is_uninformative": bool(
                e_q_scan["best_misclassifications"] >= majority
            ),
        },
        "decay_detail": decay_rows,
    }
    if cv_rows:
        summary["H6"].update(
            {
                "trajectories": len(cv_rows),
                "cv_sigma": cv_sig,
                "cv_e_q": cv_e_q,
            }
        )

    out_dir = args.census_dir
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "analysis_report.md").write_text(
        "\n".join(REPORT) + "\n", encoding="utf-8"
    )
    print("\n".join(REPORT))

    if args.write_prediction:
        # The prediction is derived ONLY from step-1 data plus each record's own
        # step-5 state, both of which exist before the sixth step is run.
        #
        # Scoring population: every route-record that REACHED step 5, i.e. that
        # emitted steps 1--4 and therefore has a fifth row. That is a superset of
        # the plateau group: a record can emit steps 1--4 and abstain at step 5,
        # in which case it is a candidate for step 6 but not a step-5 emitter.
        theta = scan["best_threshold_interval"][0]
        reached_five = [r for r in rows if len(r["steps"]) >= 5]
        plateau_keys = [(r["mixing"], r["task_index"], r["route"]) for r in plateau]
        at_five = []
        for r in reached_five:
            step5 = r["steps"][4]
            at_five.append(
                {
                    "mixing": r["mixing"],
                    "task_index": r["task_index"],
                    "route": r["route"],
                    "emitted_at_step5": bool(step5["update_emitted"]),
                    "ratio_step5": step5["ratio"],
                    "sigma_min_step5": step5["sigma_min"],
                    "e_q_step5": step5["e_q"],
                    "predicted_emit_step6_by_P2": bool(step5["ratio"] > theta),
                }
            )
        prediction = {
            "task_id": TASK_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "target_task": "FP-ITER6-001",
            "scoring_population": (
                "every route-record that reached step 5 in FP-ITER5-001, i.e. "
                "emitted steps 1--4 and therefore has a fifth row"
            ),
            "scoring_population_size": len(reached_five),
            "theta_source": (
                "best step-1 threshold for predicting emission, lowest value of "
                "the attaining interval, from FP-CENSUS-001's own step-1 scan"
            ),
            "theta": theta,
            "theta_step1_misclassifications": scan["best_misclassifications"],
            "theta_step1_population": scan["population"],
            "predictions": {
                "P1_plateau_carries_on": {
                    "rule": (
                        "every route-record that emitted at step 5 emits at step "
                        "6; every other member of the scoring population abstains"
                    ),
                    "n_predicted_emit": len(plateau_keys),
                    "predicted_emitters": [
                        {"mixing": m, "task_index": t, "route": ro}
                        for m, t, ro in plateau_keys
                    ],
                },
                "P2_step1_ratio_threshold": {
                    "rule": (
                        "emit at step 6 iff ratio(step 5) > theta, with theta "
                        "frozen from step 1"
                    ),
                    "theta": theta,
                    "n_predicted_emit": sum(
                        1 for a in at_five if a["predicted_emit_step6_by_P2"]
                    ),
                    "rows": at_five,
                },
            },
            "scoring": (
                "Against the FP-ITER6-001 numpy bundle, over the scoring "
                "population: emission count, emitting-set equality, and "
                "misclassification count for each rule. H6 requires P2 to "
                "misclassify strictly fewer than P1."
            ),
        }
        out_path = out_dir / "prediction_step6.json"
        out_path.write_text(
            json.dumps(prediction, indent=2, sort_keys=True), encoding="utf-8"
        )
        REPORT.append(f"\nwrote {out_path}")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
