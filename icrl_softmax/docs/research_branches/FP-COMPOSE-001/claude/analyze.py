"""FP-COMPOSE-001 / claude: analysis of the sealed formal run.

Reads results/FP-COMPOSE-001/claude/formal/task_results.json (produced by evaluate.py)
and scores the pre-registered hypotheses H1-H3 plus the delivery criteria D1-D4 of the
frozen task sheet section 5. It performs no simulation and no new sampling: every number
is derived from the sealed records.

    python analyze.py --results <.../formal/task_results.json> --output-dir <.../formal>

Hypotheses (task sheet section 5):
  H1 safety failure detection -- every step's E_Q >= that cell's own ||Qhat - Qpi||inf,
     and every emitted step's v_piNew - v_pi is componentwise >= 0.
  H2 non-empty multi-step composition -- within each (family, route) group the network
     producer has at least one trajectory emitting two consecutive updates.
  H3 producer correspondence -- for all 192 route-records the step-wise emit/abstain
     sequence AND the per-row chosen eta (including the no-update marker) are identical
     across the two producers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

PRODUCERS = ("numpy", "network")


def cells_of(record) -> dict[str, dict]:
    """Flatten a task-instance record into cell key -> {'route', 'producer', ...}."""
    out = {}
    for route, per_cell in record["routes"].items():
        for cell, payload in per_cell.items():
            out[cell] = {"route": route, "producer": cell.split("|")[0], **payload}
    return out


def route_records(bundle) -> list[dict[str, Any]]:
    """The 192 analysis units: one per (family, mixing, task_index, route)."""
    out = []
    for rec in bundle["records"]:
        flat = cells_of(rec)
        for route in bundle["routes"]:
            out.append({
                "family": rec["family"], "mixing": rec["mixing"],
                "task_index": rec["task_index"], "route": route,
                "cells": {p: flat[f"{p}|perstate|L12"] for p in PRODUCERS
                          if f"{p}|perstate|L12" in flat},
            })
    return out


# --------------------------------------------------------------------------- #
# H1 -- safety failure detection
# --------------------------------------------------------------------------- #
def score_h1(rrs: list[dict]) -> dict[str, Any]:
    n_steps = 0
    violations: list[dict] = []
    margins: list[float] = []
    min_delta: float | None = None
    min_delta_where: dict | None = None
    nonfinite: list[dict] = []
    abstain_reasons: dict[str, int] = {}
    emitted_steps = 0
    emitted_degrading = 0
    tiny: list[dict] = []                      # |margin| or |delta| <= 1e-10 (section 6)
    for rr in rrs:
        for producer, cell in rr["cells"].items():
            for e in cell["steps"]:
                tag = {"family": rr["family"], "mixing": rr["mixing"],
                       "task_index": rr["task_index"], "route": rr["route"],
                       "producer": producer, "step": e["step"]}
                n_steps += 1
                for reason in e["ordered_reasons"]:
                    abstain_reasons[reason] = abstain_reasons.get(reason, 0) + 1
                if e["e_q"] is None:
                    continue
                if not math.isfinite(e["e_q"]):
                    nonfinite.append({**tag, "field": "e_q", "value": e["e_q"]})
                m = e["audit"]["safety_margin"]
                margins.append(m)
                if not math.isfinite(m):
                    nonfinite.append({**tag, "field": "safety_margin", "value": m})
                if not e["audit"]["covers"] or m < 0.0:
                    violations.append({**tag, "kind": "coverage",
                                       "e_q": e["e_q"],
                                       "realized_sup_error": e["audit"]["realized_sup_error"],
                                       "margin": m})
                if abs(m) <= 1e-10:
                    tiny.append({**tag, "field": "safety_margin", "value": m})
                if e["emitted"]:
                    emitted_steps += 1
                    dv = e["audit"].get("value_delta")
                    if dv is None:
                        violations.append({**tag, "kind": "missing_value_delta"})
                        continue
                    dv = np.asarray(dv, dtype=np.float64)
                    if not np.all(np.isfinite(dv)):
                        nonfinite.append({**tag, "field": "value_delta"})
                    if float(np.min(dv)) < 0.0:
                        emitted_degrading += 1
                        violations.append({**tag, "kind": "componentwise_degrading",
                                           "min_value_delta": float(np.min(dv)),
                                           "value_delta": dv.tolist()})
                    if min_delta is None or float(np.min(dv)) < min_delta:
                        min_delta = float(np.min(dv))
                        min_delta_where = tag
                    if float(np.min(np.abs(dv))) <= 1e-10:
                        tiny.append({**tag, "field": "min_abs_value_delta",
                                     "value": float(np.min(np.abs(dv)))})
    return {
        "evaluated_steps": n_steps,
        "steps_with_certificate": len(margins),
        "steps_abstained_no_certificate": n_steps - len(margins),
        "emitted_steps": emitted_steps,
        "coverage_violations": len(violations),
        "componentwise_degrading_emitted_steps": emitted_degrading,
        "min_safety_margin": (min(margins) if margins else None),
        "mean_safety_margin": (float(np.mean(margins)) if margins else None),
        "min_value_delta_over_emitted_steps": min_delta,
        "min_value_delta_at": min_delta_where,
        "nonfinite_points": nonfinite,
        "points_within_1e-10_of_boundary": tiny,
        "abstention_reasons": abstain_reasons,
        "violations": violations[:50],
        "violation_count_total": len(violations),
        "verdict": "PASS" if not violations and not nonfinite else "FALSIFIED",
        "note": ("Zero violations is an empirical failure-detection check only; it is not "
                 "a probabilistic proof."),
    }


# --------------------------------------------------------------------------- #
# H2 -- non-empty multi-step composition
# --------------------------------------------------------------------------- #
def consecutive_emits(steps: list[dict]) -> int:
    """Longest run of consecutive emitted steps."""
    best = run = 0
    for e in steps:
        run = run + 1 if e["emitted"] else 0
        best = max(best, run)
    return best


def score_h2(rrs: list[dict]) -> dict[str, Any]:
    groups: dict[tuple, list] = {}
    for rr in rrs:
        groups.setdefault((rr["family"], rr["route"]), []).append(rr)
    out = {}
    for (fam, route), members in sorted(groups.items()):
        for producer in PRODUCERS:
            runs = [consecutive_emits(m["cells"][producer]["steps"]) for m in members]
            emitters = [r for r in runs if r >= 2]
            out[f"{fam}|{route}|{producer}"] = {
                "route_records": len(members),
                "trajectories_with_two_consecutive_emits": len(emitters),
                "max_consecutive_emit_run": (max(runs) if runs else 0),
                "verdict": "PASS" if emitters else "FALSIFIED",
            }
    net = {k: v for k, v in out.items() if k.endswith("|network")}
    return {"groups": out,
            "network_groups": net,
            "verdict": "PASS" if net and all(v["verdict"] == "PASS"
                                             for v in net.values()) else "FALSIFIED"}


# --------------------------------------------------------------------------- #
# H3 -- producer correspondence
# --------------------------------------------------------------------------- #
def signature(cell: dict) -> dict:
    return {"emit": [e["emitted"] for e in cell["steps"]],
            "rows_eta": [e["rows_eta"] for e in cell["steps"]],
            "mask": [e["update_mask"] for e in cell["steps"]],
            "reasons": [e["ordered_reasons"] for e in cell["steps"]]}


def score_h3(rrs: list[dict]) -> dict[str, Any]:
    mismatch: list[dict] = []
    exact = 0
    for rr in rrs:
        a, b = (signature(rr["cells"][p]) for p in PRODUCERS)
        if a == b:
            exact += 1
        else:
            same_len = len(a["emit"]) == len(b["emit"])
            mismatch.append({
                "family": rr["family"], "mixing": rr["mixing"],
                "task_index": rr["task_index"], "route": rr["route"],
                "same_step_count": same_len,
                "emit_numpy": a["emit"], "emit_network": b["emit"],
                "rows_eta_numpy": a["rows_eta"], "rows_eta_network": b["rows_eta"],
                "first_divergent_step": next(
                    (i + 1 for i in range(min(len(a["emit"]), len(b["emit"])))
                     if a["emit"][i] != b["emit"][i]
                     or a["rows_eta"][i] != b["rows_eta"][i]), None),
            })
    return {
        "route_records": len(rrs),
        "identical": exact,
        "mismatched": len(mismatch),
        "mismatch_detail": mismatch[:40],
        "verdict": "PASS" if not mismatch and rrs else "FALSIFIED",
        "note": ("Identity of the emit/abstain sequence and per-row eta is required; an "
                 "equal emission total does not substitute. Numerical equality of Qhat is "
                 "explicitly NOT required."),
    }


# --------------------------------------------------------------------------- #
# D1 -- delivery completeness
# --------------------------------------------------------------------------- #
REQUIRED_STEP_FIELDS = ("step", "producer", "route", "pi_before", "q_hat", "pair_sizes",
                        "pair_means", "pair_vars", "pair_radii", "pair_eps", "e_q",
                        "y_range", "delta_dir", "delta_step", "delta_total", "lb_table",
                        "rows_eta", "update_mask", "states_updated", "pi_after",
                        "emitted", "ordered_reasons", "audit", "items_this_step",
                        "batch_hash", "producer_gap_same_policy")
AUDIT_FIELDS = ("q_pi_audit", "v_audit", "realized_sup_error", "covers", "safety_margin")


def score_d1(bundle, rrs: list[dict]) -> dict[str, Any]:
    missing: list[dict] = []
    steps = 0
    for rr in rrs:
        for producer, cell in rr["cells"].items():
            for e in cell["steps"]:
                steps += 1
                for f in REQUIRED_STEP_FIELDS:
                    if f not in e:
                        missing.append({"pair": [rr["family"], rr["task_index"], rr["route"],
                                                 producer, e.get("step")], "field": f})
                for f in AUDIT_FIELDS:
                    if f not in e.get("audit", {}):
                        missing.append({"pair": [rr["family"], rr["task_index"], rr["route"],
                                                 producer, e.get("step")], "audit_field": f})
                if e["emitted"]:
                    for f in ("v_after", "value_delta", "componentwise_nondegrading",
                              "total_value_gain", "closure_fraction"):
                        if f not in e["audit"]:
                            missing.append({"pair": [rr["family"], rr["task_index"],
                                                     rr["route"], producer, e["step"]],
                                            "emitted_audit_field": f})
    return {"steps_checked": steps, "missing_fields": missing,
            "missing_count": len(missing),
            "verdict": "PASS" if not missing else "FAIL"}


# --------------------------------------------------------------------------- #
# D2 -- cost
# --------------------------------------------------------------------------- #
def score_d2(bundle) -> dict[str, Any]:
    cost = bundle["cost"]
    per_cell = {}
    for c in bundle["cells"]:
        items = 0
        steps = 0
        for rec in bundle["records"]:
            for route, per_cell_rec in rec["routes"].items():
                if c in per_cell_rec:
                    for e in per_cell_rec[c]["steps"]:
                        items += e["items_this_step"]
                        steps += 1
        per_cell[c] = {"producer_steps": steps, "items": items}
    wall = cost["wall"]
    return {
        "items_per_batch": cost["items_per_batch"],
        "unique_batches_actually_drawn": cost["unique_batches"],
        "items_actually_drawn_paired": cost["items_unique"],
        "items_if_each_cell_run_alone": cost["items_if_run_alone"],
        "items_if_each_cell_run_alone_total": sum(cost["items_if_run_alone"].values()),
        "per_cell": per_cell,
        "network_forwards": bundle["net_forwards"],
        "wall_seconds": wall,
        "wall_shares": {k: (v / wall["total"] if wall["total"] else None)
                        for k, v in wall.items() if k != "total"},
        "note": ("Data budget, counterfactual standalone volume and the paired unique-batch "
                 "volume are reported separately. Shared sampling wall clock is NOT split "
                 "into fictitious per-cell clocks, and no equal-cost advantage is claimed."),
    }


# --------------------------------------------------------------------------- #
# D3 -- full population value curve
# --------------------------------------------------------------------------- #
def closure_curve(cell: dict) -> list[float]:
    """Closure fraction after each step 1..K, holding the last value once stopped."""
    denom = cell["gap_denom"]
    v0 = np.asarray(cell["steps"][0]["audit"]["v_audit"], dtype=np.float64)
    base = float(np.sum(v0))
    out = []
    cur = 0.0
    for e in cell["steps"]:
        if e["emitted"]:
            vt = np.asarray(e["audit"]["v_after"], dtype=np.float64)
            cur = float(np.sum(vt) - base) / denom
        out.append(cur)
    return out


def score_d3(bundle, rrs: list[dict]) -> dict[str, Any]:
    horizon = bundle["horizon"]
    per_producer: dict[str, list[list[float]]] = {p: [] for p in PRODUCERS}
    per_record: list[list[float]] = []
    for rr in rrs:
        curves = {}
        for p in PRODUCERS:
            cell = rr["cells"][p]
            c = closure_curve(cell)
            c = c + [c[-1]] * (horizon - len(c))
            curves[p] = c
            per_producer[p].append(c)
        per_record.append([float(np.mean([curves[p][k] for p in PRODUCERS]))
                           for k in range(horizon)])

    def summarise(rows):
        arr = np.asarray(rows, dtype=np.float64)
        return {"denominator": int(arr.shape[0]),
                "mean_by_step": [float(x) for x in arr.mean(axis=0)],
                "median_by_step": [float(x) for x in np.median(arr, axis=0)],
                "sd_by_step": [float(x) for x in arr.std(axis=0, ddof=1)],
                "share_positive_by_step": [float(x) for x in (arr > 0).mean(axis=0)],
                "share_at_ceiling_by_step": [float(x) for x in (arr >= 1.0 - 1e-9).mean(axis=0)]}

    population = summarise(per_record)
    survivors = {}
    arr = np.asarray(per_record)
    for k in range(horizon):
        active = arr[:, k] > 0.0
        survivors[f"step_{k + 1}"] = {
            "n_positive": int(active.sum()),
            "mean_among_positive": (float(arr[active, k].mean()) if active.any() else None),
        }
    # per-group curves, descriptive
    groups: dict[str, list[list[float]]] = {}
    for rr, row in zip(rrs, per_record):
        groups.setdefault(f"{rr['family']}|{rr['route']}", []).append(row)
    return {
        "population": population,
        "population_by_producer": {p: summarise(rows) for p, rows in per_producer.items()},
        "survivor_conditioned": survivors,
        "by_group": {k: summarise(v) for k, v in sorted(groups.items())},
        "note": ("Fixed denominator over all route-records; stopped trajectories hold their "
                 "last value. Survivor-conditioned figures are labelled with their per-step "
                 "counts. Descriptive only -- no late-stage gain or geometric rate is tested."),
    }


# --------------------------------------------------------------------------- #
# D4 -- paired producer comparison
# --------------------------------------------------------------------------- #
def score_d4(rrs: list[dict]) -> dict[str, Any]:
    same_policy_gaps: list[float] = []          # network cells only: the real recompute
    same_policy_gaps_all: list[float] = []      # incl. numpy cells, 0 by construction
    own_path_gaps: list[float] = []
    own_path_rows: list[float] = []
    divergent_pairs = 0
    pair_total = 0
    qhat_equal_rows = 0
    worst: list[dict] = []
    step_count_mismatch = []
    for rr in rrs:
        a, b = rr["cells"]["numpy"], rr["cells"]["network"]
        if len(a["steps"]) != len(b["steps"]):
            step_count_mismatch.append({"family": rr["family"], "mixing": rr["mixing"],
                                        "task_index": rr["task_index"], "route": rr["route"],
                                        "numpy_steps": len(a["steps"]),
                                        "network_steps": len(b["steps"])})
        for ea, eb in zip(a["steps"], b["steps"]):
            qa = np.asarray(ea["q_hat"], dtype=np.float64)
            qb = np.asarray(eb["q_hat"], dtype=np.float64)
            d_own = float(np.max(np.abs(qa - qb)))
            own_path_gaps.append(d_own)
            own_path_rows.append(float(np.mean(np.abs(qa - qb))))
            same_policy_gaps_all.append(float(eb["producer_gap_same_policy"]))
            if rr["cells"]["network"] is b:
                same_policy_gaps.append(float(eb["producer_gap_same_policy"]))
            d = np.abs(qa - qb)
            pair_total += int(d.size)
            divergent_pairs += int(np.count_nonzero(d > 0.0))
            qhat_equal_rows += int(np.count_nonzero(d == 0.0))
            worst.append({"family": rr["family"], "mixing": rr["mixing"],
                          "task_index": rr["task_index"], "route": rr["route"],
                          "step": ea["step"], "own_path_max_abs_diff": d_own,
                          "same_policy_max_abs_diff": float(eb["producer_gap_same_policy"]),
                          "own_policy_identical": bool(np.array_equal(
                              np.asarray(ea["pi_before"]), np.asarray(eb["pi_before"])))})
    worst.sort(key=lambda w: -w["own_path_max_abs_diff"])
    sp = np.asarray(same_policy_gaps)
    sp_all = np.asarray(same_policy_gaps_all)
    op = np.asarray(own_path_gaps)

    def q(v):
        return {f"p{p}": float(np.percentile(v, p)) for p in (0, 50, 90, 99, 100)}

    return {
        "step_pairs": len(own_path_gaps),
        "own_policy_path_qhat_diff": {"max": float(op.max()), "mean": float(op.mean()),
                                      "quantiles": q(op)},
        "same_policy_recompute_diff": {
            "measured_on": "network cells only (the numpy cell needs no recompute)",
            "n": int(sp.size),
            "max": float(sp.max()), "mean": float(sp.mean()), "quantiles": q(sp),
            "n_exactly_zero": int(np.count_nonzero(sp == 0.0))},
        "same_policy_recompute_diff_incl_numpy_cells_zero_by_construction": {
            "n": int(sp_all.size), "max": float(sp_all.max())},
        "per_row_abs_diff_mean_over_steps": float(np.mean(own_path_rows)),
        "pairs_compared": pair_total,
        "pairs_with_any_diff": divergent_pairs,
        "pairs_bitwise_equal": qhat_equal_rows,
        "all_pi_paths_identical": all(w["own_policy_identical"] for w in worst),
        "pi_path_mismatch_note": (
            "pi_before values may differ in the last bits because the network produces Qhat "
            "in float32; the frozen H3 requirement is identity of the emit/abstain sequence "
            "and of the per-row chosen eta, which is checked separately and does not require "
            "numerical equality of Qhat."),
        "step_count_mismatches": step_count_mismatch,
        "worst_20": worst[:20],
        "note": ("'own_policy_path_qhat_diff' compares the two producers along their own "
                 "policy paths (the main-grid quantity). 'same_policy_recompute_diff' is the "
                 "extra numpy recompute under the SAME policy and is audit-only; it never "
                 "drives a main-grid update."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    bundle = json.loads(args.results.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if bundle.get("smoke"):
        print("WARNING: this bundle is a smoke run, not the frozen protocol")

    rrs = route_records(bundle)
    analysis = {
        "task_id": bundle["task_id"], "actor": bundle["actor"],
        "baseline": bundle["baseline"], "smoke": bundle.get("smoke", False),
        "route_records_analysed": len(rrs),
        "families": sorted({r["family"] for r in rrs}),
        "H1": score_h1(rrs), "H2": score_h2(rrs), "H3": score_h3(rrs),
        "D1": score_d1(bundle, rrs), "D2": score_d2(bundle),
        "D3": score_d3(bundle, rrs), "D4": score_d4(rrs),
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")

    a = analysis
    lines = [
        "# FP-COMPOSE-001 / claude -- analysis of the sealed formal run", "",
        f"- route-records analysed: {a['route_records_analysed']}",
        f"- families: {', '.join(a['families'])}   smoke: {a['smoke']}", "",
        "## H1 safety failure detection",
        f"- evaluated producer-steps: {a['H1']['evaluated_steps']}"
        f" (with certificate {a['H1']['steps_with_certificate']},"
        f" abstained {a['H1']['steps_abstained_no_certificate']})",
        f"- emitted steps: {a['H1']['emitted_steps']}",
        f"- coverage violations: {a['H1']['coverage_violations']}",
        f"- componentwise-degrading emitted steps: "
        f"{a['H1']['componentwise_degrading_emitted_steps']}",
        f"- min safety margin: {a['H1']['min_safety_margin']}",
        f"- min value delta at an emitted step: {a['H1']['min_value_delta_over_emitted_steps']}",
        f"- non-finite points: {len(a['H1']['nonfinite_points'])}",
        f"- points within 1e-10 of the boundary: "
        f"{len(a['H1']['points_within_1e-10_of_boundary'])}",
        f"- abstention reasons: {json.dumps(a['H1']['abstention_reasons'], sort_keys=True)}",
        f"- **H1 verdict: {a['H1']['verdict']}**", "",
        "## H2 non-empty multi-step composition",
    ]
    for k, v in a["H2"]["groups"].items():
        lines.append(f"- {k}: {v['trajectories_with_two_consecutive_emits']}/"
                     f"{v['route_records']} trajectories with >=2 consecutive emits "
                     f"(max run {v['max_consecutive_emit_run']}) -> {v['verdict']}")
    lines += [f"- **H2 verdict: {a['H2']['verdict']}**", "",
              "## H3 producer correspondence",
              f"- identical route-records: {a['H3']['identical']}/{a['H3']['route_records']}",
              f"- mismatched: {a['H3']['mismatched']}",
              f"- **H3 verdict: {a['H3']['verdict']}**", "",
              "## D1 delivery completeness",
              f"- steps checked: {a['D1']['steps_checked']}, missing fields: "
              f"{a['D1']['missing_count']} -> {a['D1']['verdict']}", "",
              "## D2 cost",
              f"- items per batch: {a['D2']['items_per_batch']}",
              f"- unique batches drawn: {a['D2']['unique_batches_actually_drawn']}",
              f"- items actually drawn (paired): {a['D2']['items_actually_drawn_paired']}",
              f"- items if each cell ran alone: "
              f"{a['D2']['items_if_each_cell_run_alone_total']}",
              f"- network forwards: {a['D2']['network_forwards']}",
              f"- wall seconds: {json.dumps(a['D2']['wall_seconds'], sort_keys=True)}", "",
              "## D3 population value curve",
              f"- denominator: {a['D3']['population']['denominator']}",
              f"- mean closure by step: "
              f"{[round(x, 4) for x in a['D3']['population']['mean_by_step']]}",
              f"- share positive by step: "
              f"{[round(x, 4) for x in a['D3']['population']['share_positive_by_step']]}",
              f"- share at ceiling by step: "
              f"{[round(x, 4) for x in a['D3']['population']['share_at_ceiling_by_step']]}",
              f"- survivors: {json.dumps(a['D3']['survivor_conditioned'], sort_keys=True)}",
              "", "## D4 paired producer comparison",
              f"- step pairs: {a['D4']['step_pairs']}",
              f"- own-path Qhat max diff: {a['D4']['own_policy_path_qhat_diff']['max']}",
              f"- same-policy recompute max diff (network cells): "
              f"{a['D4']['same_policy_recompute_diff']['max']} "
              f"over n={a['D4']['same_policy_recompute_diff']['n']}",
              f"- pairs bitwise equal: {a['D4']['pairs_bitwise_equal']}/"
              f"{a['D4']['pairs_compared']}",
              f"- all pi paths identical: {a['D4']['all_pi_paths_identical']}", "",
              "All results are preliminary until independently verified.", ""]
    (args.output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"H1": a["H1"]["verdict"], "H2": a["H2"]["verdict"],
                      "H3": a["H3"]["verdict"], "D1": a["D1"]["verdict"],
                      "route_records": a["route_records_analysed"]}, indent=2))


if __name__ == "__main__":
    main()
