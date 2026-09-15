"""FP-COMPOSE-002 / claude: analysis of the K=12 sealed run.

Scores the pre-registered hypotheses H0-H4 of docs/research_tasks/FP-COMPOSE-002.md and
diffs predictions P0-P3 of pre_registered_prediction.md against what actually happened.
No simulation: every number is derived from the sealed records plus (for the risk-budget
counterfactual) exact arithmetic on the sealed per-pair statistics.

    python analyze.py --results <.../formal/task_results.json> \
        --baseline-results <.../FP-COMPOSE-001/claude/formal/task_results.json> \
        --output-dir <.../formal>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa as es  # noqa: E402

PRODUCERS = ("numpy", "network")
ETA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
GAMMA = 0.70


def route_records(bundle) -> list[dict[str, Any]]:
    out = []
    for rec in bundle["records"]:
        for route in bundle["routes"]:
            per_cell = rec["routes"][route]
            out.append({"family": rec["family"], "mixing": rec["mixing"],
                        "task_index": rec["task_index"], "route": route,
                        "cells": {p: per_cell[f"{p}|perstate|L12"] for p in PRODUCERS
                                  if f"{p}|perstate|L12" in per_cell}})
    return out


def row_margins(pi, q_hat):
    """h_s = max_eta <dpi_s(eta), Qhat_s> / ||dpi_s(eta)||_1 -- the exact gate margin."""
    S = pi.shape[0]
    h = np.full(S, -np.inf)
    for eta in ETA_GRID:
        cand = es.relative_softmax_candidate(pi, q_hat, eta)
        d = cand - pi
        num = np.einsum("sa,sa->s", d, q_hat)
        den = np.abs(d).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(den > 0, num / np.where(den > 0, den, 1.0), -np.inf)
        h = np.maximum(h, ratio)
    return h


def closure_curve(cell, horizon):
    denom = cell["gap_denom"]
    v0 = np.asarray(cell["steps"][0]["audit"]["v_audit"], np.float64)
    base = float(v0.sum())
    cur = 0.0
    out = []
    for e in cell["steps"]:
        if e["emitted"]:
            cur = (float(np.asarray(e["audit"]["v_after"], np.float64).sum()) - base) / denom
        out.append(cur)
    return out + [cur] * (horizon - len(out))


def eq_with_delta_step(e, delta_step):
    """E_Q recomputed from the SEALED pair statistics under a different delta_step.

    Pure arithmetic; no sampling and no decision loop, so this prices the risk-budget
    reading without creating a second arm.
    """
    n = np.asarray(e["pair_sizes"], float)
    v = np.asarray(e["pair_vars"], float)
    m = np.asarray(e["pair_means"], float)
    d = float(delta_step) / (n.size)
    lt = math.log(2.0 / d)
    r = np.sqrt(2.0 * v * lt / n) + (7.0 / 3.0) * float(e["y_range"]) * lt / np.maximum(n - 1.0, 1.0)
    return float(np.max(np.abs(m) + r)) / (1.0 - GAMMA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--baseline-results", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    bundle = json.loads(args.results.read_text(encoding="utf-8"))
    horizon = int(bundle["horizon"])
    rrs = route_records(bundle)
    out: dict[str, Any] = {"task_id": bundle["task_id"], "horizon": horizon,
                           "route_records": len(rrs),
                           "delta_step": bundle["delta_step"],
                           "delta_total": bundle["delta_total"]}

    # ---------------- H0: bit-for-bit reproduction of steps 1..4 ---------------- #
    h0: dict[str, Any] = {"compared": False}
    if args.baseline_results and args.baseline_results.exists():
        base = json.loads(args.baseline_results.read_text(encoding="utf-8"))
        bmap = {}
        for rec in base["records"]:
            for route in base["routes"]:
                for p in PRODUCERS:
                    bmap[(rec["family"], round(float(rec["mixing"]), 6), int(rec["task_index"]),
                          route, p)] = rec["routes"][route][f"{p}|perstate|L12"]["steps"]
        FIELDS = ("q_hat", "pair_sizes", "pair_means", "pair_vars", "pair_radii", "pair_eps",
                  "e_q", "y_range", "delta_dir", "lb_table", "rows_eta", "update_mask",
                  "pi_before", "pi_after", "emitted", "states_updated")
        diffs: list[dict] = []
        checked = 0
        for rr in rrs:
            for p, cell in rr["cells"].items():
                key = (rr["family"], round(float(rr["mixing"]), 6), rr["task_index"],
                       rr["route"], p)
                old = bmap.get(key)
                if old is None:
                    continue
                for e_new, e_old in zip(cell["steps"], old):
                    checked += 1
                    bad = []
                    for f in FIELDS:
                        a, b = e_new.get(f), e_old.get(f)
                        if f in ("e_q", "y_range", "delta_dir"):
                            same = (a is None and b is None) or (
                                a is not None and b is not None
                                and math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=0.0))
                        elif f == "emitted":
                            same = bool(a) == bool(b)
                        elif f == "states_updated":
                            same = int(a) == int(b)
                        else:
                            same = (a is None and b is None) or (
                                a is not None and b is not None
                                and np.array_equal(np.asarray(a), np.asarray(b)))
                        if not same:
                            bad.append(f)
                    if bad:
                        diffs.append({"family": rr["family"], "mixing": rr["mixing"],
                                      "task_index": rr["task_index"], "route": rr["route"],
                                      "producer": p, "step": e_new["step"],
                                      "fields_differing": bad})
        h0 = {"compared": True, "steps_compared": checked,
              "identical_steps": checked - len(diffs), "differing_steps": len(diffs),
              "examples": diffs[:20],
              "verdict": "PASS" if not diffs and checked else "FALSIFIED"}
    out["H0"] = h0

    # ---------------- H1 + H2: safety, and the margin trajectory ---------------- #
    margins, deltas, stops, traj = [], [], [], {}
    coverage_viol, degrading = [], []
    near_boundary: list[dict] = []
    for rr in rrs:
        for p, cell in rr["cells"].items():
            for e in cell["steps"]:
                if e["e_q"] is None:
                    continue
                m = float(e["audit"]["safety_margin"])
                margins.append(m)
                if m < 0:
                    coverage_viol.append({"family": rr["family"], "task_index": rr["task_index"],
                                          "route": rr["route"], "producer": p,
                                          "step": e["step"], "margin": m})
                if e["emitted"]:
                    dv = np.asarray(e["audit"]["value_delta"], float)
                    deltas.append(float(dv.min()))
                    if float(dv.min()) < 0:
                        degrading.append({"family": rr["family"], "task_index": rr["task_index"],
                                          "route": rr["route"], "producer": p,
                                          "step": e["step"], "min_delta": float(dv.min())})
                    if abs(float(dv.min())) <= 1e-8:
                        near_boundary.append(
                            {"kind": "tiny_value_delta", "family": rr["family"],
                             "mixing": rr["mixing"], "task_index": rr["task_index"],
                             "route": rr["route"], "producer": p, "step": e["step"],
                             "min_abs_value_delta": float(np.min(np.abs(dv))),
                             "min_value_delta": float(dv.min()),
                             "e_q": float(e["e_q"]) if e["e_q"] else None})
                pi = np.asarray(e["pi_before"], float)
                qh = np.asarray(e["q_hat"], float)
                h = float(np.max(row_margins(pi, qh)))
                ratio = h / float(e["e_q"]) if e["e_q"] > 0 else None
                traj.setdefault((rr["family"], int(e["step"])), []).append(ratio)
            first_stop = next((e["step"] for e in cell["steps"] if not e["emitted"]), None)
            if first_stop is not None:
                ent = next(e for e in cell["steps"] if e["step"] == first_stop)
                pi = np.asarray(ent["pi_before"], float)
                qh = np.asarray(ent["q_hat"], float)
                h = float(np.max(row_margins(pi, qh)))
                stops.append({"family": rr["family"], "mixing": rr["mixing"],
                              "task_index": rr["task_index"], "route": rr["route"],
                              "producer": p, "first_stop_step": int(first_stop),
                              "h": h, "e_q": float(ent["e_q"]) if ent["e_q"] else None,
                              "h_over_e_q": (h / ent["e_q"]) if ent["e_q"] else None,
                              "states_updated": int(ent["states_updated"]),
                              "reasons": ent["ordered_reasons"]})
    med = {f"{k[0]}|step{k[1]}": float(np.median(v)) for k, v in sorted(traj.items())}
    out["H1"] = {
        "evaluated_steps": len(margins), "coverage_violations": len(coverage_viol),
        "componentwise_degrading_steps": len(degrading),
        "min_safety_margin": (min(margins) if margins else None),
        "min_value_delta": (min(deltas) if deltas else None),
        "near_boundary_points": near_boundary,
        "points_within_1e-10_of_zero": len(near_boundary),
        "examples": (coverage_viol + degrading)[:20],
        "verdict": "PASS" if not coverage_viol and not degrading else "FALSIFIED",
        "note": ("zero violations is a failure detector, not evidence of validity. The "
                 "near-boundary list is reported because the task sheet's section 6 pause "
                 "trigger is |margin| or |delta| <= 1e-10 and a value only an order of "
                 "magnitude above that is a numerical finding, not a formality."),
    }
    out["H2"] = {
        "margin_ratio_median_by_family_step": med,
        "first_stop_events": stops,
        "cells_that_stopped": len(stops),
        "cells_total": len(rrs) * len(PRODUCERS),
        "first_stop_step_min": (min(s["first_stop_step"] for s in stops) if stops else None),
        "verdict": "see P1/P2 comparison",
    }

    # ---------------- H3: fixed-denominator population curve ---------------- #
    per_record = []
    for rr in rrs:
        curves = [closure_curve(rr["cells"][p], horizon) for p in PRODUCERS]
        per_record.append(np.mean(curves, axis=0).tolist())
    arr = np.asarray(per_record, float)
    mean_by_step = arr.mean(axis=0).tolist()
    increments = [mean_by_step[0]] + [mean_by_step[i] - mean_by_step[i - 1]
                                      for i in range(1, horizon)]

    # Index discipline. mean_by_step[i] is the curve value AFTER step i+1, so the
    # contribution of steps a..b is mean_by_step[b-1] - mean_by_step[a-2] (with a-2 < 0
    # meaning 0, i.e. measured from the start). An earlier version wrote the labels one
    # step too low: it called mean[-1]-mean[4] "steps 5..12" when that difference is
    # exactly the contribution of steps 6..12. Caught by an external review.
    def contrib(a, b):
        lo = mean_by_step[a - 2] if a >= 2 else 0.0
        return float(mean_by_step[b - 1] - lo)

    tail_6_12 = contrib(6, horizon)
    tail_5_12 = contrib(5, horizon)
    share_cumulative_positive = (arr > 0).mean(axis=0).tolist()
    # per-STEP increment positivity -- a different and stricter statement than the
    # cumulative one, which is trivially 1.0 once anything is positive. An earlier version
    # reported the cumulative share while the text claimed every step gained.
    inc_pos = (np.diff(np.concatenate([np.zeros((arr.shape[0], 1)), arr], axis=1),
                       axis=1) > 0).mean(axis=0).tolist()
    inc_count = (np.diff(np.concatenate([np.zeros((arr.shape[0], 1)), arr], axis=1),
                         axis=1) > 0).sum(axis=0).tolist()
    out["H3"] = {
        "denominator": int(arr.shape[0]),
        "mean_closure_by_step": mean_by_step,
        "per_step_increment": increments,
        "tail_contribution_k6_to_k12_points": tail_6_12 * 100.0,
        "tail_contribution_k5_to_k12_points": tail_5_12 * 100.0,
        "share_cumulative_positive_by_step": share_cumulative_positive,
        "share_with_positive_STEP_increment_by_step": inc_pos,
        "count_with_positive_STEP_increment_by_step": inc_count,
        "share_at_ceiling_by_step": (arr >= 1 - 1e-9).mean(axis=0).tolist(),
        "verdict": ("non-negligible" if tail_6_12 * 100.0 >= 2.0
                    else "negligible" if tail_6_12 * 100.0 < 1.0 else "inconclusive"),
        "note": ("fixed denominator over all route-records, stopped trajectories hold their "
                 "last value, no rows dropped. Descriptive only; no convergence rate is "
                 "claimed and no equal-cost advantage is implied. The CUMULATIVE share stays "
                 "1.0 by construction once any gain exists; the per-STEP share is the "
                 "stricter statistic and is the one to quote."),
    }

    # ---------------- H4: producer correspondence ---------------- #
    def sig(cell):
        return ([e["emitted"] for e in cell["steps"]],
                [e["rows_eta"] for e in cell["steps"]])
    bad = []
    for rr in rrs:
        if sig(rr["cells"]["numpy"]) != sig(rr["cells"]["network"]):
            bad.append({"family": rr["family"], "mixing": rr["mixing"],
                        "task_index": rr["task_index"], "route": rr["route"]})
    out["H4"] = {"route_records": len(rrs), "mismatched": len(bad),
                 "examples": bad[:20],
                 "verdict": "PASS" if not bad and rrs else "FALSIFIED"}

    # ---------------- risk-budget counterfactual (delta_total = 0.05 reading) -------- #
    cf_delta = 0.05 / horizon
    old_ratio, new_ratio = [], []
    for rr in rrs:
        for p, cell in rr["cells"].items():
            for e in cell["steps"]:
                if e["pair_sizes"] is None:
                    continue
                pi = np.asarray(e["pi_before"], float)
                qh = np.asarray(e["q_hat"], float)
                h = float(np.max(row_margins(pi, qh)))
                e_new = eq_with_delta_step(e, cf_delta)
                old_ratio.append(h / e_new)
                new_ratio.append(h / float(e["e_q"]))
    out["risk_budget_counterfactual"] = {
        "sealed_delta_step": bundle["delta_step"],
        "sealed_delta_total": bundle["delta_total"],
        "counterfactual_delta_step": cf_delta,
        "counterfactual_delta_total": 0.05,
        "margin_ratio_median_sealed": float(np.median(new_ratio)),
        "margin_ratio_median_counterfactual": float(np.median(old_ratio)),
        "e_q_inflation_factor_median": float(np.median(
            [a / b for a, b in zip(new_ratio, old_ratio)])),
        "cell_steps_with_closed_gate_sealed": int(sum(1 for x in new_ratio if x <= 1.0)),
        "cell_steps_with_closed_gate_counterfactual": int(sum(1 for x in old_ratio if x <= 1.0)),
        "cell_steps_total": len(new_ratio),
        "note": ("EXACT ARITHMETIC on the sealed per-pair statistics under a smaller "
                 "delta_step. Emission COUNTS are deliberately NOT reported: a different "
                 "E_Q changes the decisions, which changes the policy path, which would "
                 "require re-running the gate -- that would be a second arm, which the task "
                 "sheet forbids."),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"H0": out["H0"].get("verdict"), "H1": out["H1"]["verdict"],
                      "H3": out["H3"]["verdict"], "H4": out["H4"]["verdict"],
                      "cells_that_stopped": out["H2"]["cells_that_stopped"],
                      "cells_total": out["H2"]["cells_total"],
                      "first_stop_step_min": out["H2"]["first_stop_step_min"],
                      "tail_k6_12_points": out["H3"]["tail_contribution_k6_to_k12_points"],
                      "tail_k5_12_points": out["H3"]["tail_contribution_k5_to_k12_points"],
                      "positive_step_increment_counts":
                          out["H3"]["count_with_positive_STEP_increment_by_step"],
                      "mean_closure_by_step": [round(x, 4) for x in
                                               out["H3"]["mean_closure_by_step"]]},
                     indent=2))


if __name__ == "__main__":
    main()
