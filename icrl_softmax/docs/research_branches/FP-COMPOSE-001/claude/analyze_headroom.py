"""FP-COMPOSE-001 / claude: how much headroom does the gate actually have, step by step.

This analyses the SEALED bundle only. It runs no new protocol, adds no arm, and changes no
frozen quantity -- it answers a question the adversarial checks left open: y_range grows
with the step because ||Vhat||_inf grows, so how close did the run come to the point where
no row clears LB_s > 0?

The gate is exact and re-derivable from sealed quantities. For a row s and an eta,

    LB_s(eta) = <dpi_s(eta), Qhat_s> - E_Q * ||dpi_s(eta)||_1

and the row is updated iff some eta gives LB_s > 0, i.e. iff

    E_Q < h_s := max_eta  <dpi_s(eta), Qhat_s> / ||dpi_s(eta)||_1

so the cell emits iff E_Q < h := max_s h_s. The quantity h is the exact gate margin: no
probability, no sampling, a deterministic function of the sealed (pi_before, Qhat). The
headroom ratio h / E_Q therefore says how much the certified error could still grow before
this cell would go silent, and E_Q* = h is the critical value.

    python analyze_headroom.py --results <.../formal/task_results.json> \
        --output <.../verification/headroom_analysis.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa as es  # noqa: E402
from evaluate_fp_xfam_001 import FAMILIES  # noqa: E402

ETA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


def row_margins(pi, q_hat):
    """h_s = max_eta <dpi_s(eta), Qhat_s> / ||dpi_s(eta)||_1, plus the argmax eta."""
    S = pi.shape[0]
    h = np.zeros(S)
    arg = [None] * S
    for eta in ETA_GRID:
        cand = es.relative_softmax_candidate(pi, q_hat, eta)
        d = cand - pi
        num = np.einsum("sa,sa->s", d, q_hat)
        den = np.abs(d).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(den > 0, num / np.where(den > 0, den, 1.0), -np.inf)
        take = ratio > h
        h = np.where(take, ratio, h)
        for s in np.flatnonzero(take):
            arg[s] = float(eta)
    return h, arg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    bundle = json.loads(args.results.read_text(encoding="utf-8"))
    horizon = int(bundle["horizon"])
    producers = tuple(bundle["producers"]); routes = tuple(bundle["routes"])

    t0 = time.time()
    rows: list[dict[str, Any]] = []
    for task in bundle["records"]:
        fam_name = task["family"]; fam = FAMILIES[fam_name]
        S = fam["n_states"]
        for route in routes:
            for producer in producers:
                cell = task["routes"][route][f"{producer}|perstate|L12"]
                for e in cell["steps"]:
                    if e["pair_sizes"] is None:
                        continue
                    pi = np.asarray(e["pi_before"], np.float64)
                    qh = np.asarray(e["q_hat"], np.float64)
                    h, arg = row_margins(pi, qh)
                    e_q = float(e["e_q"])
                    v_hat = np.einsum("sa,sa->s", pi, qh)
                    rows.append({
                        "family": fam_name, "mixing": task["mixing"],
                        "task_index": task["task_index"], "route": route,
                        "producer": producer, "step": int(e["step"]),
                        "y_range": float(e["y_range"]), "e_q": e_q,
                        "safety_margin": float(e["audit"]["safety_margin"]),
                        "h_cell": float(np.max(h)), "h_min_row": float(np.min(h)),
                        "headroom_ratio": float(np.max(h)) / e_q if e_q > 0 else None,
                        "v_hat_inf": float(np.max(np.abs(v_hat))),
                        "q_hat_inf": float(np.max(np.abs(qh))),
                        "rows_above_e_q": int(np.sum(h > e_q)),
                        "rows_total": int(S),
                        "states_updated": int(e["states_updated"]),
                        "emitted": bool(e["emitted"]),
                    })
    arr = rows

    def agg(sel, key):
        v = [r[key] for r in arr if sel(r) and r[key] is not None]
        return {"n": len(v), "min": (min(v) if v else None),
                "median": (float(np.median(v)) if v else None),
                "max": (max(v) if v else None)}

    summary: dict[str, Any] = {}
    for fam in ("f1", "f2"):
        for step in range(1, horizon + 1):
            sel = (lambda r, f=fam, s=step: r["family"] == f and r["step"] == s)
            g = lambda k, sl=sel: agg(sl, k)              # noqa: E731
            summary[f"{fam}|step{step}"] = {
                "n": sum(1 for r in arr if sel(r)),
                "y_range": g("y_range"), "e_q": g("e_q"),
                "safety_margin": g("safety_margin"),
                "h_cell": g("h_cell"), "headroom_ratio": g("headroom_ratio"),
                "v_hat_inf": g("v_hat_inf"), "q_hat_inf": g("q_hat_inf"),
                "rows_above_e_q_median": float(np.median(
                    [r["rows_above_e_q"] for r in arr if sel(r)])),
                "rows_total": FAMILIES[fam]["n_states"],
            }

    # growth of the radius ingredients across steps, per family
    growth: dict[str, Any] = {}
    for fam in ("f1", "f2"):
        for key in ("y_range", "e_q", "v_hat_inf", "q_hat_inf", "headroom_ratio"):
            med = [summary[f"{fam}|step{s}"][key]["median"] for s in range(1, horizon + 1)]
            growth[f"{fam}|{key}"] = {
                "by_step_median": med,
                "ratio_last_over_first": (med[-1] / med[0]
                                          if med[0] not in (None, 0) else None),
                "mean_step_multiplier": ((med[-1] / med[0]) ** (1 / (horizon - 1))
                                         if med[0] not in (None, 0) else None),
            }

    worst = min((r for r in arr if r["headroom_ratio"] is not None),
                key=lambda r: r["headroom_ratio"])
    tightest_margin = min(arr, key=lambda r: r["safety_margin"])
    out = {
        "framing": ("Analysis of the SEALED bundle only; no new protocol, no new arm, no "
                    "changed frozen quantity. The gate margin h is exact and deterministic "
                    "in the sealed (pi_before, Qhat), so the stopping condition can be "
                    "located without sampling."),
        "steps_analysed": len(arr),
        "gate_algebra": {
            "row_condition": "LB_s(eta) = <dpi_s(eta), Qhat_s> - E_Q * ||dpi_s(eta)||_1 > 0",
            "equivalently": "E_Q < h_s = max_eta <dpi_s(eta), Qhat_s> / ||dpi_s(eta)||_1",
            "cell_condition": "E_Q < h = max_s h_s",
            "note": ("E_Q* = h is the critical certified error at which this cell would go "
                     "silent. h depends only on (pi_before, Qhat), both sealed."),
        },
        "by_family_step": summary,
        "growth_across_steps": growth,
        "tightest": {
            "worst_headroom_ratio": worst["headroom_ratio"],
            "worst_headroom_at": {k: worst[k] for k in
                                  ("family", "mixing", "task_index", "route", "producer",
                                   "step", "h_cell", "e_q")},
            "tightest_safety_margin": tightest_margin["safety_margin"],
            "tightest_margin_at": {k: tightest_margin[k] for k in
                                   ("family", "mixing", "task_index", "route", "producer",
                                    "step", "e_q", "y_range")},
        },
        "what_this_does_not_show": [
            "The step-wise growth of y_range is measured over K=4 only; extrapolating it to "
            "K>4 is an extrapolation, not a measurement, and no K>4 run exists.",
            "The observed trend cannot separate 'the radius grows' from 'the policy "
            "converges so that ||dpi|| shrinks'; both push the headroom in the same "
            "direction only if the numerator falls faster, which is not established here.",
            "This does not create a new claim about the composition; it bounds how far the "
            "sealed run sat from its own stopping condition.",
        ],
        "wall_seconds": None,
    }
    out["wall_seconds"] = time.time() - t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "steps_analysed": out["steps_analysed"],
        "worst_headroom_ratio": out["tightest"]["worst_headroom_ratio"],
        "worst_headroom_at": out["tightest"]["worst_headroom_at"],
        "tightest_safety_margin": out["tightest"]["tightest_safety_margin"],
        "growth": {k: {"by_step_median": [None if x is None else round(x, 4) for x in
                                          v["by_step_median"]],
                       "ratio_last_over_first": (None if v["ratio_last_over_first"] is None
                                                 else round(v["ratio_last_over_first"], 3))}
                   for k, v in growth.items()
                   if k.split("|")[1] in ("y_range", "e_q", "headroom_ratio")},
    }, indent=2))


if __name__ == "__main__":
    main()
