"""FP-INDEP-001 / claude: analysis of the sealed formal run.

Scores P1, R1, R2 from the sealed bundle and compares against the FP-EARLYSTOP-001 v2
sealed bundle for R3 (numpy producer only, since v2 has no network producer). Cross-actor
checks (G1, P2, P3, X1) belong to the later cross-verification stage and are NOT scored here.

    python analyze.py --results <.../formal/task_results.json> \
        --v2-bundle <.../FP-EARLYSTOP-001/claude/formal_fv/task_results.json> \
        --output-dir <.../formal>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

TIE_EPS = 1e-9          # frozen tie epsilon (task sheet section 2)
VAL_TOL_NP = 1e-9       # R3 value tolerance
SUMMARY_TOL = 1e-6      # R3 arm-summary tolerance
GRID = {1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01}


def cells(bundle):
    """Yield (record, algorithm, arm, producer, payload)."""
    for rec in bundle["records"]:
        for algorithm, arms in rec["routes"].items():
            for arm, producers in arms.items():
                for producer, payload in producers.items():
                    yield rec, algorithm, arm, producer, payload


# --------------------------------------------------------------------------- #
# P1 -- safety failure detection
# --------------------------------------------------------------------------- #
def score_p1(bundle) -> dict[str, Any]:
    steps_eval = 0
    emitted = 0
    violations: list[dict] = []
    degrading: list[dict] = []
    margins: list[float] = []
    min_delta = None
    for rec, algorithm, arm, producer, payload in cells(bundle):
        for e in payload["steps"]:
            steps_eval += 1
            oa = e["oracle_audit"]
            if e["e_q"] is not None:
                margins.append(float(e["e_q"]) - float(oa["realized_q_sup_error_vs_current_target_pi"]))
            if oa["certificate_violation"]:
                violations.append({"record": [rec["mixing"], rec["task_index"]],
                                   "cell": [algorithm, arm, producer], "step": e["step"]})
            if e["emitted"]:
                emitted += 1
                dv = np.asarray(oa["value_delta_vs_previous"], float)
                if float(dv.min()) < -1e-12:
                    degrading.append({"record": [rec["mixing"], rec["task_index"]],
                                      "cell": [algorithm, arm, producer], "step": e["step"],
                                      "min_delta": float(dv.min())})
                if min_delta is None or float(dv.min()) < min_delta:
                    min_delta = float(dv.min())
    return {"steps_evaluated": steps_eval, "emitted_steps": emitted,
            "coverage_violations": len(violations),
            "componentwise_degrading_steps": len(degrading),
            "min_safety_margin": (min(margins) if margins else None),
            "min_value_delta": min_delta,
            "violations": violations[:20], "degrading": degrading[:20],
            "verdict": "PASS" if not violations and not degrading else "FALSIFIED"}


# --------------------------------------------------------------------------- #
# R1 -- the limited conclusion, per (producer) cell
# --------------------------------------------------------------------------- #
def arm_metrics(bundle, producer):
    """Per-record per-arm aggregates over all (env, algorithm) records."""
    per_rec = {}        # (mixing, task_index, algorithm) -> {arm: payload}
    for rec, algorithm, arm, prod, payload in cells(bundle):
        if prod != producer:
            continue
        per_rec.setdefault((rec["mixing"], rec["task_index"], algorithm), {})[arm] = payload
    return per_rec


def r1_for(bundle, producer):
    per_rec = arm_metrics(bundle, producer)
    records = sorted(per_rec)
    n = len(records)
    closure = {arm: [] for arm in ("conj_n16k", "conj_n64k", "perstate_n16k")}
    first_emit = {arm: 0 for arm in closure}
    win = loss = tie = 0
    rescue_from = rescue = 0
    for key in records:
        arms = per_rec[key]
        for arm in closure:
            closure[arm].append(arms[arm]["fraction_gap_closed"])
            steps = arms[arm]["steps"]
            if steps and steps[0]["emitted"]:
                first_emit[arm] += 1
        a, c = arms["conj_n16k"]["fraction_gap_closed"], arms["perstate_n16k"]["fraction_gap_closed"]
        if c > a + TIE_EPS:
            win += 1
        elif c < a - TIE_EPS:
            loss += 1
        else:
            tie += 1
        a_zero = arms["conj_n16k"]["emitted_steps"] == 0
        if a_zero:
            rescue_from += 1
            if arms["perstate_n16k"]["emitted_steps"] > 0:
                rescue += 1
    mean_closure = {arm: float(np.mean(v)) for arm, v in closure.items()}
    return {
        "records": n,
        "mean_closure": mean_closure,
        "closure_gap_C_minus_A": mean_closure["perstate_n16k"] - mean_closure["conj_n16k"],
        "first_step_emissions": first_emit,
        "win_count": win, "loss_count": loss, "tie_count": tie,
        "A_zero_emission_records": rescue_from, "rescued_by_C": rescue,
        "criteria": {
            "closure_gap >= 0.40": mean_closure["perstate_n16k"] - mean_closure["conj_n16k"] >= 0.40,
            "C_first >= 40": first_emit["perstate_n16k"] >= 40,
            "A_first <= 20": first_emit["conj_n16k"] <= 20,
            "C_first >= 2*A_first": first_emit["perstate_n16k"] >= 2 * first_emit["conj_n16k"],
            "win >= 40": win >= 40,
            "loss <= 2": loss <= 2,
            "rescue >= 25": rescue >= 25,
        },
    }


# --------------------------------------------------------------------------- #
# R2 -- network vs numpy decision identity, per (record, arm)
# --------------------------------------------------------------------------- #
def decision_seq(payload):
    return [(e["step"], e["emitted"], json.dumps(e["eta_selected"], sort_keys=True),
             e["states_updated"], tuple(e["ordered_reasons"])) for e in payload["steps"]]


def score_r2(bundle):
    mismatches = []
    n = 0
    for rec, algorithm, arm, producer, payload in cells(bundle):
        if producer != "numpy":
            continue
        other = rec["routes"][algorithm][arm]["network"]
        n += 1
        if decision_seq(payload) != decision_seq(other):
            mismatches.append({"record": [rec["mixing"], rec["task_index"]],
                               "algorithm": algorithm, "arm": arm})
    return {"compared": n, "mismatched": len(mismatches), "examples": mismatches[:20],
            "verdict": "PASS" if not mismatches and n else "FALSIFIED"}


# --------------------------------------------------------------------------- #
# R3 -- numpy producer vs the v2 sealed bundle
# --------------------------------------------------------------------------- #
def score_r3(bundle, v2):
    v2_by_key = {}
    for rec in v2["records"]:
        v2_by_key[(round(float(rec["mixing"]), 6), int(rec["task_index"]))] = rec
    field_bit = []
    field_tol = []
    summary_tol = []
    counts_bad = []
    n_compared = 0
    for rec in bundle["records"]:
        key = (round(float(rec["mixing"]), 6), int(rec["task_index"]))
        old = v2_by_key.get(key)
        if old is None:
            counts_bad.append({"key": key, "issue": "missing in v2 bundle"})
            continue
        for algorithm, arms in rec["routes"].items():
            old_route = old["routes"][algorithm]
            for arm, producers in arms.items():
                new_steps = producers["numpy"]["steps"]
                old_steps = old_route[arm]["steps"]
                if len(new_steps) != len(old_steps):
                    counts_bad.append({"key": key, "algorithm": algorithm, "arm": arm,
                                       "issue": f"step count {len(new_steps)} vs {len(old_steps)}"})
                for sn, so in zip(new_steps, old_steps):
                    n_compared += 1
                    for f in ("emitted", "states_updated", "min_pair_count_observed"):
                        if sn[f] != so[f]:
                            field_bit.append({"key": key, "algorithm": algorithm, "arm": arm,
                                              "step": sn["step"], "field": f,
                                              "new": sn[f], "old": so[f]})
                    if json.dumps(sn["eta_selected"], sort_keys=True) != json.dumps(
                            so["eta_selected"], sort_keys=True):
                        field_bit.append({"key": key, "algorithm": algorithm, "arm": arm,
                                          "step": sn["step"], "field": "eta_selected",
                                          "new": sn["eta_selected"], "old": so["eta_selected"]})
                    if tuple(sn["ordered_reasons"]) != tuple(so["ordered_reasons"]):
                        field_bit.append({"key": key, "algorithm": algorithm, "arm": arm,
                                          "step": sn["step"], "field": "ordered_reasons",
                                          "new": sn["ordered_reasons"], "old": so["ordered_reasons"]})
                    for f in ("e_q", "min_lb"):
                        a, b = sn[f], so[f]
                        if (a is None) != (b is None):
                            field_tol.append({"key": key, "field": f, "new": a, "old": b})
                        elif a is not None and abs(float(a) - float(b)) > VAL_TOL_NP:
                            field_tol.append({"key": key, "field": f, "new": a, "old": b})
                    ra = abs(float(sn["oracle_audit"]["realized_q_sup_error_vs_current_target_pi"])
                             - float(so["oracle_audit"]["realized_q_sup_error_vs_current_target_pi"]))
                    if ra > VAL_TOL_NP:
                        field_tol.append({"key": key, "field": "realized_error", "diff": ra})
                # arm summaries
                for f, tol in (("initial_v_sum", SUMMARY_TOL), ("final_v_sum", SUMMARY_TOL),
                               ("initial_suboptimality", SUMMARY_TOL),
                               ("fraction_gap_closed", SUMMARY_TOL)):
                    if abs(float(producers["numpy"][f]) - float(old_route[arm][f])) > tol:
                        summary_tol.append({"key": key, "algorithm": algorithm, "arm": arm,
                                            "field": f, "new": producers["numpy"][f],
                                            "old": old_route[arm][f]})
                if producers["numpy"]["emitted_steps"] != old_route[arm]["emitted_steps"]:
                    counts_bad.append({"key": key, "algorithm": algorithm, "arm": arm,
                                       "issue": "emitted_steps "
                                       f"{producers['numpy']['emitted_steps']} vs "
                                       f"{old_route[arm]['emitted_steps']}"})
    return {"steps_compared": n_compared,
            "bitwise_field_mismatches": field_bit[:20], "n_bitwise_mismatches": len(field_bit),
            "tolerance_field_mismatches": field_tol[:20], "n_tolerance_mismatches": len(field_tol),
            "summary_mismatches": summary_tol[:20], "n_summary_mismatches": len(summary_tol),
            "count_mismatches": counts_bad[:20],
            "verdict": "PASS" if not field_bit and not field_tol and not summary_tol
            and not counts_bad else "FALSIFIED"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--v2-bundle", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    bundle = json.loads(args.results.read_text(encoding="utf-8"))
    v2 = json.loads(args.v2_bundle.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p1 = score_p1(bundle)
    r1 = {p: r1_for(bundle, p) for p in ("numpy", "network")}
    for p in r1:
        r1[p]["verdict"] = "PASS" if all(r1[p]["criteria"].values()) else "FALSIFIED"
    r2 = score_r2(bundle)
    r3 = score_r3(bundle, v2)

    out = {"task_id": bundle["task_id"], "actor": bundle["actor"],
           "baseline": bundle["baseline"], "smoke": bundle.get("smoke", False),
           "P1": p1, "R1": r1, "R2": r2, "R3": r3,
           "note": "G1/P2/P3/X1 are cross-actor checks and belong to the cross-verification "
                   "stage after both routes seal; they are deliberately not scored here."}
    (args.output_dir / "analysis.json").write_text(json.dumps(out, indent=2, sort_keys=True),
                                                   encoding="utf-8")
    print(json.dumps({
        "P1": p1["verdict"],
        "R1_numpy": r1["numpy"]["verdict"], "R1_network": r1["network"]["verdict"],
        "R2": r2["verdict"], "R3": r3["verdict"],
        "closure_gap": {p: round(r1[p]["closure_gap_C_minus_A"], 4) for p in r1},
        "first_step": {p: r1[p]["first_step_emissions"] for p in r1},
        "rescue": {p: f"{r1[p]['rescued_by_C']}/{r1[p]['A_zero_emission_records']}" for p in r1},
    }, indent=2))


if __name__ == "__main__":
    main()
