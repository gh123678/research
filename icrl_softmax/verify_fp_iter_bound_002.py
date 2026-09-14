"""FP-ITER-BOUND-002 verifier: replay all four (arm, K) cells from the bundle.

Same construction as `verify_fp_iter_bound_001.py` (rebuild each arm's policy
chain from the recorded eta selections, re-derive q_hat, the certificate, the
decision and the exact values), extended to the `(lever, horizon)` spec: the
per-step risk is `delta_total / K` for that cell, so the replay also checks that
the right budget was used at the right step.

C1  emitted <=> E_Q < h, with h recomputed, per cell.
C2  E_Q >= realized ||Qhat - Q^pi||_inf at every step of every cell.
C3  value deltas recomputed along the replayed chain; non-degradation re-checked.
C4  sealed totals (emitted_steps, total_value_gain, stopped_at, truncated).
C5  the per-step risk equals delta_total / K for the cell, and the risk split of
    the L12M cells sums to that budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa as es  # noqa: E402
import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_tight_certificate as tc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402

D = 12


def gate_margin(policy, q_hat):
    best = 0.0
    for eta in fs.ETA_CANDIDATES:
        cand = es.relative_softmax_candidate(policy, q_hat, eta)
        dpi = cand - policy
        i_hat = (dpi * q_hat).sum(axis=1)
        l1 = np.abs(dpi).sum(axis=1)
        ratios = np.where(l1 > 0.0, i_hat / np.where(l1 > 0.0, l1, 1.0), 0.0)
        best = max(best, float(np.min(ratios)))
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT / "results" / "FP-ITER-BOUND-002" / "claude" / "fresh_k4k8k16",
    )
    args = parser.parse_args()
    bundle = json.loads((args.results / "task_results.json").read_text(encoding="utf-8"))
    spec = bundle["arm_spec"]
    labels = list(bundle["arms"])
    chains, length = int(bundle["chains"]), int(bundle["chain_length"])
    delta_total = float(bundle["delta_total"])

    emit_mismatch = viol = degrade = risk_bad = total_bad = 0
    max_h_delta = max_e_q_delta = max_dv = max_gain_delta = 0.0
    steps_checked = 0

    for rec in bundle["records"]:
        mixing, task_index = float(rec["mixing"]), int(rec["task_index"])
        mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
        exact0 = policy_quantities(mdp, behaviour)
        mu = np.asarray(exact0["mu_state"], dtype=np.float64)
        train = fs.training_batch(mdp, behaviour, mu, rng)
        P = np.asarray(mdp["P"], dtype=np.float64)
        R = np.asarray(mdp["R"], dtype=np.float64)
        for route_name, route in rec["routes"].items():
            for lb in labels:
                lever = str(spec[lb]["lever"])
                K = int(spec[lb]["horizon"])
                delta_step = delta_total / K
                current = behaviour.copy()
                v = np.asarray(exact0["v_pi"], dtype=np.float64).copy()
                q_ref = np.asarray(exact0["q_pi"], dtype=np.float64).copy()
                for idx, entry in enumerate(route[lb]["steps"]):
                    step = idx + 1
                    if entry["step"] != step or abs(float(entry["delta_step"]) - delta_step) > 1e-15:
                        risk_bad += 1
                    q_hat = np.asarray(
                        fs.run_route(route_name, current, train)["q_hat"], dtype=np.float64
                    ).reshape(fs.N_STATES, fs.N_ACTIONS)
                    raw = vectorised_batch(
                        mdp, behaviour, mu,
                        step_seed_parts(fs.SEED, 90417, mixing, task_index, step),
                        chains, length,
                    )
                    reduced, _ = first_visit_batch(raw, length)
                    cert = tc.certificate(
                        q_hat, current, reduced, min_visits=int(bundle["min_visits"]),
                        delta_step=delta_step, lever=lever, transition=P, reward=R,
                        delta_prop_fraction=float(bundle["delta_prop_fraction"]),
                    )
                    if cert.get("propagation"):
                        prop = cert["propagation"]
                        spent = float(prop["delta_eps"]) + (
                            float(prop["delta_per_iteration"]) * int(prop["n_iter"]) * D
                            + float(prop["delta_final"]) * D
                        )
                        if abs(spent - delta_step) > 1e-12:
                            risk_bad += 1
                    decision = fs.improvement_for(current, q_hat, cert)
                    h = gate_margin(current, q_hat)
                    max_h_delta = max(max_h_delta, abs(h - float(entry["h"])))
                    if cert["e_q"] is not None and entry["e_q"] is not None:
                        max_e_q_delta = max(
                            max_e_q_delta, abs(float(cert["e_q"]) - float(entry["e_q"]))
                        )
                    realized = float(np.max(np.abs(q_hat - q_ref)))
                    if cert["e_q"] is None or float(cert["e_q"]) < realized:
                        viol += 1
                    emitted = decision["status"] == "safe_update_emitted"
                    if emitted != bool(entry["emitted"]):
                        emit_mismatch += 1
                        continue
                    if emitted:
                        nxt = np.asarray(decision["policy_plus"], dtype=np.float64)
                        v_next = np.asarray(policy_quantities(mdp, nxt)["v_pi"], dtype=np.float64)
                        dv = v_next - v
                        if float(np.min(dv)) < -1e-12:
                            degrade += 1
                        max_dv = max(
                            max_dv,
                            float(np.max(np.abs(dv - np.asarray(entry["value_delta"], dtype=np.float64)))),
                        )
                        q_ref = np.asarray(policy_quantities(mdp, nxt)["q_pi"], dtype=np.float64)
                        current = nxt
                        v = v_next
                    steps_checked += 1
                if route[lb]["emitted_steps"] != sum(1 for e in route[lb]["steps"] if e["emitted"]):
                    total_bad += 1
                if abs(route[lb]["total_value_gain"]
                       - float(np.sum(v) - np.sum(exact0["v_pi"]))) > 1e-9:
                    total_bad += 1
                    max_gain_delta = max(
                        max_gain_delta,
                        abs(route[lb]["total_value_gain"]
                            - float(np.sum(v) - np.sum(exact0["v_pi"]))),
                    )
                if route[lb]["truncated"] != (route[lb]["stopped_at"] is None):
                    total_bad += 1

    report = {
        "results_dir": str(args.results),
        "steps_replayed": steps_checked,
        "C1_emission_mismatches": emit_mismatch,
        "C1_max_abs_delta_h": max_h_delta,
        "C2_coverage_violations": viol,
        "C2_max_abs_delta_e_q": max_e_q_delta,
        "C3_degradations": degrade,
        "C3_max_abs_delta_value_delta": max_dv,
        "C4_total_mismatches": total_bad,
        "C4_max_abs_delta_total_gain": max_gain_delta,
        "C5_risk_schedule_bad": risk_bad,
        "failure_count": emit_mismatch + viol + degrade + total_bad + risk_bad
        + (1 if max_h_delta > 1e-12 else 0)
        + (1 if max_e_q_delta > 1e-12 else 0)
        + (1 if max_dv > 1e-9 else 0),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
