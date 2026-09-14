"""FP-NET-BOUND-001 verifier: replay all eight cells, producers included.

The evaluator stores each step's `q_hat`. This verifier rebuilds every cell's
policy chain from the recorded eta selections alone, re-runs the SAME producer
(numpy array route or the literal attention network) on the replayed policy,
re-draws the step's certification batch from the step-indexed schedule, and
re-derives the certificate, the decision, the exact values and the totals.

C1  the producer: the replayed `q_hat` equals the sealed `q_hat` (so the network
    was run as recorded, on the right policy, at the right step).
C2  emitted <=> E_Q < h, with h recomputed.
C3  E_Q >= the realized error of THAT cell's own producer.
C4  value deltas recomputed along the replayed chain; non-degradation re-checked.
C5  sealed totals (emitted_steps, total_value_gain, stopped_at).
C6  numpy vs network `q_hat` at the same replayed policy (float32 agreement).
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
from evaluate_fp_certfix_001 import make_producer  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402


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
        default=PROJECT / "results" / "FP-NET-BOUND-001" / "claude" / "fresh_K16",
    )
    parser.add_argument("--max-records", type=int, default=0,
                        help="0 = all records; otherwise verify the first N (a subset run)")
    args = parser.parse_args()
    bundle = json.loads((args.results / "task_results.json").read_text(encoding="utf-8"))
    cells = list(bundle["cells"])
    chains, length = int(bundle["chains"]), int(bundle["chain_length"])
    delta_step = float(bundle["delta_step"])
    producers = {p: make_producer(p) for p in bundle["producers"]}

    emit_mismatch = viol = degrade = total_bad = qhat_bad = 0
    max_qhat_delta = max_h_delta = max_e_q_delta = max_dv = max_producer_gap = 0.0
    steps_checked = 0

    records = bundle["records"]
    if args.max_records:
        records = records[: args.max_records]
    for rec in records:
        mixing, task_index = float(rec["mixing"]), int(rec["task_index"])
        mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
        exact0 = policy_quantities(mdp, behaviour)
        mu = np.asarray(exact0["mu_state"], dtype=np.float64)
        train = fs.training_batch(mdp, behaviour, mu, rng)
        P = np.asarray(mdp["P"], dtype=np.float64)
        R = np.asarray(mdp["R"], dtype=np.float64)
        for route_name, route in rec["routes"].items():
            for cell in cells:
                producer, lever = cell.split("|")
                current = behaviour.copy()
                v = np.asarray(exact0["v_pi"], dtype=np.float64).copy()
                q_ref = np.asarray(exact0["q_pi"], dtype=np.float64).copy()
                for idx, entry in enumerate(route[cell]["steps"]):
                    step = idx + 1
                    q_hat = np.asarray(
                        producers[producer](route_name, current, train), dtype=np.float64
                    ).reshape(fs.N_STATES, fs.N_ACTIONS)
                    stored = np.asarray(entry["q_hat"], dtype=np.float64).reshape(
                        fs.N_STATES, fs.N_ACTIONS
                    )
                    d = float(np.max(np.abs(q_hat - stored)))
                    if d > 1e-9:
                        qhat_bad += 1
                    max_qhat_delta = max(max_qhat_delta, d)
                    if producer == "network":
                        q_np = np.asarray(
                            producers["numpy"](route_name, current, train), dtype=np.float64
                        ).reshape(fs.N_STATES, fs.N_ACTIONS)
                        max_producer_gap = max(
                            max_producer_gap, float(np.max(np.abs(q_hat - q_np)))
                        )
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
                        v_next = np.asarray(
                            policy_quantities(mdp, nxt)["v_pi"], dtype=np.float64
                        )
                        dv = v_next - v
                        if float(np.min(dv)) < -1e-12:
                            degrade += 1
                        max_dv = max(
                            max_dv,
                            float(np.max(np.abs(
                                dv - np.asarray(entry["value_delta"], dtype=np.float64)
                            ))),
                        )
                        q_ref = np.asarray(
                            policy_quantities(mdp, nxt)["q_pi"], dtype=np.float64
                        )
                        current = nxt
                        v = v_next
                    steps_checked += 1
                if route[cell]["emitted_steps"] != sum(
                    1 for e in route[cell]["steps"] if e["emitted"]
                ):
                    total_bad += 1
                if abs(route[cell]["total_value_gain"]
                       - float(np.sum(v) - np.sum(exact0["v_pi"]))) > 1e-9:
                    total_bad += 1
                if route[cell]["stopped_at"] != (
                    None if len(route[cell]["steps"]) == int(bundle["horizon"]) else len(route[cell]["steps"])
                ):
                    # stopped_at is the failing step; a truncated cell has stopped_at None
                    if not (route[cell]["stopped_at"] is None
                            and len(route[cell]["steps"]) == int(bundle["horizon"])):
                        total_bad += 1

    report = {
        "results_dir": str(args.results),
        "records_verified": len(records),
        "steps_replayed": steps_checked,
        "C1_max_abs_delta_q_hat": max_qhat_delta,
        "C1_producer_mismatches": qhat_bad,
        "C2_emission_mismatches": emit_mismatch,
        "C2_max_abs_delta_h": max_h_delta,
        "C3_coverage_violations": viol,
        "C3_max_abs_delta_e_q": max_e_q_delta,
        "C4_degradations": degrade,
        "C4_max_abs_delta_value_delta": max_dv,
        "C5_total_mismatches": total_bad,
        "C6_max_abs_network_minus_numpy_q_hat": max_producer_gap,
        "failure_count": emit_mismatch + viol + degrade + total_bad + qhat_bad
        + (1 if max_h_delta > 1e-12 else 0)
        + (1 if max_e_q_delta > 1e-12 else 0)
        + (1 if max_dv > 1e-9 else 0),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
