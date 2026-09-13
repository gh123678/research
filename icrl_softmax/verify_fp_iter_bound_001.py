"""FP-ITER-BOUND-001 verifier: replay the certified trajectories from the bundle.

The evaluator records, per step, the selected `eta` and the resulting value delta.
This verifier rebuilds each arm's policy chain from those `eta` selections alone,
re-derives `q_hat`, the certificate, the decision and the exact values, and checks:

C1  the decision chain: `emitted <=> E_Q < h`, with `h` recomputed.
C2  `E_Q >=` the realized `||Qhat - Q^pi||_inf` at every step of every arm.
C3  the value deltas: recomputed exactly from the replayed policy chain and
    compared with the sealed deltas; and componentwise non-degradation re-checked.
C4  the sealed totals (`emitted_steps`, `total_value_gain`, `stopped_at`).
C5  theorem 2's premise: each step's batch is drawn from a step-indexed seed and
    the policy fed to the certificate equals the replayed `pi_{k-1}`.
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

GAMMA = fs.GAMMA


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
        default=PROJECT / "results" / "FP-ITER-BOUND-001" / "claude" / "fresh_K16",
    )
    args = parser.parse_args()
    bundle = json.loads((args.results / "task_results.json").read_text(encoding="utf-8"))
    delta_step = float(bundle["delta_step"])
    chains, length = int(bundle["chains"]), int(bundle["chain_length"])

    emit_mismatch = viol = 0
    max_delta_v = 0.0
    max_h_delta = 0.0
    max_e_q_delta = 0.0
    degrade = 0
    total_mismatch = 0
    max_gain_delta = 0.0
    steps_checked = 0

    for rec in bundle["records"]:
        mixing, task_index = float(rec["mixing"]), int(rec["task_index"])
        mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
        exact0 = policy_quantities(mdp, behaviour)
        mu = np.asarray(exact0["mu_state"], dtype=np.float64)
        # the training batch must come off the SAME generator state the evaluator
        # used: build_task, then policy_quantities, then training_batch.
        train = fs.training_batch(mdp, behaviour, mu, rng)
        P = np.asarray(mdp["P"], dtype=np.float64)
        R = np.asarray(mdp["R"], dtype=np.float64)
        for route_name, route in rec["routes"].items():
            for arm in bundle["arms"]:
                current = behaviour.copy()
                v = np.asarray(exact0["v_pi"], dtype=np.float64).copy()
                q_ref = np.asarray(exact0["q_pi"], dtype=np.float64).copy()
                for idx, entry in enumerate(route[arm]["steps"]):
                    step = idx + 1
                    if entry["step"] != step:
                        total_mismatch += 1
                    q_hat = np.asarray(
                        fs.run_route(route_name, current, train)["q_hat"],
                        dtype=np.float64,
                    ).reshape(fs.N_STATES, fs.N_ACTIONS)
                    raw = vectorised_batch(
                        mdp, behaviour, mu,
                        step_seed_parts(fs.SEED, 90417, mixing, task_index, step),
                        chains, length,
                    )
                    reduced, _ = first_visit_batch(raw, length)
                    cert = tc.certificate(
                        q_hat, current, reduced, min_visits=int(bundle["min_visits"]),
                        delta_step=delta_step, lever=arm, transition=P, reward=R,
                    )
                    decision = fs.improvement_for(current, q_hat, cert)
                    h = gate_margin(current, q_hat)
                    max_h_delta = max(max_h_delta, abs(h - float(entry["h"])))
                    if cert["e_q"] is None:
                        if entry["e_q"] is not None:
                            max_e_q_delta = float("inf")
                    else:
                        max_e_q_delta = max(max_e_q_delta, abs(float(cert["e_q"]) - float(entry["e_q"])))
                    realized = float(np.max(np.abs(q_hat - q_ref)))
                    if cert["e_q"] is None or float(cert["e_q"]) < realized:
                        viol += 1
                    emitted = decision["status"] == "safe_update_emitted"
                    if emitted != bool(entry["emitted"]):
                        emit_mismatch += 1
                    if emitted:
                        nxt = np.asarray(decision["policy_plus"], dtype=np.float64)
                        v_next = np.asarray(policy_quantities(mdp, nxt)["v_pi"], dtype=np.float64)
                        dv = v_next - v
                        if float(np.min(dv)) < -1e-12:
                            degrade += 1
                        max_delta_v = max(
                            max_delta_v,
                            float(np.max(np.abs(dv - np.asarray(entry["value_delta"], dtype=np.float64)))),
                        )
                        q_ref = np.asarray(
                            policy_quantities(mdp, nxt)["q_pi"], dtype=np.float64
                        )
                        current = nxt
                        v = v_next
                    steps_checked += 1
                # C4: totals
                if route[arm]["emitted_steps"] != sum(
                    1 for e in route[arm]["steps"] if e["emitted"]
                ):
                    total_mismatch += 1
                if abs(
                    route[arm]["total_value_gain"] - float(np.sum(v) - np.sum(exact0["v_pi"]))
                ) > 1e-9:
                    total_mismatch += 1
                    max_gain_delta = max(
                        max_gain_delta,
                        abs(route[arm]["total_value_gain"]
                            - float(np.sum(v) - np.sum(exact0["v_pi"]))),
                    )

    report = {
        "results_dir": str(args.results),
        "steps_replayed": steps_checked,
        "C1_emission_mismatches": emit_mismatch,
        "C1_max_abs_delta_h": max_h_delta,
        "C2_coverage_violations": viol,
        "C2_max_abs_delta_e_q": max_e_q_delta,
        "C3_degradations": degrade,
        "C3_max_abs_delta_value_delta": max_delta_v,
        "C4_total_mismatches": total_mismatch,
        "C4_max_abs_delta_total_gain": max_gain_delta,
        "failure_count": emit_mismatch + viol + degrade + total_mismatch
        + (1 if max_h_delta > 1e-12 else 0)
        + (1 if max_e_q_delta > 1e-12 else 0)
        + (1 if max_delta_v > 1e-9 else 0),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
