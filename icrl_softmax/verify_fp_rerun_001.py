"""FP-RERUN-001 verifier: replay all eight cells, producers and `L12S` included.

Rebuilds each cell's policy chain from the recorded eta selections alone, re-runs the
SAME producer (array route or literal attention network) on the replayed policy,
re-draws the step's batch from the step-indexed schedule, and re-derives the
certificate, the decision, the exact values, the totals and the cost fields.

C1  the producer: replayed `q_hat` equals the sealed `q_hat`.
C2  emitted <=> E_Q < h, with h recomputed.
C3  E_Q >= the realized error of THAT cell's own producer.
C4  value deltas recomputed along the replayed chain; non-degradation re-checked.
C5  sealed totals: emitted_steps, simulated_steps, items_if_run_alone, stopped_at,
    total_value_gain.
C6  the L12S cells really used the registered `(split_fraction, delta_prop_fraction)`,
    and the split halves sum to the retained count.
C7  network vs numpy `q_hat` at the same replayed policy (float32 agreement).
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
        default=PROJECT / "results" / "FP-RERUN-001" / "claude" / "fresh_K16",
    )
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    bundle = json.loads((args.results / "task_results.json").read_text(encoding="utf-8"))
    cells = list(bundle["cells"])
    chains, length = int(bundle["chains"]), int(bundle["chain_length"])
    items_per_step = int(bundle["items_per_step"])
    delta_step = float(bundle["delta_step"])
    producers = {p: make_producer(p) for p in bundle["producers"]}

    emit_bad = cov = deg = total_bad = qhat_bad = cfg_bad = 0
    max_qhat = max_h = max_e_q = max_dv = max_gap = 0.0
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
                    max_qhat = max(max_qhat, d)
                    if producer == "network":
                        q_np = np.asarray(
                            producers["numpy"](route_name, current, train), dtype=np.float64
                        ).reshape(fs.N_STATES, fs.N_ACTIONS)
                        max_gap = max(max_gap, float(np.max(np.abs(q_hat - q_np))))
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
                        split_fraction=float(bundle["split_fraction"]),
                    )
                    if lever == "L12S" and cert.get("propagation"):
                        prop = cert["propagation"]
                        if abs(float(prop["split_fraction"]) - float(bundle["split_fraction"])) > 1e-15:
                            cfg_bad += 1
                        if abs(float(prop["delta_prop_fraction"])
                               - float(bundle["delta_prop_fraction"])) > 1e-15:
                            cfg_bad += 1
                        sa = np.asarray(prop["sizes_half_a"], dtype=np.int64)
                        sb = np.asarray(prop["sizes_half_b"], dtype=np.int64)
                        if not np.array_equal(sa + sb, np.asarray(cert["pair_sizes"])):
                            cfg_bad += 1
                        spent = float(prop["delta_eps"]) + (
                            float(prop["delta_per_iteration"]) * int(prop["n_iter"]) * sa.size
                            + float(prop["delta_final"]) * sa.size
                        )
                        if abs(spent - delta_step) > 1e-12:
                            cfg_bad += 1
                    decision = fs.improvement_for(current, q_hat, cert)
                    h = gate_margin(current, q_hat)
                    max_h = max(max_h, abs(h - float(entry["h"])))
                    if cert["e_q"] is not None and entry["e_q"] is not None:
                        max_e_q = max(max_e_q, abs(float(cert["e_q"]) - float(entry["e_q"])))
                    realized = float(np.max(np.abs(q_hat - q_ref)))
                    if cert["e_q"] is None or float(cert["e_q"]) < realized:
                        cov += 1
                    emitted = decision["status"] == "safe_update_emitted"
                    if emitted != bool(entry["emitted"]):
                        emit_bad += 1
                        continue
                    if emitted:
                        nxt = np.asarray(decision["policy_plus"], dtype=np.float64)
                        v_next = np.asarray(
                            policy_quantities(mdp, nxt)["v_pi"], dtype=np.float64
                        )
                        dv = v_next - v
                        if float(np.min(dv)) < -1e-12:
                            deg += 1
                        max_dv = max(max_dv, float(np.max(np.abs(
                            dv - np.asarray(entry["value_delta"], dtype=np.float64)
                        ))))
                        q_ref = np.asarray(
                            policy_quantities(mdp, nxt)["q_pi"], dtype=np.float64
                        )
                        current = nxt
                        v = v_next
                    steps_checked += 1
                r = route[cell]
                if r["emitted_steps"] != sum(1 for e in r["steps"] if e["emitted"]):
                    total_bad += 1
                if r["simulated_steps"] != len(r["steps"]):
                    total_bad += 1
                if r["items_if_run_alone"] != len(r["steps"]) * items_per_step:
                    total_bad += 1
                if abs(r["total_value_gain"] - float(np.sum(v) - np.sum(exact0["v_pi"]))) > 1e-9:
                    total_bad += 1
                if r["stopped_at"] is not None and int(r["stopped_at"]) != len(r["steps"]):
                    total_bad += 1

    report = {
        "results_dir": str(args.results),
        "records_verified": len(records),
        "steps_replayed": steps_checked,
        "C1_max_abs_delta_q_hat": max_qhat,
        "C1_producer_mismatches": qhat_bad,
        "C2_emission_mismatches": emit_bad,
        "C2_max_abs_delta_h": max_h,
        "C3_coverage_violations": cov,
        "C3_max_abs_delta_e_q": max_e_q,
        "C4_degradations": deg,
        "C4_max_abs_delta_value_delta": max_dv,
        "C5_total_mismatches": total_bad,
        "C6_l12s_config_mismatches": cfg_bad,
        "C7_max_abs_network_minus_numpy_q_hat": max_gap,
        "failure_count": emit_bad + cov + deg + total_bad + qhat_bad + cfg_bad
        + (1 if max_h > 1e-12 else 0) + (1 if max_e_q > 1e-12 else 0)
        + (1 if max_dv > 1e-9 else 0),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
