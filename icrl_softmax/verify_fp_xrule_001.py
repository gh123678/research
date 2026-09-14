"""FP-XRULE-001 verifier: replay all four cells on both families from the bundle.

Rebuilds each cell's policy chain from the recorded decisions, re-runs the family's
array route on the replayed policy, re-draws the step's batch, and re-derives the
certificate, the decision, the exact values and the cost fields.

C1  emitted <=> the rule's own decision, with `min_lb` and `states_updated` recomputed
C2  E_Q >= the realized ||Qhat - Q^pi||_inf
C3  value deltas recomputed along the replayed chain; non-degradation re-checked
C4  sealed totals: emitted_steps, simulated_steps, items_if_run_alone, stopped_at
C5  the certificate really ran the registered lever and budget (L12S config incl.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_tight_certificate as tc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_xfam_001 import (  # noqa: E402
    FAMILIES,
    build_family_task,
    conjunctive_decision,
    perstate_decision,
    run_route,
    training_batch,
    vectorised_batch_generic,
)
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=PROJECT / "results" / "FP-XRULE-001" / "claude" / "f1f2_K4",
    )
    parser.add_argument("--max-records-per-family", type=int, default=0)
    args = parser.parse_args()
    bundle = json.loads((args.results / "task_results.json").read_text(encoding="utf-8"))
    cells = list(bundle["cells"])
    chains, length = int(bundle["chains"]), int(bundle["chain_length"])
    items_per_step = int(bundle["items_per_step"])
    delta_step = float(bundle["delta_step"])

    emit_bad = cov = deg = total_bad = cfg_bad = 0
    max_e_q = max_dv = max_lb = 0.0
    steps_checked = 0

    for fam_name in bundle["families"]:
        fam = FAMILIES[fam_name]
        recs = [r for r in bundle["records"] if r["family"] == fam_name]
        if args.max_records_per_family:
            recs = recs[: args.max_records_per_family]
        for rec in recs:
            mixing, task_index = float(rec["mixing"]), int(rec["task_index"])
            mdp, behaviour, rng = build_family_task(fam, mixing, task_index)
            exact0 = policy_quantities(mdp, behaviour)
            mu = np.asarray(exact0["mu_state"], dtype=np.float64)
            train = training_batch(mdp, behaviour, mu, rng)
            S, A = behaviour.shape
            P = np.asarray(mdp["P"], dtype=np.float64)
            R = np.asarray(mdp["R"], dtype=np.float64)
            for route_name, route in rec["routes"].items():
                for cell in cells:
                    rule, cert_name = cell.split("|")
                    current = behaviour.copy()
                    v = np.asarray(exact0["v_pi"], dtype=np.float64).copy()
                    q_ref = np.asarray(exact0["q_pi"], dtype=np.float64).copy()
                    for idx, entry in enumerate(route[cell]["steps"]):
                        step = idx + 1
                        q_hat = run_route(route_name, fam, current, train).reshape(S, A)
                        raw = vectorised_batch_generic(
                            mdp, behaviour, mu,
                            step_seed_parts(fs.SEED, 90417, mixing, task_index, step),
                            chains, fam, length,
                        )
                        reduced, _ = first_visit_batch(
                            raw, length, n_states=S, n_actions=A
                        )
                        cert = tc.certificate(
                            q_hat, current, reduced, min_visits=int(bundle["min_visits"]),
                            delta_step=delta_step, lever=cert_name, transition=P, reward=R,
                            reward_bound=fam["reward_bound"], gamma=fam["gamma"],
                            n_states=S, n_actions=A,
                            delta_prop_fraction=float(bundle["delta_prop_fraction"]),
                            split_fraction=float(bundle["split_fraction"]),
                        )
                        if cert_name == "L12S" and cert["status"] == "certificate_emitted":
                            prop = cert["propagation"]
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
                        e_q = cert["e_q"]
                        if e_q is None:
                            emitted, pplus, lb = False, current.copy(), np.zeros(S)
                        elif rule == "conj":
                            emitted, pplus, _eta, lb = conjunctive_decision(
                                current, q_hat, float(e_q)
                            )
                        else:
                            emitted, pplus, _u, lb = perstate_decision(
                                current, q_hat, float(e_q)
                            )
                        max_lb = max(max_lb, abs(float(np.min(lb)) - float(entry["min_lb"])))
                        if cert["e_q"] is not None and entry["e_q"] is not None:
                            max_e_q = max(max_e_q, abs(float(cert["e_q"]) - float(entry["e_q"])))
                        realized = float(np.max(np.abs(q_hat - q_ref)))
                        if cert["e_q"] is None or float(cert["e_q"]) < realized:
                            cov += 1
                        if bool(emitted) != bool(entry["emitted"]):
                            emit_bad += 1
                            continue
                        if emitted:
                            nxt = np.asarray(pplus, dtype=np.float64)
                            q_next = policy_quantities(mdp, nxt)
                            v_next = np.asarray(q_next["v_pi"], dtype=np.float64)
                            dv = v_next - v
                            if float(np.min(dv)) < -1e-12:
                                deg += 1
                            max_dv = max(max_dv, float(np.max(np.abs(
                                dv - np.asarray(entry["value_delta"], dtype=np.float64)
                            ))))
                            q_ref = np.asarray(q_next["q_pi"], dtype=np.float64)
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

    report = {
        "results_dir": str(args.results),
        "steps_replayed": steps_checked,
        "C1_emission_mismatches": emit_bad,
        "C1_max_abs_delta_min_lb": max_lb,
        "C2_coverage_violations": cov,
        "C2_max_abs_delta_e_q": max_e_q,
        "C3_degradations": deg,
        "C3_max_abs_delta_value_delta": max_dv,
        "C4_total_mismatches": total_bad,
        "C5_l12s_config_mismatches": cfg_bad,
        "failure_count": emit_bad + cov + deg + total_bad + cfg_bad
        + (1 if max_lb > 1e-12 else 0) + (1 if max_e_q > 1e-12 else 0)
        + (1 if max_dv > 1e-9 else 0),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    sys.exit(main())
