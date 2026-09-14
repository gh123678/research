"""FP-COMPOSE-001 / claude: section 6 high-precision boundary diagnosis.

The task sheet section 6 requires that, for any point whose safety margin is negative,
non-finite, or within 1e-10 of the boundary, the affected formal run is paused and the
point is recomputed in 80-bit and 120-bit decimal precision from the SAME sealed inputs
(not re-sampled, not re-tuned). This program is that instrument.

It recomputes, at both precisions:
  * the per-pair residual mean and ddof=1 variance from the sealed first-visit batch,
  * the L12 radius, eps_x and E_Q,
  * the per-state LB table and the chosen eta,
  * Q^pi and v^pi of the sealed pi_before / pi_after by an exact rational-free solve,
  * the realized sup error ||Qhat - Q^pi||inf and the value delta.

80 and 120 decimal places are used as required. Agreement at both precisions is NOT a
formal interval proof and the report says so.

    python diagnose_boundary.py --results <.../formal/task_results.json> \
        --family f1 --mixing 0.08 --task-index 200 --route expected_exact \
        --producer network --step 1 --output <.../boundary_diagnosis.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

import mpmath as mp  # noqa: E402
from evaluate_fp_xfam_001 import (  # noqa: E402
    FAMILIES, build_family_task, training_batch, vectorised_batch_generic,
)
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_certfix_first_n import first_visit_batch  # noqa: E402

ETA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


def mp_matrix(rows) -> mp.matrix:
    return mp.matrix([[mp.mpf(repr(float(x))) for x in row] for row in rows])


def mp_q_pi(P, R, policy, gamma):
    """Q^pi = (I - gamma P_pi)^-1 r_pi in the requested working precision."""
    S = len(policy)
    A = len(policy[0])
    n = S * A
    eye = mp.eye(n)
    M = mp.matrix(n, n)
    rhs = mp.matrix(n, 1)
    for s in range(S):
        for a in range(A):
            row = s * A + a
            acc = mp.mpf(0)
            for sp in range(S):
                for ap in range(A):
                    M[row, sp * A + ap] = -gamma * P[s][a][sp] * policy[sp][ap]
                    acc += P[s][a][sp] * R[s][a][sp]
            M[row, row] += 1
            rhs[row, 0] = acc
    sol = mp.lu_solve(M, rhs)
    Q = [[sol[s * A + a, 0] for a in range(A)] for s in range(S)]
    v = [sum(policy[s][a] * Q[s][a] for a in range(A)) for s in range(S)]
    return Q, v


def diagnose(mdp, behaviour, train, chains, chain_length, seed, salt, mixing, task_index,
             step, route, producer, sealed_step, fam, dps_list):
    P = [[[mp.mpf(repr(float(x))) for x in row] for row in mat]
         for mat in np.asarray(mdp["P"], dtype=np.float64)]
    R = [[[mp.mpf(repr(float(x))) for x in row] for row in mat]
         for mat in np.asarray(mdp["R"], dtype=np.float64)]
    gamma = mp.mpf(repr(float(fam["gamma"])))
    r_star = mp.mpf(repr(float(fam["reward_bound"])))
    S, A = fam["n_states"], fam["n_actions"]
    d = S * A
    delta_step = mp.mpf(repr(float(sealed_step["delta_step"])))

    # Rebuild the certification batch and its first-visit reduction from the frozen seeds.
    raw = vectorised_batch_generic(
        mdp, behaviour,
        np.asarray(policy_quantities(mdp, behaviour)["mu_state"], dtype=np.float64),
        [seed, salt, int(round(100 * mixing)), int(task_index), step],
        chains, fam, chain_length)
    reduced, counts = first_visit_batch(raw, chain_length, n_states=S, n_actions=A)

    q_hat_in = np.asarray(sealed_step["q_hat"], dtype=np.float64)
    pi_in = np.asarray(sealed_step["pi_before"], dtype=np.float64)

    out: dict[str, Any] = {"dps": {}, "replay_vs_sealed": {}}
    for dps in dps_list:
        mp.mp.dps = dps
        qh = mp_matrix(q_hat_in)
        pi = mp_matrix(pi_in)
        v_hat = [sum(pi[s, a] * qh[s, a] for a in range(A)) for s in range(S)]
        y_range = 2 * (r_star + gamma * max(abs(x) for x in v_hat)
                       + max(abs(qh[s, a]) for s in range(S) for a in range(A)))
        delta_dir = delta_step / d
        log_term = mp.log(2 / delta_dir)

        st = np.asarray(reduced["states"], np.int64)
        ac = np.asarray(reduced["actions"], np.int64)
        rw = np.asarray(reduced["rewards"], np.float64)
        ns = np.asarray(reduced["next_states"], np.int64)
        # residuals in high precision, from the sealed Qhat and the sealed pi_before
        resid = []
        for i in range(st.size):
            succ = sum(pi[ns[i], a] * qh[ns[i], a] for a in range(A))
            resid.append(mp.mpf(repr(float(rw[i]))) + gamma * succ - qh[st[i], ac[i]])

        flat = st * A + ac
        eps = []
        stats = []
        for x in range(d):
            y = [resid[i] for i in np.flatnonzero(flat == x)]
            n = len(y)
            mean = sum(y) / n if n else mp.mpf(0)
            var = (sum((t - mean) ** 2 for t in y) / (n - 1)) if n > 1 else mp.mpf(0)
            radius = (mp.sqrt(2 * var * log_term / n)
                      + mp.mpf(7) / 3 * y_range * log_term / max(n - 1, 1)) if n else mp.mpf(0)
            eps.append(abs(mean) + radius)
            stats.append({"n": n, "mean": mp.nstr(mean, 30), "var": mp.nstr(var, 30),
                          "radius": mp.nstr(radius, 30)})
        e_q = max(eps) / (1 - gamma)

        # LB table and decision at this precision
        pi_after = [row[:] for row in pi_in.tolist()]
        rows_eta = []
        table = []
        for s in range(S):
            row = []
            pick = None
            for eta in ETA_GRID:
                logits = [mp.log(pi[s, a]) + mp.mpf(repr(eta)) * qh[s, a] for a in range(A)]
                mx = max(logits)
                w = [mp.e ** (t - mx) for t in logits]
                tot = sum(w)
                cand = [wi / tot for wi in w]
                delta = [cand[a] - pi[s, a] for a in range(A)]
                lb = sum(delta[a] * qh[s, a] for a in range(A)) - e_q * sum(
                    abs(t) for t in delta)
                row.append(mp.nstr(lb, 30))
                if pick is None and lb > 0:
                    pick = float(eta)
                    pi_after[s] = [float(c) for c in cand]
            table.append(row)
            rows_eta.append(pick)

        q1, v1 = mp_q_pi(P, R, [[mp.mpf(repr(c)) for c in row] for row in pi_after], gamma)
        Q0, V0 = mp_q_pi(P, R, [[pi[s, a] for a in range(A)] for s in range(S)], gamma)
        realized = max(abs(qh[s, a] - Q0[s][a]) for s in range(S) for a in range(A))
        margin = e_q - realized
        dv = [v1[s] - V0[s] for s in range(S)]

        out["dps"][str(dps)] = {
            "y_range": mp.nstr(y_range, 30),
            "delta_dir": mp.nstr(delta_dir, 30),
            "log_term": mp.nstr(log_term, 30),
            "e_q": mp.nstr(e_q, 30),
            "realized_sup_error": mp.nstr(realized, 30),
            "safety_margin_recomputed": mp.nstr(margin, 30),
            "safety_margin_sealed": sealed_step["audit"]["safety_margin"],
            "rows_eta_recomputed": rows_eta,
            "rows_eta_sealed": sealed_step["rows_eta"],
            "lb_table_recomputed": table,
            "value_delta_recomputed": [mp.nstr(x, 30) for x in dv],
            "min_value_delta": mp.nstr(min(dv), 30),
            "pair_stats": stats,
            "coverage_holds": bool(margin >= 0),
            "componentwise_nondegrading": bool(min(dv) >= 0),
        }
    # decimal-place agreement between the two precisions
    keys = ["e_q", "safety_margin_recomputed", "min_value_delta"]
    agree = {}
    a, b = (out["dps"][str(dps_list[0])], out["dps"][str(dps_list[-1])])
    for k in keys:
        agree[k] = mp.nstr(abs(mp.mpf(a[k]) - mp.mpf(b[k])), 10)
    out["agreement_between_precisions"] = agree
    out["limitation"] = ("Agreement at 80 and 120 decimal places is a strong numerical "
                         "consistency check, NOT a formal interval proof. It does not "
                         "upgrade the result to a verified theorem.")
    out["location"] = {"family": None, "mixing": mixing, "task_index": task_index,
                       "route": route, "producer": producer, "step": step,
                       "chains": chains, "chain_length": chain_length, "seed": seed,
                       "salt": salt}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--mixing", type=float, required=True)
    ap.add_argument("--task-index", type=int, required=True)
    ap.add_argument("--route", required=True)
    ap.add_argument("--producer", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--dps", type=int, nargs="+", default=[80, 120])
    args = ap.parse_args()

    bundle = json.loads(args.results.read_text(encoding="utf-8"))
    fam = FAMILIES[args.family]
    chains, chain_length = int(bundle["chains"]), int(bundle["chain_length"])
    seed, salt = int(bundle["seed"]), int(bundle["salt"])
    mdp, behaviour, rng = build_family_task(fam, args.mixing, args.task_index)

    sealed_step = None
    for rec in bundle["records"]:
        if (rec["family"] == args.family and abs(float(rec["mixing"]) - args.mixing) < 1e-12
                and int(rec["task_index"]) == args.task_index):
            cell = rec["routes"][args.route][f"{args.producer}|perstate|L12"]
            for e in cell["steps"]:
                if int(e["step"]) == args.step:
                    sealed_step = e
    if sealed_step is None:
        raise SystemExit("the requested sealed step was not found in the bundle")

    out = diagnose(mdp, behaviour, training_batch(
        mdp, behaviour,
        np.asarray(policy_quantities(mdp, behaviour)["mu_state"], dtype=np.float64), rng),
        chains, chain_length, seed, salt, args.mixing, args.task_index, args.step,
        args.route, args.producer, sealed_step, fam, args.dps)
    out["location"]["family"] = args.family
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({d: {k: out["dps"][d][k] for k in
                          ("safety_margin_recomputed", "min_value_delta", "coverage_holds",
                           "componentwise_nondegrading")} for d in map(str, args.dps)},
                     indent=2))


if __name__ == "__main__":
    main()
