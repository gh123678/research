"""Diagnosis: where does the residual certificate's slack actually live?

EXPLORATORY, on already-sealed data (FP-CERTFIX-001 `step1_fv_c64k`). No new
measurement is claimed here; the point is to price each candidate lever before a
task sheet is written, exactly as FP-TIGHT-001's pilot did for its own levers.

The certified quantity is a scalar `E_Q` fed to the frozen decision rule
`LB_s = I_s - E_Q * ||dpi_s||_1`. Everything below is a sound replacement for
`E_Q`; none of them touches the decision rule, the eta grid, or the sealed files.

Current bound (derivation sections 3 and 5):

    E_Q = max_x ( |mean_x| + t_x ) / (1 - gamma)
    t_x = sqrt(2 V_x log(2/delta') / n_x) + (7/3) * R * log(2/delta') / (n_x - 1)
    R   = 2E = 20,  E = R* + gamma*B + B,  B = R*/(1-gamma) = 5

Three places where that is provably not tight, each with a different price:

A. THE ENVELOPE IS WORST-CASE OVER QHAT, BUT QHAT IS KNOWN.
   `E = R* + gamma*B + B` assumes only `|Qhat| <= B`. The actual array is an
   input to the certificate and is fully known, so `E` can be replaced, at zero
   probability cost, by

       E_eff = R* + gamma*||Vhat||_inf + ||Qhat||_inf,   Vhat = pi . Qhat

   which is a deterministic constant of known quantities. No estimation, no
   confidence correction, no change of assumptions: the same theorem with a
   smaller true range. (FP-RANGE-001 tried to shrink the range with a
   data-driven truncation and paid a Cauchy-Schwarz bias for it; this costs
   nothing because nothing is estimated.)

B. THE 1/(1-GAMMA) STEP THROWS AWAY THE PER-PAIR STRUCTURE.
   `Qhat - Q^pi = (I - gamma P^pi)^{-1} rho` is an IDENTITY, not an inequality.
   With `|rho_x| <= eps_x` per pair, the exact propagation is

       w = (I - gamma K)^{-1} epsbar,   K(s,s') = sum_a pi(a|s) P(s'|s,a),
       |u(s,a)| <= eps_x + gamma * sum_s' P(s'|s,a) w(s'),

   whose max is <= max_x eps_x / (1-gamma), with equality only when `eps` is
   constant across pairs. The gain is therefore the non-uniformity of `eps`.

C. THE ORACLE SUPPORT RANGE (reference only, needs the transition kernel).
   The support of `Y_x` is the finite set `{R(s,a,s') + gamma Vhat(s') -
   Qhat(s,a) : P(s'|s,a) > 0}`, so its exact range is computable from the MDP
   tables. This is the floor of the range lever, and it is quoted to show how
   much of A's gain is left on the table -- it changes the data-access claim, so
   it is not proposed as a deliverable.

Also reported: the same three quantities under the empirical-Bernstein
(single-sample Maurer-Pontil) certificate, which is the arm that cleared its band.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402

BUNDLE = PROJECT / "results" / "FP-CERTFIX-001" / "claude" / "step1_fv_c64k"
GAMMA = fs.GAMMA
R_STAR = 1.5
D = fs.N_STATES * fs.N_ACTIONS
DELTA_TOTAL = 0.05


def propagation_matrices(mdp, policy):
    """K (S x S) and (I - gamma K)^{-1}, from the known MDP tables."""
    P = np.asarray(mdp["P"], dtype=np.float64)
    K = np.einsum("sa,sat->st", np.asarray(policy, dtype=np.float64), P)
    inv = np.linalg.inv(np.eye(K.shape[0]) - GAMMA * K)
    return K, inv


def main() -> int:
    if not (BUNDLE / "task_results.json").exists():
        raise SystemExit(f"sealed bundle missing: {BUNDLE}")
    bundle = json.loads((BUNDLE / "task_results.json").read_text(encoding="utf-8"))
    delta_dir = DELTA_TOTAL / (2.0 * D)
    log_term = math.log(2.0 / delta_dir)

    rows = []
    for rec in bundle["records"]:
        mixing, task_index = float(rec["mixing"]), int(rec["task_index"])
        mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
        exact0 = policy_quantities(mdp, behaviour)
        mu = np.asarray(exact0["mu_state"], dtype=np.float64)
        train = fs.training_batch(mdp, behaviour, mu, rng)
        P = np.asarray(mdp["P"], dtype=np.float64)
        R = np.asarray(mdp["R"], dtype=np.float64)
        _, inv = propagation_matrices(mdp, behaviour)
        for route in bundle["routes"]:
            q_hat = np.asarray(
                fs.run_route(route, behaviour, train)["q_hat"], dtype=np.float64
            ).reshape(fs.N_STATES, fs.N_ACTIONS)
            s0 = rec["routes"][route]["mp"]["steps"][0]
            if s0["status"] != "safe_update_emitted" and s0.get("e_q") is None:
                continue
            means = np.asarray(s0["cert_means"], dtype=np.float64)
            radii = np.asarray(s0["cert_radii"], dtype=np.float64)
            vars_ = np.asarray(s0["cert_sample_vars"], dtype=np.float64)
            sizes = np.asarray(s0["cert_pair_sizes"], dtype=np.int64)
            e_q_now = float(s0["e_q"])
            realized = float(
                s0["oracle_audit"]["realized_q_sup_error_vs_current_target_pi"]
            )

            # ---- lever A: envelope from the KNOWN Qhat --------------------
            v_hat = (behaviour * q_hat).sum(axis=1)
            e_eff = R_STAR + GAMMA * float(np.max(np.abs(v_hat))) + float(
                np.max(np.abs(q_hat))
            )
            range_eff = 2.0 * e_eff
            t_a = np.sqrt(2.0 * np.maximum(vars_, 0.0) * log_term / sizes) + (
                7.0 / 3.0
            ) * range_eff * log_term / np.maximum(sizes - 1, 1)
            eps_a = np.abs(means) + t_a
            e_q_a = float(np.max(eps_a)) / (1.0 - GAMMA)

            # ---- lever B: A + exact propagation ---------------------------
            epsbar = (behaviour * eps_a.reshape(fs.N_STATES, fs.N_ACTIONS)).sum(axis=1)
            w = inv @ epsbar
            prop = GAMMA * (P * w[None, None, :]).sum(axis=2)
            e_q_b = float(np.max(eps_a.reshape(fs.N_STATES, fs.N_ACTIONS) + prop))

            # ---- lever C: A + oracle support range (reference) -----------
            v_hat_s = v_hat
            y_support = R + GAMMA * v_hat_s[None, None, :] - q_hat[:, :, None]
            supp = np.where(P > 0.0, y_support, np.nan)
            range_x = np.nanmax(supp, axis=2) - np.nanmin(supp, axis=2)
            t_c = np.sqrt(2.0 * np.maximum(vars_, 0.0) * log_term / sizes) + (
                7.0 / 3.0
            ) * range_x.reshape(-1) * log_term / np.maximum(sizes - 1, 1)
            eps_c = np.abs(means) + t_c
            e_q_c = float(np.max(eps_c)) / (1.0 - GAMMA)

            # ---- decomposition of the current E_Q -------------------------
            sqrt_now = np.sqrt(2.0 * np.maximum(vars_, 0.0) * log_term / sizes)
            lin_now = (7.0 / 3.0) * 20.0 * log_term / np.maximum(sizes - 1, 1)
            # ---- lever D: recover the half of the risk budget never spent --
            # MP thm 4 is TWO-SIDED at delta, so delta' = delta_step/(2d) spends
            # only d*delta' = delta_step/2. Using delta_step/d spends the budget
            # exactly.
            log_all = math.log(2.0 * D / DELTA_TOTAL)
            lin_d = (7.0 / 3.0) * range_eff * log_all / np.maximum(sizes - 1, 1)
            t_d = np.sqrt(2.0 * np.maximum(vars_, 0.0) * log_all / sizes) + lin_d
            eps_d = np.abs(means) + t_d
            epsbar_d = (behaviour * eps_d.reshape(fs.N_STATES, fs.N_ACTIONS)).sum(axis=1)
            prop_d = GAMMA * (P * (inv @ epsbar_d)[None, None, :]).sum(axis=2)
            e_q_d = float(np.max(eps_d.reshape(fs.N_STATES, fs.N_ACTIONS) + prop_d))

            rows.append(
                {
                    "mixing": mixing,
                    "task_index": task_index,
                    "route": route,
                    "e_q_now": e_q_now,
                    "e_q_a": e_q_a,
                    "e_q_b": e_q_b,
                    "e_q_c": e_q_c,
                    "e_q_d": e_q_d,
                    "realized": realized,
                    "ratio_now": e_q_now / realized,
                    "ratio_b": e_q_b / realized,
                    "envelope_now": 10.0,
                    "envelope_eff": e_eff,
                    "range_x_max": float(np.max(range_x)),
                    "range_x_min": float(np.min(range_x)),
                    "q_sup": float(np.max(np.abs(q_hat))),
                    "eps_max_over_min": float(np.max(eps_a) / max(np.min(eps_a), 1e-12)),
                    "mean_abs_mean": float(np.mean(np.abs(means))),
                    "mean_sqrt_term": float(np.mean(sqrt_now)),
                    "mean_linear_term": float(np.mean(lin_now)),
                    "max_reward": float(np.max(np.abs(np.asarray(mdp["R"], float)))),
                }
            )

    def summarise(key):
        vals = np.array([r[key] for r in rows])
        return float(np.mean(vals)), float(np.median(vals))

    print("FP-CERTFIX-001 bound-slack diagnosis (sealed step1_fv_c64k, 48 record-routes)")
    print("=" * 92)
    now = np.array([r["e_q_now"] for r in rows])
    a = np.array([r["e_q_a"] for r in rows])
    b = np.array([r["e_q_b"] for r in rows])
    c = np.array([r["e_q_c"] for r in rows])
    realized = np.array([r["realized"] for r in rows])
    print(f"  mean E_Q now          {now.mean():.5f}   median {np.median(now):.5f}")
    print(f"  A: known-Qhat range   {a.mean():.5f}   median {np.median(a):.5f}   "
          f"({a.mean() / now.mean() - 1:+.1%} vs now)")
    print(f"  B: A + exact propagation {b.mean():.5f}   median {np.median(b):.5f}   "
          f"({b.mean() / now.mean() - 1:+.1%} vs now)")
    print(f"  C: A + oracle range   {c.mean():.5f}   median {np.median(c):.5f}   "
          f"({c.mean() / now.mean() - 1:+.1%} vs now)")
    dd = np.array([r["e_q_d"] for r in rows])
    print(f"  D: A+B + full risk budget {dd.mean():.5f}   median {np.median(dd):.5f}   "
          f"({dd.mean() / now.mean() - 1:+.1%} vs now)")
    print()
    print("  decomposition of the per-pair residual bound (means over pairs):")
    print(f"    |mean_x|        {np.mean([r['mean_abs_mean'] for r in rows]):.5f}")
    print(f"    sqrt term       {np.mean([r['mean_sqrt_term'] for r in rows]):.5f}")
    print(f"    linear term(R=20) {np.mean([r['mean_linear_term'] for r in rows]):.5f}")
    print(f"    max |reward| in the frozen MDPs: "
          f"{max(r['max_reward'] for r in rows):.3f} (declared R* = {R_STAR})")
    print()
    print(f"  mean realized ||Qhat-Q^pi||_inf   {realized.mean():.5f}")
    print(f"  realized/E_Q  now {np.mean(realized / now):.4f}  "
          f"B {np.mean(realized / b):.4f}  C {np.mean(realized / c):.4f}")
    print()
    print(f"  envelope now 10.0 -> effective {np.mean([r['envelope_eff'] for r in rows]):.3f} "
          f"(min {min(r['envelope_eff'] for r in rows):.3f}, max {max(r['envelope_eff'] for r in rows):.3f})")
    print(f"  per-pair support range: min {min(r['range_x_min'] for r in rows):.3f}, "
          f"max {max(r['range_x_max'] for r in rows):.3f}   (vs the 20 used now)")
    print(f"  ||Qhat||_inf: min {min(r['q_sup'] for r in rows):.3f}, "
          f"max {max(r['q_sup'] for r in rows):.3f}")
    print(f"  eps non-uniformity max/min: median "
          f"{np.median([r['eps_max_over_min'] for r in rows]):.2f}")
    print()
    worse = int(np.sum(b > now + 1e-12))
    print(f"  record-routes where lever B is WORSE than now: {worse} (must be 0)")
    print(f"  record-routes where lever B exceeds the realized error: "
          f"{int(np.sum(b < realized))} (must be 0)")
    (ROOT / "tmp").mkdir(exist_ok=True)
    (ROOT / "tmp" / "bound_slack_diagnosis.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
