"""Review #2 cross-check: the PRE-REGISTERED fallback protocol (oracle kernel).

The FP-CERTFIX-001 task sheet, failure criterion 1, says that if the bridge lemma
cannot be proved under behavioural sampling, the work must switch to "drawing
iid samples per pair directly from the transition kernel P(.|s,a) (oracle
kernel)". The author instead adopted a third protocol (chain-replicated first
visit, lemma A'). This script runs the PRE-REGISTERED one as an independent
construction:

  * it needs no chains at all -- n iid successors per pair are drawn from
    P(.|s,a) and Y = r + gamma*Vhat(s') - Qhat(s,a) is formed directly;
  * the residual law is the same P_x (derivation section 0), so the estimand is
    identical and the two protocols must agree up to Monte-Carlo noise;
  * agreement is therefore genuine cross-validation of lemma A' by a different
    sampling mechanism (AGENTS.md section 5: independent construction).

Also reports the exact residual mean rho_x, so the reader can see how much of
each radius is Monte-Carlo slack versus bias.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

def _repo_root() -> Path:
    for cand in Path(__file__).resolve().parents:
        if (cand / "icrl_softmax" / "fixed_policy_expected_sarsa_scaled.py").exists():
            return cand
    raise RuntimeError("repository root not found")


ROOT = _repo_root()
PROJ = ROOT / "icrl_softmax"
sys.path.insert(0, str(PROJ))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_mp_certificate as mc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402

DELTA_TOTAL = 0.05
N_PER_PAIR = 65536
TASK_SALT = 90417


def exact_rho(mdp, policy, q_hat, gamma=fs.GAMMA):
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    V = (np.asarray(policy, dtype=np.float64) * np.asarray(q_hat, dtype=np.float64)).sum(axis=1)
    return ((P * (R + gamma * V[None, None, :])).sum(axis=2) - np.asarray(q_hat, float)).reshape(-1)


def oracle_batch(mdp, pair, n, rng, gamma=fs.GAMMA):
    """Direct iid draws from the transition kernel for one pair -- pre-registered fallback."""
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    s, a = divmod(pair, fs.N_ACTIONS)
    nxt = rng.choice(fs.N_STATES, size=n, p=P[s, a])
    return {
        "states": np.full(n, s, dtype=np.int64),
        "actions": np.full(n, a, dtype=np.int64),
        "rewards": R[s, a, nxt],
        "next_states": nxt.astype(np.int64),
    }


def main() -> int:
    print("pre-registered fallback (oracle-kernel iid sampling) vs lemma A' first-visit")
    print("=" * 96)
    rows = []
    fv_total = or_total = 0
    both = agree = 0
    for mixing in fs.MIXING:
        for task_index in range(fs.TASKS_PER_CELL):
            mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact0 = policy_quantities(mdp, behaviour)
            mu = np.asarray(exact0["mu_state"], dtype=np.float64)
            train = fs.training_batch(mdp, behaviour, mu, rng)
            q_pi = np.asarray(exact0["q_pi"], dtype=np.float64)
            for route in ("expected_exact", "expected_finite"):
                q_hat = np.asarray(
                    fs.run_route(route, behaviour, train)["q_hat"], dtype=np.float64
                ).reshape(fs.N_STATES, fs.N_ACTIONS)
                rho = exact_rho(mdp, behaviour, q_hat)

                # ---- pre-registered fallback: direct kernel draws, own stream ----
                rng_k = np.random.default_rng([fs.SEED, TASK_SALT, int(round(mixing * 100)),
                                               task_index, 991])
                batch = {
                    k: np.concatenate([oracle_batch(mdp, p, N_PER_PAIR, rng_k)[k]
                                       for p in range(fs.N_STATES * fs.N_ACTIONS)])
                    for k in ("states", "actions", "rewards", "next_states")
                }
                cert_or = mc.mp_certificate_firstvisit(
                    q_hat, behaviour, batch, min_visits=N_PER_PAIR, delta_step=DELTA_TOTAL
                )
                dec_or = fs.improvement_for(behaviour, q_hat, cert_or)
                or_emit = dec_or["status"] == "safe_update_emitted"

                # ---- lemma A' first visit, read from the sealed bundle ----
                sealed = json.loads(
                    (PROJ / "results" / "FP-CERTFIX-001" / "claude" / "step1_fv_c64k"
                     / "task_results.json").read_text(encoding="utf-8")
                )
                rec = next(
                    r for r in sealed["records"]
                    if r["mixing"] == mixing and r["task_index"] == task_index
                )
                s0 = rec["routes"][route]["mp"]["steps"][0]
                fv_emit = s0["status"] == "safe_update_emitted"
                means = np.asarray(s0["cert_means"], dtype=np.float64)
                radii = np.asarray(s0["cert_radii"], dtype=np.float64)
                e_q_fv = float(s0["e_q"])
                e_q_or = float(cert_or["e_q"]) if cert_or["status"] == "certificate_emitted" else None

                fv_total += int(fv_emit)
                or_total += int(or_emit)
                both += 1
                agree += int(fv_emit == or_emit)
                rows.append({
                    "mixing": mixing, "task_index": task_index, "route": route,
                    "fv_emit": bool(fv_emit), "oracle_emit": bool(or_emit),
                    "e_q_first_visit": e_q_fv, "e_q_oracle": e_q_or,
                    "max_abs_rho": float(np.max(np.abs(rho))),
                    "first_visit_mean_minus_rho": float(np.max(np.abs(means - rho))),
                    "oracle_mean_minus_rho": float(
                        np.max(np.abs(np.asarray(cert_or["residual_means"]) - rho))
                    )
                    if e_q_or is not None else None,
                    "first_visit_radius_max": float(np.max(radii)),
                    "oracle_radius_max": float(np.max(np.asarray(cert_or["radii"])))
                    if e_q_or is not None else None,
                })
    print(f"record-routes: {both}")
    print(f"emitted  first-visit (sealed, c64k) : {fv_total}/48")
    print(f"emitted  oracle kernel (pre-registered fallback, n=65536): {or_total}/48")
    print(f"emission decisions agreeing: {agree}/{both}")
    e_fv = np.array([r["e_q_first_visit"] for r in rows])
    e_or = np.array([r["e_q_oracle"] for r in rows if r["e_q_oracle"] is not None])
    print(f"E_Q median  first-visit {np.median(e_fv):.5f}   oracle {np.median(e_or):.5f}")
    dev = np.array([abs(r["e_q_first_visit"] - r["e_q_oracle"]) for r in rows
                    if r["e_q_oracle"] is not None])
    print(f"|E_Q_firstvisit - E_Q_oracle|: median {np.median(dev):.2e}  max {dev.max():.2e}")
    print()
    print("per-record E_Q comparison (first 12):")
    for r in rows[:12]:
        print(
            f"  mix={r['mixing']} task={r['task_index']:>2} {r['route']:<16} "
            f"E_Q {r['e_q_first_visit']:.5f} vs {r['e_q_oracle']:.5f}   "
            f"max|rho| {r['max_abs_rho']:.5f}   "
            f"emit fv/or {int(r['fv_emit'])}/{int(r['oracle_emit'])}"
        )
    out = {
        "n_per_pair": N_PER_PAIR, "delta_step": DELTA_TOTAL,
        "emitted_first_visit": fv_total, "emitted_oracle": or_total,
        "agree": agree, "records": both, "rows": rows,
    }
    (ROOT / "tmp" / "review2_oracle_kernel.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
