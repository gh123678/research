"""Independent review #2 of FP-CERTFIX-001: is the repaired lemma A' actually true?

Review #1 (docs/research_branches/FP-CERTFIX-001/claude/review_of_derivation.md)
falsified the ORIGINAL lemma A (first-n fixed-count extraction). The author then
repaired it as lemma A' (one first-visit residual per independent chain, random
retained count N_x independent of the values). This script tries to break A'.

Three independent probes, none of which reuse the author's verifier:

P1  EXACT STRUCTURAL COUNTEREXAMPLE, both protocols, real certificate code.
    A 3-state chain in which a SECOND visit to the pair exists only if the FIRST
    draw took the "bad" branch. Then, under the old protocol, conditioning on
    {N_x >= n} makes the first n-1 residuals deterministic, so the sample mean
    converges to -10 while the true residual mean rho is 0 and the MP radius
    shrinks like 1/sqrt(n): a violation of size ~10 against a radius of ~0.03.
    The first-visit protocol on the SAME chains must cover at nominal rate.
    This is the sharpest possible statement of what the repair bought.

P2  iid CHECK ON REAL FROZEN ENVIRONMENTS (first-visit protocol only).
    Per pair, condition on the retained count N_x = n and test whether the
    retained residuals look like n iid draws from P_x: mean and variance of the
    residual must not drift with n (a violation of "N independent of values"
    shows up as E[Y | N = n] depending on n).

P3  COVERAGE OF THEOREM 1 at the pre-registered delta_step = 0.05.
    Many independent replications per environment; count how often
    max_x |mean_x - rho_x| > epsilon_res. Theorem 1 claims <= 5% failures.
    The OLD protocol is run on the same draws as a positive control.

Read-only with respect to the repository: writes nothing outside tmp/.
"""

from __future__ import annotations

import json
import math
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
from fp_certfix_first_n import first_n_batch, first_visit_batch  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402

GAMMA = 0.70
D = 12


def exact_rho(mdp, policy, q_hat, gamma=GAMMA):
    """rho_x = (T^pi Qhat - Qhat)(x), exactly, from the MDP tables."""
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    V = (np.asarray(policy, dtype=np.float64) * np.asarray(q_hat, dtype=np.float64)).sum(axis=1)
    TQ = (P * (R + gamma * V[None, None, :])).sum(axis=2)
    return (TQ - np.asarray(q_hat, dtype=np.float64)).reshape(-1)


# --------------------------------------------------------------------------
# P1: the structural counterexample
# --------------------------------------------------------------------------
def sample_chains(P, R, mu, seed, chains, length):
    """Independent chain sampler for the synthetic MDP (not the author's code).

    One independent start per chain, then one independent uniform per step for
    the transition. Action 0 everywhere.
    """
    P = np.asarray(P, float)
    R = np.asarray(R, float)
    rng = np.random.default_rng(seed)
    n_s = P.shape[0]
    states = np.empty((chains, length), np.int64)
    next_states = np.empty((chains, length), np.int64)
    current = rng.choice(n_s, size=chains, p=mu)
    for t in range(length):
        u = rng.random(chains)
        nxt = (u[:, None] > np.cumsum(P[current, 0], axis=1)).sum(axis=1)
        np.clip(nxt, 0, n_s - 1, out=nxt)
        states[:, t] = current
        next_states[:, t] = nxt
        current = nxt
    flat_s = states.reshape(-1)
    flat_n = next_states.reshape(-1)
    return {
        "states": flat_s,
        "actions": np.zeros(flat_s.size, np.int64),
        "rewards": R[flat_s, 0, flat_n],
        "next_states": flat_n,
    }


def coverage_synthetic(reps=200, chains=4000, length=64, n_per_pair=1000, delta_step=0.05):
    """Repeat the trap experiment: violation rate and bias of BOTH protocols.

    A pair-level violation is |mean_x - rho_x| > radius_x, the exact event that
    theorem 1 claims has probability <= delta. Reported per protocol, together
    with the mean signed bias, so a small-but-real bias is visible even when the
    radius happens to cover it.
    """
    mdp, policy, q_hat, mu = build_trap_mdp()
    rho = exact_rho(mdp, policy, q_hat, GAMMA)
    d = 3
    delta_dir = delta_step / (2.0 * d)
    stats = {
        "first_visit": {"pair_fail": 0, "pair_tot": 0, "rep_fail": 0, "reps": 0,
                        "bias": [], "radius": [], "abstain": 0},
        "first_n": {"pair_fail": 0, "pair_tot": 0, "rep_fail": 0, "reps": 0,
                    "bias": [], "radius": [], "abstain": 0},
    }
    for rep in range(reps):
        raw = sample_chains(mdp["P"], mdp["R"], mu, [999, rep], chains, length)

        reduced, counts = first_visit_batch(raw, length, n_states=3, n_actions=1)
        cert = mc.mp_certificate_firstvisit(
            q_hat, policy, reduced, min_visits=200, delta_step=delta_step,
            n_states=3, n_actions=1,
        )
        if cert["status"] != "certificate_emitted":
            stats["first_visit"]["abstain"] += 1
        else:
            means = np.asarray(cert["residual_means"], float)
            radii = np.asarray(cert["radii"], float)
            err = np.abs(means - rho)
            stats["first_visit"]["pair_fail"] += int(np.sum(err > radii))
            stats["first_visit"]["pair_tot"] += d
            stats["first_visit"]["rep_fail"] += int(np.any(err > radii))
            stats["first_visit"]["reps"] += 1
            stats["first_visit"]["bias"].append((means - rho).tolist())
            stats["first_visit"]["radius"].append(radii.tolist())

        red_old, _ = first_n_batch(raw, n_per_pair, n_states=3, n_actions=1)
        if red_old is None:
            stats["first_n"]["abstain"] += 1
        else:
            cert_old = mc.mp_certificate(
                q_hat, policy, red_old, n_per_pair=n_per_pair, delta_step=delta_step,
                n_states=3, n_actions=1,
            )
            if cert_old["status"] != "certificate_emitted":
                stats["first_n"]["abstain"] += 1
            else:
                m2 = np.asarray(cert_old["residual_means"], float)
                r2 = np.asarray(cert_old["radii"], float)
                err2 = np.abs(m2 - rho)
                stats["first_n"]["pair_fail"] += int(np.sum(err2 > r2))
                stats["first_n"]["pair_tot"] += d
                stats["first_n"]["rep_fail"] += int(np.any(err2 > r2))
                stats["first_n"]["reps"] += 1
                stats["first_n"]["bias"].append((m2 - rho).tolist())
                stats["first_n"]["radius"].append(r2.tolist())

    out = {"n_per_pair": n_per_pair, "chains": chains, "length": length,
           "delta_step": delta_step, "delta_dir": delta_dir, "rho": rho.tolist()}
    for tag, s in stats.items():
        bias = np.array(s["bias"]) if s["bias"] else np.zeros((0, d))
        radius = np.array(s["radius"]) if s["radius"] else np.zeros((0, d))
        out[tag] = {
            "reps_run": s["reps"] + s["abstain"],
            "abstentions": s["abstain"],
            "rep_failures": s["rep_fail"],
            "rep_failure_rate": (s["rep_fail"] / s["reps"]) if s["reps"] else None,
            "pair_failures": s["pair_fail"],
            "pair_total": s["pair_tot"],
            "pair_failure_rate": (s["pair_fail"] / s["pair_tot"]) if s["pair_tot"] else None,
            "mean_signed_bias": bias.mean(axis=0).tolist() if bias.size else None,
            "mean_radius": radius.mean(axis=0).tolist() if radius.size else None,
            "worst_abs_bias": float(np.max(np.abs(bias))) if bias.size else None,
            "nominal_pair_rate": delta_dir,
        }
    return out


def build_trap_mdp():
    """3 states, 1 action. s -> u/v each 1/2; u -> s; v -> v absorbing.

    r(s,.,u) = -10, r(s,.,v) = +10, all other rewards 0; Qhat = 0, so
    Vhat = 0 and Y_s = +-10 while Y_u = Y_v = 0.
    A second visit to s exists iff the first draw at s was u.
    """
    P = np.zeros((3, 1, 3))
    R = np.zeros((3, 1, 3))
    P[0, 0] = [0.0, 0.5, 0.5]
    P[1, 0] = [1.0, 0.0, 0.0]
    P[2, 0] = [0.0, 0.0, 1.0]
    R[0, 0] = [0.0, -10.0, 10.0]
    mdp = {"P": P, "R": R}
    policy = np.ones((3, 1))
    q_hat = np.zeros((3, 1))
    mu = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return mdp, policy, q_hat, mu


def run_synthetic(chains=20000, length=32, n_per_pair=2000, delta_step=0.05):
    mdp, policy, q_hat, mu = build_trap_mdp()
    rho = exact_rho(mdp, policy, q_hat, GAMMA)
    raw = sample_chains(mdp["P"], mdp["R"], mu, [12345, 1], chains, length)

    # ---- first-visit (repaired) ----
    reduced, counts_fv = first_visit_batch(raw, length, n_states=3, n_actions=1)
    cert_fv = mc.mp_certificate_firstvisit(
        q_hat, policy, reduced, min_visits=200, delta_step=delta_step,
        n_states=3, n_actions=1,
    )
    # ---- first-n (withdrawn) ----
    reduced_old, counts_old = first_n_batch(
        raw, n_per_pair, n_states=3, n_actions=1
    )
    cert_old = (
        mc.mp_certificate(
            q_hat, policy, reduced_old, n_per_pair=n_per_pair, delta_step=delta_step,
            n_states=3, n_actions=1,
        )
        if reduced_old is not None
        else None
    )

    def block(tag, cert, counts):
        if cert is None or cert["status"] != "certificate_emitted":
            return {"tag": tag, "status": "not_certified",
                    "reasons": None if cert is None else cert["failure_reasons"]}
        means = np.asarray(cert["residual_means"], float)
        radii = np.asarray(cert["radii"], float)
        err = np.abs(means - rho)
        return {
            "tag": tag,
            "status": "certificate_emitted",
            "pair_counts": [int(c) for c in counts],
            "rho": rho.tolist(),
            "means": means.tolist(),
            "radii": radii.tolist(),
            "abs_error": err.tolist(),
            "violated": bool(np.any(err > radii)),
            "violation_ratio": float(np.max(err / np.maximum(radii, 1e-300))),
            "epsilon_res": float(cert["epsilon_res"]),
            "e_q": float(cert["e_q"]),
        }

    out = {
        "exact_rho": rho.tolist(),
        "pair_counts_first_visit": [int(c) for c in counts_fv],
        "pair_counts_first_n": [int(c) for c in counts_old],
        "first_visit": block("first_visit", cert_fv, counts_fv),
        "first_n": block("first_n", cert_old, counts_old),
    }

    # Direct restatement of review #1's counterexample on the raw extractions.
    states_grid = np.asarray(raw["states"], np.int64).reshape(chains, length)
    next_grid = np.asarray(raw["next_states"], np.int64).reshape(chains, length)
    mask = states_grid == 0  # at the critical pair x = (s, 0); the only action is 0
    # residual = R + gamma*Vhat - Qhat = R here (Vhat = Qhat = 0), so -10 iff next = u
    res_grid = np.where(next_grid == 1, -10.0, 10.0)

    has = mask.any(axis=1)
    cid = np.flatnonzero(has)
    firstpos = mask[has].argmax(axis=1)
    out["first_visit_retained_chains"] = int(cid.size)
    out["first_visit_P_Y_is_minus10"] = float(
        np.mean(res_grid[cid, firstpos] == -10.0)
    )

    # old protocol: condition on {visits to x >= 2}
    sel = mask.sum(axis=1) >= 2
    cid2 = np.flatnonzero(sel)
    out["old_frac_chains_with_2plus_visits"] = float(np.mean(sel))
    fp2 = mask[cid2].argmax(axis=1)
    out["old_P_Y1_is_minus10_given_N_ge_2"] = float(
        np.mean(res_grid[cid2, fp2] == -10.0)
    )
    per_chain_mean = np.array(
        [res_grid[i][mask[i]].mean() for i in cid2[:400]]
    )
    out["old_mean_over_all_visits_given_N_ge_2"] = float(per_chain_mean.mean())
    out["old_visit_count_per_chain_given_N_ge_2"] = float(mask[cid2].sum(axis=1).mean())
    return out


# --------------------------------------------------------------------------
# P2/P3: real frozen environments
# --------------------------------------------------------------------------
def run_real(envs, reps, chains, length, min_visits, n_per_pair, delta_step, tag):
    rows = []
    for mixing, task_index in envs:
        mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
        exact = policy_quantities(mdp, behaviour)
        mu = np.asarray(exact["mu_state"], dtype=np.float64)
        train = fs.training_batch(mdp, behaviour, mu, rng)
        q_hat = np.asarray(
            fs.run_route("expected_exact", behaviour, train)["q_hat"], dtype=np.float64
        ).reshape(fs.N_STATES, fs.N_ACTIONS)
        rho = exact_rho(mdp, behaviour, q_hat, GAMMA)
        realized_inf = float(np.max(np.abs(q_hat - np.asarray(exact["q_pi"], float))))

        delta_dir = delta_step / (2.0 * D)
        fails_fv = fails_old = 0
        pair_fail_fv = pair_tot_fv = 0
        abstain_fv = abstain_old = 0
        by_n = {}
        for rep in range(reps):
            raw = vectorised_batch(
                mdp, behaviour, mu, [fs.SEED, 777001, int(round(mixing * 100)), task_index, rep],
                chains, length,
            )
            reduced, counts = first_visit_batch(raw, length)
            cert = mc.mp_certificate_firstvisit(
                q_hat, behaviour, reduced, min_visits=min_visits, delta_step=delta_step
            )
            if cert["status"] != "certificate_emitted":
                abstain_fv += 1
            else:
                means = np.asarray(cert["residual_means"], float)
                radii = np.asarray(cert["radii"], float)
                err = np.abs(means - rho)
                pair_fail_fv += int(np.sum(err > radii))
                pair_tot_fv += D
                if np.any(err > radii):
                    fails_fv += 1
                # P2: does E[Y | N = n] drift with n?
                sizes = np.asarray(cert["pair_sizes"], np.int64)
                for x in range(D):
                    by_n.setdefault(x, []).append((int(sizes[x]), float(means[x]), float(rho[x])))

            red_old, cnt_old = first_n_batch(raw, n_per_pair)
            if red_old is None:
                abstain_old += 1
            else:
                cert_old = mc.mp_certificate(
                    q_hat, behaviour, red_old, n_per_pair=n_per_pair, delta_step=delta_step
                )
                if cert_old["status"] != "certificate_emitted":
                    abstain_old += 1
                else:
                    m2 = np.asarray(cert_old["residual_means"], float)
                    r2 = np.asarray(cert_old["radii"], float)
                    if np.any(np.abs(m2 - rho) > r2):
                        fails_old += 1

        drift = {}
        for x, recs in by_n.items():
            sizes = np.array([r[0] for r in recs], float)
            means = np.array([r[1] for r in recs], float)
            if sizes.std() > 0:
                corr = float(np.corrcoef(sizes, means)[0, 1])
            else:
                corr = float("nan")
            drift[str(x)] = {
                "n_min": int(sizes.min()),
                "n_max": int(sizes.max()),
                "mean_over_reps": float(means.mean()),
                "rho": float(recs[0][2]),
                "corr_n_mean": corr,
            }

        rows.append({
            "mixing": mixing, "task_index": task_index, "tag": tag,
            "reps": reps, "chains": chains, "min_visits": min_visits,
            "n_per_pair": n_per_pair, "delta_step": delta_step, "delta_dir": delta_dir,
            "realized_q_sup_error": realized_inf,
            "first_visit_failures": fails_fv,
            "first_visit_abstentions": abstain_fv,
            "first_visit_pair_fail_rate": pair_fail_fv / max(pair_tot_fv, 1),
            "old_failures": fails_old,
            "old_abstentions": abstain_old,
            "drift_by_pair": drift,
        })
        print(
            f"  {tag} mixing={mixing} task={task_index:>2}  "
            f"fv fails {fails_fv}/{reps} (abstain {abstain_fv})   "
            f"old fails {fails_old}/{reps} (abstain {abstain_old})   "
            f"pair-fail {pair_fail_fv}/{pair_tot_fv}",
            flush=True,
        )
    return rows


def main():
    out = {}
    print("=" * 78)
    print("P1  structural counterexample (3-state trap), exact certificate code")
    print("=" * 78)
    syn = run_synthetic()
    out["P1_synthetic"] = syn
    print(json.dumps(syn, indent=2, sort_keys=True), flush=True)
    print()
    print("P1b  coverage comparison over repeated trap experiments")
    cov = coverage_synthetic()
    out["P1b_synthetic_coverage"] = cov
    print(json.dumps(cov, indent=2, sort_keys=True), flush=True)
    print()
    print("=" * 78)
    print("P2/P3  real frozen environments: coverage of theorem 1 at delta_step")
    print("=" * 78)
    envs = [(0.08, 0), (0.08, 5), (0.5, 0), (0.5, 4), (0.08, 11), (0.5, 11)]
    rows = run_real(
        envs, reps=150, chains=3000, length=64, min_visits=200, n_per_pair=500,
        delta_step=0.05, tag="real_delta05",
    )
    out["P2P3_real"] = rows
    total_fv = sum(r["first_visit_failures"] for r in rows)
    total_reps = sum(r["reps"] for r in rows)
    total_old = sum(r["old_failures"] for r in rows)
    out["P2P3_totals"] = {
        "first_visit_failures": total_fv, "reps": total_reps,
        "first_visit_rate": total_fv / total_reps,
        "old_failures": total_old, "old_rate": total_old / total_reps,
        "nominal": 0.05,
    }
    print()
    print(f"first-visit failure rate {total_fv}/{total_reps} = {total_fv/total_reps:.4f} "
          f"(nominal <= 0.05);  old protocol {total_old}/{total_reps} = {total_old/total_reps:.4f}")
    (ROOT / "tmp").mkdir(exist_ok=True)
    (ROOT / "tmp" / "review2_lemma_A_prime.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=float), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
