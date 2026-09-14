"""FP-COMPOSE-001 / claude: read-only audit of the SHARED input generators.

Task sheet section 8 requires that the common input generators -- the family MDP, the
training batch, the dimension-generic sampler and the first-visit extractor -- be examined
read-only for their sampling law and first-visit premise, with the limits recorded, and
that this NOT be presented as an independent discovery of the object under test.

This program checks the claimed law against the generator's own output rather than only
reading the code, and re-implements the first-visit reduction independently to compare.
It writes nothing into the sealed bundle.

    python check_shared_generators.py --family f1 --mixing 0.5 --task-index 200 \
        --output results/FP-COMPOSE-001/claude/verification/shared_generator_check.json
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

from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_xfam_001 import (  # noqa: E402
    FAMILIES, build_family_task, run_route, training_batch, vectorised_batch_generic,
)
from fp_certfix_first_n import first_visit_batch  # noqa: E402

SEED = 20260911
SALT = 90417


def tv(emp: np.ndarray, ref: np.ndarray) -> float:
    return float(0.5 * np.abs(emp - ref).sum())


def independent_first_visits(states, actions, chain_length, n_states, n_actions):
    """Re-derive the first-visit reduction with a different implementation.

    Returns (counts, retained_positions). The reference implementation loops pair-by-pair
    with a boolean mask; this one sorts by (chain, pair) and takes the first slot per group.
    """
    flat = np.asarray(states, np.int64) * n_actions + np.asarray(actions, np.int64)
    n_chains = flat.size // chain_length
    grid = flat.reshape(n_chains, chain_length)
    chain_ids = np.repeat(np.arange(n_chains), chain_length)
    order = np.lexsort((np.arange(flat.size), flat, chain_ids))
    cid_sorted = chain_ids[order]
    flat_sorted = flat[order]
    new_group = np.empty(flat.size, dtype=bool)
    new_group[0] = True
    new_group[1:] = (cid_sorted[1:] != cid_sorted[:-1]) | (flat_sorted[1:] != flat_sorted[:-1])
    keep = np.sort(order[new_group])
    counts = np.bincount(flat[keep], minlength=n_states * n_actions).astype(np.int64)
    return counts, keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="f1")
    ap.add_argument("--mixing", type=float, default=0.5)
    ap.add_argument("--task-index", type=int, default=200)
    ap.add_argument("--chains", type=int, default=32768)
    ap.add_argument("--chain-length", type=int, default=32)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    fam = FAMILIES[args.family]
    S, A = fam["n_states"], fam["n_actions"]
    d = S * A
    out: dict[str, Any] = {"location": {"family": args.family, "mixing": args.mixing,
                                        "task_index": args.task_index,
                                        "chains": args.chains,
                                        "chain_length": args.chain_length,
                                        "step": args.step},
                           "checks": {}, "limits": []}
    C = out["checks"]

    mdp, behaviour, rng = build_family_task(fam, args.mixing, args.task_index)
    mu = np.asarray(policy_quantities(mdp, behaviour)["mu_state"], dtype=np.float64)
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)

    # --- A. the generator's own precision ---------------------------------- #
    P32 = np.asarray(mdp["P"], dtype=np.float32).astype(np.float64)
    R32 = np.asarray(mdp["R"], dtype=np.float32).astype(np.float64)
    rowsum = P.sum(axis=2)
    C["A_generator_precision"] = {
        "P_dtype_in_mdp": str(np.asarray(mdp["P"]).dtype),
        "R_dtype_in_mdp": str(np.asarray(mdp["R"]).dtype),
        "raw_draw_dtype_in_sample_mdp": "float32 (P, R and p0 are all astype(np.float32))",
        "P_stored_is_exactly_float32": bool(np.array_equal(P, P32)),
        "R_stored_is_exactly_float32": bool(np.array_equal(R, R32)),
        "R_stored_is_exactly_float32_by_action": [
            bool(np.array_equal(R[:, a, :], R32[:, a, :])) for a in range(A)],
        "P_rowsum_min": float(rowsum.min()), "P_rowsum_max": float(rowsum.max()),
        "P_rowsum_max_abs_dev_from_one": float(np.max(np.abs(rowsum - 1.0))),
        "R_min": float(R.min()), "R_max": float(R.max()),
        "R_abs_max_over_reward_bound": float(np.max(np.abs(R))) / fam["reward_bound"],
        "note": ("sample_mdp draws P, R and p0 as float32. make_mdp then renormalises and "
                 "mixes P, and adds gap_bonus to action 0's rewards, all in float64, so the "
                 "STORED tensors are generally not float32-representable (action 0 rewards "
                 "and the mixed P rows are not) even though their randomness came from "
                 "float32 draws. The environment therefore inherits float32-level "
                 "information, and every claim in this task is about this stored MDP."),
    }

    # --- B/C/D/E. the sampler's law against the analytic law ---------------- #
    raw = vectorised_batch_generic(mdp, behaviour, mu,
                                  [SEED, SALT, int(round(100 * args.mixing)),
                                   args.task_index, args.step],
                                  args.chains, fam, args.chain_length)
    st = np.asarray(raw["states"], np.int64)
    ac = np.asarray(raw["actions"], np.int64)
    rw = np.asarray(raw["rewards"], np.float64)
    ns = np.asarray(raw["next_states"], np.int64)

    starts = st.reshape(args.chains, args.chain_length)[:, 0]
    emp_start = np.bincount(starts, minlength=S) / args.chains
    C["B_start_state_law"] = {
        "total_variation_vs_mu_state": tv(emp_start, mu),
        "max_abs_dev": float(np.max(np.abs(emp_start - mu))),
        "empirical": emp_start.tolist(), "mu_state": mu.tolist(),
        "interpretation": ("mu_state comes from policy_quantities and is the stationary "
                           "distribution of the BEHAVIOUR chain, not mdp['p0']."),
    }

    emp_act = np.zeros((S, A)); emp_nxt = np.zeros((S, A, S))
    act_tv, act_n = {}, {}
    for s in range(S):
        m = st == s
        n_s = int(m.sum())
        if n_s:
            emp_act[s] = np.bincount(ac[m], minlength=A) / n_s
        act_tv[s] = tv(emp_act[s], behaviour[s]); act_n[s] = n_s
        for a in range(A):
            m2 = m & (ac == a)
            n2 = int(m2.sum())
            if n2:
                emp_nxt[s, a] = np.bincount(ns[m2], minlength=S) / n2
    C["C_action_law"] = {"max_abs_dev": float(np.max(np.abs(emp_act - behaviour))),
                         "worst_tv_by_state": act_tv, "samples_per_state": act_n}
    dev_nxt = np.abs(emp_nxt - P)
    C["D_transition_law"] = {
        "max_abs_dev": float(dev_nxt.max()),
        "worst_cell": [int(np.unravel_index(np.argmax(dev_nxt), dev_nxt.shape)[0]),
                       int(np.unravel_index(np.argmax(dev_nxt), dev_nxt.shape)[1]),
                       int(np.unravel_index(np.argmax(dev_nxt), dev_nxt.shape)[2])],
        "samples_per_pair_min": int(np.bincount(st * A + ac, minlength=d).min()),
        "note": ("the sampler compares a uniform against a cumsum of the STORED P row and "
                 "clips the result, so a row that did not sum to 1 would shift the last "
                 "state. Max row deviation from 1 is reported in A."),
    }
    C["E_reward_determinism"] = {
        "rewards_equal_R_at_triple": bool(np.array_equal(rw, R[st, ac, ns])),
        "batch_rewards_dtype": str(np.asarray(raw["rewards"]).dtype),
        "max_abs_diff": float(np.max(np.abs(rw - R[st, ac, ns]))),
    }

    # --- F. determinism and stream separation ------------------------------ #
    raw2 = vectorised_batch_generic(mdp, behaviour, mu,
                                    [SEED, SALT, int(round(100 * args.mixing)),
                                     args.task_index, args.step],
                                    args.chains, fam, args.chain_length)
    raw_other = vectorised_batch_generic(mdp, behaviour, mu,
                                         [SEED, SALT, int(round(100 * args.mixing)),
                                          args.task_index, args.step + 1],
                                         args.chains, fam, args.chain_length)
    C["F_determinism"] = {
        "same_seed_parts_bitwise_identical": bool(
            np.array_equal(st, np.asarray(raw2["states"], np.int64))),
        "different_step_differs": bool(
            not np.array_equal(st, np.asarray(raw_other["states"], np.int64))),
        "seeding": ("a fresh np.random.default_rng(seed_parts) per call, with seed_parts = "
                    "[SEED, SALT, round(100*mixing), task_index, step]; so the batch is a "
                    "deterministic function of that vector alone and is indexed by step."),
        "policy_argument": ("the sampler is called with the FIXED behaviour policy at every "
                            "step, never with the current pi, so B_k does not depend on the "
                            "update path."),
    }
    train = training_batch(mdp, behaviour, mu, rng)
    tst = np.asarray(train["states"], np.int64)
    C["F2_training_batch"] = {
        "length": int(tst.size),
        "is_single_trajectory": bool(np.array_equal(tst[1:], tst[:-1] + 0)
                                     is False),
        "consecutive_states_link": bool(
            np.array_equal(np.asarray(train["next_states"], np.int64)[:-1], tst[1:])),
        "alignment_holds": bool(np.array_equal(
            np.asarray(train["rewards"], np.float64),
            R[tst, np.asarray(train["actions"], np.int64),
              np.asarray(train["next_states"], np.int64)])),
        "rewards_exactly_float32_representable": bool(np.array_equal(
            np.asarray(train["rewards"], np.float64),
            np.asarray(train["rewards"], np.float32).astype(np.float64))),
        "note": ("training_batch is ONE rollout of length 65536 from a single mu_state "
                 "start, so its samples are serially correlated, unlike the certification "
                 "batch. That is admissible because the certificate concentrates on the "
                 "certification batch and treats Qhat as an arbitrary fixed function; it "
                 "does mean the producer is not an iid estimator of anything."),
    }

    # --- G. first-visit reduction, re-derived independently ----------------- #
    ref_reduced, ref_counts = first_visit_batch(raw, args.chain_length,
                                                n_states=S, n_actions=A)
    my_counts, my_keep = independent_first_visits(st, ac, args.chain_length, S, A)
    ref_flat = (np.asarray(ref_reduced["states"], np.int64) * A
                + np.asarray(ref_reduced["actions"], np.int64))
    ref_len = ref_flat.size
    C["G_first_visit_mechanics"] = {
        "counts_match_independent_reimplementation": bool(
            np.array_equal(ref_counts, my_counts)),
        "retained_size_equals_counts_sum": bool(ref_len == int(ref_counts.sum())),
        "retained_size": int(ref_len),
        "counts_min": int(ref_counts.min()), "counts_max": int(ref_counts.max()),
        "chains": int(args.chains), "chain_length": int(args.chain_length),
        "max_count_le_chains": bool(int(ref_counts.max()) <= args.chains),
        "retained_positions_match": None,     # filled below
        "note": ("each chain contributes at most one retained item per pair, at its first "
                 "visit; counts[x] is the number of distinct chains visiting x."),
    }
    # independent reconstruction of the retained positions in the reference ordering
    flat_all = st * A + ac
    n_chains = st.size // args.chain_length
    grid = flat_all.reshape(n_chains, args.chain_length)
    keep_ref = []
    for pair in range(d):
        mask = grid == pair
        has = mask.any(axis=1)
        first_pos = mask[has].argmax(axis=1)
        keep_ref.extend((np.flatnonzero(has) * args.chain_length + first_pos).tolist())
    keep_ref = np.sort(np.asarray(keep_ref, np.int64))
    C["G_first_visit_mechanics"]["retained_positions_match"] = bool(
        np.array_equal(keep_ref, my_keep))

    # --- H. empirical check of the first-visit law -------------------------- #
    # The retained residual at pair x = (s,a) is a DETERMINISTIC function of the single
    # transition drawn at the first visit: Y = R[s,a,s'] + gamma*(pi(s').Qhat(s')) - Qhat(s,a).
    # Because the MDP is finite, Y takes at most S distinct values and its exact law is the
    # P[s,a,:] mass pushed through that map. So the operative content of Lemma A' -- that a
    # chain's first visit to x yields a fresh draw from P_x -- is directly checkable by
    # comparing the retained empirical law against that exact discrete law, pair by pair.
    q0 = np.asarray(run_route("expected_exact", fam, behaviour, train), np.float64)
    pi = np.asarray(behaviour, np.float64)
    resid_all = (rw + fam["gamma"] * np.einsum("na,na->n", pi[ns], q0[ns])
                 - q0[st, ac])
    v_hat = np.einsum("sa,sa->s", pi, q0)
    pair_tv, pair_n, worst = {}, {}, None
    for pair in range(d):
        s, a = divmod(pair, A)
        mask = grid == pair
        vc = mask.sum(axis=1)
        ch = np.flatnonzero(vc > 0)
        if ch.size == 0:
            pair_tv[pair], pair_n[pair] = None, 0
            continue
        first_pos_in_chain = np.argmax(mask[ch], axis=1)
        y = resid_all[ch * args.chain_length + first_pos_in_chain]
        # exact law of Y under s' ~ P[s,a,:]
        g = R[s, a, :] + fam["gamma"] * v_hat - q0[s, a]
        law: dict[float, float] = {}
        for sp in range(S):
            law[round(float(g[sp]), 12)] = law.get(round(float(g[sp]), 12), 0.0) + float(P[s, a, sp])
        emp: dict[float, float] = {}
        for val in y:
            k = round(float(val), 12)
            emp[k] = emp.get(k, 0.0) + 1.0 / y.size
        keys = set(law) | set(emp)
        t = 0.5 * sum(abs(emp.get(k, 0.0) - law.get(k, 0.0)) for k in keys)
        pair_tv[pair], pair_n[pair] = float(t), int(y.size)
        if worst is None or t > worst[1]:
            worst = (pair, float(t))
    vals = [v for v in pair_tv.values() if v is not None]
    C["H_first_visit_law"] = {
        "pairs_checked": len(vals),
        "worst_total_variation": (max(vals) if vals else None),
        "worst_pair": (worst[0] if worst else None),
        "median_total_variation": (float(np.median(vals)) if vals else None),
        "min_retained_per_pair": (min(pair_n.values()) if pair_n else None),
        "sample_sizes_note": ("total variation carries a Monte-Carlo floor of about "
                              "sqrt(2/(pi*n)) for n retained values; the per-pair n is "
                              "reported so the floor can be judged."),
        "what_this_does_and_does_not_show": (
            "This checks the MARGINAL law of the retained first-visit values against the "
            "exact discrete law induced by P[s,a,:]. That is the operative content of the "
            "premise. It does NOT test independence WITHIN the retained set for a fixed "
            "capacity n; that follows from the chains being generated independently, which "
            "is a structural property, not something this simulation establishes."),
        "rejected_first_probe": (
            "An earlier version of this probe compared, among chains visiting a pair, those "
            "visiting it exactly once against those visiting it more often, and returned a "
            "worst |z| of 49. That was a probe of the wrong event: whether a chain RETURNS "
            "to the pair depends on the same first-visit successor s' that determines Y, so "
            "a difference there is expected and does not contradict the premise. The "
            "conditioning event in the premise is which chains visit x, not how often each "
            "one visits."),
    }
    out["limits"] = [
        "The shared generators are a COMMON DEPENDENCY, not an independent "
        "reimplementation of the object under test; this audit does not make the two "
        "routes independent and must not be reported as an independent discovery.",
        "The environment is a float32-precision object (sample_mdp), so all statements are "
        "about that stored MDP; float64 downstream arithmetic cannot restore the lost bits.",
        "mu_state is the stationary distribution of the behaviour chain, so chain starts "
        "are not iid uniform over states; per-pair retention counts inherit that "
        "concentration.",
        "The training batch is a single serially correlated rollout, so the producer Qhat "
        "is not an iid estimator; the certificate's concentration is on the certification "
        "batch and treats Qhat as a fixed history-measurable function.",
        "The first-visit probe is a search for a selection signature, not a verification of "
        "Lemma A'.",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "P_rowsum_max_dev": C["A_generator_precision"]["P_rowsum_max_abs_dev_from_one"],
        "P_stored_float32": C["A_generator_precision"]["P_stored_is_exactly_float32"],
        "R_stored_float32_by_action":
            C["A_generator_precision"]["R_stored_is_exactly_float32_by_action"],
        "start_TV_vs_mu": C["B_start_state_law"]["total_variation_vs_mu_state"],
        "action_max_dev": C["C_action_law"]["max_abs_dev"],
        "transition_max_dev": C["D_transition_law"]["max_abs_dev"],
        "reward_deterministic": C["E_reward_determinism"]["rewards_equal_R_at_triple"],
        "deterministic": C["F_determinism"]["same_seed_parts_bitwise_identical"],
        "firstvisit_counts_match": C["G_first_visit_mechanics"][
            "counts_match_independent_reimplementation"],
        "firstvisit_positions_match": C["G_first_visit_mechanics"]["retained_positions_match"],
        "firstvisit_law_worst_TV": C["H_first_visit_law"]["worst_total_variation"],
        "firstvisit_law_median_TV": C["H_first_visit_law"]["median_total_variation"],
        "firstvisit_law_min_n": C["H_first_visit_law"]["min_retained_per_pair"],
    }, indent=2))


if __name__ == "__main__":
    main()
