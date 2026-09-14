"""FP-COMPOSE-001 / claude: adversarial checks against my own sealed result.

This program is not a re-verification. It takes the specific places where the sealed run
could be RIGHT FOR THE WRONG REASON and tries to falsify them. Four attempts:

1. PER-PAIR CONCENTRATION. H1 as specified in the task sheet is an aggregate check: it
   asks whether E_Q = max_x eps_x/(1-gamma) dominates the sup over pairs. The L12 proof
   asserts something strictly stronger, namely a per-pair bound
       |rho_x| <= |mean_x| + radius_x
   for EACH of the d pairs. An aggregate pass is compatible with widespread per-pair
   violations whenever the violating pairs are not the argmax. So this checks the claim
   the proof actually makes, on every pair of every step, and also reports whether the
   two-sided interval [mean-r, mean+r] happens to cover rho_x (it need not, and that is
   not required -- reporting it separates "the one-sided bound holds" from "the interval
   happens to be centred correctly").

2. y_range AS A RANGE BOUND. The MP empirical-Bernstein radius needs a genuine bound on
   the range of Y. Because Y is a deterministic function of the successor state, its exact
   supremum can be enumerated from the MDP with no sampling at all, so the bound is checked
   exactly rather than statistically.

3. PRODUCER INDEPENDENCE FROM THE CERTIFICATION BATCH. T4/T5 claim the producers read only
   (train, pi) and never B_k. That is a measurable, falsifiable statement: hand the
   producer a DIFFERENT certification batch and its Qhat must not move by one bit.

4. TRUE IMPROVEMENT PER EMITTED ROW. Non-degradation is checked in aggregate as
   v^{pi'} - v^pi >= 0 per state. The mechanism claim is per row: every row the rule
   actually changed should have A_s > 0 under the true Q^pi. A row could in principle pass
   LB_s > 0 while the true A_s is negative if E_Q were too small.

    python adversarial_checks.py --results <.../formal/task_results.json> \
        --output <.../verification/adversarial_checks.json> [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

import torch  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_xfam_001 import (  # noqa: E402
    FAMILIES, build_family_task, run_route, training_batch, vectorised_batch_generic,
)
from model import (  # noqa: E402
    EndToEndFiniteSoftmaxExpectedSARSA, EndToEndMaskedSoftmaxExpectedSARSA,
)

LAYERS, ALPHA = 160, 0.65
ZETA = XI = TAU = 8.0


def true_residual(P, R, pi, q_hat, gamma):
    """rho_x = (T^pi Qhat)(x) - Qhat(x), exactly, from the stored MDP."""
    v_hat = np.einsum("sa,sa->s", pi, q_hat)
    t_q = np.einsum("sat,sat->sa", P, R + gamma * v_hat[None, None, :])
    return t_q - q_hat


def y_sup(P, R, pi, q_hat, gamma):
    """Exact sup over the batch support of |Y| for each pair, with no sampling.

    Y(s,a,s') = R[s,a,s'] + gamma*(pi(s').Qhat(s')) - Qhat(s,a), so the support of Y for
    pair (s,a) is the finite set indexed by s'.
    """
    v_hat = np.einsum("sa,sa->s", pi, q_hat)
    Y = R + gamma * v_hat[None, None, :] - q_hat[:, :, None]
    return np.abs(Y).max(axis=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--independent-steps", type=int, default=6,
                    help="how many producer evaluations to use for check 3")
    args = ap.parse_args()

    bundle = json.loads(args.results.read_text(encoding="utf-8"))
    chains = int(bundle["chains"]); chain_length = int(bundle["chain_length"])
    horizon = int(bundle["horizon"])
    seed, salt = int(bundle["seed"]), int(bundle["salt"])
    delta_step = float(bundle["delta_step"])
    producers = tuple(bundle["producers"]); routes = tuple(bundle["routes"])

    rec: dict[str, Any] = {
        "attempts": {}, "wall_seconds": None,
        "framing": ("Adversarial checks by the SAME author on the SAME sealed bundle. This "
                    "is not independent verification and does not raise the evidence grade; "
                    "its only value is that it tries to falsify rather than confirm."),
    }
    t0 = time.time()

    # ---------------- 1 + 2 + 4: exact, per pair, no sampling ---------------- #
    pair_total = 0
    viol_onesided: list[dict] = []
    inside_interval = 0
    interval_total = 0
    yr_viol: list[dict] = []
    yr_ratio: list[float] = []
    row_total = 0
    row_negative_A: list[dict] = []
    row_zero_lb = 0
    margin_by_pair: list[float] = []
    n_tasks = 0
    n_cell_steps = 0
    for task in bundle["records"]:
        if args.limit and n_tasks >= args.limit:
            break
        n_tasks += 1
        fam_name = task["family"]; fam = FAMILIES[fam_name]
        mixing = float(task["mixing"]); tidx = int(task["task_index"])
        mdp, behaviour, rng = build_family_task(fam, mixing, tidx)
        P = np.asarray(mdp["P"], np.float64); R = np.asarray(mdp["R"], np.float64)
        gamma = fam["gamma"]; A = fam["n_actions"]; S = fam["n_states"]
        for route in routes:
            for producer in producers:
                cell = task["routes"][route][f"{producer}|perstate|L12"]
                for e in cell["steps"]:
                    if e["pair_sizes"] is None:
                        continue
                    pi = np.asarray(e["pi_before"], np.float64)
                    qh = np.asarray(e["q_hat"], np.float64)
                    means = np.asarray(e["pair_means"], np.float64)
                    radii = np.asarray(e["pair_radii"], np.float64)
                    eps = np.asarray(e["pair_eps"], np.float64)
                    rho = true_residual(P, R, pi, qh, gamma).reshape(-1)
                    bound = np.abs(means) + radii
                    pair_total += rho.size
                    n_cell_steps += 1
                    bad = np.flatnonzero(np.abs(rho) > bound)
                    for x in bad:
                        viol_onesided.append({
                            "family": fam_name, "mixing": mixing, "task_index": tidx,
                            "route": route, "producer": producer, "step": int(e["step"]),
                            "pair": int(x), "abs_rho": float(abs(rho[x])),
                            "bound": float(bound[x]),
                            "excess": float(abs(rho[x]) - bound[x])})
                    inside = (rho >= means - radii) & (rho <= means + radii)
                    inside_interval += int(inside.sum()); interval_total += rho.size
                    margin_by_pair.append(float(np.min(bound - np.abs(rho))))
                    # y_range validity, exact
                    ys = y_sup(P, R, pi, qh, gamma).reshape(-1)
                    need = 2.0 * float(ys.max())
                    y_r = float(e["y_range"])
                    yr_ratio.append(y_r / need if need > 0 else float("inf"))
                    if y_r < need - 1e-12:
                        yr_viol.append({"family": fam_name, "task_index": tidx,
                                        "route": route, "producer": producer,
                                        "step": int(e["step"]), "y_range": y_r,
                                        "needed": need})
                    # per-row true improvement for every CHANGED row
                    pi_after = np.asarray(e["pi_after"], np.float64)
                    Qpi, _ = _q_pi(P, R, pi, gamma)
                    for s in range(S):
                        if np.array_equal(pi_after[s], pi[s]):
                            continue
                        row_total += 1
                        d_row = pi_after[s] - pi[s]
                        a_true = float(np.dot(d_row, Qpi[s]))
                        if a_true <= 0.0:
                            row_negative_A.append({
                                "family": fam_name, "task_index": tidx, "route": route,
                                "producer": producer, "step": int(e["step"]), "state": s,
                                "A_true": a_true})
    rec["attempts"]["1_per_pair_concentration"] = {
        "pairs_checked": pair_total,
        "one_sided_violations": len(viol_onesided),
        "one_sided_violation_rate": (len(viol_onesided) / pair_total if pair_total else None),
        "min_slack_over_pairs": (min(margin_by_pair) if margin_by_pair else None),
        "examples": viol_onesided[:20],
        "two_sided_interval_coverage": (inside_interval / interval_total
                                        if interval_total else None),
        "violations_the_frozen_budget_permits": {
            "derivation": ("per cell-step the union over d pairs gives <= d*delta_dir = "
                           "delta_step = 0.0125, so the expected NUMBER of violating "
                           "pair-steps is bounded by the number of cell-steps times "
                           "delta_step"),
            "cell_steps": n_cell_steps,
            "delta_step": delta_step,
            "expected_upper_bound": n_cell_steps * delta_step,
            "observed": len(viol_onesided),
            "reading": ("0 observed against an allowance of this size means the bound is "
                        "LOOSE here, not tight. It is not evidence of validity -- and had "
                        "roughly this many violations appeared, that would have been "
                        "consistent with the bound rather than alarming. This is the "
                        "quantitative form of the task sheet's warning that zero coverage "
                        "violations is a failure detector, not proof."),
        },
        "reading": ("the one-sided bound |rho_x| <= |mean_x| + r_x is the claim the proof "
                    "makes and is what must hold; the two-sided interval is reported only "
                    "to show the one-sided result is not an artefact of a mis-centred "
                    "interval."),
    }
    rec["attempts"]["2_y_range_validity"] = {
        "steps_checked": len(yr_ratio),
        "violations": len(yr_viol),
        "min_ratio_y_range_over_needed": (min(yr_ratio) if yr_ratio else None),
        "median_ratio": (float(np.median(yr_ratio)) if yr_ratio else None),
        "examples": yr_viol[:10],
        "method": ("Y is a deterministic function of the successor state, so sup|Y| is "
                   "enumerated exactly from the stored MDP; no sampling is involved."),
    }
    rec["attempts"]["4_true_improvement_per_row"] = {
        "changed_rows_checked": row_total,
        "rows_with_nonpositive_true_A": len(row_negative_A),
        "examples": row_negative_A[:10],
        "reading": ("a changed row passing LB_s > 0 while its true A_s <= 0 would mean E_Q "
                    "was too small at that row; this is the per-row version of the "
                    "non-degradation check."),
    }

    # ---------------- 3: Qhat_k is a function of (pi_before, train) alone --------- #
    # The sharp, falsifiable form of the measurability claim. If the producer carried any
    # hidden state from earlier steps -- a mutated network, a cached batch, an accumulator --
    # then recomputing it at step k > 1 from the sealed pi_before and train alone would NOT
    # reproduce the sealed q_hat. Drawing a fresh batch in between makes the test bite.
    nets = {"expected_exact": EndToEndMaskedSoftmaxExpectedSARSA(gamma=FAMILIES["f1"]["gamma"],
                                                                alpha=ALPHA),
            "expected_finite": EndToEndFiniteSoftmaxExpectedSARSA(
                gamma=FAMILIES["f1"]["gamma"], alpha=ALPHA, zeta=ZETA, xi=XI, tau=TAU)}
    probes: list[dict] = []
    max_diff = {"numpy": 0.0, "network": 0.0}
    checked = 0
    for task in bundle["records"][:4]:
        if checked >= args.independent_steps:
            break
        fam_name = task["family"]; fam = FAMILIES[fam_name]
        mdp, behaviour, rng = build_family_task(fam, float(task["mixing"]),
                                                int(task["task_index"]))
        mu = np.asarray(policy_quantities(mdp, behaviour)["mu_state"], np.float64)
        train = training_batch(mdp, behaviour, mu, rng)
        nets = {"expected_exact": EndToEndMaskedSoftmaxExpectedSARSA(gamma=fam["gamma"],
                                                                    alpha=ALPHA),
                "expected_finite": EndToEndFiniteSoftmaxExpectedSARSA(
                    gamma=fam["gamma"], alpha=ALPHA, zeta=ZETA, xi=XI, tau=TAU)}
        for route in routes:
            for producer in producers:
                cell = task["routes"][route][f"{producer}|perstate|L12"]
                # deliberately consume an unrelated batch first, to poison any hidden state
                vectorised_batch_generic(mdp, behaviour, mu,
                                         [seed, salt, 999, int(task["task_index"]), 7],
                                         chains, fam, chain_length)
                for e in cell["steps"]:
                    pi = np.asarray(e["pi_before"], np.float64)
                    if producer == "numpy":
                        q = np.asarray(run_route(route, fam, pi, train), np.float64)
                    else:
                        q, _ = _net(nets[route], pi, train)
                    diff = float(np.max(np.abs(q - np.asarray(e["q_hat"], np.float64))))
                    max_diff[producer] = max(max_diff[producer], diff)
                    checked += 1
                    probes.append({"family": fam_name, "task_index": int(task["task_index"]),
                                   "route": route, "producer": producer,
                                   "step": int(e["step"]), "max_abs_diff": diff})
                if checked >= args.independent_steps:
                    break
            if checked >= args.independent_steps:
                break
    rec["attempts"]["3_qhat_is_function_of_pi_and_train_only"] = {
        "probes": checked,
        "max_abs_diff_numpy": max_diff["numpy"],
        "max_abs_diff_network": max_diff["network"],
        "all_bitwise_equal": bool(max_diff["numpy"] == 0.0 and max_diff["network"] == 0.0),
        "steps_including_k_gt_1": sorted({p["step"] for p in probes}),
        "detail": probes[:12],
        "note": ("the producer receives no argument derived from B_k, and an unrelated batch "
                 "is consumed before each recomputation; a hidden dependence on earlier "
                 "steps or on the batch would show up as a nonzero difference."),
    }

    rec["wall_seconds"] = time.time() - t0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items()
                           if kk in ("pairs_checked", "one_sided_violations",
                                     "one_sided_violation_rate", "min_slack_over_pairs",
                                     "two_sided_interval_coverage", "violations",
                                     "min_ratio_y_range_over_needed",
                                     "changed_rows_checked",
                                     "rows_with_nonpositive_true_A")})
                      for k, v in rec["attempts"].items()}, indent=2))


def _q_pi(P, R, pi, gamma):
    S, A = pi.shape
    r_pi = np.einsum("sat,sat->sa", P, np.broadcast_to(R, P.shape))
    Ppi = np.einsum("sat,tb->satb", P, pi).reshape(S * A, S * A)
    q = np.linalg.solve(np.eye(S * A) - gamma * Ppi, r_pi.reshape(-1)).reshape(S, A)
    return q, np.einsum("sa,sa->s", pi, q)


def _net(net, policy, train):
    st = torch.as_tensor(train["states"], dtype=torch.long)
    ac = torch.as_tensor(train["actions"], dtype=torch.long)
    rw = torch.as_tensor(train["rewards"], dtype=torch.float32)
    ns = torch.as_tensor(train["next_states"], dtype=torch.long)
    pi = torch.as_tensor(policy, dtype=torch.float32)
    q = torch.zeros(policy.shape, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(LAYERS):
            q, _ = net(q, st, ac, rw, ns, pi)
    return q.detach().numpy().astype(np.float64), LAYERS


if __name__ == "__main__":
    main()
