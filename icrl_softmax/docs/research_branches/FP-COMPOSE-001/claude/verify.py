"""FP-COMPOSE-001 / claude: independent replay verifier.

Rebuilds a sealed FP-COMPOSE-001 bundle from the frozen protocol and compares every
recorded quantity field by field. It reads ONLY the frozen baseline objects (family MDP
generator, training batch, dimension-generic sampler, first-visit extractor, array routes,
model.py networks) plus the sealed JSON. It does NOT import the author's evaluate.py,
analyze.py or any author-written decision helper: the certificate scalar, the per-state
rule, the linear solve and the value audit are rewritten here with a different code shape
(vectorised rather than row loops; iteratively refined solve alongside the direct solve).

Usage -- the same program verifies either route, which is what makes the two reports
comparable:

    python verify.py --results <.../formal/task_results.json> \
                     --manifest <.../formal/input_manifest.json> \
                     --output <.../verification_of_other.json> [--limit N]

Comparison thresholds are the ones frozen in the task sheet section 8: same-executor
replay of Qhat <= 1e-9; replay of E_Q and values <= 1e-10. They are consistency thresholds
only and are never used to excuse an H1 inequality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa as es  # noqa: E402
import torch  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_xfam_001 import (  # noqa: E402
    FAMILIES, build_family_task, run_route, training_batch, vectorised_batch_generic,
)
from fp_certfix_first_n import first_visit_batch  # noqa: E402
from model import (  # noqa: E402
    EndToEndFiniteSoftmaxExpectedSARSA, EndToEndMaskedSoftmaxExpectedSARSA,
)

QHAT_TOL = 1e-9          # task sheet section 8, C1
VALUE_TOL = 1e-10        # task sheet section 8, C1
LAYERS = 160
ALPHA = 0.65
ZETA = XI = TAU = 8.0
ETA_GRID = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


# --------------------------------------------------------------------------- #
# Independent numeric kernels (rewritten, not imported from the author)
# --------------------------------------------------------------------------- #
def solve_pairspace(mdp, policy, gamma, tol=1e-13):
    """Q^pi and v^pi two ways: a direct solve and a convergent iteration.

    Direct:  vec(Q) = (I - gamma P_pi)^-1 vec(r_pi).
    Iterate: Q <- r_pi + gamma P_pi Q until the sup change < tol (Richardson).
    Agreement of the two is reported so a linear-algebra slip cannot pass silently.
    """
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    pi = np.asarray(policy, dtype=np.float64)
    S, A = pi.shape
    r_pi = np.einsum("sat,sat->sa", P, np.broadcast_to(R, P.shape))
    Ppi = np.einsum("sat,tb->satb", P, pi).reshape(S * A, S * A)
    direct = np.linalg.solve(np.eye(S * A) - gamma * Ppi, r_pi.reshape(-1)).reshape(S, A)
    q = direct.copy()
    for _ in range(20000):
        nq = r_pi + gamma * (Ppi @ q.reshape(-1)).reshape(S, A)
        if np.max(np.abs(nq - q)) < tol:
            q = nq
            break
        q = nq
    v = np.einsum("sa,sa->s", pi, direct)
    return direct, v, float(np.max(np.abs(q - direct)))


def value_star(mdp, gamma, tol=1e-14, iters=100000):
    P = np.asarray(mdp["P"], dtype=np.float64)
    R = np.asarray(mdp["R"], dtype=np.float64)
    v = np.zeros(P.shape[0])
    for _ in range(iters):
        nv = np.max(np.einsum("sat,sat->sa", P, np.broadcast_to(
            R + gamma * v[None, None, :], P.shape)), axis=1)
        if np.max(np.abs(nv - v)) < tol:
            return nv
        v = nv
    return v


def certificate_from_stats(q_hat, policy, sizes, means, variances, delta_step, gamma, r_star):
    """E_Q rebuilt from the SEALED per-pair statistics, arranged differently.

    Sizes/means/variances are read from the bundle, so this isolates the arithmetic of
    the concentration step from the sampling step. The sampling step is checked
    separately by replaying the batch and recomputing these statistics.
    """
    S, A = policy.shape
    d = S * A
    v_hat = np.einsum("sa,sa->s", policy, q_hat)
    y_range = 2.0 * (r_star + gamma * float(np.max(np.abs(v_hat)))
                     + float(np.max(np.abs(q_hat))))
    delta_dir = float(delta_step) / d
    log_term = math.log(2.0 / delta_dir)
    sizes = np.asarray(sizes, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    radius = np.sqrt(2.0 * np.maximum(variances, 0.0) * log_term / sizes) + (
        (7.0 / 3.0) * y_range * log_term / np.maximum(sizes - 1.0, 1.0))
    eps = np.abs(means) + radius
    return {"e_q": float(np.max(eps)) / (1.0 - gamma), "radius": radius, "eps": eps,
            "y_range": y_range, "delta_dir": delta_dir}


def perstate_decide(policy, q_hat, e_q, eta_grid=ETA_GRID):
    """The per-state rule, written with the eta grid as an outer axis.

    Returns the full LB table (S x |eta|), the chosen eta per row (None if no eta passes),
    the update mask and pi_after. Every candidate is built from the OLD policy.
    """
    S = policy.shape[0]
    table = np.full((S, len(eta_grid)), -np.inf)
    chosen: list[Any] = [None] * S
    pi_after = policy.copy()
    for j, eta in enumerate(eta_grid):
        cand = es.relative_softmax_candidate(policy, q_hat, eta)
        delta = cand - policy
        # LB_s = <delta_s, Qhat_s> - E_Q * ||delta_s||_1
        lb = np.einsum("sa,sa->s", delta, q_hat) - e_q * np.abs(delta).sum(axis=1)
        table[:, j] = lb
        take = (np.array([c is None for c in chosen])) & (lb > 0.0)
        for s in np.flatnonzero(take):
            chosen[s] = float(eta)
            pi_after[s] = cand[s]
    mask = np.array([c is not None for c in chosen], dtype=bool)
    return {"table": table, "rows_eta": chosen, "mask": mask, "pi_after": pi_after}


def first_visit_residuals(q_hat, policy, reduced, gamma):
    st = np.asarray(reduced["states"], np.int64)
    ac = np.asarray(reduced["actions"], np.int64)
    rw = np.asarray(reduced["rewards"], np.float64)
    ns = np.asarray(reduced["next_states"], np.int64)
    pi = np.asarray(policy, np.float64)
    return rw + gamma * np.einsum("na,na->n", pi[ns], q_hat[ns]) - q_hat[st, ac]


def pair_statistics(residuals, flat, d):
    """Per-pair size / mean / variance(ddof=1) via a single grouping pass."""
    sizes = np.bincount(flat, minlength=d).astype(np.int64)
    sums = np.bincount(flat, weights=residuals, minlength=d)
    sums2 = np.bincount(flat, weights=residuals * residuals, minlength=d)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(sizes > 0, sums / np.maximum(sizes, 1), 0.0)
        # ddof=1 variance from the sum of squares
        var = np.where(sizes > 1,
                       (sums2 - sizes * means * means) / np.maximum(sizes - 1, 1), 0.0)
    return sizes, means, np.maximum(var, 0.0)


def make_networks(fam):
    return {
        "expected_exact": EndToEndMaskedSoftmaxExpectedSARSA(gamma=fam["gamma"], alpha=ALPHA),
        "expected_finite": EndToEndFiniteSoftmaxExpectedSARSA(
            gamma=fam["gamma"], alpha=ALPHA, zeta=ZETA, xi=XI, tau=TAU),
    }


def network_forward(net, policy, train):
    """One producer evaluation = LAYERS successive calls of the attention layer.

    Returns the Qhat and the number of *producer evaluations* (1), so the count is
    comparable with the author's recorded net_forwards; the layer count is reported
    separately as `layers_per_forward`.
    """
    st = torch.as_tensor(train["states"], dtype=torch.long)
    ac = torch.as_tensor(train["actions"], dtype=torch.long)
    rw = torch.as_tensor(train["rewards"], dtype=torch.float32)
    ns = torch.as_tensor(train["next_states"], dtype=torch.long)
    pi = torch.as_tensor(policy, dtype=torch.float32)
    q = torch.zeros(policy.shape, dtype=torch.float32)
    for _ in range(LAYERS):
        with torch.no_grad():
            q, _ = net(q, st, ac, rw, ns, pi)
    return q.detach().numpy().astype(np.float64), 1


def digest(*arrays) -> str:
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        m.update(str(a.dtype.str).encode()); m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


# --------------------------------------------------------------------------- #
def check(cond, tag, detail, problems, warn_only=False):
    if not cond:
        (problems.setdefault("warnings", []) if warn_only else problems["failures"]).append(
            {"check": tag, "detail": detail})
    return bool(cond)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="verify only the first N tasks")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    bundle = json.loads(args.results.read_text(encoding="utf-8"))
    manifest = None
    man_path = args.manifest or (args.results.parent / "input_manifest.json")
    if man_path.exists():
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    man_by_key = {}
    if manifest:
        for r in manifest["records"]:
            man_by_key[(r["family"], round(float(r["mixing"]), 6), int(r["task_index"]))] = r

    chains = int(bundle["chains"])
    chain_length = int(bundle["chain_length"])
    horizon = int(bundle["horizon"])
    seed, salt = int(bundle["seed"]), int(bundle["salt"])
    delta_step = float(bundle["delta_step"])
    delta_total = float(bundle["delta_total"])
    min_visits = int(bundle["min_visits"])
    producers = tuple(bundle["producers"])
    routes = tuple(bundle["routes"])
    cells = tuple(bundle["cells"])

    problems: dict[str, Any] = {"failures": [], "warnings": []}
    stats: dict[str, Any] = {
        "route_records": 0, "producer_steps": 0, "batch_hash_checks": 0,
        "batch_hash_mismatches": 0, "qhat_replayed": 0, "eq_replayed": 0,
        "abstentions_replayed": 0,
        "decisions_replayed": 0, "value_replays": 0, "net_forwards": 0,
        "max_qhat_diff": 0.0, "max_eq_diff": 0.0, "max_value_diff": 0.0,
        "max_closure_diff": 0.0, "max_sup_error_diff": 0.0,
        "max_direct_vs_iterate": 0.0,
        "decision_mismatches": [], "hash_mismatch_detail": [],
        "h1_coverage_violations": [], "h1_degrading_steps": [],
        "missing_recorded_steps": [],
    }
    t0 = time.time()
    seen_tasks: set = set()

    for rec in bundle["records"]:
        fam_name = rec["family"]
        seen_tasks.add((rec["family"], round(float(rec["mixing"]), 6), int(rec["task_index"])))
        fam = FAMILIES[fam_name]
        mixing = float(rec["mixing"])
        tidx = int(rec["task_index"])
        S, A = fam["n_states"], fam["n_actions"]
        d = S * A

        mdp, behaviour, rng = build_family_task(fam, mixing, tidx)
        mu = np.asarray(policy_quantities(mdp, behaviour)["mu_state"], dtype=np.float64)
        train = training_batch(mdp, behaviour, mu, rng)

        key = (fam_name, round(mixing, 6), tidx)
        if key in man_by_key:
            man = man_by_key[key]
            check(digest(np.asarray(mdp["P"]), np.asarray(mdp["R"])) == man["mdp_hash"],
                  "manifest_mdp_hash", {"family": fam_name, "mixing": mixing,
                                        "task_index": tidx}, problems)
            check(digest(train["states"], train["actions"], train["rewards"],
                         train["next_states"]) == man["train_hash"],
                  "manifest_train_hash", {"family": fam_name, "mixing": mixing,
                                          "task_index": tidx}, problems)
            check(digest(behaviour) == man["policy0_hash"],
                  "manifest_policy0_hash", {"family": fam_name, "mixing": mixing,
                                            "task_index": tidx}, problems)
        elif manifest is not None:
            problems["warnings"].append({"check": "manifest_record_missing",
                                         "detail": list(key)})

        q0, v0, direct_gap = solve_pairspace(mdp, behaviour, fam["gamma"])
        stats["max_direct_vs_iterate"] = max(stats["max_direct_vs_iterate"], direct_gap)
        check(direct_gap <= 1e-12, "direct_vs_iterate_agreement",
              {"task": list(key), "gap": direct_gap}, problems)
        vstar = value_star(mdp, fam["gamma"])
        gap_denom = float(np.sum(vstar) - np.sum(v0))
        if key in man_by_key:
            check(abs(float(np.sum(v0)) - man_by_key[key]["v0_sum"]) <= 1e-10,
                  "manifest_v0_sum", {"task": list(key)}, problems)
            check(abs(float(np.sum(vstar)) - man_by_key[key]["v_star_sum"]) <= 1e-10,
                  "manifest_vstar_sum", {"task": list(key)}, problems)

        # Rebuild each step's certification batch ONCE and keep only the first-visit
        # reduction plus its hash, so memory does not scale with the full raw batch.
        batches: dict[int, Any] = {}
        for step in range(1, horizon + 1):
            raw = vectorised_batch_generic(
                mdp, behaviour, mu,
                [seed, salt, int(round(100 * mixing)), tidx, step],
                chains, fam, chain_length)
            reduced, counts = first_visit_batch(raw, chain_length, n_states=S, n_actions=A)
            flat = (np.asarray(reduced["states"], np.int64) * A
                    + np.asarray(reduced["actions"], np.int64))
            batches[step] = {"reduced": reduced, "flat": flat,
                             "hash": digest(raw["states"][:1000], raw["next_states"][:1000]),
                             "n_items": int(np.asarray(raw["states"]).size)}
            del raw

        nets = make_networks(fam)
        for route in routes:
            for producer in producers:
                cellname = f"{producer}|perstate|L12"
                if cellname not in rec["routes"][route]:
                    continue
                sealed = rec["routes"][route][cellname]
                stats["route_records"] += 1
                pi = behaviour.copy()
                v = v0.copy()
                qref = q0.copy()
                for e in sealed["steps"]:
                    step = int(e["step"])
                    stats["producer_steps"] += 1
                    if step not in batches:
                        stats["missing_recorded_steps"].append(
                            {"task": list(key), "route": route, "producer": producer,
                             "step": step})
                        break
                    B = batches[step]
                    if e.get("batch_hash") is not None:
                        stats["batch_hash_checks"] += 1
                        if B["hash"] != e["batch_hash"]:
                            stats["batch_hash_mismatches"] += 1
                            stats["hash_mismatch_detail"].append(
                                {"task": list(key), "route": route, "producer": producer,
                                 "step": step, "expected": e["batch_hash"],
                                 "recomputed": B["hash"]})

                    # --- Qhat replay from the reconstructed policy ------------- #
                    if producer == "numpy":
                        qh = np.asarray(run_route(route, fam, pi, train),
                                        dtype=np.float64).reshape(S, A)
                    else:
                        qh, nf = network_forward(nets[route], pi, train)
                        stats["net_forwards"] += nf
                    recorded_q = np.asarray(e["q_hat"], dtype=np.float64)
                    dq = float(np.max(np.abs(qh - recorded_q)))
                    stats["max_qhat_diff"] = max(stats["max_qhat_diff"], dq)
                    stats["qhat_replayed"] += 1
                    check(dq <= QHAT_TOL, "qhat_replay",
                          {"task": list(key), "route": route, "producer": producer,
                           "step": step, "max_diff": dq}, problems)

                    # --- pi_before must match the reconstructed pi ------------- #
                    check(np.allclose(np.asarray(e["pi_before"], dtype=np.float64), pi,
                                      rtol=0.0, atol=1e-12), "pi_before_replay",
                          {"task": list(key), "route": route, "producer": producer,
                           "step": step}, problems, warn_only=True)

                    # --- concentration step from the SEALED statistics --------- #
                    if e["pair_sizes"] is None:
                        # No certificate was produced for this step, so there is no decision
                        # to replay: the only replayable claim is that the step abstained.
                        stats["abstentions_replayed"] += 1
                        check(not e["emitted"] and e["e_q"] is None, "abstain_replay",
                              {"task": list(key), "route": route, "producer": producer,
                               "step": step, "emitted": e["emitted"], "e_q": e["e_q"]},
                              problems)
                        dec = None
                    else:
                        sizes = np.asarray(e["pair_sizes"], dtype=np.int64)
                        cert = certificate_from_stats(
                            qh, pi, sizes, e["pair_means"], e["pair_vars"], delta_step,
                            fam["gamma"], fam["reward_bound"])
                        eq_diff = abs(cert["e_q"] - float(e["e_q"]))
                        stats["max_eq_diff"] = max(stats["max_eq_diff"], eq_diff)
                        stats["eq_replayed"] += 1
                        check(eq_diff <= VALUE_TOL, "eq_replay",
                              {"task": list(key), "route": route, "producer": producer,
                               "step": step, "abs_diff": eq_diff}, problems)

                        # recompute the pair statistics from the replay batch itself
                        resid = first_visit_residuals(qh, pi, B["reduced"], fam["gamma"])
                        rs, rm, rv = pair_statistics(resid, B["flat"], d)
                        check(np.array_equal(rs, sizes), "pair_sizes_replay",
                              {"task": list(key), "route": route, "producer": producer,
                               "step": step,
                               "max_abs_diff": int(np.max(np.abs(rs - sizes)))
                               if sizes is not None else None}, problems)
                        check(np.allclose(rm, np.asarray(e["pair_means"], dtype=np.float64),
                                          rtol=0.0, atol=1e-9), "pair_means_replay",
                              {"task": list(key), "route": route, "producer": producer,
                               "step": step}, problems)
                        check(np.allclose(rv, np.asarray(e["pair_vars"], dtype=np.float64),
                                          rtol=1e-7, atol=1e-9), "pair_vars_replay",
                              {"task": list(key), "route": route, "producer": producer,
                               "step": step}, problems, warn_only=True)

                        # --- decision replay -------------------------------------- #
                        dec = perstate_decide(pi, qh, cert["e_q"])
                        stats["decisions_replayed"] += 1
                        rec_table = np.asarray(e["lb_table"], dtype=np.float64) \
                            if e["lb_table"] else None
                        ok = True
                        if rec_table is None or rec_table.shape != dec["table"].shape:
                            ok = False
                        else:
                            ok = bool(np.allclose(rec_table, dec["table"], rtol=0.0,
                                                  atol=1e-9))
                        rows_ok = [None if c is None else float(c)
                                   for c in dec["rows_eta"]] == \
                            [None if c is None else float(c) for c in e["rows_eta"]]
                        mask_ok = dec["mask"].tolist() == list(e["update_mask"])
                        if not (ok and rows_ok and mask_ok):
                            stats["decision_mismatches"].append(
                                {"task": list(key), "route": route, "producer": producer,
                                 "step": step, "lb_table_match": ok,
                                 "rows_eta_match": rows_ok, "mask_match": mask_ok,
                                 "emitted_recorded": e["emitted"],
                                 "emitted_replayed": bool(dec["mask"].any())})
                        check(ok and rows_ok and mask_ok, "decision_replay",
                              {"task": list(key), "route": route, "producer": producer,
                               "step": step, "lb_table_match": ok,
                               "rows_eta_match": rows_ok, "mask_match": mask_ok}, problems)
                        check(bool(dec["mask"].any()) == bool(e["emitted"]),
                              "emitted_flag_replay",
                              {"task": list(key), "route": route, "producer": producer,
                               "step": step}, problems)

                        # --- H1 coverage against the verifier's own Q^pi ---------- #
                        realized = float(np.max(np.abs(qh - qref)))
                        stats["max_sup_error_diff"] = max(
                            stats["max_sup_error_diff"],
                            abs(realized - float(e["audit"]["realized_sup_error"])))
                        if cert["e_q"] < realized:
                            stats["h1_coverage_violations"].append(
                                {"task": list(key), "route": route, "producer": producer,
                                 "step": step, "e_q": cert["e_q"], "realized": realized})
                    # --- value audit ------------------------------------------- #
                    if e["emitted"] and dec is not None:
                        pi_new = np.asarray(dec["pi_after"], dtype=np.float64)
                        check(np.allclose(np.asarray(e["pi_after"], dtype=np.float64),
                                          pi_new, rtol=0.0, atol=1e-9), "pi_after_replay",
                              {"task": list(key), "route": route, "producer": producer,
                               "step": step}, problems)
                        q1, v1, g1 = solve_pairspace(mdp, pi_new, fam["gamma"])
                        stats["max_direct_vs_iterate"] = max(stats["max_direct_vs_iterate"],
                                                             g1)
                        dv = v1 - v
                        rec_dv = np.asarray(e["audit"]["value_delta"], dtype=np.float64)
                        dvd = float(np.max(np.abs(dv - rec_dv)))
                        stats["max_value_diff"] = max(stats["max_value_diff"], dvd)
                        stats["value_replays"] += 1
                        check(dvd <= VALUE_TOL, "value_delta_replay",
                              {"task": list(key), "route": route, "producer": producer,
                               "step": step, "max_diff": dvd}, problems)
                        if float(np.min(dv)) < -VALUE_TOL:
                            stats["h1_degrading_steps"].append(
                                {"task": list(key), "route": route, "producer": producer,
                                 "step": step, "min_value_delta": float(np.min(dv))})
                        cf = float(np.sum(dv)) / gap_denom
                        stats["max_closure_diff"] = max(
                            stats["max_closure_diff"],
                            abs(cf - float(e["audit"]["closure_fraction"])))
                        check(abs(cf - float(e["audit"]["closure_fraction"])) <= 1e-10,
                              "closure_fraction_replay",
                              {"task": list(key), "route": route, "producer": producer,
                               "step": step}, problems)
                        pi, v, qref = pi_new, v1, q1
        if args.limit and len(seen_tasks) >= args.limit:
            break
        print(f"  verified {fam_name} {mixing}/{tidx} "
              f"({time.time() - t0:.1f}s)", flush=True)

    stats["wall_seconds"] = time.time() - t0
    # cost accounting cross-check from the sealed bundle itself
    cost = bundle.get("cost", {})
    d2 = {
        "items_per_batch": cost.get("items_per_batch"),
        "expected_items_per_batch": chains * chain_length,
        "unique_batches": cost.get("unique_batches"),
        "recorded_producer_steps": stats["producer_steps"],
        "cells": list(cells),
        "layers_per_forward": LAYERS,
        "network_forwards_recorded": bundle.get("net_forwards"),
        "network_forwards_replayed": stats["net_forwards"],
    }
    if cost.get("items_per_batch") is not None:
        check(cost["items_per_batch"] == chains * chain_length, "d2_items_per_batch",
              d2, problems)
    if bundle.get("net_forwards") is not None:
        check(bundle["net_forwards"] == stats["net_forwards"], "d2_net_forwards",
              d2, problems, warn_only=True)

    hard = ["manifest_mdp_hash", "manifest_train_hash", "manifest_policy0_hash",
            "manifest_v0_sum", "manifest_vstar_sum", "direct_vs_iterate_agreement",
            "qhat_replay", "eq_replay", "pair_sizes_replay", "pair_means_replay",
            "decision_replay", "emitted_flag_replay", "value_delta_replay",
            "closure_fraction_replay", "abstain_replay", "pi_after_replay",
            "d2_items_per_batch"]
    hard_failures = [f for f in problems["failures"] if f["check"] in hard]
    verdict = "PASS" if not hard_failures else "FAIL"
    if stats["h1_coverage_violations"] or stats["h1_degrading_steps"]:
        verdict = "FAIL"

    report = {
        "verifier": "claude", "label": args.label,
        "verified_bundle": str(args.results), "verified_actor": bundle.get("actor"),
        "bundle_task_id": bundle.get("task_id"), "bundle_baseline": bundle.get("baseline"),
        "bundle_smoke": bundle.get("smoke", False),
        "thresholds": {"qhat_tol": QHAT_TOL, "value_tol": VALUE_TOL,
                       "note": "consistency thresholds only; never used to excuse H1"},
        "statistics": stats, "cost_cross_check": d2,
        "failures": problems["failures"][:60], "failure_count": len(problems["failures"]),
        "hard_failure_count": len(hard_failures),
        "warnings": problems["warnings"][:60], "warning_count": len(problems["warnings"]),
        "h1_coverage_violations": stats["h1_coverage_violations"][:40],
        "h1_degrading_steps": stats["h1_degrading_steps"][:40],
        "decision_mismatches": stats["decision_mismatches"][:40],
        "verdict": verdict,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "torch": torch.__version__},
        "scope_limits": [
            "The shared input generators (family MDP, sampler, first-visit extractor) are a "
            "common dependency, not an independent reimplementation; they were reviewed "
            "read-only and their limits are recorded separately.",
            "Agreement of a replay with a sealed record is a reproduction check, not a "
            "probabilistic proof, and does not by itself establish H1 as a theorem.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "hard_failures": len(hard_failures),
                      "failures": len(problems["failures"]),
                      "warnings": len(problems["warnings"]),
                      "h1_violations": len(stats["h1_coverage_violations"]),
                      "max_qhat_diff": stats["max_qhat_diff"],
                      "max_eq_diff": stats["max_eq_diff"],
                      "max_value_diff": stats["max_value_diff"],
                      "max_closure_diff": stats["max_closure_diff"],
                      "wall_s": round(stats["wall_seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
