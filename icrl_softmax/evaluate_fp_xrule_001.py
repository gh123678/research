"""FP-XRULE-001 evaluator: update rule x certificate, on two NEW MDP families.

Registered in `docs/research_tasks/FP-XRULE-001.md` before this run. Families `f1`
(6 states x 4 actions, GAP_BONUS 0.5) and `f2` (4x3, GAP_BONUS 1.0, R* = 2.0) are
the ones `FP-XFAM-001` opened and no other task has used.

Four cells per route-record, sharing each step's batch:

    rule        in {conj (conjunctive gate), perstate (per-state conservative update)}
    certificate in {frozen, L12S(f=0.9, p=0.05)}

`perstate` is sound for the same reason the conjunctive rule is: with a row-wise
update `A_s = sum_a dpi_s(a) Q^pi(s,a) >= LB_s >= 0` for every state, and the
performance-difference identity then gives `v^{pi'}(s0) - v^pi(s0) = E[sum_t
gamma^t A_{s_t}] >= 0` componentwise -- strictly positive wherever an updated
state is reachable.

Everything dimension-generic is imported from `evaluate_fp_xfam_001` so the family
definitions and the sampler are the ones FP-XFAM-001 already exercised.

    python evaluate_fp_xrule_001.py --output-dir results/FP-XRULE-001/claude/f1f2_K4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

TASK_ID = "FP-XRULE-001"
HORIZON = 4
CHAINS = 16384
CHAIN_LENGTH = 64
MIN_VISITS = 2000
DELTA_TOTAL = 0.05
SPLIT_FRACTION = 0.9
DELTA_PROP_FRACTION = 0.05
ROUTES = ("expected_exact", "expected_finite")
RULES = ("conj", "perstate")
CERTS = ("frozen", "L12S")
CELLS = tuple(f"{r}|{c}" for r in RULES for c in CERTS)
TASK_SALT = 90417

SEALED_FILES = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "fixed_policy_bernstein_certificate.py",
    "fixed_policy_mp_certificate.py",
    "model.py",
)
NEW_FILES = ("fixed_policy_tight_certificate.py", "evaluate_fp_xrule_001.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def decide(rule: str, policy, q_hat, e_q, n_states: int):
    """The registered rules, normalised to a dict (both are dimension-generic)."""
    if rule == "conj":
        emitted, pplus, eta, lb = conjunctive_decision(policy, q_hat, e_q)
        return {
            "emitted": bool(emitted), "policy_plus": pplus, "eta_selected": eta,
            "lb_by_state": lb,
            "states_updated": int(np.sum(np.any(pplus != policy, axis=1))),
        }
    emitted, pplus, updated, lb = perstate_decision(policy, q_hat, e_q)
    return {
        "emitted": bool(emitted), "policy_plus": pplus, "eta_selected": None,
        "lb_by_state": lb, "states_updated": int(updated),
    }


def run_family(fam_name: str, fam: dict) -> list[dict[str, Any]]:
    delta_step = DELTA_TOTAL / HORIZON
    items_per_step = CHAINS * CHAIN_LENGTH
    out: list[dict[str, Any]] = []
    for mixing in fam["mixings"]:
        for task_index in fam["task_indices"]:
            t0 = time.time()
            mdp, behaviour, rng = build_family_task(fam, mixing, task_index)
            exact0 = policy_quantities(mdp, behaviour)
            mu = np.asarray(exact0["mu_state"], dtype=np.float64)
            train = training_batch(mdp, behaviour, mu, rng)
            S, A = behaviour.shape
            P = np.asarray(mdp["P"], dtype=np.float64)
            R = np.asarray(mdp["R"], dtype=np.float64)
            routes: dict[str, Any] = {}
            for route in ROUTES:
                state = {
                    cell: {
                        "current": behaviour.copy(),
                        "v": np.asarray(exact0["v_pi"], dtype=np.float64).copy(),
                        "q_ref": np.asarray(exact0["q_pi"], dtype=np.float64).copy(),
                        "steps": [],
                        "stopped_at": None,
                    }
                    for cell in CELLS
                }
                for step in range(1, HORIZON + 1):
                    if all(s["stopped_at"] is not None for s in state.values()):
                        break
                    raw = vectorised_batch_generic(
                        mdp, behaviour, mu,
                        step_seed_parts(fs.SEED, TASK_SALT, mixing, task_index, step),
                        CHAINS, fam, CHAIN_LENGTH,
                    )
                    reduced, _ = first_visit_batch(
                        raw, CHAIN_LENGTH, n_states=S, n_actions=A
                    )
                    for cell in CELLS:
                        s = state[cell]
                        if s["stopped_at"] is not None:
                            continue
                        rule, cert_name = cell.split("|")
                        q_hat = run_route(route, fam, s["current"], train).reshape(S, A)
                        cert = tc.certificate(
                            q_hat, s["current"], reduced,
                            min_visits=MIN_VISITS, delta_step=delta_step, lever=cert_name,
                            transition=P, reward=R, reward_bound=fam["reward_bound"],
                            gamma=fam["gamma"], n_states=S, n_actions=A,
                            delta_prop_fraction=DELTA_PROP_FRACTION,
                            split_fraction=SPLIT_FRACTION,
                        )
                        e_q = cert["e_q"]
                        if e_q is None:
                            decision = {"emitted": False, "policy_plus": s["current"].copy(),
                                        "eta_selected": None,
                                        "lb_by_state": np.zeros(S), "states_updated": 0}
                            reasons = list(cert["failure_reasons"])
                        else:
                            decision = decide(rule, s["current"], q_hat, float(e_q), S)
                            reasons = [] if decision["emitted"] else [
                                "improvement_lcb_nonpositive"
                            ]
                        realized = float(np.max(np.abs(q_hat - s["q_ref"])))
                        emitted = bool(decision["emitted"])
                        entry: dict[str, Any] = {
                            "step": step, "rule": rule, "certificate": cert_name,
                            "e_q": e_q, "delta_step": delta_step,
                            "min_lb": float(np.min(decision["lb_by_state"])),
                            "emitted": emitted,
                            "states_updated": int(decision["states_updated"]),
                            "ordered_reasons": reasons,
                            "realized_q_sup_error": realized,
                            "covers_realized": bool(e_q is not None and float(e_q) >= realized),
                            "items_this_step": items_per_step,
                        }
                        if emitted:
                            nxt = np.asarray(decision["policy_plus"], dtype=np.float64)
                            q_next = policy_quantities(mdp, nxt)
                            v_next = np.asarray(q_next["v_pi"], dtype=np.float64)
                            dv = v_next - s["v"]
                            entry["value_delta"] = dv.tolist()
                            entry["componentwise_nondegrading"] = bool(float(np.min(dv)) >= -1e-12)
                            entry["total_value_gain"] = float(np.sum(dv))
                            s["v"] = v_next
                            s["q_ref"] = np.asarray(q_next["q_pi"], dtype=np.float64)
                            s["current"] = nxt
                        else:
                            s["stopped_at"] = step
                        s["steps"].append(entry)
                routes[route] = {
                    cell: {
                        "steps": state[cell]["steps"],
                        "emitted_steps": sum(1 for e in state[cell]["steps"] if e["emitted"]),
                        "simulated_steps": len(state[cell]["steps"]),
                        "items_if_run_alone": len(state[cell]["steps"]) * items_per_step,
                        "stopped_at": state[cell]["stopped_at"],
                        "total_value_gain": float(
                            np.sum(state[cell]["v"]) - np.sum(exact0["v_pi"])
                        ),
                    }
                    for cell in CELLS
                }
            out.append({
                "task_id": TASK_ID, "family": fam_name, "mixing": float(mixing),
                "task_index": int(task_index), "wall_seconds": time.time() - t0,
                "routes": routes,
            })
            print(f"  {fam_name} {mixing}/{task_index} ({time.time() - t0:.1f}s) "
                  + "  ".join(
                      f"{c}:{sum(1 for rt in routes.values() for e in rt[c]['steps'] if e['emitted'])}"
                      for c in CELLS), flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="f1f2_K4")
    parser.add_argument("--families", default="f1,f2")
    args = parser.parse_args()

    started = time.time()
    records: list[dict[str, Any]] = []
    for fam_name in args.families.split(","):
        records += run_family(fam_name, FAMILIES[fam_name])
    bundle = {
        "task_id": TASK_ID, "cells": list(CELLS), "rules": list(RULES),
        "certificates": list(CERTS), "routes": list(ROUTES),
        "families": args.families.split(","), "horizon": HORIZON, "chains": CHAINS,
        "chain_length": CHAIN_LENGTH, "items_per_step": CHAINS * CHAIN_LENGTH,
        "min_visits": MIN_VISITS, "delta_total": DELTA_TOTAL,
        "delta_step": DELTA_TOTAL / HORIZON, "split_fraction": SPLIT_FRACTION,
        "delta_prop_fraction": DELTA_PROP_FRACTION, "task_salt": TASK_SALT,
        "record_count": len(records), "wall_seconds_total": time.time() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_results.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps({
            "task_id": TASK_ID, "label": args.label, "families": args.families.split(","),
            "family_specs": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                                 for kk, vv in FAMILIES[k].items() if kk != "rng"}
                             for k in args.families.split(",")},
            "horizon": HORIZON, "chains": CHAINS, "chain_length": CHAIN_LENGTH,
            "cells": list(CELLS), "min_visits": MIN_VISITS, "delta_total": DELTA_TOTAL,
            "delta_step": bundle["delta_step"], "split_fraction": SPLIT_FRACTION,
            "delta_prop_fraction": DELTA_PROP_FRACTION,
            "sealed_file_hashes": {n: sha256(PROJECT / n) for n in SEALED_FILES},
            "new_file_hashes": {n: sha256(PROJECT / n) for n in NEW_FILES},
        }, indent=2, sort_keys=True, default=str), encoding="utf-8",
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps({"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__}, indent=2, sort_keys=True), encoding="utf-8",
    )
    summary = {
        "task_id": TASK_ID, "records": bundle["record_count"],
        "wall_seconds_total": bundle["wall_seconds_total"],
        "by_family": {
            fam: {
                "total_emitted": {
                    c: sum(rt[c]["emitted_steps"] for r in records if r["family"] == fam
                           for rt in r["routes"].values())
                    for c in CELLS
                },
                "mean_value_gain": {
                    c: float(np.mean([rt[c]["total_value_gain"] for r in records
                                      if r["family"] == fam for rt in r["routes"].values()]))
                    for c in CELLS
                },
                "items_if_run_alone": {
                    c: sum(rt[c]["items_if_run_alone"] for r in records if r["family"] == fam
                           for rt in r["routes"].values())
                    for c in CELLS
                },
                "coverage_violations": {
                    c: sum(1 for r in records if r["family"] == fam
                           for rt in r["routes"].values() for e in rt[c]["steps"]
                           if not e["covers_realized"])
                    for c in CELLS
                },
                "degradations": {
                    c: sum(1 for r in records if r["family"] == fam
                           for rt in r["routes"].values() for e in rt[c]["steps"]
                           if e["emitted"] and not e["componentwise_nondegrading"])
                    for c in CELLS
                },
            }
            for fam in args.families.split(",")
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
