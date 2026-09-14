"""FP-NETX-002 evaluator: the attention network on the two NEW MDP families.

Registered in `docs/research_tasks/FP-NETX-002.md` before this run. Families,
environments, seeds, routes, `K = 4`, per-step budget and the batch schedule are
identical to FP-XRULE-001; two things change:

* the rule is fixed to `perstate`, so the cells are `producer x certificate`;
* a `network` producer is added -- model.py's two `EndToEnd*ExpectedSARSA` modules,
  run from `Q_0 = 0` for `LAYERS` layers through a dimension-generic port of
  `evaluate_fp_attn_iter_001.network_qhat` (the original hardcodes 4x3).

Because everything else matches FP-XRULE-001, the two `numpy` cells must reproduce
that task's `perstate|*` numbers bit for bit; the analyzer checks it, and a failure
invalidates this run.

    python evaluate_fp_netxfam_001.py --output-dir results/FP-NETX-002/claude/f1f2_K4
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_tight_certificate as tc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_xfam_001 import (  # noqa: E402
    FAMILIES,
    build_family_task,
    perstate_decision,
    run_route,
    training_batch,
    vectorised_batch_generic,
)
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402
from model import (  # noqa: E402
    EndToEndFiniteSoftmaxExpectedSARSA,
    EndToEndMaskedSoftmaxExpectedSARSA,
)

TASK_ID = "FP-NETX-002"
HORIZON = 4
CHAINS = 16384
CHAIN_LENGTH = 64
MIN_VISITS = 2000
DELTA_TOTAL = 0.05
SPLIT_FRACTION = 0.9
DELTA_PROP_FRACTION = 0.05
ROUTES = ("expected_exact", "expected_finite")
PRODUCERS = ("numpy", "network")
CERTS = ("frozen", "L12S")
CELLS = tuple(f"{p}|{c}" for p in PRODUCERS for c in CERTS)
TASK_SALT = 90417


def make_networks(fam: dict) -> dict[str, torch.nn.Module]:
    return {
        "expected_exact": EndToEndMaskedSoftmaxExpectedSARSA(
            gamma=fam["gamma"], alpha=fs.ALPHA
        ),
        "expected_finite": EndToEndFiniteSoftmaxExpectedSARSA(
            gamma=fam["gamma"], alpha=fs.ALPHA, zeta=fs.ZETA, xi=fs.XI, tau=fs.TAU
        ),
    }


def network_qhat_generic(
    network: torch.nn.Module, policy: np.ndarray, train: dict[str, Any]
) -> np.ndarray:
    """`network_qhat` without the 4x3 hardcoding: Q_0 = 0 of the policy's shape."""
    states = torch.as_tensor(train["states"], dtype=torch.long)
    actions = torch.as_tensor(train["actions"], dtype=torch.long)
    rewards = torch.as_tensor(train["rewards"], dtype=torch.float32)
    next_states = torch.as_tensor(train["next_states"], dtype=torch.long)
    policy_t = torch.as_tensor(policy, dtype=torch.float32)
    q = torch.zeros(policy.shape, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(fs.LAYERS):
            q, _ = network(q, states, actions, rewards, next_states, policy_t)
    return q.detach().numpy().astype(np.float64)


def run_family(fam_name: str, fam: dict) -> list[dict[str, Any]]:
    delta_step = DELTA_TOTAL / HORIZON
    items_per_step = CHAINS * CHAIN_LENGTH
    networks = make_networks(fam)
    out: list[dict[str, Any]] = []
    for mixing in fam["mixings"]:
        for task_index in fam["task_indices"]:
            t0 = time.time()
            calls = 0
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
                    reduced, _ = first_visit_batch(raw, CHAIN_LENGTH, n_states=S, n_actions=A)
                    for cell in CELLS:
                        s = state[cell]
                        if s["stopped_at"] is not None:
                            continue
                        producer, cert_name = cell.split("|")
                        if producer == "numpy":
                            q_hat = run_route(route, fam, s["current"], train).reshape(S, A)
                        else:
                            q_hat = network_qhat_generic(
                                networks[route], s["current"], train
                            ).reshape(S, A)
                            calls += 1
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
                            emitted, pplus, lb, updated = (
                                False, s["current"].copy(), np.zeros(S), 0
                            )
                            reasons = list(cert["failure_reasons"])
                        else:
                            emitted, pplus, updated, lb = perstate_decision(
                                s["current"], q_hat, float(e_q)
                            )
                            reasons = [] if emitted else ["improvement_lcb_nonpositive"]
                        realized = float(np.max(np.abs(q_hat - s["q_ref"])))
                        entry: dict[str, Any] = {
                            "step": step, "producer": producer, "certificate": cert_name,
                            "e_q": e_q, "delta_step": delta_step,
                            "min_lb": float(np.min(lb)), "emitted": bool(emitted),
                            "states_updated": int(updated), "ordered_reasons": reasons,
                            "realized_q_sup_error": realized,
                            "covers_realized": bool(e_q is not None and float(e_q) >= realized),
                            "items_this_step": items_per_step,
                            "q_hat": q_hat.tolist(),
                        }
                        if emitted:
                            nxt = np.asarray(pplus, dtype=np.float64)
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
                "network_calls": calls, "routes": routes,
            })
            print(f"  {fam_name} {mixing}/{task_index} ({time.time() - t0:.1f}s, "
                  f"{calls} net calls) "
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
        "task_id": TASK_ID, "cells": list(CELLS), "producers": list(PRODUCERS),
        "certificates": list(CERTS), "routes": list(ROUTES),
        "families": args.families.split(","), "horizon": HORIZON, "chains": CHAINS,
        "chain_length": CHAIN_LENGTH, "items_per_step": CHAINS * CHAIN_LENGTH,
        "min_visits": MIN_VISITS, "delta_total": DELTA_TOTAL,
        "delta_step": DELTA_TOTAL / HORIZON, "split_fraction": SPLIT_FRACTION,
        "delta_prop_fraction": DELTA_PROP_FRACTION, "task_salt": TASK_SALT,
        "record_count": len(records), "wall_seconds_total": time.time() - started,
        "network_calls_total": sum(r["network_calls"] for r in records),
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
            "horizon": HORIZON, "chains": CHAINS, "cells": list(CELLS),
            "network_calls_total": bundle["network_calls_total"],
            "delta_step": bundle["delta_step"], "split_fraction": SPLIT_FRACTION,
            "delta_prop_fraction": DELTA_PROP_FRACTION,
        }, indent=2, sort_keys=True), encoding="utf-8",
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps({"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__, "torch": torch.__version__},
                   indent=2, sort_keys=True), encoding="utf-8",
    )
    print(f"done: {len(records)} records, {bundle['network_calls_total']} network calls, "
          f"{bundle['wall_seconds_total'] / 60:.1f} min")


if __name__ == "__main__":
    main()
