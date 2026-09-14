"""FP-RERUN-001 evaluator: the certified iteration with the SOUND arm `L12S`.

Hypotheses are registered in `docs/research_tasks/FP-RERUN-001.md` before this run.
Eight cells per record-route, sharing each step's certification batch:

    producer in {numpy, network}     network = model.py literal attention net, Q_0 = 0, LAYERS = 160
    lever    in {frozen, L12, L12S, L123}

`L12S` runs at the grid cell selected on step 1 in the repair note
(`split_fraction = 0.9`, `delta_prop_fraction = 0.05`). `L123` reads the kernel and
is an oracle reference only.

Cost instrumentation, which the audit asked for and the earlier runs lacked: every
step records the items that step's batch contains, so each cell's certification
data cost "if run alone" is derivable, and each record's wall clock is recorded.

    python evaluate_fp_rerun_001.py --output-dir results/FP-RERUN-001/claude/fresh_K16
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
from evaluate_fp_bound_001 import CHAIN_LENGTH, FRESH_TASKS, TASK_SALT, gate_margin  # noqa: E402
from evaluate_fp_certfix_001 import make_producer  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402

TASK_ID = "FP-RERUN-001"
HORIZON = 16
CHAINS = 16384
MIN_VISITS = 2000
DELTA_TOTAL = 0.05
SPLIT_FRACTION = 0.9
DELTA_PROP_FRACTION = 0.05
ROUTES = ("expected_exact", "expected_finite")
PRODUCERS = ("numpy", "network")
LEVERS = ("frozen", "L12", "L12S", "L123")
CELLS = tuple(f"{p}|{lv}" for p in PRODUCERS for lv in LEVERS)

SEALED_FILES = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "fixed_policy_bernstein_certificate.py",
    "fixed_policy_mp_certificate.py",
    "model.py",
)
NEW_FILES = ("fixed_policy_tight_certificate.py", "evaluate_fp_rerun_001.py")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run() -> dict[str, Any]:
    producers = {name: make_producer(name) for name in PRODUCERS}
    delta_step = DELTA_TOTAL / HORIZON
    items_per_step = CHAINS * CHAIN_LENGTH
    records: list[dict[str, Any]] = []
    started = time.time()
    for mixing in fs.MIXING:
        for task_index in FRESH_TASKS:
            t_rec = time.time()
            mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact0 = policy_quantities(mdp, behaviour)
            mu = np.asarray(exact0["mu_state"], dtype=np.float64)
            train = fs.training_batch(mdp, behaviour, mu, rng)
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
                    raw = vectorised_batch(
                        mdp, behaviour, mu,
                        step_seed_parts(fs.SEED, TASK_SALT, mixing, task_index, step),
                        CHAINS, CHAIN_LENGTH,
                    )
                    reduced, _ = first_visit_batch(raw, CHAIN_LENGTH)
                    for cell in CELLS:
                        s = state[cell]
                        if s["stopped_at"] is not None:
                            continue
                        producer, lever = cell.split("|")
                        q_hat = np.asarray(
                            producers[producer](route, s["current"], train),
                            dtype=np.float64,
                        ).reshape(fs.N_STATES, fs.N_ACTIONS)
                        cert = tc.certificate(
                            q_hat, s["current"], reduced,
                            min_visits=MIN_VISITS, delta_step=delta_step, lever=lever,
                            transition=P, reward=R,
                            delta_prop_fraction=DELTA_PROP_FRACTION,
                            split_fraction=SPLIT_FRACTION,
                        )
                        decision = fs.improvement_for(s["current"], q_hat, cert)
                        realized = float(np.max(np.abs(q_hat - s["q_ref"])))
                        h = gate_margin(s["current"], q_hat)
                        emitted = decision["status"] == "safe_update_emitted"
                        e_q = cert["e_q"]
                        entry: dict[str, Any] = {
                            "step": step,
                            "producer": producer,
                            "lever": lever,
                            "status": decision["status"],
                            "eta_selected": decision["eta_selected"],
                            "e_q": e_q,
                            "delta_step": delta_step,
                            "min_lb": float(np.min(decision["lb_by_state"])),
                            "h": h,
                            "e_q_over_h": (float(e_q) / h if e_q and h > 0 else None),
                            "emitted": emitted,
                            "ordered_reasons": fs.route_failure_reasons(cert, decision),
                            "realized_q_sup_error": realized,
                            "covers_realized": bool(e_q is not None and float(e_q) >= realized),
                            "items_this_step": items_per_step,
                            "q_hat": q_hat.tolist(),
                        }
                        if emitted:
                            nxt = np.asarray(decision["policy_plus"], dtype=np.float64)
                            q_next = policy_quantities(mdp, nxt)
                            v_next = np.asarray(q_next["v_pi"], dtype=np.float64)
                            dv = v_next - s["v"]
                            entry["value_delta"] = dv.tolist()
                            entry["componentwise_nondegrading"] = bool(
                                float(np.min(dv)) >= -1e-12
                            )
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
            records.append({
                "task_id": TASK_ID, "mixing": float(mixing), "task_index": int(task_index),
                "wall_seconds": time.time() - t_rec, "routes": routes,
            })
            print(f"  {mixing}/{task_index:>2} ({time.time() - t_rec:.1f}s) "
                  + "  ".join(
                      f"{c}:{sum(1 for rt in routes.values() for e in rt[c]['steps'] if e['emitted'])}"
                      for c in CELLS), flush=True)
    return {
        "task_id": TASK_ID,
        "cells": list(CELLS),
        "producers": list(PRODUCERS),
        "levers": list(LEVERS),
        "routes": list(ROUTES),
        "fresh_tasks": list(FRESH_TASKS),
        "mixings": list(fs.MIXING),
        "horizon": HORIZON,
        "chains": CHAINS,
        "chain_length": CHAIN_LENGTH,
        "items_per_step": items_per_step,
        "min_visits": MIN_VISITS,
        "delta_total": DELTA_TOTAL,
        "delta_step": delta_step,
        "split_fraction": SPLIT_FRACTION,
        "delta_prop_fraction": DELTA_PROP_FRACTION,
        "task_salt": TASK_SALT,
        "record_count": len(records),
        "wall_seconds_total": time.time() - started,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="fresh_K16")
    args = parser.parse_args()

    out = run()
    out["finished_utc"] = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_results.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps({
            "task_id": TASK_ID, "label": args.label, "horizon": HORIZON, "chains": CHAINS,
            "chain_length": CHAIN_LENGTH, "items_per_step": out["items_per_step"],
            "min_visits": MIN_VISITS, "delta_total": DELTA_TOTAL,
            "delta_step": out["delta_step"], "split_fraction": SPLIT_FRACTION,
            "delta_prop_fraction": DELTA_PROP_FRACTION, "cells": list(CELLS),
            "fresh_tasks": list(FRESH_TASKS), "mixings": list(fs.MIXING),
            "sealed_file_hashes": {n: sha256(PROJECT / n) for n in SEALED_FILES},
            "new_file_hashes": {n: sha256(PROJECT / n) for n in NEW_FILES},
        }, indent=2, sort_keys=True), encoding="utf-8",
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps({"python": sys.version, "platform": platform.platform(),
                    "numpy": np.__version__, "torch": _torch()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "task_id": TASK_ID,
        "records": out["record_count"],
        "wall_seconds_total": out["wall_seconds_total"],
        "total_emitted": {
            c: sum(rt[c]["emitted_steps"] for r in out["records"] for rt in r["routes"].values())
            for c in CELLS
        },
        "items_if_run_alone": {
            c: sum(rt[c]["items_if_run_alone"] for r in out["records"] for rt in r["routes"].values())
            for c in CELLS
        },
        "mean_total_value_gain": {
            c: float(np.mean([rt[c]["total_value_gain"] for r in out["records"]
                              for rt in r["routes"].values()]))
            for c in CELLS
        },
        "coverage_violations": {
            c: sum(1 for r in out["records"] for rt in r["routes"].values()
                   for e in rt[c]["steps"] if not e["covers_realized"])
            for c in CELLS
        },
        "degradations": {
            c: sum(1 for r in out["records"] for rt in r["routes"].values()
                   for e in rt[c]["steps"] if e["emitted"] and not e["componentwise_nondegrading"])
            for c in CELLS
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _torch() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:  # pragma: no cover
        return "unavailable"


if __name__ == "__main__":
    main()
