"""FP-ITER-BOUND-001 evaluator: push the tightened certificate back into the loop.

Four arms run in lockstep over the same per-step certification batches:

    frozen   the current MP certificate                       (no kernel)
    L12      known-Qhat envelope + full risk budget           (no kernel)
    L12M     L12 + MODEL-FREE propagation                     (no kernel, MAIN ARM)
    L123     L12 + exact propagation                          (reads the kernel)

All four share the step-k batch, so the comparison is paired and carries no
sampler confound; each arm's own trajectory is certified separately, which is
what theorem 2 requires (`B_k` is independent of every arm's history because the
seed schedule is indexed by step alone).

Frozen by pre-registration: `K = 16`, `chains = 16384`, `delta_k = 0.05/K`,
population `task_index 12..23` x both mixings, both routes = 48 record-routes.
The decision rule and the eta grid are untouched.

    python evaluate_fp_iter_bound_001.py --output-dir results/FP-ITER-BOUND-001/claude/fresh_K16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_tight_certificate as tc  # noqa: E402
from evaluate_fp_bound_001 import CHAIN_LENGTH, FRESH_TASKS, TASK_SALT, gate_margin  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_certfix_first_n import first_visit_batch, step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402

TASK_ID = "FP-ITER-BOUND-001"
ARMS = ("frozen", "L12", "L12M", "L123")
HORIZON = 16
CHAINS = 16384
MIN_VISITS = 2000
DELTA_TOTAL = 0.05
ROUTES = ("expected_exact", "expected_finite")

SEALED_FILES = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "fixed_policy_bernstein_certificate.py",
    "fixed_policy_mp_certificate.py",
    "model.py",
)
NEW_FILES = (
    "fixed_policy_tight_certificate.py",
    "evaluate_fp_iter_bound_001.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run() -> dict[str, Any]:
    delta_step = DELTA_TOTAL / HORIZON
    records: list[dict[str, Any]] = []
    for mixing in fs.MIXING:
        for task_index in FRESH_TASKS:
            mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact0 = policy_quantities(mdp, behaviour)
            mu = np.asarray(exact0["mu_state"], dtype=np.float64)
            train = fs.training_batch(mdp, behaviour, mu, rng)
            P = np.asarray(mdp["P"], dtype=np.float64)
            R = np.asarray(mdp["R"], dtype=np.float64)
            routes: dict[str, Any] = {}
            for route in ROUTES:
                state = {
                    arm: {
                        "current": behaviour.copy(),
                        "v": np.asarray(exact0["v_pi"], dtype=np.float64).copy(),
                        "q_ref": np.asarray(exact0["q_pi"], dtype=np.float64).copy(),
                        "steps": [],
                        "stopped_at": None,
                    }
                    for arm in ARMS
                }
                for step in range(1, HORIZON + 1):
                    if all(s["stopped_at"] is not None for s in state.values()):
                        break
                    raw = vectorised_batch(
                        mdp,
                        behaviour,
                        mu,
                        step_seed_parts(fs.SEED, TASK_SALT, mixing, task_index, step),
                        CHAINS,
                        CHAIN_LENGTH,
                    )
                    reduced, _ = first_visit_batch(raw, CHAIN_LENGTH)
                    for arm in ARMS:
                        s = state[arm]
                        if s["stopped_at"] is not None:
                            continue
                        q_hat = np.asarray(
                            fs.run_route(route, s["current"], train)["q_hat"],
                            dtype=np.float64,
                        ).reshape(fs.N_STATES, fs.N_ACTIONS)
                        cert = tc.certificate(
                            q_hat,
                            s["current"],
                            reduced,
                            min_visits=MIN_VISITS,
                            delta_step=delta_step,
                            lever=arm,
                            transition=P,
                            reward=R,
                        )
                        decision = fs.improvement_for(s["current"], q_hat, cert)
                        realized = float(np.max(np.abs(q_hat - s["q_ref"])))
                        h = gate_margin(s["current"], q_hat)
                        emitted = decision["status"] == "safe_update_emitted"
                        entry: dict[str, Any] = {
                            "step": step,
                            "status": decision["status"],
                            "eta_selected": decision["eta_selected"],
                            "e_q": cert["e_q"],
                            "delta_step": delta_step,
                            "delta_each": cert["delta_each"],
                            "min_lb": float(np.min(decision["lb_by_state"])),
                            "h": h,
                            "e_q_over_h": (float(cert["e_q"]) / h if cert["e_q"] and h > 0 else None),
                            "emitted": emitted,
                            "ordered_reasons": fs.route_failure_reasons(cert, decision),
                            "realized_q_sup_error": realized,
                            "covers_realized": bool(
                                cert["e_q"] is not None and float(cert["e_q"]) >= realized
                            ),
                        }
                        if emitted:
                            nxt = np.asarray(decision["policy_plus"], dtype=np.float64)
                            q_next = policy_quantities(mdp, nxt)
                            v_next = np.asarray(q_next["v_pi"], dtype=np.float64)
                            delta_v = v_next - s["v"]
                            entry["value_delta"] = delta_v.tolist()
                            entry["componentwise_nondegrading"] = bool(
                                float(np.min(delta_v)) >= -1e-12
                            )
                            entry["total_value_gain"] = float(np.sum(delta_v))
                            s["v"] = v_next
                            s["q_ref"] = np.asarray(q_next["q_pi"], dtype=np.float64)
                            s["current"] = nxt
                        else:
                            s["stopped_at"] = step
                        s["steps"].append(entry)
                routes[route] = {
                    arm: {
                        "steps": state[arm]["steps"],
                        "emitted_steps": sum(1 for e in state[arm]["steps"] if e["emitted"]),
                        "stopped_at": state[arm]["stopped_at"],
                        "final_v_sum": float(np.sum(state[arm]["v"])),
                        "initial_v_sum": float(np.sum(exact0["v_pi"])),
                        "total_value_gain": float(
                            np.sum(state[arm]["v"]) - np.sum(exact0["v_pi"])
                        ),
                    }
                    for arm in ARMS
                }
            records.append(
                {
                    "task_id": TASK_ID,
                    "mixing": float(mixing),
                    "task_index": int(task_index),
                    "routes": routes,
                }
            )
            print(
                f"  {mixing}/{task_index:>2} "
                + "  ".join(
                    f"{arm}:{sum(1 for rt in routes.values() for e in rt[arm]['steps'] if e['emitted'])}"
                    for arm in ARMS
                ),
                flush=True,
            )
    return {
        "task_id": TASK_ID,
        "arms": list(ARMS),
        "routes": list(ROUTES),
        "fresh_tasks": list(FRESH_TASKS),
        "mixings": list(fs.MIXING),
        "horizon": HORIZON,
        "chains": CHAINS,
        "chain_length": CHAIN_LENGTH,
        "min_visits": MIN_VISITS,
        "delta_total": DELTA_TOTAL,
        "delta_step": delta_step,
        "task_salt": TASK_SALT,
        "record_count": len(records),
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
        json.dumps(
            {
                "task_id": TASK_ID,
                "label": args.label,
                "horizon": HORIZON,
                "chains": CHAINS,
                "chain_length": CHAIN_LENGTH,
                "min_visits": MIN_VISITS,
                "delta_total": DELTA_TOTAL,
                "delta_step": out["delta_step"],
                "fresh_tasks": list(FRESH_TASKS),
                "mixings": list(fs.MIXING),
                "arms": list(ARMS),
                "sealed_file_hashes": {n: sha256(PROJECT / n) for n in SEALED_FILES},
                "new_file_hashes": {n: sha256(PROJECT / n) for n in NEW_FILES},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps(
            {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    def agg(fn):
        return {arm: fn(arm) for arm in ARMS}

    summary = {
        "task_id": TASK_ID,
        "horizon": HORIZON,
        "records": out["record_count"],
        "total_emitted": agg(
            lambda arm: sum(
                rt[arm]["emitted_steps"] for r in out["records"] for rt in r["routes"].values()
            )
        ),
        "mean_trajectory_length": agg(
            lambda arm: float(
                np.mean(
                    [
                        len(rt[arm]["steps"])
                        for r in out["records"]
                        for rt in r["routes"].values()
                    ]
                )
            )
        ),
        "reach_horizon": agg(
            lambda arm: sum(
                1
                for r in out["records"]
                for rt in r["routes"].values()
                if rt[arm]["stopped_at"] is None
            )
        ),
        "mean_total_value_gain": agg(
            lambda arm: float(
                np.mean(
                    [
                        rt[arm]["total_value_gain"]
                        for r in out["records"]
                        for rt in r["routes"].values()
                    ]
                )
            )
        ),
        "coverage_violations": agg(
            lambda arm: sum(
                1
                for r in out["records"]
                for rt in r["routes"].values()
                for e in rt[arm]["steps"]
                if not e["covers_realized"]
            )
        ),
        "degradations": agg(
            lambda arm: sum(
                1
                for r in out["records"]
                for rt in r["routes"].values()
                for e in rt[arm]["steps"]
                if e["emitted"] and not e["componentwise_nondegrading"]
            )
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
