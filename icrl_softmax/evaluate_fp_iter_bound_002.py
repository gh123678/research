"""FP-ITER-BOUND-002 evaluator: the K schedule, paired across horizons.

Hypotheses and bands were registered in `docs/research_tasks/FP-ITER-BOUND-002.md`
before this run. `HORIZONS = (4, 8, 16)` share each step's certification batch --
drawn once per (record, step) from the step-indexed seed schedule -- while each
trajectory evolves independently, so the comparison across `K` is paired and
carries no sampler confound.

Arms:
    frozen@K16                  control, delta_k = 0.05/16
    L12M(0.05)@K16              the final model-free arm at the registered horizon
    L12M(0.05)@K8               delta_k = 0.05/8
    L12M(0.05)@K4               delta_k = 0.05/4

Each trajectory stops at its own first abstention, or is truncated at its own `K`.
The guarantee of the `K=4` cell covers four steps and is quoted only as such.
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

TASK_ID = "FP-ITER-BOUND-002"
HORIZONS = (4, 8, 16)
DELTA_TOTAL = 0.05
CHAINS = 16384
MIN_VISITS = 2000
DELTA_PROP_FRACTION = 0.05
ROUTES = ("expected_exact", "expected_finite")
# (arm label, lever, horizon)
ARMS = (
    ("frozen@K16", "frozen", 16),
    ("L12M05@K16", "L12M", 16),
    ("L12M05@K8", "L12M", 8),
    ("L12M05@K4", "L12M", 4),
)

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
    "evaluate_fp_iter_bound_002.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    max_horizon = max(h for _, _, h in ARMS)
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
                    label: {
                        "current": behaviour.copy(),
                        "v": np.asarray(exact0["v_pi"], dtype=np.float64).copy(),
                        "q_ref": np.asarray(exact0["q_pi"], dtype=np.float64).copy(),
                        "steps": [],
                        "stopped_at": None,
                    }
                    for label, _, _ in ARMS
                }
                for step in range(1, max_horizon + 1):
                    active = [
                        (label, lever, K)
                        for label, lever, K in ARMS
                        if K >= step and state[label]["stopped_at"] is None
                    ]
                    if not active:
                        break
                    raw = vectorised_batch(
                        mdp, behaviour, mu,
                        step_seed_parts(fs.SEED, TASK_SALT, mixing, task_index, step),
                        CHAINS, CHAIN_LENGTH,
                    )
                    reduced, _ = first_visit_batch(raw, CHAIN_LENGTH)
                    for label, lever, K in active:
                        s = state[label]
                        delta_step = DELTA_TOTAL / K
                        q_hat = np.asarray(
                            fs.run_route(route, s["current"], train)["q_hat"],
                            dtype=np.float64,
                        ).reshape(fs.N_STATES, fs.N_ACTIONS)
                        cert = tc.certificate(
                            q_hat, s["current"], reduced,
                            min_visits=MIN_VISITS, delta_step=delta_step, lever=lever,
                            transition=P, reward=R,
                            delta_prop_fraction=DELTA_PROP_FRACTION,
                        )
                        decision = fs.improvement_for(s["current"], q_hat, cert)
                        realized = float(np.max(np.abs(q_hat - s["q_ref"])))
                        h = gate_margin(s["current"], q_hat)
                        emitted = decision["status"] == "safe_update_emitted"
                        e_q = cert["e_q"]
                        entry: dict[str, Any] = {
                            "step": step,
                            "K": K,
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
                    label: {
                        "steps": state[label]["steps"],
                        "emitted_steps": sum(
                            1 for e in state[label]["steps"] if e["emitted"]
                        ),
                        "stopped_at": state[label]["stopped_at"],
                        "truncated": state[label]["stopped_at"] is None,
                        "horizon": K,
                        "final_v_sum": float(np.sum(state[label]["v"])),
                        "total_value_gain": float(
                            np.sum(state[label]["v"]) - np.sum(exact0["v_pi"])
                        ),
                    }
                    for label, _, K in ARMS
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
                    f"{lb}:{sum(1 for rt in routes.values() for e in rt[lb]['steps'] if e['emitted'])}"
                    for lb, _, _ in ARMS
                ),
                flush=True,
            )
    return {
        "task_id": TASK_ID,
        "arms": [lb for lb, _, _ in ARMS],
        "arm_spec": {lb: {"lever": lv, "horizon": K} for lb, lv, K in ARMS},
        "routes": list(ROUTES),
        "fresh_tasks": list(FRESH_TASKS),
        "mixings": list(fs.MIXING),
        "horizons": list(HORIZONS),
        "chains": CHAINS,
        "chain_length": CHAIN_LENGTH,
        "min_visits": MIN_VISITS,
        "delta_total": DELTA_TOTAL,
        "delta_prop_fraction": DELTA_PROP_FRACTION,
        "task_salt": TASK_SALT,
        "record_count": len(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="fresh_k4k8k16")
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
                "arm_spec": out["arm_spec"],
                "chains": CHAINS,
                "chain_length": CHAIN_LENGTH,
                "min_visits": MIN_VISITS,
                "delta_total": DELTA_TOTAL,
                "delta_prop_fraction": DELTA_PROP_FRACTION,
                "fresh_tasks": list(FRESH_TASKS),
                "mixings": list(fs.MIXING),
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
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    labels = [lb for lb, _, _ in ARMS]
    summary = {
        "task_id": TASK_ID,
        "records": out["record_count"],
        "total_emitted": {
            lb: sum(
                rt[lb]["emitted_steps"] for r in out["records"] for rt in r["routes"].values()
            )
            for lb in labels
        },
        "mean_trajectory_length": {
            lb: float(
                np.mean(
                    [len(rt[lb]["steps"]) for r in out["records"] for rt in r["routes"].values()]
                )
            )
            for lb in labels
        },
        "reached_horizon": {
            lb: sum(
                1 for r in out["records"] for rt in r["routes"].values() if rt[lb]["truncated"]
            )
            for lb in labels
        },
        "mean_total_value_gain": {
            lb: float(
                np.mean(
                    [rt[lb]["total_value_gain"] for r in out["records"] for rt in r["routes"].values()]
                )
            )
            for lb in labels
        },
        "coverage_violations": {
            lb: sum(
                1
                for r in out["records"]
                for rt in r["routes"].values()
                for e in rt[lb]["steps"]
                if not e["covers_realized"]
            )
            for lb in labels
        },
        "degradations": {
            lb: sum(
                1
                for r in out["records"]
                for rt in r["routes"].values()
                for e in rt[lb]["steps"]
                if e["emitted"] and not e["componentwise_nondegrading"]
            )
            for lb in labels
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
