"""FP-EARLYSTOP-001: same-budget comparison of three remedies for early stopping,
on environments never used in any earlier task.

Pre-registered in ``docs/research_tasks/FP-EARLYSTOP-001.md`` BEFORE any run:

- New environments: task_index 12..23 (earlier tasks used 0..11 only), both
  mixings, 24 environments x 2 routes = 48 records.
- Protocol identical to FP-CERTFIX-001 (first-n fixed counts, fresh batch per
  step, delta_k = 0.05/12, K = 12, frozen eta grid). The only new ingredient is
  arm C's decision rule.
- Arm A: conjunctive gate (min_s LB_s > 0), n = 16,384 per pair per step.
- Arm B: conjunctive gate, n = 65,536 (4x data).
- Arm C: per-state first passage, n = 16,384 (same data as A): each state
  independently takes the first grid eta whose LB_s > 0; states with no passing
  eta are left unchanged. Soundness sketch (report §2): on the certificate
  event, updated states have true advantage >= LB_s > 0, untouched states have
  zero one-step advantage, so the performance difference is componentwise
  nonnegative (strict where an updated state is reachable).

One 4x batch is drawn per record-step (seed schedule of FP-CERTFIX-001 with this
task's salt); arms A/C use the first 16,384 per pair of it, arm B the first
65,536 -- so A and C see byte-identical certificate inputs.

Primary metrics (all reported per record, no survivor-only means): total value
gain, fraction of initial suboptimality closed (v* by exact policy iteration),
worst componentwise degradation (must be <= 0), false emissions (must be 0),
unique items drawn per record. Steps are a secondary descriptor.
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
import fixed_policy_mp_certificate as mc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_gap_001 import optimal_values  # noqa: E402
from fp_certfix_first_n import first_n_batch, step_seed_parts  # noqa: E402
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402

TASK_ID = "FP-EARLYSTOP-001"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("conj_n16k", "conj_n64k", "perstate_n16k")
TASK_SALT = 77531
DELTA_TOTAL = 0.05
MIXINGS = fs.MIXING
TASK_INDICES = tuple(range(12, 24))  # pre-registered NEW environments
CHAIN_LENGTH = 64
N_A = 16384
N_B = 65536
CHAINS = 65536  # 4x batch per record-step; A/C take the first 16384 per pair

ETA_GRID = fs.ETA_CANDIDATES


def strict_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): strict_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict_ready(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if not np.isfinite(value):
            raise ValueError("nonfinite value")
        return value
    if obj is None or isinstance(obj, str):
        return obj
    raise ValueError(f"unsupported {type(obj).__name__}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def conjunctive_decision(policy, q_hat, e_q):
    """Frozen rule: first eta whose min-state LB is strictly positive."""
    for eta in ETA_GRID:
        cand = fs.es.relative_softmax_candidate(policy, q_hat, eta)
        delta_pi = cand - policy
        i_hat = (delta_pi * q_hat).sum(axis=1)
        lb = i_hat - e_q * np.abs(delta_pi).sum(axis=1)
        if float(lb.min()) > 0.0:
            return {
                "emitted": True,
                "policy_plus": cand,
                "eta_selected": float(eta),
                "lb_by_state": lb,
                "states_updated": int(fs.N_STATES),
            }
    return {
        "emitted": False,
        "policy_plus": policy.copy(),
        "eta_selected": None,
        "lb_by_state": np.zeros(fs.N_STATES),
        "states_updated": 0,
    }


def perstate_decision(policy, q_hat, e_q):
    """Per-state first passage: tilt each state's row at the first eta with
    LB_s > 0; rows with no passing eta stay unchanged.

    Row-wise tilts compose exactly (the relative-softmax candidate is
    row-normalised), so the joint policy equals the row-wise choices.
    """
    new_policy = policy.copy()
    lb_by_state = np.zeros(fs.N_STATES)
    etas: list[float | None] = []
    updated = 0
    for s in range(fs.N_STATES):
        chosen = None
        chosen_lb = 0.0
        for eta in ETA_GRID:
            cand = fs.es.relative_softmax_candidate(policy, q_hat, eta)
            delta_row = cand[s] - policy[s]
            i_hat = float((delta_row * q_hat[s]).sum())
            l1 = float(np.abs(delta_row).sum())
            lb = i_hat - e_q * l1
            if lb > 0.0:
                chosen = float(eta)
                chosen_lb = lb
                new_policy[s] = cand[s]
                break
        etas.append(chosen)
        lb_by_state[s] = chosen_lb
        if chosen is not None:
            updated += 1
    return {
        "emitted": updated > 0,
        "policy_plus": new_policy,
        "eta_selected": etas,
        "lb_by_state": lb_by_state,
        "states_updated": updated,
    }


def run(args) -> dict[str, Any]:
    max_steps = int(args.max_steps)
    delta_step = DELTA_TOTAL / max_steps
    records: list[dict[str, Any]] = []
    items_drawn_total = 0
    for mixing in MIXINGS:
        for task_index in TASK_INDICES:
            mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact0 = policy_quantities(mdp, behaviour)
            mu_state = np.asarray(exact0["mu_state"], dtype=np.float64)
            v_star, _ = optimal_values(mdp)
            train = fs.training_batch(mdp, behaviour, mu_state, rng)

            routes: dict[str, Any] = {}
            for route in PRIMARY:
                # One 4x batch per record-step, shared across arms (same seed
                # schedule -> byte-identical inputs; arms differ only in rule
                # and in how many items per pair they consume).
                step_batches: list[Any] = []
                arms_out: dict[str, Any] = {}
                for arm in ARMS:
                    n_arm = N_B if arm == "conj_n64k" else N_A
                    current = behaviour.copy()
                    q_ref = np.asarray(exact0["q_pi"], dtype=np.float64).copy()
                    v_chain = [np.asarray(exact0["v_pi"], dtype=np.float64).copy()]
                    steps: list[dict[str, Any]] = []
                    for step_index in range(1, max_steps + 1):
                        q_hat = np.asarray(
                            fs.run_route(route, current, train)["q_hat"],
                            dtype=np.float64,
                        ).reshape(fs.N_STATES, fs.N_ACTIONS)
                        if step_index > len(step_batches):
                            raw = vectorised_batch(
                                mdp,
                                behaviour,
                                mu_state,
                                step_seed_parts(
                                    fs.SEED, TASK_SALT, mixing, task_index, step_index
                                ),
                                CHAINS,
                                CHAIN_LENGTH,
                            )
                            items_drawn_total += CHAINS * CHAIN_LENGTH
                            step_batches.append(raw)
                        raw = step_batches[step_index - 1]
                        reduced, counts = first_n_batch(raw, n_arm)
                        realized = float(np.max(np.abs(q_hat - q_ref)))
                        if reduced is None:
                            e_q = None
                            decision = {
                                "emitted": False,
                                "policy_plus": current.copy(),
                                "eta_selected": None,
                                "lb_by_state": np.zeros(fs.N_STATES),
                                "states_updated": 0,
                            }
                            reasons = ["heldout_pair_support_missing"]
                        else:
                            cert = mc.mp_certificate(
                                q_hat,
                                current,
                                reduced,
                                n_per_pair=n_arm,
                                delta_step=delta_step,
                            )
                            e_q = cert.get("e_q")
                            if e_q is None:
                                decision = {
                                    "emitted": False,
                                    "policy_plus": current.copy(),
                                    "eta_selected": None,
                                    "lb_by_state": np.zeros(fs.N_STATES),
                                    "states_updated": 0,
                                }
                                reasons = list(cert.get("failure_reasons", []))
                            else:
                                decision = (
                                    conjunctive_decision(current, q_hat, e_q)
                                    if arm != "perstate_n16k"
                                    else perstate_decision(current, q_hat, e_q)
                                )
                                reasons = (
                                    []
                                    if decision["emitted"]
                                    else ["improvement_lcb_nonpositive"]
                                )
                        emitted = bool(decision["emitted"])
                        lb = np.asarray(decision["lb_by_state"], dtype=np.float64)
                        entry: dict[str, Any] = {
                            "step": step_index,
                            "emitted": emitted,
                            "eta_selected": decision["eta_selected"],
                            "states_updated": int(decision["states_updated"]),
                            "e_q": e_q,
                            "min_lb": float(lb.min()) if lb.size else None,
                            "min_pair_count_observed": int(counts.min()),
                            "ordered_reasons": reasons,
                            "oracle_audit": {
                                "purpose": "truth-based audit only; never a certificate input",
                                "realized_q_sup_error_vs_current_target_pi": realized,
                                "certificate_violation": bool(
                                    e_q is not None and float(e_q) < realized
                                ),
                            },
                        }
                        if emitted:
                            nxt = np.asarray(decision["policy_plus"], dtype=np.float64)
                            q_next = policy_quantities(mdp, nxt)
                            v_next = np.asarray(q_next["v_pi"], dtype=np.float64)
                            delta_v = v_next - v_chain[-1]
                            entry["oracle_audit"].update(
                                {
                                    "value_delta_vs_previous": delta_v.tolist(),
                                    "componentwise_nondegrading": bool(
                                        float(np.min(delta_v)) >= -1e-12
                                    ),
                                    "total_value_gain": float(np.sum(delta_v)),
                                }
                            )
                            v_chain.append(v_next)
                            q_ref = np.asarray(q_next["q_pi"], dtype=np.float64)
                            current = nxt
                        steps.append(entry)
                        if not emitted:
                            break
                    v_final = v_chain[-1]
                    init_gap = float(np.sum(v_star - v_chain[0]))
                    arms_out[arm] = {
                        "steps": steps,
                        "emitted_steps": sum(1 for s in steps if s["emitted"]),
                        "initial_v_sum": float(np.sum(v_chain[0])),
                        "final_v_sum": float(np.sum(v_final)),
                        "initial_suboptimality": init_gap,
                        "fraction_gap_closed": (
                            float(np.sum(v_final - v_chain[0])) / init_gap
                            if init_gap > 0
                            else 1.0
                        ),
                        "final_gap_per_state": (v_star - v_final).tolist(),
                    }
                routes[route] = arms_out
            records.append(
                {
                    "task_id": TASK_ID,
                    "mixing": float(mixing),
                    "task_index": int(task_index),
                    "environment_is_new": True,
                    "routes": routes,
                }
            )
    return {
        "task_id": TASK_ID,
        "label": args.label,
        "arms": list(ARMS),
        "routes": list(PRIMARY),
        "task_indices": list(TASK_INDICES),
        "mixings": list(MIXINGS),
        "n_per_pair": {"conj_n16k": N_A, "conj_n64k": N_B, "perstate_n16k": N_A},
        "chains_per_step": CHAINS,
        "chain_length": CHAIN_LENGTH,
        "delta_total": DELTA_TOTAL,
        "max_steps": max_steps,
        "delta_step": DELTA_TOTAL / max_steps,
        "task_salt": TASK_SALT,
        "items_drawn_total": items_drawn_total,
        "record_count": len(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    global TASK_INDICES
    if args.smoke:
        TASK_INDICES = (12, 13)

    started = datetime.now(timezone.utc)
    out = run(args)
    out["started_utc"] = started.isoformat()
    out["finished_utc"] = datetime.now(timezone.utc).isoformat()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task_results.json").write_text(
        json.dumps(strict_ready(out), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "label": args.label,
                "max_steps": int(args.max_steps),
                "task_indices": list(TASK_INDICES),
                "mixings": list(MIXINGS),
                "arms": list(ARMS),
                "task_salt": TASK_SALT,
                "file_hashes": {
                    name: sha256(PROJECT / name)
                    for name in (
                        "fixed_policy_mp_certificate.py",
                        "fp_certfix_first_n.py",
                        "evaluate_fp_earlystop_001.py",
                        "fp_sample_vectorised_batch.py",
                        "evaluate_fp_gap_001.py",
                    )
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "records": out["record_count"],
        "items_drawn_total": out["items_drawn_total"],
        "per_arm": {
            arm: {
                "emitted_steps": sum(
                    r["routes"][route][arm]["emitted_steps"]
                    for r in out["records"]
                    for route in PRIMARY
                ),
                "records_emitting_step1": sum(
                    1
                    for r in out["records"]
                    for route in PRIMARY
                    if r["routes"][route][arm]["steps"]
                    and r["routes"][route][arm]["steps"][0]["emitted"]
                ),
                "mean_total_gain": float(
                    np.mean(
                        [
                            r["routes"][route][arm]["final_v_sum"]
                            - r["routes"][route][arm]["initial_v_sum"]
                            for r in out["records"]
                            for route in PRIMARY
                        ]
                    )
                ),
                "mean_fraction_gap_closed": float(
                    np.mean(
                        [
                            r["routes"][route][arm]["fraction_gap_closed"]
                            for r in out["records"]
                            for route in PRIMARY
                        ]
                    )
                ),
                "certificate_violations": sum(
                    1
                    for r in out["records"]
                    for route in PRIMARY
                    for s in r["routes"][route][arm]["steps"]
                    if s["oracle_audit"]["certificate_violation"]
                ),
                "componentwise_degrading_steps": sum(
                    1
                    for r in out["records"]
                    for route in PRIMARY
                    for s in r["routes"][route][arm]["steps"]
                    if s["emitted"]
                    and not s["oracle_audit"].get("componentwise_nondegrading", True)
                ),
            }
            for arm in ARMS
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
