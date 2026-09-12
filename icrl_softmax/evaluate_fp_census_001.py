"""FP-CENSUS-001: eligibility census of the certified iteration.

Walks the policy trajectory that FP-ITER5-001 produced for each of the 48
route-records, and at EVERY step — including abstaining ones — records

    sigma_min(k) = min_s ptp_a Q^{pi_k}(s,a)   (oracle side, diagnostic only)
    E_Q(k)       = the frozen variance-adaptive certificate at pi_k
    ratio(k)     = sigma_min(k) / E_Q(k)
    decision(k)  = emitted, or the frozen ordered abstention reason

so that the 26 records which never emitted still contribute their step-1 data.

The sealed FP-ITER5-001 numpy bundle is the ground truth for the emission
classification and is reproduced exactly before any ratio is interpreted.

Usage:
    python -B evaluate_fp_census_001.py --output-dir <dir> [--tasks N]
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
import fixed_policy_variance_certificate as vc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402

TASK_ID = "FP-CENSUS-001"
ITER5_BUNDLE = (
    PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy" / "task_results.json"
)
PRIMARY = ("expected_exact", "expected_finite")
MAX_STEPS = 5
MIXINGS = (0.08, 0.5)
TASKS = 12
# FP-SCALE-002 certification constants (``fs`` carries FP-SCALE-001 values).
CERT_CHAINS = 16384
CERT_CHAIN_LENGTH = 64
SEALED_FILES = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "evaluate_fp_scale_002.py",
    "evaluate_fp_attn_001.py",
    "evaluate_fp_iter2_001.py",
    "model.py",
)


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


def certification_batch(
    mdp: Any, policy: np.ndarray, mu_state: np.ndarray, seed_parts: list[Any]
) -> dict[str, np.ndarray]:
    """Certification batch, byte-for-byte the generator FP-ITER5-001 sealed.

    The draw order is per chain, interleaving one action and one next-state per
    step, and it must stay that way: a vectorised per-step rewrite of this loop
    was tried and checked against the sealed ``FP-SCALE-002`` pair counts, and
    it produced different counts on all four probed records (``19531`` vs
    ``19728``, ``25167`` vs ``25586``, ``31673`` vs ``31583``, ``31871`` vs
    ``32381``). ``Generator.choice`` with an explicit ``p`` does not consume the
    underlying stream the way a raw uniform draw does, so the census keeps the
    literal loop in order to sit on the exact batches the sealed iteration used.
    """
    rng = np.random.default_rng(seed_parts)
    chains, chain_length = CERT_CHAINS, CERT_CHAIN_LENGTH
    total = chains * chain_length
    starts = rng.choice(fs.N_STATES, size=chains, p=mu_state)
    states = np.empty(total, dtype=np.int64)
    actions = np.empty(total, dtype=np.int64)
    rewards = np.empty(total, dtype=np.float64)
    next_states = np.empty(total, dtype=np.int64)
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    for chain in range(chains):
        state = int(starts[chain])
        base = chain * chain_length
        for step in range(chain_length):
            action = int(rng.choice(fs.N_ACTIONS, p=policy[state]))
            following = int(rng.choice(fs.N_STATES, p=transition[state, action]))
            states[base + step] = state
            actions[base + step] = action
            rewards[base + step] = reward[state, action, following]
            next_states[base + step] = following
            state = following
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "next_actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=TASKS)
    parser.add_argument("--mixings", type=str, default="0.08,0.5")
    parser.add_argument("--label", type=str, default="eligibility-census")
    args = parser.parse_args()

    mixings = tuple(float(v) for v in args.mixings.split(","))
    sealed = json.loads(ITER5_BUNDLE.read_text(encoding="utf-8"))
    sealed_by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
    }

    records: list[dict[str, Any]] = []
    for mixing in mixings:
        for task_index in range(args.tasks):
            mdp, policy, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact = policy_quantities(mdp, policy)
            mu_state = np.asarray(exact["mu_state"], dtype=np.float64)
            train = fs.training_batch(mdp, policy, mu_state, rng)
            cert = certification_batch(
                mdp,
                policy,
                mu_state,
                [fs.SEED, 9001, int(round(mixing * 100)), task_index],
            )
            sealed_record = sealed_by_key.get((mixing, task_index))

            routes: dict[str, Any] = {}
            for route in PRIMARY:
                current_policy = policy.copy()
                steps: list[dict[str, Any]] = []
                for step_index in range(1, MAX_STEPS + 1):
                    # --- the frozen certificate at the CURRENT policy ----------
                    q_hat = np.asarray(
                        fs.run_route(route, current_policy, train)["q_hat"],
                        dtype=np.float64,
                    ).reshape(fs.N_STATES, fs.N_ACTIONS)
                    certificate = vc.variance_adaptive_certificate(
                        q_hat, current_policy, cert
                    )
                    decision = fs.improvement_for(
                        current_policy, q_hat, certificate
                    )
                    emitted = decision["status"] == "safe_update_emitted"

                    # --- oracle-side diagnostic yardstick ---------------------
                    exact_current = policy_quantities(mdp, current_policy)
                    q_pi = np.asarray(exact_current["q_pi"], dtype=np.float64)
                    v_pi = np.asarray(exact_current["v_pi"], dtype=np.float64)
                    spreads = np.ptp(q_pi, axis=1)
                    sigma_min = float(np.min(spreads))
                    e_q = certificate.get("e_q")
                    ratio = (
                        None
                        if e_q is None or float(e_q) <= 0.0
                        else sigma_min / float(e_q)
                    )
                    lb = np.asarray(
                        decision["lb_by_state"], dtype=np.float64
                    ).reshape(-1)

                    entry: dict[str, Any] = {
                        "step": step_index,
                        "status": decision["status"],
                        "sigma_min": sigma_min,
                        "sigma_by_state": spreads.tolist(),
                        "e_q": e_q,
                        "ratio": ratio,
                        "min_lb": float(lb.min()) if lb.size else None,
                        "max_lb": float(lb.max()) if lb.size else None,
                        "update_emitted": bool(emitted),
                        "eta_selected": decision["eta_selected"],
                        "ordered_reasons": fs.route_failure_reasons(
                            certificate, decision
                        ),
                        "oracle_audit": {
                            "purpose": "truth-based diagnostic only; never a certificate input",
                            "v_pi": v_pi.tolist(),
                            "min_state_action_gap_true": float(
                                np.min(
                                    np.sort(q_pi, axis=1)[:, -1]
                                    - np.sort(q_pi, axis=1)[:, -2]
                                )
                            ),
                        },
                    }
                    steps.append(entry)

                    if not emitted:
                        # Abstain: the policy does not move, so continue the loop
                        # anyway to record what the SAME policy would produce at
                        # the next step only if this is step 1; beyond that the
                        # trajectory is frozen and further steps add no new
                        # policy. Stop here to keep one row per reachable policy.
                        break
                    current_policy = np.asarray(
                        decision["policy_plus"], dtype=np.float64
                    )

                # --- sealed classification reproduction ----------------------
                # H0. The comparison is deliberately stronger than "same
                # emitted/abstained flag": a census whose certificate differed
                # from the sealed one would be measuring a different quantity, so
                # E_Q, the eta choice and the ordered reasons are compared too.
                repro: dict[str, Any] = {}
                if sealed_record is not None:
                    sealed_steps = sealed_record["routes"][route]["steps"]
                    decision_mismatches = 0
                    reason_mismatches = 0
                    eta_mismatches = 0
                    e_q_mismatches = 0
                    compared = min(len(steps), len(sealed_steps))
                    for i in range(compared):
                        ours, theirs = steps[i], sealed_steps[i]
                        if ours["update_emitted"] != theirs["update_emitted"]:
                            decision_mismatches += 1
                        if ours["ordered_reasons"] != theirs["ordered_reasons"]:
                            reason_mismatches += 1
                        if ours["eta_selected"] != theirs["eta_selected"]:
                            eta_mismatches += 1
                        sealed_e_q = theirs["e_q"]
                        if (ours["e_q"] is None) != (sealed_e_q is None) or (
                            ours["e_q"] is not None
                            and float(ours["e_q"]) != float(sealed_e_q)
                        ):
                            e_q_mismatches += 1
                    repro = {
                        "sealed_emitted_steps": sealed_record["routes"][route][
                            "emitted_steps"
                        ],
                        "census_emitted_steps": sum(
                            1 for s in steps if s["update_emitted"]
                        ),
                        "sealed_step_count": len(sealed_steps),
                        "census_step_count": len(steps),
                        "compared": compared,
                        "decision_mismatches": decision_mismatches,
                        "reason_mismatches": reason_mismatches,
                        "eta_mismatches": eta_mismatches,
                        "e_q_mismatches": e_q_mismatches,
                        "exact": bool(
                            compared == len(sealed_steps) == len(steps)
                            and decision_mismatches == 0
                            and reason_mismatches == 0
                            and eta_mismatches == 0
                            and e_q_mismatches == 0
                        ),
                    }

                routes[route] = {
                    "steps": steps,
                    "step1_ratio": steps[0]["ratio"],
                    "step1_e_q": steps[0]["e_q"],
                    "step1_sigma_min": steps[0]["sigma_min"],
                    "step1_emitted": steps[0]["update_emitted"],
                    "emitted_steps": sum(1 for s in steps if s["update_emitted"]),
                    "sealed_reproduction": repro,
                }

            records.append(
                {
                    "task_id": TASK_ID,
                    "mixing": float(mixing),
                    "task_index": int(task_index),
                    "routes": routes,
                }
            )
            print(f"  mixing={mixing} task={task_index:>2} done", flush=True)

    bundle = {
        "task_id": TASK_ID,
        "label": args.label,
        "record_count": len(records),
        "max_steps": MAX_STEPS,
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_results.json").write_text(
        json.dumps(strict_ready(bundle), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "label": args.label,
                "mixings": list(mixings),
                "tasks": args.tasks,
                "max_steps": MAX_STEPS,
                "cert_chains": CERT_CHAINS,
                "cert_chain_length": CERT_CHAIN_LENGTH,
                "reference_bundle": str(ITER5_BUNDLE),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "captured_utc": datetime.now(timezone.utc).isoformat(),
                "sealed_file_hashes": {
                    name: sha256(PROJECT / name) for name in SEALED_FILES
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "commands.log").write_text(
        " ".join(sys.argv) + "\n", encoding="utf-8"
    )

    total = sum(1 for r in records for _ in PRIMARY)
    repros = [
        r["routes"][route]["sealed_reproduction"]
        for r in records
        for route in PRIMARY
    ]
    bad = sum(1 for x in repros if not x.get("exact", False))
    decision_mismatches = sum(x.get("decision_mismatches", 0) for x in repros)
    e_q_mismatches = sum(x.get("e_q_mismatches", 0) for x in repros)
    eta_mismatches = sum(x.get("eta_mismatches", 0) for x in repros)
    reason_mismatches = sum(x.get("reason_mismatches", 0) for x in repros)
    print(
        f"{TASK_ID}: {total} route-records censused; H0 exact = "
        f"{total - bad}/{total} (decision {decision_mismatches}, e_q "
        f"{e_q_mismatches}, eta {eta_mismatches}, reasons {reason_mismatches})"
    )


if __name__ == "__main__":
    main()
