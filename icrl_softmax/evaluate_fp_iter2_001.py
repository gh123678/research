"""FP-ITER2-001 evaluator: a second certified relative-softmax step.

For every frozen FP-SCALE-002 record this rebuilds the identical training and
certification batches, reproduces step 1 to confirm it matches the sealed
result, and then applies the SAME frozen certificate and decision rule a second
time with ``pi_1`` as the new target policy.

Nothing is retuned. Both steps use the identical batches, verified executably.

Usage:
    python -B evaluate_fp_iter2_001.py --output-dir <dir> [--tasks N] [--mixings a,b]
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

TASK_ID = "FP-ITER2-001"
SEALED_BUNDLE = (
    PROJECT / "results" / "FP-SCALE-002" / "claude" / "formal" / "task_results.json"
)
PRIMARY = ("expected_exact", "expected_finite")
SEALED_LABEL = {
    "expected_exact": "variance_adaptive_exact",
    "expected_finite": "variance_adaptive_finite",
}
MAX_STEPS = 2
# FP-ITER3-001 extended the horizon to 3 and FP-ITER4-001 to 4. The code path is
# otherwise unchanged, and both tasks require proving that raising the ceiling
# leaves the earlier steps bit-identical, so that the horizon change is inert
# rather than a new method.
ALLOWED_MAX_STEPS = (2, 3, 4)
MIXINGS = (0.08, 0.5)
TASKS = 12
# FP-SCALE-002 certification constants. ``fs`` carries the FP-SCALE-001 values
# (262144 x 16), so they are frozen locally.
CERT_CHAINS = 16384
CERT_CHAIN_LENGTH = 64
SEALED_FILES = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "evaluate_fp_scale_002.py",
    "evaluate_fp_attn_001.py",
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


def batch_digest(batch: dict[str, np.ndarray]) -> str:
    """Stable digest of a batch, so identical batches are provable."""
    digest = hashlib.sha256()
    for key in sorted(batch):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(batch[key]).tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=TASKS)
    parser.add_argument("--mixings", type=str, default="0.08,0.5")
    parser.add_argument("--label", type=str, default="second-certified-step")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        choices=ALLOWED_MAX_STEPS,
        help="iteration horizon; FP-ITER2-001 froze 2, FP-ITER3-001 freezes 3",
    )
    args = parser.parse_args()
    max_steps = int(args.max_steps)

    mixings = tuple(float(v) for v in args.mixings.split(","))
    debug = bool(args.debug)
    sealed = json.loads(SEALED_BUNDLE.read_text(encoding="utf-8"))
    sealed_by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
    }

    records: list[dict[str, Any]] = []
    for mixing in mixings:
        for task_index in range(args.tasks):
            mdp, policy, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact = policy_quantities(mdp, policy)
            mu_state = np.asarray(exact["mu_state"], dtype=np.float64)
            v_chain = [np.asarray(exact["v_pi"], dtype=np.float64)]

            train = fs.training_batch(mdp, policy, mu_state, rng)
            cert = certification_batch(
                mdp,
                policy,
                mu_state,
                [fs.SEED, 9001, int(round(mixing * 100)), task_index],
            )
            train_digest = batch_digest(train)
            cert_digest = batch_digest(cert)

            sealed_record = sealed_by_key.get((mixing, task_index))
            routes: dict[str, Any] = {}
            for route in PRIMARY:
                current_policy = policy.copy()
                steps: list[dict[str, Any]] = []
                # Per-route value chain. It MUST be reset here: each route starts
                # from the same pi_0, and carrying the previous route's chain
                # forward made the second route compare against the first
                # route's final value. That produced large negative deltas and a
                # spurious "non-degrading" failure in the first smoke run.
                v_chain = [np.asarray(exact["v_pi"], dtype=np.float64).copy()]
                # Q^pi for the policy CURRENTLY being evaluated. The certificate
                # bounds ||Q_k - Q^{pi_{k-1}}||, not ||Q_k - Q^{pi_0}||, so the
                # audit must use the matching fixed point at every step.
                q_ref = np.asarray(exact["q_pi"], dtype=np.float64).copy()
                for step_index in range(1, max_steps + 1):
                    result = fs.run_route(route, current_policy, train)
                    q_hat = np.asarray(result["q_hat"], dtype=np.float64).reshape(
                        fs.N_STATES, fs.N_ACTIONS
                    )
                    certificate = vc.variance_adaptive_certificate(
                        q_hat, current_policy, cert
                    )
                    decision = fs.improvement_for(current_policy, q_hat, certificate)
                    emitted = decision["status"] == "safe_update_emitted"
                    lb = np.asarray(
                        decision["lb_by_state"], dtype=np.float64
                    ).reshape(-1)

                    realized = float(np.max(np.abs(q_hat - q_ref)))
                    entry: dict[str, Any] = {
                        "step": step_index,
                        "status": decision["status"],
                        "eta_selected": decision["eta_selected"],
                        "e_q": certificate.get("e_q"),
                        "min_lb": float(lb.min()) if lb.size else None,
                        "max_lb": float(lb.max()) if lb.size else None,
                        "update_emitted": bool(emitted),
                        "ordered_reasons": fs.route_failure_reasons(
                            certificate, decision
                        ),
                        "oracle_audit": {
                            "purpose": "truth-based audit only; never a certificate input",
                            "realized_q_sup_error_vs_current_target_pi": realized,
                            "certificate_violation": bool(
                                certificate.get("e_q") is not None
                                and float(certificate["e_q"]) < realized
                            ),
                        },
                    }
                    if emitted:
                        next_policy = np.asarray(
                            decision["policy_plus"], dtype=np.float64
                        )
                        v_next = np.asarray(
                            policy_quantities(mdp, next_policy)["v_pi"],
                            dtype=np.float64,
                        )
                        delta_v = v_next - v_chain[-1]
                        if debug:
                            print(
                                f"DEBUG mix={mixing} task={task_index} route={route} "
                                f"step={step_index} prev={np.round(v_chain[-1], 6)} "
                                f"vnext={np.round(v_next, 6)} "
                                f"delta={np.round(delta_v, 6)}"
                            )
                        entry["oracle_audit"].update(
                            {
                                "value_delta_vs_previous": delta_v.tolist(),
                                "componentwise_nondegrading": bool(
                                    float(np.min(delta_v)) >= -1e-12
                                ),
                                "total_value_gain": float(np.sum(delta_v)),
                                "value_delta_vs_start": (
                                    v_next - v_chain[0]
                                ).tolist(),
                            }
                        )
                        v_chain.append(v_next)
                        q_ref = np.asarray(
                            policy_quantities(mdp, next_policy)["q_pi"],
                            dtype=np.float64,
                        )
                        current_policy = next_policy
                    steps.append(entry)
                    if not emitted:
                        break

                # Step-1 reproduction against the sealed bundle.
                repro: dict[str, Any] = {}
                if sealed_record is not None and steps:
                    sealed_route = sealed_record["routes"][SEALED_LABEL[route]]
                    first = steps[0]
                    sealed_e_q = sealed_route["e_q"]
                    repro["sealed_e_q"] = sealed_e_q
                    repro["e_q_gap"] = (
                        None
                        if sealed_e_q is None or first["e_q"] is None
                        else float(abs(float(first["e_q"]) - float(sealed_e_q)))
                    )
                    repro["eta_matches"] = bool(
                        sealed_route["eta_selected"] == first["eta_selected"]
                    )
                    repro["decision_matches"] = bool(
                        sealed_route["update_emitted"] == first["update_emitted"]
                    )
                    repro["exact"] = bool(
                        repro["e_q_gap"] == 0.0
                        and repro["eta_matches"]
                        and repro["decision_matches"]
                    )

                routes[route] = {
                    "steps": steps,
                    "step1_reproduction": repro,
                    "emitted_steps": sum(1 for s in steps if s["update_emitted"]),
                    "final_policy_value": v_chain[-1].tolist(),
                    "start_policy_value": v_chain[0].tolist(),
                }

            records.append(
                {
                    "task_id": TASK_ID,
                    "mixing": float(mixing),
                    "task_index": int(task_index),
                    "n_states": fs.N_STATES,
                    "n_actions": fs.N_ACTIONS,
                    "layers": fs.LAYERS,
                    "train_length": fs.TRAIN_LENGTH,
                    "cert_count": int(cert["states"].size),
                    "train_batch_digest": train_digest,
                    "cert_batch_digest": cert_digest,
                    "routes": routes,
                }
            )
            two = sum(
                1
                for r in routes.values()
                if r["emitted_steps"] >= 2
            )
            print(
                f"  mixing={mixing} task={task_index:>2} emitted_two_steps={two}",
                flush=True,
            )

    bundle = {
        "task_id": TASK_ID,
        "label": args.label,
        "record_count": len(records),
        "max_steps": max_steps,
        "cert_chains": CERT_CHAINS,
        "cert_chain_length": CERT_CHAIN_LENGTH,
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
                "max_steps": max_steps,
                "cert_chains": CERT_CHAINS,
                "cert_chain_length": CERT_CHAIN_LENGTH,
                "routes": list(PRIMARY),
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

    two_step = sum(
        1
        for record in records
        for route in record["routes"].values()
        if route["emitted_steps"] >= 2
    )
    one_step = sum(
        1
        for record in records
        for route in record["routes"].values()
        if route["emitted_steps"] == 1
    )
    print(
        f"{TASK_ID}: {len(records)} records; routes emitting 2 steps = {two_step}, "
        f"1 step = {one_step}"
    )


if __name__ == "__main__":
    main()


