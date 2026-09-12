"""FP-ATTN-ITER-001 evaluator: three certified steps driven by the literal network.

At every iteration step the Q estimate comes from the literal softmax attention
network in ``model.py``; the frozen certificate and decision rule are then
applied to that network output. The numpy route is computed alongside purely as
the comparison baseline, and a provenance flag records which producer generated
each step's Qhat.

Nothing is retuned. Both steps use the identical training and certification
batches, verified by digest equality.

Usage:
    python -B evaluate_fp_attn_iter_001.py --output-dir <dir> [--tasks N] [--mixings a,b]
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
import torch

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_variance_certificate as vc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from model import (  # noqa: E402
    EndToEndFiniteSoftmaxExpectedSARSA,
    EndToEndMaskedSoftmaxExpectedSARSA,
)

TASK_ID = "FP-ATTN-ITER-001"
ITER3_BUNDLE = (
    PROJECT / "results" / "FP-ITER3-001" / "claude" / "formal" / "task_results.json"
)
# FP-ITER4-001 sealed the numpy four-step route; FP-ATTN-ITER4-001 compares
# against it so that step 4 is matched against the matching numpy horizon.
ITER4_BUNDLE = (
    PROJECT / "results" / "FP-ITER4-001" / "claude" / "formal" / "task_results.json"
)
# FP-ITER5-001 sealed the numpy five-step route; FP-ITER6-001 sealed the six-step
# one. Each network horizon compares against the numpy horizon of the same length,
# which is the only apples-to-apples path comparison.
ITER5_BUNDLE = (
    PROJECT / "results" / "FP-ITER5-001" / "claude" / "numpy" / "task_results.json"
)
ITER6_BUNDLE = (
    PROJECT / "results" / "FP-ITER6-001" / "claude" / "numpy" / "task_results.json"
)
REFERENCE_BUNDLES = {
    "iter3": ITER3_BUNDLE,
    "iter4": ITER4_BUNDLE,
    "iter5": ITER5_BUNDLE,
    "iter6": ITER6_BUNDLE,
}
PRIMARY = ("expected_exact", "expected_finite")
MAX_STEPS = 3
# FP-ATTN-ITER-001 (3), FP-ATTN-ITER4-001 (4), FP-ITER5-001 (5) and
# FP-ATTN-ITER6-001 (6) freeze the horizon here. The code path is otherwise
# unchanged, and each task must prove that raising the ceiling leaves the earlier
# steps bit-identical, so that the horizon change is inert rather than a new
# method.
ALLOWED_MAX_STEPS = (3, 4, 5, 6)
ATOL = 1e-4
MIXINGS = (0.08, 0.5)
TASKS = 12
# FP-SCALE-002 certification constants (``fs`` carries the FP-SCALE-001 values).
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
    digest = hashlib.sha256()
    for key in sorted(batch):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(batch[key]).tobytes())
    return digest.hexdigest()


def network_qhat(
    network: torch.nn.Module,
    policy: np.ndarray,
    train: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run the literal network for the frozen layer count from Q_0 = 0."""
    states = torch.as_tensor(train["states"], dtype=torch.long)
    actions = torch.as_tensor(train["actions"], dtype=torch.long)
    rewards = torch.as_tensor(train["rewards"], dtype=torch.float32)
    next_states = torch.as_tensor(train["next_states"], dtype=torch.long)
    policy_t = torch.as_tensor(policy, dtype=torch.float32)
    q = torch.zeros((fs.N_STATES, fs.N_ACTIONS), dtype=torch.float32)
    first: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for layer in range(fs.LAYERS):
            q, diagnostics = network(q, states, actions, rewards, next_states, policy_t)
            if layer == 0:
                first = {
                    key: value.detach().numpy().copy()
                    for key, value in diagnostics.items()
                    if isinstance(value, torch.Tensor)
                }
    return q.detach().numpy().astype(np.float64), first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=TASKS)
    parser.add_argument("--mixings", type=str, default="0.08,0.5")
    parser.add_argument("--label", type=str, default="literal-iteration")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        choices=ALLOWED_MAX_STEPS,
        help="network iteration horizon; FP-ATTN-ITER-001 froze 3, "
        "FP-ATTN-ITER4-001 froze 4, FP-ITER5-001 froze 5, FP-ATTN-ITER6-001 "
        "freezes 6",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="iter3",
        choices=sorted(REFERENCE_BUNDLES),
        help="numpy comparison baseline: iter3 (FP-ITER3-001), iter4 "
        "(FP-ITER4-001), iter5 (FP-ITER5-001) or iter6 (FP-ITER6-001)",
    )
    args = parser.parse_args()
    max_steps = int(args.max_steps)

    mixings = tuple(float(v) for v in args.mixings.split(","))
    reference_path = REFERENCE_BUNDLES[args.reference]
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    ref_by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in reference["records"]
    }

    masked = EndToEndMaskedSoftmaxExpectedSARSA(gamma=fs.GAMMA, alpha=fs.ALPHA)
    finite = EndToEndFiniteSoftmaxExpectedSARSA(
        gamma=fs.GAMMA, alpha=fs.ALPHA, zeta=fs.ZETA, xi=fs.XI, tau=fs.TAU
    )
    networks = {"expected_exact": masked, "expected_finite": finite}

    records: list[dict[str, Any]] = []
    for mixing in mixings:
        for task_index in range(args.tasks):
            mdp, policy, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact = policy_quantities(mdp, policy)
            mu_state = np.asarray(exact["mu_state"], dtype=np.float64)
            train = fs.training_batch(mdp, policy, mu_state, rng)
            cert = certification_batch(
                mdp, policy, mu_state, [fs.SEED, 9001, int(round(mixing * 100)), task_index]
            )
            train_digest = batch_digest(train)
            cert_digest = batch_digest(cert)
            ref_record = ref_by_key.get((mixing, task_index))

            routes: dict[str, Any] = {}
            for route in PRIMARY:
                # ---- literal-network driven iteration -------------------------
                current_policy = policy.copy()
                v_chain = [np.asarray(exact["v_pi"], dtype=np.float64).copy()]
                q_ref = np.asarray(exact["q_pi"], dtype=np.float64).copy()
                network = networks[route]
                steps: list[dict[str, Any]] = []
                for step_index in range(1, max_steps + 1):
                    q_hat_literal, first_layer = network_qhat(
                        network, current_policy, train
                    )
                    # Provenance: the certificate is scored on the NETWORK output.
                    certificate = vc.variance_adaptive_certificate(
                        q_hat_literal, current_policy, cert
                    )
                    decision = fs.improvement_for(
                        current_policy, q_hat_literal, certificate
                    )
                    emitted = decision["status"] == "safe_update_emitted"
                    lb = np.asarray(
                        decision["lb_by_state"], dtype=np.float64
                    ).reshape(-1)
                    realized = float(np.max(np.abs(q_hat_literal - q_ref)))

                    # numpy comparator for the same step, same batches.
                    q_hat_numpy = np.asarray(
                        fs.run_route(route, current_policy, train)["q_hat"],
                        dtype=np.float64,
                    ).reshape(fs.N_STATES, fs.N_ACTIONS)
                    cert_numpy = vc.variance_adaptive_certificate(
                        q_hat_numpy, current_policy, cert
                    )
                    decision_numpy = fs.improvement_for(
                        current_policy, q_hat_numpy, cert_numpy
                    )
                    lb_numpy = np.asarray(
                        decision_numpy["lb_by_state"], dtype=np.float64
                    ).reshape(-1)

                    entry: dict[str, Any] = {
                        "step": step_index,
                        "qhat_producer": "literal_attention_network",
                        "q_hat_gap_vs_numpy": float(
                            np.max(np.abs(q_hat_literal - q_hat_numpy))
                        ),
                        "status": decision["status"],
                        "eta_selected": decision["eta_selected"],
                        "e_q": certificate.get("e_q"),
                        "min_lb": float(lb.min()) if lb.size else None,
                        "max_lb": float(lb.max()) if lb.size else None,
                        "update_emitted": bool(emitted),
                        "ordered_reasons": fs.route_failure_reasons(
                            certificate, decision
                        ),
                        "numpy": {
                            "status": decision_numpy["status"],
                            "eta_selected": decision_numpy["eta_selected"],
                            "e_q": cert_numpy.get("e_q"),
                            "min_lb": float(lb_numpy.min()) if lb_numpy.size else None,
                            "update_emitted": bool(
                                decision_numpy["status"] == "safe_update_emitted"
                            ),
                        },
                        "decision_flip_vs_numpy": bool(
                            emitted
                            != (decision_numpy["status"] == "safe_update_emitted")
                        ),
                        "eta_flip_vs_numpy": bool(
                            decision["eta_selected"] != decision_numpy["eta_selected"]
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
                        q_next = policy_quantities(mdp, next_policy)
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
                        current_policy = next_policy
                    steps.append(entry)
                    if not emitted:
                        break

                # reference (numpy) three-step block from FP-ITER3-001
                ref_steps = (
                    ref_record["routes"][route]["steps"] if ref_record else []
                )
                comparisons = []
                for index, step in enumerate(steps):
                    if index < len(ref_steps):
                        ref = ref_steps[index]
                        comparisons.append(
                            {
                                "step": step["step"],
                                "decision_match": bool(
                                    step["update_emitted"] == ref["update_emitted"]
                                ),
                                "eta_match": bool(
                                    step["eta_selected"] == ref["eta_selected"]
                                ),
                            }
                        )

                routes[route] = {
                    "steps": steps,
                    "emitted_steps": sum(1 for s in steps if s["update_emitted"]),
                    "reference_emitted_steps": (
                        ref_record["routes"][route]["emitted_steps"]
                        if ref_record
                        else None
                    ),
                    "reference_comparisons": comparisons,
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
                    "atol": ATOL,
                    "cert_count": int(cert["states"].size),
                    "train_batch_digest": train_digest,
                    "cert_batch_digest": cert_digest,
                    "routes": routes,
                }
            )
            three = sum(1 for r in routes.values() if r["emitted_steps"] >= 3)
            print(
                f"  mixing={mixing} task={task_index:>2} network_three_step_routes={three}",
                flush=True,
            )

    bundle = {
        "task_id": TASK_ID,
        "label": args.label,
        "record_count": len(records),
        "max_steps": max_steps,
        "atol": ATOL,
        "torch": torch.__version__,
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
                "atol": ATOL,
                "cert_chains": CERT_CHAINS,
                "cert_chain_length": CERT_CHAIN_LENGTH,
                "route_network_map": {
                    "expected_exact": "EndToEndMaskedSoftmaxExpectedSARSA",
                    "expected_finite": "EndToEndFiniteSoftmaxExpectedSARSA",
                },
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
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
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

    three = sum(
        1
        for record in records
        for block in record["routes"].values()
        if block["emitted_steps"] >= 3
    )
    flips = sum(
        1
        for record in records
        for block in record["routes"].values()
        for step in block["steps"]
        if step["decision_flip_vs_numpy"]
    )
    print(
        f"{TASK_ID}: {len(records)} records; network three-step routes = {three}; "
        f"decision flips vs numpy = {flips}"
    )


if __name__ == "__main__":
    main()


