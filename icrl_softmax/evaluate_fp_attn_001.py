"""FP-ATTN-001 evaluator: literal softmax-attention execution of the frozen records.

Runs the literal attention networks from ``model.py`` on the frozen
FP-SCALE-002 protocol and compares them, record by record, against the numpy
routes that produced the sealed result.

Design points that matter:

- The training batches are REGENERATED from the frozen seed schedule because the
  sealed bundle does not serialize trajectories. Regeneration is validated: the
  numpy route run on the regenerated batch must reproduce the sealed ``Qhat``,
  otherwise the record is reported as a reproduction failure.
- Both sides are scored with the SAME frozen certificate and decision code, so
  the comparison isolates the ``Qhat`` the network produces.
- Literal networks run in float32 (their native dtype) under ``torch.no_grad``;
  numpy routes run in float64. ``ATOL`` is frozen in the task sheet.

Usage:
    python -B evaluate_fp_attn_001.py --output-dir <dir> [--tasks N] [--mixings a,b]
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

TASK_ID = "FP-ATTN-001"
SEALED_BUNDLE = (
    PROJECT / "results" / "FP-SCALE-002" / "claude" / "formal" / "task_results.json"
)
ATOL = 1e-4
MIXINGS = (0.08, 0.5)
TASKS = 12
# Certification constants of the FP-SCALE-002 protocol, NOT fs.CERT_CHAINS.
# ``fixed_policy_expected_sarsa_scaled`` carries the FP-SCALE-001 values
# (262144 x 16); FP-SCALE-002 froze 16384 x 64 in its own evaluator and the
# sealed corpus was produced with those.
CERT_CHAINS = 16384
CERT_CHAIN_LENGTH = 64
# numpy route -> (literal network name, sealed route label)
PAIRS = {
    "expected_exact": ("literal_masked_exact", "variance_adaptive_exact"),
    "expected_finite": ("literal_finite", "variance_adaptive_finite"),
}

SEALED_FILES = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "verify_variance_adaptive_certificate.py",
    "evaluate_fp_scale_002.py",
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
    """Recreate the frozen certification batch exactly as FP-SCALE-002 did."""
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


def run_literal(
    network: torch.nn.Module,
    train: dict[str, np.ndarray],
    *,
    layers: int,
    capture_first_layer: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply the literal network ``layers`` times from Q_0 = 0."""
    states = torch.as_tensor(train["states"], dtype=torch.long)
    actions = torch.as_tensor(train["actions"], dtype=torch.long)
    rewards = torch.as_tensor(train["rewards"], dtype=torch.float32)
    next_states = torch.as_tensor(train["next_states"], dtype=torch.long)
    policy = torch.as_tensor(train["policy"], dtype=torch.float32)
    q = torch.zeros((fs.N_STATES, fs.N_ACTIONS), dtype=torch.float32)
    first: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for layer in range(layers):
            q, diagnostics = network(q, states, actions, rewards, next_states, policy)
            if capture_first_layer and layer == 0:
                first = {
                    key: value.detach().numpy().copy()
                    for key, value in diagnostics.items()
                    if isinstance(value, torch.Tensor)
                }
    return q.detach().numpy().astype(np.float64), first


def numpy_layer0(train: dict[str, np.ndarray], policy: np.ndarray) -> dict[str, np.ndarray]:
    """Numpy reference for the layer-0 quantities the diagnostics report.

    Computed from the first application of the exact grouped update from Q_0=0,
    which is what the literal networks also start from.
    """
    q0 = np.zeros((fs.N_STATES, fs.N_ACTIONS), dtype=np.float64)
    states = train["states"]
    actions = train["actions"]
    rewards = train["rewards"]
    next_states = train["next_states"]
    successor = (policy[next_states] * q0[next_states]).sum(axis=1)
    residuals = rewards + fs.GAMMA * successor - q0[states, actions]
    zeros = np.zeros(states.size, dtype=np.float64)
    return {
        "read": zeros.copy(),
        "successor": successor,
        "residuals": residuals,
        "qbar": successor,
    }


def min_lb(decision: dict[str, Any]) -> float | None:
    """Smallest per-state lower bound, or None when unavailable."""
    values = np.asarray(decision.get("lb_by_state", []), dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None
    return float(np.min(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=TASKS)
    parser.add_argument("--mixings", type=str, default="0.08,0.5")
    parser.add_argument("--label", type=str, default="literal-attention")
    args = parser.parse_args()

    mixings = tuple(float(v) for v in args.mixings.split(","))
    sealed = json.loads(SEALED_BUNDLE.read_text(encoding="utf-8"))
    sealed_by_key = {
        (float(r["mixing"]), int(r["task_index"])): r for r in sealed["records"]
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
            q_pi = np.asarray(exact["q_pi"], dtype=np.float64)
            train = fs.training_batch(mdp, policy, mu_state, rng)
            train_with_policy = dict(train)
            train_with_policy["policy"] = policy

            cert = certification_batch(
                mdp, policy, mu_state, [fs.SEED, 9001, int(round(mixing * 100)), task_index]
            )

            sealed_record = sealed_by_key.get((mixing, task_index))
            routes: dict[str, Any] = {}
            for numpy_route, (literal_name, sealed_label) in PAIRS.items():
                # --- numpy side -------------------------------------------------
                numpy_result = fs.run_route(numpy_route, policy, train)
                q_numpy = np.asarray(numpy_result["q_hat"], dtype=np.float64).reshape(
                    fs.N_STATES, fs.N_ACTIONS
                )
                cert_numpy = vc.variance_adaptive_certificate(q_numpy, policy, cert)
                decision_numpy = fs.improvement_for(policy, q_numpy, cert_numpy)

                # --- sealed side, for regeneration validation -------------------
                # The sealed bundle does not serialize q_hat, so regeneration is
                # validated against the sealed end-to-end quantities instead:
                # certified error, selected eta, per-state lower bounds and the
                # emission decision. That is a stronger check than a Q-vector
                # comparison because it exercises the whole downstream pipeline.
                sealed_route = (
                    sealed_record["routes"][sealed_label]
                    if sealed_record is not None
                    else None
                )
                regen = {}
                if sealed_route is not None:
                    sealed_e_q = sealed_route["e_q"]
                    numpy_e_q = cert_numpy.get("e_q")
                    regen["e_q_gap"] = (
                        None
                        if sealed_e_q is None or numpy_e_q is None
                        else float(abs(float(numpy_e_q) - float(sealed_e_q)))
                    )
                    regen["sealed_e_q"] = sealed_e_q
                    regen["e_q_matches"] = (
                        None
                        if regen["e_q_gap"] is None
                        else bool(regen["e_q_gap"] <= ATOL)
                    )
                    regen["sealed_eta"] = sealed_route["eta_selected"]
                    regen["eta_matches"] = bool(
                        sealed_route["eta_selected"] == decision_numpy["eta_selected"]
                    )
                    sealed_lb = np.asarray(
                        sealed_route["lb_by_state"], dtype=np.float64
                    ).reshape(-1)
                    numpy_lb = np.asarray(
                        decision_numpy["lb_by_state"], dtype=np.float64
                    ).reshape(-1)
                    regen["lb_gap"] = (
                        float(np.max(np.abs(sealed_lb - numpy_lb)))
                        if sealed_lb.size and sealed_lb.size == numpy_lb.size
                        else None
                    )
                    regen["sealed_emitted"] = bool(sealed_route["update_emitted"])
                    regen["decision_matches"] = bool(
                        sealed_route["update_emitted"]
                        == (decision_numpy["status"] == "safe_update_emitted")
                    )
                    regen["all_match"] = bool(
                        regen["e_q_matches"] is not False
                        and regen["eta_matches"]
                        and regen["decision_matches"]
                    )
                else:
                    regen["all_match"] = None

                # --- literal side ----------------------------------------------
                q_literal, first_layer = run_literal(
                    networks[numpy_route],
                    train_with_policy,
                    layers=fs.LAYERS,
                    capture_first_layer=True,
                )
                cert_literal = vc.variance_adaptive_certificate(
                    q_literal, policy, cert
                )
                decision_literal = fs.improvement_for(policy, q_literal, cert_literal)

                numpy_layer = numpy_layer0(train_with_policy, policy)
                diagnostic_gap = {}
                for field, reference in numpy_layer.items():
                    if field in first_layer:
                        diagnostic_gap[field] = float(
                            np.max(np.abs(np.asarray(first_layer[field]).reshape(-1) - reference))
                        )

                q_gap = float(np.max(np.abs(q_literal - q_numpy)))
                finite_ok = bool(np.all(np.isfinite(q_literal)))
                guard_ok = bool(
                    float(np.max(np.abs(q_literal))) <= fs.VALUE_BOUND
                )
                routes[numpy_route] = {
                    "literal_network": literal_name,
                    "sealed_route_label": sealed_label,
                    "numpy_e_q": cert_numpy.get("e_q"),
                    "literal_e_q": cert_literal.get("e_q"),
                    "numpy_status": decision_numpy["status"],
                    "literal_status": decision_literal["status"],
                    "numpy_eta": decision_numpy["eta_selected"],
                    "literal_eta": decision_literal["eta_selected"],
                    "numpy_update_emitted": bool(decision_numpy["status"] == "safe_update_emitted"),
                    "literal_update_emitted": bool(
                        decision_literal["status"] == "safe_update_emitted"
                    ),
                    "decision_flip": bool(
                        (decision_numpy["status"] == "safe_update_emitted")
                        != (decision_literal["status"] == "safe_update_emitted")
                    ),
                    "eta_flip": bool(
                        decision_numpy["eta_selected"] != decision_literal["eta_selected"]
                    ),
                    "q_hat_gap_inf": q_gap,
                    "q_hat_gap_within_atol": bool(q_gap <= ATOL),
                    "sealed_regeneration": regen,
                    "literal_finite": finite_ok,
                    "literal_within_divergence_guard": guard_ok,
                    "layer0_diagnostic_gap": diagnostic_gap,
                    "numpy_min_lb": min_lb(decision_numpy),
                    "literal_min_lb": min_lb(decision_literal),
                    "oracle_audit": {
                        "purpose": "truth-based audit only; not a certificate input",
                        "numpy_realized_q_sup_error": float(np.max(np.abs(q_numpy - q_pi))),
                        "literal_realized_q_sup_error": float(
                            np.max(np.abs(q_literal - q_pi))
                        ),
                    },
                }
            records.append(
                {
                    "task_id": TASK_ID,
                    "mixing": float(mixing),
                    "task_index": int(task_index),
                    "n_states": fs.N_STATES,
                    "n_actions": fs.N_ACTIONS,
                    "layers": fs.LAYERS,
                    "gamma": fs.GAMMA,
                    "alpha": fs.ALPHA,
                    "train_length": fs.TRAIN_LENGTH,
                    "atol": ATOL,
                    "cert_count": int(cert["states"].size),
                    "routes": routes,
                }
            )
            print(
                f"  mixing={mixing} task={task_index:>2} "
                + " ".join(
                    f"{r}={records[-1]['routes'][r]['literal_update_emitted']}"
                    for r in PAIRS
                ),
                flush=True,
            )

    bundle = {
        "task_id": TASK_ID,
        "label": args.label,
        "record_count": len(records),
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
                "atol": ATOL,
                "layers": fs.LAYERS,
                "pairs": {k: list(v) for k, v in PAIRS.items()},
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

    flips = sum(
        1
        for record in records
        for route in record["routes"].values()
        if route["decision_flip"]
    )
    print(f"{TASK_ID}: {len(records)} records, decision flips = {flips}")


if __name__ == "__main__":
    main()
