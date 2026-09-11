"""FP-SCALE-002 frozen evaluator.

Runs the frozen FP-SCALE-002 matrix: two mixing settings, twelve tasks each,
three routes per record, with the variance-adaptive residual certificate.

Every record carries:
  - ``variance_adaptive_exact``   primary, exact grouped Expected SARSA;
  - ``variance_adaptive_finite``  primary, finite-logit mask-free route;
  - ``envelope_control_exact``    the ``2B`` envelope certificate on the SAME
                                  tasks, batches and per-pair split sizes, as the
                                  attribution control for H5.

Exact truth is computed only inside ``oracle_audit``. The estimator, the
certificate, and the decision rule never receive it.

Usage:
    python -B evaluate_fp_scale_002.py --mode smoke|formal --output-dir <dir>
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

TASK_ID = "FP-SCALE-002"
ROUTES = ("variance_adaptive_exact", "variance_adaptive_finite", "envelope_control_exact")
PRIMARY = ("variance_adaptive_exact", "variance_adaptive_finite")
SEED = 20260911
MIXINGS = (0.08, 0.5)
TASKS_PER_CELL = 12
TRAIN_LENGTH = 65536
CERT_CHAINS = 16384
CERT_CHAIN_LENGTH = 64
SMOKE_TASKS = 2
# Exercise the real certification path in smoke, including emission: at 65536
# chains the smallest pairs clear the frozen 10000-item floor.
SMOKE_CHAINS = 65536
BASELINE_MODULE = "fixed_policy_expected_sarsa.py"


def strict_json_ready(obj: Any) -> Any:
    """Convert to plain JSON-safe Python without losing finiteness checks."""
    if isinstance(obj, dict):
        return {str(k): strict_json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [strict_json_ready(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        if not np.isfinite(value):
            raise ValueError("nonfinite value cannot be serialised")
        return value
    if obj is None or isinstance(obj, str):
        return obj
    raise ValueError(f"unsupported type {type(obj).__name__}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def certification_batch(
    mdp: Any,
    policy: np.ndarray,
    mu_state: np.ndarray,
    seed_parts: list[Any],
    chains: int,
    chain_length: int,
) -> dict[str, np.ndarray]:
    """Independent certification batch, restarted from the stationary law.

    Kept local to the evaluator so the frozen constants used by
    ``fixed_policy_expected_sarsa_scaled`` are never mutated.
    """
    rng = np.random.default_rng(seed_parts)
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


def run_record(
    mixing: float, task_index: int, chains: int, chain_length: int, train_length: int
) -> dict[str, Any]:
    mdp, policy, rng = fs.build_task(task_index=task_index, mixing=mixing)
    exact = policy_quantities(mdp, policy)
    mu_state = np.asarray(exact["mu_state"], dtype=np.float64)
    q_pi = np.asarray(exact["q_pi"], dtype=np.float64)
    v_pi = np.asarray(exact["v_pi"], dtype=np.float64)

    # Training batch: dedicated stream, used only to build Qhat.
    train = fs.training_batch(mdp, policy, mu_state, rng)
    if train_length != fs.TRAIN_LENGTH:
        raise ValueError("train length must equal the frozen constant")

    # Certification batch: independent stream, never touches Qhat.
    cert = certification_batch(
        mdp,
        policy,
        mu_state,
        [SEED, 9001, int(round(mixing * 100)), int(task_index)],
        chains,
        chain_length,
    )
    counts = fs.pair_counts(cert["states"], cert["actions"])

    routes: dict[str, Any] = {}
    for route in ROUTES:
        if route == "envelope_control_exact":
            source_route = "expected_exact"
        elif route == "variance_adaptive_exact":
            source_route = "expected_exact"
        else:
            source_route = "expected_finite"

        result = fs.run_route(source_route, policy, train)
        q_hat = np.asarray(result["q_hat"], dtype=np.float64).reshape(
            fs.N_STATES, fs.N_ACTIONS
        )

        if route == "envelope_control_exact":
            certificate = vc.envelope_control_certificate(q_hat, policy, cert)
        else:
            certificate = vc.variance_adaptive_certificate(q_hat, policy, cert)

        improvement = fs.improvement_for(policy, q_hat, certificate)
        reasons = fs.route_failure_reasons(certificate, improvement)

        realized = float(np.max(np.abs(q_hat - q_pi)))
        emitted = improvement["status"] == "safe_update_emitted"
        value_delta = None
        componentwise_ok = None
        total_gain = None
        if emitted:
            new_policy = np.asarray(improvement["policy_plus"], dtype=np.float64)
            new_exact = policy_quantities(mdp, new_policy)
            new_v = np.asarray(new_exact["v_pi"], dtype=np.float64)
            delta_v = new_v - v_pi
            value_delta = delta_v.tolist()
            componentwise_ok = bool(float(np.min(delta_v)) >= -1e-12)
            total_gain = float(np.sum(delta_v))

        routes[route] = {
            "status": result.get("status"),
            "route_failure_reasons": result.get("failure_reasons", []),
            "certificate_status": certificate["status"],
            "certificate_failure_reasons": certificate.get("failure_reasons", []),
            "e_q": certificate.get("e_q"),
            "epsilon_res": certificate.get("epsilon_res"),
            "radii": np.asarray(certificate["radii"], dtype=np.float64).tolist(),
            "residual_means": np.asarray(
                certificate["residual_means"], dtype=np.float64
            ).tolist(),
            "scales": np.asarray(
                certificate.get("scales", np.zeros(vc.D)), dtype=np.float64
            ).tolist(),
            "improvement_status": improvement["status"],
            "eta_selected": improvement["eta_selected"],
            "lb_by_state": np.asarray(
                improvement["lb_by_state"], dtype=np.float64
            ).tolist(),
            "update_emitted": bool(emitted),
            "ordered_reasons": reasons,
        }

        routes[route]["oracle_audit"] = {
            "purpose": "empirical truth-based audit only; not proof or certificate input",
            "realized_q_sup_error": realized,
            "certificate_violation": bool(
                certificate.get("e_q") is not None and float(certificate["e_q"]) < realized
            ),
            "value_delta": value_delta,
            "componentwise_nondegrading": componentwise_ok,
            "total_value_gain": total_gain,
        }

    return {
        "task_id": TASK_ID,
        "mixing": float(mixing),
        "task_index": int(task_index),
        "trajectory_length": int(train_length),
        "train_length": int(train_length),
        "cert_chains": int(chains),
        "cert_chain_length": int(chain_length),
        "cert_count": int(chains * chain_length),
        "min_cert_count_observed": int(counts.min()),
        "cert_pair_counts": counts.tolist(),
        "n_states": fs.N_STATES,
        "n_actions": fs.N_ACTIONS,
        "pi_min": fs.PI_MIN,
        "gamma": fs.GAMMA,
        "alpha": fs.ALPHA,
        "layers": fs.LAYERS,
        "reward_bound": fs.REWARD_BOUND,
        "delta": vc.DELTA,
        "min_half_count": vc.MIN_HALF_COUNT,
        "eta_candidates": list(fs.ETA_CANDIDATES),
        "seed": SEED,
        "routes": routes,
        "oracle_audit_only": {
            "q_pi": q_pi.tolist(),
            "v_pi": v_pi.tolist(),
            "mu_state": mu_state.tolist(),
            "within_state_q_spread": np.ptp(q_pi, axis=1).tolist(),
        },
    }


def run_matrix(mode: str, output_dir: Path, label: str) -> dict[str, Any]:
    if mode == "smoke":
        tasks = SMOKE_TASKS
        chains = SMOKE_CHAINS
        chain_length = CERT_CHAIN_LENGTH
        train_length = TRAIN_LENGTH
    else:
        tasks = TASKS_PER_CELL
        chains = CERT_CHAINS
        chain_length = CERT_CHAIN_LENGTH
        train_length = TRAIN_LENGTH

    records: list[dict[str, Any]] = []
    for mixing in MIXINGS:
        for task_index in range(tasks):
            records.append(
                run_record(mixing, task_index, chains, chain_length, train_length)
            )

    bundle = {
        "task_id": TASK_ID,
        "mode": mode,
        "label": label,
        "record_count": len(records),
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task_results.json").write_text(
        json.dumps(strict_json_ready(bundle), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    config = {
        "task_id": TASK_ID,
        "mode": mode,
        "label": label,
        "mixings": list(MIXINGS),
        "tasks_per_cell": tasks,
        "train_length": train_length,
        "cert_chains": chains,
        "cert_chain_length": chain_length,
        "cert_count": chains * chain_length,
        "n_states": fs.N_STATES,
        "n_actions": fs.N_ACTIONS,
        "pi_min": fs.PI_MIN,
        "gamma": fs.GAMMA,
        "alpha": fs.ALPHA,
        "layers": fs.LAYERS,
        "reward_bound": fs.REWARD_BOUND,
        "delta": vc.DELTA,
        "min_half_count": vc.MIN_HALF_COUNT,
        "eta_candidates": list(fs.ETA_CANDIDATES),
        "seed": SEED,
        "routes": list(ROUTES),
        "primary_routes": list(PRIMARY),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "project_dir": str(PROJECT),
        "baseline_module_sha256": sha256_file(PROJECT / BASELINE_MODULE),
        "certificate_module_sha256": sha256_file(
            PROJECT / "fixed_policy_variance_certificate.py"
        ),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "commands.log").write_text(
        " ".join(sys.argv) + "\n", encoding="utf-8"
    )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to write into a nonempty directory: {args.output_dir}")

    bundle = run_matrix(args.mode, args.output_dir, args.label or args.mode)
    emitted = sum(
        1
        for record in bundle["records"]
        for route in PRIMARY
        if record["routes"][route]["update_emitted"]
    )
    attempted = len(bundle["records"]) * len(PRIMARY)
    print(
        f"{TASK_ID} {args.mode}: {bundle['record_count']} records, "
        f"primary emissions {emitted}/{attempted}"
    )


if __name__ == "__main__":
    main()
