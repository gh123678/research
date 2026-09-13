"""FP-GAP-001: does the certified iteration close the optimality gap?

No new simulation. FP-HORIZON-001 already sealed, for every emitted step of every
trajectory, the per-state change in `v^{pi_k}`; cumulating those deltas reconstructs
`v^{pi_k}` exactly, and the reconstruction is checkable against the recorded final
value. The only missing ingredient is `v*`, computed exactly here by policy iteration
on a 4-state, 3-action MDP.

The measurement is the gap trajectory `gap_k = sum_s (v*_s - v^{pi_k}_s)`, the
per-step gain relative to the remaining gap, and the share of the initial
suboptimality closed.

CAVEAT the task sheet requires be repeated wherever a floor is reported: the policy
class is the relative-softmax family with `pi_min = 0.15`, not the whole simplex, so
the gap to `v*` is not expected to reach zero and part of any floor belongs to the
policy class rather than to the iteration. This evaluator does NOT decompose the two.

Usage:
    python -B evaluate_fp_gap_001.py --output-dir <dir>
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
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402

TASK_ID = "FP-GAP-001"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("frozen", "empirical_bernstein")
HORIZON_BUNDLE = (
    PROJECT / "results" / "FP-HORIZON-001" / "claude" / "formal" / "task_results.json"
)
SEALED_FILES = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "model.py",
    "evaluate_fixed_policy_q_routes.py",
)
POLICY_CLASS_CAVEAT = (
    "the policy class is the relative-softmax family with pi_min = 0.15, not the "
    "whole simplex, so the gap to v* is not expected to reach zero and part of any "
    "floor belongs to the policy class rather than to the iteration; this task does "
    "NOT decompose the two"
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


def optimal_values(mdp: Any, tolerance: float = 1e-13) -> tuple[np.ndarray, np.ndarray]:
    """Exact `v*` and a greedy optimal policy by policy iteration.

    Policy iteration on a 4x3 MDP terminates in a handful of sweeps and gives the
    exact optimum, which value iteration would only approach.
    """
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    n_states, n_actions = fs.N_STATES, fs.N_ACTIONS
    # Expected immediate reward per (s,a).
    r_sa = np.einsum("sap,sap->sa", transition, reward)
    # q from v: r_sa + gamma * sum_s' P(s'|s,a) v(s')
    greedy = np.full((n_states, n_actions), 1.0 / n_actions)
    for _ in range(1000):
        q = r_sa + fs.GAMMA * np.einsum("sap,p->sa", transition, _value_of(greedy, mdp))
        best = np.argmax(q, axis=1)
        new = np.zeros_like(greedy)
        new[np.arange(n_states), best] = 1.0
        if np.array_equal(new, greedy):
            break
        greedy = new
    v_star = _value_of(greedy, mdp)
    return v_star, greedy


def _value_of(policy: np.ndarray, mdp: Any) -> np.ndarray:
    """Exact `v^pi` by solving the linear system."""
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    r_sa = np.einsum("sap,sap->sa", transition, reward)
    n_states = policy.shape[0]
    p_pi = np.einsum("sa,sap->sp", policy, transition)
    r_pi = np.einsum("sa,sa->s", policy, r_sa)
    return np.linalg.solve(np.eye(n_states) - fs.GAMMA * p_pi, r_pi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon-bundle", type=Path, default=HORIZON_BUNDLE)
    parser.add_argument("--label", type=str, default="optimality-gap")
    args = parser.parse_args()

    source = json.loads(args.horizon_bundle.read_text(encoding="utf-8"))
    horizon = int(source["max_steps"])

    records: list[dict[str, Any]] = []
    for record in source["records"]:
        mixing = float(record["mixing"])
        task_index = int(record["task_index"])
        mdp, policy0, _ = fs.build_task(task_index=task_index, mixing=mixing)
        v_star, greedy = optimal_values(mdp)
        # Verify v* against the Bellman optimality equation rather than trusting it.
        transition = np.asarray(mdp["P"], dtype=np.float64)
        reward = np.asarray(mdp["R"], dtype=np.float64)
        r_sa = np.einsum("sap,sap->sa", transition, reward)
        q_star = r_sa + fs.GAMMA * np.einsum("sap,p->sa", transition, v_star)
        bellman_residual = float(np.max(np.abs(v_star - np.max(q_star, axis=1))))
        exact0 = policy_quantities(mdp, policy0)
        v0 = np.asarray(exact0["v_pi"], dtype=np.float64)

        routes: dict[str, Any] = {}
        for route in PRIMARY:
            arms_out: dict[str, Any] = {}
            for arm in ARMS:
                block = record["routes"][route][arm]
                start = np.asarray(block["start_policy_value"], dtype=np.float64)
                deltas = [
                    np.asarray(s["oracle_audit"]["value_delta_vs_previous"],
                               dtype=np.float64)
                    for s in block["steps"]
                    if s["update_emitted"]
                ]
                final_recorded = np.asarray(
                    block["final_policy_value"], dtype=np.float64
                )
                reconstructed = start.copy()
                trajectory = [reconstructed.copy()]
                for delta in deltas:
                    reconstructed = reconstructed + delta
                    trajectory.append(reconstructed.copy())
                reconstruction_gap = float(
                    np.max(np.abs(reconstructed - final_recorded))
                )
                gaps = [
                    float(np.sum(v_star - v)) for v in trajectory
                ]
                arms_out[arm] = {
                    "start_policy_value": start.tolist(),
                    "reconstruction_gap": reconstruction_gap,
                    "trajectory_length": len(trajectory),
                    "gap_trajectory": gaps,
                    "emitted_steps": int(block["emitted_steps"]),
                }
            routes[route] = arms_out
        records.append(
            {
                "task_id": TASK_ID,
                "mixing": mixing,
                "task_index": task_index,
                "v_star": v_star.tolist(),
                "greedy_policy": greedy.tolist(),
                "bellman_residual_of_v_star": bellman_residual,
                "v_pi_0": v0.tolist(),
                "routes": routes,
            }
        )
        gap0 = {
            arm: records[-1]["routes"][PRIMARY[0]][arm]["gap_trajectory"][0]
            for arm in ARMS
        }
        print(
            f"  mixing={mixing} task={task_index:>2} gap0 {gap0} "
            f"bellman {bellman_residual:.2e}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_results.json").write_text(
        json.dumps(
            strict_ready(
                {
                    "task_id": TASK_ID,
                    "label": args.label,
                    "record_count": len(records),
                    "horizon": horizon,
                    "source_bundle": str(args.horizon_bundle),
                    "source_bundle_sha256": hashlib.sha256(
                        args.horizon_bundle.read_bytes()
                    ).hexdigest(),
                    "policy_class_caveat": POLICY_CLASS_CAVEAT,
                    "records": records,
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "label": args.label,
                "source_bundle": str(args.horizon_bundle),
                "method": (
                    "exact v* by policy iteration; v^{pi_k} by cumulating the sealed "
                    "per-step value deltas; no new simulation"
                ),
                "policy_class_caveat": POLICY_CLASS_CAVEAT,
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
    print(f"{TASK_ID}: {len(records)} records -> {args.output_dir}")


if __name__ == "__main__":
    main()
