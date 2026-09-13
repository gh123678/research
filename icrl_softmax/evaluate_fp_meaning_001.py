"""FP-MEANING-001: measure the arithmetic floor of the value computation.

`FP-HORIZON-001` records `min_lb` -- the CERTIFIED lower bound on the worst-state
improvement -- decaying geometrically to `2.164e-16` by step 32. That is within a
factor of 1.03 of float64 machine epsilon. Whether that means the tail updates prove
nothing depends entirely on the resolution of the arithmetic, so this measures it
rather than assuming it.

Two routes per record, sharing no code:

  policy_quantities(mdp, policy)          the routine the whole line uses
  value iteration under pi to 1e-14       written from the Bellman operator

and the disagreement is the floor. `numpy.finfo(float64).eps` is recorded alongside so
the measured and assumed values can be compared rather than conflated.

Usage:
    python -B evaluate_fp_meaning_001.py --output-dir <dir>
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

TASK_ID = "FP-MEANING-001"
SEALED_FILES = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "model.py",
)


def value_iteration(mdp: Any, policy: np.ndarray, gamma: float) -> np.ndarray:
    """v^pi by iterating v <- sum_a pi(a|s)[R(s,a) + gamma P(s'|s,a) v] to 1e-14."""
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    n_states = policy.shape[0]
    r_sa = np.einsum("sap,sap->sa", transition, reward)
    v = np.zeros(n_states)
    for _ in range(500000):
        q = r_sa + gamma * np.einsum("sap,p->sa", transition, v)
        v_new = np.einsum("sa,sa->s", policy, q)
        delta = float(np.max(np.abs(v_new - v)))
        v = v_new
        if delta < 1e-14:
            break
    return v


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--mixings", type=str, default="0.08,0.5")
    parser.add_argument("--label", type=str, default="arithmetic-floor")
    args = parser.parse_args()

    mixings = tuple(float(v) for v in args.mixings.split(","))
    eps = float(np.finfo(np.float64).eps)

    records: list[dict[str, Any]] = []
    for mixing in mixings:
        for task_index in range(args.tasks):
            mdp, policy, _ = fs.build_task(task_index=task_index, mixing=mixing)
            exact = policy_quantities(mdp, policy)
            v_a = np.asarray(exact["v_pi"], dtype=np.float64)
            v_b = value_iteration(mdp, policy, fs.GAMMA)
            gap = float(np.max(np.abs(v_a - v_b)))
            scale = float(np.max(np.abs(v_a)))
            records.append(
                {
                    "task_id": TASK_ID,
                    "mixing": float(mixing),
                    "task_index": int(task_index),
                    "v_policy_quantities": v_a.tolist(),
                    "v_value_iteration": v_b.tolist(),
                    "absolute_gap": gap,
                    "relative_gap": gap / scale if scale > 0 else None,
                    "value_scale": scale,
                }
            )
            print(
                f"  mixing={mixing} task={task_index:>2} "
                f"|dv| = {gap:.3e}  (|v| ~ {scale:.3f}, eps = {eps:.3e})",
                flush=True,
            )

    gaps = [r["absolute_gap"] for r in records]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_results.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "label": args.label,
                "record_count": len(records),
                "float64_eps": eps,
                "max_absolute_gap": max(gaps),
                "median_absolute_gap": float(np.median(gaps)),
                "records": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "label": args.label,
                "mixings": list(mixings),
                "tasks": args.tasks,
                "method_a": "evaluate_fixed_policy_q_routes.policy_quantities",
                "method_b": "value iteration to a 1e-14 tolerance",
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
                "float64_eps": eps,
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
    print(
        f"{TASK_ID}: {len(records)} records; "
        f"max |dv| = {max(gaps):.3e}, median = {np.median(gaps):.3e}, "
        f"eps = {eps:.3e}"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


if __name__ == "__main__":
    main()
