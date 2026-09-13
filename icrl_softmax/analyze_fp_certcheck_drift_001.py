"""FP-CERTCHECK-001: network-vs-numpy estimate drift along the sealed trajectories.

Decisions under the corrected protocol were already shown to be identical
(0 divergences). This script measures the quantity behind that: at every sealed
step of every trajectory, recompute ``q_hat`` with BOTH producers (sealed numpy
routes and the literal networks) for the SAME policy and training batch, and
record the sup-norm gap. Because the decisions are identical, one replay of the
sealed policy trajectory suffices -- both producers are evaluated at each
replayed step.

Output: per-step max gap, plus the per-record gap sequences, sealed to
``results/FP-CERTCHECK-001/claude/drift/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from evaluate_fp_attn_iter_001 import network_qhat  # noqa: E402
from model import (  # noqa: E402
    EndToEndFiniteSoftmaxExpectedSARSA,
    EndToEndMaskedSoftmaxExpectedSARSA,
)

TOLERANCE = 1e-4  # frozen tolerance used across the FP line


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.results / "task_results.json").read_text(encoding="utf-8"))
    routes = list(data["routes"])

    networks = {
        "expected_exact": EndToEndMaskedSoftmaxExpectedSARSA(
            gamma=fs.GAMMA, alpha=fs.ALPHA
        ),
        "expected_finite": EndToEndFiniteSoftmaxExpectedSARSA(
            gamma=fs.GAMMA, alpha=fs.ALPHA, zeta=fs.ZETA, xi=fs.XI, tau=fs.TAU
        ),
    }

    per_step_worst: dict[str, list[float]] = {}
    details: list[dict[str, Any]] = []
    worst = 0.0
    for rec in data["records"]:
        mixing = float(rec["mixing"])
        task_index = int(rec["task_index"])
        mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
        exact0 = policy_quantities(mdp, behaviour)
        mu_state = np.asarray(exact0["mu_state"], dtype=np.float64)
        train = fs.training_batch(mdp, behaviour, mu_state, rng)
        for route in routes:
            # decisions were identical across producers, so the mp arm's sealed
            # trajectory is THE trajectory both producers walk.
            sealed_steps = rec["routes"][route]["mp"]["steps"]
            current = behaviour.copy()
            for s in sealed_steps:
                q_np = np.asarray(
                    fs.run_route(route, current, train)["q_hat"], dtype=np.float64
                ).reshape(fs.N_STATES, fs.N_ACTIONS)
                q_net, _ = network_qhat(networks[route], current, train)
                q_net = np.asarray(q_net, dtype=np.float64).reshape(
                    fs.N_STATES, fs.N_ACTIONS
                )
                gap = float(np.max(np.abs(q_np - q_net)))
                worst = max(worst, gap)
                per_step_worst.setdefault(str(s["step"]), []).append(gap)
                details.append(
                    {
                        "record": f"{mixing}/{task_index}/{route}",
                        "step": int(s["step"]),
                        "gap": gap,
                    }
                )
                if not bool(s["update_emitted"]):
                    break
                cand = fs.es.relative_softmax_candidate(
                    current, q_np, float(s["eta_selected"])
                )
                current = np.asarray(cand, dtype=np.float64)

    per_step = {
        k: {"max": max(v), "mean": sum(v) / len(v), "n": len(v)}
        for k, v in sorted(per_step_worst.items(), key=lambda kv: int(kv[0]))
    }
    out = {
        "source_results": str(args.results),
        "tolerance": TOLERANCE,
        "worst_gap": worst,
        "within_tolerance": worst <= TOLERANCE,
        "per_step": per_step,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "drift.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "drift_steps.json").write_text(
        json.dumps(details, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
