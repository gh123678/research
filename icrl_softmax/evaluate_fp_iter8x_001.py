"""FP-ITER8X-001: does eight times the certification data extend the iteration?

FP-SAMPLE-001 established the lever: at step 1, `8x` certification raises the number
of eligible route-records from `30` to `44` of `48`, soundly. FP-ITER6-001 and
FP-ATTN-ITER6-001 established the ceiling with `1x`: the certified iteration reaches
six steps on both implementations, the emitting population falls `22, 20, 15, 12,
12, 9`, and the minimum gain drops below the `0.05` floor.

Nobody has yet put the two together. This evaluator does, by running the certified
iteration on `8x` certification with a long horizon and two certificate arms:

  ``frozen``               the sealed certificate, so the DATA is the only change
  ``empirical_bernstein``  the sound repair plus the data, so both levers act

The frozen arm is the control that separates "more data" from "a better bound".
Without it, a longer iteration could not be attributed to either.

The trajectory is per arm, because each arm's decision at step `k` determines the
policy at step `k+1`. The sealed protocol is otherwise inherited verbatim: identical
batches at every step, `pi_{k-1}` as the step-`k` target, the frozen decision rule,
no retuning.

Usage:
    python -B evaluate_fp_iter8x_001.py --output-dir <dir> [--max-steps 12]
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

import fixed_policy_bernstein_certificate as bc  # noqa: E402
import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_variance_certificate as vc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_sample_vectorised_batch import CERT_CHAINS, vectorised_batch  # noqa: E402

TASK_ID = "FP-ITER8X-001"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("frozen", "empirical_bernstein")
SEED_OFFSET = 5881
SEALED_FILES = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
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


def certificate_for(arm: str, q_hat: np.ndarray, policy: np.ndarray, cert: dict):
    if arm == "frozen":
        return vc.variance_adaptive_certificate(q_hat, policy, cert)
    if arm == "empirical_bernstein":
        return bc.empirical_bernstein_certificate(q_hat, policy, cert)
    raise ValueError(f"unknown arm {arm!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--mixings", type=str, default="0.08,0.5")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--multiplier", type=int, default=8)
    parser.add_argument("--label", type=str, default="iteration-at-8x")
    args = parser.parse_args()

    mixings = tuple(float(v) for v in args.mixings.split(","))
    chains = CERT_CHAINS * int(args.multiplier)

    records: list[dict[str, Any]] = []
    for mixing in mixings:
        for task_index in range(args.tasks):
            mdp, policy, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact = policy_quantities(mdp, policy)
            mu_state = np.asarray(exact["mu_state"], dtype=np.float64)
            train = fs.training_batch(mdp, policy, mu_state, rng)
            cert = vectorised_batch(
                mdp,
                policy,
                mu_state,
                [fs.SEED, SEED_OFFSET, int(round(mixing * 100)), task_index],
                chains,
            )

            routes: dict[str, Any] = {}
            for route in PRIMARY:
                arms_out: dict[str, Any] = {}
                for arm in ARMS:
                    current = policy.copy()
                    v_chain = [np.asarray(exact["v_pi"], dtype=np.float64).copy()]
                    # The certificate bounds ||Q_k - Q^{pi_{k-1}}||, so the audit must
                    # use the matching fixed point at every step. Comparing against
                    # Q^{pi_0} at every step was a bug found and fixed in FP-ITER2-001.
                    q_ref = np.asarray(exact["q_pi"], dtype=np.float64).copy()
                    steps: list[dict[str, Any]] = []
                    for step_index in range(1, int(args.max_steps) + 1):
                        q_hat = np.asarray(
                            fs.run_route(route, current, train)["q_hat"],
                            dtype=np.float64,
                        ).reshape(fs.N_STATES, fs.N_ACTIONS)
                        certificate = certificate_for(arm, q_hat, current, cert)
                        decision = fs.improvement_for(current, q_hat, certificate)
                        emitted = decision["status"] == "safe_update_emitted"
                        realized = float(np.max(np.abs(q_hat - q_ref)))
                        e_q = certificate.get("e_q")
                        lb = np.asarray(
                            decision["lb_by_state"], dtype=np.float64
                        ).reshape(-1)
                        entry: dict[str, Any] = {
                            "step": step_index,
                            "status": decision["status"],
                            "eta_selected": decision["eta_selected"],
                            "e_q": e_q,
                            "min_lb": float(lb.min()) if lb.size else None,
                            "max_lb": float(lb.max()) if lb.size else None,
                            "update_emitted": bool(emitted),
                            "ordered_reasons": fs.route_failure_reasons(
                                certificate, decision
                            ),
                            "oracle_audit": {
                                "purpose": (
                                    "truth-based audit only; never a certificate input"
                                ),
                                "realized_q_sup_error_vs_current_target_pi": realized,
                                "certificate_violation": bool(
                                    e_q is not None and float(e_q) < realized
                                ),
                            },
                        }
                        if emitted:
                            nxt = np.asarray(
                                decision["policy_plus"], dtype=np.float64
                            )
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
                    arms_out[arm] = {
                        "steps": steps,
                        "emitted_steps": sum(
                            1 for s in steps if s["update_emitted"]
                        ),
                        "final_policy_value": v_chain[-1].tolist(),
                        "start_policy_value": v_chain[0].tolist(),
                    }
                routes[route] = arms_out
            records.append(
                {
                    "task_id": TASK_ID,
                    "mixing": float(mixing),
                    "task_index": int(task_index),
                    "routes": routes,
                }
            )
            summary = {
                arm: sum(
                    1
                    for route in PRIMARY
                    if routes[route][arm]["emitted_steps"] >= int(args.max_steps)
                )
                for arm in ARMS
            }
            print(
                f"  mixing={mixing} task={task_index:>2} full-horizon routes {summary}",
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
                    "max_steps": int(args.max_steps),
                    "multiplier": int(args.multiplier),
                    "cert_chains": chains,
                    "arms": list(ARMS),
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
                "mixings": list(mixings),
                "tasks": args.tasks,
                "max_steps": int(args.max_steps),
                "multiplier": int(args.multiplier),
                "cert_chains": chains,
                "arms": list(ARMS),
                "sampler": "fp_sample_vectorised_batch.vectorised_batch",
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
                "certificate_module_hash": sha256(
                    PROJECT / "fixed_policy_bernstein_certificate.py"
                ),
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
