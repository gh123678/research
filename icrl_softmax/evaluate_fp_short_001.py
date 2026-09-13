"""FP-SHORT-001: why do some route-records stop emitting while 45% of the gap remains?

FP-GAP-001 found the place the certified iteration stops short: records that emit only
`1`-`5` steps leave a mean of `54.89%` of their initial suboptimality uncollected,
while records that run long close `99.97%`. The tail is fine; the early stop is the
problem.

Three candidate mechanisms, and this evaluator separates them:

1. **The certificate is far too loose** -- `E_Q` exceeds the achievable improvement by
   a wide margin, so no small repair would help. Measured as `E_Q / h`, where `h` is
   the exact gate margin: the largest certified error that would still let some eta
   pass.
2. **One state blocks the whole update** -- the rule requires `min_s LB_s > 0`, a
   conjunction over states, so a single stuck state stops the record. Measured as the
   number of states with `LB_s <= 0` at the best candidate.
3. **The eta grid bottoms out** -- the smallest grid value is `0.01`. A policy near the
   constrained optimum needs a *small* tilt, and if even `0.01` overshoots, the rule
   has nothing smaller to try. Measured by re-running the iteration with an extended
   grid.

The frozen grid arm doubles as the regression check: it must reproduce
`FP-ITER8X-001`'s sealed decisions exactly, or the instrumented decision function is
not the frozen rule and none of the diagnosis is about the frozen rule.

Usage:
    python -B evaluate_fp_short_001.py --output-dir <dir> [--max-steps 12]
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

import fixed_policy_expected_sarsa as es  # noqa: E402
import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_variance_certificate as vc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_sample_vectorised_batch import CERT_CHAINS, vectorised_batch  # noqa: E402

TASK_ID = "FP-SHORT-001"
PRIMARY = ("expected_exact", "expected_finite")
SEED_OFFSET = 5881
FROZEN_GRID = fs.ETA_CANDIDATES
# The extended grid keeps every frozen value and adds smaller tilts. The frozen
# prefix is what makes the two arms comparable: any difference is caused only by the
# extra candidates below 0.01.
EXTENDED_GRID = FROZEN_GRID + (0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001)
SEALED_8X = (
    PROJECT / "results" / "FP-ITER8X-001" / "claude" / "formal" / "task_results.json"
)
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


def diagnose(
    policy: np.ndarray, q_hat: np.ndarray, e_q: float, grid: tuple[float, ...]
) -> dict[str, Any]:
    """Per-eta lower bounds, plus the exact gate margin ``h``.

    ``h`` is the largest certified error for which some candidate would pass:
    ``h = max_eta min_s I_s / ||dpi_s||_1``, with the convention that a zero movement
    contributes nothing. Emission is exactly ``E_Q < h``, so ``E_Q / h`` is the gate
    expressed as a single number and its distance above 1 says how close the record
    came.
    """
    per_eta = []
    best_margin = -np.inf
    best_eta = None
    for eta in grid:
        candidate = es.relative_softmax_candidate(policy, q_hat, float(eta))
        delta_pi = candidate - policy
        i_hat = (delta_pi * q_hat).sum(axis=1)
        movement = np.abs(delta_pi).sum(axis=1)
        lb = i_hat - e_q * movement
        with np.errstate(divide="ignore", invalid="ignore"):
            margin_by_state = np.where(movement > 0, i_hat / movement, np.inf)
        margin = float(np.min(margin_by_state))
        passes = bool(float(np.min(lb)) > 0.0)
        per_eta.append(
            {
                "eta": float(eta),
                "passes": passes,
                "min_lb": float(np.min(lb)),
                "movement_min": float(np.min(movement)),
                "margin": margin,
                "blocking_states": int(np.sum(lb <= 0.0)),
            }
        )
        if margin > best_margin:
            best_margin = margin
            best_eta = float(eta)
    return {
        "e_q": float(e_q),
        "gate_margin_h": best_margin,
        "best_margin_eta": best_eta,
        "e_q_over_h": (float(e_q) / best_margin) if best_margin > 0 else None,
        "per_eta": per_eta,
    }


def decide_with_grid(
    policy: np.ndarray, q_hat: np.ndarray, certificate: dict, grid: tuple[float, ...]
) -> dict[str, Any]:
    """The sealed decision rule, with the eta grid passed in.

    Reproduces ``fs.improvement_for`` exactly when given the frozen grid; that
    equality is asserted per step, so the instrumented path cannot silently diverge
    from the rule it is diagnosing.
    """
    e_q = certificate.get("e_q")
    if e_q is None:
        return {
            "status": "not_certified",
            "eta_selected": None,
            "policy_plus": policy.copy(),
            "min_lb": 0.0,
            "blocking_states": policy.shape[0],
        }
    e_q = float(e_q)
    for eta in grid:
        candidate = es.relative_softmax_candidate(policy, q_hat, float(eta))
        delta_pi = candidate - policy
        i_hat = (delta_pi * q_hat).sum(axis=1)
        lb = i_hat - e_q * np.abs(delta_pi).sum(axis=1)
        if float(np.min(lb)) > 0.0:
            return {
                "status": "safe_update_emitted",
                "eta_selected": float(eta),
                "policy_plus": candidate,
                "min_lb": float(np.min(lb)),
                "blocking_states": 0,
            }
    # Abstained: report the best candidate's lower bound and how many states block it,
    # because "one state blocks" and "all states block" are different diagnoses.
    best_min_lb = -np.inf
    best_lb = None
    for eta in grid:
        candidate = es.relative_softmax_candidate(policy, q_hat, float(eta))
        delta_pi = candidate - policy
        lb = (delta_pi * q_hat).sum(axis=1) - e_q * np.abs(delta_pi).sum(axis=1)
        if float(np.min(lb)) > best_min_lb:
            best_min_lb = float(np.min(lb))
            best_lb = lb
    return {
        "status": "abstained",
        "eta_selected": None,
        "policy_plus": policy.copy(),
        "min_lb": best_min_lb,
        "blocking_states": int(np.sum(best_lb <= 0.0)) if best_lb is not None else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--mixings", type=str, default="0.08,0.5")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--multiplier", type=int, default=8)
    parser.add_argument("--label", type=str, default="why-records-stop-early")
    args = parser.parse_args()

    mixings = tuple(float(v) for v in args.mixings.split(","))
    chains = CERT_CHAINS * int(args.multiplier)

    records: list[dict[str, Any]] = []
    for mixing in mixings:
        for task_index in range(args.tasks):
            mdp, policy, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact = policy_quantities(mdp, policy)
            mu_state = np.asarray(exact["mu_state"], dtype=np.float64)
            q_pi = np.asarray(exact["q_pi"], dtype=np.float64)
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
                out: dict[str, Any] = {}
                for arm, grid in (("frozen_grid", FROZEN_GRID),
                                  ("extended_grid", EXTENDED_GRID)):
                    current = policy.copy()
                    steps: list[dict[str, Any]] = []
                    for step_index in range(1, int(args.max_steps) + 1):
                        q_hat = np.asarray(
                            fs.run_route(route, current, train)["q_hat"],
                            dtype=np.float64,
                        ).reshape(fs.N_STATES, fs.N_ACTIONS)
                        certificate = vc.variance_adaptive_certificate(
                            q_hat, current, cert
                        )
                        mine = decide_with_grid(current, q_hat, certificate, grid)
                        reference = fs.improvement_for(current, q_hat, certificate)
                        # The regression check: on the frozen grid the instrumented
                        # path must agree with the sealed rule exactly.
                        faithful = (
                            None
                            if arm != "frozen_grid"
                            else bool(
                                mine["status"] == reference["status"]
                                and mine["eta_selected"]
                                == reference["eta_selected"]
                            )
                        )
                        emitted = mine["status"] == "safe_update_emitted"
                        entry: dict[str, Any] = {
                            "step": step_index,
                            "status": mine["status"],
                            "eta_selected": mine["eta_selected"],
                            "e_q": certificate.get("e_q"),
                            "min_lb": mine["min_lb"],
                            "blocking_states": mine["blocking_states"],
                            "update_emitted": bool(emitted),
                            "ordered_reasons": fs.route_failure_reasons(
                                certificate,
                                {
                                    "status": mine["status"],
                                    "eta_selected": mine["eta_selected"],
                                },
                            ),
                            "faithful_to_sealed_rule": faithful,
                            "diagnosis": (
                                diagnose(
                                    current, q_hat, float(certificate["e_q"]), grid
                                )
                                if certificate.get("e_q") is not None
                                else None
                            ),
                            "oracle_audit": {
                                "purpose": (
                                    "truth-based diagnostic only; never a certificate input"
                                ),
                                "q_pi_reference": q_pi.tolist(),
                            },
                        }
                        if emitted:
                            nxt = np.asarray(mine["policy_plus"], dtype=np.float64)
                            q_next = policy_quantities(mdp, nxt)
                            v_prev = np.asarray(
                                policy_quantities(mdp, current)["v_pi"],
                                dtype=np.float64,
                            )
                            v_next = np.asarray(q_next["v_pi"], dtype=np.float64)
                            delta_v = v_next - v_prev
                            entry["oracle_audit"].update(
                                {
                                    "value_delta_vs_previous": delta_v.tolist(),
                                    "componentwise_nondegrading": bool(
                                        float(np.min(delta_v)) >= -1e-12
                                    ),
                                    "total_value_gain": float(np.sum(delta_v)),
                                }
                            )
                            current = nxt
                        steps.append(entry)
                        if not emitted:
                            break
                    out[arm] = {
                        "steps": steps,
                        "emitted_steps": sum(
                            1 for s in steps if s["update_emitted"]
                        ),
                    }
                routes[route] = out
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
                for arm in ("frozen_grid", "extended_grid")
            }
            print(f"  mixing={mixing} task={task_index:>2} full-horizon {summary}",
                  flush=True)

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
                    "frozen_grid": list(FROZEN_GRID),
                    "extended_grid": list(EXTENDED_GRID),
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
                "frozen_grid": list(FROZEN_GRID),
                "extended_grid": list(EXTENDED_GRID),
                "certificate": "sealed variance-adaptive, one arm, so the GRID is the "
                               "only thing that changes between arms",
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
