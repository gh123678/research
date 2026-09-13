"""FP-CERTFIX-001 evaluator: the corrected certificate at step 1 and over a short
multi-step horizon, with fresh independent certification data at EVERY step.

What this fixes relative to ``evaluate_fp_iter8x_001.py`` (derivation
``docs/derivations/FP-CERTFIX-001-certificate-rederivation.md``):

1. Per-pair sample size is a FIXED, pre-registered ``n_per_pair``; the batch
   keeps each pair's FIRST n visits (lemma A). No splitting by realised counts.
2. The certificate is Maurer-Pontil on a single iid fixed-n sample (arm ``mp``),
   or the minimally repaired split-half Bernstein (arm ``split``), both with the
   KNOWN range and correct constants. Neither substitutes an estimated scale
   for the range; no ``sqrt(2)`` is dropped.
3. Every step draws a FRESH batch from the step-indexed seed schedule, so the
   step-k certificate sees a batch independent of the history that produced
   ``pi_{k-1}`` and ``q_hat_k`` (theorem 2). The old protocol reused one batch
   for all 12 steps, which broke the fixed-object premise.

Guarantee scope: per-trajectory, over the frozen horizon ``K``, with
``delta_k = DELTA_TOTAL / K`` per step; step-1-only runs use ``K = 1`` so their
risk is ``0.05`` per record, matching the sealed runs' per-call budget. Joint
statements across records/arms are NOT allocated and are labelled as such.

Arms share the same step-k batch between them; each arm's certificate is valid
conditionally on the history, so each arm's trajectory carries its own
``delta_total`` statement.
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
import fixed_policy_mp_certificate as mc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402
from fp_certfix_first_n import (  # noqa: E402
    first_n_batch,
    first_visit_batch,
    step_seed_parts,
)
from fp_sample_vectorised_batch import vectorised_batch  # noqa: E402

TASK_ID = "FP-CERTFIX-001"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("mp", "split")
TASK_SALT = 90417  # pre-registered stream salt for this task's certification data
DELTA_TOTAL = 0.05
MIXINGS = fs.MIXING
TASKS_PER_CELL = fs.TASKS_PER_CELL
CHAIN_LENGTH = 64

SEALED_FILES = (
    "fixed_policy_expected_sarsa.py",
    "fixed_policy_expected_sarsa_scaled.py",
    "fixed_policy_variance_certificate.py",
    "fixed_policy_bernstein_certificate.py",
    "model.py",
)
NEW_FILES = (
    "fixed_policy_mp_certificate.py",
    "fp_certfix_first_n.py",
    "evaluate_fp_certfix_001.py",
    "fp_sample_vectorised_batch.py",
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


def certificate_for(
    arm: str,
    q_hat: np.ndarray,
    policy: np.ndarray,
    batch: dict,
    n_per_pair: int,
    delta_step: float,
    extraction: str = "first_n",
    min_visits: int = 2000,
) -> dict:
    if extraction == "first_visit":
        if arm == "mp":
            return mc.mp_certificate_firstvisit(
                q_hat, policy, batch, min_visits=min_visits, delta_step=delta_step
            )
        return mc.split_bernstein_firstvisit(
            q_hat, policy, batch, min_visits=min_visits, delta_step=delta_step
        )
    # Deprecated path: reproduces the 2026-09-13 morning artifacts only.
    if arm == "mp":
        return mc.mp_certificate(
            q_hat, policy, batch, n_per_pair=n_per_pair, delta_step=delta_step
        )
    return mc.split_bernstein_certificate(
        q_hat, policy, batch, n_per_pair=n_per_pair, delta_step=delta_step
    )


def make_producer(name: str):
    """Return a ``produce(route, policy, train) -> q_hat`` callable.

    ``numpy`` uses the sealed array-formula routes; ``network`` (FP-CERTCHECK-001)
    uses the literal attention networks of ``model.py`` via the FP-ATTN-001
    ``network_qhat`` helper. Everything downstream (certificate, decision,
    audit) is identical for both producers.
    """
    if name == "numpy":
        return lambda route, policy, train: fs.run_route(route, policy, train)["q_hat"]
    if name == "network":
        import torch  # local import: numpy mode must not require torch

        from evaluate_fp_attn_iter_001 import network_qhat
        from model import (
            EndToEndFiniteSoftmaxExpectedSARSA,
            EndToEndMaskedSoftmaxExpectedSARSA,
        )

        networks = {
            "expected_exact": EndToEndMaskedSoftmaxExpectedSARSA(
                gamma=fs.GAMMA, alpha=fs.ALPHA
            ),
            "expected_finite": EndToEndFiniteSoftmaxExpectedSARSA(
                gamma=fs.GAMMA, alpha=fs.ALPHA, zeta=fs.ZETA, xi=fs.XI, tau=fs.TAU
            ),
        }

        def produce(route, policy, train):
            q_literal, _ = network_qhat(networks[route], policy, train)
            return np.asarray(q_literal, dtype=np.float64)

        return produce
    raise ValueError(f"unknown producer {name!r}")


def run_records(args, mixings, tasks) -> dict[str, Any]:
    max_steps = 1 if args.mode == "step1" else int(args.max_steps)
    delta_step = DELTA_TOTAL / max_steps  # theorem 2: uniform allocation, K frozen
    produce = make_producer(args.producer)
    records: list[dict[str, Any]] = []
    items_drawn_total = 0
    for mixing in mixings:
        for task_index in range(tasks):
            mdp, behaviour, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact0 = policy_quantities(mdp, behaviour)
            mu_state = np.asarray(exact0["mu_state"], dtype=np.float64)
            train = fs.training_batch(mdp, behaviour, mu_state, rng)

            routes: dict[str, Any] = {}
            for route in PRIMARY:
                # Shared per step across arms: the step-k batch is drawn fresh
                # from the step-indexed schedule, independent of BOTH arms'
                # histories; each arm's trajectory keeps its own delta_total
                # statement (theorem 2 applied per arm).
                step_batches: list[Any] = []
                arms_out: dict[str, Any] = {}
                for arm in ARMS:
                    current = behaviour.copy()
                    q_ref = np.asarray(exact0["q_pi"], dtype=np.float64).copy()
                    v_chain = [np.asarray(exact0["v_pi"], dtype=np.float64).copy()]
                    steps: list[dict[str, Any]] = []
                    for step_index in range(1, max_steps + 1):
                        # q_hat_k is a function of (train, pi_{k-1}) ONLY.
                        q_hat = np.asarray(
                            produce(route, current, train),
                            dtype=np.float64,
                        ).reshape(fs.N_STATES, fs.N_ACTIONS)
                        if step_index > len(step_batches):
                            raw = vectorised_batch(
                                mdp,
                                behaviour,
                                mu_state,
                                step_seed_parts(
                                    fs.SEED, TASK_SALT, mixing, task_index, step_index
                                ),
                                int(args.chains),
                                CHAIN_LENGTH,
                            )
                            items_drawn_total += int(args.chains) * CHAIN_LENGTH
                            if args.extraction == "first_visit":
                                step_batches.append(
                                    first_visit_batch(raw, CHAIN_LENGTH)
                                )
                            else:
                                step_batches.append(
                                    first_n_batch(raw, int(args.n_per_pair))
                                )
                        reduced, counts = step_batches[step_index - 1]
                        realized = float(np.max(np.abs(q_hat - q_ref)))
                        if reduced is None:
                            certificate = {
                                "status": "not_certified",
                                "failure_reasons": ["heldout_pair_support_missing"],
                                "e_q": None,
                                "delta_step": delta_step,
                                "delta_each": None,
                            }
                            decision = {
                                "status": "not_certified",
                                "eta_selected": None,
                                "lb_by_state": np.zeros(fs.N_STATES),
                            }
                        else:
                            certificate = certificate_for(
                                arm,
                                q_hat,
                                current,
                                reduced,
                                int(args.n_per_pair),
                                delta_step,
                                extraction=args.extraction,
                                min_visits=int(args.min_visits),
                            )
                            decision = fs.improvement_for(current, q_hat, certificate)
                        emitted = decision["status"] == "safe_update_emitted"
                        e_q = certificate.get("e_q")
                        lb = np.asarray(
                            decision["lb_by_state"], dtype=np.float64
                        ).reshape(-1)
                        entry: dict[str, Any] = {
                            "step": step_index,
                            "status": decision["status"],
                            "eta_selected": decision["eta_selected"],
                            "e_q": e_q,
                            "delta_step": delta_step,
                            "delta_each": certificate.get("delta_each"),
                            "min_lb": float(lb.min()) if lb.size else None,
                            "max_lb": float(lb.max()) if lb.size else None,
                            "update_emitted": bool(emitted),
                            "min_pair_count_observed": int(counts.min()),
                            "cert_means": np.asarray(
                                certificate.get("residual_means", []), dtype=np.float64
                            ).tolist(),
                            "cert_radii": np.asarray(
                                certificate.get("radii", []), dtype=np.float64
                            ).tolist(),
                            "cert_sample_vars": np.asarray(
                                certificate.get("sample_vars", []), dtype=np.float64
                            ).tolist()
                            if arm == "mp" and certificate.get("sample_vars") is not None
                            else None,
                            "cert_pair_sizes": np.asarray(
                                certificate.get("pair_sizes", []), dtype=np.int64
                            ).tolist()
                            if certificate.get("pair_sizes") is not None
                            else None,
                            "ordered_reasons": fs.route_failure_reasons(
                                certificate, decision
                            ),
                            "oracle_audit": {
                                "purpose": "truth-based audit only; never a certificate input",
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
                        "final_v_sum": float(np.sum(v_chain[-1])),
                        "initial_v_sum": float(np.sum(v_chain[0])),
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
    return {
        "task_id": TASK_ID,
        "mode": args.mode,
        "label": args.label,
        "producer": args.producer,
        "extraction": args.extraction,
        "min_visits": int(args.min_visits),
        "arms": list(ARMS),
        "routes": list(PRIMARY),
        "n_per_pair": int(args.n_per_pair),
        "chains_per_step": int(args.chains),
        "chain_length": CHAIN_LENGTH,
        "delta_total": DELTA_TOTAL,
        "max_steps": max_steps,
        "delta_step": delta_step,
        "task_salt": TASK_SALT,
        "items_drawn_total": items_drawn_total,
        "record_count": len(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=["smoke", "step1", "multi"], required=True)
    parser.add_argument("--n-per-pair", type=int, required=True)
    parser.add_argument("--chains", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--producer", choices=["numpy", "network"], default="numpy")
    parser.add_argument(
        "--extraction",
        choices=["first_visit", "first_n"],
        default="first_visit",
        help="first_visit = valid chain-replicated sample (lemma A'); "
        "first_n = DEPRECATED, reproduces the 2026-09-13 morning artifacts only",
    )
    parser.add_argument("--min-visits", type=int, default=2000)
    args = parser.parse_args()

    if args.mode == "smoke":
        mixings, tasks = (0.08,), 2
    else:
        mixings, tasks = MIXINGS, TASKS_PER_CELL

    started = datetime.now(timezone.utc)
    out = run_records(args, mixings, tasks)
    out["started_utc"] = started.isoformat()
    out["finished_utc"] = datetime.now(timezone.utc).isoformat()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "task_results.json"
    results_path.write_text(
        json.dumps(strict_ready(out), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "mode": args.mode,
                "label": args.label,
                "producer": args.producer,
                "extraction": args.extraction,
                "min_visits": int(args.min_visits),
                "n_per_pair": int(args.n_per_pair),
                "chains_per_step": int(args.chains),
                "chain_length": CHAIN_LENGTH,
                "delta_total": DELTA_TOTAL,
                "max_steps": out["max_steps"],
                "delta_step": out["delta_step"],
                "task_salt": TASK_SALT,
                "mixings": list(mixings),
                "tasks_per_cell": int(tasks),
                "sealed_file_hashes": {
                    name: sha256(PROJECT / name) for name in SEALED_FILES
                },
                "new_file_hashes": {name: sha256(PROJECT / name) for name in NEW_FILES},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "mode": args.mode,
        "records": out["record_count"],
        "items_drawn_total": out["items_drawn_total"],
        "emitted_by_arm": {
            arm: sum(
                r["routes"][route][arm]["emitted_steps"]
                for r in out["records"]
                for route in PRIMARY
            )
            for arm in ARMS
        },
        "step1_emitted_by_arm": {
            arm: sum(
                1
                for r in out["records"]
                for route in PRIMARY
                if r["routes"][route][arm]["steps"]
                and r["routes"][route][arm]["steps"][0]["update_emitted"]
            )
            for arm in ARMS
        },
        "certificate_violations": sum(
            1
            for r in out["records"]
            for route in PRIMARY
            for arm in ARMS
            for s in r["routes"][route][arm]["steps"]
            if s["oracle_audit"]["certificate_violation"]
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
