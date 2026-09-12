"""FP-RANGE-001: can the envelope ceiling be reached soundly?

At step 1 on all 48 route-records and at every rung of the certification ladder,
compares:

  frozen                 the sealed certificate
  empirical_bernstein    the inequality repair
  data_range             truncation with the tail price paid (SOUND)
  counterfactual         envelope deleted (the unsound ceiling, for reference only)

and records the price decomposition of the data-range certificate: the threshold
tau, the empirical tail mass, the Cauchy-Schwarz bias, and the concentration term.

Uses the vectorised sampler validated by FP-SAMPLE-001, so the rungs are affordable.

Usage:
    python -B evaluate_fp_range_001.py --output-dir <dir> [--rungs 1,2,4,8]
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

TASK_ID = "FP-RANGE-001"
PRIMARY = ("expected_exact", "expected_finite")
RUNGS = (1, 2, 4, 8)
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


def arm(certificate: dict[str, Any], policy: np.ndarray, q_hat: np.ndarray,
        realized: float) -> dict[str, Any]:
    decision = fs.improvement_for(policy, q_hat, certificate)
    e_q = certificate.get("e_q")
    out = {
        "e_q": e_q,
        "epsilon_res": certificate.get("epsilon_res"),
        "status_certificate": certificate["status"],
        "emitted": bool(decision["status"] == "safe_update_emitted"),
        "eta_selected": decision["eta_selected"],
        "ordered_reasons": fs.route_failure_reasons(certificate, decision),
        "covers_realized": None if e_q is None else bool(float(e_q) >= realized),
    }
    if "taus" in certificate:
        # The price decomposition, which is the point of the exercise.
        radii = np.asarray(certificate["radii"], dtype=np.float64)
        tail = np.asarray(certificate["tail_mass"], dtype=np.float64)
        bias = np.asarray(certificate["bias_terms"], dtype=np.float64)
        means = np.asarray(certificate["residual_means"], dtype=np.float64)
        binding = int(np.argmax(np.abs(means) + radii))
        conc = float(radii[binding] - tail[binding] - bias[binding])
        out["price_decomposition"] = {
            "binding_pair": binding,
            "tau": float(np.asarray(certificate["taus"])[binding]),
            "empirical_tail_mass": float(tail[binding]),
            "cauchy_schwarz_bias": float(bias[binding]),
            "concentration": conc,
            "tail_price_total": float(tail[binding] + bias[binding]),
            "price_exceeds_concentration": bool(
                tail[binding] + bias[binding] > conc
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rungs", type=str, default=",".join(str(r) for r in RUNGS))
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--mixings", type=str, default="0.08,0.5")
    parser.add_argument("--label", type=str, default="data-driven-range")
    args = parser.parse_args()

    rungs = tuple(int(v) for v in args.rungs.split(","))
    mixings = tuple(float(v) for v in args.mixings.split(","))

    records: list[dict[str, Any]] = []
    for mixing in mixings:
        for task_index in range(args.tasks):
            mdp, policy, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact = policy_quantities(mdp, policy)
            mu_state = np.asarray(exact["mu_state"], dtype=np.float64)
            q_pi = np.asarray(exact["q_pi"], dtype=np.float64)
            sigma_by_state = np.ptp(q_pi, axis=1)
            train = fs.training_batch(mdp, policy, mu_state, rng)
            seed_parts = [fs.SEED, SEED_OFFSET, int(round(mixing * 100)), task_index]

            routes: dict[str, Any] = {}
            for route in PRIMARY:
                q_hat = np.asarray(
                    fs.run_route(route, policy, train)["q_hat"], dtype=np.float64
                ).reshape(fs.N_STATES, fs.N_ACTIONS)
                realized = float(np.max(np.abs(q_hat - q_pi)))
                rung_out: dict[str, Any] = {}
                for rung in rungs:
                    batch = vectorised_batch(
                        mdp, policy, mu_state, seed_parts, CERT_CHAINS * rung
                    )
                    entry = {
                        "items": int(batch["states"].size),
                        "data_range": arm(
                            bc.data_range_certificate(q_hat, policy, batch),
                            policy, q_hat, realized,
                        ),
                        "empirical_bernstein": arm(
                            bc.empirical_bernstein_certificate(q_hat, policy, batch),
                            policy, q_hat, realized,
                        ),
                    }
                    if rung == rungs[0]:
                        # The unsound ceiling is only needed at 1x: it is a fixed
                        # reference, not a lever.
                        entry["counterfactual_no_envelope"] = arm(
                            bc.counterfactual_no_envelope(q_hat, policy, batch),
                            policy, q_hat, realized,
                        )
                        # The frozen certificate on the SAME sample, so H5's
                        # bias-versus-radius comparison is like-for-like rather than
                        # borrowed from a bundle that records only epsilon_res.
                        frozen = vc.variance_adaptive_certificate(
                            q_hat, policy, batch
                        )
                        frozen_entry = arm(frozen, policy, q_hat, realized)
                        fr_radii = np.asarray(frozen["radii"], dtype=np.float64)
                        fr_means = np.asarray(
                            frozen["residual_means"], dtype=np.float64
                        )
                        fr_binding = int(np.argmax(np.abs(fr_means) + fr_radii))
                        dr_binding = entry["data_range"]["price_decomposition"][
                            "binding_pair"
                        ]
                        frozen_entry["radius_at_dr_binding_pair"] = float(
                            fr_radii[dr_binding]
                        )
                        frozen_entry["radius_at_own_binding_pair"] = float(
                            fr_radii[fr_binding]
                        )
                        entry["frozen_same_sample"] = frozen_entry
                    e_q = entry["data_range"]["e_q"]
                    entry["sigma_min_over_e_q"] = (
                        None
                        if e_q in (None, 0.0)
                        else float(np.min(sigma_by_state)) / float(e_q)
                    )
                    rung_out[str(rung)] = entry
                    del batch
                routes[route] = {
                    "rungs": rung_out,
                    "oracle_audit": {
                        "purpose": (
                            "truth-based diagnostic only; never a certificate input"
                        ),
                        "realized_q_sup_error": realized,
                        "sigma_min": float(np.min(sigma_by_state)),
                    },
                }
            records.append(
                {
                    "task_id": TASK_ID,
                    "mixing": float(mixing),
                    "task_index": int(task_index),
                    "routes": routes,
                }
            )
            by_rung = {
                r: sum(
                    1
                    for route in PRIMARY
                    if routes[route]["rungs"][str(r)]["data_range"]["emitted"]
                )
                for r in rungs
            }
            print(f"  mixing={mixing} task={task_index:>2} data_range emits {by_rung}",
                  flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_results.json").write_text(
        json.dumps(
            strict_ready(
                {
                    "task_id": TASK_ID,
                    "label": args.label,
                    "record_count": len(records),
                    "rungs": list(rungs),
                    "cert_chains": CERT_CHAINS,
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
                "rungs": list(rungs),
                "cert_chains": CERT_CHAINS,
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
    print(f"{TASK_ID}: {len(records)} records, rungs {rungs} -> {args.output_dir}")


if __name__ == "__main__":
    main()
