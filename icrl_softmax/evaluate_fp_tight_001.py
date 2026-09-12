"""FP-TIGHT-001: do sound certificate repairs buy enough to flip the abstainers?

At step 1 of the frozen pi_0 for every one of the 48 route-records, this computes

  * the sealed certificate's E_Q                         (reference)
  * the Bernstein repair's E_Q                           (corrected inequality)
  * the empirical-Bernstein repair's E_Q                 (observed variance too)
  * the realized oracle sup-error                        (coverage audit only)
  * sigma_min                                             (oracle, audit only)
  * the decision each certificate produces under the UNCHANGED frozen rule

and, optionally with ``--sample-multiplier K``, the empirical-Bernstein certificate
on a K-times larger certification sample, to price the sample-size lever.

The frozen certificate module is imported READ-ONLY and is never modified; the
sealed batch generator is reproduced exactly so all arms sit on the same items.

Usage:
    python -B evaluate_fp_tight_001.py --output-dir <dir> [--tasks N]
                                       [--sample-multiplier K] [--label NAME]
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

TASK_ID = "FP-TIGHT-001"
PRIMARY = ("expected_exact", "expected_finite")
MIXINGS = (0.08, 0.5)
TASKS = 12
# FP-SCALE-002 certification constants, exactly as the sealed evaluators use them.
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
    mdp: Any,
    policy: np.ndarray,
    mu_state: np.ndarray,
    seed_parts: list[Any],
    chains: int = CERT_CHAINS,
    chain_length: int = CERT_CHAIN_LENGTH,
) -> dict[str, np.ndarray]:
    """The sealed generator, with the chain count left as a parameter.

    The draw order is per chain, one action then one next state per step, exactly as
    the sealed evaluators do it. ``chains`` is a parameter only so the sample-size
    arm can ask for more data; at the default it is the sealed batch.
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


def decision_of(certificate: dict[str, Any], policy: np.ndarray, q_hat: np.ndarray):
    """The frozen decision rule, applied to whichever certificate is supplied."""
    decision = fs.improvement_for(policy, q_hat, certificate)
    return {
        "status": decision["status"],
        "emitted": bool(decision["status"] == "safe_update_emitted"),
        "eta_selected": decision["eta_selected"],
        "min_lb": (
            float(np.min(np.asarray(decision["lb_by_state"], dtype=np.float64)))
            if certificate.get("e_q") is not None
            else None
        ),
        "ordered_reasons": fs.route_failure_reasons(certificate, decision),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=TASKS)
    parser.add_argument("--mixings", type=str, default="0.08,0.5")
    parser.add_argument("--label", type=str, default="tightened-certificate")
    parser.add_argument(
        "--sample-multiplier",
        type=int,
        default=1,
        help="certification chains multiplier for the empirical-Bernstein arm",
    )
    args = parser.parse_args()

    mixings = tuple(float(v) for v in args.mixings.split(","))
    multiplier = int(args.sample_multiplier)

    records: list[dict[str, Any]] = []
    for mixing in mixings:
        for task_index in range(args.tasks):
            mdp, policy, rng = fs.build_task(task_index=task_index, mixing=mixing)
            exact = policy_quantities(mdp, policy)
            mu_state = np.asarray(exact["mu_state"], dtype=np.float64)
            q_pi = np.asarray(exact["q_pi"], dtype=np.float64)
            sigma_by_state = np.ptp(q_pi, axis=1)
            train = fs.training_batch(mdp, policy, mu_state, rng)
            seed_parts = [fs.SEED, 9001, int(round(mixing * 100)), task_index]
            cert = certification_batch(mdp, policy, mu_state, seed_parts)

            routes: dict[str, Any] = {}
            for route in PRIMARY:
                q_hat = np.asarray(
                    fs.run_route(route, policy, train)["q_hat"], dtype=np.float64
                ).reshape(fs.N_STATES, fs.N_ACTIONS)
                realized = float(np.max(np.abs(q_hat - q_pi)))

                frozen = vc.variance_adaptive_certificate(q_hat, policy, cert)
                bern = bc.bernstein_certificate(q_hat, policy, cert)
                empb = bc.empirical_bernstein_certificate(q_hat, policy, cert)

                arms = {
                    "frozen": frozen,
                    "bernstein": bern,
                    "empirical_bernstein": empb,
                    # Counterfactual floor, NOT a certificate: the frozen
                    # construction with the envelope term deleted. Included so H6
                    # ("is deleting the envelope a bigger lever than correcting the
                    # inequality?") is answered by a number. Excluded from the
                    # coverage audit, which applies only to sound arms.
                    "counterfactual_no_envelope": bc.counterfactual_no_envelope(
                        q_hat, policy, cert
                    ),
                }
                unsound = {"counterfactual_no_envelope"}
                if multiplier > 1:
                    big = certification_batch(
                        mdp,
                        policy,
                        mu_state,
                        seed_parts,
                        chains=CERT_CHAINS * multiplier,
                    )
                    arms[f"empirical_bernstein_x{multiplier}"] = (
                        bc.empirical_bernstein_certificate(q_hat, policy, big)
                    )
                    del big

                summary: dict[str, Any] = {}
                for name, certificate in arms.items():
                    entry = decision_of(certificate, policy, q_hat)
                    e_q = certificate.get("e_q")
                    entry.update(
                        {
                            "e_q": e_q,
                            "epsilon_res": certificate.get("epsilon_res"),
                            "status_certificate": certificate["status"],
                            "certificate_reasons": list(
                                certificate.get("failure_reasons", [])
                            ),
                            "sigma_min_over_e_q": (
                                None
                                if e_q in (None, 0.0)
                                else float(np.min(sigma_by_state)) / float(e_q)
                            ),
                            "covers_realized": (
                                None if e_q is None else bool(float(e_q) >= realized)
                            ),
                            "sound": name not in unsound,
                        }
                    )
                    summary[name] = entry

                routes[route] = {
                    "q_hat": q_hat.tolist(),
                    "arms": summary,
                    "oracle_audit": {
                        "purpose": (
                            "truth-based diagnostic only; never a certificate input"
                        ),
                        "q_pi": q_pi.tolist(),
                        "realized_q_sup_error": realized,
                        "sigma_min": float(np.min(sigma_by_state)),
                        "sigma_by_state": sigma_by_state.tolist(),
                        "min_state_action_gap_true": float(
                            np.min(
                                np.sort(q_pi, axis=1)[:, -1]
                                - np.sort(q_pi, axis=1)[:, -2]
                            )
                        ),
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
            emitted = {
                name: sum(
                    1
                    for route in PRIMARY
                    if routes[route]["arms"][name]["emitted"]
                )
                for name in routes[PRIMARY[0]]["arms"]
            }
            print(f"  mixing={mixing} task={task_index:>2} {emitted}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "task_id": TASK_ID,
        "label": args.label,
        "record_count": len(records),
        "sample_multiplier": multiplier,
        "cert_chains": CERT_CHAINS,
        "cert_chain_length": CERT_CHAIN_LENGTH,
        "records": records,
    }
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
                "sample_multiplier": multiplier,
                "cert_chains": CERT_CHAINS,
                "cert_chain_length": CERT_CHAIN_LENGTH,
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
                "new_module_hash": sha256(
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
    print(f"{TASK_ID}: {len(records)} records written to {args.output_dir}")


if __name__ == "__main__":
    main()
