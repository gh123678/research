"""FP-SCALE-002 seed-sensitivity check (read-only, does NOT touch the formal corpus).

The frozen formal run used one seed schedule and produced 22/48 primary
emissions. This check asks whether that rate is a property of the construction
or an accident of the batch draws, by re-running the identical pipeline with
different training-batch and certification-batch seeds.

Scope discipline:
  - the frozen formal bundle in results/FP-SCALE-002/claude/formal/ is read-only
    and is never rewritten; this script writes nothing;
  - the task seed schedule, matrix shape, dimensions and every frozen constant
    are unchanged: only the two batch RNG streams move;
  - this is robustness evidence for the reported result. It is not a rerun of
    the frozen formal matrix and cannot replace or amend it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent / "icrl_softmax"
sys.path.insert(0, str(PROJECT))

import fixed_policy_expected_sarsa_scaled as fs  # noqa: E402
import fixed_policy_variance_certificate as vc  # noqa: E402
from evaluate_fixed_policy_q_routes import policy_quantities  # noqa: E402

MIXINGS = (0.08, 0.5)
TASKS = 6
CHAIN_LENGTH = 64
CHAINS = 16384
TRAIN_LEN = 65536
PRIMARY = ("expected_exact", "expected_finite")


def certification_batch(mdp, policy, mu_state, seed_parts, chains, chain_length, n_states, n_actions):
    rng = np.random.default_rng(seed_parts)
    total = chains * chain_length
    starts = rng.choice(n_states, size=chains, p=mu_state)
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
            action = int(rng.choice(n_actions, p=policy[state]))
            following = int(rng.choice(n_states, p=transition[state, action]))
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


def run_variant(train_tag: int, cert_tag: int, tasks: int) -> dict:
    attempted = emitted = nondegrading = strict = violations = 0
    ratios: list[float] = []
    for mixing in MIXINGS:
        for task in range(tasks):
            mdp, policy, rng = fs.build_task(task_index=task, mixing=mixing)
            exact = policy_quantities(mdp, policy)
            mu_state = np.asarray(exact["mu_state"], dtype=np.float64)
            q_pi = np.asarray(exact["q_pi"], dtype=np.float64)
            v_pi = np.asarray(exact["v_pi"], dtype=np.float64)

            # Training stream: tag only this draw, nothing else.
            train_rng = np.random.default_rng(
                [fs.SEED, train_tag, int(round(mixing * 100)), task]
            )
            train = fs.training_batch(mdp, policy, mu_state, train_rng)

            cert = certification_batch(
                mdp,
                policy,
                mu_state,
                [fs.SEED, cert_tag, int(round(mixing * 100)), task],
                CHAINS,
                CHAIN_LENGTH,
                fs.N_STATES,
                fs.N_ACTIONS,
            )

            route_names = {
                "expected_exact": "variance_adaptive_exact",
                "expected_finite": "variance_adaptive_finite",
            }
            for source, label in route_names.items():
                result = fs.run_route(source, policy, train)
                q_hat = np.asarray(result["q_hat"], dtype=np.float64).reshape(
                    fs.N_STATES, fs.N_ACTIONS
                )
                certificate = vc.variance_adaptive_certificate(q_hat, policy, cert)
                control = vc.envelope_control_certificate(q_hat, policy, cert)
                attempted += 1
                if certificate["status"] != "certificate_emitted":
                    continue
                realized = float(np.max(np.abs(q_hat - q_pi)))
                if float(certificate["e_q"]) < realized:
                    violations += 1
                if control["e_q"] is not None and float(certificate["e_q"]) > 0:
                    ratios.append(float(control["e_q"]) / float(certificate["e_q"]))
                improvement = fs.improvement_for(policy, q_hat, certificate)
                if improvement["status"] != "safe_update_emitted":
                    continue
                emitted += 1
                new_policy = np.asarray(improvement["policy_plus"], dtype=np.float64)
                new_v = np.asarray(
                    policy_quantities(mdp, new_policy)["v_pi"], dtype=np.float64
                )
                delta_v = new_v - v_pi
                if float(np.min(delta_v)) >= -1e-12:
                    nondegrading += 1
                if float(np.sum(delta_v)) > 0.0:
                    strict += 1
    return {
        "attempted": attempted,
        "emitted": emitted,
        "nondegrading": nondegrading,
        "strict": strict,
        "violations": violations,
        "mean_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "min_ratio": float(np.min(ratios)) if ratios else float("nan"),
    }


def main() -> None:
    started = time.perf_counter()
    variants = [
        ("frozen schedule subset", 0, 9001),
        ("train seed +1000", 1000, 9001),
        ("cert seed +1000", 0, 10001),
        ("both +1000", 1000, 10001),
    ]
    print("FP-SCALE-002 seed sensitivity (read-only; formal corpus untouched)")
    print(f"tasks per mixing = {TASKS}, chains x length = {CHAINS} x {CHAIN_LENGTH}")
    print()
    header = (
        f"{'variant':>24} {'attempted':>9} {'emitted':>8} {'rate':>7} "
        f"{'nondeg':>7} {'strict':>7} {'viol':>5} {'minRatio':>9}"
    )
    print(header)
    print("-" * len(header))
    rates: list[float] = []
    for label, train_tag, cert_tag in variants:
        stats = run_variant(train_tag, cert_tag, TASKS)
        rate = stats["emitted"] / stats["attempted"] if stats["attempted"] else 0.0
        rates.append(rate)
        print(
            f"{label:>24} {stats['attempted']:>9} {stats['emitted']:>8} {rate:>7.3f} "
            f"{stats['nondegrading']:>7} {stats['strict']:>7} {stats['violations']:>5} "
            f"{stats['min_ratio']:>9.3f}"
        )
    print()
    print(f"emission-rate range across variants: {min(rates):.3f} .. {max(rates):.3f}")
    print(f"elapsed: {time.perf_counter() - started:.1f} s")
    print()
    print("This is robustness evidence only. It does not amend, replace or rerun")
    print("the frozen formal matrix, whose single execution stands as sealed.")


if __name__ == "__main__":
    main()
