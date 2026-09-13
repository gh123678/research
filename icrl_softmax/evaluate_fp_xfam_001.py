"""FP-XFAM-001: does the corrected protocol + per-state rule generalise across
MDP families?

Pre-registered in ``docs/research_tasks/FP-XFAM-001.md`` BEFORE any run.

Families (both never used in any earlier task):
- F1: 6 states x 4 actions, GAP_BONUS=0.5 (reward bound 1.5, same envelope as
  the original family), mixing in {0.08, 0.5}, task_index 200..223.
- F2: 4 states x 3 actions, GAP_BONUS=1.0 (reward bound 2.0 -> envelope
  E = 2 + 0.7*(2/0.3) + 2/0.3 = 13.33), mixing in {0.2, 0.7}, task_index
  300..323.

Protocol: identical to FP-CERTFIX-001 (MP certificate, first-n fixed counts,
fresh batch per step, delta_k = 0.05/12, K=12, frozen eta grid). Arms:
``conj`` (conjunctive gate) and ``perstate`` (per-state first passage), both at
n = 16,384 per pair per step. numpy producers only -- the network
correspondence was only ever established on the original family.

Everything dimension-dependent is local to this module; sealed files are
untouched.
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
import fixed_policy_mp_certificate as mc  # noqa: E402
from evaluate_fixed_policy_q_routes import make_mdp, make_policy, policy_quantities  # noqa: E402
from fp_certfix_first_n import first_n_batch, step_seed_parts  # noqa: E402
from mdps import rollout  # noqa: E402

TASK_ID = "FP-XFAM-001"
PRIMARY = ("expected_exact", "expected_finite")
ARMS = ("conj", "perstate")
TASK_SALT = 66337
DELTA_TOTAL = 0.05
K_STEPS = 12
N_PER_PAIR = 16384
CHAIN_LENGTH = 64
PI_MIN = 0.15
ETA_GRID = fs.ETA_CANDIDATES
SEED = fs.SEED

FAMILIES = {
    "f1": {
        "n_states": 6,
        "n_actions": 4,
        "gamma": 0.70,
        "gap_bonus": 0.5,
        "reward_bound": 1.5,
        "mixings": (0.08, 0.5),
        "task_indices": tuple(range(200, 224)),
        "chains": 131072,
    },
    "f2": {
        "n_states": 4,
        "n_actions": 3,
        "gamma": 0.70,
        "gap_bonus": 1.0,
        "reward_bound": 2.0,
        "mixings": (0.2, 0.7),
        "task_indices": tuple(range(300, 324)),
        "chains": 65536,
    },
}


# --------------------------------------------------------------------------- #
# Generic plumbing (dimension-driven; sealed modules stay untouched)
# --------------------------------------------------------------------------- #


def build_family_task(fam: dict, mixing: float, task_index: int):
    rng = np.random.default_rng([SEED, int(round(mixing * 100)), int(task_index)])
    mdp = make_mdp(
        fam["n_states"], fam["n_actions"], fam["gamma"], float(mixing), fam["gap_bonus"], rng
    )
    policy = make_policy(fam["n_states"], fam["n_actions"], PI_MIN, rng)
    return mdp, policy, rng


def training_batch(mdp, policy, mu_state, rng, length=65536):
    start = int(rng.choice(policy.shape[0], p=mu_state))
    sampled = rollout(mdp, policy, start, length, rng)
    all_actions = np.asarray(sampled[1])
    states = np.asarray(sampled[0])[:length]
    actions = all_actions[:length]
    rewards = np.asarray(sampled[2], dtype=np.float64)[1 : length + 1]
    next_states = np.asarray(sampled[0])[1 : length + 1]
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "next_actions": all_actions[1 : length + 1],
    }


def run_route(route: str, fam: dict, policy: np.ndarray, train: dict[str, Any]) -> np.ndarray:
    common = {
        "q0": np.zeros((fam["n_states"], fam["n_actions"]), dtype=np.float64),
        "policy": policy,
        "states": train["states"],
        "actions": train["actions"],
        "rewards": train["rewards"],
        "next_states": train["next_states"],
        "alpha": fs.ALPHA,
        "gamma": fam["gamma"],
        "layers": fs.LAYERS,
        "value_bound": fam["reward_bound"] / (1.0 - fam["gamma"]),
    }
    if route == "expected_exact":
        out = es.run_expected_exact(**common)
    elif route == "expected_finite":
        out = es.run_expected_finite(**common, zeta=fs.ZETA, xi=fs.XI, tau=fs.TAU)
    else:
        raise ValueError(route)
    return np.asarray(out["q_hat"], dtype=np.float64).reshape(
        fam["n_states"], fam["n_actions"]
    )


def vectorised_batch_generic(mdp, policy, mu_state, seed_parts, chains, fam, chain_length=CHAIN_LENGTH):
    """Same law as fp_sample_vectorised_batch.vectorised_batch, dimension-generic."""
    n_states, n_actions = fam["n_states"], fam["n_actions"]
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    rng = np.random.default_rng(seed_parts)
    policy_cdf = np.cumsum(policy, axis=1)
    policy_cdf[:, -1] = 1.0
    total = chains * chain_length
    states = np.empty(total, dtype=np.int32)
    actions = np.empty(total, dtype=np.int32)
    rewards = np.empty(total, dtype=np.float64)
    next_states = np.empty(total, dtype=np.int32)
    done = 0
    chunk = 32768
    while done < chains:
        size = min(chunk, chains - done)
        current = rng.choice(n_states, size=size, p=mu_state).astype(np.int32)
        for step in range(chain_length):
            uniform = rng.random(size)
            chosen = (uniform[:, None] > policy_cdf[current]).sum(axis=1)
            np.clip(chosen, 0, n_actions - 1, out=chosen)
            following_uniform = rng.random(size)
            rows = transition[current, chosen]
            following = (following_uniform[:, None] > np.cumsum(rows, axis=1)).sum(axis=1)
            np.clip(following, 0, n_states - 1, out=following)
            following = following.astype(np.int32)
            base = (done + np.arange(size)) * chain_length + step
            states[base] = current
            actions[base] = chosen.astype(np.int32)
            rewards[base] = reward[current, chosen, following]
            next_states[base] = following
            current = following
        done += size
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "next_actions": actions,
    }


def value_of(policy: np.ndarray, mdp: Any, gamma: float) -> np.ndarray:
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    r_sa = np.einsum("sap,sap->sa", transition, reward)
    p_pi = np.einsum("sa,sap->sp", policy, transition)
    r_pi = np.einsum("sa,sa->s", policy, r_sa)
    return np.linalg.solve(np.eye(policy.shape[0]) - gamma * p_pi, r_pi)


def optimal_values_generic(mdp: Any, gamma: float) -> np.ndarray:
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    n_states, n_actions = reward.shape[0], reward.shape[1]
    r_sa = np.einsum("sap,sap->sa", transition, reward)
    greedy = np.full((n_states, n_actions), 1.0 / n_actions)
    for _ in range(1000):
        q = r_sa + gamma * np.einsum("sap,p->sa", transition, value_of(greedy, mdp, gamma))
        best = np.argmax(q, axis=1)
        new = np.zeros_like(greedy)
        new[np.arange(n_states), best] = 1.0
        if np.array_equal(new, greedy):
            break
        greedy = new
    return value_of(greedy, mdp, gamma)


def conjunctive_decision(policy, q_hat, e_q):
    for eta in ETA_GRID:
        cand = es.relative_softmax_candidate(policy, q_hat, eta)
        delta_pi = cand - policy
        i_hat = (delta_pi * q_hat).sum(axis=1)
        lb = i_hat - e_q * np.abs(delta_pi).sum(axis=1)
        if float(lb.min()) > 0.0:
            return True, cand, float(eta), lb
    return False, policy.copy(), None, np.zeros(policy.shape[0])


def perstate_decision(policy, q_hat, e_q):
    new_policy = policy.copy()
    lb_by_state = np.zeros(policy.shape[0])
    updated = 0
    for s in range(policy.shape[0]):
        for eta in ETA_GRID:
            cand = es.relative_softmax_candidate(policy, q_hat, eta)
            delta_row = cand[s] - policy[s]
            lb = float((delta_row * q_hat[s]).sum()) - e_q * float(np.abs(delta_row).sum())
            if lb > 0.0:
                new_policy[s] = cand[s]
                lb_by_state[s] = lb
                updated += 1
                break
    return updated > 0, new_policy, updated, lb_by_state


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


def run_family(fam_name: str, args) -> dict[str, Any]:
    fam = FAMILIES[fam_name]
    n_states, n_actions = fam["n_states"], fam["n_actions"]
    delta_step = DELTA_TOTAL / K_STEPS
    records: list[dict[str, Any]] = []
    items_drawn_total = 0
    task_indices = fam["task_indices"][: args.limit] if args.limit else fam["task_indices"]
    for mixing in fam["mixings"]:
        for task_index in task_indices:
            mdp, behaviour, rng = build_family_task(fam, mixing, task_index)
            exact0 = policy_quantities(mdp, behaviour)
            mu_state = np.asarray(exact0["mu_state"], dtype=np.float64)
            v_star = optimal_values_generic(mdp, fam["gamma"])
            train = training_batch(mdp, behaviour, mu_state, rng)

            routes: dict[str, Any] = {}
            for route in PRIMARY:
                step_batches: list[Any] = []
                arms_out: dict[str, Any] = {}
                for arm in ARMS:
                    current = behaviour.copy()
                    q_ref = np.asarray(exact0["q_pi"], dtype=np.float64).copy()
                    v_chain = [np.asarray(exact0["v_pi"], dtype=np.float64).copy()]
                    steps: list[dict[str, Any]] = []
                    for step_index in range(1, K_STEPS + 1):
                        q_hat = run_route(route, fam, current, train)
                        if step_index > len(step_batches):
                            raw = vectorised_batch_generic(
                                mdp,
                                behaviour,
                                mu_state,
                                step_seed_parts(
                                    SEED, TASK_SALT, mixing, task_index, step_index
                                ),
                                fam["chains"],
                                fam,
                            )
                            items_drawn_total += fam["chains"] * CHAIN_LENGTH
                            step_batches.append(raw)
                        raw = step_batches[step_index - 1]
                        reduced, counts = first_n_batch(
                            raw, N_PER_PAIR, n_states=n_states, n_actions=n_actions
                        )
                        realized = float(np.max(np.abs(q_hat - q_ref)))
                        if reduced is None:
                            e_q = None
                            emitted, nxt, eta_info, lb = False, current.copy(), None, np.zeros(n_states)
                            reasons = ["heldout_pair_support_missing"]
                        else:
                            cert = mc.mp_certificate(
                                q_hat,
                                current,
                                reduced,
                                n_per_pair=N_PER_PAIR,
                                delta_step=delta_step,
                                n_states=n_states,
                                n_actions=n_actions,
                                reward_bound=fam["reward_bound"],
                                gamma=fam["gamma"],
                            )
                            e_q = cert.get("e_q")
                            if e_q is None:
                                emitted, nxt, eta_info, lb = False, current.copy(), None, np.zeros(n_states)
                                reasons = list(cert.get("failure_reasons", []))
                            elif arm == "conj":
                                emitted, nxt, eta_info, lb = conjunctive_decision(current, q_hat, e_q)
                                reasons = [] if emitted else ["improvement_lcb_nonpositive"]
                            else:
                                emitted, nxt, eta_info, lb = perstate_decision(current, q_hat, e_q)
                                reasons = [] if emitted else ["improvement_lcb_nonpositive"]
                        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
                        entry: dict[str, Any] = {
                            "step": step_index,
                            "emitted": bool(emitted),
                            "eta_info": eta_info if not isinstance(eta_info, np.ndarray) else eta_info.tolist(),
                            "e_q": e_q,
                            "min_lb": float(lb.min()) if lb.size else None,
                            "min_pair_count_observed": int(counts.min()),
                            "ordered_reasons": reasons,
                            "oracle_audit": {
                                "purpose": "truth-based audit only; never a certificate input",
                                "realized_q_sup_error_vs_current_target_pi": realized,
                                "certificate_violation": bool(
                                    e_q is not None and float(e_q) < realized
                                ),
                            },
                        }
                        if emitted:
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
                            current = np.asarray(nxt, dtype=np.float64)
                        steps.append(entry)
                        if not emitted:
                            break
                    v_final = v_chain[-1]
                    init_gap = float(np.sum(v_star - v_chain[0]))
                    arms_out[arm] = {
                        "steps": steps,
                        "emitted_steps": sum(1 for s in steps if s["emitted"]),
                        "initial_v_sum": float(np.sum(v_chain[0])),
                        "final_v_sum": float(np.sum(v_final)),
                        "initial_suboptimality": init_gap,
                        "fraction_gap_closed": (
                            float(np.sum(v_final - v_chain[0])) / init_gap
                            if init_gap > 0
                            else 1.0
                        ),
                    }
                routes[route] = arms_out
            records.append(
                {
                    "task_id": TASK_ID,
                    "family": fam_name,
                    "mixing": float(mixing),
                    "task_index": int(task_index),
                    "routes": routes,
                }
            )
    return {
        "task_id": TASK_ID,
        "family": fam_name,
        "family_spec": {k: v for k, v in fam.items() if k != "task_indices"}
        | {"task_indices": list(task_indices)},
        "label": args.label,
        "arms": list(ARMS),
        "routes": list(PRIMARY),
        "n_per_pair": N_PER_PAIR,
        "delta_total": DELTA_TOTAL,
        "max_steps": K_STEPS,
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
    parser.add_argument("--family", choices=["f1", "f2"], required=True)
    parser.add_argument("--limit", type=int, default=0, help="smoke: limit tasks per mixing")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    out = run_family(args.family, args)
    out["started_utc"] = started.isoformat()
    out["finished_utc"] = datetime.now(timezone.utc).isoformat()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task_results.json").write_text(
        json.dumps(strict_ready(out), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "label": args.label,
                "family": args.family,
                "file_hashes": {
                    name: sha256(PROJECT / name)
                    for name in (
                        "fixed_policy_mp_certificate.py",
                        "fp_certfix_first_n.py",
                        "evaluate_fp_xfam_001.py",
                    )
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "environment.json").write_text(
        json.dumps(
            {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "family": args.family,
        "records": out["record_count"],
        "items_drawn_total": out["items_drawn_total"],
        "per_arm": {
            arm: {
                "emitted_steps": sum(
                    r["routes"][route][arm]["emitted_steps"]
                    for r in out["records"]
                    for route in PRIMARY
                ),
                "records_emitting_step1": sum(
                    1
                    for r in out["records"]
                    for route in PRIMARY
                    if r["routes"][route][arm]["steps"]
                    and r["routes"][route][arm]["steps"][0]["emitted"]
                ),
                "mean_total_gain": float(
                    np.mean(
                        [
                            r["routes"][route][arm]["final_v_sum"]
                            - r["routes"][route][arm]["initial_v_sum"]
                            for r in out["records"]
                            for route in PRIMARY
                        ]
                    )
                ),
                "mean_fraction_gap_closed": float(
                    np.mean(
                        [
                            r["routes"][route][arm]["fraction_gap_closed"]
                            for r in out["records"]
                            for route in PRIMARY
                        ]
                    )
                ),
                "certificate_violations": sum(
                    1
                    for r in out["records"]
                    for route in PRIMARY
                    for s in r["routes"][route][arm]["steps"]
                    if s["oracle_audit"]["certificate_violation"]
                ),
                "componentwise_degrading_steps": sum(
                    1
                    for r in out["records"]
                    for route in PRIMARY
                    for s in r["routes"][route][arm]["steps"]
                    if s["emitted"]
                    and not s["oracle_audit"].get("componentwise_nondegrading", True)
                ),
            }
            for arm in ARMS
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
