"""MDP families for the FP-KERN-001 feasibility study."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluate_fixed_policy_q_routes import make_mdp


PROTOTYPE_WEIGHT = 0.90
INDEPENDENT_WEIGHT = 0.10


def make_hidden_cluster_mdp(
    n_states: int,
    n_actions: int,
    gamma: float,
    mixing: float,
    gap_bonus: float,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a two-cluster MDP while hiding structure from estimators."""
    if n_states != 6 or n_actions != 4:
        raise ValueError("FP-KERN-001 freezes n_states=6 and n_actions=4")
    if not np.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    if not np.isfinite(mixing) or not 0.0 < mixing <= 1.0:
        raise ValueError("mixing must lie in (0, 1]")
    if not np.isfinite(gap_bonus) or gap_bonus < 0.0:
        raise ValueError("gap_bonus must be finite and nonnegative")

    labels = np.repeat(np.arange(2, dtype=np.int64), n_states // 2)
    labels = labels[rng.permutation(n_states)]
    prototype_transition = rng.dirichlet(
        np.ones(n_states), size=(2, n_actions)
    ).astype(np.float64)
    independent_transition = rng.dirichlet(
        np.ones(n_states), size=(n_states, n_actions)
    ).astype(np.float64)
    structured = (
        PROTOTYPE_WEIGHT * prototype_transition[labels]
        + INDEPENDENT_WEIGHT * independent_transition
    )
    sticky = np.zeros_like(structured)
    for state in range(n_states):
        sticky[state, :, state] = 1.0
    transition = (1.0 - mixing) * sticky + mixing * structured
    transition /= transition.sum(axis=2, keepdims=True)

    prototype_reward = rng.uniform(
        -1.0, 1.0, size=(2, n_actions, n_states)
    ).astype(np.float64)
    independent_reward = rng.uniform(
        -1.0, 1.0, size=(n_states, n_actions, n_states)
    ).astype(np.float64)
    reward = (
        PROTOTYPE_WEIGHT * prototype_reward[labels]
        + INDEPENDENT_WEIGHT * independent_reward
    )
    reward[:, 0, :] += gap_bonus
    p0 = rng.dirichlet(np.ones(n_states)).astype(np.float64)
    mdp: dict[str, Any] = {
        "nS": n_states,
        "nA": n_actions,
        "gamma": float(gamma),
        "P": transition,
        "R": reward,
        "p0": p0,
    }
    audit = {
        "cluster_by_state": labels.tolist(),
        "prototype_weight": PROTOTYPE_WEIGHT,
        "independent_weight": INDEPENDENT_WEIGHT,
        "prototype_transition": prototype_transition.tolist(),
        "independent_transition": independent_transition.tolist(),
        "prototype_reward": prototype_reward.tolist(),
        "independent_reward": independent_reward.tolist(),
    }
    return mdp, audit


def make_environment(
    family: str,
    n_states: int,
    n_actions: int,
    gamma: float,
    mixing: float,
    gap_bonus: float,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if family == "current_unstructured":
        return (
            make_mdp(
                n_states,
                n_actions,
                gamma,
                mixing,
                gap_bonus,
                rng,
            ),
            {"family": family},
        )
    if family == "hidden_cluster":
        mdp, audit = make_hidden_cluster_mdp(
            n_states,
            n_actions,
            gamma,
            mixing,
            gap_bonus,
            rng,
        )
        audit["family"] = family
        return mdp, audit
    raise ValueError(f"unknown environment family: {family}")

