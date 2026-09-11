"""Environment families for FP-KERN-001 kernel generalization.

Two frozen families share the existing sticky-mixing and reward-gap
conventions:

1. ``current_unstructured``: a narrow wrapper around the unchanged
   ``evaluate_fixed_policy_q_routes.make_mdp``/``make_policy`` construction.
2. ``hidden_cluster``: two balanced latent clusters of states with frozen
   0.90/0.10 prototype/independent convex structure.

Cluster labels, prototypes, and independent perturbations are returned in a
separate hidden mapping; they are oracle-audit fields only and are never part
of the observable MDP dictionary consumed by estimators.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluate_fixed_policy_q_routes import make_mdp, make_policy


STRUCTURE_WEIGHT = 0.90
INDEPENDENT_WEIGHT = 0.10


def _validate_mdp(mdp: dict[str, Any], reward_bound: float) -> None:
    transition = np.asarray(mdp["P"], dtype=np.float64)
    reward = np.asarray(mdp["R"], dtype=np.float64)
    if not np.all(np.isfinite(transition)) or np.any(transition < 0.0):
        raise ValueError("transition rows must be finite and nonnegative")
    if not np.allclose(transition.sum(axis=2), 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("transition rows must lie on the simplex")
    if not np.all(np.isfinite(reward)):
        raise ValueError("rewards must be finite")
    if float(np.max(np.abs(reward))) > reward_bound + 1e-9:
        raise ValueError("rewards exceed the declared bound")
    p0 = np.asarray(mdp["p0"], dtype=np.float64)
    if not np.allclose(p0.sum(), 1.0, rtol=0.0, atol=1e-9) or np.any(p0 < 0.0):
        raise ValueError("p0 must lie on the simplex")


def make_current_mdp(
    n_states: int,
    n_actions: int,
    gamma: float,
    mixing: float,
    gap_bonus: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Unchanged current unstructured family (sticky mixing + gap bonus)."""
    if not 0.0 <= mixing <= 1.0:
        raise ValueError("mixing must lie in [0, 1]")
    if gap_bonus < 0.0:
        raise ValueError("gap_bonus must be nonnegative")
    mdp = make_mdp(n_states, n_actions, gamma, mixing, gap_bonus, rng)
    p0 = np.asarray(mdp["p0"], dtype=np.float64)
    mdp["p0"] = p0 / p0.sum()
    _validate_mdp(mdp, 1.0 + gap_bonus)
    return mdp


def make_hidden_cluster_mdp(
    n_states: int,
    n_actions: int,
    gamma: float,
    mixing: float,
    gap_bonus: float,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hidden two-cluster positive control with frozen 0.90/0.10 structure.

    Returns (observable_mdp, hidden). The observable MDP has exactly the
    sample_mdp key set; the hidden mapping carries cluster labels,
    prototypes, and independent perturbations for the oracle audit only.
    """
    if n_states % 2 != 0 or n_states < 4:
        raise ValueError("hidden-cluster family needs an even n_states >= 4")
    if not 0.0 <= mixing <= 1.0:
        raise ValueError("mixing must lie in [0, 1]")
    if gap_bonus < 0.0:
        raise ValueError("gap_bonus must be nonnegative")

    labels = np.repeat(np.arange(2), n_states // 2).astype(np.int64)
    rng.shuffle(labels)
    p_proto = rng.dirichlet(np.ones(n_states), size=(2, n_actions))
    r_proto = rng.uniform(-1.0, 1.0, size=(2, n_actions, n_states))
    p_independent = rng.dirichlet(np.ones(n_states), size=(n_states, n_actions))
    r_independent = rng.uniform(-1.0, 1.0, size=(n_states, n_actions, n_states))

    p_struct = (
        STRUCTURE_WEIGHT * p_proto[labels]
        + INDEPENDENT_WEIGHT * p_independent
    )
    p_struct /= p_struct.sum(axis=2, keepdims=True)
    sticky = np.zeros((n_states, n_actions, n_states), dtype=np.float64)
    for state in range(n_states):
        sticky[state, :, state] = 1.0
    transition = (1.0 - mixing) * sticky + mixing * p_struct

    reward = (
        STRUCTURE_WEIGHT * r_proto[labels]
        + INDEPENDENT_WEIGHT * r_independent
    )
    reward[:, 0, :] += gap_bonus
    p0 = rng.dirichlet(np.ones(n_states))

    mdp: dict[str, Any] = {
        "nS": n_states,
        "nA": n_actions,
        "gamma": gamma,
        "P": transition,
        "R": reward,
        "p0": p0,
    }
    hidden = {
        "cluster_labels": labels,
        "p_proto": p_proto,
        "r_proto": r_proto,
        "p_independent": p_independent,
        "r_independent": r_independent,
        "structure_weight": STRUCTURE_WEIGHT,
        "independent_weight": INDEPENDENT_WEIGHT,
    }
    _validate_mdp(mdp, 1.0 + gap_bonus)
    return mdp, hidden


__all__ = [
    "INDEPENDENT_WEIGHT",
    "STRUCTURE_WEIGHT",
    "make_current_mdp",
    "make_hidden_cluster_mdp",
    "make_policy",
]
