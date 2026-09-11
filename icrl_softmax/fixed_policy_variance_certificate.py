"""FP-SCALE-002: variance-adaptive residual certificate.

This module replaces exactly one ingredient of the inherited, verified
FP-ESARSA-001 certificate: the worst-case residual envelope ``2B`` that supplies
its sub-Gaussian parameter. Everything else -- the routes, the residual
definition, the decision rule, the ordered abstention reasons, and the protocol
-- is inherited and untouched.

The replacement is a two-half construction with explicit Hoeffding constants and
no fitted, asserted, or tuned scaling factor:

  For each pair x, split its certification items into disjoint halves A and B.

  (1) Second-moment bound on A. With ``Z_i = Y_i^2`` in ``[0, (2B)^2]``,
      Hoeffding gives, at confidence ``1 - delta_A``,

          E[Y^2 | x] <= mean_A(Y^2) + 2 B^2 sqrt(2 log(1/delta_A) / N_A) =: V_x.

      Bounding the second moment rather than the variance keeps the statement
      valid without assuming the residual mean vanishes, which it does not when
      ``Qhat != Q^pi``.

  (2) Mean bound on B. With ``s_x = sqrt(V_x) >= sqrt(Var(Y))``, Hoeffding's
      lemma bounds the half-B mean deviation, at confidence ``1 - delta_B``, by

          r_x = 2 s_x sqrt(log(1/delta_B) / N_B).

  Taking ``delta_A = delta_B = delta / (2d)`` and union-bounding over the d
  pairs makes the whole event simultaneous at risk ``delta``.

Because the halves are a deterministic function of the items, replacing ``N_A``
and ``N_B`` by their realized sizes is valid.

Oracle quantities are never accepted: the certificate receives only ``q_hat``,
the frozen policy, held-out transition fields, ``R_star``, ``gamma``, and
``delta``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

N_STATES = 4
N_ACTIONS = 3
D = N_STATES * N_ACTIONS
GAMMA = 0.70
R_STAR = 1.5
ENVELOPE = 2.0 * (R_STAR / (1.0 - GAMMA))
DELTA = 0.05
MIN_HALF_COUNT = 5000
MIN_CERT_COUNT = 2 * MIN_HALF_COUNT

# Frozen ordered non-emission reasons, inherited verbatim from FP-ESARSA-001.
REASON_ORDER = (
    "algorithm_mode_mismatch",
    "duplicate_q_memory",
    "divergence_guard_triggered",
    "heldout_pair_support_missing",
    "mixture_inversion_unbracketed",
    "mixture_inversion_not_converged",
    "mixture_root_not_conservative",
    "numerical_nonfinite",
    "policy_invalid",
    "improvement_lcb_nonpositive",
    "policy_unchanged",
)


def residuals_for(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
) -> np.ndarray:
    """Held-out Bellman residuals ``Y_t(Qhat)`` for the frozen policy."""
    del reward_bound, n_states  # carried for signature symmetry; unused here
    q_hat = np.asarray(q_hat, dtype=np.float64)
    policy = np.asarray(policy, dtype=np.float64)
    states = np.asarray(batch["states"], dtype=np.int64)
    actions = np.asarray(batch["actions"], dtype=np.int64)
    rewards = np.asarray(batch["rewards"], dtype=np.float64)
    next_states = np.asarray(batch["next_states"], dtype=np.int64)
    successor = (policy[next_states] * q_hat[next_states]).sum(axis=1)
    return rewards + float(gamma) * successor - q_hat[states, actions]


def split_pairs(
    batch: dict[str, Any], *, n_states: int = N_STATES, n_actions: int = N_ACTIONS
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Deterministic disjoint halves of each pair's certification items.

    Members are visited in ascending index order, so the split depends only on
    the item ordering and never on any value of the data.
    """
    flat = np.asarray(batch["states"], dtype=np.int64) * int(n_actions) + np.asarray(
        batch["actions"], dtype=np.int64
    )
    split: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pair in range(int(n_states) * int(n_actions)):
        members = np.flatnonzero(flat == pair)
        half = members.size // 2
        split[pair] = (members[:half], members[half:])
    return split


def variance_adaptive_certificate(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
    delta: float = DELTA,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    min_half_count: int = MIN_HALF_COUNT,
) -> dict[str, Any]:
    """Certify ``||Qhat - Q^pi||_infinity`` with an estimated sub-Gaussian scale."""
    n_groups = int(n_states) * int(n_actions)
    value_bound = float(reward_bound) / (1.0 - float(gamma))
    envelope = 2.0 * value_bound
    q_hat = np.asarray(q_hat, dtype=np.float64)
    policy = np.asarray(policy, dtype=np.float64)

    reasons: list[str] = []
    if q_hat.shape != (int(n_states), int(n_actions)):
        raise ValueError("q_hat must match the declared dimensions")
    if not np.all(np.isfinite(q_hat)):
        reasons.append("numerical_nonfinite")
    elif float(np.max(np.abs(q_hat))) > value_bound:
        reasons.append("divergence_guard_triggered")

    residuals = residuals_for(
        q_hat,
        policy,
        batch,
        reward_bound=reward_bound,
        gamma=gamma,
        n_states=n_states,
        n_actions=n_actions,
    )
    split = split_pairs(batch, n_states=n_states, n_actions=n_actions)

    for pair in range(n_groups):
        first, second = split[pair]
        if first.size < min_half_count or second.size < min_half_count:
            reasons.append("heldout_pair_support_missing")
            break

    delta_each = float(delta) / (2.0 * n_groups)
    scales = np.zeros(n_groups, dtype=np.float64)
    radii = np.zeros(n_groups, dtype=np.float64)
    means = np.zeros(n_groups, dtype=np.float64)

    if not reasons:
        log_a = math.log(1.0 / delta_each)
        log_b = math.log(1.0 / delta_each)
        for pair in range(n_groups):
            first, second = split[pair]
            n_a = int(first.size)
            n_b = int(second.size)
            # Hoeffding on Z = Y^2 in [0, (2B)^2]: range/2 = (2B)^2 / 2.
            second_moment = float(np.mean(residuals[first] ** 2))
            slack = (envelope**2 / 2.0) * math.sqrt(2.0 * log_a / n_a)
            v_x = second_moment + slack
            s_x = math.sqrt(max(v_x, 0.0))
            scales[pair] = s_x
            # Hoeffding's lemma on the bounded residuals of half B.
            radii[pair] = 2.0 * s_x * math.sqrt(log_b / n_b)
            means[pair] = float(np.mean(residuals[second]))

    if reasons:
        ordered = sorted(set(reasons), key=REASON_ORDER.index)
        return {
            "status": "not_certified",
            "failure_reasons": ordered,
            "e_q": None,
            "epsilon_res": None,
            "residual_means": means,
            "radii": radii,
            "scales": scales,
            "n_groups": n_groups,
            "delta_each": delta_each,
        }

    epsilon_res = float(np.max(np.abs(means) + radii))
    return {
        "status": "certificate_emitted",
        "failure_reasons": [],
        "e_q": epsilon_res / (1.0 - float(gamma)),
        "epsilon_res": epsilon_res,
        "residual_means": means,
        "radii": radii,
        "scales": scales,
        "n_groups": n_groups,
        "delta_each": delta_each,
    }


def provenance_probe(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
    delta: float = DELTA,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    min_half_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the scales and means from halves A and B in isolation.

    Exists so the verifier can prove the scale never sees the half it
    certifies. Returns ``(scales_from_A_only, means_from_B_only)``.
    """
    n_groups = int(n_states) * int(n_actions)
    value_bound = float(reward_bound) / (1.0 - float(gamma))
    envelope = 2.0 * value_bound
    residuals = residuals_for(
        q_hat,
        policy,
        batch,
        reward_bound=reward_bound,
        gamma=gamma,
        n_states=n_states,
        n_actions=n_actions,
    )
    split = split_pairs(batch, n_states=n_states, n_actions=n_actions)
    delta_each = float(delta) / (2.0 * n_groups)
    scales = np.zeros(n_groups, dtype=np.float64)
    means = np.zeros(n_groups, dtype=np.float64)
    for pair in range(n_groups):
        first, second = split[pair]
        if first.size == 0 or second.size == 0:
            continue
        if first.size >= min_half_count:
            second_moment = float(np.mean(residuals[first] ** 2))
            slack = (envelope**2 / 2.0) * math.sqrt(
                2.0 * math.log(1.0 / delta_each) / int(first.size)
            )
            scales[pair] = math.sqrt(max(second_moment + slack, 0.0))
        means[pair] = float(np.mean(residuals[second]))
    return scales, means


def envelope_control_certificate(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
    delta: float = DELTA,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
) -> dict[str, Any]:
    """The frozen ``2B`` envelope certificate on the same counts, for H5.

    This is the worst-case-envelope analogue of the variance-adaptive
    certificate with the identical per-pair split sizes and risk allocation, so
    the two certified errors are directly comparable. It is a control only and
    makes no claim of being the sealed inherited implementation.
    """
    n_groups = int(n_states) * int(n_actions)
    value_bound = float(reward_bound) / (1.0 - float(gamma))
    envelope = 2.0 * value_bound
    residuals = residuals_for(
        q_hat,
        policy,
        batch,
        reward_bound=reward_bound,
        gamma=gamma,
        n_states=n_states,
        n_actions=n_actions,
    )
    split = split_pairs(batch, n_states=n_states, n_actions=n_actions)
    delta_each = float(delta) / (2.0 * n_groups)
    radii = np.zeros(n_groups, dtype=np.float64)
    means = np.zeros(n_groups, dtype=np.float64)
    for pair in range(n_groups):
        first, second = split[pair]
        n_b = max(int(second.size), 1)
        # Hoeffding on the bounded half-B mean (range 2*envelope), per pair.
        radii[pair] = envelope * math.sqrt(2.0 * math.log(1.0 / delta_each) / n_b)
        if second.size:
            means[pair] = float(np.mean(residuals[second]))
    epsilon_res = float(np.max(np.abs(means) + radii))
    return {
        "status": "certificate_emitted",
        "failure_reasons": [],
        "e_q": epsilon_res / (1.0 - float(gamma)),
        "epsilon_res": epsilon_res,
        "residual_means": means,
        "radii": radii,
        "n_groups": n_groups,
        "delta_each": delta_each,
    }
