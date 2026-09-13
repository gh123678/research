"""FP-BOUND-001: sound tightenings of the residual certificate's scalar ``E_Q``.

ADDITIVE MODULE. It imports nothing sealed for its arithmetic and modifies no
sealed file; the frozen decision rule, the eta grid and ``delta_total`` are
untouched. Every function here returns the same scalar contract as
``fixed_policy_mp_certificate.mp_certificate_firstvisit`` so that
``fs.improvement_for`` can be called on it unchanged.

WHY THE FROZEN ``E_Q`` IS NOT TIGHT
-----------------------------------
``E_Q = max_x (|mean_x| + t_x) / (1 - gamma)`` with
``t_x = sqrt(2 V_x log(2/delta')/n_x) + (7/3) R log(2/delta')/(n_x - 1)``,
``R = 2E``, ``E = R* + gamma B + B``, ``B = R*/(1-gamma) = 5``, ``delta' =
delta_step/(2d)``. Three provable slack sources, each priced in the task sheet:

L1  THE ENVELOPE IS WORST-CASE OVER QHAT, BUT QHAT IS KNOWN.
    ``E = R* + gamma*B + B`` only assumes ``|Qhat| <= B``. Since ``Qhat`` is an
    input the certificate already holds, the true range of
    ``Y = r + gamma Vhat(s') - Qhat(s,a)`` is bounded, deterministically, by
    ``E_eff = R* + gamma ||Vhat||_inf + ||Qhat||_inf`` with
    ``Vhat = pi . Qhat``. No estimate, no confidence correction, no new premise:
    the same Maurer-Pontil theorem with a smaller TRUE range.

L2  THE IMPLEMENTATION NEVER SPENDS HALF ITS RISK BUDGET.
    Maurer-Pontil Theorem 4 is TWO-SIDED at level ``delta``. Allocating
    ``delta' = delta_step/(2d)`` and then reasoning about "two directions"
    therefore spends only ``d * delta' = delta_step/2``. ``delta' = delta_step/d``
    spends the budget exactly. ``log(2/delta')`` falls from 6.87 to 6.17.

L3  THE ``1/(1-gamma)`` STEP THROWS AWAY THE PER-PAIR STRUCTURE.
    ``Qhat - Q^pi = (I - gamma P^pi)^{-1} rho`` is an IDENTITY. With
    ``|rho_x| <= eps_x`` per pair, exact propagation gives

        w        = (I - gamma K)^{-1} epsbar,
        K(s,s')  = sum_a pi(a|s) P(s'|s,a),
        epsbar(s)= sum_a pi(a|s) eps(s,a),
        |u(s,a)| <= eps(s,a) + gamma * sum_s' P(s'|s,a) w(s'),

    whose max is ``<= max_x eps_x/(1-gamma)`` with equality only when ``eps`` is
    constant across pairs. The gain is exactly the non-uniformity of ``eps``
    (measured median max/min ratio: 4.40).

    L3 READS THE TRANSITION KERNEL. It is the same family as the pre-registered
    oracle-kernel arm and MUST be labelled as such everywhere: it corroborates,
    but cannot be used for any claim about learning from behavioural data alone.

Reference-only (NOT a deliverable, also needs the kernel): the exact support
range of ``Y_x`` from the MDP tables, ``support_range_certificate``, quoted to
show how much of the range lever L1 leaves on the table.
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
VALUE_BOUND = R_STAR / (1.0 - GAMMA)  # B = 5
ENVELOPE = R_STAR + GAMMA * VALUE_BOUND + VALUE_BOUND  # E = 10 (the frozen one)
Y_RANGE = 2.0 * ENVELOPE  # 20
MP_CONSTANT = 7.0 / 3.0

REASONS = ("divergence_guard_triggered", "heldout_pair_support_missing", "numerical_nonfinite")

LEVERS = ("frozen", "L1", "L12", "L123", "support_range")


def effective_envelope(
    q_hat: Any, policy: Any, *, reward_bound: float = R_STAR, gamma: float = GAMMA
) -> float:
    """``E_eff = R* + gamma*||Vhat||_inf + ||Qhat||_inf`` -- L1's range constant.

    A deterministic function of the KNOWN ``(q_hat, policy)``. Since
    ``|r| <= R*`` and ``|Vhat(s')| <= ||Vhat||_inf`` and ``|Qhat(s,a)| <=
    ||Qhat||_inf`` pointwise, ``|Y_x| <= E_eff`` for every pair, so the residual
    range is at most ``2 E_eff``. This is strictly sharper than the frozen
    ``E = R* + gamma*B + B`` whenever ``max(||Vhat||_inf, ||Qhat||_inf) < B``,
    which holds for every route in this project.
    """
    q = np.asarray(q_hat, dtype=np.float64)
    pi = np.asarray(policy, dtype=np.float64)
    v_hat = (pi * q).sum(axis=1)
    return float(reward_bound) + float(gamma) * float(
        np.max(np.abs(v_hat))
    ) + float(np.max(np.abs(q)))


def support_range(
    q_hat: Any,
    policy: Any,
    transition: Any,
    reward: Any,
    *,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Exact support range of ``Y_x`` from the MDP tables (REFERENCE ONLY).

    ``Y_x`` can only take the values ``R(s,a,s') + gamma Vhat(s') - Qhat(s,a)``
    for successors ``s'`` with ``P(s'|s,a) > 0``, so its exact range is a finite,
    computable constant -- no estimation and no probability cost. Quoted to price
    what L1 leaves on the table; needs the kernel, so not a deliverable.
    """
    q = np.asarray(q_hat, dtype=np.float64)
    pi = np.asarray(policy, dtype=np.float64)
    P = np.asarray(transition, dtype=np.float64)
    R = np.asarray(reward, dtype=np.float64)
    v_hat = (pi * q).sum(axis=1)
    y = R + float(gamma) * v_hat[None, None, :] - q[:, :, None]
    live = P > 0.0
    hi = np.where(live, y, -np.inf).max(axis=2)
    lo = np.where(live, y, np.inf).min(axis=2)
    return (hi - lo).reshape(-1)


def _guards(q_hat: Any, *, reward_bound: float, gamma: float) -> list[str]:
    q = np.asarray(q_hat, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        return ["numerical_nonfinite"]
    if float(np.max(np.abs(q))) > float(reward_bound) / (1.0 - float(gamma)):
        return ["divergence_guard_triggered"]
    return []


def _pair_slices(batch: dict[str, Any], d: int) -> dict[int, np.ndarray]:
    flat = np.asarray(batch["states"], dtype=np.int64) * N_ACTIONS + np.asarray(
        batch["actions"], dtype=np.int64
    )
    return {pair: np.flatnonzero(flat == pair) for pair in range(d)}


def _residuals(q_hat, policy, batch, gamma=GAMMA):
    q = np.asarray(q_hat, dtype=np.float64)
    pi = np.asarray(policy, dtype=np.float64)
    st = np.asarray(batch["states"], dtype=np.int64)
    ac = np.asarray(batch["actions"], dtype=np.int64)
    rw = np.asarray(batch["rewards"], dtype=np.float64)
    ns = np.asarray(batch["next_states"], dtype=np.int64)
    return rw + float(gamma) * (pi[ns] * q[ns]).sum(axis=1) - q[st, ac]


def certificate(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    min_visits: int,
    delta_step: float,
    lever: str = "L123",
    transition: Any = None,
    reward: Any = None,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
) -> dict[str, Any]:
    """One ``E_Q`` per lever, from the chain-replicated first-visit sample.

    ``lever`` is one of ``LEVERS``:

    ``frozen``        the current MP certificate (E = 10, delta' = delta_step/(2d));
    ``L1``            frozen + the known-Qhat envelope;
    ``L12``           L1 + the full risk budget;
    ``L123``          L12 + exact propagation (needs ``transition``);
    ``support_range`` L1 + the oracle support range (reference only).
    """
    if lever not in LEVERS:
        raise ValueError(f"unknown lever {lever!r}")
    d = int(n_states) * int(n_actions)
    reasons = _guards(q_hat, reward_bound=reward_bound, gamma=gamma)
    q = np.asarray(q_hat, dtype=np.float64)
    pi = np.asarray(policy, dtype=np.float64)
    residuals = _residuals(q, pi, batch, gamma=gamma)
    groups = _pair_slices(batch, d)
    sizes = np.array([groups[p].size for p in range(d)], dtype=np.int64)
    if int(sizes.min()) < int(min_visits):
        reasons = reasons + ["heldout_pair_support_missing"]

    use_eff = lever != "frozen"
    use_full_budget = lever in ("L12", "L123")
    y_range = 2.0 * effective_envelope(
        q, pi, reward_bound=reward_bound, gamma=gamma
    ) if use_eff else Y_RANGE
    if lever == "support_range":
        if transition is None or reward is None:
            raise ValueError("support_range needs the MDP tables")
        y_range = None  # per-pair, computed below

    delta_dir = float(delta_step) / (d if use_full_budget else 2.0 * d)
    log_term = math.log(2.0 / delta_dir)

    means = np.zeros(d)
    radii = np.zeros(d)
    eps = np.zeros(d)
    per_pair_range = np.full(d, y_range if y_range is not None else np.nan)
    if not reasons:
        if lever == "support_range":
            per_pair_range = support_range(q, pi, transition, reward, gamma=gamma)
        for pair in range(d):
            y = residuals[groups[pair]]
            n = int(sizes[pair])
            v = float(np.var(y, ddof=1)) if n > 1 else 0.0
            means[pair] = float(np.mean(y))
            radii[pair] = math.sqrt(2.0 * max(v, 0.0) * log_term / n) + MP_CONSTANT * (
                per_pair_range[pair] if lever == "support_range" else y_range
            ) * log_term / max(n - 1, 1)
            eps[pair] = abs(means[pair]) + radii[pair]

    out: dict[str, Any] = {
        "status": "not_certified" if reasons else "certificate_emitted",
        "failure_reasons": reasons,
        "lever": lever,
        "residual_means": means,
        "radii": radii,
        "epsilon_res_by_pair": eps,
        "pair_sizes": sizes,
        "delta_step": float(delta_step),
        "delta_each": delta_dir,
        "y_range": float(y_range) if y_range is not None else None,
        "n_groups": d,
        "e_q": None,
        "propagation": None,
    }
    if reasons:
        return out

    if lever in ("L123",):
        if transition is None:
            raise ValueError("L123 needs the transition kernel")
        P = np.asarray(transition, dtype=np.float64)
        K = np.einsum("sa,sat->st", pi, P)
        inv = np.linalg.inv(np.eye(int(n_states)) - float(gamma) * K)
        epsbar = (pi * eps.reshape(int(n_states), int(n_actions))).sum(axis=1)
        w = inv @ epsbar
        prop = float(gamma) * (P * w[None, None, :]).sum(axis=2)
        bound = eps.reshape(int(n_states), int(n_actions)) + prop
        out["propagation"] = {
            "K": K.tolist(),
            "inverse": inv.tolist(),
            "epsbar": epsbar.tolist(),
            "w": w.tolist(),
            "uniform_bound": float(np.max(eps)) / (1.0 - float(gamma)),
        }
        out["e_q"] = float(np.max(bound))
    else:
        out["e_q"] = float(np.max(eps)) / (1.0 - float(gamma))
    return out
