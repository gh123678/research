"""FP-CERTFIX-001: a residual certificate whose premises are actually met.

This module is NEW and additive. It does not modify, import-patch, or shadow any
sealed file. It implements the two constructions of
``docs/derivations/FP-CERTFIX-001-certificate-rederivation.md``:

``mp_certificate`` -- the reference arm (derivation section 2 + 5). One fixed-size
iid sample per pair (the first ``n_per_pair`` visits, extracted upstream by
``fp_certfix_first_n``), one Maurer-Pontil empirical Bernstein bound per pair per
direction, with the KNOWN range ``2*ENVELOPE = 20`` of the residual ``Y``. No
split halves, no estimated second-moment scale, no ordinary Bernstein with a
plugged-in sample variance.

``split_bernstein_certificate`` -- the minimal-repair control arm (derivation
section 7). Keeps the frozen two-step structure but fixes all three defects:
fixed half size ``m`` (first ``2m`` visits, lemma A), the second-moment Hoeffding
with the correct ``sqrt(2)`` (range ``ENVELOPE**2`` of ``Z = Y^2``), and the mean
step as ordinary Bernstein with variance proxy ``v_x`` and TRUE range ``2E``.

Premises both arms rely on, and where each is enforced:

- iid, fixed-size samples: lemma A of the derivation, enforced by the upstream
  first-n extraction (this module REFUSES batches whose per-pair counts are not
  exactly the frozen count);
- boundedness: ``|Y| <= ENVELOPE = R* + gamma*B + B`` with
  ``B = R*/(1-gamma)``, enforced by the ``divergence_guard`` check below;
- independence of ``(policy, q_hat)`` from the certification batch: protocol
  level -- the evaluator draws a FRESH batch per step AFTER the policy and
  q_hat of that step are fixed (theorem 2, derivation section 6).

Risk accounting is explicit: every certificate reports ``delta_step`` and
``delta_each`` so a reader can re-add the ``2d`` (MP) or ``2d`` (split: second
moment + two-sided mean share) bounds and confirm the sum is exactly
``delta_step``.
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
ENVELOPE = R_STAR + GAMMA * VALUE_BOUND + VALUE_BOUND  # E = 10, derivation section 0
Y_RANGE = 2.0 * ENVELOPE  # 20

# Maurer-Pontil constant 7/3 (range-scaled), derivation section 2.
MP_CONSTANT = 7.0 / 3.0

REASON_ORDER = (
    "divergence_guard_triggered",
    "heldout_pair_support_missing",
    "numerical_nonfinite",
    "pair_count_mismatch",
)


def residuals_for(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Held-out Bellman residuals ``Y_t(Qhat)`` for the TARGET ``policy``.

    Identical formula to the sealed modules; the residual law per pair depends
    only on ``(s, a)`` and the fixed ``(policy, q_hat)``, never on the behaviour
    policy that generated the visits.
    """
    q_hat = np.asarray(q_hat, dtype=np.float64)
    policy = np.asarray(policy, dtype=np.float64)
    states = np.asarray(batch["states"], dtype=np.int64)
    actions = np.asarray(batch["actions"], dtype=np.int64)
    rewards = np.asarray(batch["rewards"], dtype=np.float64)
    next_states = np.asarray(batch["next_states"], dtype=np.int64)
    successor = (policy[next_states] * q_hat[next_states]).sum(axis=1)
    return rewards + float(gamma) * successor - q_hat[states, actions]


def _prepare(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    n_per_pair: int,
    n_states: int,
    n_actions: int,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
) -> tuple[list[str], np.ndarray, dict[int, np.ndarray], int]:
    """Guards, residuals, and per-pair indices with the fixed-count check."""
    value_bound = float(reward_bound) / (1.0 - float(gamma))
    d = int(n_states) * int(n_actions)
    reasons: list[str] = []
    q_hat = np.asarray(q_hat, dtype=np.float64)
    if q_hat.shape != (int(n_states), int(n_actions)):
        raise ValueError("q_hat must match the declared dimensions")
    if not np.all(np.isfinite(q_hat)):
        reasons.append("numerical_nonfinite")
    elif float(np.max(np.abs(q_hat))) > value_bound:
        # Enforces the premise |Qhat| <= B on which the ENVELOPE range rests.
        reasons.append("divergence_guard_triggered")

    policy = np.asarray(policy, dtype=np.float64)
    residuals = residuals_for(q_hat, policy, batch, gamma=gamma)
    flat = np.asarray(batch["states"], dtype=np.int64) * int(n_actions) + np.asarray(
        batch["actions"], dtype=np.int64
    )
    groups: dict[int, np.ndarray] = {}
    for pair in range(d):
        members = np.flatnonzero(flat == pair)
        if members.size < int(n_per_pair):
            reasons.append("heldout_pair_support_missing")
            break
        if members.size != int(n_per_pair):
            # The upstream extractor must deliver EXACTLY the frozen count;
            # anything else means the fixed-n premise is not in force.
            reasons.append("pair_count_mismatch")
            break
        groups[pair] = members
    return sorted(set(reasons), key=REASON_ORDER.index), residuals, groups, d


def _finish(
    reasons: list[str],
    means: np.ndarray,
    radii: np.ndarray,
    d: int,
    delta_step: float,
    delta_each: float,
    gamma: float = GAMMA,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if reasons:
        return {
            "status": "not_certified",
            "failure_reasons": reasons,
            "e_q": None,
            "epsilon_res": None,
            "residual_means": means,
            "radii": radii,
            "n_groups": d,
            "delta_step": float(delta_step),
            "delta_each": float(delta_each),
            **(extra or {}),
        }
    epsilon_res = float(np.max(np.abs(means) + radii))
    return {
        "status": "certificate_emitted",
        "failure_reasons": [],
        # Lemma C: ||Qhat - Q^pi||_inf <= ||rho||_inf / (1 - gamma).
        "e_q": epsilon_res / (1.0 - float(gamma)),
        "epsilon_res": epsilon_res,
        "residual_means": means,
        "radii": radii,
        "n_groups": d,
        "delta_step": float(delta_step),
        "delta_each": float(delta_each),
        **(extra or {}),
    }


def mp_certificate(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    n_per_pair: int,
    delta_step: float,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
) -> dict[str, Any]:
    """Maurer-Pontil empirical Bernstein per pair, derivation sections 2 and 5.

    Per pair ``x`` with its fixed ``n = n_per_pair`` iid residuals (lemma A):

        |mean_x - E[Y_x]| <= sqrt(2 V_x log(2/delta_dir) / n)
                             + (7/3) * Y_RANGE * log(2/delta_dir) / (n - 1)

    with probability >= ``1 - delta_dir`` per DIRECTION, where
    ``delta_dir = delta_step / (2d)``. Union over ``d`` pairs and two
    directions costs exactly ``delta_step`` (theorem 1). ``Y_RANGE`` is derived
    from ``reward_bound``/``gamma``: ``E = R* + gamma*B + B``, ``B =
    R*/(1-gamma)``, ``Y_RANGE = 2E`` (derivation section 0).
    """
    value_bound = float(reward_bound) / (1.0 - float(gamma))
    envelope = float(reward_bound) + float(gamma) * value_bound + value_bound
    y_range = 2.0 * envelope
    reasons, residuals, groups, d = _prepare(
        q_hat, policy, batch, n_per_pair, n_states, n_actions, reward_bound, gamma
    )
    delta_dir = float(delta_step) / (2.0 * d)
    log_term = math.log(2.0 / delta_dir)
    means = np.zeros(d, dtype=np.float64)
    radii = np.zeros(d, dtype=np.float64)
    sample_vars = np.zeros(d, dtype=np.float64)
    if not reasons:
        n = int(n_per_pair)
        for pair in range(d):
            y = residuals[groups[pair]]
            v = float(np.var(y, ddof=1)) if n > 1 else 0.0
            sample_vars[pair] = v
            means[pair] = float(np.mean(y))
            radii[pair] = math.sqrt(
                2.0 * max(v, 0.0) * log_term / n
            ) + MP_CONSTANT * y_range * log_term / max(n - 1, 1)
    return _finish(
        reasons,
        means,
        radii,
        d,
        delta_step,
        delta_dir,
        gamma,
        {"arm": "mp", "n_per_pair": int(n_per_pair), "sample_vars": sample_vars},
    )


def _bernstein_t(variance_ub: float, range_: float, log_term: float, n: int) -> float:
    """Smallest ``t`` with ``t^2 = (2 V + (4/3) R t) log_term / n`` (fixed point)."""
    if n <= 0:
        return float("inf")
    t = math.sqrt(2.0 * max(variance_ub, 0.0) * log_term / n)
    for _ in range(200):
        nxt = math.sqrt(
            (2.0 * max(variance_ub, 0.0) + (4.0 / 3.0) * range_ * t) * log_term / n
        )
        if abs(nxt - t) <= 1e-15 * max(1.0, t):
            return nxt
        t = nxt
    return t


def split_bernstein_certificate(
    q_hat: Any,
    policy: Any,
    batch: dict[str, Any],
    *,
    n_per_pair: int,
    delta_step: float,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    reward_bound: float = R_STAR,
    gamma: float = GAMMA,
) -> dict[str, Any]:
    """Minimal-repair split-half Bernstein, derivation section 7.

    The batch must hold the FIRST ``2*m`` visits per pair (``n_per_pair`` is
    ``2*m`` here); half A = visits ``1..m``, half B = visits ``m+1..2m``, both
    fixed-size iid and mutually independent (lemma A).

    Step 1 (corrected): ``Z = Y^2 in [0, E^2]``, Hoeffding one-sided,
        ``v_x = mean_A(Z) + E^2 * sqrt(log(1/delta_1) / (2m))``,
    which upper-bounds ``E[Y^2] >= Var(Y)`` with probability ``>= 1 - delta_1``.
    (This is the frozen slack WITH the ``sqrt(2)`` restored: the sealed
    Bernstein module's ``(E^2/2) * sqrt(log(1/d) / n)`` is ``1/sqrt(2)`` of the
    correct ``E^2 * sqrt(log(1/d) / (2n))``.)

    Step 2: ordinary Bernstein on half B with variance proxy ``v_x`` and TRUE
    range ``2E``, two-sided, risk ``delta_2``:
        ``t`` solves ``t^2 = (2 v_x + (4/3) (2E) t) log(2/delta_2) / m``.
    On ``{v_x >= Var(Y)}`` the deviation bound holds with prob ``>= 1-delta_2``.

    Risk per pair: ``delta_1 + delta_2`` with ``delta_1 = delta_2 =
    delta_step/(2d)``; over ``d`` pairs the total is exactly ``delta_step``.
    """
    value_bound = float(reward_bound) / (1.0 - float(gamma))
    envelope = float(reward_bound) + float(gamma) * value_bound + value_bound
    y_range = 2.0 * envelope
    reasons, residuals, groups, d = _prepare(
        q_hat, policy, batch, n_per_pair, n_states, n_actions, reward_bound, gamma
    )
    delta_1 = float(delta_step) / (2.0 * d)
    delta_2 = float(delta_step) / (2.0 * d)
    means = np.zeros(d, dtype=np.float64)
    radii = np.zeros(d, dtype=np.float64)
    scales = np.zeros(d, dtype=np.float64)
    if not reasons:
        two_m = int(n_per_pair)
        if two_m % 2 != 0:
            raise ValueError("split_bernstein_certificate needs an even n_per_pair")
        m = two_m // 2
        for pair in range(d):
            idx = groups[pair]
            first, second = idx[:m], idx[m:]
            y_a = residuals[first]
            y_b = residuals[second]
            # Step 1: corrected Hoeffding on Z = Y^2 in [0, E^2].
            m2 = float(np.mean(y_a**2))
            slack = (envelope**2) * math.sqrt(math.log(1.0 / delta_1) / (2.0 * m))
            v_x = m2 + slack
            scales[pair] = math.sqrt(max(v_x, 0.0))
            # Step 2: Bernstein on half B with the true range.
            radii[pair] = _bernstein_t(
                v_x, y_range, math.log(2.0 / delta_2), m
            )
            means[pair] = float(np.mean(y_b))
    return _finish(
        reasons,
        means,
        radii,
        d,
        delta_step,
        delta_1,
        gamma,
        {
            "arm": "split_bernstein",
            "n_per_pair": int(n_per_pair),
            "half_size": int(n_per_pair) // 2,
            "delta_1": delta_1,
            "delta_2": delta_2,
            "scales": scales,
        },
    )
